/* Copyright (c) 2026 OceanBase. Licensed under the Apache License, Version 2.0. */
#define USING_LOG_PREFIX SQL_RESV
#include "ai_column_sql.h"
#include "common/mysqlclient/ob_mysql_proxy.h"
#include "query/session/ob_inner_sql_connection_access.h"
#include "share/ai_service/ai_column_definition.h"
#include "share/ob_server_struct.h"
#include "share/schema/ob_multi_version_schema_service.h"
#include "share/schema/ob_schema_getter_guard.h"
#include "sql/engine/ob_exec_context.h"
#include "sql/parser/ob_parser.h"
#include "sql/resolver/ob_resolver_define.h"
#include "sql/resolver/ob_schema_checker.h"
#include "sql/session/ob_inner_sql_connection.h"
#include "sql/session/ob_sql_session_info.h"

namespace oceanbase
{
namespace sql
{
namespace ai_column
{
using namespace common;
using namespace share::schema;
namespace ac = share::ai_column;
thread_local bool resolving_retry = false;
namespace
{
std::string str(const ObString &s)
{
  return s.empty() ? std::string() : std::string(s.ptr(), s.length());
}
std::string retry_node_text(const ParseNode &n)
{
  return n.str_value_ == nullptr ? std::string() : std::string(n.str_value_, n.str_len_);
}
void check(int ret)
{
  if (ret != OB_SUCCESS)
    throw ac::Error(ret, "AI COLUMN resolution failed");
}
void replace_status(ParseNode *node, const std::string &name, ObIAllocator &allocator,
                    unsigned depth = 0)
{
  if (node == nullptr)
    return;
  if (depth > 128)
    throw ac::Error(OB_SIZE_OVERFLOW, "RETRY predicate nesting limit");
  if (node->type_ == T_COLUMN_REF && node->num_child_ == 3 && node->children_[0] == nullptr &&
      node->children_[1] == nullptr && node->children_[2] != nullptr)
  {
    ParseNode *column = node->children_[2];
    ObString value(column->str_len_, column->str_value_);
    if (!column->is_input_quoted_ && value.case_compare("status") == 0)
    {
      ObString copy;
      check(ob_write_string(allocator, ObString(name.size(), name.data()), copy));
      column->str_value_ = copy.ptr();
      column->str_len_ = copy.length();
      return;
    }
  }
  if (node->type_ == T_SELECT)
    throw ac::Error(OB_NOT_SUPPORTED, "subqueries in RETRY are unsupported");
  for (int i = 0; i < node->num_child_; ++i)
    replace_status(node->children_[i], name, allocator, depth + 1);
}
} // namespace
int rewrite(ObResolverParams &params, const ParseNode &source, const ParseNode *&target)
{
  int ret = OB_SUCCESS;
  if (source.type_ == T_ALTER_TABLE && source.num_child_ > 1 && source.children_[1] != nullptr)
  {
    const auto *actions = source.children_[1];
    for (int i = 0; i < actions->num_child_; ++i)
      if (actions->children_[i] != nullptr && actions->children_[i]->type_ == T_AI_COLUMN_RETRY)
      {
        ret = OB_NOT_SUPPORTED;
        LOG_USER_ERROR(OB_NOT_SUPPORTED, "combining AI COLUMN RETRY with another ALTER action");
        return ret;
      }
  }
  if (source.type_ != T_AI_COLUMN_RETRY && source.type_ != T_SHOW_AI_COLUMNS)
    return ret;
  try
  {
    if (source.num_child_ < 1 || source.children_[0] == nullptr)
      throw ac::Error(OB_INVALID_ARGUMENT, "malformed AI column statement");
    const ParseNode &relation = *source.children_[0];
    if (relation.num_child_ < 2 || relation.children_[1] == nullptr)
      throw ac::Error(OB_INVALID_ARGUMENT, "missing AI column table");
    ObString database = params.session_info_->get_database_name();
    if (relation.children_[0] != nullptr)
      database.assign_ptr(relation.children_[0]->str_value_, relation.children_[0]->str_len_);
    ObString table_name(relation.children_[1]->str_len_, relation.children_[1]->str_value_);
    const ObTableSchema *table = nullptr;
    check(params.schema_checker_->get_table_schema(database, table_name, false, table));
    if (table == nullptr)
      throw ac::Error(OB_TABLE_NOT_EXIST, "AI column table does not exist");
    std::string sql;
    ParseNode *where = nullptr;
    if (source.type_ == T_SHOW_AI_COLUMNS)
    {
      sql = "SELECT column_name,expression,normalized_expr,sync_mode,max_retries,dimension FROM "
            "oceanbase.__all_ai_column WHERE table_id=" +
            std::to_string(table->get_table_id()) + " ORDER BY column_id";
    }
    else
    {
      if (source.num_child_ != 3 || source.children_[1] == nullptr)
        throw ac::Error(OB_NOT_SUPPORTED, "RETRY must specify exactly one AI column");
      std::string name = retry_node_text(*source.children_[1]);
      const auto *column = table->get_column_schema(ObString(name.size(), name.data()));
      if (column == nullptr || !column->is_ai_column())
        throw ac::Error(OB_NOT_SUPPORTED, "retrying a column that is not an AI column");
      auto d = ac::decode(str(column->get_ai_column_info_str()));
      ObObj visible;
      visible.set_varchar("true");
      auto &hint = const_cast<ObGlobalHint &>(params.query_ctx_->get_global_hint());
      check(hint.opt_params_.add_opt_param_hint(ObOptParamHint::HIDDEN_COLUMN_VISIBLE, visible));
      hint.plan_cache_policy_ = OB_USE_PLAN_CACHE_NONE;
      sql = "UPDATE /*+ OPT_PARAM('HIDDEN_COLUMN_VISIBLE' 'TRUE') USE_PLAN_CACHE(NONE) */ " +
            ac::identifier(str(database)) + "." + ac::identifier(str(table_name)) + " SET " +
            ac::identifier(name) + "=NULL," + ac::identifier(d.state_column_name) + "='PENDING'," +
            ac::identifier(d.request_column_name) + "=UUID()," +
            ac::identifier(d.result_spec_column_name) + "=" + ac::literal(d.version);
      where = source.children_[2];
      replace_status(where, d.state_column_name, *params.allocator_);
    }
    ObString copy;
    check(ob_write_string(*params.allocator_, ObString(sql.size(), sql.data()), copy));
    ObParser parser(*params.allocator_, params.session_info_->get_sql_mode(),
                    params.session_info_->get_charsets4parser());
    ParseResult result;
    check(parser.parse(copy, result));
    ParseNode *root = result.result_tree_;
    if (root != nullptr && root->type_ == T_STMT_LIST && root->num_child_ == 1)
      root = root->children_[0];
    if (root == nullptr)
      throw ac::Error(OB_ERR_UNEXPECTED, "cannot resolve AI column statement");
    if (source.type_ == T_AI_COLUMN_RETRY)
      root->children_[3] = where;
    target = root;
  }
  catch (const ac::Error &error)
  {
    ret = error.code;
    if (ret == OB_NOT_SUPPORTED)
      LOG_USER_ERROR(OB_NOT_SUPPORTED, error.what());
  }
  catch (const std::exception &)
  {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  }
  return ret;
}
int check_write(ObSQLSessionInfo &session, uint64_t flags)
{
  int ret = OB_SUCCESS;
  if ((!session.is_inner() || session.is_user_session()) && !resolving_retry &&
      (flags & (AI_COLUMN_FLAG | AI_INTERNAL_COLUMN_FLAG)) != 0)
  {
    ret = OB_NOT_SUPPORTED;
    LOG_USER_ERROR(OB_NOT_SUPPORTED, "assignment to an AI column");
  }
  return ret;
}

int strict_read(ObExecContext &context, const ObIArray<uint64_t> &references)
{
  int ret = OB_SUCCESS;
  uint64_t pending_table = 0, pending_column = 0;
  auto *session = context.get_my_session();
  ObObj option;
  if (session == nullptr || (session->is_inner() && !session->is_user_session()) ||
      references.empty())
    return ret;
  if (OB_FAIL(session->get_sys_variable(share::SYS_VAR_AI_COLUMN_STRICT_READ, option)))
    return ret;
  if (option.get_int() == 0)
    return ret;
  sqlclient::ObISQLConnectionGuard guard;
  bool created = false, begun = false;
  ObIInnerSQLConnection *connection = nullptr;
  ObSQLSessionInfo::StmtSavedValue saved_session;
  ObIInnerSQLConnection::SavedValue saved_connection;
  try
  {
    if (session->get_inner_conn() == nullptr)
    {
      check(query::ObInnerSQLConnectionAccess::create_connection_with_external_session(session,
                                                                                       guard));
      session->set_inner_conn(guard.get_ptr());
      created = true;
    }
    connection = as_inner_sql_connection(session->get_inner_conn());
    if (connection == nullptr)
      throw ac::Error(OB_ERR_UNEXPECTED, "missing inner connection");
    check(connection->begin_nested_session(saved_session, saved_connection, true));
    begun = true;
    ObSchemaGetterGuard schema;
    check(GCTX.schema_service_->get_runtime_schema_guard(schema));
    for (int64_t i = 0; i + 1 < references.count(); i += 2)
    {
      const ObTableSchema *table = nullptr;
      check(schema.get_table_schema(references.at(i), table));
      if (table == nullptr)
        continue;
      const auto *column = table->get_column_schema(references.at(i + 1));
      if (column == nullptr || !column->is_ai_column())
        continue;
      auto d = ac::decode(str(column->get_ai_column_info_str()));
      const ObDatabaseSchema *database = nullptr;
      check(schema.get_database_schema(table->get_database_id(), database));
      if (database == nullptr)
        throw ac::Error(OB_ERR_UNEXPECTED, "missing database");
      std::string sql = "SELECT /*+ OPT_PARAM('HIDDEN_COLUMN_VISIBLE' 'TRUE') */ 1 FROM " +
                        ac::identifier(str(database->get_database_name_str())) + "." +
                        ac::identifier(str(table->get_table_name_str())) + " WHERE NOT(" +
                        ac::identifier(d.state_column_name) + " <=> 'READY') LIMIT 1";
      alignas(16) ObMySQLProxy::MySQLResult holder;
      check(connection->execute_read(sql.c_str(), holder));
      auto *result = holder.get_result();
      if (result == nullptr)
        throw ac::Error(OB_ERR_UNEXPECTED, "missing readiness result");
      int next = result->next();
      if (next == OB_SUCCESS)
      {
        ret = OB_AI_COLUMN_NOT_READY;
        pending_table = references.at(i);
        pending_column = references.at(i + 1);
        break;
      }
      if (next != OB_ITER_END)
        check(next);
    }
  }
  catch (const ac::Error &error)
  {
    ret = error.code;
  }
  catch (const std::exception &)
  {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  }
  if (begun)
  {
    int end = connection->end_nested_session(saved_session, saved_connection);
    if (ret == OB_SUCCESS)
      ret = end;
  }
  if (created)
  {
    session->set_inner_conn(nullptr);
    guard.reset();
  }
  if (ret == OB_AI_COLUMN_NOT_READY)
    FORWARD_USER_ERROR_MSG(ret, "AI column has unfinished tasks for table_id=%lu, column_id=%lu",
                           pending_table, pending_column);
  return ret;
}
} // namespace ai_column
} // namespace sql
} // namespace oceanbase
