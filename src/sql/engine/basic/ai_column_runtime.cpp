/* Copyright (c) 2026 OceanBase. Licensed under the Apache License, Version 2.0. */
#define USING_LOG_PREFIX SQL_ENG
#include "ai_column_runtime.h"
#include "common/mysqlclient/ob_mysql_proxy.h"
#include "common/mysqlclient/ob_mysql_transaction.h"
#include "lib/time/ob_time_utility.h"
#include "query/ai/ob_ai_endpoint_resolver.h"
#include "share/ai_service/ai_column_definition.h"
#include "share/ob_server_struct.h"
#include "share/ob_thread_pool.h"
#include "share/rc/ob_server_runtime.h"
#include "share/schema/ob_multi_version_schema_service.h"
#include "share/schema/ob_schema_getter_guard.h"
#include "share/schema/ob_trigger_info.h"
#include "sql/engine/expr/ob_expr_ai/ob_ai_func_utils.h"
#include <algorithm>
#include <atomic>
#include <cctype>
#include <map>
#include <mutex>

namespace oceanbase
{
namespace sql
{
namespace ai_column
{
using namespace common;
using namespace share::schema;
namespace ac = share::ai_column;
namespace runtime_detail
{
const std::string queue = "oceanbase.__all_ai_column_task";
struct Column
{
  uint64_t id;
  std::string name;
  ac::Definition definition;
};
struct Table
{
  uint64_t id = 0, database_id = 0;
  std::string database, name;
  std::vector<Column> columns;
  std::vector<std::pair<std::string, bool>> key;
};
struct Task
{
  uint64_t table = 0, column = 0;
  int state = 0, retries = 0, error = 0;
  bool null_input = false, null_result = false, invalid_input = false;
  std::string hash, request, version, predicate, input, config, result, owner;
};
std::string str(const ObString &value)
{
  return value.empty() ? std::string() : std::string(value.ptr(), value.length());
}
ObString ob(std::string &value)
{
  return ObString(value.size(), value.data());
}
std::string number(uint64_t value)
{
  return std::to_string(value);
}
std::string qualified(const Table &table)
{
  return ac::identifier(table.database) + "." + ac::identifier(table.name);
}
void check(int code)
{
  if (code != OB_SUCCESS)
    throw ac::Error(code, "AI COLUMN internal SQL failed");
}
std::string string_field(sqlclient::ObMySQLResult &result, const char *name)
{
  ObString value;
  int ret = result.get_varchar(name, value);
  if (ret == OB_ERR_NULL_VALUE)
    return {};
  check(ret);
  return str(value);
}
std::string input_field(sqlclient::ObMySQLResult &result, const char *name, size_t &bytes,
                        bool &invalid)
{
  ObString value;
  int ret = result.get_varchar(name, value);
  if (ret == OB_ERR_NULL_VALUE)
    return {};
  check(ret);
  if (static_cast<size_t>(value.length()) > ac::MAX_GROUP_INPUT_BYTES - bytes)
  {
    invalid = true;
    return {};
  }
  bytes += value.length();
  return str(value);
}
int64_t int_field(sqlclient::ObMySQLResult &result, const char *name)
{
  int64_t value = 0;
  check(result.get_int(name, value));
  return value;
}
int64_t write(ObISQLClient &client, const std::string &sql)
{
  int64_t count = 0;
  check(client.write(sql.c_str(), count));
  return count;
}
Table load_table(uint64_t id)
{
  Table result;
  result.id = id;
  ObSchemaGetterGuard guard;
  if (GCTX.schema_service_ == nullptr)
    throw ac::Error(OB_NOT_INIT, "AI COLUMN schema is unavailable");
  check(GCTX.schema_service_->get_runtime_schema_guard(guard));
  const ObTableSchema *table = nullptr;
  check(guard.get_table_schema(id, table));
  if (table == nullptr)
    return result;
  const ObDatabaseSchema *database = nullptr;
  check(guard.get_database_schema(table->get_database_id(), database));
  if (database == nullptr)
    return result;
  result.database_id = table->get_database_id();
  result.name = str(table->get_table_name_str());
  result.database = str(database->get_database_name_str());
  for (auto it = table->column_begin(); it != table->column_end(); ++it)
  {
    if ((*it)->is_ai_column())
      result.columns.push_back({(*it)->get_column_id(), str((*it)->get_column_name_str()),
                                ac::decode(str((*it)->get_ai_column_info_str()))});
  }
  const auto &key = table->get_rowkey_info();
  for (int64_t i = 0; i < key.get_size(); ++i)
  {
    uint64_t column_id = 0;
    check(key.get_column_id(i, column_id));
    const auto *column = table->get_column_schema(column_id);
    if (column == nullptr)
      throw ac::Error(OB_ERR_UNEXPECTED, "AI COLUMN row key is missing");
    result.key.push_back(
        {str(column->get_column_name_str()), ob_is_numeric_type(column->get_data_type())});
  }
  return result;
}
std::string ref(const std::string &prefix, const std::string &name)
{
  return prefix + "." + ac::identifier(name);
}
// Definitions have already been parsed and bound. Only replace identifier tokens,
// preserving quoted strings, escapes, function names and all expression operators.
std::string new_expression(const std::string &expression, const ac::Definition &definition)
{
  std::string out;
  size_t i = 0;
  auto fold = [](std::string s)
  {
    for (char &c : s)
      if (c >= 'A' && c <= 'Z')
        c += 'a' - 'A';
    return s;
  };
  while (i < expression.size())
  {
    char c = expression[i];
    size_t begin = i;
    if (c == '\'' || c == '"')
    {
      ++i;
      while (i < expression.size())
      {
        char n = expression[i++];
        if (n == '\\' && i < expression.size())
          ++i;
        else if (n == c)
        {
          if (i < expression.size() && expression[i] == c)
            ++i;
          else
            break;
        }
      }
      out += expression.substr(begin, i - begin);
      continue;
    }
    std::string token;
    if (c == '`')
    {
      ++i;
      while (i < expression.size())
      {
        char n = expression[i++];
        if (n == '`')
        {
          if (i < expression.size() && expression[i] == '`')
          {
            token += '`';
            ++i;
          }
          else
            break;
        }
        else
          token += n;
      }
    }
    else if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' ||
             static_cast<unsigned char>(c) >= 128)
    {
      ++i;
      while (i < expression.size() &&
             (std::isalnum(static_cast<unsigned char>(expression[i])) || expression[i] == '_' ||
              expression[i] == '$' || static_cast<unsigned char>(expression[i]) >= 128))
        ++i;
      token = expression.substr(begin, i - begin);
    }
    else
    {
      out += c;
      ++i;
      continue;
    }
    size_t next = i;
    while (next < expression.size() && std::isspace(static_cast<unsigned char>(expression[next])))
      ++next;
    bool function = next < expression.size() && expression[next] == '(';
    bool dependency =
        !function &&
        std::any_of(definition.dependencies.begin(), definition.dependencies.end(),
                    [&](const ac::Dependency &d) { return fold(d.name) == fold(token); });
    out += dependency ? ref("NEW", token) : expression.substr(begin, i - begin);
  }
  return out;
}
std::string predicate_expression(const Table &table, const std::string &prefix)
{
  std::string result = "CONCAT(";
  bool first = true;
  for (const auto &key : table.key)
  {
    if (!first)
      result += "," + ac::literal(" AND ") + ",";
    result += ac::literal(ac::identifier(key.first) + " = ") + ",";
    std::string value = "CAST(" + ref(prefix, key.first) + " AS CHAR CHARACTER SET utf8mb4)";
    // Hex string literals are independent of NO_BACKSLASH_ESCAPES and connection charset.
    result += key.second
                  ? value
                  : "CONCAT('CONVERT(X',CHAR(39),HEX(" + value + "),CHAR(39),' USING utf8mb4)')";
    first = false;
  }
  return result + ")";
}
std::string key_changed(const Table &table)
{
  std::string value = "FALSE";
  for (const auto &key : table.key)
    value += " OR NOT(" + ref("NEW", key.first) + " <=> " + ref("OLD", key.first) + ")";
  return "(" + value + ")";
}
std::string changed(const Table &table, const Column &column)
{
  std::string value = key_changed(table);
  for (const auto &dependency : column.definition.dependencies)
    value += " OR NOT(" + ref("NEW", dependency.name) + " <=> " + ref("OLD", dependency.name) + ")";
  return "(" + value + ")";
}
std::string invalidation(const Column &column, const std::string &request)
{
  const auto &d = column.definition;
  return "SET " + ref("NEW", column.name) + "=NULL," + ref("NEW", d.state_column_name) +
         "='PENDING'," + ref("NEW", d.request_column_name) + "=" + request + "," +
         ref("NEW", d.result_spec_column_name) + "=" + ac::literal(d.version) + "; ";
}
std::string enqueue(const Table &table, const Column &column)
{
  const auto &d = column.definition;
  std::string pred = predicate_expression(table, "NEW");
  std::string input = new_expression(d.input_expression, d),
              config = new_expression(d.config_expression, d);
  return "INSERT INTO " + queue +
         "(table_id,column_id,rowkey_hash,request_id,definition_version,rowkey_predicate,input_"
         "value,config_value,input_is_null,state,retry_count,eligible_ts) VALUES(" +
         number(table.id) + "," + number(column.id) + ",SHA2(" + pred + ",256)," +
         ref("NEW", d.request_column_name) + "," + ac::literal(d.version) + "," + pred + ",CAST((" +
         input + ") AS CHAR CHARACTER SET utf8mb4),CAST((" + config +
         ") AS CHAR CHARACTER SET utf8mb4),(" + input +
         ") IS NULL,1,0,NOW(6)) ON DUPLICATE KEY UPDATE "
         "request_id=VALUES(request_id),definition_version=VALUES(definition_version),rowkey_"
         "predicate=VALUES(rowkey_predicate),input_value=VALUES(input_value),config_value=VALUES("
         "config_value),input_is_null=VALUES(input_is_null),state=1,retry_count=0,eligible_ts=NOW("
         "6),owner_id=NULL,lease_until=NULL,result_value=NULL,result_is_null=0,error_code=0; ";
}
std::mutex installation_mutex, claim_mutex, active_mutex;
std::map<std::string, Task> active_claims;
struct ActiveClaim
{
  std::string owner;
  explicit ActiveClaim(const Task &task) : owner(task.owner)
  {
    std::lock_guard<std::mutex> lock(active_mutex);
    active_claims[owner] = task;
  }
  ~ActiveClaim()
  {
    std::lock_guard<std::mutex> lock(active_mutex);
    active_claims.erase(owner);
  }
};
std::map<uint64_t, std::string> installed;
int install_impl(uint64_t table_id)
{
  std::lock_guard<std::mutex> lock(installation_mutex);
  try
  {
    Table table = load_table(table_id);
    if (table.columns.empty())
      return OB_SUCCESS;
    std::string signature = qualified(table);
    for (const auto &key : table.key)
      signature += ac::identifier(key.first) + (key.second ? "N" : "S");
    for (const auto &c : table.columns)
      signature += number(c.id) + ":" + c.definition.version + ";";
    if (installed[table_id] == signature)
      return OB_SUCCESS;
    if (table.key.empty())
      throw ac::Error(OB_NOT_SUPPORTED, "AI COLUMN requires a row key");
    std::string before_insert = "BEGIN DECLARE ai_request VARCHAR(64); SET ai_request=UUID(); ";
    std::string before_update = "BEGIN DECLARE ai_request VARCHAR(64); SET ai_request=UUID(); ";
    std::string after_insert = "BEGIN ", after_update = "BEGIN ";
    for (const auto &column : table.columns)
    {
      before_insert += invalidation(column, "ai_request");
      before_update += "IF " + changed(table, column) + " OR " +
                       ref("NEW", column.definition.request_column_name) +
                       "='__AI_REGENERATE__' THEN " + invalidation(column, "ai_request") +
                       " END IF; ";
      after_insert += enqueue(table, column);
      after_update += "IF NOT(" + ref("NEW", column.definition.request_column_name) + " <=> " +
                      ref("OLD", column.definition.request_column_name) + ") THEN " +
                      enqueue(table, column) + " END IF; ";
    }
    std::string erase = "DELETE FROM " + queue + " WHERE table_id=" + number(table.id) +
                        " AND rowkey_hash=SHA2(" + predicate_expression(table, "OLD") + ",256); ";
    after_update =
        "BEGIN IF " + key_changed(table) + " THEN " + erase + "END IF; " + after_update.substr(6);
    struct Trigger
    {
      const char *suffix, *when;
      std::string body;
    };
    std::vector<Trigger> triggers = {{"bi", "BEFORE INSERT", before_insert + "END"},
                                     {"bu", "BEFORE UPDATE", before_update + "END"},
                                     {"ai", "AFTER INSERT", after_insert + "END"},
                                     {"au", "AFTER UPDATE", after_update + "END"},
                                     {"ad", "AFTER DELETE", "BEGIN " + erase + "END"}};
    for (const auto &trigger : triggers)
    {
      std::string name = ac::identifier(table.database) + "." +
                         ac::identifier("__ai_" + number(table.id) + "_" + trigger.suffix);
      ObSchemaGetterGuard guard;
      check(GCTX.schema_service_->get_runtime_schema_guard(guard));
      const ObTriggerInfo *existing = nullptr;
      std::string local_name = "__ai_" + number(table.id) + "_" + trigger.suffix;
      check(guard.get_trigger_info(table.database_id,
                                   ObString(local_name.size(), local_name.data()), existing));
      if (existing != nullptr && str(existing->get_trigger_body()) == trigger.body)
        continue;
      if (existing != nullptr)
        write(*GCTX.sql_proxy_, "DROP TRIGGER " + name);
      write(*GCTX.sql_proxy_, "CREATE TRIGGER " + name + " " + trigger.when + " ON " +
                                  qualified(table) + " FOR EACH ROW " + trigger.body);
    }
    // Existing rows after ADD/MODIFY use the same transactional trigger path as
    // business DML. A single UPDATE groups all newly invalidated columns per row.
    std::string assignments, condition;
    for (const auto &column : table.columns)
    {
      const auto &d = column.definition;
      std::string mismatch = "NOT(" + ac::identifier(d.result_spec_column_name) + " <=> " +
                             ac::literal(d.version) + ")";
      if (!assignments.empty())
      {
        assignments += ",";
        condition += " OR ";
      }
      assignments += ac::identifier(d.request_column_name) + "=IF(" + mismatch +
                     ",'__AI_REGENERATE__'," + ac::identifier(d.request_column_name) + ")";
      condition += mismatch;
    }
    write(*GCTX.sql_proxy_, "UPDATE /*+ OPT_PARAM('HIDDEN_COLUMN_VISIBLE' 'TRUE') */ " +
                                qualified(table) + " SET " + assignments + " WHERE " + condition);
    installed[table_id] = signature;
    return OB_SUCCESS;
  }
  catch (const ac::Error &error)
  {
    int ret = error.code;
    LOG_WARN("AI COLUMN trigger installation failed", K(ret), K(table_id));
    return ret;
  }
  catch (const std::exception &)
  {
    return OB_ALLOCATE_MEMORY_FAILED;
  }
}
std::string task_where(const Task &task, bool owner = true)
{
  return "table_id=" + number(task.table) + " AND column_id=" + number(task.column) +
         " AND rowkey_hash=" + ac::literal(task.hash) +
         " AND request_id=" + ac::literal(task.request) +
         (owner ? " AND owner_id=" + ac::literal(task.owner) : "");
}
std::vector<Task> read_tasks(ObISQLClient &client, const std::string &where)
{
  alignas(16) ObMySQLProxy::MySQLResult holder;
  check(client.read(holder, ("SELECT "
                             "table_id,column_id,rowkey_hash,request_id,definition_version,rowkey_"
                             "predicate,CAST(input_value AS CHAR) input_value,CAST(config_value AS "
                             "CHAR) config_value,input_is_null,state,retry_count,CAST(result_value "
                             "AS CHAR) result_value,result_is_null,error_code,owner_id FROM " +
                             queue + " WHERE " + where)
                                .c_str()));
  auto *result = holder.get_result();
  if (result == nullptr)
    throw ac::Error(OB_ERR_UNEXPECTED, "missing task result");
  std::vector<Task> tasks;
  int ret = 0;
  size_t input_bytes = 0;
  while ((ret = result->next()) == OB_SUCCESS)
  {
    Task t;
    t.table = int_field(*result, "table_id");
    t.column = int_field(*result, "column_id");
    t.hash = string_field(*result, "rowkey_hash");
    t.request = string_field(*result, "request_id");
    t.version = string_field(*result, "definition_version");
    t.predicate = string_field(*result, "rowkey_predicate");
    t.input = input_field(*result, "input_value", input_bytes, t.invalid_input);
    t.config = input_field(*result, "config_value", input_bytes, t.invalid_input);
    t.null_input = int_field(*result, "input_is_null") != 0;
    t.state = int_field(*result, "state");
    t.retries = int_field(*result, "retry_count");
    t.result = string_field(*result, "result_value");
    t.null_result = int_field(*result, "result_is_null") != 0;
    t.error = int_field(*result, "error_code");
    t.owner = string_field(*result, "owner_id");
    tasks.push_back(std::move(t));
    if (tasks.size() > 1024)
      throw ac::Error(OB_SIZE_OVERFLOW, "too many AI columns in one group");
  }
  if (ret != OB_ITER_END)
    check(ret);
  return tasks;
}
std::atomic<uint64_t> owner_sequence{0};
std::vector<Task> claim()
{
  std::lock_guard<std::mutex> lock(claim_mutex);
  ObMySQLTransaction transaction;
  check(transaction.start(GCTX.sql_proxy_));
  try
  {
    auto candidates = read_tasks(
        transaction, "state IN(1,2,3,4,5) AND eligible_ts<=NOW(6) AND (owner_id IS NULL OR "
                     "lease_until<NOW(6)) ORDER BY eligible_ts,table_id,column_id LIMIT 1");
    if (candidates.empty())
    {
      check(transaction.end(true));
      return {};
    }
    const auto &first = candidates.front();
    std::string where = "table_id=" + number(first.table) +
                        " AND rowkey_hash=" + ac::literal(first.hash) +
                        " AND request_id=" + ac::literal(first.request);
    std::string owner = number(ObTimeUtility::current_time()) + "-" + number(++owner_sequence);
    int64_t count = write(transaction, "UPDATE " + queue + " SET owner_id=" + ac::literal(owner) +
                                           ",lease_until=DATE_ADD(NOW(6),INTERVAL 15 "
                                           "SECOND),state=IF(state IN(1,2,3),2,state) WHERE " +
                                           where + " AND (owner_id IS NULL OR lease_until<NOW(6))");
    std::vector<Task> tasks;
    if (count > 0)
      tasks = read_tasks(transaction,
                         where + " AND owner_id=" + ac::literal(owner) + " ORDER BY column_id");
    check(transaction.end(true));
    return tasks;
  }
  catch (...)
  {
    transaction.end(false);
    throw;
  }
}
const Column *find_column(const Table &table, uint64_t id)
{
  for (const auto &c : table.columns)
    if (c.id == id)
      return &c;
  return nullptr;
}
int model_call(const Column &column, Task &task)
{
  const auto &definition = column.definition;
  if (task.null_input)
  {
    task.null_result = true;
    return OB_SUCCESS;
  }
  if (task.invalid_input)
    return OB_SIZE_OVERFLOW;
  ObArenaAllocator allocator("AIColumnCall");
  ObAIFuncExprInfo *info = nullptr;
  ObString model_name(definition.model.size(), definition.model.data());
  int ret = ObAIFuncUtils::get_ai_func_info(allocator, model_name, info);
  auto *resolver = share::server_service<query::ObIAiEndpointResolver>();
  share::ObAiModelEndpointInfo endpoint;
  if (ret == OB_SUCCESS && resolver == nullptr)
    ret = OB_NOT_INIT;
  if (ret == OB_SUCCESS)
    ret = resolver->resolve_by_model_name(model_name, allocator, endpoint, false);
  if (ret == OB_SUCCESS)
  {
    ObJsonObject *config = nullptr;
    ObString prompt = ob(task.input), result;
    std::string config_text = task.config;
    if (definition.embedding)
      config_text = "{\"dimension\":" + std::to_string(definition.dimension) + "}";
    if (!config_text.empty() && config_text != "null" && config_text != "NULL")
    {
      ObString value = ob(config_text);
      ret = ObAIFuncJsonUtils::get_json_object_form_str(allocator, value, config);
    }
    if (ret == OB_SUCCESS && !definition.embedding && !task.input.empty() &&
        task.input.front() == '{')
    {
      ObJsonObject *object = nullptr;
      ObString value = ob(task.input);
      if (ObAIFuncJsonUtils::get_json_object_form_str(allocator, value, object) == OB_SUCCESS &&
          ObAIFuncPromptObjectUtils::is_valid_prompt_object(object))
        ret =
            ObAIFuncPromptObjectUtils::replace_all_str_args_in_template(allocator, object, prompt);
    }
    if (ret == OB_SUCCESS)
    {
      ObAIFuncModel model(allocator, *info, endpoint);
      ObString instruction(definition.type_prompt.size(), definition.type_prompt.data());
      ret = definition.embedding ? model.call_dense_embedding(prompt, config, result)
                                 : model.call_completion(prompt, config, result, instruction);
      if (ret == OB_SUCCESS)
      {
        task.result = str(result);
        ac::result_literal(definition, task.result);
      }
    }
  }
  return ret;
}
void stage(const Column &column, Task &task)
{
  int ret = OB_SUCCESS;
  try
  {
    ret = model_call(column, task);
  }
  catch (const ac::Error &e)
  {
    ret = e.code;
  }
  catch (const std::exception &)
  {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  }
  task.error = ret;
  bool retry = ret != OB_SUCCESS && task.retries < column.definition.max_retries;
  int state = ret == OB_SUCCESS ? 4 : retry ? 3 : 5;
  std::string values = "state=" + std::to_string(state) + ",error_code=" + std::to_string(ret) +
                       ",result_value=" + (ret == OB_SUCCESS ? ac::literal(task.result) : "NULL") +
                       ",result_is_null=" + (task.null_result ? "1" : "0");
  if (retry)
    values += ",retry_count=retry_count+1,eligible_ts=DATE_ADD(NOW(6),INTERVAL 1 SECOND)";
  write(*GCTX.sql_proxy_, "UPDATE " + queue + " SET " + values + " WHERE " + task_where(task));
}
void publish(const Table &table, const std::vector<Task> &original)
{
  if (original.empty())
    return;
  const auto &first = original.front();
  std::string group =
      "table_id=" + number(first.table) + " AND rowkey_hash=" + ac::literal(first.hash) +
      " AND request_id=" + ac::literal(first.request) + " AND owner_id=" + ac::literal(first.owner);
  ObMySQLTransaction transaction;
  check(transaction.start(GCTX.sql_proxy_));
  try
  {
    // Lock the business row before the outbox, matching business DML lock order.
    if (!table.name.empty())
    {
      alignas(16) ObMySQLProxy::MySQLResult row;
      check(
          transaction.read(row, ("SELECT /*+ OPT_PARAM('HIDDEN_COLUMN_VISIBLE' 'TRUE') */ 1 FROM " +
                                 qualified(table) + " WHERE " + first.predicate + " FOR UPDATE")
                                    .c_str()));
      if (row.get_result() != nullptr)
      {
        int next = row.get_result()->next();
        if (next != OB_SUCCESS && next != OB_ITER_END)
          check(next);
      }
    }
    auto tasks = read_tasks(transaction, group + " ORDER BY column_id FOR UPDATE");
    bool terminal = !tasks.empty();
    for (const auto &task : tasks)
      terminal = terminal && (task.state == 4 || task.state == 5);
    if (terminal)
    {
      std::string assignments, conditions;
      for (const auto &task : tasks)
      {
        const Column *column = find_column(table, task.column);
        if (column == nullptr || column->definition.version != task.version)
        {
          terminal = false;
          break;
        }
        const auto &d = column->definition;
        if (!assignments.empty())
          assignments += ",";
        assignments +=
            ac::identifier(column->name) + "=" +
            (task.state == 4 && !task.null_result ? ac::result_literal(d, task.result) : "NULL") +
            "," + ac::identifier(d.state_column_name) + "=" +
            ac::literal(task.state == 4 ? "READY" : "FAILED");
        conditions += " AND " + ac::identifier(d.request_column_name) + "=" +
                      ac::literal(task.request) + " AND " +
                      ac::identifier(d.result_spec_column_name) + "=" + ac::literal(task.version);
      }
      if (terminal && !table.name.empty())
        write(transaction, "UPDATE /*+ OPT_PARAM('HIDDEN_COLUMN_VISIBLE' 'TRUE') */ " +
                               qualified(table) + " SET " + assignments + " WHERE " +
                               first.predicate + conditions);
      write(transaction, "DELETE FROM " + queue + " WHERE " + group);
    }
    else
    {
      write(transaction, "UPDATE " + queue +
                             " SET "
                             "owner_id=NULL,lease_until=NULL,eligible_ts=GREATEST(eligible_ts,DATE_"
                             "ADD(NOW(6),INTERVAL 1 SECOND)) WHERE " +
                             group);
    }
    check(transaction.end(true));
  }
  catch (...)
  {
    transaction.end(false);
    throw;
  }
}
class Worker final : public share::ObThreadPool
{
public:
  Worker() : share::ObThreadPool(6) {}

private:
  void run1() override
  {
    int ret = OB_SUCCESS;
    lib::set_thread_name("AIColumn");
    int64_t last_reconcile = 0;
    while (!has_set_stop())
    {
      try
      {
        if (GCTX.sql_proxy_ != nullptr && GCTX.schema_service_ != nullptr &&
            GCTX.sys_package_ready_ && share::server_is_write_enabled())
        {
          if (get_thread_idx() == 5)
          {
            int64_t now = ObTimeUtility::current_time();
            if (now - last_reconcile >= 5000000)
            {
              std::vector<uint64_t> ids;
              {
                alignas(16) ObMySQLProxy::MySQLResult holder;
                check(GCTX.sql_proxy_->read(
                    holder, "SELECT DISTINCT table_id FROM oceanbase.__all_ai_column"));
                auto *rows = holder.get_result();
                if (rows != nullptr)
                {
                  int next = 0;
                  while ((next = rows->next()) == OB_SUCCESS)
                    ids.push_back(int_field(*rows, "table_id"));
                  if (next != OB_ITER_END)
                    check(next);
                }
              }
              for (uint64_t id : ids)
              {
                if (has_set_stop())
                  break;
                int installed_ret = install_impl(id);
                if (installed_ret != OB_SUCCESS)
                  LOG_WARN("AI COLUMN reconciliation failed", K(installed_ret), K(id));
              }
              last_reconcile = now;
            }
            ob_usleep(1000000);
            continue;
          }
          if (get_thread_idx() == 4)
          {
            std::vector<Task> active;
            {
              std::lock_guard<std::mutex> lock(active_mutex);
              for (const auto &entry : active_claims)
                active.push_back(entry.second);
            }
            for (const auto &task : active)
              write(*GCTX.sql_proxy_,
                    "UPDATE " + queue +
                        " SET lease_until=DATE_ADD(NOW(6),INTERVAL 15 SECOND) WHERE table_id=" +
                        number(task.table) + " AND rowkey_hash=" + ac::literal(task.hash) +
                        " AND request_id=" + ac::literal(task.request) +
                        " AND owner_id=" + ac::literal(task.owner));
            ob_usleep(1000000);
            continue;
          }
          auto tasks = claim();
          if (!tasks.empty())
          {
            ActiveClaim active(tasks.front());
            Table table = load_table(tasks.front().table);
            if (table.columns.empty())
              write(*GCTX.sql_proxy_,
                    "DELETE FROM " + queue + " WHERE table_id=" + number(table.id));
            else
            {
              for (auto &task : tasks)
              {
                if (has_set_stop())
                  break;
                if (task.state == 4 || task.state == 5)
                  continue;
                const Column *column = find_column(table, task.column);
                if (column != nullptr && column->definition.version == task.version)
                  stage(*column, task);
                else
                  write(*GCTX.sql_proxy_, "DELETE FROM " + queue + " WHERE " + task_where(task));
              }
              publish(table, tasks);
            }
            continue;
          }
        }
      }
      catch (const ac::Error &error)
      {
        int ret = error.code;
        LOG_WARN("AI COLUMN worker iteration failed", K(ret));
      }
      catch (const std::exception &)
      {
        LOG_WARN("AI COLUMN worker allocation failed");
      }
      ob_usleep(100000);
    }
  }
};
Worker worker;
} // namespace runtime_detail
int install(uint64_t table_id)
{
  return runtime_detail::install_impl(table_id);
}
int start_worker()
{
  return runtime_detail::worker.start();
}
void stop_worker()
{
  runtime_detail::worker.stop();
}
void wait_worker()
{
  runtime_detail::worker.wait();
}
} // namespace ai_column
} // namespace sql
} // namespace oceanbase
