/* Copyright (c) 2026 OceanBase. Licensed under the Apache License, Version 2.0. */
#define USING_LOG_PREFIX SQL_ENG
#include "file_sql_engine.h"
#include "share/ob_server_struct.h"
#include "share/schema/ob_schema_utils.h"
#include "share/system_variable/ob_sys_var_class_type.h"
#include <set>

namespace oceanbase { namespace sql { namespace file_sql {
using namespace common;
int access_root(ObSQLSessionInfo &session, ObIAllocator &allocator, ObString &root)
{
  int ret = OB_SUCCESS;
  bool enabled = false;
  if (OB_FAIL(session.get_sys_variable(share::SYS_VAR_ENABLE_FILE_SQL, enabled))) {
  } else if (!enabled) {
    ret = OB_NOT_SUPPORTED;
    LOG_USER_ERROR(OB_NOT_SUPPORTED, "file SQL is disabled; set enable_file_sql = ON");
  } else if (OB_ISNULL(GCTX.schema_service_)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_FAIL(share::schema::ObSchemaUtils::get_runtime_varchar_variable(
      *GCTX.schema_service_, share::SYS_VAR_SECURE_FILE_PRIV, allocator, root))) {
  }
  return ret;
}
namespace {
std::string text(const ParseNode &node)
{
  if (node.type_ != T_CHAR && node.type_ != T_VARCHAR) throw FileSqlError("file SQL: expected a string literal");
  return std::string(node.str_value_, static_cast<size_t>(node.str_len_));
}
std::string constant(const ParseNode &node)
{
  if (node.type_ == T_CHAR || node.type_ == T_VARCHAR) return text(node);
  if (node.type_ == T_INT || node.type_ == T_BOOL) return std::to_string(node.value_);
  if (node.type_ == T_OP_NEG && node.num_child_ == 1 && node.children_[0]->type_ == T_INT)
    return "-" + std::to_string(node.children_[0]->value_);
  throw FileSqlError("file SQL: options require constant strings, integers or booleans");
}
}
Plan source_from_node(const ParseNode &node)
{
  if (node.type_ != T_FILE_TABLE_EXPRESSION || node.num_child_ != 4 || node.children_[0] == nullptr || node.children_[1] == nullptr)
    throw FileSqlError("file SQL: malformed table function");
  Plan plan;
  plan.format.assign(node.children_[0]->str_value_, node.children_[0]->str_len_);
  plan.path = text(*node.children_[1]);
  std::set<std::string> seen;
  const ParseNode *options = node.children_[2];
  for (int64_t i = 0; options != nullptr && i < options->num_child_; ++i) {
    const ParseNode *option = options->children_[i];
    if (option == nullptr || option->num_child_ != 2 || option->children_[0] == nullptr || option->children_[1] == nullptr)
      throw FileSqlError("file SQL: malformed named option");
    std::string key(option->children_[0]->str_value_, option->children_[0]->str_len_);
    for (char &c : key) if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
    if (!seen.insert(key).second) throw FileSqlError("file SQL: duplicate option: " + key);
    std::string value = constant(*option->children_[1]);
    if (key == "sample_size") {
      if (value != "-1") throw FileSqlError("file SQL: sample_size only supports -1");
    } else if (key == "format") {
      for (char &c : value) if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
      if (value != plan.format && !(plan.format == "json" && (value == "newline_delimited" || value == "ndjson")))
        throw FileSqlError("file SQL: unsupported format option");
    } else if (plan.format == "csv") {
      if (key == "header") {
        if (value != "0" && value != "1" && value != "true" && value != "false") throw FileSqlError("file SQL: header must be boolean");
        plan.options.header = value == "1" || value == "true";
      } else if (key == "nullstr" || key == "null_string" || key == "null") plan.options.null_text = value;
      else if (key == "delimiter" || key == "delim" || key == "quote" || key == "escape") {
        if (value.size() != 1 || value[0] == '\0' || value[0] == '\r' || value[0] == '\n') throw FileSqlError("file SQL: separator must be one non-newline byte");
        if (key == "delimiter" || key == "delim") plan.options.delimiter = value[0];
        else if (key == "quote") plan.options.quote = value[0];
        else plan.options.escape = value[0];
      } else throw FileSqlError("file SQL: unknown CSV option: " + key);
    } else throw FileSqlError("file SQL: unknown option: " + key);
  }
  if (plan.options.delimiter == plan.options.quote) throw FileSqlError("file SQL: delimiter and quote must differ");
  return plan;
}
ObDataType engine_type(const Column &column)
{
  const ObObjType types[] = {ObNullType, ObIntType, ObDoubleType, ObTinyIntType, ObVarcharType, ObDateType, ObDateTimeType};
  ObObjType type = types[static_cast<size_t>(column.type)];
  ObDataType result;
  result.set_obj_type(type);
  result.set_accuracy(ObAccuracy::DDL_DEFAULT_ACCURACY[type]);
  if (type == ObVarcharType) {
    result.set_length(16 * 1024 * 1024);
    result.set_collation_type(CS_TYPE_UTF8MB4_BIN);
    result.set_collation_level(CS_LEVEL_IMPLICIT);
  }
  if (type == ObDateTimeType) result.set_scale(column.scale);
  return result;
}
}}}
