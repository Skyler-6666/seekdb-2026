/* Copyright (c) 2026 OceanBase. Licensed under the Apache License, Version 2.0. */
#define USING_LOG_PREFIX SQL_RESV
#include "ai_column_ddl.h"
#include "lib/time/ob_time_utility.h"
#include "share/ai_service/ai_column_definition.h"
#include "share/schema/ob_schema_service.h"
#include "sql/engine/expr/ob_expr_ai/ob_ai_func_utils.h"
#include <algorithm>
#include <atomic>
#include <charconv>
#include <map>
#include <set>

namespace oceanbase
{
namespace sql
{
namespace ai_column
{
using namespace common;
using namespace share::schema;
namespace ac = share::ai_column;
namespace
{
std::string text(const ObString &s)
{
  return s.empty() ? std::string() : std::string(s.ptr(), s.length());
}
std::string node_text(const ParseNode &node)
{
  return node.str_value_ == nullptr ? std::string() : std::string(node.str_value_, node.str_len_);
}
std::string lower(std::string value)
{
  for (char &c : value)
    if (c >= 'A' && c <= 'Z')
      c += 'a' - 'A';
  return value;
}
void reject(const std::string &message)
{
  throw ac::Error(OB_NOT_SUPPORTED, "AI COLUMN: " + message);
}
void dependencies(const ParseNode &node, std::vector<ac::Dependency> &found, unsigned depth = 0)
{
  if (depth > 64)
    reject("expression nesting exceeds 64");
  if (node.type_ == T_SELECT || node.type_ == T_REF_QUERY || node.type_ == T_OP_GET_USER_VAR ||
      node.type_ == T_OP_ASSIGN)
    reject("subqueries and user variables are unsupported");
  if (node.type_ == T_COLUMN_REF)
  {
    if (node.num_child_ != 3 || node.children_[2] == nullptr || node.children_[0] != nullptr ||
        node.children_[1] != nullptr)
      reject("dependencies must be unqualified ordinary columns");
    std::string name = node_text(*node.children_[2]);
    if (std::none_of(found.begin(), found.end(),
                     [&](const ac::Dependency &d) { return lower(d.name) == lower(name); }))
      found.push_back({name, 0});
    return;
  }
  if (node.type_ == T_FUN_SYS && node.num_child_ > 0 && node.children_[0] != nullptr)
  {
    std::string function = lower(node_text(*node.children_[0]));
    if (function == "ai_complete" || function == "ai_embed" || function == "ai_rerank" ||
        function == "rand" || function == "uuid" || function == "sleep" || function == "now")
      reject("input expressions must be deterministic and cannot invoke a model");
  }
  for (int i = 0; i < node.num_child_; ++i)
    if (node.children_[i] != nullptr)
      dependencies(*node.children_[i], found, depth + 1);
}
ac::ResultType result_type(const ObColumnSchemaV2 &column, int &dimension)
{
  auto type = column.get_data_type();
  if (type == ObJsonType)
    return ac::ResultType::Json;
  if (ob_is_integer_type(type))
    return ob_is_unsigned_type(type) ? ac::ResultType::Unsigned : ac::ResultType::Integer;
  if (ob_is_numeric_type(type))
    return ac::ResultType::Double;
  if (ob_is_string_or_lob_type(type))
    return ac::ResultType::Text;
  if (type == ObDateType)
    return ac::ResultType::Date;
  if (type == ObDateTimeType || type == ObTimestampType)
    return ac::ResultType::Datetime;
  if (column.is_collection() && column.get_extended_type_info().count() == 1)
  {
    std::string info = lower(text(column.get_extended_type_info().at(0)));
    if (info.compare(0, 7, "vector(") != 0 || info.back() != ')')
      reject("only fixed-dimension VECTOR is supported");
    auto parsed = std::from_chars(info.data() + 7, info.data() + info.size() - 1, dimension);
    if (parsed.ec != std::errc() || parsed.ptr != info.data() + info.size() - 1 || dimension <= 0)
      reject("invalid VECTOR dimension");
    return ac::ResultType::Vector;
  }
  reject("unsupported result type");
  return ac::ResultType::Text;
}
ObColumnSchemaV2 *find_column(ObTableSchema &schema, const std::string &name)
{
  ObString value(name.size(), name.data());
  for (auto it = schema.column_begin(); it != schema.column_end(); ++it)
    if ((*it)->get_column_name_str().case_compare(value) == 0)
      return *it;
  return nullptr;
}
void store(ObColumnSchemaV2 &column, const ac::Definition &definition)
{
  std::string encoded = ac::encode(definition);
  int ret = column.set_ai_column_info(ObString(encoded.size(), encoded.data()));
  if (ret != OB_SUCCESS)
    throw ac::Error(ret, "AI COLUMN: cannot allocate definition");
  column.add_column_flag(AI_COLUMN_FLAG);
}
} // namespace
int resolve_attribute(const ParseNode &attribute, ObColumnSchemaV2 &column)
{
  int ret = OB_SUCCESS;
  try
  {
    if (attribute.type_ != T_AI_COLUMN || attribute.num_child_ != 2 ||
        attribute.children_[0] == nullptr)
      reject("malformed or duplicate declaration");
    const ParseNode &expression = *attribute.children_[0];
    if (expression.type_ != T_FUN_SYS || expression.num_child_ < 2 ||
        expression.children_[0] == nullptr || expression.children_[1] == nullptr)
      reject("expected AI_COMPLETE or AI_EMBED");
    std::string function = lower(node_text(*expression.children_[0]));
    if (function != "ai_complete" && function != "ai_embed")
      reject("expected AI_COMPLETE or AI_EMBED");
    const ParseNode &arguments = *expression.children_[1];
    if (arguments.num_child_ < 2 || arguments.num_child_ > 3 || arguments.children_[0] == nullptr ||
        arguments.children_[1] == nullptr)
      reject("invalid AI function arguments");
    const ParseNode &model = *arguments.children_[0];
    if ((model.type_ != T_VARCHAR && model.type_ != T_CHAR) || model.str_len_ == 0)
      reject("model must be a nonempty string literal");
    ac::Definition d;
    d.expression = node_text(expression);
    d.model = node_text(model);
    d.embedding = function == "ai_embed";
    auto values = ac::arguments(d.expression);
    if (values.size() != static_cast<size_t>(arguments.num_child_))
      reject("invalid argument text");
    d.input_expression = values[1];
    d.config_expression = values.size() == 3 && !d.embedding ? values[2] : "NULL";
    ObArenaAllocator model_allocator("AIColumnDDL");
    ObAIFuncExprInfo *model_info = nullptr;
    if ((ret = ObAIFuncUtils::get_ai_func_info(
             model_allocator, ObString(d.model.size(), d.model.data()), model_info)) != OB_SUCCESS)
      return ret;
    if (model_info == nullptr || (d.embedding ? !ObAIFuncUtils::is_dense_embedding_type(model_info)
                                              : !ObAIFuncUtils::is_completion_type(model_info)))
      reject("model type does not match the AI function");
    d.type = result_type(column, d.dimension);
    switch (column.get_data_type())
    {
    case ObTinyIntType:
    case ObUTinyIntType:
      d.integer_bits = 8;
      break;
    case ObSmallIntType:
    case ObUSmallIntType:
      d.integer_bits = 16;
      break;
    case ObMediumIntType:
    case ObUMediumIntType:
      d.integer_bits = 24;
      break;
    case ObInt32Type:
    case ObUInt32Type:
      d.integer_bits = 32;
      break;
    default:
      break;
    }
    if (d.embedding != (d.type == ac::ResultType::Vector))
      reject("AI_EMBED requires VECTOR; AI_COMPLETE requires a scalar result column");
    if (d.embedding && arguments.num_child_ == 3)
    {
      const ParseNode &dim = *arguments.children_[2];
      if (dim.type_ != T_INT || dim.value_ <= 0 || dim.value_ != d.dimension)
      {
        ret = OB_ERR_INVALID_VECTOR_DIM;
        LOG_USER_ERROR(OB_ERR_INVALID_VECTOR_DIM, d.dimension, static_cast<unsigned>(dim.value_));
        return ret;
      }
    }
    dependencies(*arguments.children_[1], d.dependencies);
    if (arguments.num_child_ == 3 && !d.embedding)
      dependencies(*arguments.children_[2], d.dependencies);
    if (attribute.children_[1] != nullptr)
    {
      std::set<std::string> seen;
      for (int i = 0; i < attribute.children_[1]->num_child_; ++i)
      {
        const ParseNode *option = attribute.children_[1]->children_[i];
        if (option == nullptr || option->num_child_ != 2 || option->children_[0] == nullptr ||
            option->children_[1] == nullptr)
          reject("malformed option");
        std::string key = lower(node_text(*option->children_[0]));
        const ParseNode &value = *option->children_[1];
        if (!seen.insert(key).second)
          reject("duplicate option");
        if (key == "sync_mode")
        {
          if ((value.type_ != T_VARCHAR && value.type_ != T_CHAR) ||
              lower(node_text(value)) != "async")
            reject("only sync_mode='async' is supported");
        }
        else if (key == "max_retries")
        {
          if (value.type_ != T_INT || value.value_ < 0 || value.value_ > 100)
            reject("max_retries must be between 0 and 100");
          d.max_retries = value.value_;
        }
        else
          reject("unknown option: " + key);
      }
    }
    static std::atomic<uint64_t> sequence{0};
    d.version =
        std::to_string(ObTimeUtility::current_time()) + "_" + std::to_string(sequence.fetch_add(1));
    d.state_column_name = "__ai_s_" + d.version;
    d.request_column_name = "__ai_r_" + d.version;
    d.result_spec_column_name = "__ai_v_" + d.version;
    d.length = std::max(0, column.get_data_length());
    d.scale = std::max(0, static_cast<int>(column.get_data_scale()));
    if (d.type != ac::ResultType::Text)
      d.length = 0;
    d.type_prompt = ac::instruction(d.type);
    d.normalized_expression =
        d.expression + (d.type_prompt.empty() ? "" : " /* " + d.type_prompt + " */");
    store(column, d);
  }
  catch (const ac::Error &error)
  {
    ret = error.code;
    if (ret == OB_NOT_SUPPORTED)
      LOG_USER_ERROR(OB_NOT_SUPPORTED, error.what());
    else
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, error.what());
  }
  catch (const std::bad_alloc &)
  {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  }
  return ret;
}
int finalize_schema(ObTableSchema &schema, const ObTableSchema *original)
{
  int ret = OB_SUCCESS;
  try
  {
    std::vector<std::string> names;
    for (auto it = schema.column_begin(); it != schema.column_end(); ++it)
      if ((*it)->is_ai_column())
      {
        names.push_back(text((*it)->get_column_name_str()));
      }
    size_t count = names.size();
    if (original != nullptr)
      for (auto it = original->column_begin(); it != original->column_end(); ++it)
      {
        if ((*it)->is_ai_column() &&
            std::none_of(names.begin(), names.end(), [&](const std::string &name)
                         { return lower(name) == lower(text((*it)->get_column_name_str())); }))
          ++count;
      }
    if (count > ac::MAX_COLUMNS)
      reject("at most 16 AI columns are supported per table");
    uint64_t next =
        original == nullptr ? schema.get_max_used_column_id() : original->get_max_used_column_id();
    std::map<std::string, uint64_t> added_ids;
    if (original != nullptr)
      for (auto it = schema.column_begin(); it != schema.column_end(); ++it)
      {
        const auto *added = static_cast<const AlterColumnSchema *>(*it);
        if (added->alter_type_ == OB_DDL_ADD_COLUMN)
          added_ids[lower(text(added->get_column_name_str()))] = ++next;
      }
    for (const auto &name : names)
    {
      ObColumnSchemaV2 *column = find_column(schema, name);
      if (column == nullptr)
        reject("missing result column");
      if (!column->is_nullable() || column->is_generated_column() || column->is_rowkey_column())
        reject("AI result columns must be nullable physical non-key columns");
      ac::Definition d = ac::decode(text(column->get_ai_column_info_str()));
      for (auto &dep : d.dependencies)
      {
        const ObColumnSchemaV2 *source = find_column(schema, dep.name);
        if (source == nullptr && original != nullptr)
          source = original->get_column_schema(ObString(dep.name.size(), dep.name.data()));
        if (source == nullptr)
          reject("unknown dependency: " + dep.name);
        if (source->is_ai_column() || source->is_hidden())
        {
          ret = OB_NOT_SUPPORTED;
          FORWARD_USER_ERROR(ret, "Not supported feature or function");
          return ret;
        }
        dep.id = source->get_column_id();
        dep.name = text(source->get_column_name_str());
        auto added = added_ids.find(lower(dep.name));
        if (added != added_ids.end())
          dep.id = added->second;
      }
      std::string internal_names[] = {d.state_column_name, d.request_column_name,
                                      d.result_spec_column_name};
      uint64_t *internal_ids[] = {&d.state_column_id, &d.request_column_id,
                                  &d.result_spec_column_id};
      for (int i = 0; i < 3; ++i)
      {
        const auto *existing =
            original == nullptr ? nullptr
                                : original->get_column_schema(
                                      ObString(internal_names[i].size(), internal_names[i].data()));
        if (existing != nullptr)
        {
          *internal_ids[i] = existing->get_column_id();
          continue;
        }
        ObColumnSchemaV2 hidden;
        hidden.set_table_id(schema.get_table_id());
        hidden.set_column_id(original == nullptr ? ++next : OB_INVALID_ID);
        hidden.set_data_type(ObVarcharType);
        hidden.set_data_length(i == 0 ? 16 : 128);
        hidden.set_nullable(true);
        hidden.set_is_hidden(true);
        hidden.add_column_flag(AI_INTERNAL_COLUMN_FLAG);
        hidden.set_charset_type(CHARSET_UTF8MB4);
        hidden.set_collation_type(CS_TYPE_UTF8MB4_BIN);
        if ((ret = hidden.set_column_name(
                 ObString(internal_names[i].size(), internal_names[i].data()))) != OB_SUCCESS)
          throw ac::Error(ret, "hidden name allocation");
        if (original == nullptr)
          ret = schema.add_column(hidden);
        else
        {
          AlterColumnSchema added;
          ret = added.ObColumnSchemaV2::assign(hidden);
          added.alter_type_ = OB_DDL_ADD_COLUMN;
          if (ret == OB_SUCCESS)
            ret = static_cast<AlterTableSchema &>(schema).add_alter_column(added, true);
        }
        if (ret != OB_SUCCESS)
          throw ac::Error(ret, "cannot append hidden state column");
        *internal_ids[i] = original == nullptr ? hidden.get_column_id() : ++next;
      }
      store(*column, d);
    }
    if (original == nullptr && !names.empty())
      schema.set_max_used_column_id(next);
  }
  catch (const ac::Error &error)
  {
    ret = error.code;
    LOG_USER_ERROR(OB_NOT_SUPPORTED, error.what());
  }
  catch (const std::bad_alloc &)
  {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  }
  return ret;
}
} // namespace ai_column
} // namespace sql
} // namespace oceanbase
