/*
 * Copyright (c) 2025 OceanBase.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define USING_LOG_PREFIX SQL_RESV

#include "sql/table_function/file_sql_bind.h"
#include "sql/resolver/dml/ob_dml_resolver.h"
#include "sql/resolver/expr/ob_raw_expr_util.h"
#include "sql/resolver/ob_resolver_utils.h"
#include "sql/table_function/table_function.h"
#include "sql/table_function/file_sql_utils.h"
#include "sql/session/ob_basic_session_info.h"
#include "sql/parser/parse_node.h"
#include "common/object/ob_object.h"
#include <string.h>
#include <strings.h>
#include <string>

namespace oceanbase
{
namespace sql
{
using namespace common;

namespace
{

bool is_named_argument(const ParseNode *node)
{
  return NULL != node && T_OP_EQ == node->type_ && 2 == node->num_child_ &&
         NULL != node->children_[0] && T_IDENT == node->children_[0]->type_;
}

bool option_name_is(const ParseNode *node, const char *name)
{
  const ParseNode *name_node = node->children_[0];
  const int64_t length = static_cast<int64_t>(strlen(name));
  return length == static_cast<int64_t>(name_node->str_len_) &&
         0 == strncasecmp(name_node->str_value_, name, static_cast<size_t>(length));
}

int file_kind_to_data_type(file_sql::ColumnKind kind, ObDataType &data_type)
{
  int ret = OB_SUCCESS;
  switch (kind) {
    case file_sql::COLUMN_KIND_BIGINT:
      data_type.set_int();
      break;
    case file_sql::COLUMN_KIND_DOUBLE:
      data_type.set_double();
      break;
    case file_sql::COLUMN_KIND_BOOLEAN:
      data_type.set_tinyint();
      break;
    case file_sql::COLUMN_KIND_DATE:
      data_type.set_date();
      break;
    case file_sql::COLUMN_KIND_DATETIME:
      data_type.set_datetime();
      break;
    default:
      data_type.set_varchar();
      break;
  }
  return ret;
}

char file_kind_to_char(file_sql::ColumnKind kind)
{
  switch (kind) {
    case file_sql::COLUMN_KIND_BIGINT: return 'i';
    case file_sql::COLUMN_KIND_DOUBLE: return 'd';
    case file_sql::COLUMN_KIND_BOOLEAN: return 'b';
    case file_sql::COLUMN_KIND_DATE: return 'a';
    case file_sql::COLUMN_KIND_DATETIME: return 't';
    case file_sql::COLUMN_KIND_VARCHAR: return 's';
    default: return 'n';
  }
}

} // namespace

int bind_read_csv(ObDMLResolver &resolver, const TableFunctionCall &call, TableItem *&item)
{
  return resolver.bind_file_table_function(call, item, file_sql::FILE_TYPE_CSV);
}

int bind_read_ndjson(ObDMLResolver &resolver, const TableFunctionCall &call, TableItem *&item)
{
  return resolver.bind_file_table_function(call, item, file_sql::FILE_TYPE_JSONL);
}

int bind_read_json(ObDMLResolver &resolver, const TableFunctionCall &call, TableItem *&item)
{
  return resolver.bind_file_table_function(call, item, file_sql::FILE_TYPE_JSONL);
}

int bind_read_parquet(ObDMLResolver &resolver, const TableFunctionCall &call, TableItem *&item)
{
  return resolver.bind_file_table_function(call, item, file_sql::FILE_TYPE_PARQUET);
}

int bind_glob(ObDMLResolver &resolver, const TableFunctionCall &call, TableItem *&item)
{
  return resolver.bind_glob_table_function(call, item);
}

int ObDMLResolver::bind_glob_table_function(const TableFunctionCall &call, TableItem *&item)
{
  int ret = OB_SUCCESS;
  if (OB_FAIL(create_function_scan_item(call, item))) {
  } else {
    // glob exposes a single VARCHAR column named "file".
    ObDataType data_type;
    ObAccuracy accuracy;
    ColumnItem *column = NULL;
    data_type.set_varchar();
    accuracy.set_length(65536);
    if (OB_FAIL(resolve_function_table_column_item(*item, data_type.get_meta_type(),
              accuracy, ObString::make_string("file"), OB_APP_MIN_COLUMN_ID, column))) {
    }
  }
  return ret;
}

int ObDMLResolver::eval_file_table_function_const(const ParseNode &node, ObObj &value)
{
  int ret = OB_SUCCESS;
  ObRawExpr *expr = NULL;
  if (OB_FAIL(resolve_sql_expr(node, expr))) {
  } else if (OB_FAIL(expr->deduce_type(session_info_))) {
  } else if (OB_FAIL(expr->extract_info())) {
  } else {
    const ParamStore no_parameters;
    const ParamStore &parameters = params_.param_list_ == NULL ? no_parameters : *params_.param_list_;
    if (OB_FAIL(ObSQLUtils::calc_const_expr(session_info_, *expr, value, *allocator_, parameters,
                                            session_info_->get_cur_exec_ctx()))) {
    }
  }
  return ret;
}

int ObDMLResolver::bind_file_table_function(const TableFunctionCall &call, TableItem *&item,
                                            int64_t file_type)
{
  int ret = OB_SUCCESS;
  bool enabled = false;
  file_sql::Options options;
  file_sql::Schema schema;
  ObString file_path;
  ObObj options_text;
  if (OB_ISNULL(session_info_) || OB_ISNULL(call.arguments_) ||
      OB_ISNULL(get_stmt()) || OB_ISNULL(params_.expr_factory_)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (file_type < static_cast<int64_t>(file_sql::FILE_TYPE_INVALID) ||
             file_type > static_cast<int64_t>(file_sql::FILE_TYPE_PARQUET)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_FAIL(session_info_->get_enable_file_sql(enabled))) {
  } else if (!enabled) {
    ret = OB_NOT_SUPPORTED;
    LOG_USER_ERROR(OB_NOT_SUPPORTED, "file SQL requires enable_file_sql");
  } else if (0 == call.arguments_->num_child_ || NULL == call.arguments_->children_[0]) {
    ret = OB_ERR_PARAM_SIZE;
    LOG_USER_ERROR(OB_ERR_PARAM_SIZE, "file SQL table function");
  } else {
    options.type_ = static_cast<file_sql::FileType>(file_type);
    // Positional arguments: the path is the only positional argument. A second
    // positional argument is a syntax error, matching DuckDB's named options.
    for (int64_t i = 0; OB_SUCC(ret) && i < call.arguments_->num_child_; ++i) {
      const ParseNode *argument = call.arguments_->children_[i];
      if (0 == i) {
        ObObj value;
        if (is_named_argument(argument)) {
          ret = OB_ERR_PARSE;
        } else if (OB_FAIL(eval_file_table_function_const(*argument, value))) {
        } else if (!value.is_string_type()) {
          ret = OB_ERR_INVALID_TYPE_FOR_OP;
        } else {
          file_path = value.get_string();
        }
      } else if (!is_named_argument(argument)) {
        // A second positional argument (e.g. read_csv(path, 'xml')) is rejected
        // as a syntax error by DuckDB and by this implementation.
        ret = OB_ERR_PARSE;
      } else if (option_name_is(argument, "header")) {
        ObObj value;
        if (OB_FAIL(eval_file_table_function_const(*argument->children_[1], value))) {
        } else if (value.is_null()) {
          ret = OB_ERR_INVALID_TYPE_FOR_OP;
        } else {
          options.header_ = value.get_bool();
        }
      } else if (option_name_is(argument, "sample_size")) {
        ObObj value;
        if (OB_FAIL(eval_file_table_function_const(*argument->children_[1], value))) {
        } else if (!value.is_integer_type()) {
          ret = OB_ERR_INVALID_TYPE_FOR_OP;
        } else if (-1 != static_cast<int64_t>(value.get_int())) {
          ret = OB_NOT_SUPPORTED;
          LOG_USER_ERROR(OB_NOT_SUPPORTED, "sample_size other than -1");
        }
      } else if (option_name_is(argument, "format")) {
        ObObj value;
        if (OB_FAIL(eval_file_table_function_const(*argument->children_[1], value))) {
        } else if (!value.is_string_type()) {
          ret = OB_ERR_INVALID_TYPE_FOR_OP;
        } else {
          const ObString format = value.get_string();
          const char *text = format.ptr();
          const int64_t length = format.length();
          if ((10 == length && 0 == strncasecmp(text, "newline_delimited", 10)) ||
              (6 == length && 0 == strncasecmp(text, "ndjson", 6))) {
            options.type_ = file_sql::FILE_TYPE_JSONL;
          } else if (3 == length && 0 == strncasecmp(text, "csv", 3)) {
            options.type_ = file_sql::FILE_TYPE_CSV;
          } else if (7 == length && 0 == strncasecmp(text, "parquet", 7)) {
            options.type_ = file_sql::FILE_TYPE_PARQUET;
          } else {
            ret = OB_ERR_PARSE;
          }
        }
      } else {
        ret = OB_NOT_SUPPORTED;
        LOG_USER_ERROR(OB_NOT_SUPPORTED, "this file SQL option");
      }
    }
  }

  if (OB_SUCC(ret)) {
    if (OB_FAIL(file_sql::check_extension_supported(std::string(file_path.ptr(),
                 static_cast<size_t>(file_path.length())), options))) {
      ret = OB_NOT_SUPPORTED;
      LOG_USER_ERROR(OB_NOT_SUPPORTED, "this file extension");
    } else {
      ObString secure_file_priv;
      if (OB_FAIL(session_info_->get_secure_file_priv(secure_file_priv))) {
      } else if (OB_FAIL(ObResolverUtils::check_secure_path(secure_file_priv, file_path))) {
        LOG_WARN("file path is outside secure_file_priv", K(ret), K(secure_file_priv), K(file_path));
        LOG_USER_ERROR(OB_NOT_SUPPORTED, "file path outside secure_file_priv");
        ret = OB_NOT_SUPPORTED;
      }
    }
  }

  if (OB_SUCC(ret)) {
    const std::string path(file_path.ptr(), static_cast<size_t>(file_path.length()));
    file_sql::FileStamp stamp;
    if (OB_FAIL(file_sql::infer_schema(path, options, schema))) {
      LOG_WARN("failed to infer file schema", K(ret), K(path));
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "failed to read file schema");
      ret = OB_INVALID_ARGUMENT;
    } else if (schema.columns_.empty()) {
      ret = OB_INVALID_ARGUMENT;
      LOG_USER_ERROR(OB_INVALID_ARGUMENT, "file has no columns");
    } else if (schema.columns_.size() > 64) {
      ret = OB_NOT_SUPPORTED;
      LOG_USER_ERROR(OB_NOT_SUPPORTED, "files with more than 64 columns");
    } else if (OB_FAIL(file_sql::read_file_stamp(path, stamp))) {
      ret = OB_INVALID_ARGUMENT;
    } else {
      std::string serialized;
      serialized += "type=" + std::string(1, static_cast<char>('0' + file_type)) + ";";
      serialized += std::string("header=") + (options.header_ ? "1" : "0") + ";";
      serialized += "size=" + file_sql::format_int64(stamp.size_) + ";";
      serialized += "mtime=" + file_sql::format_int64(stamp.mtime_) + ";";
      serialized += "cols=";
      for (size_t i = 0; i < schema.columns_.size(); ++i) {
        serialized += file_kind_to_char(schema.columns_[i].kind_);
      }
      char *buffer = static_cast<char *>(allocator_->alloc(static_cast<int64_t>(serialized.size())));
      if (NULL == buffer) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
      } else {
        MEMCPY(buffer, serialized.data(), serialized.size());
        options_text.set_varchar(ObString(static_cast<int32_t>(serialized.size()), buffer));
        options_text.set_collation_type(CS_TYPE_UTF8MB4_BIN);
      }
    }
  }

  if (OB_SUCC(ret)) {
    ObOpRawExpr *arguments = NULL;
    ObRawExpr *path_expr = NULL;
    ObConstRawExpr *options_expr = NULL;
    if (OB_FAIL(params_.expr_factory_->create_raw_expr(T_OP_ROW, arguments))) {
    } else if (OB_FAIL(arguments->init_param_exprs(2))) {
    } else if (OB_FAIL(resolve_sql_expr(*call.arguments_->children_[0], path_expr))) {
    } else if (OB_FAIL(path_expr->deduce_type(session_info_))) {
    } else if (OB_FAIL(ObRawExprUtils::build_const_string_expr(*params_.expr_factory_,
                 ObVarcharType, options_text.get_string(), CS_TYPE_UTF8MB4_BIN, options_expr))) {
    } else if (OB_FAIL(arguments->add_param_expr(path_expr))) {
    } else if (OB_FAIL(arguments->add_param_expr(options_expr))) {
    } else if (OB_FAIL(arguments->extract_info())) {
    } else if (OB_ISNULL(item = get_stmt()->create_table_item(*allocator_))) {
      ret = OB_ALLOCATE_MEMORY_FAILED;
    } else {
      item->type_ = TableItem::FUNCTION_TABLE;
      item->table_function_id_ = call.descriptor_->id_;
      item->table_function_requires_all_columns_ = call.descriptor_->requires_all_columns_;
      item->table_id_ = generate_table_id();
      item->table_name_ = ObString::make_string(call.descriptor_->name_);
      item->alias_name_ = call.alias_ == NULL ? item->table_name_ :
          ObString(call.alias_->str_len_, call.alias_->str_value_);
      item->function_table_expr_ = arguments;
      if (OB_FAIL(get_stmt()->add_table_item(session_info_, item))) {
      } else {
        for (int64_t i = 0; OB_SUCC(ret) && i < static_cast<int64_t>(schema.columns_.size()); ++i) {
          ObDataType data_type;
          ObAccuracy accuracy;
          ColumnItem *column = NULL;
          if (OB_FAIL(file_kind_to_data_type(schema.columns_[i].kind_, data_type))) {
          } else {
            ObString column_name;
            if (OB_FAIL(ob_write_string(*allocator_, ObString(static_cast<int32_t>(
                       schema.columns_[i].name_.size()), schema.columns_[i].name_.data()),
                       column_name))) {
            } else if (OB_FAIL(resolve_function_table_column_item(*item, data_type.get_meta_type(),
                       accuracy, column_name, OB_APP_MIN_COLUMN_ID + i, column))) {
            } else if (OB_SUCC(ret) && !schema.columns_[i].nullable_) {
              column->expr_->set_result_flag(NOT_NULL_FLAG);
            }
          }
        }
        if (OB_SUCC(ret) && OB_FAIL(apply_table_function_aliases(call, *item))) {
        }
      }
    }
  }
  return ret;
}

} // namespace sql
} // namespace oceanbase
