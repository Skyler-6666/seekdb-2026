/* Copyright (c) 2026 OceanBase. Licensed under the Apache License, Version 2.0. */
#define USING_LOG_PREFIX SQL_ENG
#include "ob_json_table_op.h"
#include "file_sql_engine.h"
#include "common/timezone/ob_time_convert.h"
#include <charconv>

namespace oceanbase { namespace sql {
using namespace common;
int ObJsonTableOp::open_file_scan()
{
  int ret = OB_SUCCESS;
  const auto &spec = static_cast<const ObJsonTableSpec &>(spec_);
  ObString secure_root;
  ObDatum *definition = nullptr;
  if (OB_ISNULL(ctx_.get_my_session())) ret = OB_ERR_UNEXPECTED;
  else if (OB_FAIL(file_sql::access_root(*ctx_.get_my_session(), *allocator_, secure_root))) {
  } else if (spec.value_exprs_.count() != 1) ret = OB_ERR_UNEXPECTED;
  else if (OB_FAIL(spec.value_exprs_[0]->eval(eval_ctx_, definition))) {
  } else if (OB_ISNULL(definition) || definition->is_null()) ret = OB_INVALID_ARGUMENT;
  else {
    try {
      ObString encoded = definition->get_string();
      file_plan_ = file_sql::decode(std::string(encoded.ptr(), encoded.length()));
      file_reader_.reset(new file_sql::Reader(file_plan_, std::string(secure_root.ptr(), secure_root.length()), [this] {
        if (ctx_.check_status() != OB_SUCCESS) throw file_sql::FileSqlError("file SQL: query canceled during scan");
      }));
      std::vector<size_t> projection;
      for (int64_t i = 1; i < spec.cols_def_.count(); ++i) {
        if (spec.cols_def_[i] != nullptr && spec.cols_def_[i]->output_column_idx_ >= 0) projection.push_back(i - 1);
      }
      file_reader_->project(projection);
    } catch (const std::bad_alloc &) { ret = OB_ALLOCATE_MEMORY_FAILED; }
    catch (const std::exception &error) { int status = ctx_.check_status(); ret = status == OB_SUCCESS ? OB_INVALID_ARGUMENT : status; if (status == OB_SUCCESS) LOG_USER_ERROR(OB_INVALID_ARGUMENT, error.what()); }
  }
  return ret;
}
int ObJsonTableOp::next_file_row()
{
  int ret = OB_SUCCESS;
  const auto &spec = static_cast<const ObJsonTableSpec &>(spec_);
  try {
    clear_evaluated_flag();
    if (!file_reader_) ret = OB_NOT_INIT;
    else if (!file_reader_->next(file_row_)) ret = OB_ITER_END;
    for (int64_t i = 1; OB_SUCC(ret) && i < spec.cols_def_.count(); ++i) {
      const ObJtColInfo *info = spec.cols_def_[i];
      if (OB_ISNULL(info)) ret = OB_ERR_UNEXPECTED;
      else if (info->output_column_idx_ < 0) continue;
      else if (info->output_column_idx_ >= spec.column_exprs_.count()
               || static_cast<size_t>(i - 1) >= file_row_.size()) ret = OB_ERR_UNEXPECTED;
      else {
        ObExpr *expr = spec.column_exprs_[info->output_column_idx_];
        ObDatum &datum = expr->locate_datum_for_write(eval_ctx_);
        const auto &cell = file_row_[i - 1];
        if (!cell) datum.set_null();
        else {
          ObString text(cell->size(), cell->data());
          const char *begin = cell->data() + (!cell->empty() && (*cell)[0] == '+' ? 1 : 0);
          const char *end = cell->data() + cell->size();
          switch (file_plan_.columns[i - 1].type) {
          case file_sql::Type::Null: datum.set_null(); break;
          case file_sql::Type::Boolean: datum.set_int(!cell->empty() && ((*cell)[0] == 't' || (*cell)[0] == 'T')); break;
          case file_sql::Type::Bigint: {
            int64_t value = 0;
            auto parsed = std::from_chars(begin, end, value);
            if (parsed.ec != std::errc() || parsed.ptr != end) ret = OB_INVALID_DATA;
            else datum.set_int(value);
            break;
          }
          case file_sql::Type::Double: {
            double value = 0;
            auto parsed = std::from_chars(begin, end, value);
            if (parsed.ec != std::errc() || parsed.ptr != end) ret = OB_INVALID_DATA;
            else datum.set_double(value);
            break;
          }
          case file_sql::Type::Date: {
            int32_t value = 0;
            if (OB_FAIL(ObTimeConverter::str_to_date(text, value))) {} else datum.set_date(value);
            break;
          }
          case file_sql::Type::Datetime: {
            int64_t value = 0;
            ObTimeConvertCtx convert(nullptr, false);
            if (OB_FAIL(ObTimeConverter::str_to_datetime(text, convert, value))) {} else datum.set_datetime(value);
            break;
          }
          case file_sql::Type::Varchar: {
            char *memory = expr->get_str_res_mem(eval_ctx_, cell->size());
            if (OB_ISNULL(memory)) ret = OB_ALLOCATE_MEMORY_FAILED;
            else { MEMCPY(memory, cell->data(), cell->size()); datum.set_string(memory, cell->size()); }
            break;
          }
          }
        }
        if (OB_SUCC(ret)) expr->get_eval_info(eval_ctx_).evaluated_ = true;
      }
    }
  } catch (const std::bad_alloc &) { ret = OB_ALLOCATE_MEMORY_FAILED; }
  catch (const std::exception &error) { int status = ctx_.check_status(); ret = status == OB_SUCCESS ? OB_INVALID_ARGUMENT : status; if (status == OB_SUCCESS) LOG_USER_ERROR(OB_INVALID_ARGUMENT, error.what()); }
  return ret;
}
}}
