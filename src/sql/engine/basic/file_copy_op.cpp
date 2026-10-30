/* Copyright (c) 2026 OceanBase. Licensed under the Apache License, Version 2.0. */
#define USING_LOG_PREFIX SQL_ENG
#include "ob_select_into_op.h"
#include "file_copy_writer.h"
#include "file_sql_engine.h"
#include "share/ob_lob_access_utils.h"
#include "sql/engine/expr/ob_expr_lob_utils.h"

namespace oceanbase { namespace sql {
using namespace common;
int ObSelectIntoOp::run_file_copy(bool batch)
{
  int ret = OB_SUCCESS;
  if (!is_first_) return batch ? OB_SUCCESS : OB_ITER_END;
  is_first_ = false;
  const auto &spec = static_cast<const ObSelectIntoSpec &>(spec_);
  ObString root;
  if (OB_ISNULL(ctx_.get_my_session()) || OB_ISNULL(child_) || OB_ISNULL(ctx_.get_physical_plan_ctx())) ret = OB_ERR_UNEXPECTED;
  else if (OB_FAIL(file_sql::access_root(*ctx_.get_my_session(), ctx_.get_allocator(), root))) {
  } else {
    try {
      ObString encoded = spec.outfile_name_.get_string();
      file_sql::Plan plan = file_sql::decode(std::string(encoded.ptr(), encoded.length()));
      file_sql::CsvCopyWriter writer(plan, std::string(root.ptr(), root.length()));
      int64_t count = 0;
      bool finished = false;
      ObArenaAllocator temporary("FileCopy");
      auto write_row = [&]() {
        file_sql::Row row;
        temporary.reuse();
        for (int64_t i = 0; OB_SUCC(ret) && i < spec.select_exprs_.count(); ++i) {
          ObExpr *expr = spec.select_exprs_[i]; ObDatum *datum = nullptr;
          if (OB_FAIL(expr->eval(eval_ctx_, datum))) {
          } else if (datum->is_null()) row.emplace_back(std::nullopt);
          else if (ob_is_string_or_lob_type(expr->datum_meta_.type_)) {
            ObString value;
            if (OB_FAIL(ObTextStringHelper::read_real_string_data(ctx_, temporary, *datum,
                       expr->datum_meta_, expr->obj_meta_.has_lob_header(), value))) {
            } else row.emplace_back(std::string(value.ptr(), value.length()));
          } else {
            ObObj object;
            char buffer[1024]; int64_t position = 0;
            ObObjPrintParams params; params.tz_info_ = ctx_.get_my_session()->get_timezone_info();
            if (OB_FAIL(datum->to_obj(object, expr->obj_meta_))) {
            } else if (OB_FAIL(object.print_plain_str_literal(buffer, sizeof(buffer), position, params))) {
            } else row.emplace_back(std::string(buffer, position));
          }
        }
        if (OB_SUCC(ret)) { writer.write(row); ++count; }
      };
      while (OB_SUCC(ret) && !finished) {
        clear_evaluated_flag();
        if (OB_FAIL(ctx_.check_status())) {
        } else if (batch) {
          const ObBatchRows *rows = nullptr;
          if (OB_FAIL(child_->get_next_batch(spec.max_batch_size_, rows))) {
            if (ret == OB_ITER_END) { ret = OB_SUCCESS; finished = true; }
          } else if (OB_ISNULL(rows)) ret = OB_ERR_UNEXPECTED;
          else {
            ObEvalCtx::BatchInfoScopeGuard guard(eval_ctx_);
            guard.set_batch_size(rows->size_);
            for (int64_t i = 0; OB_SUCC(ret) && i < rows->size_; ++i) {
              if (!rows->skip_->contain(i)) { guard.set_batch_idx(i); write_row(); }
            }
            finished = rows->end_;
          }
        } else if (OB_FAIL(child_->get_next_row())) {
          if (ret == OB_ITER_END) { ret = OB_SUCCESS; finished = true; }
        } else write_row();
      }
      if (OB_SUCC(ret) && OB_SUCC(ctx_.check_status())) {
        writer.commit(); ctx_.get_physical_plan_ctx()->set_affected_rows(count);
      } else if (OB_SUCC(ret)) ret = ctx_.check_status();
    } catch (const std::bad_alloc &) { ret = OB_ALLOCATE_MEMORY_FAILED; }
    catch (const std::exception &error) { ret = OB_INVALID_ARGUMENT; LOG_USER_ERROR(OB_INVALID_ARGUMENT, error.what()); }
  }
  brs_.size_ = 0; brs_.end_ = true;
  return OB_SUCC(ret) && !batch ? OB_ITER_END : ret;
}
}} // namespace oceanbase::sql
