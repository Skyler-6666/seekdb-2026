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

#define USING_LOG_PREFIX STORAGE

#include "ob_text_daat_iter.h"

namespace oceanbase
{
namespace storage
{
int ObTextDaaTIter::init(const ObTextDaaTParam &param)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(param.base_param_) || OB_ISNULL(param.dim_iters_) || OB_ISNULL(param.allocator_)
      || OB_ISNULL(param.relevance_collector_)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(ObSRDaaTIterImpl::init(*param.base_param_, *param.dim_iters_,
                                            *param.allocator_, *param.relevance_collector_))) {
  } else if (OB_FAIL(bm25_param_estimator_.init(param.bm25_param_est_ctx_))) {
  } else {
    mode_flag_ = param.mode_flag_;
    function_lookup_mode_ = param.function_lookup_mode_;
    const ObDatumMeta &id_meta = param.base_param_->id_proj_expr_->datum_meta_;
    const bool supported_id = ObUInt64Type == id_meta.type_ || ObIntType == id_meta.type_
        || (CS_TYPE_BINARY == id_meta.cs_type_
            && (ObVarcharType == id_meta.type_ || ObCharType == id_meta.type_));
    doc_only_union_ = param.doc_only_union_allowed_ && supported_id
        && sql::NATURAL_LANGUAGE_MODE == mode_flag_ && !function_lookup_mode_
        && nullptr == param.base_param_->relevance_expr_
        && nullptr == param.base_param_->dim_weights_
        && param.base_param_->match_filter_trivial_
        && param.base_param_->eval_ctx_->is_vectorized()
        && param.dim_iters_->count() > 0 && param.dim_iters_->count() <= MAX_DOC_ONLY_UNION_DIMS;
    cardinality_only_ = doc_only_union_ && param.cardinality_only_
        && !param.base_param_->limit_param_->is_valid() && 0 == param.base_param_->limit_param_->offset_;
    if (doc_only_union_) {
      doc_only_uint64_ = ObUInt64Type == id_meta.type_;
      doc_only_int64_ = ObIntType == id_meta.type_;
      // Use exactly the comparator used by the heap, including encoded rowkeys.
      doc_only_cmp_ = ObDatumFuncs::get_basic_func(id_meta.type_, id_meta.cs_type_)->null_first_cmp_;
      if (OB_ISNULL(doc_only_cmp_)) {
        ret = OB_ERR_UNEXPECTED;
      } else if (CS_TYPE_BINARY == id_meta.cs_type_
          && (ObVarcharType == id_meta.type_ || ObCharType == id_meta.type_)) {
        const ObCharsetInfo *charset = ObCharset::get_charset(CS_TYPE_BINARY);
        if (nullptr != charset && nullptr != charset->coll
            && nullptr != charset->coll->strnncollsp) {
          doc_only_charset_ = charset;
        }
      }
    }
  }
  return ret;
}

void ObTextDaaTIter::reset()
{
  doc_only_union_ = false;
  cardinality_only_ = false;
  doc_only_uint64_ = false;
  doc_only_int64_ = false;
  doc_only_cmp_ = nullptr;
  doc_only_charset_ = nullptr;
  for (int64_t i = 0; i < MAX_DOC_ONLY_UNION_DIMS; ++i) {
    doc_only_cursors_[i] = DocOnlyCursor();
  }
  bm25_param_estimator_.reset();
  // this class can know the type of dim_iters_, but ObSRDaaTIterImpl maybe not
  if (OB_NOT_NULL(dim_iters_)) {
    for (int64_t i = 0; i < dim_iters_->count(); ++i) {
      static_cast<ObTextRetrievalDaaTTokenIter *>(dim_iters_->at(i))->reset();
    }
  }
  ObSRDaaTIterImpl::reset();
}

void ObTextDaaTIter::reuse(const bool switch_tablet)
{
  // The expression type and immutable collation remain valid across rescans;
  // only posting positions need resetting. reset() clears the cached collation.
  for (int64_t i = 0; i < MAX_DOC_ONLY_UNION_DIMS; ++i) {
    doc_only_cursors_[i] = DocOnlyCursor();
  }
  if ((OB_NOT_NULL(dim_iters_) && 0 == dim_iters_->count())) {
    // do nothing
  } else {
    bm25_param_estimator_.reuse(switch_tablet);
  }
  if (OB_NOT_NULL(dim_iters_)) {
    for (int64_t i = 0; i < dim_iters_->count(); ++i) {
      static_cast<ObTextRetrievalDaaTTokenIter *>(dim_iters_->at(i))->reuse();
    }
  }
  ObSRDaaTIterImpl::reuse(switch_tablet);
}

int ObTextDaaTIter::pre_process()
{
  int ret = OB_SUCCESS;
  if (dim_iters_->count() == 0) {
    ret = OB_ITER_END;
  } else if (iter_param_->need_project_relevance() && nullptr != iter_param_->relevance_expr_) {
    if (OB_FAIL(bm25_param_estimator_.do_estimation(*iter_param_->eval_ctx_))) {
    }
  }
  return ret;
}

int ObTextDaaTIter::get_next_rows(const int64_t capacity, int64_t &count)
{
  return doc_only_union_ ? get_doc_only_rows(capacity, count)
                         : ObSRDaaTIterImpl::get_next_rows(capacity, count);
}

int ObTextDaaTIter::get_doc_only_rows(const int64_t capacity, int64_t &count)
{
  int ret = OB_SUCCESS;
  count = 0;
  if (IS_NOT_INIT) {
    ret = OB_NOT_INIT;
  } else if (OB_UNLIKELY(capacity <= 0)) {
    ret = capacity == 0 ? OB_SUCCESS : OB_INVALID_ARGUMENT;
  } else if (iter_param_->limit_param_->is_valid()
      && output_row_cnt_ >= iter_param_->limit_param_->limit_) {
    ret = OB_ITER_END;
  } else {
    const int64_t real_capacity = MIN(capacity, iter_param_->max_batch_size_);
    while (OB_SUCC(ret) && count < real_capacity) {
      const ObDatum *smallest_id = nullptr;
      uint32_t equal_mask = 0;
      int64_t matching_dims = 0;
      for (int64_t i = 0; OB_SUCC(ret) && i < dim_iters_->count(); ++i) {
        DocOnlyCursor &cursor = doc_only_cursors_[i];
        if (!cursor.ended_ && cursor.pos_ == cursor.count_) {
          ObTextRetrievalDaaTTokenIter *token_iter =
              static_cast<ObTextRetrievalDaaTTokenIter *>(dim_iters_->at(i));
          int64_t fetch_capacity = INT64_MAX; // unlimited scans keep the token's original batch size
          const ObLimitParam &limit = *iter_param_->limit_param_;
          if (limit.is_valid()) {
            const int64_t remaining_output = limit.limit_ > output_row_cnt_ ? limit.limit_ - output_row_cnt_ : 0;
            const int64_t remaining_offset = limit.offset_ > input_row_cnt_ ? limit.offset_ - input_row_cnt_ : 0;
            // Saturate each addition at the allocated batch size instead of
            // adding limit + offset, which may overflow for legal large values.
            fetch_capacity = MIN(iter_param_->max_batch_size_, remaining_output);
            fetch_capacity += MIN(iter_param_->max_batch_size_ - fetch_capacity, remaining_offset);
          }
          if (OB_UNLIKELY(fetch_capacity <= 0)) {
            ret = OB_ITER_END;
          } else if (OB_FAIL(token_iter->get_doc_only_batch(fetch_capacity, cursor.ids_, cursor.count_))) {
            if (OB_ITER_END == ret) {
              cursor.ended_ = true;
              ret = OB_SUCCESS;
            }
          } else {
            cursor.pos_ = 0;
            cursor.id_ = &cursor.ids_[0].get_datum();
          }
        }
        if (OB_SUCC(ret) && !cursor.ended_) {
          const ObDatum &id = *cursor.id_;
          int cmp = 0;
          if (nullptr == smallest_id) {
            cmp = -1;
          } else if (OB_FAIL(compare_doc_only_ids(id, *smallest_id, cmp))) {
          }
          if (OB_SUCC(ret)) {
            if (cmp < 0) {
              smallest_id = &id;
              equal_mask = (1U << i);
              matching_dims = 1;
            } else if (0 == cmp) {
              equal_mask |= (1U << i);
              ++matching_dims;
            }
          }
        }
      }
      if (OB_FAIL(ret)) {
      } else if (nullptr == smallest_id) {
        ret = OB_ITER_END;
      } else {
        // Each posting stream is unique by (token, document ID). Advance every
        // stream with the chosen ID so documents shared by tokens emit once.
        // Refilling is delayed until the next round: cache_result owns its copy
        // before any input buffer can be overwritten.
        for (int64_t i = 0; i < dim_iters_->count(); ++i) {
          if (equal_mask & (1U << i)) {
            DocOnlyCursor &cursor = doc_only_cursors_[i];
            ++cursor.pos_;
            if (cursor.pos_ < cursor.count_) {
              cursor.id_ = &cursor.ids_[cursor.pos_].get_datum();
            }
          }
        }
        if (cardinality_only_) {
          // Visibility and de-duplication above are unchanged. COUNT(*) needs
          // only the number of actual matching rows, with no ID/score consumer.
          ++input_row_cnt_;
          ++output_row_cnt_;
          ++count;
        } else {
          ret = cache_result(count, *smallest_id, static_cast<double>(matching_dims));
        }
      }
    }
    if ((OB_SUCC(ret) || OB_ITER_END == ret) && count > 0) {
      // As in the normal DAAT path, deliver a nonempty final batch before EOF.
      ret = cardinality_only_ ? OB_SUCCESS : project_doc_only_results(count);
    }
  }
  return ret;
}

int ObTextDaaTIter::project_doc_only_results(const int64_t count)
{
  int ret = OB_SUCCESS;
  ObExpr *id_expr = iter_param_->id_proj_expr_;
  ObExpr *relevance_expr = iter_param_->relevance_proj_expr_;
  ObEvalCtx &eval_ctx = *iter_param_->eval_ctx_;
  if (!eval_ctx.is_vectorized() || !id_expr->is_batch_result()
      || (nullptr != relevance_expr && !relevance_expr->is_batch_result())) {
    ret = ObSRDaaTIterImpl::project_results(count);
  } else {
    // Match locate_datum_for_write's reserved-buffer reset before writing each
    // slot. String IDs still borrow the owned result cache, never token buffers.
    ObDatum *ids = id_expr->locate_datums_for_update(eval_ctx, count);
    if (ObUInt64Type == id_expr->datum_meta_.type_) {
      for (int64_t i = 0; i < count; ++i) {
        set_datum_int(ids[i], buffered_domain_ids_[i]);
      }
    } else {
      for (int64_t i = 0; i < count; ++i) {
        set_datum_shallow(ids[i], buffered_domain_ids_[i]);
      }
    }
    id_expr->get_evaluated_flags(eval_ctx).set_all(count);
    id_expr->set_evaluated_projected(eval_ctx);
    if (nullptr != relevance_expr) {
      ObDatum *relevances = relevance_expr->locate_datums_for_update(eval_ctx, count);
      for (int64_t i = 0; i < count; ++i) {
        relevances[i].set_double(buffered_relevances_[i]);
      }
      relevance_expr->get_evaluated_flags(eval_ctx).set_all(count);
      relevance_expr->set_evaluated_projected(eval_ctx);
    }
  }
  return ret;
}

ObTextBMWIter::ObTextBMWIter()
  : ObSRBMWIterImpl(),
    bm25_param_estimator_() {}

void ObTextBMWIter::reuse(const bool switch_tablet)
{
  bm25_param_estimator_.reuse(switch_tablet);
  if (OB_NOT_NULL(dim_iters_)) {
    for (int64_t i = 0; i < dim_iters_->count(); ++i) {
      static_cast<ObTextRetrievalBlockMaxIter *>(dim_iters_->at(i))->reuse();
    }
  }
  ObSRBMWIterImpl::reuse(switch_tablet);
}

void ObTextBMWIter::reset()
{
  bm25_param_estimator_.reset();
  if (OB_NOT_NULL(dim_iters_)) {
    for (int64_t i = 0; i < dim_iters_->count(); ++i) {
      static_cast<ObTextRetrievalBlockMaxIter *>(dim_iters_->at(i))->reset();
    }
  }
  ObSRBMWIterImpl::reset();
}

int ObTextBMWIter::init(const ObTextDaaTParam &param)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(param.base_param_) || OB_ISNULL(param.dim_iters_) || OB_ISNULL(param.allocator_)
      || OB_ISNULL(param.relevance_collector_)) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(ObSRBMWIterImpl::init(*param.base_param_, *param.dim_iters_,
                                           *param.allocator_, *param.relevance_collector_))) {
  } else if (OB_FAIL(bm25_param_estimator_.init(param.bm25_param_est_ctx_))) {
  }
  return ret;
}

int ObTextBMWIter::get_next_rows(const int64_t capacity, int64_t &count)
{
  int ret = OB_SUCCESS;
  if (IS_NOT_INIT) {
    ret = OB_NOT_INIT;
  } else if (dim_iters_->count() == 0) {
    ret = OB_ITER_END;
  } else if (OB_FAIL(bm25_param_estimator_.do_estimation(*iter_param_->eval_ctx_))) {
  }

  if (FAILEDx(ObSRBMWIterImpl::get_next_rows(capacity, count))) {
    if (OB_UNLIKELY(OB_ITER_END != ret)) {
    }
  }
  return ret;
}

int ObTextBMWIter::init_before_wand_process()
{
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(!bm25_param_estimator_.is_estimated())) {
    ret = OB_ERR_UNEXPECTED;
  }

  for (int64_t i = 0; OB_SUCC(ret) && i < dim_iters_->count(); ++i) {
    ObTextRetrievalBlockMaxIter *block_max_iter = static_cast<ObTextRetrievalBlockMaxIter *>(dim_iters_->at(i));
    if (OB_FAIL(block_max_iter->init_block_max_iter(
        bm25_param_estimator_.get_total_doc_cnt(), bm25_param_estimator_.get_avg_doc_token_cnt()))) {
    }
  }
  return ret;
}

} // namespace storage
} // namespace oceanbase
