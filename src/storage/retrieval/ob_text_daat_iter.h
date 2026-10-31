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

#ifndef OB_TEXT_DAAT_ITER_H_
#define OB_TEXT_DAAT_ITER_H_

#include "lib/charset/ob_charset.h"
#include "query/ob_sql_define.h"
#include "ob_inv_idx_param_estimator.h"
#include "ob_sparse_bmw_iter.h"
#include "ob_sparse_daat_iter.h"
#include "ob_text_retrieval_token_iter.h"

namespace oceanbase
{
namespace sql
{
  class ObDASScanIter;
}
namespace storage
{

struct ObTextDaaTParam
{
  ObTextDaaTParam()
    : dim_iters_(nullptr), base_param_(nullptr), allocator_(nullptr), relevance_collector_(nullptr),
      bm25_param_est_ctx_(), mode_flag_(sql::ObMatchAgainstMode::NATURAL_LANGUAGE_MODE),
      function_lookup_mode_(false), doc_only_union_allowed_(false), cardinality_only_(false) {}
  ~ObTextDaaTParam() {}
  TO_STRING_KV(K_(base_param), KP_(dim_iters), K_(bm25_param_est_ctx),  K_(mode_flag), K_(function_lookup_mode));
  ObIArray<ObISRDaaTDimIter *> *dim_iters_;
  ObSparseRetrievalMergeParam *base_param_;
  common::ObArenaAllocator *allocator_;
  ObSRDaaTRelevanceCollector *relevance_collector_;
  ObBM25ParamEstCtx bm25_param_est_ctx_;
  sql::ObMatchAgainstMode mode_flag_;
  bool function_lookup_mode_;
  bool doc_only_union_allowed_; // no score consumer; minimum_should_match is 0/1
  bool cardinality_only_; // caller advertises an empty result expression array
};

class ObTextDaaTIter final : public ObSRDaaTIterImpl
{
public:
  ObTextDaaTIter() : ObSRDaaTIterImpl(),
      bm25_param_estimator_(),
      mode_flag_(sql::ObMatchAgainstMode::NATURAL_LANGUAGE_MODE),
      function_lookup_mode_(false), doc_only_union_(false), cardinality_only_(false),
      doc_only_uint64_(false), doc_only_int64_(false), doc_only_cmp_(nullptr),
      doc_only_charset_(nullptr), doc_only_cursors_() {}
  virtual ~ObTextDaaTIter() { reset(); }

  int init(const ObTextDaaTParam &param);
  virtual void reuse(const bool switch_tablet = false) override;
  virtual void reset() override;
  virtual int get_next_rows(const int64_t capacity, int64_t &count) override;
protected:
  virtual int pre_process() override;
protected:
  ObBM25ParamEstimator bm25_param_estimator_;
  sql::ObMatchAgainstMode mode_flag_;
  bool function_lookup_mode_;
private:
  static const int64_t MAX_DOC_ONLY_UNION_DIMS = 8;
  struct DocOnlyCursor
  {
    DocOnlyCursor() : ids_(nullptr), id_(nullptr), count_(0), pos_(0), ended_(false) {}
    const ObDocIdExt *ids_;
    const common::ObDatum *id_;
    int64_t count_;
    int64_t pos_;
    bool ended_;
  };
  int get_doc_only_rows(const int64_t capacity, int64_t &count);
  int project_doc_only_results(const int64_t count);
  inline int compare_doc_only_ids(const common::ObDatum &left,
                                 const common::ObDatum &right, int &cmp) const
  {
    int ret = common::OB_SUCCESS;
    if (doc_only_uint64_ && !left.is_null() && !right.is_null()) {
      const uint64_t left_id = left.get_uint64();
      const uint64_t right_id = right.get_uint64();
      cmp = left_id < right_id ? -1 : (left_id > right_id ? 1 : 0);
    } else if (doc_only_int64_ && !left.is_null() && !right.is_null()) {
      const int64_t left_id = left.get_int();
      const int64_t right_id = right.get_int();
      cmp = left_id < right_id ? -1 : (left_id > right_id ? 1 : 0);
    } else if (nullptr != doc_only_charset_ && !left.is_null() && !right.is_null()
        && nullptr != left.ptr_ && nullptr != right.ptr_) {
      // CS_TYPE_BINARY's strnncollsp compares the common byte prefix, then
      // subtracts the lengths. Preserve its integer narrowing and normalize as
      // the typed comparator does; trailing spaces and NUL bytes are significant.
      const size_t left_len = left.len_;
      const size_t right_len = right.len_;
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && defined(__ORDER_BIG_ENDIAN__)
      if (16 == left_len && 16 == right_len
          && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__ || __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)) {
        // Common encoded document IDs fit two words. memcpy permits unaligned
        // inputs; compare the words in byte order to match binary memcmp.
        uint64_t left_word = 0;
        uint64_t right_word = 0;
        for (int64_t offset = 0; offset < 16; offset += sizeof(uint64_t)) {
          MEMCPY(&left_word, left.ptr_ + offset, sizeof(left_word));
          MEMCPY(&right_word, right.ptr_ + offset, sizeof(right_word));
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
          left_word = __builtin_bswap64(left_word);
          right_word = __builtin_bswap64(right_word);
#endif
          if (left_word != right_word) {
            break;
          }
        }
        cmp = left_word > right_word ? 1 : (left_word < right_word ? -1 : 0);
      } else {
#endif
        cmp = MEMCMP(left.ptr_, right.ptr_, MIN(left_len, right_len));
        if (0 == cmp) {
          cmp = static_cast<int>(left_len - right_len);
        }
        cmp = cmp > 0 ? 1 : (cmp < 0 ? -1 : 0);
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && defined(__ORDER_BIG_ENDIAN__)
      }
#endif
    } else {
      ret = doc_only_cmp_(left, right, cmp, nullptr);
    }
    return ret;
  }
  bool doc_only_union_;
  bool cardinality_only_;
  bool doc_only_uint64_;
  bool doc_only_int64_;
  common::ObDatumCmpFuncType doc_only_cmp_;
  const ObCharsetInfo *doc_only_charset_; // immutable global binary collation; not owned
  DocOnlyCursor doc_only_cursors_[MAX_DOC_ONLY_UNION_DIMS];
  DISALLOW_COPY_AND_ASSIGN(ObTextDaaTIter);
};


class ObTextBMWIter final : public ObSRBMWIterImpl
{
public:
  ObTextBMWIter();
  virtual ~ObTextBMWIter() { reset(); }
  virtual void reuse(const bool switch_tablet = false) override;
  virtual void reset() override;
  int init(const ObTextDaaTParam &param);
protected:
  virtual int get_next_rows(const int64_t capacity, int64_t &count) override;
  virtual int init_before_wand_process() override;
protected:
  ObBM25ParamEstimator bm25_param_estimator_;
private:
  DISALLOW_COPY_AND_ASSIGN(ObTextBMWIter);
};

} // namespace storage
} // namespace oceanbase

#endif
