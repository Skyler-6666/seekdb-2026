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

#define USING_LOG_PREFIX LIB

#include "ob_vsag_adaptor.h"
#include <cmath>
#include <cstring>
#include <map>
#include <sstream>
#include <vector>
#include "vsag/vsag.h"
#include "vsag/errors.h"
#include "vsag/dataset.h"
#include "vsag/search_param.h"
#include "vsag/index.h"
#include "vsag/options.h"
#include "vsag/factory.h"
#include "lib/utility/ob_print_utils.h"
#include "lib/oblog/ob_log.h"
#include "lib/time/ob_time_utility.h"
#include "lib/worker.h"
#include "lib/vector/ob_hipvs_adaptor.h"

namespace oceanbase {
namespace common {
namespace obvsag {

using namespace vsag;

static constexpr int64_t VSAG_SINDI_N_CANDIDATE_FACTOR = 500;

/**
 * The widest search the GPU backend asks for. It matches the upper bound the
 * observer uses for the CPU backend.
 */
static constexpr int64_t HIPVS_MAX_EF_SEARCH = 160000;

static int vsag_errcode2ob(vsag::ErrorType vsag_errcode)
{
  int ret = OB_ERR_VSAG_RETURN_ERROR;
  switch (vsag_errcode) {
    case vsag::ErrorType::INVALID_ARGUMENT: {
      ret = OB_INVALID_ARGUMENT;
      break;
    }
    case vsag::ErrorType::UNSUPPORTED_INDEX:
    case vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION: {
      ret = OB_NOT_SUPPORTED;
      break;
    }
    case vsag::ErrorType::DIMENSION_NOT_EQUAL: {
      ret = OB_INVALID_ARGUMENT;
      break;
    }
    case vsag::ErrorType::INDEX_EMPTY: {
      ret = OB_OP_NOT_ALLOW;
      break;
    }
    case vsag::ErrorType::NO_ENOUGH_MEMORY: {
      ret = OB_ALLOCATE_MEMORY_FAILED;
      break;
    }
    default: {
      ret = OB_ERR_VSAG_RETURN_ERROR;
      break;
    }
  }
  return ret;
}

static void fill_vsag_error_message(const vsag::Error &error, std::string &err_msg)
{
  err_msg = error.message;
}

static void adjust_create_index_max_degree(const IndexType index_type, int &max_degree)
{
  // hgraph of vsag needs to be multiplied by 2 so as to align recall with hnsw
  if (HNSW_SQ_TYPE == index_type || HNSW_BQ_TYPE == index_type || HGRAPH_TYPE == index_type) {
    max_degree *= 2;
    LOG_INFO("change max_degree for hgraph", K(index_type), K(max_degree), K(lbt()));
  }
}

class ObVasgFilter final : public vsag::Filter {
public:
  ObVasgFilter(float valid_ratio,
               const std::function<bool(int64_t)> &vid_fallback_func,
               const std::function<bool(const char *)> &exinfo_fallback_func)
      : valid_ratio_(valid_ratio), vid_fallback_func_(vid_fallback_func),
        exinfo_fallback_func_(exinfo_fallback_func){};

  ~ObVasgFilter() {}

  bool CheckValid(int64_t id) const override { return !vid_fallback_func_(id); }

  bool CheckValid(const char *data) const override {
    return !exinfo_fallback_func_(data);
  }

  float ValidRatio() const override { return valid_ratio_; }

private:
  float valid_ratio_;
  std::function<bool(int64_t)> vid_fallback_func_{nullptr};
  std::function<bool(const char *)> exinfo_fallback_func_{nullptr};
};

/**
 * The base class of every vector index handle.
 *
 * A `VectorIndexPtr` is an opaque pointer to one of the handlers below. The
 * handlers are distinguished by their backend, so that the top level
 * functions of this file can dispatch to the right implementation.
 */
class VectorIndexHandler {
public:
  virtual ~VectorIndexHandler() {}
  /**
   * @brief The backend that owns this index, one of the `IndexLib` values.
   */
  virtual int get_backend() const = 0;
  /**
   * @brief Load the index from a stream.
   *
   * The format is self describing, so that an index that was written by one
   * backend can be read back by the right one.
   */
  virtual int deserialize(std::istream &in_stream) = 0;
  /**
   * @brief Write the index to a stream.
   */
  virtual int serialize(std::ostream &out_stream) = 0;
};

class HnswIndexHandler;

/**
 * An HNSW index owned by the AMD hipVS backend.
 *
 * The graph is built on the GPU: `cuvsCagraBuild` constructs the k-NN graph
 * and `cuvsHnswFromCagra` turns it into a regular HNSW index. That graph
 * construction is the expensive part of an HNSW build, and it is where the
 * acceleration comes from. The index is then served by hipVS as well.
 *
 * The handler type is decided by the `lib` option of the index, not by
 * whether a GPU happens to be present, so that the serialized format of an
 * index never depends on the node that wrote it. When the GPU backend is not
 * usable on this node, the handler falls back to the CPU backend and writes
 * the CPU format, which the CPU backend reads back.
 */
class HipvsIndexHandler : public VectorIndexHandler {
public:
  HipvsIndexHandler(const char *dtype,
                    const char *metric,
                    int max_degree,
                    int ef_construction,
                    int ef_search,
                    int dim,
                    const seekdb_hipvs_index_params_t &params,
                    const obhipvs::HipvsBridge *bridge,
                    void *index)
      : dtype_(dtype),
        metric_(metric),
        max_degree_(max_degree),
        ef_construction_(ef_construction),
        ef_search_(ef_search),
        dim_(dim),
        params_(params),
        bridge_(bridge),
        index_(index),
        fallback_(nullptr)
  {}

  ~HipvsIndexHandler() override;

  int get_backend() const override { return LIB_HIPVS; }

  /**
   * @brief Whether the index is really served by the GPU backend.
   *
   * It is false when the GPU backend was not usable on this node and the
   * handler fell back to the CPU backend.
   */
  bool is_gpu_backed() const { return index_ != nullptr; }

  int build_index(float *vector_list, int64_t *ids, int dim, int size);
  int add_index(float *vector_list, int64_t *ids, int dim, int size);
  int get_index_number(int64_t &num_elements);
  int get_vid_bound(int64_t &min_vid, int64_t &max_vid);
  int knn_search(const float *query,
                 int dim,
                 int64_t topk,
                 int ef_search,
                 const float *&dist,
                 const int64_t *&ids,
                 int64_t &result_size,
                 float valid_ratio,
                 FilterInterface *bitmap,
                 bool reverse_filter,
                 float distance_threshold);
  int serialize(std::ostream &out_stream);
  int deserialize(std::istream &in_stream) override;
  uint64_t estimate_memory(uint64_t row_count, bool is_build);

  const char *get_dtype() const { return dtype_; }
  const char *get_metric() const { return metric_; }
  int get_max_degree() const { return max_degree_; }
  int get_ef_construction() const { return ef_construction_; }
  int get_ef_search() const { return ef_search_; }
  int get_dim() const { return dim_; }

  TO_STRING_KV(KP(this), KCSTRING_(dtype), KCSTRING_(metric), K_(max_degree), K_(ef_construction),
               K_(ef_search), K_(dim), KP(index_));

private:
  /**
   * @brief Create the CPU backend that is used when the GPU is not available.
   */
  int ensure_fallback();

  const char *dtype_;
  const char *metric_;
  int max_degree_;
  int ef_construction_;
  int ef_search_;
  int dim_;
  seekdb_hipvs_index_params_t params_;
  const obhipvs::HipvsBridge *bridge_;
  void *index_;
  HnswIndexHandler *fallback_;
  /* The buffers holding the result of the last search. They stay alive as
   * long as the index, exactly like the buffers VSAG hands out. */
  std::vector<int64_t> ids_buffer_;
  std::vector<float> dist_buffer_;
};

class HnswIndexHandler : public VectorIndexHandler {
public:
  HnswIndexHandler(bool is_create, bool is_build, bool use_static,
                   const char *dtype, const char *metric, int max_degree,
                   int ef_construction, int ef_search, int dim,
                   IndexType index_type, std::shared_ptr<vsag::Index> index,
                   vsag::Allocator *allocator, uint64_t extra_info_size,
                   int16_t refine_type, int16_t bq_bits_query, bool bq_use_fht)
      : is_created_(is_create), is_build_(is_build), use_static_(use_static),
        dtype_(dtype), metric_(metric), max_degree_(max_degree),
        ef_construction_(ef_construction), ef_search_(ef_search), dim_(dim),
        index_type_(index_type), index_(index), allocator_(allocator),
        extra_info_size_(extra_info_size), refine_type_(refine_type),
        bq_bits_query_(bq_bits_query), bq_use_fht_(bq_use_fht) {}

  HnswIndexHandler(bool is_create, bool is_build, bool use_static, const char *dtype, const char *metric,
      IndexType index_type, std::shared_ptr<vsag::Index> index, vsag::Allocator *allocator, uint64_t extra_info_size,
      bool use_reorder, float doc_prune_ratio, int window_size)
      : is_created_(is_create),
        is_build_(is_build),
        use_static_(use_static),
        dtype_(dtype),
        metric_(metric),
        index_type_(index_type),
        index_(index),
        allocator_(allocator),
        extra_info_size_(extra_info_size),
        use_reorder_(use_reorder),
        doc_prune_ratio_(doc_prune_ratio),
        window_size_(window_size)
  {}

  ~HnswIndexHandler() {
    index_ = nullptr;
    LOG_INFO("[OBVSAG] after deconstruction, hnsw index", KP(allocator_), K(index_.use_count()), K(lbt()));
  }
  int get_backend() const override { return LIB_VSAG; }
  int build_index(const vsag::DatasetPtr &base);
  int get_index_number();
  int add_index(const vsag::DatasetPtr &incremental);
  int cal_distance_by_id(const float *vector, const int64_t *ids, int64_t count,
                         const float *&dist);
  int cal_distance_by_id(uint32_t len, uint32_t *dims, float *vals,
                        const int64_t *ids, int64_t count, const float *&dist);
  int get_extra_info_by_ids(const int64_t *ids, int64_t count,
                            char *extra_infos);
  int get_vid_bound(int64_t &min_vid, int64_t &max_vid);
  uint64_t estimate_memory(const uint64_t row_count, const bool is_build);
  int knn_search(const vsag::DatasetPtr &query, int64_t topk,
                 const std::string &parameters, const float *&dist,
                 const int64_t *&ids, int64_t &result_size, float valid_ratio,
                 int index_type, FilterInterface *bitmap, bool reverse_filter,
                 bool need_extra_info, const char *&extra_infos,
                 void *allocator, float distance_threshold = FLT_MAX);
  int knn_search(const vsag::DatasetPtr &query, int64_t topk,
                 const std::string &parameters, const float *&dist,
                 const int64_t *&ids, int64_t &result_size, float valid_ratio,
                 int index_type, FilterInterface *bitmap, bool reverse_filter,
                 bool need_extra_info, const char *&extra_infos,
                 void *&iter_ctx, bool is_last_search, void *allocator);
  int immutable_optimize();
  int serialize(std::ostream &out_stream) override;
  int deserialize(std::istream &in_stream) override;

  std::shared_ptr<vsag::Index> &get_index() { return index_; }
  void set_index(std::shared_ptr<vsag::Index> hnsw) { index_ = hnsw; }
  vsag::Allocator *get_allocator() const { return allocator_; }
  inline bool get_use_static() const { return use_static_; }
  inline int get_max_degree() const { return max_degree_; }
  inline int get_ef_construction() const { return ef_construction_; }
  inline int get_index_type() const { return (int)index_type_; }
  const char *get_dtype() const { return dtype_; }
  const char *get_metric() const { return metric_; }
  inline int get_ef_search() const { return ef_search_; }
  inline int get_dim() const { return dim_; }
  inline uint64_t get_extra_info_size() const { return extra_info_size_; }
  inline int16_t get_refine_type() const { return refine_type_; }
  inline int16_t get_bq_bits_query() const { return bq_bits_query_; }
  inline bool get_bq_use_fht() const { return bq_use_fht_; };
  inline bool get_use_reorder() const { return use_reorder_; }
  inline float get_doc_prune_ratio() const { return doc_prune_ratio_; }
  inline int get_window_size() const { return window_size_; }

  TO_STRING_KV(KP(this), K_(is_created), K_(is_build), K_(use_static), KCSTRING_(dtype),
      KCSTRING_(metric), K_(max_degree), K_(ef_construction), K_(ef_search), K_(dim),
      K_(ef_search), K_(index_type), KP(index_.get()), KP_(allocator), K_(extra_info_size),
      K_(refine_type), K_(bq_bits_query), K_(bq_use_fht));

private:
  bool is_created_;
  bool is_build_;
  bool use_static_;
  const char *dtype_;
  const char *metric_;
  int max_degree_;
  int ef_construction_;
  int ef_search_;
  int dim_;
  IndexType index_type_;
  std::shared_ptr<vsag::Index> index_;
  vsag::Allocator *allocator_;
  uint64_t extra_info_size_;
  int16_t refine_type_;
  int16_t bq_bits_query_;
  bool bq_use_fht_;
  bool use_reorder_;
  float doc_prune_ratio_;
  int window_size_;
};

/* The destructor is defined here, and not in the class body, because the CPU
 * fallback is an `HnswIndexHandler` and the type has to be complete to
 * destroy it. */
HipvsIndexHandler::~HipvsIndexHandler()
{
  if (index_ != nullptr && bridge_ != nullptr) {
    bridge_->index_destroy(index_);
    index_ = nullptr;
  }
  delete fallback_;
  fallback_ = nullptr;
}

int HipvsIndexHandler::ensure_fallback()
{
  int ret = OB_SUCCESS;
  if (fallback_ != nullptr) {
    // do nothing
  } else {
    char result_param_str[1024] = {0};
    if (OB_FAIL(construct_vsag_create_param(uint8_t(HNSW_TYPE), dtype_, metric_, dim_,
                                             max_degree_, ef_construction_, ef_search_, nullptr,
                                             0 /* extra_info_size */, 0 /* refine_type */,
                                             32 /* bq_bits_query */, false /* bq_use_fht */,
                                             result_param_str))) {
    } else {
      const std::string input_json_str(result_param_str);
      tl::expected<std::shared_ptr<Index>, Error> index =
          vsag::Factory::CreateIndex(get_index_type_str(HNSW_TYPE), input_json_str, nullptr);
      if (index.has_value()) {
        fallback_ = new HnswIndexHandler(true, false, false, dtype_, metric_, max_degree_,
                                         ef_construction_, ef_search_, dim_, HNSW_TYPE,
                                         index.value(), nullptr, 0, 0, 32, false);
        if (OB_ISNULL(fallback_)) {
          ret = OB_ALLOCATE_MEMORY_FAILED;
        }
      } else {
        ret = vsag_errcode2ob(index.error().type);
        LOG_WARN("[OBHIPVS] create the CPU fallback index failed", K(ret));
      }
    }
  }
  return ret;
}

int HipvsIndexHandler::build_index(float *vector_list, int64_t *ids, int dim, int size)
{
  int ret = OB_SUCCESS;
  if (vector_list == nullptr || ids == nullptr || dim <= 0 || size <= 0) {
    ret = OB_INVALID_ARGUMENT;
  } else if (index_ != nullptr) {
    /* This is the GPU accelerated path: the k-NN graph is constructed on the
     * AMD GPU and then converted into an HNSW index. */
    const int64_t start_us = ObTimeUtility::fast_current_time();
    const int32_t status = bridge_->index_build(index_, vector_list, dim, size, ids);
    const int64_t cost_us = ObTimeUtility::fast_current_time() - start_us;
    if (status != SEEKDB_HIPVS_OK) {
      ret = obhipvs::to_ob_error(status);
      LOG_WARN("[OBHIPVS] build index failed", K(ret), K(dim), K(size), K(cost_us));
    } else {
      /* The log line carries the size of the graph and the time it took to
       * build it, so that the acceleration can be read back from the observer
       * log without running a benchmark. */
      LOG_INFO("[OBHIPVS] build index on the GPU", K(dim), K(size), K(cost_us),
               K(params_.graph_algo == SEEKDB_HIPVS_GRAPH_ALGO_NN_DESCENT ? "nn_descent" : "cagra"));
    }
  } else if (OB_FAIL(ensure_fallback())) {
  } else {
    const int64_t start_us = ObTimeUtility::fast_current_time();
    DatasetPtr dataset = vsag::Dataset::Make();
    dataset->Dim(dim)
        ->NumElements(size)
        ->Ids(ids)
        ->Float32Vectors(vector_list)
        ->Owner(false);
    ret = fallback_->build_index(dataset);
    const int64_t cost_us = ObTimeUtility::fast_current_time() - start_us;
    /* Logged with the same shape as the GPU line, so that the two can be
     * compared directly from the observer log. */
    LOG_INFO("[OBHIPVS] build index on the CPU backend", K(dim), K(size), K(cost_us));
  }
  return ret;
}

int HipvsIndexHandler::add_index(float *vector_list, int64_t *ids, int dim, int size)
{
  int ret = OB_SUCCESS;
  if (vector_list == nullptr || ids == nullptr || dim <= 0 || size < 0) {
    ret = OB_INVALID_ARGUMENT;
  } else if (size == 0) {
    // nothing to add
  } else if (index_ != nullptr) {
    const int32_t status = bridge_->index_add(index_, vector_list, dim, size, ids);
    if (status != SEEKDB_HIPVS_OK) {
      ret = obhipvs::to_ob_error(status);
      LOG_WARN("[OBHIPVS] add index failed", K(ret), K(dim), K(size));
    }
  } else if (OB_FAIL(ensure_fallback())) {
  } else {
    DatasetPtr incremental = vsag::Dataset::Make();
    incremental->Dim(dim)
        ->NumElements(size)
        ->Ids(ids)
        ->Float32Vectors(vector_list)
        ->Owner(false);
    ret = fallback_->add_index(incremental);
  }
  return ret;
}

int HipvsIndexHandler::get_index_number(int64_t &num_elements)
{
  int ret = OB_SUCCESS;
  num_elements = 0;
  if (index_ != nullptr) {
    const int32_t status = bridge_->index_num_elements(index_, &num_elements);
    if (status != SEEKDB_HIPVS_OK) {
      ret = obhipvs::to_ob_error(status);
      num_elements = 0;
    }
  } else if (OB_FAIL(ensure_fallback())) {
  } else {
    num_elements = fallback_->get_index_number();
  }
  return ret;
}

int HipvsIndexHandler::get_vid_bound(int64_t &min_vid, int64_t &max_vid)
{
  int ret = OB_SUCCESS;
  if (index_ != nullptr) {
    const int32_t status = bridge_->index_vid_bound(index_, &min_vid, &max_vid);
    if (status != SEEKDB_HIPVS_OK) {
      ret = obhipvs::to_ob_error(status);
    }
  } else if (OB_FAIL(ensure_fallback())) {
  } else {
    ret = fallback_->get_vid_bound(min_vid, max_vid);
  }
  return ret;
}

int HipvsIndexHandler::knn_search(const float *query,
                                  int dim,
                                  int64_t topk,
                                  int ef_search,
                                  const float *&dist,
                                  const int64_t *&ids,
                                  int64_t &result_size,
                                  float valid_ratio,
                                  FilterInterface *bitmap,
                                  bool reverse_filter,
                                  float distance_threshold)
{
  int ret = OB_SUCCESS;
  result_size = 0;
  dist = nullptr;
  ids = nullptr;
  if (query == nullptr || dim <= 0 || topk <= 0) {
    ret = OB_INVALID_ARGUMENT;
    return ret;
  }

  if (index_ == nullptr) {
    /* The GPU backend is not usable on this node, the CPU backend answers. */
    if (OB_FAIL(ensure_fallback())) {
      return ret;
    }
    char result_param_str[1024] = {0};
    if (OB_FAIL(construct_vsag_search_param(uint8_t(HNSW_TYPE), ef_search, false, result_param_str))) {
      return ret;
    }
    const std::string input_json_string(result_param_str);
    DatasetPtr query_dataset = vsag::Dataset::Make();
    query_dataset->NumElements(1)->Dim(dim)->Float32Vectors(const_cast<float *>(query))->Owner(false);
    return fallback_->knn_search(query_dataset, topk, input_json_string, dist, ids, result_size,
                                 valid_ratio, HNSW_TYPE, bitmap, reverse_filter, false, fallback_extra_infos,
                                 nullptr, distance_threshold);
  }

  int64_t index_number = 0;
  const int32_t count_status = bridge_->index_num_elements(index_, &index_number);
  if (count_status != SEEKDB_HIPVS_OK) {
    return obhipvs::to_ob_error(count_status);
  }
  if (index_number <= 0) {
    return OB_SUCCESS;
  }

  /* Clamp the search breadth the same way the CPU backend does, otherwise a
   * large ef_search makes the GPU backend slower than the CPU one. */
  const int64_t EF_SEARCH_LIMIT = 1000L;
  const int64_t AMPLIFICATION_FACTOR = 10;
  if (ef_search > EF_SEARCH_LIMIT) {
    topk = topk < index_number ? topk : index_number;
    const int64_t ef_search_threshold = AMPLIFICATION_FACTOR * topk > EF_SEARCH_LIMIT
                                          ? AMPLIFICATION_FACTOR * topk
                                          : EF_SEARCH_LIMIT;
    ef_search = ef_search < ef_search_threshold ? ef_search : ef_search_threshold;
  }

  int64_t fetch = topk < index_number ? topk : index_number;
  int32_t ef = ef_search > 0 ? ef_search : params_.ef_search;
  if (bitmap != nullptr) {
    /* hipVS has no native filtering, so the candidates are filtered here.
     * To keep the recall, more candidates than requested are fetched, the
     * same way the CPU backend widens its search when a filter is present. */
    const double ratio = valid_ratio > 0.01 ? static_cast<double>(valid_ratio) : 0.01;
    const int64_t amplified = static_cast<int64_t>(std::ceil(static_cast<double>(topk) / ratio));
    fetch = amplified > fetch ? (amplified < index_number ? amplified : index_number) : fetch;
    if (ef < fetch) {
      const int64_t widened = fetch < HIPVS_MAX_EF_SEARCH ? fetch : HIPVS_MAX_EF_SEARCH;
      ef = static_cast<int32_t>(widened);
    }
  }
  if (distance_threshold < FLT_MAX) {
    /* The threshold is applied after the search, so more candidates are
     * fetched to still be able to answer with the requested number of rows. */
    const int64_t amplified = fetch * 2;
    fetch = amplified < index_number ? amplified : index_number;
    if (ef < fetch) {
      const int64_t widened = fetch < HIPVS_MAX_EF_SEARCH ? fetch : HIPVS_MAX_EF_SEARCH;
      ef = static_cast<int32_t>(widened);
    }
  }

  ids_buffer_.assign(static_cast<size_t>(fetch), -1);
  dist_buffer_.assign(static_cast<size_t>(fetch), 0.0f);
  int64_t written = 0;
  const int32_t status = bridge_->index_search(index_, query, dim, fetch, ef, ids_buffer_.data(),
                                               dist_buffer_.data(), &written);
  if (status != SEEKDB_HIPVS_OK) {
    return obhipvs::to_ob_error(status);
  }

  int64_t kept = 0;
  for (int64_t i = 0; i < written && kept < topk; ++i) {
    const int64_t id = ids_buffer_[static_cast<size_t>(i)];
    if (id < 0) {
      continue;
    }
    if (bitmap != nullptr) {
      const bool excluded = reverse_filter ? !bitmap->test(id) : bitmap->test(id);
      if (excluded) {
        continue;
      }
    }
    if (distance_threshold < FLT_MAX && dist_buffer_[static_cast<size_t>(i)] > distance_threshold) {
      continue;
    }
    ids_buffer_[static_cast<size_t>(kept)] = id;
    dist_buffer_[static_cast<size_t>(kept)] = dist_buffer_[static_cast<size_t>(i)];
    ++kept;
  }
  result_size = kept;
  ids = ids_buffer_.data();
  dist = dist_buffer_.data();
  return ret;
}

int HipvsIndexHandler::serialize(std::ostream &out_stream)
{
  int ret = OB_SUCCESS;
  if (index_ != nullptr) {
    char *buffer = nullptr;
    int64_t length = 0;
    const int32_t status = bridge_->index_serialize(index_, &buffer, &length);
    if (status != SEEKDB_HIPVS_OK) {
      ret = obhipvs::to_ob_error(status);
    } else {
      out_stream.write(buffer, length);
      if (!out_stream.good()) {
        ret = OB_ERR_UNEXPECTED;
      }
      bridge_->buffer_free(buffer);
    }
  } else if (OB_FAIL(ensure_fallback())) {
  } else {
    ret = fallback_->serialize(out_stream);
  }
  return ret;
}

int HipvsIndexHandler::deserialize(std::istream &in_stream)
{
  int ret = OB_SUCCESS;
  std::string payload;
  char chunk[65536];
  while (in_stream.good()) {
    in_stream.read(chunk, sizeof(chunk));
    const std::streamsize got = in_stream.gcount();
    if (got > 0) {
      payload.append(chunk, static_cast<size_t>(got));
    }
    if (!in_stream) {
      break;
    }
  }

  /* The payload is self describing: the GPU format starts with a magic, the
   * CPU format does not. This keeps an index readable by the backend that
   * wrote it, whatever node it is read on. */
  static const char HIPVS_PAYLOAD_MAGIC[8] = {'S', 'K', 'H', 'I', 'P', 'V', 'S', '1'};
  const bool is_gpu_payload = payload.size() >= sizeof(HIPVS_PAYLOAD_MAGIC)
                              && memcmp(payload.data(), HIPVS_PAYLOAD_MAGIC, sizeof(HIPVS_PAYLOAD_MAGIC)) == 0;
  if (is_gpu_payload) {
    if (index_ == nullptr) {
      LOG_WARN("[OBHIPVS] the index was written by the GPU backend, but the GPU backend is not available");
      ret = OB_NOT_SUPPORTED;
    } else {
      void *index = nullptr;
      const int32_t status = bridge_->index_deserialize(payload.data(),
                                                         static_cast<int64_t>(payload.size()),
                                                         &params_, &index);
      if (status != SEEKDB_HIPVS_OK) {
        ret = obhipvs::to_ob_error(status);
        LOG_WARN("[OBHIPVS] deserialize index failed", K(ret), K(payload.size()));
      } else {
        /* The bridge returns a new index, the empty one is dropped. */
        if (index_ != nullptr) {
          bridge_->index_destroy(index_);
        }
        index_ = index;
        LOG_INFO("[OBHIPVS] deserialize index success", K(payload.size()));
      }
    }
  } else if (OB_FAIL(ensure_fallback())) {
  } else {
    std::istringstream payload_stream(payload);
    ret = fallback_->deserialize(payload_stream);
  }
  return ret;
}

uint64_t HipvsIndexHandler::estimate_memory(uint64_t row_count, bool is_build)
{
  if (index_ == nullptr && fallback_ != nullptr) {
    return fallback_->estimate_memory(row_count, is_build);
  }
  /* An hnswlib index keeps the vectors, the graph links and the per element
   * bookkeeping. The estimate mirrors the CPU backend: the vectors plus the
   * links of the base layer. */
  UNUSED(is_build);
  const uint64_t dim = dim_ > 0 ? static_cast<uint64_t>(dim_) : 0;
  const uint64_t degree = max_degree_ > 0 ? static_cast<uint64_t>(max_degree_) * 2 : 32;
  uint64_t size = row_count * dim * sizeof(float);
  size += row_count * degree * sizeof(int32_t);
  size += row_count * sizeof(int64_t);
  return size;
}

int HnswIndexHandler::serialize(std::ostream &out_stream)
{
  int ret = OB_SUCCESS;
  tl::expected<void, Error> bs = index_->Serialize(out_stream);
  if (bs.has_value()) {
    LOG_INFO("[OBVSAG] serialize index success");
  } else {
    ret = vsag_errcode2ob(bs.error().type);
    LOG_WARN("[OBVSAG] fserialize error happend", K(ret), K(bs.error().type));
  }
  return ret;
}

int HnswIndexHandler::deserialize(std::istream &in_stream)
{
  int ret = OB_SUCCESS;
  std::shared_ptr<vsag::Index> hnsw_index;
  bool use_static = get_use_static();
  const char *metric = get_metric();
  const char *dtype = get_dtype();
  int max_degree = get_max_degree();
  int ef_construction = get_ef_construction();
  int ef_search = get_ef_search();
  int dim = get_dim();
  int index_type = get_index_type();
  uint64_t extra_info_size = get_extra_info_size();
  const char* index_type_str = get_index_type_str(index_type);
  int16_t refine_type = get_refine_type();
  int16_t bq_bits_query = get_bq_bits_query();
  bool bq_use_fht = get_bq_use_fht();
  bool use_reorder = get_use_reorder();
  float doc_prune_ratio = get_doc_prune_ratio();
  int window_size = get_window_size();

  char result_param_str[1024] = {0};
  if ((IndexType)index_type == IndexType::IPIVF_TYPE) {
    if (OB_FAIL(construct_vsag_sindi_create_param(uint8_t(index_type),
            dtype,
            metric,
            get_allocator(),
            extra_info_size,
            use_reorder,
            doc_prune_ratio,
            window_size,
            result_param_str))) {
    }
  } else {
    if (OB_FAIL(construct_vsag_create_param(
      uint8_t(index_type), dtype, metric, dim, max_degree,
      ef_construction, ef_search, get_allocator(),
      extra_info_size, refine_type, bq_bits_query, bq_use_fht, result_param_str))) {
    }
  }
  if (OB_FAIL(ret)) {
  } else {
    const std::string input_json_str(result_param_str);
    tl::expected<std::shared_ptr<Index>, Error> index = vsag::Factory::CreateIndex(index_type_str, input_json_str, get_allocator());
    if (index.has_value()) {
      hnsw_index = index.value();
      tl::expected<void, Error> bs = hnsw_index->Deserialize(in_stream);
      if (bs.has_value()) {
        set_index(hnsw_index);
        LOG_INFO("[OBVSAG] fdeserialize success", KCSTRING(result_param_str));
      } else {
        ret = vsag_errcode2ob(bs.error().type);
        LOG_WARN("[OBVSAG] fdeserialize error", K(ret), K(bs.error().type));
      }
    } else {
      ret = vsag_errcode2ob(index.error().type);
      LOG_WARN("[OBVSAG] create index error", K(ret), K(index.error().type));
    }
  }
  return ret;
}

int HnswIndexHandler::build_index(const vsag::DatasetPtr &base){
  int ret = OB_SUCCESS;
  try {
    tl::expected<std::vector<int64_t>, Error> result = index_->Build(base);
    if (result.has_value()) {
    } else {
      ret = vsag_errcode2ob(result.error().type);
    }
  } catch (const std::exception &e) {
    ret = OB_ERR_VSAG_RETURN_ERROR;
    LOG_WARN("[OBVSAG] exception caught in build_index", "what", e.what());
  } catch (...) {
    ret = OB_ERR_VSAG_RETURN_ERROR;
    LOG_WARN("[OBVSAG] unknown exception caught in build_index");
  }
  return ret;
}

int HnswIndexHandler::get_index_number()
{
  return index_->GetNumElements();
}

int HnswIndexHandler::add_index(const vsag::DatasetPtr &incremental)
{
  int ret = OB_SUCCESS;
  tl::expected<std::vector<int64_t>, Error> result = index_->Add(incremental);
  if (result.has_value()) {
    LOG_DEBUG("add index success", K(get_index_number()));
  } else {
    ret = vsag_errcode2ob(result.error().type);
  }
  return ret;
}

int HnswIndexHandler::cal_distance_by_id(uint32_t len, uint32_t *dims, float *vals,
                                         const int64_t *ids, int64_t count,
                                         const float *&dist)
{
  int ret = OB_SUCCESS;
  vsag::SparseVector sparse;
  sparse.len_ = len;
  sparse.ids_ = dims;
  sparse.vals_ = vals;
  DatasetPtr query = vsag::Dataset::Make();
  query->NumElements(1)->SparseVectors(&sparse)->Owner(false);
  float *dist_tmp = (float*)allocator_->Allocate(count * sizeof(float));
  if (OB_ISNULL(dist_tmp)) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  }
  // TODO(ningxin.ning): support CalcDistanceById in sparse vector
  for (int i = 0; i < count && OB_SUCC(ret); ++i) {
    // tl::expected<float, Error> result = index_->CalcDistanceById(query, ids[i]);
    // if (result.has_value()) {
    //   dist_tmp[i] = result.value();
    // } else {
    //   ret = vsag_errcode2ob(result.error().type);
    // }
    dist_tmp[i] = 0.1;
  }
  dist = dist_tmp;
  return ret;
}

int HnswIndexHandler::cal_distance_by_id(const float *vector,
                                         const int64_t *ids, int64_t count,
                                         const float *&dist)
{
  int ret = OB_SUCCESS;
  tl::expected<DatasetPtr, Error> result = index_->CalDistanceById(vector, ids, count);
  if (result.has_value()) {
    result.value()->Owner(false);
    dist = result.value()->GetDistances();
  } else {
    ret = vsag_errcode2ob(result.error().type);
  }
  return ret;
}

int HnswIndexHandler::get_extra_info_by_ids(const int64_t *ids, int64_t count,
                                            char *extra_infos)
{
  int ret = OB_SUCCESS;
  tl::expected<void, Error> result = index_->GetExtraInfoByIds(ids, count, extra_infos);
  if (result.has_value()) {
  } else {
    ret = vsag_errcode2ob(result.error().type);
  }
  return ret;
}

int HnswIndexHandler::get_vid_bound(int64_t &min_vid, int64_t &max_vid)
{
  int ret = OB_SUCCESS;
  int64_t element_cnt = index_->GetNumElements();
  if (element_cnt == 0) {
  } else {
    tl::expected<std::pair<int64_t, int64_t>, Error> result = index_->GetMinAndMaxId();
    if (result.has_value()) {
      min_vid = result.value().first;
      max_vid = result.value().second;
    } else {
      ret = vsag_errcode2ob(result.error().type);
    }
  }
  return ret;
}

uint64_t HnswIndexHandler::estimate_memory(const uint64_t row_count, const bool is_build)
{
  
  uint64_t size = 0;
  if (IPIVF_TYPE == index_type_) {
    // TODO(ningxin.ning): use vsag EstimateMemory
    size += 2 * sizeof(int64_t) * row_count;
    // nonzero dim = 100
    size += 100 * row_count * sizeof(float) * 2;
    if (use_reorder_) {
      size *= 2;
    }
  } else {
    size = index_->EstimateMemory(row_count);
  }
  if (HNSW_BQ_TYPE == index_type_ && is_build) {
    if (QuantizationType::SQ8 == refine_type_) {
      size += (row_count * dim_ * sizeof(uint8_t));
    } else {
      size += (row_count * dim_ * sizeof(float));
    }
  }
  return size;
}

int HnswIndexHandler::immutable_optimize()
{
  int ret = OB_SUCCESS;
  if (index_type_ == IPIVF_TYPE) {
    // TODO(ningxin.ning): support SetImmutable for sparse vector index
  } else {
    tl::expected<void, Error> res = index_->SetImmutable();
    if (res.has_value()) {
      LOG_INFO("[OBVSAG] set immutable success", KPC(this));
    } else {
      ret = vsag_errcode2ob(res.error().type);
      LOG_WARN("[OBVSAG] index set immutable error", K(ret), K(res.error().type));
    }
  }
  return ret;
}

int HnswIndexHandler::knn_search(const vsag::DatasetPtr &query, int64_t topk,
                                 const std::string &parameters,
                                 const float *&dist, const int64_t *&ids,
                                 int64_t &result_size, float valid_ratio,
                                 int index_type, FilterInterface *bitmap,
                                 bool reverse_filter, bool need_extra_info,
                                 const char *&extra_infos, void *allocator,
                                 float distance_threshold)
{
  int ret = OB_SUCCESS;
  std::function<bool(int64_t)> vid_filter = [bitmap, reverse_filter](int64_t id) -> bool {
    if (!reverse_filter) {
      return bitmap->test(id);
    } else {
      return !(bitmap->test(id));
    }
  };
  std::function<bool(const char *)> exinfo_filter = [bitmap, reverse_filter](const char *data) -> bool {
    if (!reverse_filter) {
      return bitmap->test(data);
    } else {
      return !(bitmap->test(data));
    }
  };

  std::shared_ptr<ObVasgFilter> vsag_filter = std::make_shared<ObVasgFilter>(valid_ratio, vid_filter, exinfo_filter);
  vsag::Allocator *vsag_allocator = nullptr;
  if (allocator != nullptr) vsag_allocator = static_cast<vsag::Allocator *>(allocator);
  tl::expected<std::shared_ptr<vsag::Dataset>, vsag::Error> result;
  if (index_type_ == IPIVF_TYPE) {
    result = index_->KnnSearch(query, topk, parameters, bitmap == nullptr ? nullptr : vsag_filter);
  } else {
    vsag::SearchParam search_param(false, parameters,
                                 bitmap == nullptr ? nullptr : vsag_filter,
                                 vsag_allocator);
    result = index_->KnnSearch(query, topk, search_param);
  }
  if (result.has_value()) {
    // the lifecycle of result
    result.value()->Owner(false);
    ids = result.value()->GetIds();
    dist = result.value()->GetDistances();
    result_size = result.value()->GetDim();
    if (need_extra_info) {
      extra_infos = result.value()->GetExtraInfos();
    }
  } else {
    ret = vsag_errcode2ob(result.error().type);
  }
  return ret;
}

int HnswIndexHandler::knn_search(const vsag::DatasetPtr &query, int64_t topk,
                                 const std::string &parameters,
                                 const float *&dist, const int64_t *&ids,
                                 int64_t &result_size, float valid_ratio,
                                 int index_type, FilterInterface *bitmap,
                                 bool reverse_filter, bool need_extra_info,
                                 const char *&extra_infos, void *&iter_ctx,
                                 bool is_last_search, void *allocator)
{
  int ret = OB_SUCCESS;
  std::function<bool(int64_t)> filter = [bitmap, reverse_filter](int64_t id) -> bool {
    if (!reverse_filter) {
      return bitmap->test(id);
    } else {
      return !(bitmap->test(id));
    }
  };
  std::function<bool(const char *)> exinfo_filter = [bitmap, reverse_filter](const char *data) -> bool {
    if (!reverse_filter) {
      return bitmap->test(data);
    } else {
      return !(bitmap->test(data));
    }
  };

  std::shared_ptr<ObVasgFilter> vsag_filter = std::make_shared<ObVasgFilter>(valid_ratio, filter, exinfo_filter);
  vsag::Allocator *vsag_allocator = nullptr;
  if (allocator != nullptr) vsag_allocator = static_cast<vsag::Allocator *>(allocator);
  vsag::IteratorContext *input_iter = static_cast<vsag::IteratorContext *>(iter_ctx);
  vsag::SearchParam search_param(true, parameters,
                                 bitmap == nullptr ? nullptr : vsag_filter,
                                 vsag_allocator, input_iter, is_last_search);
  tl::expected<std::shared_ptr<vsag::Dataset>, vsag::Error> result = index_->KnnSearch(query, topk, search_param);
  if (result.has_value()) {
    iter_ctx = search_param.iter_ctx;
    result.value()->Owner(false);
    ids = result.value()->GetIds();
    dist = result.value()->GetDistances();
    result_size = result.value()->GetDim();
    if (need_extra_info) {
      extra_infos = result.value()->GetExtraInfos();
    }
  } else {
    ret = vsag_errcode2ob(result.error().type);
  }
  return ret;
}

void set_log_level(int32_t ob_level_num)
{
  static std::map<int32_t, int32_t> ob2vsag_log_level = {
      {0 /*ERROR*/, vsag::Logger::Level::kERR},
      {1 /*WARN*/, vsag::Logger::Level::kWARN},
      {2 /*INFO*/, vsag::Logger::Level::kINFO},
      {3 /*EDIAG*/, vsag::Logger::Level::kERR},
      {4 /*WDIAG*/, vsag::Logger::Level::kWARN},
      {5 /*TRACE*/, vsag::Logger::Level::kTRACE},
      {6 /*DEBUG*/, vsag::Logger::Level::kDEBUG},
  };
  vsag::Options::Instance().logger()->SetLevel(
      static_cast<vsag::Logger::Level>(ob2vsag_log_level[ob_level_num]));
}

bool is_init_ = vsag::init();
bool is_init()
{
    LOG_INFO("[OBVSAG] Init VsagLib]:");
    if (is_init_) {
        LOG_INFO("[OBVSAG] Init VsagLib success");
    } else {
        LOG_INFO("[OBVSAG] Init VsagLib fail");
    }
    return is_init_; 
}

void set_logger(void *logger_ptr)
{
  vsag::Options::Instance().set_logger(static_cast<vsag::Logger *>(logger_ptr));
  vsag::Logger::Level log_level = static_cast<vsag::Logger::Level>(1); // default is debug level
  vsag::Options::Instance().logger()->SetLevel(log_level);
}

void set_block_size_limit(uint64_t size)
{
  vsag::Options::Instance().set_block_size_limit(size);
}

bool get_is_hgraph_type(uint8_t create_type) 
{
  bool res = false;
  switch (create_type) {
    case HNSW_TYPE: {
      res = false;
      break;
    }
    case HNSW_SQ_TYPE: 
    case HNSW_BQ_TYPE:
    case HGRAPH_TYPE: {
      res = true;
      break;
    }
  }
  return res;
}

const char* get_index_type_str(uint8_t create_type)
{
  const char* res;
  switch (create_type) {
    case HNSW_TYPE: {
      res = "hnsw";
      break;
    }
    case HNSW_SQ_TYPE: 
    case HNSW_BQ_TYPE:
    case HGRAPH_TYPE: {
      res = "hgraph";
      break;
    }
    case IPIVF_TYPE: {
      res = "sindi";
      break;
    }
  }
  return res;
}

const char* get_precise_quantization_type(const uint8_t type)
{
  const char* res = nullptr;
  if (type == QuantizationType::SQ8) {
    res = "sq8";
  } else {
    res = "fp32";
  }
  return res;
}


/**
  eg:
    hnsw: {
            "dtype": dtype, "metric_type": metric, "dim": dim, 
            "hnsw": {
              "max_degree": max_degree, "ef_construction": ef_construction, "ef_search": ef_search, "use_static": use_static
            }
          }
    hgraph: {
              "dtype": dtype, "metric_type": metric, "dim": dim, "extra_info_size": extra_info_size,
              "index_param": {
                "base_quantization_type": "fp32", "max_degree": max_degree, "ef_construction": ef_construction, "build_thread_count": 0
              }
            }
    sq: {
          "dtype": dtype, "metric_type": metric, "dim": dim, "extra_info_size": extra_info_size,
          "index_param": {
            "base_quantization_type": "sq8", "max_degree": max_degree, "ef_construction": ef_construction, "build_thread_count": 0
          }
        }
    bq: {
          "dtype": dtype, "metric_type": metric, "dim": dim, "extra_info_size": extra_info_size,
          "index_param": {
            "base_quantization_type": "rabitq", "max_degree": max_degree, "ef_construction": ef_construction, "build_thread_count": 0,
            "use_reorder": true, "ignore_reorder": true, "precise_quantization_type": "fp32", "precise_io_type": "block_memory_io"
          }
        }
*/
int construct_vsag_create_param(
    uint8_t create_type, const char *dtype, const char *metric, int dim,
    int max_degree, int ef_construction, int ef_search, void *allocator,
    int extra_info_size, int16_t refine_type, int16_t bq_bits_query,
    bool bq_use_fht, char *result_param_str)
{
  int ret = OB_SUCCESS;
  bool is_hgraph_type = get_is_hgraph_type(create_type);
  const char *index_type_str = is_hgraph_type ? "index_param" : "hnsw";
  const char *base_quantization_type;
  const int64_t buf_len = 1024;
  switch (create_type) {
  case HNSW_SQ_TYPE: {
    base_quantization_type = "sq8";
    break;
  }
  case HNSW_BQ_TYPE: {
    base_quantization_type = "rabitq";
    break;
  }
  case HGRAPH_TYPE: {
    base_quantization_type = "fp32";
    break;
  }
  default: {
    break;
  }
  }
  // ObIStreamBuf only supports seeking within the current callback buffer, while
  // VSAG's new format seeks to the footer. Keep the legacy format until global seek is supported.
  int64_t pos = 0;
  int64_t buff_size = 0;
  if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos, "{\"dim\":%d",
                              int(dim)))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos,
                                     ",\"dtype\":\"%s\"",
                                     dtype))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos,
                                     ",\"metric_type\":\"%s\"",
                                     metric))) {
  } else if (extra_info_size > 0 &&
             OB_FAIL(databuff_printf(result_param_str, buf_len, pos,
                                 ",\"extra_info_size\": %d",
                                 extra_info_size))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos,
                                 ",\"use_old_serial_format\":true"))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos,
                                     ",\"%s\":{",
                                     index_type_str))) {
  } else if (OB_FAIL(databuff_printf(
                 result_param_str, buf_len, pos, "\"ef_construction\":%d",
                 ef_construction))) {
  } else if (! is_hgraph_type && OB_FAIL(databuff_printf(result_param_str,
                                 buf_len, pos, ",\"ef_search\":%d",
                                 ef_search))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos,
                                     ",\"max_degree\":%d",
                                     max_degree))) {
  } else if (is_hgraph_type &&
      OB_FAIL(databuff_printf(
          result_param_str, buf_len, pos,
          ",\"base_quantization_type\":\"%s\"",
          base_quantization_type))) {
  } else if (is_hgraph_type &&
             OB_FAIL(databuff_printf(result_param_str, buf_len, pos,
                                     ",\"build_thread_count\":%d",
                                     0))) {
  } else if (create_type == HNSW_BQ_TYPE &&
             OB_FAIL(databuff_printf(
                 result_param_str, buf_len, pos,
                 ",\"use_reorder\":true"))) {
  } else if (create_type == HNSW_BQ_TYPE &&
             OB_FAIL(databuff_printf(
                 result_param_str, buf_len, pos,
                 ",\"ignore_reorder\":true"))) {
  } else if (create_type == HNSW_BQ_TYPE &&
             OB_FAIL(databuff_printf(
                 result_param_str, buf_len, pos,
                 ",\"precise_quantization_type\":\"%s\"", get_precise_quantization_type(refine_type)))) {
  } else if (create_type == HNSW_BQ_TYPE &&
             OB_FAIL(databuff_printf(result_param_str, buf_len, pos,
                                     ",\"precise_io_type\":\"block_memory_io\""))) {
  } else if (create_type == HNSW_BQ_TYPE &&
             OB_FAIL(databuff_printf(result_param_str, buf_len, pos,
                                     ",\"rabitq_bits_per_dim_query\":%d", bq_bits_query))) {
  } else if (create_type == HNSW_BQ_TYPE &&
             OB_FAIL(databuff_printf(result_param_str, buf_len, pos,
                                     ",\"rabitq_use_fht\":%s", (bq_use_fht ? "true" : "false")))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos,
                                     "}}"))) {
  }
  if (OB_SUCC(ret)) {
    LOG_INFO("build param", K(create_type), KCSTRING(result_param_str), K(lbt()));
  }
  return ret;
}

int construct_vsag_sindi_create_param(uint8_t create_type, const char *dtype, const char *metric, 
    void *allocator, int extra_info_size, bool use_reorder, float doc_prune_ratio, int window_size,
    char *result_param_str)
{
  int ret = OB_SUCCESS;
  const char *index_type_str = "index_param";
  const int64_t buf_len = 1024;

  int64_t pos = 0;
  int64_t buff_size = 0;
  // ObIStreamBuf exposes the serialized index through callback-backed chunks.
  // Skip seek-based footer handling and let SINDI read from that stream directly;
  // BufferStreamReader otherwise treats the current chunk length as the full stream.
  const bool deserialize_without_footer = true;
  const bool deserialize_without_buffer = true;
  if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos, "{\"dtype\":\"%s\"", dtype))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos, ",\"metric_type\":\"%s\"", metric))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos, ",\"dim\": 1024"))) {
  } else if (extra_info_size > 0 &&
             OB_FAIL(databuff_printf(result_param_str, buf_len, pos, ",\"extra_info_size\": %d", extra_info_size))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos, ",\"%s\":{", index_type_str))) {
  } else if (OB_FAIL(databuff_printf(
                 result_param_str, buf_len, pos, "\"use_reorder\":%s", use_reorder ? "true" : "false"))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos, ",\"doc_prune_ratio\":%f", doc_prune_ratio))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos, ",\"window_size\":%d", window_size))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos,
                                 ",\"deserialize_without_footer\":%s",
                                 (deserialize_without_footer ? "true": "false")))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos,
                                 ",\"deserialize_without_buffer\":%s",
                                 (deserialize_without_buffer ? "true": "false")))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos, "}}"))) {
  }
  if (OB_SUCC(ret)) {
    LOG_INFO("build param", K(create_type), KCSTRING(result_param_str), K(lbt()));
  }
  return ret;
}

/**
  eg:
    hnsw : {"hnsw": {"ef_search": ef_search, "skip_ratio": 0.7}}
    hgraph : {"hgraph": {"ef_search": ef_search, "use_extra_info_filter": use_extra_info_filter}}
*/
int construct_vsag_search_param(uint8_t create_type, 
                                int64_t ef_search, 
                                bool use_extra_info_filter, 
                                char *result_param_str)
{
  int ret = OB_SUCCESS;
  bool is_hgraph_type = get_is_hgraph_type(create_type);
  const char *index_type_str = is_hgraph_type ? "hgraph" : "hnsw";
  int64_t pos = 0;
  int64_t buff_size = 0;
  int64_t buf_len = 1024;
  if (OB_FAIL(databuff_printf(result_param_str, 
                        buf_len, 
                        pos, 
                        "{\"%s\":{", index_type_str))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, 
                        buf_len, 
                        pos, 
                        "\"ef_search\":%d", int(ef_search)))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, 
                        buf_len, 
                        pos, 
                        ",\"skip_ratio\":%f", 0.7))) {
  } else if (is_hgraph_type && OB_FAIL(databuff_printf(result_param_str, 
                        buf_len, 
                        pos, 
                        ",\"use_extra_info_filter\":%s", use_extra_info_filter ? "true" : "false"))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, 
                        buf_len, 
                        pos, 
                        "}}"))) {
  }
  if (OB_SUCC(ret)) {
    LOG_TRACE("search param", KCSTRING(result_param_str), K(lbt()));
  }
  return ret;
}

int construct_vsag_sindi_search_param(float query_prune_ratio, uint64_t n_candidate, 
                                char *result_param_str)
{
  int ret = OB_SUCCESS;
  const char *index_type_str = "sindi";
  int64_t pos = 0;
  int64_t buff_size = 0;
  int64_t buf_len = 1024;
  if (OB_FAIL(databuff_printf(result_param_str, 
                        buf_len, 
                        pos, 
                        "{\"%s\":{", index_type_str))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, 
                        buf_len, 
                        pos, 
                        "\"query_prune_ratio\":%f", query_prune_ratio))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, 
                        buf_len, 
                        pos, 
                        ",\"n_candidate\":%lu}}", n_candidate))) {
  }
  if (OB_SUCC(ret)) {
    LOG_TRACE("search param", KCSTRING(result_param_str), K(lbt()));
  }
  return ret;
}

/**
 * @brief Create an empty index owned by the AMD hipVS backend.
 *
 * The index starts empty; the GPU graph is built when the vectors are added
 * to it.
 */
static int create_hipvs_index(VectorIndexPtr &index_handler,
                               const char *dtype,
                               const char *metric,
                               int dim,
                               int max_degree,
                               int ef_construction,
                               int ef_search)
{
  int ret = OB_SUCCESS;
  const obhipvs::HipvsBridge *bridge = obhipvs::HipvsLoader::instance().get_bridge();
  seekdb_hipvs_index_params_t params = {};
  params.dim = dim;
  params.m = max_degree;
  params.ef_construction = ef_construction;
  params.ef_search = ef_search;
  params.graph_algo = SEEKDB_HIPVS_GRAPH_ALGO_CAGRA;
  /* The CPU hierarchy keeps the index mutable, so that the rows written after
   * the initial build can still be added to it. */
  params.hierarchy = SEEKDB_HIPVS_HIERARCHY_CPU;
  if (!obhipvs::to_bridge_metric(metric, params.metric)) {
    LOG_WARN("[OBHIPVS] the metric is not supported by the GPU backend", KCSTRING(metric));
    ret = OB_NOT_SUPPORTED;
  } else {
    void *index = nullptr;
    if (bridge != nullptr) {
      const int32_t status = bridge->index_create(&params, &index);
      if (status != SEEKDB_HIPVS_OK) {
        ret = obhipvs::to_ob_error(status);
        LOG_WARN("[OBHIPVS] create index failed", K(ret));
        index = nullptr;
      }
    } else {
      /* The GPU backend is not usable on this node. The handler is still
       * created, it falls back to the CPU backend. */
      LOG_WARN("[OBHIPVS] the GPU backend is not available, the index is built by the CPU backend");
    }
    if (OB_SUCC(ret)) {
      HipvsIndexHandler *handler = new HipvsIndexHandler(dtype, metric, max_degree, ef_construction,
                                                         ef_search, dim, params, bridge, index);
      if (OB_ISNULL(handler)) {
        if (index != nullptr && bridge != nullptr) {
          bridge->index_destroy(index);
        }
        ret = OB_ALLOCATE_MEMORY_FAILED;
      } else {
        index_handler = static_cast<VectorIndexPtr>(handler);
        LOG_INFO("[OBHIPVS] create index on the GPU backend", K(dim), K(max_degree),
                 K(ef_construction), K(ef_search), K(index != nullptr));
      }
    }
  }
  return ret;
}

int create_index(VectorIndexPtr &index_handler,
                 IndexType index_type, const char *dtype,
                 const char *metric, int dim, int max_degree,
                 int ef_construction, int ef_search, void *allocator,
                 int extra_info_size /* = 0*/, int16_t refine_type /*= 0*/,
                 int16_t bq_bits_query /*= 32*/, bool bq_use_fht /*= false*/,
                 int lib /* = LIB_VSAG */)
{
  int ret = OB_SUCCESS;
  if (dtype == nullptr || metric == nullptr) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    /* The GPU backend only supports the dense HNSW index. */
    const bool want_gpu = (lib == LIB_HIPVS)
                          && (index_type == HNSW_TYPE)
                          && (extra_info_size == 0)
                          && (refine_type == 0);
    if (want_gpu) {
      obhipvs::log_availability();
      /* The handler type is decided by the requested backend, not by whether
       * a GPU happens to be present, so that the format of an index never
       * depends on the node that wrote it. The handler falls back to the CPU
       * backend when the GPU backend is not usable. */
      ret = create_hipvs_index(index_handler, dtype, metric, dim, max_degree, ef_construction, ef_search);
      if (OB_SUCC(ret)) {
        return ret;
      }
      LOG_WARN("[OBHIPVS] create index on the GPU failed, falling back to the CPU backend", K(ret));
    }

    vsag::Allocator *vsag_allocator = nullptr;
    if (allocator == nullptr) {
      vsag_allocator = nullptr;
      LOG_INFO("[OBVSAG] allocator is null , use default_allocator", K(index_type), K(lbt()));
    } else {
      vsag_allocator = static_cast<vsag::Allocator *>(allocator);
      LOG_INFO("[OBVSAG] use caller allocator ", K(index_type), K(lbt()));
    }

    adjust_create_index_max_degree(index_type, max_degree);

    const char* index_type_str = get_index_type_str(index_type);
    char result_param_str[1024] = {0};
    if (OB_FAIL(construct_vsag_create_param(
        uint8_t(index_type), dtype, metric, dim, max_degree,
        ef_construction, ef_search, allocator, extra_info_size,
        refine_type, bq_bits_query, bq_use_fht, result_param_str))) {
    } else {
      const std::string input_json_str(result_param_str);
      tl::expected<std::shared_ptr<Index>, Error> index = vsag::Factory::CreateIndex(index_type_str, input_json_str, vsag_allocator);
      if (index.has_value()) {
        std::shared_ptr<vsag::Index> hnsw;
        hnsw = index.value();
        HnswIndexHandler *hnsw_index = new HnswIndexHandler(
            true, false, false, dtype, metric, max_degree, ef_construction,
            ef_search, dim, index_type, hnsw, vsag_allocator, extra_info_size,
            refine_type, bq_bits_query, bq_use_fht);
        if (OB_ISNULL(hnsw_index)) {
          ret = OB_ALLOCATE_MEMORY_FAILED;
        } else {
          index_handler = static_cast<VectorIndexPtr>(hnsw_index);
        }
      } else {
        ret = vsag_errcode2ob(index.error().type);
        LOG_WARN("[OBVSAG] create index error happend",
            K(ret), KCSTRING(result_param_str), K(index.error().type), KCSTRING(index.error().message.c_str()));
      }
    }
  }
  return ret;
}

int validate_create_index(const CreateIndexParam &param, std::string &err_msg)
{
  int ret = OB_SUCCESS;
  err_msg.clear();
  if (param.dtype_ == nullptr || param.metric_ == nullptr) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    vsag::Allocator *vsag_allocator = nullptr;
    if (param.allocator_ == nullptr) {
      vsag_allocator = nullptr;
      LOG_INFO("[OBVSAG] allocator is null , use default_allocator", K(param.index_type_), K(lbt()));
    } else {
      vsag_allocator = static_cast<vsag::Allocator *>(param.allocator_);
      LOG_INFO("[OBVSAG] use caller allocator ", K(param.index_type_), K(lbt()));
    }

    const char *index_type_str = get_index_type_str(param.index_type_);
    char result_param_str[1024] = {0};
    if (param.is_sparse_) {
      if (OB_FAIL(construct_vsag_sindi_create_param(uint8_t(param.index_type_),
                                                    param.dtype_,
                                                    param.metric_,
                                                    param.allocator_,
                                                    param.extra_info_size_,
                                                    param.use_reorder_,
                                                    param.doc_prune_ratio_,
                                                    param.window_size_,
                                                    result_param_str))) {
      }
    } else {
      int max_degree = param.max_degree_;
      adjust_create_index_max_degree(param.index_type_, max_degree);
      if (OB_FAIL(construct_vsag_create_param(
          uint8_t(param.index_type_), param.dtype_, param.metric_, param.dim_, max_degree,
          param.ef_construction_, param.ef_search_, param.allocator_, param.extra_info_size_,
          param.refine_type_, param.bq_bits_query_, param.bq_use_fht_, result_param_str))) {
      }
    }
    if (OB_SUCC(ret)) {
      const std::string input_json_str(result_param_str);
      tl::expected<std::shared_ptr<Index>, Error> index =
          vsag::Factory::CreateIndex(index_type_str, input_json_str, vsag_allocator);
      if (!index.has_value()) {
        ret = vsag_errcode2ob(index.error().type);
        fill_vsag_error_message(index.error(), err_msg);
        LOG_WARN("[OBVSAG] validate create index error",
            K(ret), KCSTRING(result_param_str), K(index.error().type), KCSTRING(index.error().message.c_str()));
      }
    }
  }
  return ret;
}

int create_index(VectorIndexPtr &index_handler, IndexType index_type, const char *dtype, const char *metric,
    bool use_reorder, float doc_prune_ratio, int window_size, void *allocator, int extra_info_size /* = 0*/)
{
  int ret = OB_SUCCESS;
  if (dtype == nullptr || metric == nullptr) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    vsag::Allocator *vsag_allocator = nullptr;
    if (allocator == nullptr) {
      vsag_allocator = nullptr;
      LOG_INFO("[OBVSAG] allocator is null , use default_allocator", K(index_type), K(lbt()));
    } else {
      vsag_allocator = static_cast<vsag::Allocator *>(allocator);
      LOG_INFO("[OBVSAG] use caller allocator ", K(index_type), K(lbt()));
    }

    const char *index_type_str = get_index_type_str(index_type);
    char result_param_str[1024] = {0};
    if (OB_FAIL(construct_vsag_sindi_create_param(uint8_t(index_type),
            dtype,
            metric,
            allocator,
            extra_info_size,
            use_reorder,
            doc_prune_ratio,
            window_size,
            result_param_str))) {
    } else {
      const std::string input_json_str(result_param_str);
      tl::expected<std::shared_ptr<Index>, Error> index =
          vsag::Factory::CreateIndex(index_type_str, input_json_str, vsag_allocator);
      if (index.has_value()) {
        std::shared_ptr<vsag::Index> hnsw;
        hnsw = index.value();
        HnswIndexHandler *hnsw_index = new HnswIndexHandler(true,
            false,
            false,
            dtype,
            metric,
            index_type,
            hnsw,
            vsag_allocator,
            extra_info_size,
            use_reorder,
            doc_prune_ratio,
            window_size);
        if (OB_ISNULL(hnsw_index)) {
          ret = OB_ALLOCATE_MEMORY_FAILED;
        } else {
          index_handler = static_cast<VectorIndexPtr>(hnsw_index);
        }
      } else {
        ret = vsag_errcode2ob(index.error().type);
        LOG_WARN("[OBVSAG] create index error happend", K(ret), KCSTRING(result_param_str), K(index.error().type));
      }
    }
  }
  return ret;
}

int build_index(VectorIndexPtr &index_handler, float *vector_list,
                int64_t *ids, int dim, int size, char *extra_infos /* = nullptr*/)
{
  int ret = OB_SUCCESS;
  if (index_handler == nullptr || vector_list == nullptr || ids == nullptr) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    VectorIndexHandler *handler = static_cast<VectorIndexHandler *>(index_handler);
    if (handler->get_backend() == LIB_HIPVS) {
      /* The GPU backend does not support the extra info of the HGRAPH index,
       * such an index is never created with this backend. */
      if (OB_FAIL(static_cast<HipvsIndexHandler *>(handler)->build_index(vector_list, ids, dim, size))) {
      }
    } else {
      HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(handler);
      DatasetPtr dataset = vsag::Dataset::Make();
      dataset->Dim(dim)
          ->NumElements(size)
          ->Ids(ids)
          ->Float32Vectors(vector_list)
          ->Owner(false);
      if (extra_infos != nullptr) {
        dataset->ExtraInfos(extra_infos);
      }
      if (OB_FAIL(hnsw->build_index(dataset))) {
      }
    }
  }
  return ret;
}

int build_index(VectorIndexPtr &index_handler, uint32_t *lens, uint32_t *dims, float *vals, int64_t *ids, int size,
    char *extra_info /* = nullptr*/)
{
  int ret = OB_SUCCESS;
  if (index_handler == nullptr || lens == nullptr || dims == nullptr || vals == nullptr || ids == nullptr) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    uint32_t *cur_dims_ptr = dims;
    float *cur_vals_ptr = vals;
    std::vector<vsag::SparseVector> sparse_vectors(size);
    for (int i = 0; i < size; i++) {
      sparse_vectors[i].len_ = lens[i];
      sparse_vectors[i].ids_ = cur_dims_ptr;
      sparse_vectors[i].vals_ = cur_vals_ptr;
      cur_dims_ptr += lens[i];
      cur_vals_ptr += lens[i];
    }
    HnswIndexHandler *handler = static_cast<HnswIndexHandler *>(index_handler);
    DatasetPtr dataset = vsag::Dataset::Make();
    dataset->NumElements(size)->Ids(ids)->SparseVectors(sparse_vectors.data())->Owner(false);
    if (extra_info != nullptr) {
      dataset->ExtraInfos(extra_info);
    }
    if (OB_FAIL(handler->build_index(dataset))) {
    }
  }
  return ret;
}

int add_index(VectorIndexPtr &index_handler, float *vector,
              int64_t *ids, int dim, int size,
              char *extra_info /* = nullptr*/)
{
  int ret = OB_SUCCESS;
  if (index_handler == nullptr || vector == nullptr || ids == nullptr) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    VectorIndexHandler *handler = static_cast<VectorIndexHandler *>(index_handler);
    if (handler->get_backend() == LIB_HIPVS) {
      if (OB_FAIL(static_cast<HipvsIndexHandler *>(handler)->add_index(vector, ids, dim, size))) {
      }
    } else {
      HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(handler);
      // add index
      DatasetPtr incremental = vsag::Dataset::Make();
      incremental->Dim(dim)
          ->NumElements(size)
          ->Ids(ids)
          ->Float32Vectors(vector)
          ->Owner(false);
      if (extra_info != nullptr) {
        incremental->ExtraInfos(extra_info);
      }
      if (OB_FAIL(hnsw->add_index(incremental))) {
      }
    }
  }
  return ret;
}

int add_index(VectorIndexPtr &index_handler, uint32_t *lens, uint32_t *dims, float *vals, int64_t *ids, int size,
    char *extra_info /* = nullptr*/)
{
  int ret = OB_SUCCESS;
  if (index_handler == nullptr || lens == nullptr || dims == nullptr || vals == nullptr || ids == nullptr) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    uint32_t *cur_dims_ptr = dims;
    float *cur_vals_ptr = vals;
    std::vector<vsag::SparseVector> sparse_vectors(size);
    for (int i = 0; i < size; i++) {
      sparse_vectors[i].len_ = lens[i];
      sparse_vectors[i].ids_ = cur_dims_ptr;
      sparse_vectors[i].vals_ = cur_vals_ptr;
      cur_dims_ptr += lens[i];
      cur_vals_ptr += lens[i];
    }
    const uint32_t MAX_DIM_LIMIT = 500000;
    uint32_t max_dim = 0;
    for (int i = 0; i < size && OB_SUCC(ret); i++) {
      uint32_t length = sparse_vectors[i].len_;
      for (int j = 0; j < length && OB_SUCC(ret); j++) {
        max_dim = MAX(max_dim, sparse_vectors[i].ids_[j]);
        if (OB_UNLIKELY(max_dim > MAX_DIM_LIMIT)) {
          ret = OB_NOT_SUPPORTED;
          LOG_USER_ERROR(OB_NOT_SUPPORTED, "sparse vector dimension greater than 500000 is");
        }
      }
    }
    if (OB_FAIL(ret)) {
      // do nothing
    } else {
      HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
      // add index
      DatasetPtr incremental = vsag::Dataset::Make();
      incremental->NumElements(size)->Ids(ids)->SparseVectors(sparse_vectors.data())->Owner(false);
      if (extra_info != nullptr) {
        incremental->ExtraInfos(extra_info);
      }
      if (OB_FAIL(hnsw->add_index(incremental))) {
      }
    }
  }
  return ret;
}

int get_index_type(VectorIndexPtr &index_handler)
{
  if (index_handler == nullptr) {
    return INVALID_INDEX_TYPE;
  }
  VectorIndexHandler *handler = static_cast<VectorIndexHandler *>(index_handler);
  if (handler->get_backend() == LIB_HIPVS) {
    return HNSW_TYPE;
  }
  HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(handler);
  return hnsw->get_index_type();
}

int get_index_number(VectorIndexPtr &index_handler, int64_t &size)
{
  int ret = OB_SUCCESS;
  if (index_handler == nullptr) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    VectorIndexHandler *handler = static_cast<VectorIndexHandler *>(index_handler);
    if (handler->get_backend() == LIB_HIPVS) {
      ret = static_cast<HipvsIndexHandler *>(handler)->get_index_number(size);
    } else {
      HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(handler);
      size = hnsw->get_index_number();
    }
  }
  return ret;
}

/**
 * @brief Whether the index is really served by the GPU backend.
 *
 * It is false when the index was created with `LIB_HIPVS` but the GPU
 * backend was not usable on this node, in which case the index is served by
 * the CPU backend.
 */
bool is_gpu_backed(VectorIndexPtr &index_handler)
{
  bool ret = false;
  if (index_handler != nullptr) {
    VectorIndexHandler *handler = static_cast<VectorIndexHandler *>(index_handler);
    if (handler->get_backend() == LIB_HIPVS) {
      ret = static_cast<HipvsIndexHandler *>(handler)->is_gpu_backed();
    }
  }
  return ret;
}

int cal_distance_by_id(VectorIndexPtr &index_handler,
                       const float *vector, const int64_t *ids, int64_t count,
                       const float *&distances)
{
  int ret = OB_SUCCESS;
  if (index_handler == nullptr) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    VectorIndexHandler *handler = static_cast<VectorIndexHandler *>(index_handler);
    if (handler->get_backend() == LIB_HIPVS) {
      /* The GPU backend does not keep a copy of the vectors around, so the
       * exact distance between a query and a set of ids cannot be computed.
       * The query path handles this the same way it handles the indexes that
       * do not support it either. */
      ret = OB_NOT_SUPPORTED;
    } else {
      HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(handler);
      if (OB_FAIL(hnsw->cal_distance_by_id(vector, ids, count, distances))) {
      }
    }
  }
  return ret;
}

int cal_distance_by_id(VectorIndexPtr &index_handler, uint32_t len, uint32_t *dims, float *vals, const int64_t *ids,
    int64_t count, const float *&distances)
{
  int ret = OB_SUCCESS;
  if (index_handler == nullptr) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    VectorIndexHandler *handler = static_cast<VectorIndexHandler *>(index_handler);
    if (handler->get_backend() == LIB_HIPVS) {
      ret = OB_NOT_SUPPORTED;
    } else {
      HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(handler);
      if (OB_FAIL(hnsw->cal_distance_by_id(len, dims, vals, ids, count, distances))) {
      }
    }
  }
  return ret;
}

int get_vid_bound(VectorIndexPtr &index_handler,
                         int64_t &min_vid, int64_t &max_vid)
{
  int ret = OB_SUCCESS;
  if (nullptr == index_handler) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    VectorIndexHandler *handler = static_cast<VectorIndexHandler *>(index_handler);
    if (handler->get_backend() == LIB_HIPVS) {
      if (OB_FAIL(static_cast<HipvsIndexHandler *>(handler)->get_vid_bound(min_vid, max_vid))) {
      }
    } else {
      HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(handler);
      const IndexType index_type = static_cast<IndexType>(hnsw->get_index_type());
      if (index_type == IPIVF_TYPE) {
        // TODO(ningxin.ning): support get_vid_bound for ipivf
        min_vid = 0;
        max_vid = 0;
      } else {
        if (OB_FAIL(hnsw->get_vid_bound(min_vid, max_vid))) {
        }
      }
    }
  }
  return ret;
}

int knn_search(VectorIndexPtr &index_handler, float *query_vector,
               int dim, int64_t topk, const float *&dist, const int64_t *&ids,
               int64_t &result_size, int ef_search, bool need_extra_info,
               const char *&extra_infos, void *invalid, bool reverse_filter,
               bool use_extra_info_filter, void *allocator, float valid_ratio,
               float distance_threshold)
{
  int ret = OB_SUCCESS;
  if (index_handler == nullptr || query_vector == nullptr) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    FilterInterface *bitmap = static_cast<FilterInterface *>(invalid);
    VectorIndexHandler *handler = static_cast<VectorIndexHandler *>(index_handler);
    if (handler->get_backend() == LIB_HIPVS) {
      if (need_extra_info) {
        /* The GPU backend does not store the extra info of the HGRAPH index. */
        ret = OB_NOT_SUPPORTED;
      } else {
        ret = static_cast<HipvsIndexHandler *>(handler)->knn_search(query_vector, dim, topk,
                                                                   ef_search, dist, ids,
                                                                   result_size, valid_ratio,
                                                                   bitmap, reverse_filter,
                                                                   distance_threshold);
      }
    } else {
      HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(handler);
      const IndexType index_type = static_cast<IndexType>(hnsw->get_index_type());
      char result_param_str[1024]= {0};
      const int64_t EF_SEARCH_LIMIT = 1000L;
      const int64_t AMPLIFICATION_FACTOR = 10;
      if (ef_search > EF_SEARCH_LIMIT) {
        int64_t index_number = hnsw->get_index_number();
        if (0 != index_number) {
          topk = topk < index_number ? topk : index_number;
        }
        int64_t ef_search_threshold = AMPLIFICATION_FACTOR * topk > EF_SEARCH_LIMIT ? AMPLIFICATION_FACTOR * topk : EF_SEARCH_LIMIT;
        ef_search = ef_search < ef_search_threshold ? ef_search : ef_search_threshold;
      }
      if (OB_FAIL(construct_vsag_search_param(uint8_t(index_type), ef_search, use_extra_info_filter, result_param_str))) {
      } else {
        const std::string input_json_string(result_param_str);
        DatasetPtr query = vsag::Dataset::Make();
        query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector)->Owner(false);
        if (OB_FAIL(hnsw->knn_search(query, topk, input_json_string, dist, ids,
                                     result_size, valid_ratio, index_type, bitmap,
                                     reverse_filter, need_extra_info, extra_infos, allocator, distance_threshold))) {
        }
      }
    }
  }
  return ret;
}

int knn_search(VectorIndexPtr &index_handler, float *query_vector,
               int dim, int64_t topk, const float *&dist, const int64_t *&ids,
               int64_t &result_size, int ef_search, bool need_extra_info,
               const char *&extra_infos, void *invalid, bool reverse_filter,
               bool use_extra_info_filter, float valid_ratio, void *&iter_ctx,
               bool is_last_search, void *allocator)
{
  int ret = OB_SUCCESS;
  if (index_handler == nullptr || query_vector == nullptr) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    FilterInterface *bitmap = static_cast<FilterInterface *>(invalid);
    VectorIndexHandler *handler = static_cast<VectorIndexHandler *>(index_handler);
    if (handler->get_backend() == LIB_HIPVS) {
      /* The GPU backend has no iterator context, the search is answered in one
       * shot. */
      iter_ctx = nullptr;
      if (need_extra_info) {
        ret = OB_NOT_SUPPORTED;
      } else {
        ret = static_cast<HipvsIndexHandler *>(handler)->knn_search(query_vector, dim, topk,
                                                                   ef_search, dist, ids,
                                                                   result_size, valid_ratio,
                                                                   bitmap, reverse_filter,
                                                                   FLT_MAX);
      }
    } else {
      HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(handler);
      const IndexType index_type = static_cast<IndexType>(hnsw->get_index_type());
      char result_param_str[1024]= {0};
      const int64_t EF_SEARCH_LIMIT = 1000L;
      const int64_t AMPLIFICATION_FACTOR = 10;
      if (ef_search > EF_SEARCH_LIMIT) {
        int64_t index_number = hnsw->get_index_number();
        if (0 != index_number) {
          topk = topk < index_number ? topk : index_number;
        }
        int64_t ef_search_threshold = AMPLIFICATION_FACTOR * topk > EF_SEARCH_LIMIT ? AMPLIFICATION_FACTOR * topk : EF_SEARCH_LIMIT;
        ef_search = ef_search < ef_search_threshold ? ef_search : ef_search_threshold;
      }
      if (OB_FAIL(construct_vsag_search_param(uint8_t(index_type), ef_search, use_extra_info_filter, result_param_str))) {
      } else {
        const std::string input_json_string(result_param_str);
        DatasetPtr query = vsag::Dataset::Make();
        query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector)->Owner(false);
        if (OB_FAIL(hnsw->knn_search(query, topk, input_json_string, dist, ids,
                              result_size, valid_ratio, index_type, bitmap,
                              reverse_filter, need_extra_info, extra_infos, iter_ctx,
                              is_last_search, allocator))) {
        }
      }
    }
  }
  return ret;
}

int knn_search(obvsag::VectorIndexPtr &index_handler, uint32_t len, uint32_t *dims, float *vals, int64_t topk,
    const float *&result_dist, const int64_t *&result_ids, const char *&extra_infos, int64_t &result_size,
    float query_prune_ratio, int64_t n_candidate, void *invalid, bool reverse_filter,
    bool is_extra_info_filter, float valid_ratio, void *allocator, bool need_extra_info)
{
  int ret = OB_SUCCESS;
  if (index_handler == nullptr || dims == nullptr || vals == nullptr) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    VectorIndexHandler *handler = static_cast<VectorIndexHandler *>(index_handler);
    if (handler->get_backend() == LIB_HIPVS) {
      /* The GPU backend only handles the dense index. */
      ret = OB_NOT_SUPPORTED;
    } else {
      FilterInterface *bitmap = static_cast<FilterInterface *>(invalid);
      HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(handler);
      const IndexType index_type = static_cast<IndexType>(hnsw->get_index_type());
      char result_param_str[1024]= {0};
      if (OB_FAIL(construct_vsag_sindi_search_param(query_prune_ratio, n_candidate, result_param_str))) {
      } else if (len == 0) {
        result_size = 0;
      } else {
        // VSAG SINDI requires n_candidate <= 500 * k. SeekDB supports refine_k up to 1000,
        // so use the minimum internal k accepted by VSAG and let the caller merge and trim
        // the extra results to the SQL query limit. This preserves the requested candidate pool.
        const int64_t min_vsag_topk = n_candidate / VSAG_SINDI_N_CANDIDATE_FACTOR
                                      + (n_candidate % VSAG_SINDI_N_CANDIDATE_FACTOR != 0);
        const int64_t vsag_topk = std::max(topk, min_vsag_topk);
        const std::string input_json_string(result_param_str);
        vsag::SparseVector sparse;
        sparse.len_ = len;
        sparse.ids_ = dims;
        sparse.vals_ = vals;
        DatasetPtr query = vsag::Dataset::Make();
        query->NumElements(1)->SparseVectors(&sparse)->Owner(false);
        if (OB_FAIL(hnsw->knn_search(query, vsag_topk, input_json_string, result_dist, result_ids,
                              result_size, valid_ratio, index_type, bitmap,
                              reverse_filter, need_extra_info, extra_infos, allocator))) {
        }
      }
    }
  }
  return ret;
}

int fserialize(VectorIndexPtr &index_handler, std::ostream &out_stream)
{
  int ret = OB_SUCCESS;
  if (index_handler == nullptr) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    VectorIndexHandler *handler = static_cast<VectorIndexHandler *>(index_handler);
    ret = handler->serialize(out_stream);
  }
  return ret;
}

int fdeserialize(VectorIndexPtr &index_handler,
                 std::istream &in_stream)
{
  int ret = OB_SUCCESS;
  if (index_handler == nullptr) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    VectorIndexHandler *handler = static_cast<VectorIndexHandler *>(index_handler);
    ret = handler->deserialize(in_stream);
  }
  return ret;
}

int delete_index(VectorIndexPtr &index_handler)
{
  int ret = OB_SUCCESS;
  if (index_handler != nullptr) {
    VectorIndexHandler *handler = static_cast<VectorIndexHandler *>(index_handler);
    if (handler->get_backend() == LIB_HIPVS) {
      delete static_cast<HipvsIndexHandler *>(handler);
    } else {
      HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(handler);
      LOG_INFO("[OBVSAG] delete index ",
          KP((void *)hnsw->get_index().get()), K(hnsw->get_index().use_count()), K(lbt()));
      delete hnsw;
    }
    index_handler = nullptr;
  }
  return ret;
}

void delete_iter_ctx(void *iter_ctx)
{
  LOG_TRACE("[OBVAG] delete_iter_ctx", KP(iter_ctx), K(lbt()));
  if (iter_ctx != nullptr) {
    delete static_cast<vsag::IteratorContext *>(iter_ctx);
    iter_ctx = nullptr;
  }
}

int get_extra_info_by_ids(VectorIndexPtr &index_handler,
                          const int64_t *ids, int64_t count,
                          char *extra_infos)
{
  int ret = OB_SUCCESS;
  if (index_handler == nullptr) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    VectorIndexHandler *handler = static_cast<VectorIndexHandler *>(index_handler);
    if (handler->get_backend() == LIB_HIPVS) {
      /* The GPU backend does not store the extra info of the HGRAPH index. */
      ret = OB_NOT_SUPPORTED;
    } else {
      HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(handler);
      if (OB_FAIL(hnsw->get_extra_info_by_ids(ids, count, extra_infos))) {
      }
    }
  }
  return ret;
}

uint64_t estimate_memory(VectorIndexPtr &index_handler, const uint64_t row_count, const bool is_build)
{
  uint64_t estimate_memory_size = 0;
  if (index_handler != nullptr) {
    VectorIndexHandler *handler = static_cast<VectorIndexHandler *>(index_handler);
    if (handler->get_backend() == LIB_HIPVS) {
      estimate_memory_size = static_cast<HipvsIndexHandler *>(handler)->estimate_memory(row_count, is_build);
    } else {
      HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(handler);
      estimate_memory_size = hnsw->estimate_memory(row_count, is_build);
    }
  }
  return estimate_memory_size;
}

int immutable_optimize(VectorIndexPtr& index_handler)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(index_handler)) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    VectorIndexHandler *handler = static_cast<VectorIndexHandler *>(index_handler);
    if (handler->get_backend() == LIB_HIPVS) {
      /* The GPU backend builds an immutable index, there is nothing to do. */
    } else {
      HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(handler);
      if (OB_FAIL(hnsw->immutable_optimize())) {
      }
    }
  }
  return ret;
}

} // namespace obvsag
} // namespace common
} // namespace oceanbase
