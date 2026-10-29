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
#include <map>
#include "vsag/vsag.h"
#include "vsag/errors.h"
#include "vsag/dataset.h"
#include "vsag/search_param.h"
#include "vsag/index.h"
#include "vsag/options.h"
#include "vsag/factory.h"
#include "lib/utility/ob_print_utils.h"
#include "lib/oblog/ob_log.h"
#include "lib/worker.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef SEEKDB_WITH_HIPVS
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/hnsw.h>
#include <dlpack/dlpack.h>
#include <unistd.h>
#endif

namespace oceanbase {
namespace common {
namespace obvsag {

using namespace vsag;

#ifdef SEEKDB_WITH_HIPVS

namespace {

constexpr uint64_t HIPVS_HNSW_MAGIC = 0x5344424849505648ULL;  // SDBHIPVH
constexpr uint64_t HIPVS_HNSW_VERSION = 1;
constexpr uint64_t HIPVS_HNSW_MAX_SERIALIZED_BYTES = 1ULL << 40;

std::string lower_ascii(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

bool env_disables_hipvs()
{
  const char *value = std::getenv("SEEKDB_HIPVS");
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  const std::string normalized = lower_ascii(value);
  return normalized == "0" || normalized == "false" || normalized == "off" || normalized == "disabled";
}

bool use_gpu_hnsw_hierarchy()
{
  const char *value = std::getenv("SEEKDB_HIPVS_GPU_HIERARCHY");
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  const std::string normalized = lower_ascii(value);
  return normalized == "1" || normalized == "true" || normalized == "on" || normalized == "enabled";
}

bool hipvs_float_dtype(const char *dtype)
{
  if (dtype == nullptr) {
    return false;
  }
  const std::string normalized = lower_ascii(dtype);
  return normalized == "float32" || normalized == "float";
}

size_t cagra_graph_degree(int hnsw_m)
{
  size_t degree = 32;
  while (degree < static_cast<size_t>(std::max(hnsw_m, 1)) && degree < 128) {
    degree <<= 1U;
  }
  return degree;
}

bool hipvs_metric(const char *metric, cuvsDistanceType &result)
{
  if (metric == nullptr) {
    return false;
  }
  const std::string normalized = lower_ascii(metric);
  if (normalized == "l2" || normalized == "l2sq" || normalized == "l2_squared") {
    result = L2Expanded;
    return true;
  }
  if (normalized == "ip" || normalized == "inner_product" || normalized == "innerproduct") {
    result = InnerProduct;
    return true;
  }
  return false;
}

vsag::Error make_hipvs_error(const char *operation)
{
  const char *detail = cuvsGetLastErrorText();
  std::string message(operation == nullptr ? "hipVS operation failed" : operation);
  if (detail != nullptr && detail[0] != '\0') {
    message.append(": ");
    message.append(detail);
  }
  return vsag::Error(vsag::ErrorType::UNKNOWN_ERROR, std::move(message));
}

std::string make_hipvs_temp_file()
{
  char path[] = "/tmp/seekdb_hipvs_hnsw_XXXXXX";
  const int fd = mkstemp(path);
  if (fd >= 0) {
    close(fd);
    return std::string(path);
  }
  return std::string();
}

void remove_hipvs_temp_file(const std::string &path)
{
  if (!path.empty()) {
    std::remove(path.c_str());
  }
}

void init_host_float_tensor(DLManagedTensor &tensor, void *data, int64_t *shape)
{
  tensor = DLManagedTensor{};
  tensor.dl_tensor.data = data;
  tensor.dl_tensor.device.device_type = kDLCPU;
  tensor.dl_tensor.device.device_id = 0;
  tensor.dl_tensor.ndim = 2;
  tensor.dl_tensor.dtype.code = kDLFloat;
  tensor.dl_tensor.dtype.bits = 32;
  tensor.dl_tensor.dtype.lanes = 1;
  tensor.dl_tensor.shape = shape;
  tensor.dl_tensor.strides = nullptr;
  tensor.dl_tensor.byte_offset = 0;
}

void init_host_uint64_tensor(DLManagedTensor &tensor, void *data, int64_t *shape)
{
  tensor = DLManagedTensor{};
  tensor.dl_tensor.data = data;
  tensor.dl_tensor.device.device_type = kDLCPU;
  tensor.dl_tensor.device.device_id = 0;
  tensor.dl_tensor.ndim = 2;
  tensor.dl_tensor.dtype.code = kDLUInt;
  tensor.dl_tensor.dtype.bits = 64;
  tensor.dl_tensor.dtype.lanes = 1;
  tensor.dl_tensor.shape = shape;
  tensor.dl_tensor.strides = nullptr;
  tensor.dl_tensor.byte_offset = 0;
}

int parse_ef_search(const std::string &parameters, int fallback)
{
  const std::string key = "\"ef_search\"";
  const auto key_pos = parameters.find(key);
  if (key_pos == std::string::npos) {
    return fallback;
  }
  const auto colon_pos = parameters.find(':', key_pos + key.size());
  if (colon_pos == std::string::npos) {
    return fallback;
  }
  const char *begin = parameters.c_str() + colon_pos + 1;
  char *end = nullptr;
  const long value = std::strtol(begin, &end, 10);
  if (end == begin || value <= 0 || value > std::numeric_limits<int>::max()) {
    return fallback;
  }
  return static_cast<int>(value);
}

template <typename T>
bool write_scalar(std::ostream &stream, T value)
{
  stream.write(reinterpret_cast<const char *>(&value), sizeof(T));
  return stream.good();
}

template <typename T>
bool read_scalar(std::istream &stream, T &value)
{
  stream.read(reinterpret_cast<char *>(&value), sizeof(T));
  return stream.good();
}

class HipVSHnswIndex final : public vsag::Index {
public:
  HipVSHnswIndex(int dim,
                 cuvsDistanceType metric,
                 int max_degree,
                 int ef_construction,
                 int ef_search,
                 vsag::Allocator *allocator)
      : dim_(dim),
        metric_(metric),
        max_degree_(std::max(max_degree, 2)),
        ef_construction_(std::max(ef_construction, 1)),
        ef_search_(std::max(ef_search, 1)),
        allocator_(allocator),
        use_gpu_hierarchy_(use_gpu_hnsw_hierarchy())
  {
    if (cuvsResourcesCreate(&resources_) == CUVS_SUCCESS &&
        cuvsHnswIndexCreate(&index_) == CUVS_SUCCESS) {
      available_ = true;
    } else {
      const char *detail = cuvsGetLastErrorText();
      LOG_WARN("[OBVSAG][HIPVS] failed to create hipVS resources or HNSW index",
               KCSTRING(detail == nullptr ? "" : detail));
      if (index_ != nullptr) {
        cuvsHnswIndexDestroy(index_);
        index_ = nullptr;
      }
      if (resources_ != 0) {
        cuvsResourcesDestroy(resources_);
        resources_ = 0;
      }
    }
  }

  ~HipVSHnswIndex() override
  {
    if (index_ != nullptr) {
      cuvsHnswIndexDestroy(index_);
      index_ = nullptr;
    }
    if (resources_ != 0) {
      cuvsResourcesDestroy(resources_);
      resources_ = 0;
    }
  }

  [[nodiscard]] bool available() const { return available_; }

  [[nodiscard]] vsag::IndexType GetIndexType() const override
  {
    return vsag::IndexType::HGRAPH;
  }

  tl::expected<std::vector<int64_t>, vsag::Error> Build(const vsag::DatasetPtr &base) override
  {
    if (!available_ || base == nullptr) {
      return tl::unexpected(vsag::Error(vsag::ErrorType::INVALID_ARGUMENT,
                                        "hipVS HNSW build requires an available backend and dataset"));
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (built_ || !external_ids_.empty()) {
      return tl::unexpected(vsag::Error(vsag::ErrorType::BUILD_TWICE,
                                        "hipVS HNSW index has already been built"));
    }
    const int64_t count = base->GetNumElements();
    if (dim_ <= 0 || count < 0 || base->GetDim() != dim_) {
      return tl::unexpected(vsag::Error(vsag::ErrorType::DIMENSION_NOT_EQUAL,
                                        "hipVS HNSW build dimension mismatch"));
    }
    if (count == 0) {
      return std::vector<int64_t>();
    }
    const float *vectors = base->GetFloat32Vectors();
    if (vectors == nullptr) {
      return tl::unexpected(vsag::Error(vsag::ErrorType::INVALID_ARGUMENT,
                                        "hipVS HNSW build requires float32 vectors"));
    }
    if (count > static_cast<int64_t>(std::numeric_limits<size_t>::max() / sizeof(float) /
                                     static_cast<size_t>(dim_))) {
      return tl::unexpected(vsag::Error(vsag::ErrorType::NO_ENOUGH_MEMORY,
                                        "hipVS HNSW vector payload is too large"));
    }

    cuvsCagraIndexParams_t cagra_params = nullptr;
    cuvsCagraIndex_t cagra_index = nullptr;
    cuvsHnswIndexParams_t hnsw_params = nullptr;
    auto cleanup = [&]() {
      if (hnsw_params != nullptr) {
        cuvsHnswIndexParamsDestroy(hnsw_params);
      }
      if (cagra_index != nullptr) {
        cuvsCagraIndexDestroy(cagra_index);
      }
      if (cagra_params != nullptr) {
        cuvsCagraIndexParamsDestroy(cagra_params);
      }
    };

    if (cuvsCagraIndexParamsCreate(&cagra_params) != CUVS_SUCCESS ||
        cuvsCagraIndexCreate(&cagra_index) != CUVS_SUCCESS) {
      cleanup();
      return tl::unexpected(make_hipvs_error("create hipVS CAGRA parameters"));
    }

    const size_t cagra_degree = cagra_graph_degree(max_degree_);
    cagra_params->metric = metric_;
    cagra_params->graph_degree = cagra_degree;
    cagra_params->intermediate_graph_degree = std::min<size_t>(cagra_degree * 2U, 256U);
    cagra_params->build_algo = AUTO_SELECT;
    cagra_params->nn_descent_niter = 0;
    cagra_params->compression = nullptr;

    int64_t dataset_shape[2] = {count, dim_};
    DLManagedTensor dataset_tensor{};
    init_host_float_tensor(dataset_tensor, const_cast<float *>(vectors), dataset_shape);
    if (cuvsCagraBuild(resources_, cagra_params, &dataset_tensor, cagra_index) != CUVS_SUCCESS) {
      cleanup();
      return tl::unexpected(make_hipvs_error("build hipVS CAGRA graph"));
    }

    if (cuvsHnswIndexParamsCreate(&hnsw_params) != CUVS_SUCCESS) {
      cleanup();
      return tl::unexpected(make_hipvs_error("create hipVS HNSW parameters"));
    }
    hnsw_params->hierarchy = use_gpu_hierarchy_ ? cuvsHnswHierarchy::GPU : cuvsHnswHierarchy::CPU;
    hnsw_params->ef_construction = ef_construction_;
    hnsw_params->num_threads = 0;
    if (cuvsHnswFromCagra(resources_, hnsw_params, cagra_index, index_) != CUVS_SUCCESS) {
      cleanup();
      return tl::unexpected(make_hipvs_error("convert CAGRA graph to hipVS HNSW"));
    }
    cleanup();
    index_->dtype.code = kDLFloat;
    index_->dtype.bits = 32;
    index_->dtype.lanes = 1;

    external_ids_.resize(static_cast<size_t>(count));
    vectors_.resize(static_cast<size_t>(count) * static_cast<size_t>(dim_));
    row_by_id_.clear();
    row_by_id_.reserve(static_cast<size_t>(count));
    const int64_t *ids = base->GetIds();
    for (int64_t i = 0; i < count; ++i) {
      external_ids_[static_cast<size_t>(i)] = ids == nullptr ? i : ids[i];
      row_by_id_[external_ids_[static_cast<size_t>(i)]] = static_cast<size_t>(i);
    }
    std::copy(vectors, vectors + static_cast<size_t>(count) * static_cast<size_t>(dim_),
              vectors_.begin());
    built_ = true;
    serialized_bytes_ = 0;
    LOG_INFO("[OBVSAG][HIPVS] built HNSW index on GPU",
             K(count), K(dim_), K(max_degree_), K(ef_construction_),
             K(use_gpu_hierarchy_));
    return std::vector<int64_t>();
  }

  tl::expected<std::vector<int64_t>, vsag::Error> Add(const vsag::DatasetPtr &base) override
  {
    if (!built_ || !available_ || base == nullptr) {
      return tl::unexpected(vsag::Error(vsag::ErrorType::INDEX_EMPTY,
                                        "hipVS HNSW add requires a built index"));
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (use_gpu_hierarchy_) {
      return tl::unexpected(vsag::Error(
          vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION,
          "hipVS GPU hierarchy is immutable; use CPU hierarchy for incremental HNSW adds"));
    }
    const int64_t count = base->GetNumElements();
    const float *vectors = base->GetFloat32Vectors();
    if (dim_ <= 0 || count <= 0 || vectors == nullptr || base->GetDim() != dim_) {
      return tl::unexpected(vsag::Error(vsag::ErrorType::INVALID_ARGUMENT,
                                        "invalid hipVS HNSW add dataset"));
    }
    if (count > static_cast<int64_t>(std::numeric_limits<size_t>::max() / sizeof(float) /
                                     static_cast<size_t>(dim_)) ||
        vectors_.size() > std::numeric_limits<size_t>::max() -
                              static_cast<size_t>(count) * static_cast<size_t>(dim_)) {
      return tl::unexpected(vsag::Error(vsag::ErrorType::NO_ENOUGH_MEMORY,
                                        "hipVS HNSW vector payload is too large"));
    }
    cuvsHnswExtendParams_t extend_params = nullptr;
    if (cuvsHnswExtendParamsCreate(&extend_params) != CUVS_SUCCESS || extend_params == nullptr) {
      return tl::unexpected(make_hipvs_error("create hipVS HNSW extend parameters"));
    }
    extend_params->num_threads = 0;

    int64_t dataset_shape[2] = {count, dim_};
    DLManagedTensor dataset_tensor{};
    init_host_float_tensor(dataset_tensor, const_cast<float *>(vectors), dataset_shape);
    const cuvsError_t result = cuvsHnswExtend(resources_, extend_params, &dataset_tensor, index_);
    cuvsHnswExtendParamsDestroy(extend_params);
    if (result != CUVS_SUCCESS) {
      return tl::unexpected(make_hipvs_error("extend hipVS HNSW index"));
    }

    const int64_t *ids = base->GetIds();
    const size_t old_count = external_ids_.size();
    external_ids_.reserve(old_count + static_cast<size_t>(count));
    vectors_.reserve(vectors_.size() + static_cast<size_t>(count) * static_cast<size_t>(dim_));
    for (int64_t i = 0; i < count; ++i) {
      const int64_t id = ids == nullptr ? static_cast<int64_t>(old_count + static_cast<size_t>(i)) : ids[i];
      external_ids_.push_back(id);
      row_by_id_[id] = old_count + static_cast<size_t>(i);
    }
    vectors_.insert(vectors_.end(), vectors,
                    vectors + static_cast<size_t>(count) * static_cast<size_t>(dim_));
    serialized_bytes_ = 0;
    return std::vector<int64_t>();
  }

  [[nodiscard]] tl::expected<vsag::DatasetPtr, vsag::Error> KnnSearch(
      const vsag::DatasetPtr &query,
      int64_t k,
      const std::string &parameters,
      vsag::BitsetPtr invalid = nullptr) const override
  {
    if (invalid == nullptr) {
      return do_knn_search(query, k, parameters, allocator_, nullptr, 1.0F);
    }
    std::function<bool(int64_t)> valid_filter = [invalid](int64_t id) {
      return !invalid->Test(id);
    };
    const int64_t total = static_cast<int64_t>(external_ids_.size());
    const int64_t invalid_count = static_cast<int64_t>(invalid->Count());
    const float valid_ratio = total <= 0 ? 0.0F
                                         : static_cast<float>(std::max<int64_t>(0, total - invalid_count)) /
                                               static_cast<float>(total);
    return do_knn_search(query, k, parameters, allocator_, &valid_filter, valid_ratio);
  }

  [[nodiscard]] tl::expected<vsag::DatasetPtr, vsag::Error> KnnSearch(
      const vsag::DatasetPtr &query,
      int64_t k,
      const std::string &parameters,
      const std::function<bool(int64_t)> &filter) const override
  {
    return do_knn_search(query, k, parameters, allocator_, filter ? &filter : nullptr, 0.5F);
  }

  [[nodiscard]] tl::expected<vsag::DatasetPtr, vsag::Error> KnnSearch(
      const vsag::DatasetPtr &query,
      int64_t k,
      vsag::SearchParam &search_param) const override
  {
    if (search_param.iter_ctx != nullptr) {
      return tl::unexpected(vsag::Error(
          vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION,
          "hipVS HNSW does not support iterative search"));
    }
    if (search_param.filter == nullptr) {
      return do_knn_search(query, k, search_param.parameters, search_param.allocator, nullptr, 1.0F);
    }
    const vsag::FilterPtr filter = search_param.filter;
    std::function<bool(int64_t)> valid_filter = [filter](int64_t id) {
      return filter->CheckValid(id);
    };
    return do_knn_search(query, k, search_param.parameters, search_param.allocator,
                         &valid_filter, filter->ValidRatio());
  }

  [[nodiscard]] tl::expected<vsag::DatasetPtr, vsag::Error> RangeSearch(
      const vsag::DatasetPtr &,
      float,
      const std::string &,
      int64_t = -1) const override
  {
    return tl::unexpected(vsag::Error(vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION,
                                      "hipVS HNSW does not support range search"));
  }

  [[nodiscard]] tl::expected<vsag::DatasetPtr, vsag::Error> RangeSearch(
      const vsag::DatasetPtr &,
      float,
      const std::string &,
      vsag::BitsetPtr,
      int64_t = -1) const override
  {
    return tl::unexpected(vsag::Error(vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION,
                                      "hipVS HNSW does not support range search"));
  }

  [[nodiscard]] tl::expected<vsag::DatasetPtr, vsag::Error> RangeSearch(
      const vsag::DatasetPtr &,
      float,
      const std::string &,
      const std::function<bool(int64_t)> &,
      int64_t = -1) const override
  {
    return tl::unexpected(vsag::Error(vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION,
                                      "hipVS HNSW does not support range search"));
  }

  [[nodiscard]] tl::expected<vsag::BinarySet, vsag::Error> Serialize() const override
  {
    return tl::unexpected(vsag::Error(vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION,
                                      "use streaming serialization for hipVS HNSW"));
  }

  tl::expected<void, vsag::Error> Serialize(std::ostream &out_stream) override
  {
    if (!built_ || index_ == nullptr) {
      return tl::unexpected(vsag::Error(vsag::ErrorType::INDEX_EMPTY,
                                        "cannot serialize an empty hipVS HNSW index"));
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const std::string temp_path = make_hipvs_temp_file();
    if (temp_path.empty()) {
      return tl::unexpected(vsag::Error(vsag::ErrorType::UNKNOWN_ERROR,
                                        "failed to create temporary hipVS serialization file"));
    }
    const cuvsError_t result = cuvsHnswSerialize(resources_, temp_path.c_str(), index_);
    if (result != CUVS_SUCCESS) {
      remove_hipvs_temp_file(temp_path);
      return tl::unexpected(make_hipvs_error("serialize hipVS HNSW index"));
    }
    std::ifstream input(temp_path, std::ios::binary | std::ios::ate);
    if (!input.good()) {
      remove_hipvs_temp_file(temp_path);
      return tl::unexpected(vsag::Error(vsag::ErrorType::READ_ERROR,
                                        "failed to read temporary hipVS serialization file"));
    }
    const std::streamoff file_size = input.tellg();
    input.seekg(0, std::ios::beg);
    if (file_size < 0) {
      remove_hipvs_temp_file(temp_path);
      return tl::unexpected(vsag::Error(vsag::ErrorType::READ_ERROR,
                                        "invalid hipVS serialization size"));
    }
    std::vector<char> blob(static_cast<size_t>(file_size));
    if (!blob.empty()) {
      input.read(blob.data(), static_cast<std::streamsize>(blob.size()));
    }
    const bool read_ok = input.good() || input.eof();
    input.close();
    remove_hipvs_temp_file(temp_path);
    if (!read_ok) {
      return tl::unexpected(vsag::Error(vsag::ErrorType::READ_ERROR,
                                        "failed to read hipVS serialization payload"));
    }

    const uint64_t count = static_cast<uint64_t>(external_ids_.size());
    const uint64_t blob_size = static_cast<uint64_t>(blob.size());
    if (!write_scalar(out_stream, HIPVS_HNSW_MAGIC) ||
        !write_scalar(out_stream, HIPVS_HNSW_VERSION) ||
        !write_scalar(out_stream, static_cast<uint64_t>(dim_)) ||
        !write_scalar(out_stream, static_cast<uint64_t>(metric_)) ||
        !write_scalar(out_stream, static_cast<uint64_t>(max_degree_)) ||
        !write_scalar(out_stream, static_cast<uint64_t>(ef_construction_)) ||
        !write_scalar(out_stream, static_cast<uint64_t>(ef_search_)) ||
        !write_scalar(out_stream, static_cast<uint64_t>(use_gpu_hierarchy_ ? 1 : 0)) ||
        !write_scalar(out_stream, count) ||
        (!external_ids_.empty() &&
         !out_stream.write(reinterpret_cast<const char *>(external_ids_.data()),
                           static_cast<std::streamsize>(count * sizeof(int64_t)))) ||
        (!vectors_.empty() &&
         !out_stream.write(reinterpret_cast<const char *>(vectors_.data()),
                           static_cast<std::streamsize>(vectors_.size() * sizeof(float)))) ||
        !write_scalar(out_stream, blob_size) ||
        (!blob.empty() &&
         !out_stream.write(blob.data(), static_cast<std::streamsize>(blob.size())))) {
      return tl::unexpected(vsag::Error(vsag::ErrorType::UNKNOWN_ERROR,
                                        "failed to write hipVS HNSW serialization stream"));
    }
    return {};
  }

  tl::expected<void, vsag::Error> Deserialize(const vsag::BinarySet &) override
  {
    return tl::unexpected(vsag::Error(vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION,
                                      "use streaming deserialization for hipVS HNSW"));
  }

  tl::expected<void, vsag::Error> Deserialize(const vsag::ReaderSet &) override
  {
    return tl::unexpected(vsag::Error(vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION,
                                      "use streaming deserialization for hipVS HNSW"));
  }

  tl::expected<void, vsag::Error> Deserialize(std::istream &in_stream) override
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (built_ || !external_ids_.empty()) {
      return tl::unexpected(vsag::Error(vsag::ErrorType::INDEX_NOT_EMPTY,
                                        "cannot deserialize into a non-empty hipVS HNSW index"));
    }
    uint64_t magic = 0;
    uint64_t version = 0;
    uint64_t dim = 0;
    uint64_t metric = 0;
    uint64_t max_degree = 0;
    uint64_t ef_construction = 0;
    uint64_t ef_search = 0;
    uint64_t gpu_hierarchy = 0;
    uint64_t count = 0;
    if (!read_scalar(in_stream, magic) || magic != HIPVS_HNSW_MAGIC ||
        !read_scalar(in_stream, version) || version != HIPVS_HNSW_VERSION ||
        !read_scalar(in_stream, dim) || dim != static_cast<uint64_t>(dim_) ||
        !read_scalar(in_stream, metric) || metric != static_cast<uint64_t>(metric_) ||
        !read_scalar(in_stream, max_degree) ||
        !read_scalar(in_stream, ef_construction) ||
        !read_scalar(in_stream, ef_search) ||
        !read_scalar(in_stream, gpu_hierarchy) ||
        !read_scalar(in_stream, count)) {
      return tl::unexpected(vsag::Error(vsag::ErrorType::INVALID_BINARY,
                                        "invalid hipVS HNSW header"));
    }
    max_degree_ = static_cast<int>(max_degree);
    ef_construction_ = static_cast<int>(ef_construction);
    ef_search_ = static_cast<int>(ef_search);
    use_gpu_hierarchy_ = gpu_hierarchy != 0;

    if (count > std::numeric_limits<size_t>::max() / sizeof(int64_t) ||
        count > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max()) / sizeof(int64_t)) {
      return tl::unexpected(vsag::Error(vsag::ErrorType::INVALID_BINARY,
                                        "hipVS HNSW id table is too large"));
    }
    std::vector<int64_t> ids(static_cast<size_t>(count));
    if (count > 0 &&
        !in_stream.read(reinterpret_cast<char *>(ids.data()),
                        static_cast<std::streamsize>(count * sizeof(int64_t)))) {
      return tl::unexpected(vsag::Error(vsag::ErrorType::INVALID_BINARY,
                                        "truncated hipVS HNSW id table"));
    }
    if (count > std::numeric_limits<size_t>::max() / sizeof(float) /
                    static_cast<size_t>(dim_) ||
        count * static_cast<uint64_t>(dim_) * sizeof(float) >
            static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
      return tl::unexpected(vsag::Error(vsag::ErrorType::INVALID_BINARY,
                                        "hipVS HNSW vector payload is too large"));
    }
    std::vector<float> vectors(static_cast<size_t>(count) * static_cast<size_t>(dim_));
    if (!vectors.empty() &&
        !in_stream.read(reinterpret_cast<char *>(vectors.data()),
                        static_cast<std::streamsize>(vectors.size() * sizeof(float)))) {
      return tl::unexpected(vsag::Error(vsag::ErrorType::INVALID_BINARY,
                                        "truncated hipVS HNSW vector payload"));
    }
    uint64_t blob_size = 0;
    if (!read_scalar(in_stream, blob_size) || blob_size > HIPVS_HNSW_MAX_SERIALIZED_BYTES ||
        blob_size > std::numeric_limits<size_t>::max()) {
      return tl::unexpected(vsag::Error(vsag::ErrorType::INVALID_BINARY,
                                        "invalid hipVS HNSW payload size"));
    }
    std::vector<char> blob(static_cast<size_t>(blob_size));
    if (blob_size > 0 &&
        !in_stream.read(blob.data(), static_cast<std::streamsize>(blob.size()))) {
      return tl::unexpected(vsag::Error(vsag::ErrorType::INVALID_BINARY,
                                        "truncated hipVS HNSW payload"));
    }

    const std::string temp_path = make_hipvs_temp_file();
    if (temp_path.empty()) {
      return tl::unexpected(vsag::Error(vsag::ErrorType::UNKNOWN_ERROR,
                                        "failed to create temporary hipVS deserialization file"));
    }
    {
      std::ofstream output(temp_path, std::ios::binary | std::ios::trunc);
      if (!output.good() ||
          (!blob.empty() &&
           !output.write(blob.data(), static_cast<std::streamsize>(blob.size())))) {
        remove_hipvs_temp_file(temp_path);
        return tl::unexpected(vsag::Error(vsag::ErrorType::READ_ERROR,
                                          "failed to stage hipVS HNSW payload"));
      }
    }

    index_->dtype.code = kDLFloat;
    index_->dtype.bits = 32;
    index_->dtype.lanes = 1;
    cuvsHnswIndexParams_t hnsw_params = nullptr;
    if (cuvsHnswIndexParamsCreate(&hnsw_params) != CUVS_SUCCESS || hnsw_params == nullptr) {
      remove_hipvs_temp_file(temp_path);
      return tl::unexpected(make_hipvs_error("create hipVS HNSW deserialize parameters"));
    }
    hnsw_params->hierarchy = use_gpu_hierarchy_ ? cuvsHnswHierarchy::GPU : cuvsHnswHierarchy::CPU;
    hnsw_params->ef_construction = ef_construction_;
    hnsw_params->num_threads = 0;
    const cuvsError_t result = cuvsHnswDeserialize(
        resources_, hnsw_params, temp_path.c_str(), dim_, metric_, index_);
    cuvsHnswIndexParamsDestroy(hnsw_params);
    remove_hipvs_temp_file(temp_path);
    if (result != CUVS_SUCCESS) {
      return tl::unexpected(make_hipvs_error("deserialize hipVS HNSW index"));
    }
    external_ids_ = std::move(ids);
    vectors_ = std::move(vectors);
    row_by_id_.clear();
    row_by_id_.reserve(external_ids_.size());
    for (size_t i = 0; i < external_ids_.size(); ++i) {
      row_by_id_[external_ids_[i]] = i;
    }
    serialized_bytes_ = blob_size;
    built_ = true;
    return {};
  }

  [[nodiscard]] int64_t GetNumElements() const override
  {
    return static_cast<int64_t>(external_ids_.size());
  }

  [[nodiscard]] tl::expected<float, vsag::Error> CalcDistanceById(
      const float *vector,
      int64_t id,
      bool = true) const override
  {
    if (vector == nullptr || dim_ <= 0) {
      return tl::unexpected(vsag::Error(vsag::ErrorType::INVALID_ARGUMENT,
                                        "invalid hipVS HNSW distance query"));
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = row_by_id_.find(id);
    if (it == row_by_id_.end()) {
      return -1.0F;
    }
    const float *stored = vectors_.data() + it->second * static_cast<size_t>(dim_);
    double value = 0.0;
    if (metric_ == L2Expanded) {
      for (int i = 0; i < dim_; ++i) {
        const double diff = static_cast<double>(vector[i]) - static_cast<double>(stored[i]);
        value += diff * diff;
      }
    } else {
      double dot = 0.0;
      for (int i = 0; i < dim_; ++i) {
        dot += static_cast<double>(vector[i]) * static_cast<double>(stored[i]);
      }
      value = 1.0 - dot;
    }
    return static_cast<float>(value);
  }

  [[nodiscard]] tl::expected<std::pair<int64_t, int64_t>, vsag::Error>
  GetMinAndMaxId() const override
  {
    if (external_ids_.empty()) {
      return tl::unexpected(vsag::Error(vsag::ErrorType::INDEX_EMPTY,
                                        "hipVS HNSW index has no ids"));
    }
    const auto bounds = std::minmax_element(external_ids_.begin(), external_ids_.end());
    return std::make_pair(*bounds.first, *bounds.second);
  }

  [[nodiscard]] uint64_t GetMemoryUsage() const override
  {
    const uint64_t vector_bytes = vectors_.capacity() * sizeof(float);
    const uint64_t graph_bytes =
        static_cast<uint64_t>(external_ids_.size()) * static_cast<uint64_t>(max_degree_) * 2U * sizeof(uint32_t);
    return vector_bytes + graph_bytes + external_ids_.capacity() * sizeof(int64_t) + serialized_bytes_;
  }

  [[nodiscard]] uint64_t EstimateMemory(uint64_t num_elements) const override
  {
    return num_elements * (static_cast<uint64_t>(dim_) * sizeof(float) +
                           static_cast<uint64_t>(max_degree_) * 2U * sizeof(uint32_t) +
                           sizeof(int64_t));
  }

  [[nodiscard]] uint64_t EstimateBuildMemory(uint64_t num_elements) const override
  {
    return EstimateMemory(num_elements) * 2U;
  }

private:
  tl::expected<vsag::DatasetPtr, vsag::Error> do_knn_search(
      const vsag::DatasetPtr &query,
      int64_t k,
      const std::string &parameters,
      vsag::Allocator *result_allocator,
      const std::function<bool(int64_t)> *valid_filter,
      float valid_ratio) const
  {
    if (!built_ || index_ == nullptr) {
      return tl::unexpected(vsag::Error(vsag::ErrorType::INDEX_EMPTY,
                                        "hipVS HNSW index is empty"));
    }
    if (query == nullptr || k <= 0 || query->GetDim() != dim_) {
      return tl::unexpected(vsag::Error(vsag::ErrorType::INVALID_ARGUMENT,
                                        "invalid hipVS HNSW query"));
    }
    const int64_t query_count = query->GetNumElements();
    if (query->GetFloat32Vectors() == nullptr) {
      return tl::unexpected(vsag::Error(vsag::ErrorType::INVALID_ARGUMENT,
                                        "hipVS HNSW query requires float32 vectors"));
    }
    const int64_t total = static_cast<int64_t>(external_ids_.size());
    const int64_t effective_k = std::min<int64_t>(k, total);
    if (query_count <= 0 || effective_k <= 0) {
      return tl::unexpected(vsag::Error(vsag::ErrorType::INVALID_ARGUMENT,
                                        "hipVS HNSW query is empty"));
    }
    int64_t candidate_k = effective_k;
    if (valid_filter != nullptr) {
      const float safe_ratio = std::max(valid_ratio, 0.001F);
      const int64_t ratio_candidates = static_cast<int64_t>(
          std::ceil(static_cast<double>(effective_k) / static_cast<double>(safe_ratio)));
      candidate_k = std::min<int64_t>(total, std::max(ratio_candidates, effective_k * 8));
    }
    if (query_count > std::numeric_limits<int64_t>::max() / candidate_k ||
        query_count > std::numeric_limits<int64_t>::max() / effective_k) {
      return tl::unexpected(vsag::Error(vsag::ErrorType::INVALID_ARGUMENT,
                                        "hipVS HNSW result dimensions overflow"));
    }
    const uint64_t candidate_count =
        static_cast<uint64_t>(query_count) * static_cast<uint64_t>(candidate_k);
    const uint64_t output_count =
        static_cast<uint64_t>(query_count) * static_cast<uint64_t>(effective_k);
    if (candidate_count > std::numeric_limits<size_t>::max() / sizeof(uint64_t) ||
        output_count > std::numeric_limits<size_t>::max() / sizeof(uint64_t)) {
      return tl::unexpected(vsag::Error(vsag::ErrorType::INVALID_ARGUMENT,
                                        "hipVS HNSW search result is too large"));
    }
    std::vector<uint64_t> neighbors(static_cast<size_t>(candidate_count));
    std::vector<float> distances(static_cast<size_t>(candidate_count));

    cuvsHnswSearchParams_t search_params = nullptr;
    if (cuvsHnswSearchParamsCreate(&search_params) != CUVS_SUCCESS || search_params == nullptr) {
      return tl::unexpected(make_hipvs_error("create hipVS HNSW search parameters"));
    }
    const int candidate_ef = candidate_k > std::numeric_limits<int>::max()
                                 ? std::numeric_limits<int>::max()
                                 : static_cast<int>(candidate_k);
    search_params->ef = std::max(parse_ef_search(parameters, ef_search_), candidate_ef);
    search_params->num_threads = 0;

    int64_t query_shape[2] = {query_count, dim_};
    int64_t output_shape[2] = {query_count, candidate_k};
    DLManagedTensor query_tensor{};
    DLManagedTensor neighbor_tensor{};
    DLManagedTensor distance_tensor{};
    init_host_float_tensor(query_tensor,
                           const_cast<float *>(query->GetFloat32Vectors()),
                           query_shape);
    init_host_uint64_tensor(neighbor_tensor, neighbors.data(), output_shape);
    init_host_float_tensor(distance_tensor, distances.data(), output_shape);
    const cuvsError_t result = cuvsHnswSearch(
        resources_, search_params, index_, &query_tensor, &neighbor_tensor, &distance_tensor);
    cuvsHnswSearchParamsDestroy(search_params);
    if (result != CUVS_SUCCESS) {
      return tl::unexpected(make_hipvs_error("search hipVS HNSW index"));
    }

    int64_t *result_ids = nullptr;
    float *result_distances = nullptr;
    if (result_allocator != nullptr) {
      result_ids = static_cast<int64_t *>(result_allocator->Allocate(output_count * sizeof(int64_t)));
      result_distances = static_cast<float *>(result_allocator->Allocate(output_count * sizeof(float)));
      if (result_ids == nullptr || result_distances == nullptr) {
        if (result_ids != nullptr) {
          result_allocator->Deallocate(result_ids);
        }
        if (result_distances != nullptr) {
          result_allocator->Deallocate(result_distances);
        }
        return tl::unexpected(vsag::Error(vsag::ErrorType::NO_ENOUGH_MEMORY,
                                          "failed to allocate hipVS HNSW search result"));
      }
    } else {
      try {
        result_ids = new int64_t[static_cast<size_t>(output_count)];
        result_distances = new float[static_cast<size_t>(output_count)];
      } catch (const std::bad_alloc &) {
        delete[] result_ids;
        delete[] result_distances;
        return tl::unexpected(vsag::Error(vsag::ErrorType::NO_ENOUGH_MEMORY,
                                          "failed to allocate hipVS HNSW search result"));
      }
    }

    for (int64_t query_idx = 0; query_idx < query_count; ++query_idx) {
      int64_t emitted = 0;
      const uint64_t candidate_base = static_cast<uint64_t>(query_idx) * static_cast<uint64_t>(candidate_k);
      const uint64_t output_base = static_cast<uint64_t>(query_idx) * static_cast<uint64_t>(effective_k);
      for (int64_t candidate_idx = 0;
           candidate_idx < candidate_k && emitted < effective_k;
           ++candidate_idx) {
        const uint64_t offset = candidate_base + static_cast<uint64_t>(candidate_idx);
        const uint64_t inner_id = neighbors[static_cast<size_t>(offset)];
        if (inner_id >= external_ids_.size()) {
          continue;
        }
        const int64_t external_id = external_ids_[static_cast<size_t>(inner_id)];
        if (valid_filter != nullptr && !(*valid_filter)(external_id)) {
          continue;
        }
        result_ids[output_base + static_cast<uint64_t>(emitted)] = external_id;
        result_distances[output_base + static_cast<uint64_t>(emitted)] = distances[static_cast<size_t>(offset)];
        ++emitted;
      }
      for (; emitted < effective_k; ++emitted) {
        result_ids[output_base + static_cast<uint64_t>(emitted)] = -1;
        result_distances[output_base + static_cast<uint64_t>(emitted)] =
            std::numeric_limits<float>::infinity();
      }
    }

    vsag::DatasetPtr result_dataset = vsag::Dataset::Make();
    result_dataset->NumElements(query_count)
        ->Dim(effective_k)
        ->Ids(result_ids)
        ->Distances(result_distances)
        ->Owner(true, result_allocator);
    return result_dataset;
  }

  int dim_{0};
  cuvsDistanceType metric_{L2Expanded};
  int max_degree_{16};
  int ef_construction_{200};
  int ef_search_{200};
  vsag::Allocator *allocator_{nullptr};
  cuvsResources_t resources_{0};
  cuvsHnswIndex_t index_{nullptr};
  std::vector<int64_t> external_ids_;
  std::vector<float> vectors_;
  std::unordered_map<int64_t, size_t> row_by_id_;
  bool available_{false};
  bool built_{false};
  bool use_gpu_hierarchy_{false};
  uint64_t serialized_bytes_{0};
  mutable std::mutex mutex_;
};

std::shared_ptr<vsag::Index> create_hipvs_hnsw_if_supported(
    int dim,
    const char *dtype,
    const char *metric,
    int max_degree,
    int ef_construction,
    int ef_search,
    vsag::Allocator *allocator,
    int extra_info_size)
{
  if (dim <= 0 || env_disables_hipvs() || extra_info_size != 0 || !hipvs_float_dtype(dtype)) {
    return nullptr;
  }
  cuvsDistanceType hipvs_metric = L2Expanded;
  if (!hipvs_metric(metric, hipvs_metric)) {
    return nullptr;
  }
  auto index = std::make_shared<HipVSHnswIndex>(
      dim, hipvs_metric, max_degree, ef_construction, ef_search, allocator);
  if (!index->available()) {
    LOG_WARN("[OBVSAG][HIPVS] backend unavailable, falling back to VSAG");
    return nullptr;
  }
  return index;
}

}  // namespace

#endif  // SEEKDB_WITH_HIPVS

static constexpr int64_t VSAG_SINDI_N_CANDIDATE_FACTOR = 500;

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

class HnswIndexHandler {
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

int HnswIndexHandler::build_index(const vsag::DatasetPtr &base)
{
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

int create_index(VectorIndexPtr &index_handler,
                 IndexType index_type, const char *dtype,
                 const char *metric, int dim, int max_degree,
                 int ef_construction, int ef_search, void *allocator,
                 int extra_info_size /* = 0*/, int16_t refine_type /*= 0*/,
                 int16_t bq_bits_query /*= 32*/, bool bq_use_fht /*= false*/)
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
  
    adjust_create_index_max_degree(index_type, max_degree);

#ifdef SEEKDB_WITH_HIPVS
    if (index_type == HNSW_TYPE) {
      std::shared_ptr<vsag::Index> hipvs_index = create_hipvs_hnsw_if_supported(
          dim, dtype, metric, max_degree, ef_construction, ef_search,
          vsag_allocator, extra_info_size);
      if (hipvs_index != nullptr) {
        HnswIndexHandler *hnsw_index = new HnswIndexHandler(
            true, false, false, dtype, metric, max_degree, ef_construction,
            ef_search, dim, index_type, hipvs_index, vsag_allocator, extra_info_size,
            refine_type, bq_bits_query, bq_use_fht);
        if (OB_ISNULL(hnsw_index)) {
          ret = OB_ALLOCATE_MEMORY_FAILED;
        } else {
          index_handler = static_cast<VectorIndexPtr>(hnsw_index);
          LOG_INFO("[OBVSAG][HIPVS] selected hipVS HNSW backend", K(dim), K(max_degree));
        }
        return ret;
      }
    }
#endif

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
    HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
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
    HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
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
  HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
  return hnsw->get_index_type();
}

int get_index_number(VectorIndexPtr &index_handler, int64_t &size)
{
  int ret = OB_SUCCESS;
  if (index_handler == nullptr) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
    size = hnsw->get_index_number();
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
    HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
    if (OB_FAIL(hnsw->cal_distance_by_id(vector, ids, count, distances))) {
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
    HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
    if (OB_FAIL(hnsw->cal_distance_by_id(len, dims, vals, ids, count, distances))) {
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
    HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
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
    HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
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
    HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
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
    FilterInterface *bitmap = static_cast<FilterInterface *>(invalid);
    HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
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
  return ret;
}

int fserialize(VectorIndexPtr &index_handler, std::ostream &out_stream)
{
  int ret = OB_SUCCESS;
  if (index_handler == nullptr) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
    tl::expected<void, Error> bs = hnsw->get_index()->Serialize(out_stream);
    if (bs.has_value()) {
      LOG_INFO("[OBVSAG] serialize index success");
    } else {
      ret = vsag_errcode2ob(bs.error().type);
      LOG_WARN("[OBVSAG] fserialize error happend", K(ret), K(bs.error().type));
    }
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
    HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
    std::shared_ptr<vsag::Index> hnsw_index;
    bool use_static = hnsw->get_use_static();
    const char *metric = hnsw->get_metric();
    const char *dtype = hnsw->get_dtype();
    int max_degree = hnsw->get_max_degree();
    int ef_construction = hnsw->get_ef_construction();
    int ef_search = hnsw->get_ef_search();
    int dim = hnsw->get_dim();
    int index_type = hnsw->get_index_type();
    uint64_t extra_info_size = hnsw->get_extra_info_size();
    const char* index_type_str = get_index_type_str(index_type);
    int16_t refine_type = hnsw->get_refine_type();
    int16_t bq_bits_query = hnsw->get_bq_bits_query();
    bool bq_use_fht = hnsw->get_bq_use_fht();
    bool use_reorder = hnsw->get_use_reorder();
    float doc_prune_ratio = hnsw->get_doc_prune_ratio();
    int window_size = hnsw->get_window_size();

    char result_param_str[1024] = {0};
    if ((IndexType)index_type == IndexType::IPIVF_TYPE) {
      if (OB_FAIL(construct_vsag_sindi_create_param(uint8_t(index_type),
              dtype,
              metric,
              hnsw->get_allocator(),
              extra_info_size,
              use_reorder,
              doc_prune_ratio,
              window_size,
              result_param_str))) {
      }
    } else {
      if (OB_FAIL(construct_vsag_create_param(
        uint8_t(index_type), dtype, metric, dim, max_degree,
        ef_construction, ef_search, hnsw->get_allocator(),
        extra_info_size, refine_type, bq_bits_query, bq_use_fht, result_param_str))) {
      } 
    }
    if (OB_FAIL(ret)) {
    } else {
      const std::string input_json_str(result_param_str);
#ifdef SEEKDB_WITH_HIPVS
      if (index_type == HNSW_TYPE) {
        hnsw_index = create_hipvs_hnsw_if_supported(
            dim, dtype, metric, max_degree, ef_construction, ef_search,
            hnsw->get_allocator(), static_cast<int>(extra_info_size));
      }
#endif
      if (hnsw_index == nullptr) {
        tl::expected<std::shared_ptr<Index>, Error> index =
            vsag::Factory::CreateIndex(index_type_str, input_json_str, hnsw->get_allocator());
        if (index.has_value()) {
          hnsw_index = index.value();
        } else {
          ret = vsag_errcode2ob(index.error().type);
          LOG_WARN("[OBVSAG] create index error", K(ret), K(index.error().type));
        }
      }
      if (OB_SUCC(ret)) {
        tl::expected<void, Error> bs = hnsw_index->Deserialize(in_stream);
        if (bs.has_value()) {
          hnsw->set_index(hnsw_index);
          LOG_INFO("[OBVSAG] fdeserialize success", KCSTRING(result_param_str));
        } else {
          ret = vsag_errcode2ob(bs.error().type);
          LOG_WARN("[OBVSAG] fdeserialize error", K(ret), K(bs.error().type));
        }
      }
    }
  }
  return ret;
}

int delete_index(VectorIndexPtr &index_handler)
{
  int ret = OB_SUCCESS;
  LOG_INFO("[OBVSAG] delete index ",
      KP((void *)static_cast<HnswIndexHandler *>(index_handler)->get_index().get()),
      K(static_cast<HnswIndexHandler *>(index_handler)->get_index().use_count()), K(lbt()));
  if (index_handler != nullptr) {
    delete static_cast<HnswIndexHandler *>(index_handler);
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
    HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
    if (OB_FAIL(hnsw->get_extra_info_by_ids(ids, count, extra_infos))) {
    }
  }
  return ret;
}

uint64_t estimate_memory(VectorIndexPtr &index_handler, const uint64_t row_count, const bool is_build) 
{
  uint64_t estimate_memory_size = 0;
  if (index_handler != nullptr) {
    HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
    estimate_memory_size = hnsw->estimate_memory(row_count, is_build);
  }
  return estimate_memory_size;
}

int immutable_optimize(VectorIndexPtr& index_handler)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(index_handler)) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
    if (OB_FAIL(hnsw->immutable_optimize())) {
    }
  }
  return ret;
}

} // namespace obvsag
} // namespace common
} // namespace oceanbase
