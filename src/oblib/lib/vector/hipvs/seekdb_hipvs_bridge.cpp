/*
 * Copyright (c) 2026 OceanBase.
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

/**
 * @file seekdb_hipvs_bridge.cpp
 *
 * AMD hipVS backed implementation of the seekdb <-> GPU bridge ABI declared in
 * `seekdb_hipvs_bridge.h`.
 *
 * This translation unit is only compiled when a ROCm toolchain and the hipVS
 * headers are available; see `cmake/hipvs.cmake`. The seekdb binary itself
 * never links against hipVS, it loads the resulting shared library at runtime.
 *
 * The HNSW build performed here is the canonical hipVS one:
 *
 *   cuvsCagraBuild()     -> build the k-NN graph on the AMD GPU
 *   cuvsHnswFromCagra()  -> turn that graph into a real HNSW index
 *
 * The second step produces an hnswlib index, so the resulting index keeps the
 * semantics of the CPU implementation and can still be extended with
 * `cuvsHnswExtend()`.
 */

#include "lib/vector/hipvs/seekdb_hipvs_bridge.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance.h>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/hnsw.h>
#include <dlpack/dlpack.h>

namespace {

/* The seekdb specific part of the serialized payload. The hnswlib blob that
 * follows it carries the graph and the vectors. */
const char SEEKDB_HIPVS_PAYLOAD_MAGIC[8] = {'S', 'K', 'H', 'I', 'P', 'V', 'S', '1'};
const int64_t SEEKDB_HIPVS_PAYLOAD_HEADER = 24;

/* The bridge owns every hipVS resource it hands out. */
struct BridgeResources {
  cuvsResources_t res{0};
  bool res_valid{false};
};

struct BridgeIndex {
  /* Parameters the index was created with; needed to deserialize it again. */
  seekdb_hipvs_index_params_t params{};

  /* The hnswlib index that answers the queries. */
  cuvsHnswIndex_t hnsw{nullptr};

  /* Kept alive between the build and the conversion to HNSW. */
  cuvsCagraIndex_t cagra{nullptr};

  /* The external ids, indexed by the hnswlib ordinal of the element. */
  std::vector<int64_t> ids;

  /* Number of vectors currently stored. */
  int64_t num_elements{0};
};

inline void copy_string(char *buf, int32_t buf_size, const std::string &value)
{
  if (buf == nullptr || buf_size <= 0) {
    return;
  }
  const int32_t len = static_cast<int32_t>(value.size()) < buf_size - 1 ? static_cast<int32_t>(value.size())
                                                                       : buf_size - 1;
  memcpy(buf, value.c_str(), len);
  buf[len] = '\0';
}

inline cuvsDistanceType to_cuvs_metric(int32_t metric)
{
  switch (metric) {
    case SEEKDB_HIPVS_METRIC_IP:
      return InnerProduct;
    case SEEKDB_HIPVS_METRIC_COSINE:
      return CosineExpanded;
    case SEEKDB_HIPVS_METRIC_L2:
    default:
      return L2Expanded;
  }
}

inline cuvsHnswHierarchy to_cuvs_hierarchy(int32_t hierarchy)
{
  switch (hierarchy) {
    case SEEKDB_HIPVS_HIERARCHY_NONE:
      return cuvsHnswHierarchy::NONE;
    case SEEKDB_HIPVS_HIERARCHY_GPU:
      return cuvsHnswHierarchy::GPU;
    case SEEKDB_HIPVS_HIERARCHY_CPU:
    default:
      return cuvsHnswHierarchy::CPU;
  }
}

/**
 * Build a host resident dlpack tensor over @p data.
 *
 * The tensor does not own the memory: every hipVS call that consumes it is
 * synchronous with respect to the host buffer, and the caller keeps the buffer
 * alive for the duration of the call.
 */
class HostTensor {
public:
  HostTensor() = default;
  HostTensor(const HostTensor &) = delete;
  HostTensor &operator=(const HostTensor &) = delete;
  ~HostTensor() { reset(); }

  void init(void *data, int64_t rows, int64_t cols, DLDataType dtype)
  {
    reset();
    shape_[0] = rows;
    shape_[1] = cols;
    strides_[0] = cols;
    strides_[1] = 1;
    tensor_.dl_tensor.device.device_type = kDLCPU;
    tensor_.dl_tensor.device.device_id = 0;
    tensor_.dl_tensor.ndim = 2;
    tensor_.dl_tensor.dtype = dtype;
    tensor_.dl_tensor.shape = shape_;
    tensor_.dl_tensor.strides = strides_;
    tensor_.dl_tensor.byte_offset = 0;
    tensor_.dl_tensor.data = data;
    tensor_.manager_ctx = nullptr;
    tensor_.deleter = nullptr;
  }

  void reset()
  {
    tensor_.dl_tensor.shape = nullptr;
    tensor_.dl_tensor.strides = nullptr;
    tensor_.dl_tensor.data = nullptr;
  }

  DLManagedTensor *get() { return &tensor_; }

private:
  DLManagedTensor tensor_{};
  int64_t shape_[2]{0, 0};
  int64_t strides_[2]{0, 0};
};

inline DLDataType float_dtype() { return DLDataType{kDLFloat, 32, 1}; }
/* cuvsHnswSearch documents the neighbours output as unsigned 64 bit. The
 * layout is identical to int64, the bridge just reports the documented code. */
inline DLDataType neighbor_dtype() { return DLDataType{kDLUInt, 64, 1}; }

/**
 * Create the CAGRA index parameters derived from the bridge parameters.
 */
int32_t create_cagra_params(const seekdb_hipvs_index_params_t &params,
                            cuvsCagraIndexParams_t *out)
{
  if (out == nullptr) {
    return SEEKDB_HIPVS_ERR_INVALID_ARGUMENT;
  }
  cuvsCagraIndexParams_t cagra_params = nullptr;
  if (cuvsCagraIndexParamsCreate(&cagra_params) != CUVS_SUCCESS) {
    return SEEKDB_HIPVS_ERR_HIPVS;
  }
  cagra_params->metric = to_cuvs_metric(params.metric);
  /* The output graph degree is the HNSW degree. */
  cagra_params->graph_degree = params.m > 0 ? static_cast<size_t>(params.m) : 32;
  /* The intermediate graph is deliberately a bit wider: it is pruned down to
   * `graph_degree` afterwards, and a wider intermediate graph gives the GPU
   * more material to pick good edges from. */
  cagra_params->intermediate_graph_degree = params.intermediate_degree > 0
                                                ? static_cast<size_t>(params.intermediate_degree)
                                                : cagra_params->graph_degree * 2;
  if (params.graph_algo == SEEKDB_HIPVS_GRAPH_ALGO_NN_DESCENT) {
    cagra_params->build_algo = cuvsCagraGraphBuildAlgo::NN_DESCENT;
    cagra_params->nn_descent_niter = params.nn_descent_iters > 0 ? static_cast<size_t>(params.nn_descent_iters) : 20;
  } else {
    cagra_params->build_algo = cuvsCagraGraphBuildAlgo::AUTO_SELECT;
  }
  *out = cagra_params;
  return SEEKDB_HIPVS_OK;
}

/**
 * Convert a CAGRA index into an hnswlib index.
 */
int32_t convert_to_hnsw(BridgeResources *resources,
                         const seekdb_hipvs_index_params_t &params,
                         cuvsCagraIndex_t cagra,
                         cuvsHnswIndex_t *out)
{
  cuvsHnswIndexParams_t hnsw_params = nullptr;
  if (cuvsHnswIndexParamsCreate(&hnsw_params) != CUVS_SUCCESS) {
    return SEEKDB_HIPVS_ERR_HIPVS;
  }
  hnsw_params->hierarchy = to_cuvs_hierarchy(params.hierarchy);
  hnsw_params->ef_construction = params.ef_construction > 0 ? params.ef_construction : 200;
  hnsw_params->num_threads = 0;

  cuvsHnswIndex_t hnsw = nullptr;
  if (cuvsHnswIndexCreate(&hnsw) != CUVS_SUCCESS) {
    cuvsHnswIndexParamsDestroy(hnsw_params);
    return SEEKDB_HIPVS_ERR_HIPVS;
  }
  const cuvsError_t status = cuvsHnswFromCagra(resources->res, hnsw_params, cagra, hnsw);
  cuvsHnswIndexParamsDestroy(hnsw_params);
  if (status != CUVS_SUCCESS) {
    cuvsHnswIndexDestroy(hnsw);
    return SEEKDB_HIPVS_ERR_HIPVS;
  }
  *out = hnsw;
  return SEEKDB_HIPVS_OK;
}

/**
 * Record the external ids of the vectors that were just added.
 */
void remember_ids(BridgeIndex *index, int64_t count, const int64_t *ids)
{
  for (int64_t i = 0; i < count; ++i) {
    index->ids.push_back(ids == nullptr ? index->num_elements + i : ids[i]);
  }
  index->num_elements += count;
}

/**
 * Read a whole file into @p out. Used to turn the hnswlib file based
 * serialization into a seekdb owned byte buffer.
 */
int32_t read_file(const std::string &path, std::string *out)
{
  FILE *file = fopen(path.c_str(), "rb");
  if (file == nullptr) {
    return SEEKDB_HIPVS_ERR_IO;
  }
  char buffer[65536];
  out->clear();
  while (true) {
    const size_t read = fread(buffer, 1, sizeof(buffer), file);
    if (read > 0) {
      out->append(buffer, read);
    }
    if (read < sizeof(buffer)) {
      break;
    }
  }
  const bool failed = ferror(file) != 0;
  fclose(file);
  return failed ? SEEKDB_HIPVS_ERR_IO : SEEKDB_HIPVS_OK;
}

int32_t write_file(const std::string &path, const char *data, int64_t len)
{
  FILE *file = fopen(path.c_str(), "wb");
  if (file == nullptr) {
    return SEEKDB_HIPVS_ERR_IO;
  }
  int32_t ret = SEEKDB_HIPVS_OK;
  if (len > 0 && fwrite(data, 1, static_cast<size_t>(len), file) != static_cast<size_t>(len)) {
    ret = SEEKDB_HIPVS_ERR_IO;
  }
  if (fclose(file) != 0) {
    ret = SEEKDB_HIPVS_ERR_IO;
  }
  return ret;
}

/**
 * Build a unique temporary path for the hnswlib serialization round trip.
 */
std::string make_temp_path(const char *suffix)
{
  static std::atomic<int64_t> counter(0);
  const char *tmp_dir = getenv("TMPDIR");
  char buffer[512];
  snprintf(buffer, sizeof(buffer), "%s/seekdb_hipvs_%d_%ld_%ld_%s",
           tmp_dir == nullptr ? "/tmp" : tmp_dir,
           static_cast<int>(getpid()),
           static_cast<long>(counter.fetch_add(1)),
           static_cast<long>(time(nullptr)),
           suffix);
  return std::string(buffer);
}

} // namespace

extern "C" {

int32_t seekdb_hipvs_abi_version(void) { return SEEKDB_HIPVS_BRIDGE_ABI_VERSION; }

void seekdb_hipvs_version(char *buf, int32_t buf_size)
{
  uint16_t major = 0;
  uint16_t minor = 0;
  uint16_t patch = 0;
  if (cuvsVersionGet(&major, &minor, &patch) != CUVS_SUCCESS) {
    copy_string(buf, buf_size, "unknown");
    return;
  }
  char version[64];
  snprintf(version, sizeof(version), "%d.%d.%d", static_cast<int>(major), static_cast<int>(minor),
           static_cast<int>(patch));
  copy_string(buf, buf_size, version);
}

int32_t seekdb_hipvs_gpu_info(char *buf, int32_t buf_size)
{
  int device_id = 0;
  cuvsResources_t res = nullptr;
  if (cuvsResourcesCreate(&res) != CUVS_SUCCESS) {
    copy_string(buf, buf_size, "no gpu");
    return SEEKDB_HIPVS_ERR_NO_GPU;
  }
  const cuvsError_t status = cuvsDeviceIdGet(res, &device_id);
  cuvsResourcesDestroy(res);
  if (status != CUVS_SUCCESS) {
    copy_string(buf, buf_size, "no gpu");
    return SEEKDB_HIPVS_ERR_NO_GPU;
  }
  char info[256];
  snprintf(info, sizeof(info), "device %d", device_id);
  copy_string(buf, buf_size, info);
  return SEEKDB_HIPVS_OK;
}

int32_t seekdb_hipvs_resources_create(void **res)
{
  if (res == nullptr) {
    return SEEKDB_HIPVS_ERR_INVALID_ARGUMENT;
  }
  BridgeResources *resources = new (std::nothrow) BridgeResources();
  if (resources == nullptr) {
    return SEEKDB_HIPVS_ERR_NO_MEMORY;
  }
  if (cuvsResourcesCreate(&resources->res) != CUVS_SUCCESS) {
    delete resources;
    return SEEKDB_HIPVS_ERR_NO_GPU;
  }
  resources->res_valid = true;
  *res = resources;
  return SEEKDB_HIPVS_OK;
}

int32_t seekdb_hipvs_resources_destroy(void *res)
{
  if (res == nullptr) {
    return SEEKDB_HIPVS_OK;
  }
  BridgeResources *resources = static_cast<BridgeResources *>(res);
  if (resources->res_valid && resources->res != nullptr) {
    cuvsResourcesDestroy(resources->res);
  }
  delete resources;
  return SEEKDB_HIPVS_OK;
}

int32_t seekdb_hipvs_index_create(const seekdb_hipvs_index_params_t *params, void **idx)
{
  if (params == nullptr || idx == nullptr) {
    return SEEKDB_HIPVS_ERR_INVALID_ARGUMENT;
  }
  if (params->dim <= 0 || params->m <= 0) {
    return SEEKDB_HIPVS_ERR_INVALID_ARGUMENT;
  }
  BridgeIndex *index = new (std::nothrow) BridgeIndex();
  if (index == nullptr) {
    return SEEKDB_HIPVS_ERR_NO_MEMORY;
  }
  index->params = *params;
  *idx = index;
  return SEEKDB_HIPVS_OK;
}

int32_t seekdb_hipvs_index_destroy(void *idx)
{
  if (idx == nullptr) {
    return SEEKDB_HIPVS_OK;
  }
  BridgeIndex *index = static_cast<BridgeIndex *>(idx);
  if (index->hnsw != nullptr) {
    cuvsHnswIndexDestroy(index->hnsw);
    index->hnsw = nullptr;
  }
  if (index->cagra != nullptr) {
    cuvsCagraIndexDestroy(index->cagra);
    index->cagra = nullptr;
  }
  delete index;
  return SEEKDB_HIPVS_OK;
}

int32_t seekdb_hipvs_index_build(void *idx, const float *vectors, int64_t dim, int64_t count,
                                  const int64_t *ids)
{
  if (idx == nullptr || vectors == nullptr || dim <= 0 || count <= 0) {
    return SEEKDB_HIPVS_ERR_INVALID_ARGUMENT;
  }
  BridgeIndex *index = static_cast<BridgeIndex *>(idx);
  if (index->hnsw != nullptr) {
    /* The index was already built, this is an incremental add. */
    return seekdb_hipvs_index_add(idx, vectors, dim, count, ids);
  }

  void *raw_resources = nullptr;
  if (seekdb_hipvs_resources_create(&raw_resources) != SEEKDB_HIPVS_OK) {
    return SEEKDB_HIPVS_ERR_NO_GPU;
  }
  BridgeResources *resources = static_cast<BridgeResources *>(raw_resources);

  int32_t ret = SEEKDB_HIPVS_OK;
  HostTensor dataset_tensor;
  dataset_tensor.init(const_cast<float *>(vectors), count, dim, float_dtype());

  cuvsCagraIndexParams_t cagra_params = nullptr;
  cuvsCagraIndex_t cagra = nullptr;
  if (create_cagra_params(index->params, &cagra_params) != SEEKDB_HIPVS_OK) {
    ret = SEEKDB_HIPVS_ERR_HIPVS;
  } else if (cuvsCagraIndexCreate(&cagra) != CUVS_SUCCESS) {
    ret = SEEKDB_HIPVS_ERR_HIPVS;
  } else if (cuvsCagraBuild(resources->res, cagra_params, dataset_tensor.get(), cagra) != CUVS_SUCCESS) {
    ret = SEEKDB_HIPVS_ERR_HIPVS;
  } else {
    /* The graph is built on the GPU. Converting it into an hnswlib index is
     * the cheap part. */
    cuvsHnswIndex_t hnsw = nullptr;
    ret = convert_to_hnsw(resources, index->params, cagra, &hnsw);
    if (ret == SEEKDB_HIPVS_OK) {
      index->hnsw = hnsw;
      remember_ids(index, count, ids);
      /* The CAGRA index is only needed for the conversion. Keeping it alive
       * would hold the whole dataset and the k-NN graph in device memory for
       * the lifetime of the index. */
      cuvsCagraIndexDestroy(cagra);
      cagra = nullptr;
    }
  }

  if (cagra_params != nullptr) {
    cuvsCagraIndexParamsDestroy(cagra_params);
  }
  if (cagra != nullptr) {
    cuvsCagraIndexDestroy(cagra);
  }
  seekdb_hipvs_resources_destroy(raw_resources);
  return ret;
}

int32_t seekdb_hipvs_index_add(void *idx, const float *vectors, int64_t dim, int64_t count,
                                const int64_t *ids)
{
  if (idx == nullptr || vectors == nullptr || dim <= 0 || count <= 0) {
    return SEEKDB_HIPVS_ERR_INVALID_ARGUMENT;
  }
  BridgeIndex *index = static_cast<BridgeIndex *>(idx);
  if (index->hnsw == nullptr) {
    /* The index has not been built yet, so this batch is the first one: the
     * whole graph is built on the GPU. */
    return seekdb_hipvs_index_build(idx, vectors, dim, count, ids);
  }
  if (to_cuvs_hierarchy(index->params.hierarchy) != cuvsHnswHierarchy::CPU) {
    /* cuvsHnswExtend only works on a CPU hierarchy index. */
    return SEEKDB_HIPVS_ERR_UNSUPPORTED;
  }

  void *raw_resources = nullptr;
  if (seekdb_hipvs_resources_create(&raw_resources) != SEEKDB_HIPVS_OK) {
    return SEEKDB_HIPVS_ERR_NO_GPU;
  }
  BridgeResources *resources = static_cast<BridgeResources *>(raw_resources);

  int32_t ret = SEEKDB_HIPVS_OK;
  HostTensor dataset_tensor;
  dataset_tensor.init(const_cast<float *>(vectors), count, dim, float_dtype());

  cuvsHnswExtendParams_t extend_params = nullptr;
  if (cuvsHnswExtendParamsCreate(&extend_params) != CUVS_SUCCESS) {
    ret = SEEKDB_HIPVS_ERR_HIPVS;
  } else if (cuvsHnswExtend(resources->res, extend_params, dataset_tensor.get(), index->hnsw) !=
             CUVS_SUCCESS) {
    ret = SEEKDB_HIPVS_ERR_HIPVS;
  } else {
    remember_ids(index, count, ids);
  }
  if (extend_params != nullptr) {
    cuvsHnswExtendParamsDestroy(extend_params);
  }
  seekdb_hipvs_resources_destroy(raw_resources);
  return ret;
}

int32_t seekdb_hipvs_index_search(void *idx, const float *query, int64_t dim, int64_t topk, int32_t ef,
                                  int64_t *ids, float *distances, int64_t *result_size)
{
  if (idx == nullptr || query == nullptr || dim <= 0 || topk <= 0 || ids == nullptr || distances == nullptr) {
    return SEEKDB_HIPVS_ERR_INVALID_ARGUMENT;
  }
  BridgeIndex *index = static_cast<BridgeIndex *>(idx);
  if (index->hnsw == nullptr) {
    return SEEKDB_HIPVS_ERR_UNSUPPORTED;
  }

  void *raw_resources = nullptr;
  if (seekdb_hipvs_resources_create(&raw_resources) != SEEKDB_HIPVS_OK) {
    return SEEKDB_HIPVS_ERR_NO_GPU;
  }
  BridgeResources *resources = static_cast<BridgeResources *>(raw_resources);

  int32_t ret = SEEKDB_HIPVS_OK;
  int64_t written = 0;

  /* The hnswlib index answers with the ordinal of the element, the caller
   * expects the external id. Both are written into the same buffer: the
   * ordinals first, then they are translated in place. */
  HostTensor query_tensor;
  HostTensor neighbor_tensor;
  HostTensor distance_tensor;
  query_tensor.init(const_cast<float *>(query), 1, dim, float_dtype());
  neighbor_tensor.init(ids, 1, topk, neighbor_dtype());
  distance_tensor.init(distances, 1, topk, float_dtype());

  cuvsHnswSearchParams_t search_params = nullptr;
  if (cuvsHnswSearchParamsCreate(&search_params) != CUVS_SUCCESS) {
    ret = SEEKDB_HIPVS_ERR_HIPVS;
  } else {
    if (ef > 0) {
      search_params->ef = ef;
    }
    const cuvsError_t status = cuvsHnswSearch(resources->res, search_params, index->hnsw,
                                               query_tensor.get(), neighbor_tensor.get(),
                                               distance_tensor.get());
    if (status != CUVS_SUCCESS) {
      ret = SEEKDB_HIPVS_ERR_HIPVS;
    } else {
      written = topk;
      for (int64_t i = 0; i < written; ++i) {
        const int64_t ordinal = ids[i];
        if (ordinal < 0 || ordinal >= static_cast<int64_t>(index->ids.size())) {
          ids[i] = -1;
        } else {
          ids[i] = index->ids[static_cast<size_t>(ordinal)];
        }
      }
    }
    cuvsHnswSearchParamsDestroy(search_params);
  }

  seekdb_hipvs_resources_destroy(raw_resources);
  if (result_size != nullptr) {
    *result_size = written;
  }
  return ret;
}

int32_t seekdb_hipvs_index_num_elements(void *idx, int64_t *num_elements)
{
  if (idx == nullptr || num_elements == nullptr) {
    return SEEKDB_HIPVS_ERR_INVALID_ARGUMENT;
  }
  BridgeIndex *index = static_cast<BridgeIndex *>(idx);
  *num_elements = index->num_elements;
  return SEEKDB_HIPVS_OK;
}

int32_t seekdb_hipvs_index_vid_bound(void *idx, int64_t *min_vid, int64_t *max_vid)
{
  if (idx == nullptr || min_vid == nullptr || max_vid == nullptr) {
    return SEEKDB_HIPVS_ERR_INVALID_ARGUMENT;
  }
  BridgeIndex *index = static_cast<BridgeIndex *>(idx);
  if (index->ids.empty()) {
    *min_vid = 0;
    *max_vid = 0;
    return SEEKDB_HIPVS_OK;
  }
  int64_t low = index->ids[0];
  int64_t high = index->ids[0];
  for (size_t i = 1; i < index->ids.size(); ++i) {
    if (index->ids[i] < low) {
      low = index->ids[i];
    }
    if (index->ids[i] > high) {
      high = index->ids[i];
    }
  }
  *min_vid = low;
  *max_vid = high;
  return SEEKDB_HIPVS_OK;
}

int32_t seekdb_hipvs_index_serialize(void *idx, char **buf, int64_t *len)
{
  if (idx == nullptr || buf == nullptr || len == nullptr) {
    return SEEKDB_HIPVS_ERR_INVALID_ARGUMENT;
  }
  BridgeIndex *index = static_cast<BridgeIndex *>(idx);
  if (index->hnsw == nullptr) {
    return SEEKDB_HIPVS_ERR_UNSUPPORTED;
  }

  void *raw_resources = nullptr;
  if (seekdb_hipvs_resources_create(&raw_resources) != SEEKDB_HIPVS_OK) {
    return SEEKDB_HIPVS_ERR_NO_GPU;
  }
  BridgeResources *resources = static_cast<BridgeResources *>(raw_resources);

  int32_t ret = SEEKDB_HIPVS_OK;
  std::string hnsw_payload;
  const std::string path = make_temp_path("hnsw");
  if (cuvsHnswSerialize(resources->res, path.c_str(), index->hnsw) != CUVS_SUCCESS) {
    ret = SEEKDB_HIPVS_ERR_HIPVS;
  } else if (read_file(path, &hnsw_payload) != SEEKDB_HIPVS_OK) {
    ret = SEEKDB_HIPVS_ERR_IO;
  }
  remove(path.c_str());
  seekdb_hipvs_resources_destroy(raw_resources);

  if (ret != SEEKDB_HIPVS_OK) {
    return ret;
  }

  /* Layout: magic(8) | id count(8) | dim(4) | flags(4) | ids | hnswlib blob */
  const int64_t id_bytes = static_cast<int64_t>(index->ids.size()) * static_cast<int64_t>(sizeof(int64_t));
  const int64_t total = SEEKDB_HIPVS_PAYLOAD_HEADER + id_bytes +
                        static_cast<int64_t>(hnsw_payload.size());
  char *out = static_cast<char *>(malloc(static_cast<size_t>(total)));
  if (out == nullptr) {
    return SEEKDB_HIPVS_ERR_NO_MEMORY;
  }
  memcpy(out, SEEKDB_HIPVS_PAYLOAD_MAGIC, sizeof(SEEKDB_HIPVS_PAYLOAD_MAGIC));
  const int64_t id_count = static_cast<int64_t>(index->ids.size());
  memcpy(out + 8, &id_count, sizeof(id_count));
  const int32_t dim = index->params.dim;
  memcpy(out + 16, &dim, sizeof(dim));
  const int32_t flags = 0;
  memcpy(out + 20, &flags, sizeof(flags));
  if (id_bytes > 0) {
    memcpy(out + SEEKDB_HIPVS_PAYLOAD_HEADER, index->ids.data(), static_cast<size_t>(id_bytes));
  }
  memcpy(out + SEEKDB_HIPVS_PAYLOAD_HEADER + id_bytes, hnsw_payload.data(), hnsw_payload.size());
  *buf = out;
  *len = total;
  return SEEKDB_HIPVS_OK;
}

int32_t seekdb_hipvs_index_deserialize(const char *buf, int64_t len, const seekdb_hipvs_index_params_t *params,
                                        void **idx)
{
  if (buf == nullptr || len < SEEKDB_HIPVS_PAYLOAD_HEADER || params == nullptr || idx == nullptr) {
    return SEEKDB_HIPVS_ERR_INVALID_ARGUMENT;
  }
  if (memcmp(buf, SEEKDB_HIPVS_PAYLOAD_MAGIC, sizeof(SEEKDB_HIPVS_PAYLOAD_MAGIC)) != 0) {
    return SEEKDB_HIPVS_ERR_IO;
  }
  int64_t id_count = 0;
  int32_t dim = 0;
  int32_t flags = 0;
  memcpy(&id_count, buf + 8, sizeof(id_count));
  memcpy(&dim, buf + 16, sizeof(dim));
  memcpy(&flags, buf + 20, sizeof(flags));
  (void)flags;
  if (id_count < 0 || id_count > (len - SEEKDB_HIPVS_PAYLOAD_HEADER) / static_cast<int64_t>(sizeof(int64_t))) {
    /* The payload is corrupt, the id array does not fit into it. */
    return SEEKDB_HIPVS_ERR_IO;
  }
  const int64_t id_bytes = id_count * static_cast<int64_t>(sizeof(int64_t));
  const int64_t hnsw_len = len - SEEKDB_HIPVS_PAYLOAD_HEADER - id_bytes;
  if (hnsw_len <= 0) {
    return SEEKDB_HIPVS_ERR_IO;
  }

  void *raw_resources = nullptr;
  if (seekdb_hipvs_resources_create(&raw_resources) != SEEKDB_HIPVS_OK) {
    return SEEKDB_HIPVS_ERR_NO_GPU;
  }
  BridgeResources *resources = static_cast<BridgeResources *>(raw_resources);

  int32_t ret = SEEKDB_HIPVS_OK;
  const std::string path = make_temp_path("hnsw");
  if (write_file(path, buf + SEEKDB_HIPVS_PAYLOAD_HEADER + id_bytes, hnsw_len) != SEEKDB_HIPVS_OK) {
    ret = SEEKDB_HIPVS_ERR_IO;
  }

  BridgeIndex *index = nullptr;
  if (ret == SEEKDB_HIPVS_OK) {
    index = new (std::nothrow) BridgeIndex();
    if (index == nullptr) {
      ret = SEEKDB_HIPVS_ERR_NO_MEMORY;
    } else {
      index->params = *params;
      if (dim > 0) {
        index->params.dim = dim;
      }
      index->ids.resize(static_cast<size_t>(id_count));
      if (id_bytes > 0) {
        memcpy(index->ids.data(), buf + SEEKDB_HIPVS_PAYLOAD_HEADER, static_cast<size_t>(id_bytes));
      }
      index->num_elements = id_count;

      cuvsHnswIndexParams_t hnsw_params = nullptr;
      if (cuvsHnswIndexParamsCreate(&hnsw_params) != CUVS_SUCCESS) {
        ret = SEEKDB_HIPVS_ERR_HIPVS;
      } else {
        hnsw_params->hierarchy = to_cuvs_hierarchy(params->hierarchy);
        hnsw_params->ef_construction = params->ef_construction > 0 ? params->ef_construction : 200;
        hnsw_params->num_threads = 0;
        cuvsHnswIndex_t hnsw = nullptr;
        if (cuvsHnswIndexCreate(&hnsw) != CUVS_SUCCESS) {
          ret = SEEKDB_HIPVS_ERR_HIPVS;
        } else if (cuvsHnswDeserialize(resources->res, hnsw_params, path.c_str(), index->params.dim,
                                       to_cuvs_metric(params->metric), hnsw) != CUVS_SUCCESS) {
          cuvsHnswIndexDestroy(hnsw);
          ret = SEEKDB_HIPVS_ERR_HIPVS;
        } else {
          index->hnsw = hnsw;
        }
        cuvsHnswIndexParamsDestroy(hnsw_params);
      }
    }
  }
  remove(path.c_str());
  seekdb_hipvs_resources_destroy(raw_resources);

  if (ret != SEEKDB_HIPVS_OK) {
    if (index != nullptr) {
      seekdb_hipvs_index_destroy(index);
    }
    return ret;
  }
  *idx = index;
  return SEEKDB_HIPVS_OK;
}

void seekdb_hipvs_buffer_free(char *buf) { free(buf); }

} // extern "C"
