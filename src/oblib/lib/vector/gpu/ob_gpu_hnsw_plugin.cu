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

/**
 * @file
 *
 * GPU HNSW build plugin, compiled with hipcc against hipVS.
 *
 * The expensive part of building an HNSW index is the all-neighbours KNN graph.
 * hipVS builds that graph on the GPU with CAGRA; the cheap upper layers of the
 * hierarchy are then built on the GPU as well, with the same nn-descent that
 * CAGRA uses for the graph of the higher levels.
 *
 * The result is written in the nmslib/hnswlib v0.7.0 format, which is the same
 * format VSAG reads, so seekdb only has to rewrite the external labels of the
 * elements before handing the bytes over to VSAG.
 */

#include "ob_gpu_hnsw_plugin.h"

#include <hip/hip_runtime.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance.h>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/hnsw.h>
#include <dlpack/dlpack.h>

#define OB_GPU_PLUGIN_VERSION "ob-gpu-hnsw-plugin/1.0 (hipVS CAGRA -> HNSW)"

namespace
{

using Clock = std::chrono::steady_clock;

double elapsed_seconds(const Clock::time_point &start)
{
  return std::chrono::duration<double>(Clock::now() - start).count();
}

/**
 * The plugin owns the host copy of the vectors. cuVS only takes a non owning
 * view of the tensor, so the deleter below is a no-op that keeps the plugin
 * in charge of the memory in case cuVS ever releases the tensor itself.
 */
void noop_tensor_deleter(DLManagedTensor *tensor)
{
  (void)tensor;
}

void set_message(char *message, size_t message_size, const std::string &text)
{
  if (message == nullptr || message_size == 0) {
    return;
  }
  snprintf(message, message_size, "%s", text.c_str());
}

/* Map the plugin metric to the cuVS distance type. */
cuvsDistanceType to_cuvs_metric(int32_t metric)
{
  switch (metric) {
    case OB_GPU_METRIC_INNER_PRODUCT:
      return InnerProduct;
    case OB_GPU_METRIC_L2:
    default:
      return L2Expanded;
  }
}

cuvsCagraGraphBuildAlgo to_cuvs_build_algo(int32_t algo)
{
  switch (algo) {
    case OB_GPU_BUILD_IVF_PQ:
      return IVF_PQ;
    case OB_GPU_BUILD_NN_DESCENT:
      return NN_DESCENT;
    case OB_GPU_BUILD_ITERATIVE_CAGRA_SEARCH:
      return ITERATIVE_CAGRA_SEARCH;
    case OB_GPU_BUILD_AUTO_SELECT:
    default:
      return AUTO_SELECT;
  }
}

/* Read a whole file into a heap buffer. */
int read_whole_file(const char *path, char *&data, size_t &size)
{
  int status = OB_GPU_BUILD_OK;
  FILE *file = fopen(path, "rb");
  data = nullptr;
  size = 0;
  if (file == nullptr) {
    status = OB_GPU_BUILD_ERR_BUILD_FAILED;
  } else {
    if (fseek(file, 0, SEEK_END) != 0) {
      status = OB_GPU_BUILD_ERR_BUILD_FAILED;
    } else {
      const long file_size = ftell(file);
      if (file_size <= 0) {
        status = OB_GPU_BUILD_ERR_BUILD_FAILED;
      } else {
        rewind(file);
        data = static_cast<char *>(malloc(static_cast<size_t>(file_size)));
        if (data == nullptr) {
          status = OB_GPU_BUILD_ERR_OUT_OF_MEMORY;
        } else if (fread(data, 1, static_cast<size_t>(file_size), file) != static_cast<size_t>(file_size)) {
          free(data);
          data = nullptr;
          status = OB_GPU_BUILD_ERR_BUILD_FAILED;
        } else {
          size = static_cast<size_t>(file_size);
        }
      }
    }
    fclose(file);
  }
  return status;
}

} // namespace

extern "C" int ob_gpu_hnsw_available(void)
{
  int device_count = 0;
  if (hipGetDeviceCount(&device_count) != hipSuccess || device_count <= 0) {
    return 0;
  }
  return 1;
}

extern "C" const char *ob_gpu_hnsw_version(void)
{
  return OB_GPU_PLUGIN_VERSION;
}

extern "C" void ob_gpu_hnsw_free_result(ObGpuHnswBuildResult *result)
{
  if (result == nullptr) {
    return;
  }
  free(result->data_);
  result->data_ = nullptr;
  result->size_ = 0;
}

extern "C" int ob_gpu_hnsw_build(const ObGpuHnswBuildParam *param, ObGpuHnswBuildResult *result)
{
  if (param == nullptr || result == nullptr) {
    return OB_GPU_BUILD_ERR_INVALID_PARAM;
  }
  memset(result, 0, sizeof(*result));

  cuvsResources_t resources = nullptr;
  cuvsCagraIndexParams_t cagra_params = nullptr;
  cuvsCagraIndex_t cagra_index = nullptr;
  cuvsHnswIndexParams_t hnsw_params = nullptr;
  cuvsHnswIndex_t hnsw_index = nullptr;
  float *host_vectors = nullptr;
  std::vector<int64_t> shape(2);
  std::vector<int64_t> strides(2);
  DLManagedTensor dataset;
  char temp_path[] = "/tmp/ob_gpu_hnsw_XXXXXX";
  bool temp_file_created = false;

  const Clock::time_point start = Clock::now();

  if (param->vectors_ == nullptr || param->n_rows_ <= 0 || param->dim_ <= 0
      || param->max_degree_ <= 0) {
    set_message(result->message_, sizeof(result->message_), "invalid build parameter");
    return OB_GPU_BUILD_ERR_INVALID_PARAM;
  }
  if (ob_gpu_hnsw_available() == 0) {
    set_message(result->message_, sizeof(result->message_), "no usable AMD GPU");
    return OB_GPU_BUILD_ERR_UNAVAILABLE;
  }

  /*
   * hipVS derives the hnswlib M from half of the graph degree, while VSAG
   * expects M to be exactly the max_degree of the index it creates. Building
   * the graph with a degree of 2 * max_degree therefore reproduces the layout
   * VSAG expects.
   */
  const size_t graph_degree = static_cast<size_t>(param->max_degree_) * 2;
  if (static_cast<size_t>(param->n_rows_) < graph_degree) {
    set_message(result->message_, sizeof(result->message_), "not enough rows for the requested graph degree");
    return OB_GPU_BUILD_ERR_INVALID_PARAM;
  }

  const size_t vector_count = static_cast<size_t>(param->n_rows_) * static_cast<size_t>(param->dim_);
  host_vectors = static_cast<float *>(malloc(vector_count * sizeof(float)));
  if (host_vectors == nullptr) {
    set_message(result->message_, sizeof(result->message_), "out of memory while copying vectors");
    return OB_GPU_BUILD_ERR_OUT_OF_MEMORY;
  }
  memcpy(host_vectors, param->vectors_, vector_count * sizeof(float));

  shape[0] = param->n_rows_;
  shape[1] = param->dim_;
  strides[0] = param->dim_;
  strides[1] = 1;
  memset(&dataset, 0, sizeof(dataset));
  dataset.dl_tensor.data = host_vectors;
  dataset.dl_tensor.device = DLDevice{kDLCPU, 0};
  dataset.dl_tensor.ndim = 2;
  dataset.dl_tensor.dtype = DLDataType{kDLFloat, 32, 1};
  dataset.dl_tensor.shape = shape.data();
  dataset.dl_tensor.strides = strides.data();
  dataset.manager_ctx = nullptr;
  dataset.deleter = noop_tensor_deleter;

  int status = OB_GPU_BUILD_OK;
  if (cuvsResourcesCreate(&resources) != CUVS_SUCCESS) {
    set_message(result->message_, sizeof(result->message_), "cuvsResourcesCreate failed");
    status = OB_GPU_BUILD_ERR_UNAVAILABLE;
    goto cleanup;
  }

  if (cuvsCagraIndexParamsCreate(&cagra_params) != CUVS_SUCCESS) {
    set_message(result->message_, sizeof(result->message_), "cuvsCagraIndexParamsCreate failed");
    status = OB_GPU_BUILD_ERR_BUILD_FAILED;
    goto cleanup;
  }
  cagra_params->metric = to_cuvs_metric(param->metric_);
  cagra_params->graph_degree = graph_degree;
  cagra_params->build_algo = to_cuvs_build_algo(param->build_algo_);
  if (param->nn_descent_iters_ > 0) {
    cagra_params->nn_descent_niter = static_cast<size_t>(param->nn_descent_iters_);
  }
  {
    /*
     * The intermediate graph has to be at least as wide as the output graph,
     * which is how cuVS derives its own defaults.
     */
    size_t intermediate = std::max<size_t>(graph_degree * 2, 128);
    if (param->intermediate_graph_degree_ > 0) {
      intermediate = static_cast<size_t>(param->intermediate_graph_degree_);
    }
    if (intermediate > static_cast<size_t>(param->n_rows_)) {
      intermediate = static_cast<size_t>(param->n_rows_);
    }
    if (intermediate < graph_degree) {
      intermediate = graph_degree;
    }
    cagra_params->intermediate_graph_degree = intermediate;
  }
  result->graph_degree_ = static_cast<int32_t>(graph_degree);

  if (cuvsCagraIndexCreate(&cagra_index) != CUVS_SUCCESS) {
    set_message(result->message_, sizeof(result->message_), "cuvsCagraIndexCreate failed");
    status = OB_GPU_BUILD_ERR_BUILD_FAILED;
    goto cleanup;
  }

  if (cuvsCagraBuild(resources, cagra_params, &dataset, cagra_index) != CUVS_SUCCESS) {
    set_message(result->message_, sizeof(result->message_), "cuvsCagraBuild failed");
    status = OB_GPU_BUILD_ERR_BUILD_FAILED;
    goto cleanup;
  }

  if (cuvsHnswIndexParamsCreate(&hnsw_params) != CUVS_SUCCESS) {
    set_message(result->message_, sizeof(result->message_), "cuvsHnswIndexParamsCreate failed");
    status = OB_GPU_BUILD_ERR_BUILD_FAILED;
    goto cleanup;
  }
  /*
   * The hierarchy has to be built on the GPU. The CPU hierarchy of hipVS adds
   * the points with hnswlib and leaves the entry point at the first point that
   * got a level above zero, which is not necessarily the point of the highest
   * level. The GPU hierarchy picks the point of the highest level as the entry
   * point, the serialized image is the same nmslib layout in both cases.
   */
  hnsw_params->hierarchy = cuvsHnswHierarchy::GPU;
  hnsw_params->ef_construction = param->ef_construction_ > 0 ? param->ef_construction_ : 200;
  hnsw_params->num_threads = 0;

  if (cuvsHnswIndexCreate(&hnsw_index) != CUVS_SUCCESS) {
    set_message(result->message_, sizeof(result->message_), "cuvsHnswIndexCreate failed");
    status = OB_GPU_BUILD_ERR_BUILD_FAILED;
    goto cleanup;
  }

  if (cuvsHnswFromCagraWithDataset(resources, hnsw_params, cagra_index, hnsw_index, &dataset)
      != CUVS_SUCCESS) {
    set_message(result->message_, sizeof(result->message_), "cuvsHnswFromCagraWithDataset failed");
    status = OB_GPU_BUILD_ERR_BUILD_FAILED;
    goto cleanup;
  }

  {
    const int fd = mkstemp(temp_path);
    if (fd < 0) {
      set_message(result->message_, sizeof(result->message_), "cannot create temporary file");
      status = OB_GPU_BUILD_ERR_BUILD_FAILED;
      goto cleanup;
    }
    close(fd);
    temp_file_created = true;
  }

  if (cuvsHnswSerialize(resources, temp_path, hnsw_index) != CUVS_SUCCESS) {
    set_message(result->message_, sizeof(result->message_), "cuvsHnswSerialize failed");
    status = OB_GPU_BUILD_ERR_BUILD_FAILED;
    goto cleanup;
  }

  {
    char *data = nullptr;
    size_t size = 0;
    const int read_status = read_whole_file(temp_path, data, size);
    if (read_status != OB_GPU_BUILD_OK) {
      set_message(result->message_, sizeof(result->message_), "cannot read the serialized index");
      status = read_status;
      goto cleanup;
    }
    result->data_ = data;
    result->size_ = size;
  }
  set_message(result->message_, sizeof(result->message_), "ok");

cleanup:
  if (temp_file_created) {
    unlink(temp_path);
  }
  if (hnsw_index != nullptr) {
    cuvsHnswIndexDestroy(hnsw_index);
  }
  if (hnsw_params != nullptr) {
    cuvsHnswIndexParamsDestroy(hnsw_params);
  }
  if (cagra_index != nullptr) {
    cuvsCagraIndexDestroy(cagra_index);
  }
  if (cagra_params != nullptr) {
    cuvsCagraIndexParamsDestroy(cagra_params);
  }
  if (resources != nullptr) {
    cuvsResourcesDestroy(resources);
  }
  free(host_vectors);
  result->build_seconds_ = elapsed_seconds(start);
  if (status != OB_GPU_BUILD_OK) {
    ob_gpu_hnsw_free_result(result);
  }
  return status;
}
