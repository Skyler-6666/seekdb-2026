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

#pragma once

/**
 * @file
 *
 * ABI of the optional GPU HNSW build plugin.
 *
 * The plugin is a standalone shared object that is compiled with hipcc against
 * hipVS and is loaded by seekdb with dlopen at runtime. Keeping the GPU code
 * out of the observer binary means seekdb still builds and runs on machines
 * without an AMD GPU; when the plugin is missing every build simply falls
 * back to the CPU implementation.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Distance metric, mirrors cuvsDistanceType. */
typedef enum ObGpuMetricType {
  OB_GPU_METRIC_L2 = 0,
  OB_GPU_METRIC_INNER_PRODUCT = 1
} ObGpuMetricType;

/* Graph build algorithm, mirrors cuvsCagraGraphBuildAlgo. */
typedef enum ObGpuBuildAlgo {
  OB_GPU_BUILD_AUTO_SELECT = 0,
  OB_GPU_BUILD_IVF_PQ = 1,
  OB_GPU_BUILD_NN_DESCENT = 2,
  OB_GPU_BUILD_ITERATIVE_CAGRA_SEARCH = 3
} ObGpuBuildAlgo;

/* Return codes of ob_gpu_hnsw_build(). */
typedef enum ObGpuBuildStatus {
  OB_GPU_BUILD_OK = 0,
  OB_GPU_BUILD_ERR_UNAVAILABLE = 1,
  OB_GPU_BUILD_ERR_INVALID_PARAM = 2,
  OB_GPU_BUILD_ERR_BUILD_FAILED = 3,
  OB_GPU_BUILD_ERR_OUT_OF_MEMORY = 4
} ObGpuBuildStatus;

typedef struct ObGpuHnswBuildParam {
  /* Row major float32 vectors, n_rows * dim elements. */
  const float *vectors_;
  int64_t n_rows_;
  int64_t dim_;
  /* The M of the target VSAG index. hipVS derives hnswlib M as graph_degree / 2. */
  int32_t max_degree_;
  int32_t ef_construction_;
  int32_t metric_;
  int32_t build_algo_;
  /* 0 lets hipVS pick a default. */
  int32_t intermediate_graph_degree_;
  /* 0 lets hipVS pick a default. */
  int32_t nn_descent_iters_;
} ObGpuHnswBuildParam;

typedef struct ObGpuHnswBuildResult {
  /*
   * The serialized index in the nmslib/hnswlib v0.7.0 layout, which is the same
   * layout VSAG reads. Allocated by the plugin, released with
   * ob_gpu_hnsw_free_result().
   */
  char *data_;
  size_t size_;
  /* Wall clock time spent inside the GPU build. */
  double build_seconds_;
  /* Graph degree that was actually used, for logging. */
  int32_t graph_degree_;
  /* Human readable status, always NUL terminated. */
  char message_[512];
} ObGpuHnswBuildResult;

/**
 * @brief Build an HNSW graph on the GPU and return it in hnswlib v0.7.0 format.
 * @param param  build parameters
 * @param result out parameter holding the serialized index
 * @return OB_GPU_BUILD_OK on success
 */
int ob_gpu_hnsw_build(const ObGpuHnswBuildParam *param, ObGpuHnswBuildResult *result);

/**
 * @brief Release the buffer held by a build result.
 */
void ob_gpu_hnsw_free_result(ObGpuHnswBuildResult *result);

/**
 * @brief Tell whether a usable AMD GPU and a working hipVS runtime are present.
 * @return 1 when the GPU path can be used, 0 otherwise
 */
int ob_gpu_hnsw_available(void);

/**
 * @brief Version string of the plugin, for logging.
 */
const char *ob_gpu_hnsw_version(void);

#ifdef __cplusplus
}
#endif
