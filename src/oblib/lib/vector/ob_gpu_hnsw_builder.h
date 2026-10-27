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

#include <stdint.h>
#include "lib/vector/gpu/ob_gpu_hnsw_plugin.h"

namespace oceanbase
{
namespace common
{
namespace obvectorutil
{

/**
 * @brief Client of the optional GPU HNSW build plugin.
 *
 * The plugin is a shared object that is built with hipcc against hipVS. It is
 * loaded lazily on the first use, and every failure degrades to the CPU build
 * path, so a seekdb deployment without an AMD GPU is unaffected.
 */
class GpuHnswBuilder
{
public:
  static GpuHnswBuilder &instance();

  /**
   * @brief Whether the GPU build path can be used.
   */
  bool is_available();

  /**
   * @brief Build an HNSW graph on the GPU.
   *
   * @param vectors                row major float32 vectors, count * dim elements
   * @param count                  number of vectors
   * @param dim                    dimension of a vector
   * @param max_degree             the M of the target VSAG index
   * @param ef_construction        ef_construction of the target VSAG index
   * @param metric                 "l2" or "ip"
   * @param build_algo             0 auto, 1 ivf_pq, 2 nn_descent, 3 iterative_cagra
   * @param intermediate_graph_degree  0 lets the plugin decide
   * @param nn_descent_iters       0 lets the plugin decide
   * @param[out] result            the serialized index in hnswlib v0.7.0 format
   */
  int build(const float *vectors,
            int64_t count,
            int64_t dim,
            int32_t max_degree,
            int32_t ef_construction,
            const char *metric,
            int32_t build_algo,
            int32_t intermediate_graph_degree,
            int32_t nn_descent_iters,
            ObGpuHnswBuildResult &result);

  /**
   * @brief Release the buffer of a build result.
   */
  void free_result(ObGpuHnswBuildResult &result);

  /**
   * @brief Path of the loaded plugin, empty when no plugin was loaded.
   */
  const char *plugin_path() const;

  /**
   * @brief Version of the loaded plugin, empty when no plugin was loaded.
   */
  const char *plugin_version() const;

private:
  GpuHnswBuilder() = default;
  ~GpuHnswBuilder();
  GpuHnswBuilder(const GpuHnswBuilder &) = delete;
  GpuHnswBuilder &operator=(const GpuHnswBuilder &) = delete;

  int load_plugin();
  void reset_plugin();

  void *handle_{nullptr};
  int (*build_func_)(const ObGpuHnswBuildParam *, ObGpuHnswBuildResult *) = nullptr;
  void (*free_result_func_)(ObGpuHnswBuildResult *) = nullptr;
  int (*available_func_)(void) = nullptr;
  const char *(*version_func_)(void) = nullptr;
  char plugin_path_[512]{0};
  char plugin_version_[128]{0};
  bool load_attempted_{false};
};

} // namespace obvectorutil
} // namespace common
} // namespace oceanbase
