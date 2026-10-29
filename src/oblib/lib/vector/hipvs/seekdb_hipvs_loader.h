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
 * @file seekdb_hipvs_loader.h
 *
 * Runtime loader for the optional AMD hipVS bridge.
 *
 * The loader deliberately has no OceanBase dependency: it only knows how to
 * open a shared library, resolve the bridge ABI and report whether the GPU
 * backend can be used. That keeps it unit testable on any machine, with or
 * without ROCm, and it keeps the "is the GPU backend usable" decision in one
 * place.
 *
 * The loader never fails hard. When the bridge is missing, when it has an
 * incompatible ABI, or when no GPU is present, it simply reports that the
 * backend is unavailable and the caller falls back to the CPU implementation.
 */

#ifndef SEEKDB_HIPVS_LOADER_H_
#define SEEKDB_HIPVS_LOADER_H_

#include <cstdint>
#include <string>

#include "lib/vector/hipvs/seekdb_hipvs_bridge.h"

namespace oceanbase {
namespace common {
namespace obhipvs {

/**
 * The bridge entry points, resolved at runtime.
 */
struct HipvsBridge
{
  int32_t (*abi_version)(void){nullptr};
  void (*version)(char *buf, int32_t buf_size){nullptr};
  int32_t (*gpu_info)(char *buf, int32_t buf_size){nullptr};
  int32_t (*resources_create)(void **res){nullptr};
  int32_t (*resources_destroy)(void *res){nullptr};
  int32_t (*index_create)(const seekdb_hipvs_index_params_t *params, void **idx){nullptr};
  int32_t (*index_destroy)(void *idx){nullptr};
  int32_t (*index_build)(void *idx, const float *vectors, int64_t dim, int64_t count,
                         const int64_t *ids){nullptr};
  int32_t (*index_add)(void *idx, const float *vectors, int64_t dim, int64_t count,
                        const int64_t *ids){nullptr};
  int32_t (*index_search)(void *idx, const float *query, int64_t dim, int64_t topk, int32_t ef,
                          int64_t *ids, float *distances, int64_t *result_size){nullptr};
  int32_t (*index_num_elements)(void *idx, int64_t *num_elements){nullptr};
  int32_t (*index_vid_bound)(void *idx, int64_t *min_vid, int64_t *max_vid){nullptr};
  int32_t (*index_serialize)(void *idx, char **buf, int64_t *len){nullptr};
  int32_t (*index_deserialize)(const char *buf, int64_t len, const seekdb_hipvs_index_params_t *params,
                                void **idx){nullptr};
  void (*buffer_free)(char *buf){nullptr};
};

/**
 * Loads and owns the hipVS bridge shared library.
 */
class HipvsLoader
{
public:
  HipvsLoader() = default;
  ~HipvsLoader();

  HipvsLoader(const HipvsLoader &) = delete;
  HipvsLoader &operator=(const HipvsLoader &) = delete;

  /**
   * @brief Load the bridge from the default search path.
   *
   * The search order is:
   *   1. the file named by the `SEEKDB_HIPVS_BRIDGE_PATH` environment variable,
   *   2. `$SEEKDB_HOME/lib/<bridge name>`,
   *   3. `<bridge name>`, resolved by the platform loader (LD_LIBRARY_PATH,
   *      /etc/ld.so.conf, ...).
   *
   * The result is cached; repeated calls are cheap.
   */
  static HipvsLoader &instance();

  /**
   * @brief Load the bridge from an explicit path.
   *
   * Used by the unit test to inject a mock bridge. Returns false when the
   * library cannot be opened or does not implement the expected ABI.
   */
  bool load(const std::string &path);

  /**
   * @brief Whether the bridge was loaded and implements the expected ABI.
   */
  bool is_loaded() const { return loaded_; }

  /**
   * @brief Whether a usable GPU was found by the bridge.
   */
  bool has_gpu() const { return has_gpu_; }

  /**
   * @brief The resolved bridge entry points, NULL when the bridge is missing.
   */
  const HipvsBridge *get_bridge() const { return loaded_ ? &bridge_ : nullptr; }

  /**
   * @brief The hipVS version reported by the bridge.
   */
  const std::string &get_version() const { return version_; }

  /**
   * @brief A human readable description of the GPU used for the builds.
   */
  const std::string &get_gpu_info() const { return gpu_info_; }

  /**
   * @brief Why the bridge could not be used. Empty when everything is fine.
   */
  const std::string &get_error() const { return error_; }

  /**
   * @brief The name of the bridge shared library on this platform.
   */
  static const char *get_bridge_file_name();

private:
  /**
   * @brief Resolve every symbol of the ABI. Returns false when one is missing.
   */
  bool resolve_symbols();

  void unload();

  bool loaded_{false};
  bool has_gpu_{false};
  std::string version_;
  std::string gpu_info_;
  std::string error_;
  HipvsBridge bridge_;
  void *handle_{nullptr};
};

} // namespace obhipvs
} // namespace common
} // namespace oceanbase

#endif /* SEEKDB_HIPVS_LOADER_H_ */
