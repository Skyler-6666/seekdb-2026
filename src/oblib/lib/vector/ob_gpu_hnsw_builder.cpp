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

#include "lib/vector/ob_gpu_hnsw_builder.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include "lib/oblog/ob_log.h"

#if defined(__linux__) || defined(__APPLE__) || defined(__ANDROID__)
#define OB_GPU_HNSW_PLUGIN_SUPPORTED 1
#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>
#else
#define OB_GPU_HNSW_PLUGIN_SUPPORTED 0
#endif

namespace oceanbase
{
namespace common
{
namespace obvectorutil
{

static constexpr const char *OB_GPU_HNSW_PLUGIN_ENV = "OB_GPU_HNSW_PLUGIN_PATH";
static constexpr const char *OB_GPU_HNSW_PLUGIN_SONAME = "libob_gpu_hnsw_plugin.so";

static std::mutex g_gpu_hnsw_builder_mutex;

#if OB_GPU_HNSW_PLUGIN_SUPPORTED
static bool is_regular_file(const char *path)
{
  return path != nullptr && access(path, R_OK) == 0;
}

static bool is_directory(const char *path)
{
  if (path == nullptr) {
    return false;
  }
  struct stat file_stat;
  if (stat(path, &file_stat) != 0) {
    return false;
  }
  return S_ISDIR(file_stat.st_mode);
}

/* Directory that holds the running binary, used to find a plugin installed next to it. */
static bool get_executable_dir(char *buffer, size_t buffer_size)
{
  bool found = false;
  ssize_t length = readlink("/proc/self/exe", buffer, buffer_size - 1);
  if (length > 0) {
    buffer[length] = '\0';
    char *slash = strrchr(buffer, '/');
    if (slash != nullptr) {
      *slash = '\0';
      found = true;
    }
  }
  return found;
}
#endif

GpuHnswBuilder &GpuHnswBuilder::instance()
{
  static GpuHnswBuilder builder;
  return builder;
}

GpuHnswBuilder::~GpuHnswBuilder()
{
  reset_plugin();
}

void GpuHnswBuilder::reset_plugin()
{
#if OB_GPU_HNSW_PLUGIN_SUPPORTED
  if (handle_ != nullptr) {
    dlclose(handle_);
  }
#endif
  handle_ = nullptr;
  build_func_ = nullptr;
  free_result_func_ = nullptr;
  available_func_ = nullptr;
  version_func_ = nullptr;
}

int GpuHnswBuilder::load_plugin()
{
  int ret = OB_SUCCESS;
#if OB_GPU_HNSW_PLUGIN_SUPPORTED
  char candidates[6][512];
  int candidate_count = 0;
  const char *configured = getenv(OB_GPU_HNSW_PLUGIN_ENV);
  if (configured != nullptr && configured[0] != '\0') {
    if (is_directory(configured)) {
      snprintf(candidates[candidate_count++], sizeof(candidates[0]), "%s/%s", configured,
               OB_GPU_HNSW_PLUGIN_SONAME);
    } else {
      snprintf(candidates[candidate_count++], sizeof(candidates[0]), "%s", configured);
    }
  }
  char exe_dir[512];
  if (get_executable_dir(exe_dir, sizeof(exe_dir))) {
    snprintf(candidates[candidate_count++], sizeof(candidates[0]), "%s/%s", exe_dir,
             OB_GPU_HNSW_PLUGIN_SONAME);
  }
  snprintf(candidates[candidate_count++], sizeof(candidates[0]), "/usr/local/lib/%s",
           OB_GPU_HNSW_PLUGIN_SONAME);
  snprintf(candidates[candidate_count++], sizeof(candidates[0]), "/usr/lib/%s", OB_GPU_HNSW_PLUGIN_SONAME);
  snprintf(candidates[candidate_count++], sizeof(candidates[0]), "./%s", OB_GPU_HNSW_PLUGIN_SONAME);

  for (int i = 0; i < candidate_count && handle_ == nullptr; ++i) {
    if (!is_regular_file(candidates[i])) {
      continue;
    }
    void *handle = dlopen(candidates[i], RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
      LOG_WARN("[OBGPUHNSW] dlopen plugin failed", KCSTRING(candidates[i]), KCSTRING(dlerror()));
      continue;
    }
    auto build_func = reinterpret_cast<int (*)(const ObGpuHnswBuildParam *, ObGpuHnswBuildResult *)>(
        dlsym(handle, "ob_gpu_hnsw_build"));
    auto free_result_func = reinterpret_cast<void (*)(ObGpuHnswBuildResult *)>(dlsym(handle, "ob_gpu_hnsw_free_result"));
    auto available_func = reinterpret_cast<int (*)(void)>(dlsym(handle, "ob_gpu_hnsw_available"));
    auto version_func = reinterpret_cast<const char *(*)(void)>(dlsym(handle, "ob_gpu_hnsw_version"));
    if (build_func == nullptr || free_result_func == nullptr || available_func == nullptr) {
      LOG_WARN("[OBGPUHNSW] plugin misses required symbols", KCSTRING(candidates[i]));
      dlclose(handle);
      continue;
    }
    handle_ = handle;
    build_func_ = build_func;
    free_result_func_ = free_result_func;
    available_func_ = available_func;
    version_func_ = version_func;
    snprintf(plugin_path_, sizeof(plugin_path_), "%s", candidates[i]);
    if (version_func_ != nullptr) {
      snprintf(plugin_version_, sizeof(plugin_version_), "%s", version_func_());
    }
    LOG_INFO("[OBGPUHNSW] loaded gpu hnsw plugin", KCSTRING(plugin_path_), KCSTRING(plugin_version_));
  }
  if (handle_ == nullptr) {
    ret = OB_ENTRY_NOT_EXIST;
  }
#else
  ret = OB_NOT_SUPPORTED;
#endif
  return ret;
}

bool GpuHnswBuilder::is_available()
{
  std::lock_guard<std::mutex> guard(g_gpu_hnsw_builder_mutex);
  if (!load_attempted_) {
    load_attempted_ = true;
    load_plugin();
  }
  if (handle_ == nullptr) {
    return false;
  }
  return available_func_() == 1;
}

const char *GpuHnswBuilder::plugin_path() const
{
  return plugin_path_;
}

const char *GpuHnswBuilder::plugin_version() const
{
  return plugin_version_;
}

void GpuHnswBuilder::free_result(ObGpuHnswBuildResult &result)
{
  std::lock_guard<std::mutex> guard(g_gpu_hnsw_builder_mutex);
  if (free_result_func_ != nullptr) {
    free_result_func_(&result);
  } else {
    free(result.data_);
    result.data_ = nullptr;
    result.size_ = 0;
  }
}

int GpuHnswBuilder::build(const float *vectors,
                          int64_t count,
                          int64_t dim,
                          int32_t max_degree,
                          int32_t ef_construction,
                          const char *metric,
                          int32_t build_algo,
                          int32_t intermediate_graph_degree,
                          int32_t nn_descent_iters,
                          ObGpuHnswBuildResult &result)
{
  int ret = OB_SUCCESS;
  memset(&result, 0, sizeof(result));
  if (!is_available()) {
    ret = OB_NOT_SUPPORTED;
  } else {
    ObGpuHnswBuildParam param;
    memset(&param, 0, sizeof(param));
    param.vectors_ = vectors;
    param.n_rows_ = count;
    param.dim_ = dim;
    param.max_degree_ = max_degree;
    param.ef_construction_ = ef_construction;
    param.build_algo_ = build_algo;
    param.intermediate_graph_degree_ = intermediate_graph_degree;
    param.nn_descent_iters_ = nn_descent_iters;
    if (metric == nullptr || strcmp(metric, "l2") == 0) {
      param.metric_ = OB_GPU_METRIC_L2;
    } else if (strcmp(metric, "ip") == 0) {
      param.metric_ = OB_GPU_METRIC_INNER_PRODUCT;
    } else {
      ret = OB_NOT_SUPPORTED;
    }
    if (OB_SUCC(ret)) {
      const int build_status = build_func_(&param, &result);
      if (build_status != OB_GPU_BUILD_OK) {
        LOG_WARN("[OBGPUHNSW] gpu build failed", K(build_status), KCSTRING(result.message_), K(count), K(dim),
                 K(max_degree));
        ret = OB_ERR_UNEXPECTED;
      }
    }
  }
  return ret;
}

} // namespace obvectorutil
} // namespace common
} // namespace oceanbase
