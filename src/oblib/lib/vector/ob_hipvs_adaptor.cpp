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
 * @file ob_hipvs_adaptor.cpp
 *
 * OceanBase side glue for the optional AMD hipVS backend.
 */

#include "lib/vector/ob_hipvs_adaptor.h"

#include <cstring>

#include "lib/oblog/ob_log.h"

namespace oceanbase {
namespace common {
namespace obhipvs {

bool is_available()
{
  HipvsLoader &loader = HipvsLoader::instance();
  return loader.is_loaded() && loader.has_gpu();
}

void log_availability()
{
  static bool logged = false;
  if (logged) {
    return;
  }
  logged = true;
  HipvsLoader &loader = HipvsLoader::instance();
  if (!loader.is_loaded()) {
    LOG_INFO("[OBHIPVS] the hipVS bridge is not available, the GPU backend is disabled",
             K(loader.get_error()));
  } else if (!loader.has_gpu()) {
    LOG_WARN("[OBHIPVS] the hipVS bridge is loaded but no GPU was found, the GPU backend is disabled",
             K(loader.get_version()), K(loader.get_gpu_info()));
  } else {
    LOG_INFO("[OBHIPVS] the GPU backend is enabled", K(loader.get_version()), K(loader.get_gpu_info()));
  }
}

bool to_bridge_metric(const char *metric, int32_t &out)
{
  bool ret = true;
  if (metric == nullptr) {
    ret = false;
  } else if (strcmp(metric, "l2") == 0) {
    out = SEEKDB_HIPVS_METRIC_L2;
  } else if (strcmp(metric, "ip") == 0) {
    out = SEEKDB_HIPVS_METRIC_IP;
  } else if (strcmp(metric, "cosine") == 0) {
    out = SEEKDB_HIPVS_METRIC_COSINE;
  } else {
    ret = false;
  }
  return ret;
}

int to_ob_error(int32_t status)
{
  int ret = OB_ERR_UNEXPECTED;
  switch (status) {
    case SEEKDB_HIPVS_OK:
      ret = OB_SUCCESS;
      break;
    case SEEKDB_HIPVS_ERR_INVALID_ARGUMENT:
      ret = OB_INVALID_ARGUMENT;
      break;
    case SEEKDB_HIPVS_ERR_NO_GPU:
    case SEEKDB_HIPVS_ERR_UNSUPPORTED:
      ret = OB_NOT_SUPPORTED;
      break;
    case SEEKDB_HIPVS_ERR_NO_MEMORY:
      ret = OB_ALLOCATE_MEMORY_FAILED;
      break;
    case SEEKDB_HIPVS_ERR_HIPVS:
    case SEEKDB_HIPVS_ERR_IO:
    default:
      ret = OB_ERR_UNEXPECTED;
      break;
  }
  return ret;
}

} // namespace obhipvs
} // namespace common
} // namespace oceanbase
