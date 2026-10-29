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
 * @file ob_hipvs_adaptor.h
 *
 * OceanBase side glue for the optional AMD hipVS backend.
 *
 * The heavy lifting (opening the shared library, resolving the ABI) lives in
 * `seekdb_hipvs_loader.h`, which has no OceanBase dependency. This header
 * adds the bits the observer needs: logging, the mapping between the seekdb
 * metric names and the bridge metrics, and the mapping between the bridge
 * status codes and the OceanBase error codes.
 */

#pragma once

#include "lib/vector/hipvs/seekdb_hipvs_bridge.h"
#include "lib/vector/hipvs/seekdb_hipvs_loader.h"

namespace oceanbase {
namespace common {
namespace obhipvs {

/**
 * @brief Whether the GPU backend can be used on this node.
 *
 * True when the bridge was loaded, implements the expected ABI and found a
 * usable GPU.
 */
bool is_available();

/**
 * @brief Log the state of the GPU backend.
 *
 * Called once when the backend is first used, so that the observer log says
 * whether the GPU is in play and which hipVS version is used.
 */
void log_availability();

/**
 * @brief Translate a seekdb metric name ("l2", "ip", "cosine") into a bridge
 *        metric.
 *
 * @param[in]  metric the metric name used by the DDL.
 * @param[out] out    the matching `seekdb_hipvs_metric_t`.
 * @return true when the metric is supported.
 */
bool to_bridge_metric(const char *metric, int32_t &out);

/**
 * @brief Translate a bridge status into an OceanBase error code.
 */
int to_ob_error(int32_t status);

} // namespace obhipvs
} // namespace common
} // namespace oceanbase
