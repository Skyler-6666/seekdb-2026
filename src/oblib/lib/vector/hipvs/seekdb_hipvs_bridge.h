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
 * @file seekdb_hipvs_bridge.h
 *
 * Stable C ABI between the seekdb observer and the AMD hipVS (HIP port of the
 * RAPIDS cuVS library) backend.
 *
 * The observer never links against hipVS. It loads
 * `libseekdb_hipvs_bridge.so` at runtime and resolves the symbols declared
 * here. This keeps ROCm a purely optional runtime dependency: a seekdb binary
 * built on a machine without ROCm still runs, and simply reports that the GPU
 * backend is unavailable.
 *
 * The GPU HNSW build implemented by the bridge is:
 *
 *   1. `cuvsCagraBuild`      - build the k-NN graph on the AMD GPU
 *   2. `cuvsHnswFromCagra`   - convert that graph into a real HNSW index
 *   3. `cuvsHnswSearch`      - answer queries against the resulting index
 *
 * Step 1 is the expensive part of an HNSW build (it is O(N^2) distance work
 * plus graph optimisation) and is exactly what the GPU accelerates. Step 2
 * produces a standard hnswlib-format index, so the resulting index keeps the
 * same ANN semantics as the CPU implementation, and stays mutable/extendable
 * when the CPU hierarchy is requested.
 *
 * EXTERNAL IDS
 * ------------
 * An hnswlib index labels its elements with sequential ordinals, while seekdb
 * addresses its rows with the "vid" of the vector. The bridge keeps the
 * mapping and translates the neighbour ordinals returned by the search into
 * vids. The mapping is persisted next to the hnswlib payload, so it survives
 * a restart.
 *
 * ABI STABILITY
 * -------------
 * Every struct below is plain C with a fixed layout, and every symbol is
 * looked up by name at runtime. The loader refuses a bridge whose
 * `seekdb_hipvs_abi_version()` does not match `SEEKDB_HIPVS_BRIDGE_ABI_VERSION`
 * exactly, so a stale bridge can never be mistaken for a compatible one.
 * When the ABI changes, bump the version and keep the old symbols around.
 */

#ifndef SEEKDB_HIPVS_BRIDGE_H_
#define SEEKDB_HIPVS_BRIDGE_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Version of the bridge ABI. The loader requires an exact match.
 */
#define SEEKDB_HIPVS_BRIDGE_ABI_VERSION 1

/**
 * Result codes returned by every bridge entry point.
 */
typedef enum seekdb_hipvs_status_e {
  SEEKDB_HIPVS_OK = 0,
  /** A NULL pointer or an out-of-range argument was passed in. */
  SEEKDB_HIPVS_ERR_INVALID_ARGUMENT = 1,
  /** The hipVS/ROCm runtime is present but no usable GPU was found. */
  SEEKDB_HIPVS_ERR_NO_GPU = 2,
  /** A hipVS call failed. */
  SEEKDB_HIPVS_ERR_HIPVS = 3,
  /** The index state does not allow the requested operation. */
  SEEKDB_HIPVS_ERR_UNSUPPORTED = 4,
  /** Out of memory. */
  SEEKDB_HIPVS_ERR_NO_MEMORY = 5,
  /** The serialized blob could not be read or written. */
  SEEKDB_HIPVS_ERR_IO = 6
} seekdb_hipvs_status_t;

/**
 * Distance metric. The values mirror the strings seekdb already uses for the
 * `distance` DDL option ("l2", "ip", "cosine").
 */
typedef enum seekdb_hipvs_metric_e {
  SEEKDB_HIPVS_METRIC_L2 = 0,
  SEEKDB_HIPVS_METRIC_IP = 1,
  SEEKDB_HIPVS_METRIC_COSINE = 2
} seekdb_hipvs_metric_t;

/**
 * Algorithm used to build the k-NN graph on the GPU.
 *
 * - CAGRA: the "Highly Parallel Graph Construction" algorithm. Higher graph
 *   quality, noticeably slower than NN-Descent on very large inputs.
 * - NN_DESCENT: the GPU port of NN-Descent. Faster, slightly lower recall.
 *
 * Both are implemented by hipVS (`cuvsCagraBuild` with the matching
 * `build_algo`).
 */
typedef enum seekdb_hipvs_graph_algo_e {
  SEEKDB_HIPVS_GRAPH_ALGO_CAGRA = 0,
  SEEKDB_HIPVS_GRAPH_ALGO_NN_DESCENT = 1
} seekdb_hipvs_graph_algo_t;

/**
 * Hierarchy built for the HNSW index.
 *
 * - NONE: base layer only. Cheapest, but the serialized format is only
 *   readable by the hipVS hnswlib wrapper and the index cannot be extended.
 * - CPU: the full hierarchy is built with CPU threads. The result is a
 *   standard hnswlib index: mutable, extendable, and the on-disk format is the
 *   regular hnswlib one.
 * - GPU: the full hierarchy is built on the GPU.
 */
typedef enum seekdb_hipvs_hierarchy_e {
  SEEKDB_HIPVS_HIERARCHY_NONE = 0,
  SEEKDB_HIPVS_HIERARCHY_CPU = 1,
  SEEKDB_HIPVS_HIERARCHY_GPU = 2
} seekdb_hipvs_hierarchy_t;

/**
 * Parameters describing an index to create.
 */
typedef struct seekdb_hipvs_index_params_s {
  /** Distance metric, a `seekdb_hipvs_metric_t`. */
  int32_t metric;
  /** Graph build algorithm, a `seekdb_hipvs_graph_algo_t`. */
  int32_t graph_algo;
  /** Hierarchy to build, a `seekdb_hipvs_hierarchy_t`. */
  int32_t hierarchy;
  /** Vector dimension. Must be > 0. */
  int32_t dim;
  /** HNSW graph degree ("m" in the DDL). Must be > 0. */
  int32_t m;
  /** ef_construction used while building the hierarchy. 0 selects a default. */
  int32_t ef_construction;
  /** ef used while searching. 0 selects a default. */
  int32_t ef_search;
  /** Degree of the intermediate graph used while pruning. 0 selects a default. */
  int32_t intermediate_degree;
  /** Iterations of the graph build. 0 selects a default. */
  int32_t build_iteration;
  /** Iterations of NN-Descent. 0 selects a default. */
  int32_t nn_descent_iters;
} seekdb_hipvs_index_params_t;

/**
 * @brief Return the ABI version implemented by this bridge.
 */
int32_t seekdb_hipvs_abi_version(void);

/**
 * @brief Return the hipVS version string, e.g. "25.10.0".
 *
 * @param[out] buf      destination buffer, may be NULL.
 * @param[in]  buf_size size of @p buf in bytes.
 */
void seekdb_hipvs_version(char *buf, int32_t buf_size);

/**
 * @brief Describe the GPU that would be used for the build.
 *
 * @param[out] buf      destination buffer, may be NULL.
 * @param[in]  buf_size size of @p buf in bytes.
 * @return SEEKDB_HIPVS_OK, or SEEKDB_HIPVS_ERR_NO_GPU when no usable GPU is
 *         present.
 */
int32_t seekdb_hipvs_gpu_info(char *buf, int32_t buf_size);

/**
 * @brief Create the resources shared by every index (HIP stream, BLAS handle).
 *
 * @param[out] res opaque resource handle.
 */
int32_t seekdb_hipvs_resources_create(void **res);

/**
 * @brief Destroy resources created by `seekdb_hipvs_resources_create`.
 */
int32_t seekdb_hipvs_resources_destroy(void *res);

/**
 * @brief Create an empty index.
 *
 * @param[in]  params index parameters, copied by the bridge.
 * @param[out] idx    opaque index handle.
 */
int32_t seekdb_hipvs_index_create(const seekdb_hipvs_index_params_t *params, void **idx);

/**
 * @brief Destroy an index created by `seekdb_hipvs_index_create`.
 */
int32_t seekdb_hipvs_index_destroy(void *idx);

/**
 * @brief Build the index from @p count vectors.
 *
 * This is the GPU accelerated part: the k-NN graph is constructed on the AMD
 * GPU and then converted into an HNSW index.
 *
 * @param[in] idx     index handle.
 * @param[in] vectors row-major buffer of @p count * @p dim float32 values.
 * @param[in] dim     vector dimension.
 * @param[in] count   number of vectors, must be > 0.
 * @param[in] ids     array of @p count external ids. When NULL the bridge
 *                    uses the row ordinal.
 */
int32_t seekdb_hipvs_index_build(void *idx,
                                 const float *vectors,
                                 int64_t dim,
                                 int64_t count,
                                 const int64_t *ids);

/**
 * @brief Add vectors to an already built index.
 *
 * Only supported for indexes whose hierarchy is
 * `SEEKDB_HIPVS_HIERARCHY_CPU`, matching the hipVS `cuvsHnswExtend` contract.
 *
 * @param[in] idx     index handle.
 * @param[in] vectors row-major buffer of @p count * @p dim float32 values.
 * @param[in] dim     vector dimension.
 * @param[in] count   number of vectors, must be > 0.
 * @param[in] ids     array of @p count external ids.
 */
int32_t seekdb_hipvs_index_add(void *idx,
                               const float *vectors,
                               int64_t dim,
                               int64_t count,
                               const int64_t *ids);

/**
 * @brief Run a k-NN query.
 *
 * The neighbour ordinals returned by the hnswlib index are translated into the
 * external ids the index was built with.
 *
 * @param[in]  idx        index handle.
 * @param[in]  query      query vector, @p dim float32 values.
 * @param[in]  dim        vector dimension.
 * @param[in]  topk       number of neighbours to return.
 * @param[in]  ef         search breadth. 0 selects the index default.
 * @param[out] ids        output buffer of at least @p topk int64 values.
 * @param[out] distances  output buffer of at least @p topk float values.
 * @param[out] result_size number of entries actually written.
 */
int32_t seekdb_hipvs_index_search(void *idx,
                                  const float *query,
                                  int64_t dim,
                                  int64_t topk,
                                  int32_t ef,
                                  int64_t *ids,
                                  float *distances,
                                  int64_t *result_size);

/**
 * @brief Number of vectors currently stored in the index.
 */
int32_t seekdb_hipvs_index_num_elements(void *idx, int64_t *num_elements);

/**
 * @brief Smallest and largest external id stored in the index.
 */
int32_t seekdb_hipvs_index_vid_bound(void *idx, int64_t *min_vid, int64_t *max_vid);

/**
 * @brief Serialize the index into a freshly allocated buffer.
 *
 * The payload holds the external ids followed by the hnswlib representation of
 * the index. The buffer is allocated with `malloc` and must be released with
 * `seekdb_hipvs_buffer_free`.
 *
 * @param[out] buf  buffer holding the serialized index.
 * @param[out] len  length of @p buf in bytes.
 */
int32_t seekdb_hipvs_index_serialize(void *idx, char **buf, int64_t *len);

/**
 * @brief Load an index previously produced by `seekdb_hipvs_index_serialize`.
 *
 * @param[in]  buf    serialized bytes.
 * @param[in]  len    length of @p buf in bytes.
 * @param[in]  params index parameters, used for the metric and the dimension.
 * @param[out] idx    opaque index handle.
 */
int32_t seekdb_hipvs_index_deserialize(const char *buf,
                                       int64_t len,
                                       const seekdb_hipvs_index_params_t *params,
                                       void **idx);

/**
 * @brief Release a buffer returned by `seekdb_hipvs_index_serialize`.
 */
void seekdb_hipvs_buffer_free(char *buf);

#ifdef __cplusplus
}
#endif

#endif /* SEEKDB_HIPVS_BRIDGE_H_ */
