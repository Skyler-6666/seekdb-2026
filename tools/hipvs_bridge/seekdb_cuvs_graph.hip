// Bounded GPU construction only. Query execution stays in the fixed cuVS C API.
#include <hip/hip_runtime.h>
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/nn_descent.h>
#include "seekdb_cuvs_graph_algorithms.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace seekdb_gpu_graph;
void check_cuvs(int status)
{
  if (status != CUVS_SUCCESS) {
    const char *error = cuvsGetLastErrorText();
    throw std::runtime_error(error ? error : "cuVS construction call failed without error text");
  }
}

void check_hip(hipError_t status)
{
  if (status != hipSuccess) { throw std::runtime_error(hipGetErrorString(status)); }
}

struct DeviceBuffer {
  cuvsResources_t res;
  void *ptr = nullptr;
  size_t bytes;
  DeviceBuffer(cuvsResources_t resource, size_t size) : res(resource), bytes(size)
  {
    check_cuvs(cuvsRMMAlloc(res, &ptr, bytes));
  }
  ~DeviceBuffer() { if (ptr) { cuvsRMMFree(res, ptr, bytes); } }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
};

// Opt-in diagnostics only: synchronization changes execution scheduling.
struct BuildTrace {
  enum Stage { train, lookup, refine, prune, reverse, wrap, stage_count };
  cuvsResources_t res;
  bool enabled;
  double milliseconds[stage_count]{};
  std::chrono::steady_clock::time_point started;
  explicit BuildTrace(cuvsResources_t resource) : res(resource)
  {
    const char *value = std::getenv("SEEKDB_CUVS_BUILD_TRACE");
    enabled = value && value[0] == '1' && value[1] == '\0';
  }
  void begin()
  {
    if (enabled) { check_cuvs(cuvsStreamSync(res)); started = std::chrono::steady_clock::now(); }
  }
  void end(Stage stage)
  {
    if (enabled) {
      check_cuvs(cuvsStreamSync(res));
      milliseconds[stage] += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    }
  }
  void report(uint32_t rows) const
  {
    if (enabled) {
      std::fprintf(stderr, "[seekdb_cuvs] build_stage_ms rows=%u train=%.3f lookup=%.3f "
                   "refine=%.3f prune=%.3f reverse=%.3f wrap=%.3f\n", rows,
                   milliseconds[train], milliseconds[lookup], milliseconds[refine],
                   milliseconds[prune], milliseconds[reverse], milliseconds[wrap]);
    }
  }
};

template<unsigned Count>
__device__ void sort_keys(uint64_t *keys)
{
  for (unsigned size = 2; size <= Count; size <<= 1) {
    for (unsigned stride = size >> 1; stride; stride >>= 1) {
      for (unsigned i = threadIdx.x; i < Count; i += blockDim.x) {
        const unsigned partner = i ^ stride;
        if (partner > i) { exchange_keys(keys[i], keys[partner], (i & size) == 0); }
      }
      __syncthreads();
    }
  }
}

__global__ void sort_nn_graph(const float *data, uint32_t rows, unsigned dim,
                              uint32_t *knn, int *error)
{
  __shared__ uint64_t keys[intermediate_degree];
  const uint32_t row = blockIdx.x;
  const unsigned rank = threadIdx.x;
  const uint32_t id = knn[size_t(row) * intermediate_degree + rank];
  if (id >= rows || id == row) {
    atomicExch(error, 1);
    keys[rank] = UINT64_MAX;
  } else {
    float distance = 0;
    for (unsigned d = 0; d < dim; ++d) {
      const float delta = data[size_t(row) * dim + d] - data[size_t(id) * dim + d];
      distance += delta * delta;
    }
    if (!isfinite(distance)) { atomicExch(error, 1); keys[rank] = UINT64_MAX; }
    else { keys[rank] = (uint64_t(__float_as_uint(distance)) << 32) | id; }
  }
  __syncthreads();
  sort_keys<intermediate_degree>(keys);
  __syncthreads();
  const uint32_t sorted_id = uint32_t(keys[rank]);
  if (rank < intermediate_degree && (rank == 0 || sorted_id != uint32_t(keys[rank - 1]))) {
    knn[size_t(row) * intermediate_degree + rank] = sorted_id;
  } else if (rank < intermediate_degree) {
    atomicExch(error, 1);
  }
}

__global__ void prune_graph(const uint32_t *knn, uint32_t rows, uint32_t *graph)
{
  __shared__ uint64_t rank_table[rank_table_capacity];
  __shared__ unsigned detours[intermediate_degree];
  __shared__ uint64_t scores[intermediate_degree];
  const uint32_t row = blockIdx.x;
  const unsigned rank = threadIdx.x;
  const unsigned degree = rows - 1 < intermediate_degree ? rows - 1 : intermediate_degree;
  for (unsigned i = rank; i < rank_table_capacity; i += blockDim.x) { rank_table[i] = UINT64_MAX; }
  detours[rank] = 0;
  __syncthreads();
  if (rank < degree) {
    const uint32_t id = knn[size_t(row) * intermediate_degree + rank];
    unsigned slot = hash_id(id) & (rank_table_capacity - 1);
    while (atomicCAS(reinterpret_cast<unsigned long long *>(rank_table + slot),
                     ~0ULL, static_cast<unsigned long long>(rank_key(id, rank))) != ~0ULL) {
      slot = (slot + 1) & (rank_table_capacity - 1);
    }
  }
  __syncthreads();
  // Read each neighbor-of-neighbor row once, coalesced across the block.
  // A shared rank lookup counts the same exact two-hop detours as before.
  for (unsigned via_rank = 0; via_rank + 1 < degree; ++via_rank) {
    const uint32_t via = knn[size_t(row) * intermediate_degree + via_rank];
    if (rank < degree) {
      const uint32_t candidate = knn[size_t(via) * intermediate_degree + rank];
      const unsigned target_rank = find_rank(rank_table, candidate);
      if (target_rank < degree && target_rank > via_rank) { atomicAdd(detours + target_rank, 1U); }
    }
  }
  __syncthreads();
  scores[rank] = rank < degree ? (uint64_t(detours[rank]) << 32) | rank : UINT64_MAX;
  __syncthreads();
  sort_keys<intermediate_degree>(scores);
  if (rank < graph_degree) { graph[size_t(row) * graph_degree + rank] = knn[size_t(row) * intermediate_degree + uint32_t(scores[rank])]; }
}

__global__ void reverse_edges(const uint32_t *graph, uint32_t rows,
                              unsigned rank, uint32_t *incoming, uint32_t *counts)
{
  const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows) {
    const uint32_t target = graph[size_t(row) * graph_degree + rank];
    const uint32_t slot = atomicAdd(counts + target, 1U);
    if (slot < graph_degree) { incoming[size_t(target) * graph_degree + slot] = row; }
  }
}

__global__ void merge_graph(uint32_t *graph, const uint32_t *incoming,
                            const uint32_t *counts, uint32_t rows, int *error)
{
  __shared__ uint32_t scratch[graph_degree];
  if (threadIdx.x == 0) {
    const uint32_t row = blockIdx.x;
    const unsigned count = counts[row] < graph_degree ? counts[row] : graph_degree;
    if (!merge_reverse(graph + size_t(row) * graph_degree,
                       incoming + size_t(row) * graph_degree, count, row, rows, scratch)) { atomicExch(error, 1); }
  }
}

void check_graph_error(cuvsResources_t res, hipStream_t stream, int *device_error)
{
  int error = 0;
  check_hip(hipMemcpyAsync(&error, device_error, sizeof(error), hipMemcpyDeviceToHost, stream));
  check_cuvs(cuvsStreamSync(res));
  if (error) { throw std::runtime_error("GPU graph construction produced invalid or duplicate neighbors"); }
}
} // namespace

extern "C" int seekdb_cuvs_build_graph(cuvsResources_t res, cuvsCagraIndexParams_t params,
                                      const float *host_data, uint32_t rows, unsigned dim,
                                      unsigned int **graph)
{
  cuvsNNDescentIndexParams_t nn_params = nullptr;
  cuvsNNDescentIndex_t nn_index = nullptr;
  auto cleanup = [&] {
    cuvsStreamSync(res);
    if (nn_index) { cuvsNNDescentIndexDestroy(nn_index); nn_index = nullptr; }
    if (nn_params) { cuvsNNDescentIndexParamsDestroy(nn_params); nn_params = nullptr; }
  };
  try {
    if (params->graph_degree != graph_degree || params->intermediate_graph_degree != intermediate_degree) {
      throw std::runtime_error("NN graph builder requires graph degrees 128/256");
    }
    if (!host_data || rows <= intermediate_degree) { throw std::runtime_error("invalid NN graph input"); }
    hipStream_t stream;
    check_cuvs(cuvsStreamGet(res, &stream));
    BuildTrace trace(res);
    DeviceBuffer flag(res, sizeof(int));
    auto *error = static_cast<int *>(flag.ptr);
    check_hip(hipMemsetAsync(error, 0, sizeof(int), stream));
    trace.begin();
    check_cuvs(cuvsNNDescentIndexParamsCreate(&nn_params));
    nn_params->metric = L2Expanded;
    nn_params->graph_degree = intermediate_degree;
    nn_params->intermediate_graph_degree = intermediate_degree * 3 / 2;
    nn_params->max_iterations = params->nn_descent_niter;
    nn_params->termination_threshold = 0.0001f;
    nn_params->return_distances = false;
    check_cuvs(cuvsNNDescentIndexCreate(&nn_index));
    int64_t dataset_shape[2] = {rows, dim};
    DLManagedTensor dataset = {};
    dataset.dl_tensor.data = const_cast<float *>(host_data);
    dataset.dl_tensor.device.device_type = kDLCPU;
    dataset.dl_tensor.device.device_id = 0;
    dataset.dl_tensor.ndim = 2;
    dataset.dl_tensor.dtype.code = kDLFloat;
    dataset.dl_tensor.dtype.bits = 32;
    dataset.dl_tensor.dtype.lanes = 1;
    dataset.dl_tensor.shape = dataset_shape;
    dataset.dl_tensor.byte_offset = 0;
    std::vector<uint32_t> host_graph(size_t(rows) * intermediate_degree);
    int64_t graph_shape[2] = {rows, intermediate_degree};
    DLManagedTensor graph_tensor = {};
    graph_tensor.dl_tensor.data = host_graph.data();
    graph_tensor.dl_tensor.device.device_type = kDLCPU;
    graph_tensor.dl_tensor.device.device_id = 0;
    graph_tensor.dl_tensor.ndim = 2;
    graph_tensor.dl_tensor.dtype.code = kDLUInt;
    graph_tensor.dl_tensor.dtype.bits = 32;
    graph_tensor.dl_tensor.dtype.lanes = 1;
    graph_tensor.dl_tensor.shape = graph_shape;
    graph_tensor.dl_tensor.byte_offset = 0;
    check_cuvs(cuvsNNDescentBuild(res, nn_params, &dataset, &graph_tensor, nn_index));
    check_cuvs(cuvsStreamSync(res));
    trace.end(BuildTrace::train);
    cleanup();
    uint32_t *output_graph = nullptr;
    {
      DeviceBuffer knn(res, size_t(rows) * intermediate_degree * sizeof(uint32_t));
      check_hip(hipMemcpyAsync(knn.ptr, host_graph.data(), knn.bytes, hipMemcpyHostToDevice, stream));
      {
        DeviceBuffer data(res, size_t(rows) * dim * sizeof(float));
        check_hip(hipMemcpyAsync(data.ptr, host_data, data.bytes, hipMemcpyHostToDevice, stream));
        trace.begin();
        hipLaunchKernelGGL(sort_nn_graph, dim3(rows), dim3(intermediate_degree), 0, stream,
                           static_cast<const float *>(data.ptr), rows, dim,
                           static_cast<uint32_t *>(knn.ptr), error);
        check_hip(hipGetLastError());
        check_graph_error(res, stream, error);
        trace.end(BuildTrace::lookup);
      }
      trace.begin();
      check_cuvs(cuvsRMMAlloc(res, reinterpret_cast<void **>(graph), size_t(rows) * graph_degree * sizeof(uint32_t)));
      output_graph = *graph;
      hipLaunchKernelGGL(prune_graph, dim3(rows), dim3(256), 0, stream,
                         static_cast<const uint32_t *>(knn.ptr), rows, output_graph);
      check_hip(hipGetLastError());
      check_cuvs(cuvsStreamSync(res));
      trace.end(BuildTrace::prune);
    }
    {
      trace.begin();
      DeviceBuffer incoming(res, size_t(rows) * graph_degree * sizeof(uint32_t));
      DeviceBuffer counts(res, size_t(rows) * sizeof(uint32_t));
      check_hip(hipMemsetAsync(counts.ptr, 0, counts.bytes, stream));
      // Rank-ordered launches prevent low-priority edges from filling the cap
      // before higher-priority edges, as in the fixed CAGRA optimizer.
      for (unsigned rank = 0; rank < graph_degree; ++rank) {
        hipLaunchKernelGGL(reverse_edges, dim3(unsigned((uint64_t(rows) + 255) / 256)),
                           dim3(256), 0, stream, output_graph, rows, rank,
                           static_cast<uint32_t *>(incoming.ptr), static_cast<uint32_t *>(counts.ptr));
        check_hip(hipGetLastError());
      }
      hipLaunchKernelGGL(merge_graph, dim3(rows), dim3(256), 0, stream, output_graph,
                         static_cast<const uint32_t *>(incoming.ptr),
                         static_cast<const uint32_t *>(counts.ptr), rows, error);
      check_hip(hipGetLastError());
      check_graph_error(res, stream, error);
      trace.end(BuildTrace::reverse);
    }
    trace.report(rows);
    cuvsSetLastErrorText(nullptr);
    return CUVS_SUCCESS;
  } catch (const std::exception &error) {
    const std::string message = error.what();
    cleanup();
    cuvsSetLastErrorText(message.c_str());
    return CUVS_ERROR;
  }
}
