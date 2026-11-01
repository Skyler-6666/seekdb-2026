// Configure-time compile-and-link probe for the cuVS/hipVS C API surface used
// by src/oblib/lib/vector/ob_vsag_adaptor.cpp. It is never executed; compiling
// and linking it verifies that the installed headers and libcuvs_c.so provide
// every type, field, and symbol the hipVS backend calls. SeekdbHipvs.cmake
// falls back to the VSAG CPU path when this probe does not pass.

#include <cstdint>

#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/hnsw.h>
#include <dlpack/dlpack.h>

namespace {
volatile int g_keep = 0;

template <typename T>
void use(const T &value)
{
  if (value == 0) {
    g_keep = 1;
  }
}

void fill_float_tensor(DLManagedTensor &tensor, void *data, int64_t *shape)
{
  tensor.dl_tensor.data = data;
  tensor.dl_tensor.device.device_type = kDLCPU;
  tensor.dl_tensor.device.device_id = 0;
  tensor.dl_tensor.ndim = 2;
  tensor.dl_tensor.dtype.code = kDLFloat;
  tensor.dl_tensor.dtype.bits = 32;
  tensor.dl_tensor.dtype.lanes = 1;
  tensor.dl_tensor.shape = shape;
  tensor.dl_tensor.strides = nullptr;
  tensor.dl_tensor.byte_offset = 0;
}
}  // namespace

int main()
{
  cuvsResources_t resources = 0;
  cuvsHnswIndex_t hnsw_index = nullptr;
  cuvsCagraIndex_t cagra_index = nullptr;
  cuvsHnswIndexParams_t hnsw_params = nullptr;
  cuvsHnswExtendParams_t extend_params = nullptr;
  cuvsHnswSearchParams_t search_params = nullptr;
  cuvsCagraIndexParams_t cagra_params = nullptr;
  cuvsDistanceType metric = L2Expanded;
  cuvsError_t result = CUVS_SUCCESS;
  use(InnerProduct);
  use(cuvsGetLastErrorText());

  if (cuvsResourcesCreate(&resources) != CUVS_SUCCESS ||
      cuvsHnswIndexCreate(&hnsw_index) != CUVS_SUCCESS) {
    return 1;
  }

  int64_t shape[2] = {1, 1};
  float data = 0.0F;
  uint64_t neighbor = 0;
  float distance = 0.0F;
  DLManagedTensor tensor{};
  DLManagedTensor neighbor_tensor{};
  DLManagedTensor distance_tensor{};
  fill_float_tensor(tensor, &data, shape);
  fill_float_tensor(distance_tensor, &distance, shape);
  neighbor_tensor.dl_tensor.data = &neighbor;
  neighbor_tensor.dl_tensor.device.device_type = kDLCPU;
  neighbor_tensor.dl_tensor.device.device_id = 0;
  neighbor_tensor.dl_tensor.ndim = 2;
  neighbor_tensor.dl_tensor.dtype.code = kDLUInt;
  neighbor_tensor.dl_tensor.dtype.bits = 64;
  neighbor_tensor.dl_tensor.dtype.lanes = 1;
  neighbor_tensor.dl_tensor.shape = shape;
  neighbor_tensor.dl_tensor.strides = nullptr;
  neighbor_tensor.dl_tensor.byte_offset = 0;

  if (cuvsCagraIndexParamsCreate(&cagra_params) != CUVS_SUCCESS ||
      cuvsCagraIndexCreate(&cagra_index) != CUVS_SUCCESS) {
    return 1;
  }
  cagra_params->metric = metric;
  cagra_params->graph_degree = 32;
  cagra_params->intermediate_graph_degree = 64;
  cagra_params->build_algo = AUTO_SELECT;
  cagra_params->nn_descent_niter = 0;
  cagra_params->compression = nullptr;
  result = cuvsCagraBuild(resources, cagra_params, &tensor, cagra_index);
  use(result);

  if (cuvsHnswIndexParamsCreate(&hnsw_params) != CUVS_SUCCESS) {
    return 1;
  }
  hnsw_params->hierarchy = cuvsHnswHierarchy::CPU;
  hnsw_params->ef_construction = 40;
  hnsw_params->num_threads = 0;
  result = cuvsHnswFromCagra(resources, hnsw_params, cagra_index, hnsw_index);
  use(result);

  hnsw_index->dtype.code = kDLFloat;
  hnsw_index->dtype.bits = 32;
  hnsw_index->dtype.lanes = 1;

  if (cuvsHnswExtendParamsCreate(&extend_params) != CUVS_SUCCESS) {
    return 1;
  }
  extend_params->num_threads = 0;
  result = cuvsHnswExtend(resources, extend_params, &tensor, hnsw_index);
  use(result);

  if (cuvsHnswSearchParamsCreate(&search_params) != CUVS_SUCCESS) {
    return 1;
  }
  search_params->ef = 40;
  search_params->num_threads = 0;
  result = cuvsHnswSearch(resources, search_params, hnsw_index, &tensor,
                          &neighbor_tensor, &distance_tensor);
  use(result);

  result = cuvsHnswSerialize(resources, "/tmp/seekdb_hipvs_probe.bin", hnsw_index);
  use(result);
  result = cuvsHnswDeserialize(resources, hnsw_params, "/tmp/seekdb_hipvs_probe.bin",
                               1, metric, hnsw_index);
  use(result);

  cuvsHnswSearchParamsDestroy(search_params);
  cuvsHnswExtendParamsDestroy(extend_params);
  cuvsHnswIndexParamsDestroy(hnsw_params);
  cuvsCagraIndexParamsDestroy(cagra_params);
  cuvsCagraIndexDestroy(cagra_index);
  cuvsHnswIndexDestroy(hnsw_index);
  cuvsResourcesDestroy(resources);
  return g_keep;
}
