// Minimal mock of the cuVS/hipVS C API for the configure-time probe tests.
// It is never linked into seekdb: the fake install tree in the tests uses it
// so that cmake/hipvs_api_probe.cc can compile and link against a library
// that really provides the symbols it references.
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/hnsw.h>
#include <new>

cuvsError_t cuvsResourcesCreate(cuvsResources_t *resources)
{
  *resources = 1;
  return CUVS_SUCCESS;
}

cuvsError_t cuvsResourcesDestroy(cuvsResources_t resources)
{
  (void)resources;
  return CUVS_SUCCESS;
}

const char *cuvsGetLastErrorText(void)
{
  return "";
}

cuvsError_t cuvsCagraIndexParamsCreate(cuvsCagraIndexParams_t *params)
{
  *params = new cuvsCagraIndexParams();
  return CUVS_SUCCESS;
}

cuvsError_t cuvsCagraIndexParamsDestroy(cuvsCagraIndexParams_t params)
{
  delete params;
  return CUVS_SUCCESS;
}

cuvsError_t cuvsCagraIndexCreate(cuvsCagraIndex_t *index)
{
  *index = nullptr;
  return CUVS_SUCCESS;
}

cuvsError_t cuvsCagraIndexDestroy(cuvsCagraIndex_t index)
{
  (void)index;
  return CUVS_SUCCESS;
}

cuvsError_t cuvsCagraBuild(cuvsResources_t resources, cuvsCagraIndexParams_t params,
                           DLManagedTensor *dataset, cuvsCagraIndex_t index)
{
  (void)resources;
  (void)params;
  (void)dataset;
  (void)index;
  return CUVS_SUCCESS;
}

cuvsError_t cuvsHnswIndexParamsCreate(cuvsHnswIndexParams_t *params)
{
  *params = new cuvsHnswIndexParams();
  return CUVS_SUCCESS;
}

cuvsError_t cuvsHnswIndexParamsDestroy(cuvsHnswIndexParams_t params)
{
  delete params;
  return CUVS_SUCCESS;
}

cuvsError_t cuvsHnswIndexCreate(cuvsHnswIndex_t *index)
{
  *index = new cuvsHnswIndex();
  return CUVS_SUCCESS;
}

cuvsError_t cuvsHnswIndexDestroy(cuvsHnswIndex_t index)
{
  delete index;
  return CUVS_SUCCESS;
}

cuvsError_t cuvsHnswFromCagra(cuvsResources_t resources, cuvsHnswIndexParams_t params,
                              cuvsCagraIndex_t cagra, cuvsHnswIndex_t hnsw)
{
  (void)resources;
  (void)params;
  (void)cagra;
  (void)hnsw;
  return CUVS_SUCCESS;
}

cuvsError_t cuvsHnswExtendParamsCreate(cuvsHnswExtendParams_t *params)
{
  *params = new cuvsHnswExtendParams();
  return CUVS_SUCCESS;
}

cuvsError_t cuvsHnswExtendParamsDestroy(cuvsHnswExtendParams_t params)
{
  delete params;
  return CUVS_SUCCESS;
}

cuvsError_t cuvsHnswExtend(cuvsResources_t resources, cuvsHnswExtendParams_t params,
                           DLManagedTensor *dataset, cuvsHnswIndex_t index)
{
  (void)resources;
  (void)params;
  (void)dataset;
  (void)index;
  return CUVS_SUCCESS;
}

cuvsError_t cuvsHnswSerialize(cuvsResources_t resources, const char *path, cuvsHnswIndex_t index)
{
  (void)resources;
  (void)path;
  (void)index;
  return CUVS_SUCCESS;
}

cuvsError_t cuvsHnswDeserialize(cuvsResources_t resources, cuvsHnswIndexParams_t params,
                                const char *path, int dim, cuvsDistanceType metric,
                                cuvsHnswIndex_t index)
{
  (void)resources;
  (void)params;
  (void)path;
  (void)dim;
  (void)metric;
  (void)index;
  return CUVS_SUCCESS;
}

cuvsError_t cuvsHnswSearchParamsCreate(cuvsHnswSearchParams_t *params)
{
  *params = new cuvsHnswSearchParams();
  return CUVS_SUCCESS;
}

cuvsError_t cuvsHnswSearchParamsDestroy(cuvsHnswSearchParams_t params)
{
  delete params;
  return CUVS_SUCCESS;
}

cuvsError_t cuvsHnswSearch(cuvsResources_t resources, cuvsHnswSearchParams_t params,
                           cuvsHnswIndex_t index, DLManagedTensor *query,
                           DLManagedTensor *neighbors, DLManagedTensor *distances)
{
  (void)resources;
  (void)params;
  (void)index;
  (void)query;
  (void)neighbors;
  (void)distances;
  return CUVS_SUCCESS;
}
