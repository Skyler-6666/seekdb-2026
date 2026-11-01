#ifndef CUVS_CORE_C_API_H_
#define CUVS_CORE_C_API_H_
#include <stddef.h>
#include <stdint.h>
typedef uintptr_t cuvsResources_t;
typedef int cuvsError_t;
enum { CUVS_SUCCESS = 0, CUVS_ERROR = 1 };
typedef enum { L2Expanded = 0, InnerProduct = 1 } cuvsDistanceType;
#include "cuvs/neighbors/cagra.h"
#include "cuvs/neighbors/hnsw.h"
#ifdef __cplusplus
extern "C" {
#endif
cuvsError_t cuvsResourcesCreate(cuvsResources_t *resources);
void cuvsResourcesDestroy(cuvsResources_t resources);
const char *cuvsGetLastErrorText(void);
cuvsError_t cuvsCagraIndexParamsCreate(cuvsCagraIndexParams_t *params);
void cuvsCagraIndexParamsDestroy(cuvsCagraIndexParams_t params);
cuvsError_t cuvsCagraIndexCreate(cuvsCagraIndex_t *index);
void cuvsCagraIndexDestroy(cuvsCagraIndex_t index);
cuvsError_t cuvsCagraBuild(cuvsResources_t resources, cuvsCagraIndexParams_t params,
                           DLManagedTensor *dataset, cuvsCagraIndex_t index);
cuvsError_t cuvsHnswIndexParamsCreate(cuvsHnswIndexParams_t *params);
void cuvsHnswIndexParamsDestroy(cuvsHnswIndexParams_t params);
cuvsError_t cuvsHnswIndexCreate(cuvsHnswIndex_t *index);
void cuvsHnswIndexDestroy(cuvsHnswIndex_t index);
cuvsError_t cuvsHnswFromCagra(cuvsResources_t resources, cuvsHnswIndexParams_t params,
                              cuvsCagraIndex_t cagra, cuvsHnswIndex_t hnsw);
cuvsError_t cuvsHnswExtendParamsCreate(cuvsHnswExtendParams_t *params);
void cuvsHnswExtendParamsDestroy(cuvsHnswExtendParams_t params);
cuvsError_t cuvsHnswExtend(cuvsResources_t resources, cuvsHnswExtendParams_t params,
                           DLManagedTensor *dataset, cuvsHnswIndex_t index);
cuvsError_t cuvsHnswSerialize(cuvsResources_t resources, const char *path,
                              cuvsHnswIndex_t index);
cuvsError_t cuvsHnswDeserialize(cuvsResources_t resources, cuvsHnswIndexParams_t params,
                                const char *path, int dim, cuvsDistanceType metric,
                                cuvsHnswIndex_t index);
cuvsError_t cuvsHnswSearchParamsCreate(cuvsHnswSearchParams_t *params);
void cuvsHnswSearchParamsDestroy(cuvsHnswSearchParams_t params);
cuvsError_t cuvsHnswSearch(cuvsResources_t resources, cuvsHnswSearchParams_t params,
                           cuvsHnswIndex_t index, DLManagedTensor *query,
                           DLManagedTensor *neighbors, DLManagedTensor *distances);
#ifdef __cplusplus
}
#endif
#endif
