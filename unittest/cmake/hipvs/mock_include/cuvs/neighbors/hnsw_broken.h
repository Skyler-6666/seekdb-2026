// Broken variant: the search parameter struct lacks the "ef" field. The
// configure-time probe must notice and stay on the CPU path.
#ifndef CUVS_NEIGHBORS_HNSW_H_
#define CUVS_NEIGHBORS_HNSW_H_
#include <stddef.h>
#include <dlpack/dlpack.h>
typedef struct cuvsHnswIndex *cuvsHnswIndex_t;
typedef struct cuvsHnswIndexParams *cuvsHnswIndexParams_t;
typedef struct cuvsHnswExtendParams *cuvsHnswExtendParams_t;
typedef struct cuvsHnswSearchParams *cuvsHnswSearchParams_t;
enum class cuvsHnswHierarchy { CPU = 0, GPU = 1 };
struct cuvsHnswIndexParams {
  cuvsHnswHierarchy hierarchy;
  int ef_construction;
  int num_threads;
};
struct cuvsHnswExtendParams {
  int num_threads;
};
struct cuvsHnswSearchParams {
  int num_threads;
};
struct cuvsHnswIndex {
  DLDataType dtype;
};
#endif
