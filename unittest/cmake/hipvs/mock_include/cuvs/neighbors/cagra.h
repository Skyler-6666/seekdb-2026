#ifndef CUVS_NEIGHBORS_CAGRA_H_
#define CUVS_NEIGHBORS_CAGRA_H_
#include <stddef.h>
#include <dlpack/dlpack.h>
typedef struct cuvsCagraIndex *cuvsCagraIndex_t;
typedef struct cuvsCagraIndexParams *cuvsCagraIndexParams_t;
enum { AUTO_SELECT = 0 };
struct cuvsCagraIndexParams {
  cuvsDistanceType metric;
  size_t graph_degree;
  size_t intermediate_graph_degree;
  int build_algo;
  int nn_descent_niter;
  void *compression;
};
#endif
