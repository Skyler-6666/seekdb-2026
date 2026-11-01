"""Configure-time behaviour of the optional hipVS backend.

Runs cmake against fake hipVS/cuVS install trees and checks whether
cmake/SeekdbHipvs.cmake enables SEEKDB_WITH_HIPVS. The fake tree carries
real headers and a real libcuvs_c.so (built from mock_cuvs.cc), so the
configure-time API probe is exercised end to end. No GPU, ROCm, or seekdb
build is involved.
"""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = Path(__file__).resolve().parent
MOCK_SOURCE = SOURCE_DIR / "mock_cuvs.cc"

PACKAGE_CONFIG = """
add_library(cuvs::c_api SHARED IMPORTED)
set_target_properties(cuvs::c_api PROPERTIES
  IMPORTED_LOCATION "${CMAKE_CURRENT_LIST_DIR}/../../libcuvs_c.so"
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/../../../include")
"""

DLPACK_HEADER = """
#ifndef DLPACK_DLPACK_H_
#define DLPACK_DLPACK_H_
#include <stddef.h>
#include <stdint.h>
typedef enum { kDLCPU = 1 } DLDeviceType;
typedef struct {
  DLDeviceType device_type;
  int device_id;
} DLDevice;
typedef struct {
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
} DLDataType;
typedef struct {
  void *data;
  DLDevice device;
  int32_t ndim;
  DLDataType dtype;
  int64_t *shape;
  int64_t *strides;
  uint64_t byte_offset;
} DLTensor;
typedef struct {
  DLTensor dl_tensor;
  void *manager_ctx;
  void (*deleter)(void *);
} DLManagedTensor;
enum { kDLInt = 0U, kDLUInt = 1U, kDLFloat = 2U };
#endif
"""

CAGRA_HEADER = """
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
"""

HNSW_HEADER = """
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
  int ef;
  int num_threads;
};
struct cuvsHnswIndex {
  DLDataType dtype;
};
#endif
"""

# Same header with the search "ef" field missing: the configure-time probe
# must notice the mismatch and stay on the CPU path.
HNSW_HEADER_WITHOUT_EF = HNSW_HEADER.replace("  int ef;\n", "")

C_API_HEADER = """
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
"""


def find_cxx():
    for name in ("c++", "g++", "clang++"):
        found = shutil.which(name)
        if found:
            return found
    return None


def make_install(root, *, package, hnsw_header, broken=False, cxx=None):
    include = root / "include"
    (include / "dlpack").mkdir(parents=True, exist_ok=True)
    (include / "dlpack/dlpack.h").write_text(DLPACK_HEADER, encoding="utf-8")
    (include / "cuvs/core").mkdir(parents=True, exist_ok=True)
    (include / "cuvs/neighbors").mkdir(parents=True, exist_ok=True)
    (include / "cuvs/neighbors/cagra.h").write_text(CAGRA_HEADER, encoding="utf-8")
    (include / "cuvs/core/c_api.h").write_text(C_API_HEADER, encoding="utf-8")
    if hnsw_header:
        header = HNSW_HEADER_WITHOUT_EF if broken else HNSW_HEADER
        (include / "cuvs/neighbors/hnsw.h").write_text(header, encoding="utf-8")
    lib_dir = root / "lib"
    lib_dir.mkdir(parents=True, exist_ok=True)
    if hnsw_header and cxx is not None:
        library = lib_dir / "libcuvs_c.so"
        result = subprocess.run(
            [cxx, "-shared", "-fPIC", "-std=c++11", f"-I{include}",
             str(MOCK_SOURCE), "-o", str(library)],
            text=True, encoding="utf-8", capture_output=True)
        if result.returncode != 0:
            raise AssertionError(f"mock libcuvs_c.so build failed: {result.stderr}")
    else:
        (lib_dir / "libcuvs_c.so").touch()
    if package:
        config_dir = root / "lib/cmake/cuvs"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "cuvs-config.cmake").write_text(PACKAGE_CONFIG, encoding="utf-8")
    return root


class HipvsOptionTests(unittest.TestCase):
    def configure(self, workspace, *, prefix=None, env_root=None, mode=None):
        build = workspace / "build"
        command = [
            "cmake", "-S", str(SOURCE_DIR), "-B", str(build),
            f"-DSEEKDB_REPO_ROOT={REPO_ROOT}",
            "-DCMAKE_SYSTEM_PROCESSOR=x86_64",
        ]
        if prefix is not None:
            command.append(f"-DCMAKE_PREFIX_PATH={prefix}")
        if mode is not None:
            command.append(f"-DSEEKDB_ENABLE_HIPVS={mode}")
        env = dict(os.environ)
        env.pop("HIPVS_ROOT", None)
        if env_root is not None:
            env["HIPVS_ROOT"] = str(env_root)
        result = subprocess.run(command, text=True, encoding="utf-8",
                                capture_output=True, env=env)
        return result

    def assert_enabled(self, result):
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertIn("hipVS HNSW backend: enabled", result.stdout)
        self.assertIn("CHECK_DEFS=SEEKDB_WITH_HIPVS=1", result.stdout)
        self.assertIn("cuvs::c_api", result.stdout)
        # The library directory must reach the final link as an rpath: the
        # server has to start without the caller exporting LD_LIBRARY_PATH.
        self.assertIn("LINKER:-rpath,", result.stdout)

    def assert_disabled(self, result, reason):
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertNotIn("SEEKDB_WITH_HIPVS=1", result.stdout)
        self.assertIn(reason, result.stdout + result.stderr)

    def require_cxx(self):
        cxx = find_cxx()
        if not cxx:
            self.skipTest("a C++ compiler is required to exercise the API probe")
        return cxx

    def test_auto_enables_cuvs_package_with_hnsw_extensions(self):
        cxx = self.require_cxx()
        with tempfile.TemporaryDirectory() as tmp:
            root = make_install(Path(tmp) / "hipvs", package=True,
                                hnsw_header=True, cxx=cxx)
            self.assert_enabled(self.configure(Path(tmp), prefix=root))

    def test_auto_ignores_plain_cuvs_without_hnsw_extensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_install(Path(tmp) / "cuvs", package=True, hnsw_header=False)
            self.assert_disabled(self.configure(Path(tmp), prefix=root),
                                 "no hipVS install with cuvs/neighbors/hnsw.h")

    def test_auto_finds_headers_only_install_through_hipvs_root(self):
        cxx = self.require_cxx()
        with tempfile.TemporaryDirectory() as tmp:
            root = make_install(Path(tmp) / "hipvs", package=False,
                                hnsw_header=True, cxx=cxx)
            self.assert_enabled(self.configure(Path(tmp), env_root=root))

    def test_auto_rejects_install_whose_headers_do_not_match_the_source(self):
        cxx = self.require_cxx()
        with tempfile.TemporaryDirectory() as tmp:
            root = make_install(Path(tmp) / "hipvs", package=True,
                                hnsw_header=True, broken=True, cxx=cxx)
            self.assert_disabled(
                self.configure(Path(tmp), prefix=root),
                "installed hipVS does not provide the API this source uses")

    def test_auto_is_quiet_when_no_install_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assert_disabled(self.configure(Path(tmp)),
                                 "hipVS HNSW backend: disabled")

    def test_explicit_on_without_install_warns_but_still_configures(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.configure(Path(tmp), mode="ON")
            self.assert_disabled(result, "SEEKDB_ENABLE_HIPVS=ON but no hipVS install")
            self.assertIn("CMake Warning", result.stdout + result.stderr)

    def test_off_never_looks_for_hipvs(self):
        cxx = find_cxx()
        with tempfile.TemporaryDirectory() as tmp:
            root = make_install(Path(tmp) / "hipvs", package=True,
                                hnsw_header=True, cxx=cxx)
            self.assert_disabled(self.configure(Path(tmp), prefix=root, mode="OFF"),
                                 "hipVS HNSW backend: disabled (SEEKDB_ENABLE_HIPVS=OFF)")


if __name__ == "__main__":
    unittest.main()
