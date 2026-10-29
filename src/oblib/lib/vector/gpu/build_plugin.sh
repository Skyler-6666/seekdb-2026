#!/bin/bash
#
# Build libob_gpu_hnsw_plugin.so, the optional GPU HNSW build plugin of seekdb.
#
# The plugin is compiled with hipcc against hipVS (the AMD port of NVIDIA cuVS).
# It is loaded by the observer with dlopen at runtime, so seekdb itself is
# built and deployed exactly like an ordinary build.
#
# Prerequisites:
#   * ROCm with hipcc on the PATH
#   * hipVS installed (or built) so that libcuvs.so / libcuvs_c.so and the
#     cuvs headers are available
#   * the dlpack headers (hipVS needs them too, they are fetched by its own
#     build into <hipVS build dir>/cpp/build/_deps/dlpack-src/include)
#
# Usage:
#   HIPVS_ROOT=/opt/hipvs DLPACK_INCLUDE_DIR=/opt/dlpack/include ./build_plugin.sh
#
# Environment variables:
#   HIPVS_ROOT          installation prefix of hipVS (include/ and lib/)
#   HIPVS_BUILD_DIR     hipVS source tree, used to locate the dlpack headers
#   DLPACK_INCLUDE_DIR  directory that contains dlpack/dlpack.h
#   OUTPUT_DIR          where the .so is written, defaults to the source dir
#   CMAKE_BUILD_TYPE    unused, kept for symmetry with the cmake path

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_FILE="${SCRIPT_DIR}/ob_gpu_hnsw_plugin.cu"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}}"
OUTPUT_FILE="${OUTPUT_DIR}/libob_gpu_hnsw_plugin.so"

if ! command -v hipcc >/dev/null 2>&1; then
  echo "error: hipcc not found in PATH, install ROCm first" >&2
  exit 1
fi

HIPVS_ROOT="${HIPVS_ROOT:-}"
if [[ -z "${HIPVS_ROOT}" ]]; then
  for candidate in /opt/hipvs /usr/local/hipvs /opt/rocm/hipvs /usr/local/rapids/cuvs; do
    if [[ -d "${candidate}/include/cuvs" ]]; then
      HIPVS_ROOT="${candidate}"
      break
    fi
  done
fi

if [[ -z "${HIPVS_ROOT}" || ! -d "${HIPVS_ROOT}/include/cuvs" ]]; then
  echo "error: set HIPVS_ROOT to the hipVS installation prefix" >&2
  exit 1
fi

# The dlpack headers are a public dependency of the cuvs headers, so they have
# to be on the include path of every translation unit that includes them.
DLPACK_INCLUDE_DIR="${DLPACK_INCLUDE_DIR:-}"
if [[ -z "${DLPACK_INCLUDE_DIR}" ]]; then
  for candidate in \
      "${HIPVS_ROOT}/include" \
      "${HIPVS_BUILD_DIR:-}/cpp/build/_deps/dlpack-src/include" \
      "${HIPVS_ROOT}/cpp/build/_deps/dlpack-src/include"; do
    if [[ -n "${candidate}" && -f "${candidate}/dlpack/dlpack.h" ]]; then
      DLPACK_INCLUDE_DIR="${candidate}"
      break
    fi
  done
fi

if [[ -z "${DLPACK_INCLUDE_DIR}" || ! -f "${DLPACK_INCLUDE_DIR}/dlpack/dlpack.h" ]]; then
  echo "error: set DLPACK_INCLUDE_DIR to the directory that contains dlpack/dlpack.h" >&2
  exit 1
fi

echo "hipVS      : ${HIPVS_ROOT}"
echo "dlpack     : ${DLPACK_INCLUDE_DIR}"
echo "output     : ${OUTPUT_FILE}"

mkdir -p "${OUTPUT_DIR}"

# libcuvs_c holds the C API that the plugin calls, libcuvs holds the
# implementation it is built on.
hipcc -O3 -std=c++17 -fPIC -shared -fvisibility=default \
  -Wno-unused-result \
  -I"${HIPVS_ROOT}/include" \
  -I"${DLPACK_INCLUDE_DIR}" \
  "${SOURCE_FILE}" \
  -L"${HIPVS_ROOT}/lib" \
  -Wl,-rpath,"${HIPVS_ROOT}/lib" \
  -lcuvs_c -lcuvs \
  -o "${OUTPUT_FILE}"

echo "built ${OUTPUT_FILE}"
