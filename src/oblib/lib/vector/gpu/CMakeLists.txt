cmake_minimum_required(VERSION 3.20)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(seekdb_gpu_hnsw_plugin LANGUAGES CXX)
endif()
include(GNUInstallDirs)

option(SEEKDB_ENABLE_HIP_HNSW_PLUGIN
       "Build the optional hipVS HNSW build plugin" OFF)

if(NOT SEEKDB_ENABLE_HIP_HNSW_PLUGIN)
  message(STATUS "hipVS HNSW plugin disabled; seekdb uses the CPU HNSW path")
  return()
endif()

find_program(HIPCC_EXECUTABLE NAMES hipcc REQUIRED)
set(HIPVS_ROOT "/opt/hipvs" CACHE PATH "hipVS installation prefix")
set(DLPACK_INCLUDE_DIR "${HIPVS_ROOT}/include"
    CACHE PATH "Directory containing dlpack/dlpack.h")

find_library(CUVS_C_LIBRARY NAMES cuvs_c REQUIRED
             PATHS "${HIPVS_ROOT}/lib" NO_DEFAULT_PATH)
find_library(CUVS_LIBRARY NAMES cuvs REQUIRED
             PATHS "${HIPVS_ROOT}/lib" NO_DEFAULT_PATH)

set(PLUGIN_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/libob_gpu_hnsw_plugin.so")
add_custom_command(
  OUTPUT "${PLUGIN_OUTPUT}"
  COMMAND "${CMAKE_COMMAND}" -E env
          "HIPVS_ROOT=${HIPVS_ROOT}"
          "DLPACK_INCLUDE_DIR=${DLPACK_INCLUDE_DIR}"
          "OUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}"
          bash "${CMAKE_CURRENT_SOURCE_DIR}/build_plugin.sh"
  DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/build_plugin.sh"
    "${CMAKE_CURRENT_SOURCE_DIR}/ob_gpu_hnsw_plugin.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/ob_gpu_hnsw_plugin.h"
  COMMENT "Building optional hipVS HNSW plugin"
  VERBATIM)

add_custom_target(ob_gpu_hnsw_plugin ALL DEPENDS "${PLUGIN_OUTPUT}")
install(FILES "${PLUGIN_OUTPUT}"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/seekdb")
