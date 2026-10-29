# Optional AMD hipVS GPU backend.
#
# The seekdb binary never links against hipVS. The GPU backend is reached
# through a small shared library that is built with the ROCm toolchain and is
# loaded at runtime by the observer. When ROCm or hipVS is not available the
# option stays off and the observer keeps using the CPU backend, so the
# default build is not affected in any way.
#
# Usage:
#
#   cmake -DOB_ENABLE_HIPVS=ON \
#         -DOB_ROCM_ROOT=/opt/rocm-6.3.0 \
#         -DOB_HIPVS_ROOT=/opt/rocm-6.3.0/include/rapids/cuvs \
#         ...
#
# The resulting library is written to <build>/lib/libseekdb_hipvs_bridge.so.
# At runtime the observer looks for it in
#
#   1. $SEEKDB_HIPVS_BRIDGE_PATH,
#   2. $SEEKDB_HOME/lib/libseekdb_hipvs_bridge.so,
#   3. the loader search path (LD_LIBRARY_PATH, /etc/ld.so.conf, ...).

option(OB_ENABLE_HIPVS "Build the optional AMD hipVS GPU backend bridge" OFF)
set(OB_HIPVS_ROOT "" CACHE PATH "The root of the hipVS installation")
set(OB_ROCM_ROOT "" CACHE PATH "The root of the ROCm installation")

# The name of the bridge, without the platform specific prefix and suffix.
set(OB_HIPVS_BRIDGE_NAME "seekdb_hipvs_bridge")

function(seekdb_find_hipvs_include_dir out_var)
  set(candidates
      "${OB_HIPVS_ROOT}/cpp/include"
      "${OB_HIPVS_ROOT}/include"
      "${OB_ROCM_ROOT}/include/rapids/cuvs/cpp/include"
      "${OB_ROCM_ROOT}/include/rapids/cuvs/include")
  foreach(candidate IN LISTS candidates)
    if(EXISTS "${candidate}/cuvs/neighbors/hnsw.h")
      set(${out_var} "${candidate}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
  set(${out_var} "" PARENT_SCOPE)
endfunction()

# dlpack is used by the hipVS C API. It is usually installed next to hipVS.
function(seekdb_find_dlpack_include_dir out_var)
  set(candidates
      "${OB_HIPVS_ROOT}/cpp/include"
      "${OB_HIPVS_ROOT}/include"
      "${OB_ROCM_ROOT}/include/rapids/cuvs/cpp/include"
      "${OB_ROCM_ROOT}/include/rapids/cuvs/include"
      "${OB_ROCM_ROOT}/include"
      "${OB_ROCM_ROOT}/include/rapids/dlpack/include")
  foreach(candidate IN LISTS candidates)
    if(EXISTS "${candidate}/dlpack/dlpack.h")
      set(${out_var} "${candidate}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
  set(${out_var} "" PARENT_SCOPE)
endfunction()

function(seekdb_add_hipvs_bridge)
  if(NOT OB_ENABLE_HIPVS)
    return()
  endif()

  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(WARNING "The AMD hipVS GPU backend is only supported on Linux, skipping it")
    return()
  endif()

  find_program(OB_HIPCC_EXECUTABLE
               NAMES hipcc
               HINTS "${OB_ROCM_ROOT}/bin"
               DOC "The ROCm C++ compiler")
  if(NOT OB_HIPCC_EXECUTABLE)
    message(WARNING "hipcc was not found, the AMD hipVS GPU backend is not built. "
                    "Set OB_ROCM_ROOT to the ROCm installation")
    return()
  endif()

  seekdb_find_hipvs_include_dir(OB_HIPVS_INCLUDE_DIR)
  if(NOT OB_HIPVS_INCLUDE_DIR)
    message(WARNING "The hipVS headers were not found, the AMD GPU backend is not built. "
                    "Set OB_HIPVS_ROOT to the hipVS installation")
    return()
  endif()

  seekdb_find_dlpack_include_dir(OB_DLPACK_INCLUDE_DIR)
  if(NOT OB_DLPACK_INCLUDE_DIR)
    message(WARNING "The dlpack headers were not found, the AMD GPU backend is not built")
    return()
  endif()

  set(OB_HIPVS_BRIDGE_SRC "${CMAKE_SOURCE_DIR}/src/oblib/lib/vector/hipvs/seekdb_hipvs_bridge.cpp")
  set(OB_HIPVS_BRIDGE_LIB "${CMAKE_BINARY_DIR}/lib/lib${OB_HIPVS_BRIDGE_NAME}.so")

  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
  add_custom_command(
    OUTPUT "${OB_HIPVS_BRIDGE_LIB}"
    COMMAND ${OB_HIPCC_EXECUTABLE}
            -std=c++17 -O3 -fPIC -shared
            "-I${CMAKE_SOURCE_DIR}/src/oblib"
            "-I${OB_HIPVS_INCLUDE_DIR}"
            "-I${OB_DLPACK_INCLUDE_DIR}"
            "${OB_HIPVS_BRIDGE_SRC}"
            -o "${OB_HIPVS_BRIDGE_LIB}"
            -L"${OB_HIPVS_ROOT}/lib"
            -L"${OB_ROCM_ROOT}/lib"
            -lcuvs
    DEPENDS "${OB_HIPVS_BRIDGE_SRC}"
    COMMENT "Building the AMD hipVS GPU backend bridge"
    VERBATIM)
  add_custom_target(${OB_HIPVS_BRIDGE_NAME}_all ALL DEPENDS "${OB_HIPVS_BRIDGE_LIB}")

  message(STATUS "The AMD hipVS GPU backend bridge is built into ${OB_HIPVS_BRIDGE_LIB}")
endfunction()
