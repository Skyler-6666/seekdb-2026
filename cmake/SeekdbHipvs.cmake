# Optional hipVS GPU HNSW backend (cuVS C API plus the hipVS HNSW extensions).
#
# The project is measured with its own build command line, so the backend has
# to engage by itself when a hipVS install is present. It must equally never
# break a build on a machine without hipVS, or on one that only has a plain
# cuVS without the hipVS extensions.
#
# SEEKDB_ENABLE_HIPVS:
#   AUTO (default) enable when a hipVS install is found
#   ON             same, but warn instead of silently staying on the CPU path
#   OFF            never look for hipVS

function(seekdb_configure_hipvs target)
  set(SEEKDB_ENABLE_HIPVS "AUTO" CACHE STRING
      "hipVS GPU HNSW backend: ON, OFF, or AUTO (enable when hipVS is installed)")
  set_property(CACHE SEEKDB_ENABLE_HIPVS PROPERTY STRINGS ON OFF AUTO)
  string(TOUPPER "${SEEKDB_ENABLE_HIPVS}" _mode)
  if(_mode STREQUAL "OFF")
    message(STATUS "hipVS HNSW backend: disabled (SEEKDB_ENABLE_HIPVS=OFF)")
    return()
  endif()

  set(_skip_reason "")
  if(WIN32 OR APPLE OR ANDROID)
    set(_skip_reason "unsupported platform")
  elseif(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    set(_skip_reason "unsupported architecture ${CMAKE_SYSTEM_PROCESSOR}")
  endif()

  if(_skip_reason STREQUAL "")
    # A hipVS install may expose a CMake package, only headers and a library,
    # or neither. cuvs/neighbors/hnsw.h is what separates hipVS from a plain
    # cuVS install, and it is also what the backend includes.
    find_package(cuvs CONFIG QUIET
      HINTS "$ENV{HIPVS_ROOT}" "$ENV{HIPVS_ROOT}/cpp/install"
            /opt/hipvs /opt/hipvs/cpp/install /opt/cuvs)
    set(_cuvs_lib "")
    set(_cuvs_inc "")
    if(cuvs_FOUND)
      get_target_property(_cuvs_lib cuvs::c_api IMPORTED_LOCATION)
      if(NOT _cuvs_lib)
        get_target_property(_cuvs_lib cuvs::c_api IMPORTED_LOCATION_RELEASE)
      endif()
      get_target_property(_cuvs_inc cuvs::c_api INTERFACE_INCLUDE_DIRECTORIES)
      if(_cuvs_inc)
        list(GET _cuvs_inc 0 _cuvs_inc)
      endif()
    endif()
    if(NOT _cuvs_inc)
      foreach(_prefix IN ITEMS "$ENV{HIPVS_ROOT}" "$ENV{HIPVS_ROOT}/cpp/install"
                              /opt/hipvs /opt/hipvs/cpp/install /opt/cuvs /usr/local)
        if(EXISTS "${_prefix}/include/cuvs/core/c_api.h")
          find_library(SEEKDB_CUVS_C_LIBRARY NAMES cuvs_c libcuvs_c
                       HINTS "${_prefix}/lib" "${_prefix}/lib64" NO_DEFAULT_PATH)
          if(SEEKDB_CUVS_C_LIBRARY)
            set(_cuvs_lib "${SEEKDB_CUVS_C_LIBRARY}")
            set(_cuvs_inc "${_prefix}/include")
            break()
          endif()
        endif()
      endforeach()
    endif()

    if(NOT _cuvs_inc OR NOT EXISTS "${_cuvs_inc}/cuvs/neighbors/hnsw.h")
      set(_skip_reason "no hipVS install with cuvs/neighbors/hnsw.h")
    elseif(NOT _cuvs_lib OR NOT EXISTS "${_cuvs_lib}")
      set(_skip_reason "cuVS library not found next to the hipVS headers")
    else()
      # Before enabling, verify on this machine that the installed headers and
      # library really provide the API the backend source uses. hipVS builds
      # have drifted between releases, and a mismatch here must mean "build
      # the CPU path", never "fail the whole build".
      include(CheckCXXSourceCompiles)
      file(TO_CMAKE_PATH "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/hipvs_api_probe.cc" _hipvs_probe_src)
      set(_hipvs_required_includes "${CMAKE_REQUIRED_INCLUDES}")
      set(_hipvs_required_libraries "${CMAKE_REQUIRED_LIBRARIES}")
      set(_hipvs_required_flags "${CMAKE_REQUIRED_FLAGS}")
      set(_hipvs_required_quiet "${CMAKE_REQUIRED_QUIET}")
      set(CMAKE_REQUIRED_INCLUDES "${_cuvs_inc}")
      set(CMAKE_REQUIRED_LIBRARIES "${_cuvs_lib}")
      set(CMAKE_REQUIRED_QUIET TRUE)
      # Help the linker resolve libcuvs_c.so's own dependencies even when they
      # sit outside the default search path.
      get_filename_component(_cuvs_lib_dir "${_cuvs_lib}" DIRECTORY)
      set(_hipvs_rpath_link "-Wl,-rpath-link,${_cuvs_lib_dir}")
      foreach(_rocm_lib_dir IN ITEMS "$ENV{ROCM_PATH}/lib" "/opt/rocm/lib")
        if(EXISTS "${_rocm_lib_dir}/libamdhip64.so")
          string(APPEND _hipvs_rpath_link " -Wl,-rpath-link,${_rocm_lib_dir}")
        endif()
      endforeach()
      set(_hipvs_probe_ok FALSE)
      foreach(_hipvs_flags IN ITEMS
              ""
              "-D__HIP_PLATFORM_AMD__"
              "-Wl,--allow-shlib-undefined"
              "-D__HIP_PLATFORM_AMD__ -Wl,--allow-shlib-undefined")
        set(CMAKE_REQUIRED_FLAGS "${_hipvs_flags} ${_hipvs_rpath_link}")
        unset(SEEKDB_HIPVS_API_PROBE_RESULT CACHE)
        check_cxx_source_compiles("#include \"${_hipvs_probe_src}\"" SEEKDB_HIPVS_API_PROBE_RESULT)
        if(SEEKDB_HIPVS_API_PROBE_RESULT)
          set(_hipvs_probe_ok TRUE)
          break()
        endif()
      endforeach()
      set(CMAKE_REQUIRED_INCLUDES "${_hipvs_required_includes}")
      set(CMAKE_REQUIRED_LIBRARIES "${_hipvs_required_libraries}")
      set(CMAKE_REQUIRED_FLAGS "${_hipvs_required_flags}")
      set(CMAKE_REQUIRED_QUIET "${_hipvs_required_quiet}")

      if(NOT _hipvs_probe_ok)
        set(_skip_reason "installed hipVS does not provide the API this source uses")
      else()
        if(NOT TARGET cuvs::c_api)
          add_library(cuvs::c_api UNKNOWN IMPORTED)
          set_target_properties(cuvs::c_api PROPERTIES
            IMPORTED_LOCATION "${_cuvs_lib}"
            INTERFACE_INCLUDE_DIRECTORIES "${_cuvs_inc}")
        endif()
        target_compile_definitions(${target} PRIVATE SEEKDB_WITH_HIPVS=1)
        target_link_libraries(${target} PRIVATE cuvs::c_api)
        # A server that cannot find libcuvs_c.so (or the HIP runtime it links)
        # at load time never starts, so the runtime search path is baked in
        # instead of relying on the caller's LD_LIBRARY_PATH.
        get_filename_component(_cuvs_lib_dir "${_cuvs_lib}" DIRECTORY)
        target_link_options(${target} INTERFACE
          "LINKER:-rpath,${_cuvs_lib_dir}"
          # The same paths must work at link time, when libcuvs_c.so's own
          # DT_NEEDED entries (libamdhip64.so and friends) are otherwise not
          # found outside the default search path. Allow those entries to
          # stay unresolved in the linked binary: the rpath above resolves
          # them at load time.
          "LINKER:-rpath-link,${_cuvs_lib_dir}"
          "LINKER:--allow-shlib-undefined")
        foreach(_rocm IN ITEMS "$ENV{ROCM_PATH}" /opt/rocm)
          if(EXISTS "${_rocm}/lib/libamdhip64.so")
            target_link_options(${target} INTERFACE
              "LINKER:-rpath,${_rocm}/lib"
              "LINKER:-rpath-link,${_rocm}/lib")
          endif()
        endforeach()
        message(STATUS "hipVS HNSW backend: enabled (${_cuvs_lib})")
        return()
      endif()
    endif()
  endif()

  if(_mode STREQUAL "ON")
    message(WARNING "SEEKDB_ENABLE_HIPVS=ON but ${_skip_reason}; building the VSAG CPU path")
  else()
    message(STATUS "hipVS HNSW backend: disabled (${_skip_reason})")
  endif()
endfunction()
