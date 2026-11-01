cmake_minimum_required(VERSION 3.20)
project(seekdb_hipvs_option_check C CXX)

# Exercises cmake/SeekdbHipvs.cmake without building seekdb: the function under
# test only decides whether to add SEEKDB_WITH_HIPVS and cuvs::c_api to a target.
if(NOT SEEKDB_REPO_ROOT)
  message(FATAL_ERROR "SEEKDB_REPO_ROOT must point at the repository root")
endif()
include("${SEEKDB_REPO_ROOT}/cmake/SeekdbHipvs.cmake")

add_library(check STATIC dummy.c)
seekdb_configure_hipvs(check)

get_target_property(_defs check COMPILE_DEFINITIONS)
get_target_property(_links check LINK_LIBRARIES)
get_target_property(_options check INTERFACE_LINK_OPTIONS)
message(STATUS "CHECK_DEFS=${_defs}")
message(STATUS "CHECK_LINKS=${_links}")
message(STATUS "CHECK_OPTIONS=${_options}")
