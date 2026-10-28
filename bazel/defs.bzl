"""Stable public entry point for seekdb's first-party Bazel rules.

BUILD files load project rules from here. The implementation files remain free
to evolve without spreading their layout or standard policy across packages.
"""

load("@rules_cc//cc:cc_import.bzl", _cc_import = "cc_import")
load("//bazel:platforms.bzl", _openmp_copts = "openmp_copts")
load(
    "//bazel:compile_config.bzl",
    _C_COPTS = "C_COPTS",
    _CXX_COPTS = "CXX_COPTS",
    _OBLIB_CXX_COPTS = "OBLIB_CXX_COPTS",
    _OBLIB_C_COPTS = "OBLIB_C_COPTS",
    _OBLIB_LOCAL_DEFINES = "OBLIB_LOCAL_DEFINES",
    _MODULE_CXX_COPTS = "MODULE_CXX_COPTS",
    _X86_SIMD_COPTS = "X86_SIMD_COPTS",
)
load(
    "//bazel:cc_library.bzl",
    _cc_library = "cc_library",
    _unity_cc_library = "unity_cc_library",
    _unity_source = "unity_source",
    _grouped_unity_cc_library = "grouped_unity_cc_library",
)
load("//bazel:final_link.bzl", _final_link = "final_link")
load("//bazel:localized_partial_link.bzl", _localized_partial_link = "localized_partial_link")
load("//bazel:version.bzl", _version_source = "version_source")
load("//bazel:test_interface.bzl", _seekdb_test_interface = "seekdb_test_interface")

C_COPTS = _C_COPTS
CXX_COPTS = _CXX_COPTS
OBLIB_CXX_COPTS = _OBLIB_CXX_COPTS
OBLIB_C_COPTS = _OBLIB_C_COPTS
OBLIB_LOCAL_DEFINES = _OBLIB_LOCAL_DEFINES
MODULE_CXX_COPTS = _MODULE_CXX_COPTS
X86_SIMD_COPTS = _X86_SIMD_COPTS
openmp_copts = _openmp_copts

cc_import = _cc_import
cc_library = _cc_library
final_link = _final_link
unity_cc_library = _unity_cc_library
unity_source = _unity_source
localized_partial_link = _localized_partial_link
grouped_unity_cc_library = _grouped_unity_cc_library
version_source = _version_source
seekdb_test_interface = _seekdb_test_interface

def seekdb_module_cc_test(
    name,
    srcs,
    module_interface,
    test_hdrs = [],
    unity_size = 8,
    unity_exceptions = {},
    shard_count = 1,
    data = [],
    tags = [],
    visibility = None,
    **kwargs):
  """Builds a white-box module test against the server test runtime.

  Module tests link the shared oceanbase test runtime so the cases exercise
  the same C++ objects as the server without duplicating the final link.
  """
  if not srcs:
    fail("%s has no unit-test sources" % name)
  if unity_size <= 0:
    fail("unity_size must be positive")
  unknown_exceptions = [src for src in unity_exceptions if src not in srcs]
  if unknown_exceptions:
    fail("%s unity_exceptions are not present in srcs: %s" % (
        name, unknown_exceptions))
  missing_reasons = [
      src for src, reason in unity_exceptions.items()
      if not reason.strip()
  ]
  if missing_reasons:
    fail("%s unity_exceptions must explain isolation: %s" % (
        name, missing_reasons))

  header_target = "_" + name + "_headers"
  _cc_library(
      name = header_target,
      hdrs = test_hdrs,
      testonly = True,
      tags = tags,
      visibility = ["//visibility:private"],
  )

  unity_members = sorted([
      src for src in srcs
      if src not in unity_exceptions
  ])
  unity_groups = [
      unity_members[offset:offset + unity_size]
      for offset in range(0, len(unity_members), unity_size)
  ]
  unity_groups.extend([[src] for src in sorted(unity_exceptions)])

  unity_targets = []
  for index, members in enumerate(unity_groups):
    group_name = "_" + name + "_unity_" + str(index)
    member_target = group_name + "_members"
    _cc_library(
        name = member_target,
        textual_hdrs = members,
        testonly = True,
        tags = tags,
        visibility = ["//visibility:private"],
    )
    unity = _unity_source(
        name = group_name,
        unity_members = members,
        tags = tags,
    )
    _cc_library(
        name = group_name,
        srcs = [unity],
        alwayslink = True,
        deps = ["//bazel/third_party:gtest"],
        implementation_deps = [
            ":" + header_target,
            ":" + member_target,
            module_interface,
        ],
        testonly = True,
        tags = tags,
        visibility = ["//visibility:private"],
    )
    unity_targets.append(":" + group_name)

  main_target = "_" + name + "_main"
  _cc_library(
      name = main_target,
      srcs = ["//unittest:all_tests_main.cpp"],
      alwayslink = True,
      deps = ["//bazel/third_party:gtest"],
      testonly = True,
      tags = tags,
      visibility = ["//visibility:private"],
  )

  package = native.package_name()
  package_parent = "../" * len(package.split("/"))
  native.cc_test(
      name = name,
      data = data + ["//src/observer:liboceanbase"],
      deps = [":" + main_target] + unity_targets + [
          module_interface,
          "//bazel/third_party:gtest",
          "//src/observer:liboceanbase",
      ],
      features = ["static_link_cpp_runtimes"],
      linkstatic = True,
      linkopts = [
          "-pthread",
          "-ldl",
          "-Wl,-rpath,$ORIGIN/%ssrc/observer" % package_parent,
      ],
      shard_count = shard_count,
      tags = tags,
      visibility = visibility,
      **kwargs
  )
