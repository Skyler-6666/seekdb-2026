/*
 * Copyright (c) 2026 OceanBase.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file seekdb_hipvs_loader_test.cpp
 *
 * Standalone regression test for the optional hipVS bridge loader.
 *
 * The test drives the loader against mock bridges that reproduce every
 * failure mode the loader has to survive:
 *
 *   1. a healthy bridge          -> loads, reports a version and a GPU
 *   2. a bridge with a wrong ABI  -> refused with an explicit error
 *   3. a bridge missing a symbol  -> refused with an explicit error
 *   4. a bridge without a GPU     -> loads, but the backend stays unusable
 *   5. a path that does not exist -> refused, no crash
 *
 * It also exercises a full build -> search -> serialize -> deserialize round
 * trip through the resolved ABI, so that a mistake in the ABI declarations is
 * caught without a GPU.
 *
 * Build and run it with `make -C unittest/vector/hipvs test`.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "lib/vector/hipvs/seekdb_hipvs_loader.h"

using oceanbase::common::obhipvs::HipvsLoader;

namespace {

int g_failures = 0;

void check(bool condition, const char *what)
{
  if (condition) {
    printf("  ok   - %s\n", what);
  } else {
    printf("  FAIL - %s\n", what);
    ++g_failures;
  }
}

std::string mock_path(const char *variant)
{
  const char *dir = getenv("SEEKDB_HIPVS_TEST_MOCK_DIR");
  const char *prefix = getenv("SEEKDB_HIPVS_TEST_MOCK_PREFIX");
  const char *suffix = getenv("SEEKDB_HIPVS_TEST_MOCK_SUFFIX");
  std::string result = dir == nullptr ? "." : dir;
  result += "/";
  result += prefix == nullptr ? "libmock_hipvs_bridge" : prefix;
  result += variant;
  result += suffix == nullptr ? ".so" : suffix;
  return result;
}

void test_healthy_bridge()
{
  printf("[case] healthy bridge\n");
  HipvsLoader loader;
  check(loader.load(mock_path("good")), "load succeeds");
  check(loader.is_loaded(), "loader reports the bridge as loaded");
  check(loader.get_bridge() != nullptr, "the resolved bridge is exposed");
  check(loader.has_gpu(), "a GPU is reported");
  check(!loader.get_version().empty(), "the hipVS version is reported");
  check(loader.get_error().empty(), "no error is reported");

  const oceanbase::common::obhipvs::HipvsBridge *bridge = loader.get_bridge();
  check(bridge->abi_version != nullptr && bridge->index_build != nullptr &&
            bridge->index_search != nullptr && bridge->index_serialize != nullptr &&
            bridge->index_deserialize != nullptr && bridge->buffer_free != nullptr,
        "every entry point of the ABI is resolved");
}

void test_abi_mismatch()
{
  printf("[case] bridge with an incompatible ABI\n");
  HipvsLoader loader;
  check(!loader.load(mock_path("badabi")), "load is refused");
  check(!loader.is_loaded(), "the loader does not claim to be loaded");
  check(loader.get_bridge() == nullptr, "no bridge is exposed");
  check(loader.get_error().find("ABI") != std::string::npos, "the error mentions the ABI mismatch");
}

void test_missing_symbol()
{
  printf("[case] bridge missing a symbol\n");
  HipvsLoader loader;
  check(!loader.load(mock_path("nosym")), "load is refused");
  check(!loader.is_loaded(), "the loader does not claim to be loaded");
  check(loader.get_error().find("seekdb_hipvs_index_build") != std::string::npos,
        "the error names the missing symbol");
}

void test_no_gpu()
{
  printf("[case] bridge without a GPU\n");
  HipvsLoader loader;
  check(loader.load(mock_path("nogpu")), "the bridge itself loads");
  check(!loader.has_gpu(), "but the backend stays unusable");
  check(!loader.get_error().empty(), "and the reason is reported");
}

void test_missing_file()
{
  printf("[case] bridge path that does not exist\n");
  HipvsLoader loader;
  check(!loader.load("/this/path/does/not/exist/libseekdb_hipvs_bridge.so"), "load is refused");
  check(!loader.is_loaded(), "the loader does not claim to be loaded");
  check(!loader.get_error().empty(), "the reason is reported");
}

void test_build_search_roundtrip()
{
  printf("[case] build -> search -> serialize -> deserialize\n");
  HipvsLoader loader;
  if (!loader.load(mock_path("good"))) {
    check(false, "the mock bridge is available");
    return;
  }
  const oceanbase::common::obhipvs::HipvsBridge *bridge = loader.get_bridge();

  seekdb_hipvs_index_params_t params;
  memset(&params, 0, sizeof(params));
  params.metric = SEEKDB_HIPVS_METRIC_L2;
  params.graph_algo = SEEKDB_HIPVS_GRAPH_ALGO_CAGRA;
  params.hierarchy = SEEKDB_HIPVS_HIERARCHY_CPU;
  params.dim = 4;
  params.m = 16;
  params.ef_construction = 200;
  params.ef_search = 64;

  void *index = nullptr;
  check(bridge->index_create(&params, &index) == SEEKDB_HIPVS_OK && index != nullptr,
        "an index can be created");

  const int64_t count = 3;
  const float vectors[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
  const int64_t ids[3] = {10, 20, 30};
  check(bridge->index_build(index, vectors, 4, count, ids) == SEEKDB_HIPVS_OK, "the index can be built");

  int64_t num_elements = 0;
  check(bridge->index_num_elements(index, &num_elements) == SEEKDB_HIPVS_OK && num_elements == count,
        "the index reports the number of elements");

  int64_t found[3] = {0, 0, 0};
  float distances[3] = {0, 0, 0};
  int64_t result_size = 0;
  const float query[4] = {1, 0, 0, 0};
  check(bridge->index_search(index, query, 4, 3, 64, found, distances, &result_size) == SEEKDB_HIPVS_OK &&
            result_size == 3,
        "the index can be searched");
  check(found[0] == 10 && found[1] == 20 && found[2] == 30,
        "the search answers with the external ids, not with the ordinals");

  int64_t min_vid = 0;
  int64_t max_vid = 0;
  check(bridge->index_vid_bound(index, &min_vid, &max_vid) == SEEKDB_HIPVS_OK && min_vid == 10 &&
            max_vid == 30,
        "the index reports the vid bounds");

  char *buffer = nullptr;
  int64_t buffer_len = 0;
  check(bridge->index_serialize(index, &buffer, &buffer_len) == SEEKDB_HIPVS_OK && buffer != nullptr &&
            buffer_len > 0,
        "the index can be serialized");

  void *restored = nullptr;
  check(bridge->index_deserialize(buffer, buffer_len, &params, &restored) == SEEKDB_HIPVS_OK &&
            restored != nullptr,
        "the index can be deserialized");
  int64_t restored_count = 0;
  check(bridge->index_num_elements(restored, &restored_count) == SEEKDB_HIPVS_OK &&
            restored_count == count,
        "the restored index reports the same number of elements");
  check(bridge->index_vid_bound(restored, &min_vid, &max_vid) == SEEKDB_HIPVS_OK && min_vid == 10 &&
            max_vid == 30,
        "the restored index reports the same vid bounds");

  bridge->buffer_free(buffer);
  bridge->index_destroy(restored);
  bridge->index_destroy(index);
}

} // namespace

int main()
{
  printf("== seekdb hipVS loader test ==\n");
  test_healthy_bridge();
  test_abi_mismatch();
  test_missing_symbol();
  test_no_gpu();
  test_missing_file();
  test_build_search_roundtrip();
  if (g_failures == 0) {
    printf("== all checks passed ==\n");
    return 0;
  }
  printf("== %d check(s) failed ==\n", g_failures);
  return 1;
}
