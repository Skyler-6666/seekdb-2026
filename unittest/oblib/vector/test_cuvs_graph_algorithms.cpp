#include "../../../tools/hipvs_bridge/seekdb_cuvs_graph_algorithms.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdio>
#include <random>

using namespace seekdb_gpu_graph;

// Exercise the same compare-exchange schedule used by the HIP block sort.
static void test_sort()
{
  std::mt19937 generator(20261030);
  for (unsigned round = 0; round < 8; ++round) {
    std::array<uint64_t, candidate_capacity> keys, expected;
    for (unsigned i = 0; i < keys.size(); ++i) {
      keys[i] = i % 5 == 0 ? UINT64_MAX : generator() % (round + 1);
    }
    expected = keys;
    std::sort(expected.begin(), expected.end());
    for (unsigned size = 2; size <= keys.size(); size <<= 1) {
      for (unsigned stride = size >> 1; stride; stride >>= 1) {
        for (unsigned i = 0; i < keys.size(); ++i) {
          const unsigned partner = i ^ stride;
          if (partner > i) { exchange_keys(keys[i], keys[partner], (i & size) == 0); }
        }
      }
    }
    assert(keys == expected);
  }
}

static void test_merge()
{
  constexpr uint32_t rows = 1024;
  for (uint32_t row : {0U, 100U, rows - 1}) {
    for (unsigned mode = 0; mode < 4; ++mode) {
      std::array<uint32_t, graph_degree> graph, incoming, scratch;
      for (unsigned i = 0; i < graph_degree; ++i) {
        graph[i] = (row + 2 + i) % rows;
        incoming[i] = mode == 0 ? row : mode == 1 ? rows : (row + 200 + i / 2) % rows;
      }
      if (mode == 3) { graph.back() = row == 0 ? rows - 1 : row - 1; }
      const auto original = graph;
      assert(merge_reverse(graph.data(), incoming.data(), incoming.size(), row, rows, scratch.data()));
      for (unsigned i = 0; i < graph_degree; ++i) {
        assert(graph[i] < rows && graph[i] != row);
        assert(!contains(graph.data(), i, graph[i]));
        if (i < graph_degree / 2) { assert(graph[i] == original[i]); }
      }
      assert(contains(graph.data(), graph_degree, row + 1 == rows ? 0 : row + 1));
      assert(contains(graph.data(), graph_degree, row == 0 ? rows - 1 : row - 1));
    }
  }
  std::array<uint32_t, graph_degree> duplicate{}, scratch{};
  assert(!merge_reverse(duplicate.data(), nullptr, 0, 1, rows, scratch.data()));
}

int main()
{
  test_sort();
  test_merge();
  std::puts("PASS: sort ties/sentinels, reverse-edge merge, validity, ring preservation");
}
