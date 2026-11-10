#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <vector>

struct Candidate {
  float distance;
  uint32_t position;
};

extern "C" int mvd_topk_l2(
    const float* base,
    size_t n,
    size_t dim,
    const float* query,
    const uint32_t* allowed,
    size_t n_allowed,
    size_t k,
    uint32_t* out_positions,
    float* out_distances) {
  if (!base || !query || !out_positions || !out_distances || dim == 0) return -1;
  const size_t count = allowed ? n_allowed : n;
  if (count == 0 || k == 0) return 0;
  std::vector<Candidate> candidates;
  candidates.reserve(count);
  for (size_t j = 0; j < count; ++j) {
    const uint32_t pos = allowed ? allowed[j] : static_cast<uint32_t>(j);
    if (pos >= n) continue;
    const float* row = base + static_cast<size_t>(pos) * dim;
    float sum = 0.0f;
    for (size_t d = 0; d < dim; ++d) {
      const float delta = row[d] - query[d];
      sum += delta * delta;
    }
    candidates.push_back(Candidate{std::sqrt(sum), pos});
  }
  const size_t take = std::min(k, candidates.size());
  std::partial_sort(
      candidates.begin(), candidates.begin() + take, candidates.end(),
      [](const Candidate& a, const Candidate& b) {
        if (a.distance < b.distance) return true;
        if (a.distance > b.distance) return false;
        return a.position < b.position;
      });
  for (size_t i = 0; i < take; ++i) {
    out_positions[i] = candidates[i].position;
    out_distances[i] = candidates[i].distance;
  }
  return static_cast<int>(take);
}
