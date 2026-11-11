#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <unistd.h>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

struct Candidate {
  float distance;
  uint32_t position;
};

// Smaller values are better. A max-heap built with this comparator keeps
// the worst candidate at the root, which makes bounded TopK maintenance cheap.
struct CandidateLess {
  bool operator()(const Candidate& a, const Candidate& b) const {
    if (a.distance < b.distance) return true;
    if (a.distance > b.distance) return false;
    return a.position < b.position;
  }
};

static inline float l2_distance(const float* row, const float* query, size_t dim) {
#if defined(__AVX2__)
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();
  size_t d = 0;
  for (; d + 32 <= dim; d += 32) {
    const __m256 d0 = _mm256_sub_ps(
        _mm256_loadu_ps(row + d), _mm256_loadu_ps(query + d));
    const __m256 d1 = _mm256_sub_ps(
        _mm256_loadu_ps(row + d + 8), _mm256_loadu_ps(query + d + 8));
    const __m256 d2 = _mm256_sub_ps(
        _mm256_loadu_ps(row + d + 16), _mm256_loadu_ps(query + d + 16));
    const __m256 d3 = _mm256_sub_ps(
        _mm256_loadu_ps(row + d + 24), _mm256_loadu_ps(query + d + 24));
#if defined(__FMA__)
    acc0 = _mm256_fmadd_ps(d0, d0, acc0);
    acc1 = _mm256_fmadd_ps(d1, d1, acc1);
    acc2 = _mm256_fmadd_ps(d2, d2, acc2);
    acc3 = _mm256_fmadd_ps(d3, d3, acc3);
#else
    acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(d0, d0));
    acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(d1, d1));
    acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(d2, d2));
    acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(d3, d3));
#endif
  }
  __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1),
                             _mm256_add_ps(acc2, acc3));
  for (; d + 8 <= dim; d += 8) {
    const __m256 diff = _mm256_sub_ps(
        _mm256_loadu_ps(row + d), _mm256_loadu_ps(query + d));
#if defined(__FMA__)
    acc = _mm256_fmadd_ps(diff, diff, acc);
#else
    acc = _mm256_add_ps(acc, _mm256_mul_ps(diff, diff));
#endif
  }
  __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(acc),
                             _mm256_extractf128_ps(acc, 1));
  sum128 = _mm_hadd_ps(sum128, sum128);
  sum128 = _mm_hadd_ps(sum128, sum128);
  float sum = _mm_cvtss_f32(sum128);
  for (; d < dim; ++d) {
    const float delta = row[d] - query[d];
    sum += delta * delta;
  }
  return std::sqrt(sum);
#else
  float sum = 0.0f;
  for (size_t d = 0; d < dim; ++d) {
    const float delta = row[d] - query[d];
    sum += delta * delta;
  }
  return std::sqrt(sum);
#endif
}

extern "C" int mvd_has_bad_row_separator(const char* text, size_t length) {
  if (!text || length < 3) return 0;
  char quote = 0;
  for (size_t i = 0; i < length; ++i) {
    const char ch = text[i];
    if (quote) {
      if (ch == '\\' && i + 1 < length) {
        ++i;
        continue;
      }
      if (ch == quote) {
        if (i + 1 < length && text[i + 1] == quote) {
          ++i;
          continue;
        }
        quote = 0;
        continue;
      }
      // A parenthesis inside a SQL string can make a regex row split unsafe.
      if (ch == '(' || ch == ')') return 1;
      continue;
    }
    if (ch == '\'' || ch == '"') quote = ch;
  }
  return 0;
}

extern "C" int mvd_parse_floats(
    const char* text,
    size_t length,
    float* output,
    size_t capacity) {
  if (!text || !output) return -1;
  const char* p = text;
  const char* end = text + length;
  size_t count = 0;
  while (p < end) {
    while (p < end && (*p == ',' || *p == ' ' || *p == '\t' ||
                       *p == '\n' || *p == '\r')) {
      ++p;
    }
    if (p == end) break;
    if (count >= capacity) return -1;
    bool negative = false;
    if (*p == '+' || *p == '-') {
      negative = (*p == '-');
      ++p;
    }
    double value = 0.0;
    bool any_digit = false;
    while (p < end && *p >= '0' && *p <= '9') {
      value = value * 10.0 + static_cast<double>(*p - '0');
      any_digit = true;
      ++p;
    }
    if (p < end && *p == '.') {
      ++p;
      double scale = 0.1;
      while (p < end && *p >= '0' && *p <= '9') {
        value += static_cast<double>(*p - '0') * scale;
        scale *= 0.1;
        any_digit = true;
        ++p;
      }
    }
    if (!any_digit) return -1;
    if (p < end && (*p == 'e' || *p == 'E')) {
      ++p;
      bool exponent_negative = false;
      if (p < end && (*p == '+' || *p == '-')) {
        exponent_negative = (*p == '-');
        ++p;
      }
      int exponent = 0;
      bool exponent_digit = false;
      while (p < end && *p >= '0' && *p <= '9') {
        exponent = exponent * 10 + (*p - '0');
        exponent_digit = true;
        ++p;
      }
      if (!exponent_digit) return -1;
      value *= std::pow(10.0, exponent_negative ? -exponent : exponent);
    }
    if (p < end && *p != ',' && *p != ' ' && *p != '\t' &&
        *p != '\n' && *p != '\r') {
      return -1;
    }
    output[count++] = static_cast<float>(negative ? -value : value);
  }
  return static_cast<int>(count);
}

static int configured_thread_count(size_t rows) {
  if (rows <= 1) return 1;
  const char* env = std::getenv("MVD_THREADS");
  int count = 0;
  if (env && *env) {
    char* end = nullptr;
    long parsed = std::strtol(env, &end, 10);
    if (end && *end == '\0' && parsed > 0) count = static_cast<int>(parsed);
  }
  if (count <= 0) {
    // Docker enforces CPU limits through cgroup v2. hardware_concurrency()
    // reports the host's cores here, so prefer the actual cgroup quota.
    std::FILE* file = std::fopen("/sys/fs/cgroup/cpu.max", "r");
    if (file) {
      char quota[64] = {0};
      long long period = 0;
      if (std::fscanf(file, "%63s %lld", quota, &period) == 2 &&
          std::strcmp(quota, "max") != 0 && period > 0) {
        long long value = std::atoll(quota);
        if (value > 0) {
          long long rounded = (value + period - 1) / period;
          count = static_cast<int>(std::max(1LL, rounded));
        }
      }
      std::fclose(file);
    }
  }
  if (count <= 0) {
    const long online = sysconf(_SC_NPROCESSORS_ONLN);
    count = online > 0 ? static_cast<int>(online) : 1;
  }
  if (count <= 0) count = 1;
  count = std::min<int>(count, 16);
  count = std::min<int>(count, static_cast<int>(rows));
  return count;
}

static void scan_range(
    const float* base,
    size_t n,
    size_t dim,
    const float* query,
    const uint32_t* allowed,
    size_t n_allowed,
    size_t begin,
    size_t end,
    size_t k,
    std::vector<Candidate>& heap) {
  heap.reserve(k);
  CandidateLess less;
  for (size_t j = begin; j < end; ++j) {
    const uint32_t position = allowed ? allowed[j] : static_cast<uint32_t>(j);
    if (position >= n) continue;
    const Candidate candidate{
        l2_distance(base + static_cast<size_t>(position) * dim, query, dim),
        position};
    if (heap.size() < k) {
      heap.push_back(candidate);
      std::push_heap(heap.begin(), heap.end(), less);
    } else if (less(candidate, heap.front())) {
      std::pop_heap(heap.begin(), heap.end(), less);
      heap.back() = candidate;
      std::push_heap(heap.begin(), heap.end(), less);
    }
  }
}

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
  const size_t take = std::min(k, count);
  const int thread_count = configured_thread_count(count);
  std::vector<std::vector<Candidate>> heaps(static_cast<size_t>(thread_count));

#pragma omp parallel num_threads(thread_count)
  {
    const int thread_id = omp_get_thread_num();
    const int active_threads = omp_get_num_threads();
    const size_t begin = count * static_cast<size_t>(thread_id) /
                         static_cast<size_t>(active_threads);
    const size_t end = count * static_cast<size_t>(thread_id + 1) /
                       static_cast<size_t>(active_threads);
    scan_range(base, n, dim, query, allowed, n_allowed, begin, end, take,
               heaps[static_cast<size_t>(thread_id)]);
  }

  std::vector<Candidate> merged;
  merged.reserve(take * static_cast<size_t>(thread_count));
  for (const auto& heap : heaps) {
    merged.insert(merged.end(), heap.begin(), heap.end());
  }
  std::sort(merged.begin(), merged.end(), CandidateLess());
  const size_t result_count = std::min(take, merged.size());
  for (size_t i = 0; i < result_count; ++i) {
    out_positions[i] = merged[i].position;
    out_distances[i] = merged[i].distance;
  }
  return static_cast<int>(result_count);
}
