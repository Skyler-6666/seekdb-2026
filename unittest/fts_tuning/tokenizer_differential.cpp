/* Copyright (c) 2026 OceanBase.
 * SPDX-License-Identifier: Apache-2.0
 * Compile through tokenizer_differential.py: it supplies the actual old/new
 * production headers and actual Unicode predicates in separate translation
 * units. Collation dispatch/logging are boundary glue, not a server build.
 */
#include "tokenizer_classes.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#ifdef _MSC_VER
#define TEST_NOINLINE __declspec(noinline)
#else
#define TEST_NOINLINE __attribute__((noinline))
#endif

using namespace oceanbase::common;
using oceanbase::storage::LegacyFTCharUtil;
using oceanbase::storage::ObFTCharUtil;

struct Result { int ret; int type; };
struct Input { std::array<char, 4> bytes; uint8_t length; };

TEST_NOINLINE Result legacy(ObCharsetType charset, const char *p, uint8_t size)
{
  auto type = LegacyFTCharUtil::CharType::SURROGATE_LOW;
  int ret = LegacyFTCharUtil::classify_first_char(charset, p, size, type);
  return {ret, static_cast<int>(type)};
}

TEST_NOINLINE Result candidate(ObCharsetType charset, const char *p, uint8_t size)
{
  auto type = ObFTCharUtil::CharType::SURROGATE_LOW;
  int ret = ObFTCharUtil::classify_first_char(charset, p, size, type);
  return {ret, static_cast<int>(type)};
}

uint64_t checks = 0;
void compare(ObCharsetType charset, const char *p, uint8_t size)
{
  const auto a = legacy(charset, p, size);
  const auto b = candidate(charset, p, size);
  ++checks;
  if (a.ret != b.ret || a.type != b.type) {
    std::cerr << "Mismatch charset=" << charset << " length=" << int(size)
              << " legacy=" << a.ret << "," << a.type << " candidate=" << b.ret << "," << b.type << '\n';
    std::exit(1);
  }
}

Input encode(uint32_t cp)
{
  Input input{};
  int n = ob_charset_encode_unicode<CHARSET_UTF8MB4>(cp,
      reinterpret_cast<unsigned char *>(input.bytes.data()),
      reinterpret_cast<unsigned char *>(input.bytes.data()) + input.bytes.size());
  if (n < 1 || n > 4) std::abort();
  input.length = static_cast<uint8_t>(n);
  return input;
}

void regression()
{
  // Every scalar, plus surrogate encodings: preserve legacy invalid-input
  // behavior, including its OB_CS_ILSEQ == 0 handling, without repairing it.
  for (uint32_t cp = 0; cp <= 0x10ffff; ++cp) {
    const Input input = encode(cp);
    for (uint8_t n = 0; n <= input.length; ++n) {
      compare(CHARSET_UTF8MB4, input.bytes.data(), n);
    }
  }
  // Complete 1-byte/2-byte input space; generic UTF16 and UTF16LE decoder
  // behavior must remain equivalent to the baseline, even for odd lengths.
  for (uint32_t raw = 0; raw <= 0xffff; ++raw) {
    const char bytes[] = {static_cast<char>(raw >> 8), static_cast<char>(raw), 0, 0};
    for (auto charset : {CHARSET_UTF8MB4, CHARSET_UTF16, CHARSET_UTF16LE}) {
      for (uint8_t n = 0; n <= 2; ++n) compare(charset, bytes, n);
    }
  }
  // Exhaust all malformed/valid 3-byte prefixes in the E0..EF lead range.
  for (uint32_t raw = 0; raw < 0x100000; ++raw) {
    const char bytes[] = {static_cast<char>(0xe0 | (raw >> 16)),
                          static_cast<char>(raw >> 8), static_cast<char>(raw), 0};
    compare(CHARSET_UTF8MB4, bytes, 3);
  }
  // Four-byte boundaries and random bytes (including out-of-range leads).
  uint32_t rng = 0x7a49ed03;
  for (int i = 0; i < 1000000; ++i) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    const char bytes[] = {static_cast<char>(rng), static_cast<char>(rng >> 8),
                          static_cast<char>(rng >> 16), static_cast<char>(rng >> 24)};
    compare(CHARSET_UTF8MB4, bytes, 4);
    if (i < 65536) {
      compare(CHARSET_UTF16, bytes, 4);
      compare(CHARSET_UTF16LE, bytes, 4);
    }
  }
  const uint32_t edges[] = {0, 0x7f, 0x80, 0x7ff, 0x800, 0xd7ff, 0xd800,
    0xdbff, 0xdc00, 0xdfff, 0xe000, 0xffff, 0x10000, 0x10ffff, 0x110000, 0x1fffff};
  for (uint32_t cp : edges) {
    auto input = encode(cp);
    for (uint8_t n = 0; n <= 4; ++n) compare(CHARSET_UTF8MB4, input.bytes.data(), n);
  }
  for (auto charset : {CHARSET_UTF8MB4, CHARSET_UTF16, CHARSET_UTF16LE, CHARSET_BINARY, CHARSET_GBK, CHARSET_INVALID}) {
    for (uint8_t n = 0; n <= 4; ++n) {
      compare(charset, nullptr, n);
      compare(charset, "ABCD", n);
    }
  }
  // Assert semantic category precedence explicitly, independently of old/new.
  const std::pair<uint32_t, int> expected[] = {
    {'A', 2}, {'z', 2}, {'0', 1}, {0xff21, 2}, {0xff41, 2}, {0xff10, 1},
    {0x4e00, 3}, {0xf900, 3}, {0x3400, 3}, {0x3042, 4}, {0xac00, 4},
    {0x3000, 0}, {0xff00, 4}, {0xffef, 4}, {0x1f600, 0}, {0xd800, 0}};
  for (const auto &entry : expected) {
    const Input input = encode(entry.first);
    const auto actual = candidate(CHARSET_UTF8MB4, input.bytes.data(), input.length);
    if (actual.ret != OB_SUCCESS || actual.type != entry.second) {
      std::cerr << "Unexpected category for U+" << std::hex << entry.first << '\n';
      std::exit(1);
    }
  }
}

template<bool OPTIMIZED>
double run_benchmark(const std::vector<Input> &inputs, uint64_t &checksum)
{
  const auto begin = std::chrono::steady_clock::now();
  uint64_t sum = 0;
  for (int repeat = 0; repeat < 3000; ++repeat) {
    for (const auto &input : inputs) {
      const auto value = OPTIMIZED ? candidate(CHARSET_UTF8MB4, input.bytes.data(), input.length)
                                   : legacy(CHARSET_UTF8MB4, input.bytes.data(), input.length);
      sum += static_cast<uint64_t>(value.type + value.ret);
    }
  }
  checksum = sum;
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
}

int main(int argc, char **argv)
{
  regression();
  std::cout << "{\"status\":\"passed\",\"differential_cases\":" << checks;
  if (argc > 1) {
    // Synthetic deterministic mixture, not the official workload or score.
    const uint32_t points[] = {'A', 'b', '7', ' ', 0x4e00, 0x56fd, 0x6c34,
      0x3042, 0x30a2, 0xac00, 0xff21, 0xff10, 0x1f600, 0x3002, 0x1100, 0x3400};
    std::vector<Input> inputs;
    for (int i = 0; i < 4096; ++i) inputs.push_back(encode(points[(i * 13) % 16]));
    std::vector<double> old_times, new_times;
    uint64_t old_sum = 0, new_sum = 0;
    std::cout << ",\"unofficial_microbenchmark\":{\"calls_per_sample\":" << inputs.size() * 3000
              << ",\"samples\":[";
    for (int round = 0; round < 5; ++round) {
      double a, b;
      if (round % 2 == 0) { a = run_benchmark<false>(inputs, old_sum); b = run_benchmark<true>(inputs, new_sum); }
      else { b = run_benchmark<true>(inputs, new_sum); a = run_benchmark<false>(inputs, old_sum); }
      if (old_sum != new_sum) return 2;
      old_times.push_back(a); new_times.push_back(b);
      if (round) std::cout << ',';
      std::cout << "{\"baseline_ms\":" << a << ",\"candidate_ms\":" << b << '}';
    }
    std::sort(old_times.begin(), old_times.end());
    std::sort(new_times.begin(), new_times.end());
    std::cout << "],\"checksum\":" << old_sum << ",\"baseline_median_ms\":" << old_times[2]
              << ",\"candidate_median_ms\":" << new_times[2]
              << ",\"ratio_baseline_over_candidate\":" << old_times[2] / new_times[2] << '}';
  }
  std::cout << "}\n";
}
