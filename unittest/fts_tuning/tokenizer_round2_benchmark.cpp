// Isolated production-classifier microbenchmark, not a database performance score.
// Reuse the original source-extracted baseline/current wrappers and encoder.
#include "tokenizer_round2_support.h"

void measure(const char *name, const std::vector<Input> &inputs)
{
  std::vector<double> old_times, new_times;
  uint64_t old_sum = 0, new_sum = 0;
  std::cout << '"' << name << "\":{\"calls_per_sample\":" << inputs.size() * 3000
            << ",\"samples\":[";
  for (int round = 0; round < 5; ++round) {
    double old_time, new_time;
    if (round % 2 == 0) {
      old_time = run_benchmark<false>(inputs, old_sum);
      new_time = run_benchmark<true>(inputs, new_sum);
    } else {
      new_time = run_benchmark<true>(inputs, new_sum);
      old_time = run_benchmark<false>(inputs, old_sum);
    }
    if (old_sum != new_sum) {
      std::cerr << "Checksum differs for " << name << '\n';
      std::exit(2);
    }
    old_times.push_back(old_time);
    new_times.push_back(new_time);
    if (round != 0) std::cout << ',';
    std::cout << "{\"baseline_ms\":" << old_time << ",\"candidate_ms\":" << new_time << '}';
  }
  std::sort(old_times.begin(), old_times.end());
  std::sort(new_times.begin(), new_times.end());
  std::cout << "],\"baseline_median_ms\":" << old_times[2]
            << ",\"candidate_median_ms\":" << new_times[2]
            << ",\"ratio_baseline_over_candidate\":" << old_times[2] / new_times[2]
            << ",\"checksum\":" << old_sum << '}';
}

int main()
{
  std::vector<Input> ascii, chinese, fullwidth, fullwidth_letters, mixed;
  const uint32_t mix[] = {'A', 'b', '7', ' ', 0x4e00, 0x56fd, 0x6c34,
    0x3042, 0x30a2, 0xac00, 0xff21, 0xff10, 0x1f600, 0x3002, 0x1100, 0x3400};
  for (uint32_t i = 0; i < 4096; ++i) {
    ascii.push_back(encode((i * 53) % 128));
    chinese.push_back(encode(0x4e00 + (i * 997) % (0x9fff - 0x4e00 + 1)));
    fullwidth.push_back(encode(0xff00 + (i * 13) % 0xf0));
    fullwidth_letters.push_back(encode((i % 3 == 0 ? 0xff10 : (i % 3 == 1 ? 0xff21 : 0xff41)) + i % 10));
    mixed.push_back(encode(mix[(i * 13) % 16]));
  }
  std::cout << "{\"scope\":\"unofficial isolated character classifier; no server or official score\",\"datasets\":{";
  measure("ascii_all_128", ascii);
  std::cout << ',';
  measure("chinese_basic", chinese);
  std::cout << ',';
  measure("fullwidth_block", fullwidth);
  std::cout << ',';
  measure("fullwidth_letters_digits", fullwidth_letters);
  std::cout << ',';
  measure("mixed_scripts", mixed);
  std::cout << "}}\n";
}
