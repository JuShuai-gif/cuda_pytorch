#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace lab {
using Clock = std::chrono::steady_clock;
inline volatile double sink = 0.0;

template <class F> double time_ms(F&& fn) {
  const auto begin = Clock::now();
  fn();
  return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}

inline double percentile(std::vector<double> values, double p) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const double pos = (values.size() - 1) * p / 100.0;
  const auto lo = static_cast<std::size_t>(pos);
  const auto hi = std::min(lo + 1, values.size() - 1);
  return values[lo] + (values[hi] - values[lo]) * (pos - lo);
}

inline void print_stats(const std::string& name, const std::vector<double>& v) {
  const double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
  double variance = 0.0;
  for (double x : v) variance += (x - mean) * (x - mean);
  variance /= v.size();
  std::cout << std::fixed << std::setprecision(3) << name
            << " mean_ms=" << mean << " median_ms=" << percentile(v, 50)
            << " p50_ms=" << percentile(v, 50) << " p90_ms=" << percentile(v, 90)
            << " p95_ms=" << percentile(v, 95) << " p99_ms=" << percentile(v, 99)
            << " min_ms=" << *std::min_element(v.begin(), v.end())
            << " max_ms=" << *std::max_element(v.begin(), v.end())
            << " stddev_ms=" << std::sqrt(variance) << '\n';
}

template <class F> std::vector<double> benchmark(F&& fn, int warmup, int iterations) {
  for (int i = 0; i < warmup; ++i) sink += static_cast<double>(fn());
  std::vector<double> samples;
  samples.reserve(iterations);
  for (int i = 0; i < iterations; ++i) {
    samples.push_back(time_ms([&] { sink += static_cast<double>(fn()); }));
  }
  return samples;
}
}  // namespace lab
