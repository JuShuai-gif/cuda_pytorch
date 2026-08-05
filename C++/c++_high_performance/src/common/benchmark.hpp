#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "compiler_barrier.hpp"
#include "statistics.hpp"

namespace chp {

struct BenchmarkResult {
    double mean_ns = 0.0;
    double median_ns = 0.0;
    double min_ns = 0.0;
    double max_ns = 0.0;
    double stddev_ns = 0.0;
    std::size_t iterations = 0;
    std::uint64_t checksum = 0;
};

// Runs `fn` repeatedly and reports per-iteration cost in nanoseconds.
//
// Fn must be callable as `void(std::uint64_t& checksum)`. The callback
// accumulates its result into the checksum argument; this, together with
// compiler_barrier(), prevents the compiler from hoisting or eliminating the
// measured work. Both implementations being compared must be measured with
// the same callback signature so the bookkeeping overhead is identical.
//
// The reported time is the average over `rounds` measured rounds, each round
// executing `iterations` calls of `fn`. `warmup_rounds` rounds are executed
// before any measurement starts (caches, branch predictors, frequency
// scaling). The checksum of all executed calls is reported so callers can
// verify that both implementations actually produced identical results.
template <typename Fn>
BenchmarkResult benchmark(std::size_t iterations, std::size_t rounds,
                          std::size_t warmup_rounds, Fn&& fn) {
    BenchmarkResult res{};
    res.iterations = iterations;

    std::vector<double> samples;
    samples.reserve(rounds);

    std::uint64_t checksum = 0;
    const std::size_t total_rounds = rounds + warmup_rounds;
    for (std::size_t r = 0; r < total_rounds; ++r) {
        std::uint64_t acc = checksum;
        const auto t0 = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < iterations; ++i) {
            fn(acc);
            compiler_barrier();
        }
        const auto t1 = std::chrono::steady_clock::now();
        checksum = acc;

        if (r >= warmup_rounds) {
            const std::chrono::duration<double> elapsed = t1 - t0;
            const double ns = elapsed.count() * 1e9;
            samples.push_back(ns / static_cast<double>(iterations));
        }
    }
    res.checksum = checksum;

    const Statistics stats = compute_statistics(samples);
    res.mean_ns = stats.mean;
    res.median_ns = stats.median;
    res.min_ns = stats.min;
    res.max_ns = stats.max;
    res.stddev_ns = stats.stddev;
    return res;
}

// Prints the result (and the current system) to stdout.
void print_result(const char* name, const BenchmarkResult& res);

}  // namespace chp
