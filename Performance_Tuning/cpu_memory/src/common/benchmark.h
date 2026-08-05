#pragma once

// Common benchmark infrastructure.
// Provides multi-round timing with mean/median/min/max/stddev,
// compiler barriers, and checksum helpers so that measurement loops
// are not optimized away.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace bm {

// Result of a multi-round benchmark.
struct BenchmarkResult {
    double mean_ms;    // arithmetic mean of round times (ms)
    double median_ms;  // median round time (ms)
    double min_ms;     // minimum round time (ms)
    double max_ms;     // maximum round time (ms)
    double stddev_ms;  // sample standard deviation (ms)
};

// Prevent the compiler from removing surrounding code.
inline void compiler_barrier() noexcept {
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

// Force a read of a value so it cannot be optimized away.
template <typename T>
inline void do_not_optimize(T const& value) noexcept {
    asm volatile("" : : "r,m"(value) : "memory");
}

using Clock = std::chrono::steady_clock;

// Measure fn() for `rounds` rounds, return statistics over round times.
// fn should do a full workload each call; total time is divided by
// repetitions performed inside fn as needed by the caller.
template <typename Fn>
BenchmarkResult time_rounds(int rounds, Fn&& fn) {
    std::vector<double> times;
    times.reserve(static_cast<size_t>(rounds));
    for (int r = 0; r < rounds; ++r) {
        auto t0 = Clock::now();
        fn();
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times.push_back(ms);
    }
    std::sort(times.begin(), times.end());

    BenchmarkResult res;
    res.min_ms = times.front();
    res.max_ms = times.back();
    res.median_ms = times[times.size() / 2];
    double sum = 0.0;
    for (double t : times) sum += t;
    res.mean_ms = sum / static_cast<double>(times.size());
    double var = 0.0;
    for (double t : times) {
        double d = t - res.mean_ms;
        var += d * d;
    }
    var /= static_cast<double>(times.size() > 1 ? times.size() - 1 : 1);
    res.stddev_ms = std::sqrt(var);
    return res;
}

// A simple deterministic hash (xorshift64-based) for checksumming.
inline uint64_t mix64(uint64_t x) noexcept {
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    return x * 0x2545F4914F6CDD1DULL;
}

// Print a BenchmarkResult in a compact table form.
inline void print_result(const char* name, const BenchmarkResult& r) {
    std::printf("%-28s mean=%10.4f ms median=%10.4f ms "
                "min=%10.4f ms max=%10.4f ms stddev=%10.4f ms\n",
                name, r.mean_ms, r.median_ms, r.min_ms, r.max_ms, r.stddev_ms);
}

}  // namespace bm
