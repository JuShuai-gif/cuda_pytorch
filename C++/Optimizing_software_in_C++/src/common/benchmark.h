// Shared benchmarking helpers (header-only).
//
// Design notes (PDF ch.16, p167-171):
//  * warm-up runs before measurement (cache + CPU frequency settle);
//  * multiple rounds, report min / median / mean;
//  * the measured callable returns a value that is folded into a volatile
//    sink so the compiler cannot eliminate the work.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

struct BenchResult {
    double min_us;
    double median_us;
    double mean_us;
};

template <class Fn>
BenchResult bench(const char* name, Fn&& fn, int warmup = 3, int rounds = 7) {
    volatile long long sink = 0;

    for (int i = 0; i < warmup; ++i) {
        sink += static_cast<long long>(fn());
    }

    std::vector<double> us;
    us.reserve(static_cast<size_t>(rounds));
    for (int r = 0; r < rounds; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        sink += static_cast<long long>(fn());
        auto t1 = std::chrono::steady_clock::now();
        us.push_back(
            std::chrono::duration<double, std::micro>(t1 - t0).count());
    }

    std::sort(us.begin(), us.end());
    double mean = 0.0;
    for (double x : us) mean += x;
    mean /= static_cast<double>(us.size());

    BenchResult res{us.front(), us[us.size() / 2], mean};
    if (name != nullptr) {
        std::printf("%-18s min=%9.2f us  median=%9.2f us  mean=%9.2f us\n",
                    name, res.min_us, res.median_us, res.mean_us);
    }
    return res;
}
