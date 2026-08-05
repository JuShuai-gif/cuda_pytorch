// 01_profiling: manual timing with std::chrono.
//
// Demonstrates the "instrument the code yourself" approach (PDF p17):
// measure each candidate with a high-resolution clock, run multiple times,
// and inspect the first (cold) vs. following (hot) readings.

#include <chrono>
#include <cstdio>
#include <vector>

#include "common/benchmark.h"

// Candidate A: naive sum with a branch.
double sum_a(const std::vector<double>& v) {
    double s = 0.0;
    for (size_t i = 0; i < v.size(); ++i) {
        s += v[i];
    }
    return s;
}

// Candidate B: 4-accumulator sum (breaks the dependency chain, PDF p114).
double sum_b(const std::vector<double>& v) {
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    size_t i = 0;
    for (; i + 3 < v.size(); i += 4) {
        s0 += v[i];
        s1 += v[i + 1];
        s2 += v[i + 2];
        s3 += v[i + 3];
    }
    for (; i < v.size(); ++i) s0 += v[i];
    return (s0 + s1) + (s2 + s3);
}

int main() {
    const size_t n = 8'000'000;
    std::vector<double> v(n, 1.0);

    std::printf("== chrono single-shot readings (cold vs warm) ==\n");
    for (int k = 0; k < 5; ++k) {
        auto t0 = std::chrono::steady_clock::now();
        volatile double r = sum_a(v);
        auto t1 = std::chrono::steady_clock::now();
        (void)r;
        std::printf("shot %d: %8.2f us\n",
                    k, std::chrono::duration<double, std::micro>(t1 - t0).count());
    }

    std::printf("\n== benchmark helper: warmup + median (PDF p168) ==\n");
    bench("sum_a", [&] { return sum_a(v); });
    bench("sum_b", [&] { return sum_b(v); });

    // Verify results are identical.
    std::printf("\nsum_a = %.0f  sum_b = %.0f\n", sum_a(v), sum_b(v));
    return 0;
}
