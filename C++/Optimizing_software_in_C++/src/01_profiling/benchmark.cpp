// 01_profiling: locate the hot function with timers and compare variants.
//
// This program measures each candidate function independently, the way the
// book suggests isolating a hot spot once it has been found (PDF p167).

#include <cmath>
#include <cstdio>
#include <vector>

#include "common/benchmark.h"

double naive_log_sum(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += std::log(x);
    return s;
}

double loop_log_sum(const std::vector<double>& v) {
    // log() calls form a serial dependency; unrolled 4 accumulators help
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    size_t i = 0;
    for (; i + 3 < v.size(); i += 4) {
        s0 += std::log(v[i]);
        s1 += std::log(v[i + 1]);
        s2 += std::log(v[i + 2]);
        s3 += std::log(v[i + 3]);
    }
    for (; i < v.size(); ++i) s0 += std::log(v[i]);
    return (s0 + s1) + (s2 + s3);
}

int main() {
    std::vector<double> v(1'000'000, 2.0);

    bench("naive_log_sum", [&] { return naive_log_sum(v); });
    bench("loop_log_sum",  [&] { return loop_log_sum(v); });

    // Identical results?
    std::printf("naive=%.6f loop=%.6f\n", naive_log_sum(v), loop_log_sum(v));
    return 0;
}
