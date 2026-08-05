// 01_profiling: a program with an obvious hot spot, intended for perf.
//
// Use with:
//   perf record -g ./build/01_profiling/01_hotspot
//   perf report
// The hot function should dominate the CPU time (PDF p16-17).

#include <cmath>
#include <cstdio>
#include <vector>

// The intended hot spot: heavy floating-point math in a loop.
double heavy_math(const std::vector<double>& x, double a) {
    double s = 0.0;
    for (double v : x) {
        s += std::sin(v) * std::cos(v) * a;  // FP math, hard to vectorize
    }
    return s;
}

// A deliberately "cold" helper that does almost nothing.
double cold_helper(double v) { return v * 0.0 + 1.0; }

int main() {
    std::vector<double> x(4'000'000, 1.5);
    double s = 0.0;

    // Alternate hot and cold work so the profiler can separate them.
    for (int rep = 0; rep < 50; ++rep) {
        s += heavy_math(x, 1.0001);
        s += cold_helper(s);
    }

    std::printf("checksum = %.6f\n", s);
    return 0;
}
