// Benchmark: parallel pipeline vs serial map/filter/reduce.

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <numeric>
#include <vector>

#include "parallel_pipeline.hpp"

namespace {

long work(long v) {
    long s = v;
    for (long i = 0; i < 60; ++i) {
        s += (i % 2) * v;  // adds v for odd i; no overflow, deterministic
    }
    return s;
}

bool even(long v) { return (v % 2) == 0; }

template <typename Fn>
double measure(Fn fn) {
    const auto t0 = std::chrono::steady_clock::now();
    fn();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

constexpr std::size_t kCount = 4'000'000;

}  // namespace

int main() {
    std::printf("== parallel_pipeline benchmark ==\n");

    std::vector<long> src(kCount);
    std::iota(src.begin(), src.end(), 0L);

    const double serial_s = measure([&] {
        volatile long sink = 0;
        for (const long v : src) {
            if (even(v)) {
                const long w = work(v);
                sink += w * w;
            }
        }
    });

    const double par_s = measure([&] {
        const auto result = chp::parallel_pipeline<long, long>(
            src, [](long v) { long w = work(v); return w * w; }, even,
            [](long a, long b) { return a + b; }, 0L);
        std::printf("  (pipeline result=%ld)\n", result);
    });

    std::printf("data: %zu longs\n", kCount);
    std::printf("serial: %7.3f s\n", serial_s);
    std::printf("pipeline (par): %7.3f s\n", par_s);
    std::printf("speedup: %5.2fx\n", serial_s / par_s);

    return 0;
}
