// Pipeline demo: find the sum of squares of odd numbers in a large range.

#include <cstdio>
#include <vector>

#include "parallel_pipeline.hpp"

namespace {

// Cost varies with v so stages have non-trivial work (60 iterations, no overflow).
long heavy_map(long v) {
    long s = v;
    for (long i = 0; i < 60; ++i) {
        s += (i % 2) * v;  // adds v on odd i; deterministic, non-zero sum
    }
    return s;
}

bool is_odd(long v) { return (v % 2) != 0; }

long add(long a, long b) { return a + b; }

}  // namespace

int main() {
    std::printf("== parallel_pipeline ==\n");

    constexpr std::size_t n = 2'000'000;
    std::vector<long> src(n);
    for (std::size_t i = 0; i < n; ++i) {
        src[i] = static_cast<long>(i);
    }

    // Pipeline: square each, keep the odd inputs, sum their squares.
    const auto result = chp::parallel_pipeline<long, long>(
        src,
        [](long v) { return heavy_map(v) * heavy_map(v); },
        [](long v) { return is_odd(v); },
        [](long a, long b) { return add(a, b); },
        0L);

    // Serial reference.
    long serial = 0;
    for (const long v : src) {
        if (is_odd(v)) {
            serial += heavy_map(v) * heavy_map(v);
        }
    }

    std::printf("pipeline result = %ld, serial = %ld, match: %d\n", result,
                serial, result == serial);

    return 0;
}
