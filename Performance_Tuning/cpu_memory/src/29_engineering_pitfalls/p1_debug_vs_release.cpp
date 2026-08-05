// Pitfall P1: Debug vs Release build.
//
// A function compiled with -O0 (simulating a Debug build, via the gcc/clang
// `optimize` attribute) runs far slower than the same logic at -O3. Measuring
// performance in a Debug build is meaningless; always benchmark Release.
//
// Related PDF: 6.2 (programmers must focus on what compilers actually emit).

#include <cstdio>
#include <cstdint>

#include "benchmark.h"

// Force unoptimized codegen for this function even inside a -O3 binary,
// to reproduce what you get in a Debug build.
__attribute__((optimize("O0"))) static uint64_t compute_slow(uint64_t n) {
    uint64_t s = 0;
    for (uint64_t i = 0; i < n; ++i) s += i & 7;
    return s;
}

static uint64_t compute_fast(uint64_t n) {
    uint64_t s = 0;
    for (uint64_t i = 0; i < n; ++i) s += i & 7;
    return s;
}

int main() {
    constexpr uint64_t N = 1u << 30;
    constexpr int kRounds = 3;

    std::printf("Pitfall P1: Debug (-O0) vs Release (-O3) build\n");
    std::printf("NOTE: -O0 is forced via __attribute__((optimize)) on one\n"
                "function; the real lesson is: never benchmark a Debug build.\n\n");

    uint64_t cs = compute_slow(1000) + compute_fast(1000);
    bm::do_not_optimize(cs);

    auto r_slow = bm::time_rounds(kRounds, [] { bm::do_not_optimize(compute_slow(N)); });
    auto r_fast = bm::time_rounds(kRounds, [] { bm::do_not_optimize(compute_fast(N)); });

    std::printf("O0 (Debug-like) : mean=%.2f ms\n", r_slow.mean_ms);
    std::printf("O3 (Release)    : mean=%.2f ms\n", r_fast.mean_ms);
    std::printf("O3 is %.1fx faster. Benchmark in Release only.\n",
                r_slow.mean_ms / r_fast.mean_ms);
    return 0;
}
