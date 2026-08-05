// Experiment 27: Instruction cache.
//
// Measures the effect of code layout on L1i. Compares a function whose hot
// path is kept linear vs one with an interleaved rarely-taken cold branch.
// Uses likely/unlikely + __builtin_expect to guide layout at -O2.
//
// Reference: PDF 6.2.2 (Optimizing L1i), 7.4.

#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include "benchmark.h"

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

static constexpr int ITER = 200'000'000;
static constexpr int kRounds = 5;

static volatile int g_flag = 0;

// Version without branch hints: cold code sits in the hot path.
static uint64_t hot_no_hint() {
    uint64_t s = 0;
    for (int i = 0; i < ITER; ++i) {
        if (g_flag != 0) {
            s += 999;  // cold, never taken
        }
        s += (uint64_t)(i & 7);
    }
    return s;
}

// Version with likely/unlikely: cold code moved out of the hot path.
static uint64_t hot_with_hint() {
    uint64_t s = 0;
    for (int i = 0; i < ITER; ++i) {
        if (unlikely(g_flag != 0)) {
            s += 999;  // cold, never taken
        }
        s += (uint64_t)(i & 7);
    }
    return s;
}

int main() {
    g_flag = 0;  // ensure the cold branch is never taken
    std::printf("Experiment 27: instruction cache / branch layout\n");

    auto r0 = bm::time_rounds(kRounds, [] { bm::do_not_optimize(hot_no_hint()); });
    auto r1 = bm::time_rounds(kRounds, [] { bm::do_not_optimize(hot_with_hint()); });

    std::printf("no-hint  : mean=%.3f ms\n", r0.mean_ms);
    std::printf("with-hint: mean=%.3f ms\n", r1.mean_ms);
    std::printf("NOTE: effect depends on compiler and CPU; use perf\n"
                "L1-icache events on this machine to confirm.\n");
    return 0;
}
