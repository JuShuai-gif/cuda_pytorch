// Pitfall P6: microbenchmark noise -- single run vs many runs.
//
// A single run is heavily affected by page faults, frequency ramps, and
// background noise. Multiple runs with warmup and median give a stable
// estimate. We run the same kernel once (cold) and then many times,
// showing how much a single measurement can mislead.
//
// Related PDF: 7 (tools), project benchmark rules (multi-round stats).

#include <cstdint>
#include <cstdio>
#include <vector>

#include "benchmark.h"

static constexpr size_t N = 1 << 25;
static constexpr int kRounds = 9;

int main() {
    std::printf("Pitfall P6: single-run benchmark noise\n");

    std::vector<int> data(N, 1);
    auto kernel = [&] {
        uint64_t s = 0;
        for (size_t i = 0; i < N; ++i) s += (uint64_t)data[i];
        bm::do_not_optimize(s);
    };

    // First call: cold caches / possible page faults.
    auto t0 = bm::Clock::now();
    kernel();
    auto t1 = bm::Clock::now();
    double cold_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto r = bm::time_rounds(kRounds, kernel);

    std::printf("cold first run          : %.2f ms\n", cold_ms);
    std::printf("median of %d runs       : %.2f ms\n", kRounds, r.median_ms);
    std::printf("min / max of %d runs    : %.2f / %.2f ms\n", kRounds,
                r.min_ms, r.max_ms);
    std::printf("stddev                  : %.2f ms\n", r.stddev_ms);
    std::printf("\nLesson: single runs are noisy; warm up, run many times,\n"
                "report median/min/stddev, not one number.\n");
    return 0;
}
