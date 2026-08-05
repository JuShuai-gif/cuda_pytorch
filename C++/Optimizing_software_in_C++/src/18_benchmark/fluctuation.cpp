// 18_benchmark: fluctuation & measurement discipline -- many rounds of the
// same workload show the natural noise (turbo, OS, interrupts). This is why
// we report min/median and warm up (PDF p168-169).
#include <cstdio>
#include <vector>

#include "common/benchmark.h"

int main() {
    std::vector<double> v(8'000'000, 1.0);

    // 9 raw rounds, printed individually, to show fluctuation.
    std::printf("== 9 raw rounds (same workload) ==\n");
    for (int r = 0; r < 9; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        volatile double s = 0.0;
        for (double x : v) s += x;
        auto t1 = std::chrono::steady_clock::now();
        std::printf("round %d: %8.2f us\n", r,
            std::chrono::duration<double, std::micro>(t1 - t0).count());
    }

    std::printf("\n== disciplined measurement (warmup + median) ==\n");
    bench("sum", [&] { double s = 0.0; for (double x : v) s += x; return s; });
    return 0;
}
