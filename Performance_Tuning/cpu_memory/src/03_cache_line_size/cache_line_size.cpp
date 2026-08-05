// Experiment 03: Cache line size detection via bandwidth vs access granularity.
//
// Reads the array touching only every k-th byte (k = 4..128). When k
// exceeds the cache line size, each accessed byte costs a new line fetch
// and bandwidth drops. The plateau boundary reveals the line size.
// Also prints the value reported by the kernel (/sys/.../coherency_line_size).

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "benchmark.h"
#include "cpu_info.h"

static constexpr size_t ARRAY_BYTES = 128u << 20;   // 128 MiB > LLC
static constexpr int kRounds = 5;

int main() {
    std::printf("Experiment 03: cache line size\n");
    long sys_line = cpuinfo::l1d_line_size();
    std::printf("kernel-reported L1d coherency_line_size: %ld bytes\n", sys_line);

    std::vector<char> data(ARRAY_BYTES, 1);

    std::printf("%-12s %-12s %-14s %-12s\n", "step(B)", "time_ms", "bytes/sec",
                "GB/s");

    for (int step : {4, 8, 16, 32, 64, 128, 256}) {
        size_t stride = static_cast<size_t>(step);
        auto fn = [&] {
            volatile char sink = 0;
            for (size_t off = 0; off < stride; ++off) {
                for (size_t i = off; i < data.size(); i += stride) {
                    sink += data[i];
                }
            }
            bm::do_not_optimize(sink);
        };
        fn();  // warmup
        auto res = bm::time_rounds(kRounds, fn);
        size_t touched = (ARRAY_BYTES / stride) * stride;  // approx bytes touched
        double gbps = (double)touched / (res.median_ms * 1e6 / 1e9) / 1e9;
        std::printf("%-12d %-12.2f %-14.0f %-12.3f\n",
                    step, res.median_ms, (double)touched / (res.median_ms * 1e-3),
                    gbps);
    }

    std::printf("\nInterpretation: after the step exceeds the cache line size,\n"
                "every touched byte requires a fresh cache line, so GB/s drops.\n");
    return 0;
}
