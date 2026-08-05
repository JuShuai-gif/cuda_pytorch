// Experiment 18: TLB capacity.
//
// Touches exactly one element per page (element at the start of each 4 KB
// page) over an array. Sweeping the number of pages reveals the DTLB
// capacity: latency jumps once the page count exceeds the DTLB entries.
// Sequential-adjacent elements would be masked by prefetching, so we touch
// a single element per page (stride = page size).
//
// Reference: PDF 4.3, 4.3.2, Figure 3.17.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "benchmark.h"
#include "cpu_info.h"

static constexpr int kRounds = 5;

int main() {
    long page = cpuinfo::page_size();
    std::printf("Experiment 18: TLB capacity (page size=%ld)\n", page);

    // Allocate enough for up to 64K pages.
    size_t pages = 1u << 16;
    std::vector<char> mem(pages * (size_t)page, 1);

    std::printf("%-12s %-12s %-14s\n", "pages", "time_ms", "ns/access");
    for (int shift = 8; shift <= 16; ++shift) {
        size_t np = (size_t)1 << shift;
        auto fn = [&] {
            volatile char sink = 0;
            for (size_t r = 0; r < 8; ++r)
                for (size_t p = 0; p < np; ++p)
                    sink += mem[p * (size_t)page];  // one element per page
            bm::do_not_optimize(sink);
        };
        fn();
        auto res = bm::time_rounds(kRounds, fn);
        double accesses = 8.0 * (double)np;
        double ns_acc = res.median_ms * 1e6 / accesses;
        std::printf("%-12zu %-12.3f %-14.2f\n", np, res.median_ms, ns_acc);
    }
    return 0;
}
