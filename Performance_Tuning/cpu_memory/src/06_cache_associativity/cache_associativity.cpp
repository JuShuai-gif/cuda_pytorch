// Experiment 06: Cache associativity probing.
//
// Constructs addresses that map to the same cache set by using a stride
// equal to the number of sets times the line size. Sweeping the number of
// simultaneously "live" addresses that share a set reveals the set
// associativity: performance degrades once the count exceeds the ways.
//
// Limitation: on modern x86, physical-address indexing (and page coloring)
// may prevent us from reliably controlling which set an address lands in.
// We therefore report results and note the caveat in the output.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "benchmark.h"
#include "cpu_info.h"

static constexpr int kRounds = 5;

int main() {
    long line = cpuinfo::l1d_line_size();
    if (line <= 0) line = 64;
    long l1d = cpuinfo::l1d_size_bytes();
    if (l1d <= 0) l1d = 32768;

    // Assume a probe using 8 candidate ways; we stride over the whole L1d
    // address space so that addresses with the same (virtual) set index are
    // separated by l1d bytes.
    std::vector<char> buf((size_t)l1d * 8, 1);

    std::printf("Experiment 06: cache associativity probe\n");
    std::printf("detected L1d size=%ld, line=%ld\n", l1d, line);
    std::printf("stride = L1d size = %ld bytes (same virtual set)\n", l1d);
    std::printf("\nCAVEAT: this uses virtual addresses; physical-address\n"
                "indexing / page coloring on the CPU may defeat the probe.\n"
                "Results are indicative, not authoritative.\n\n");

    std::printf("%-8s %-12s %-12s\n", "ways", "time_ms", "ns/elem");
    for (int ways = 1; ways <= 16; ++ways) {
        auto fn = [&] {
            volatile char sink = 0;
            for (int w = 0; w < ways; ++w) {
                size_t base = (size_t)w * (size_t)l1d;
                sink += buf[base];
            }
            bm::do_not_optimize(sink);
        };
        fn();
        auto res = bm::time_rounds(kRounds, fn);
        std::printf("%-8d %-12.3f %-12.3f\n", ways, res.median_ms,
                    res.median_ms * 1e6 / (double)ways);
    }
    return 0;
}
