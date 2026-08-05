// Experiment 07: Cache conflict misses.
//
// Two arrays sized so that their element i map to the same cache set
// (stride = L1d size). Alternating access makes each element evict the
// other's line even though total capacity is small: a conflict miss.
// A sequential traversal of the same total bytes is the control.
//
// Reference: PDF 3.3.1 (Associativity / conflict misses).

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "benchmark.h"
#include "cpu_info.h"

static constexpr int kRounds = 5;

int main() {
    long line = cpuinfo::l1d_line_size();
    long l1d = cpuinfo::l1d_size_bytes();
    if (line <= 0) line = 64;
    if (l1d <= 0) l1d = 32768;

    std::printf("Experiment 07: cache conflict misses (L1d=%ld, line=%ld)\n",
                l1d, line);
    std::printf("CAVEAT: virtual-address-based probing; physical page\n"
                "coloring may reduce the effect on modern CPUs.\n\n");

    // Two buffers, each occupying the full L1d address range.
    size_t buf_bytes = (size_t)l1d;
    std::vector<char> a(buf_bytes, 1);
    std::vector<char> b(buf_bytes, 2);

    // Conflicting: access a[i] then b[i] repeatedly -> same set evictions.
    auto conflict = [&] {
        volatile char sink = 0;
        for (int rep = 0; rep < 4096; ++rep)
            for (size_t i = 0; i < buf_bytes; i += (size_t)line)
                sink += a[i] + b[i];
        bm::do_not_optimize(sink);
    };
    // Control: sequential access to one combined buffer of same total size.
    std::vector<char> big(buf_bytes * 2, 1);
    auto seq = [&] {
        volatile char sink = 0;
        for (int rep = 0; rep < 4096; ++rep)
            for (size_t i = 0; i < big.size(); ++i) sink += big[i];
        bm::do_not_optimize(sink);
    };

    conflict();
    seq();

    auto rc = bm::time_rounds(kRounds, conflict);
    auto rs = bm::time_rounds(kRounds, seq);

    std::printf("conflict  : mean=%.3f ms\n", rc.mean_ms);
    std::printf("sequential: mean=%.3f ms\n", rs.mean_ms);
    std::printf("ratio conflict/seq = %.2fx\n", rc.mean_ms / rs.mean_ms);
    return 0;
}
