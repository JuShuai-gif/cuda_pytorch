// Experiment 04: Stride access.
//
// Sweeps the stride (in int elements: 1..256) over a large array.
// Observations: small strides reuse cache lines; strides near 64B multiples
// waste the line; large strides may trigger TLB pressure per page touched.
// Reports ns/element and effective GB/s (counting only accessed bytes).
//
// Reference: PDF 3.3.2 (Figure 3.11: element size vs prefetch), 6.3.1.

#include <cstdint>
#include <cstdio>
#include <vector>

#include "benchmark.h"

static constexpr size_t N = 1 << 25;   // 32M ints = 128 MB
static constexpr int kRounds = 5;

int main() {
    std::printf("Experiment 04: stride access (N=%zu ints)\n", N);
    std::vector<int> data(N, 1);

    std::printf("%-12s %-12s %-12s %-14s\n", "stride(el)", "ns/elem", "time_ms",
                "GB/s(used)");

    for (int stride : {1, 2, 4, 8, 16, 32, 64, 128, 256}) {
        auto fn = [&] {
            uint64_t s = 0;
            for (size_t i = 0; i < N; i += (size_t)stride) s += (uint64_t)data[i];
            bm::do_not_optimize(s);
        };
        fn();  // warmup
        auto res = bm::time_rounds(kRounds, fn);
        double elems = (double)N / (double)stride;
        double ns_elem = res.median_ms * 1e6 / elems;
        double used_bytes = elems * sizeof(int);
        double gbps = used_bytes / (res.median_ms * 1e6 / 1e9) / 1e9;
        std::printf("%-12d %-12.2f %-12.2f %-14.3f\n",
                    stride, ns_elem, res.median_ms, gbps);
    }
    return 0;
}
