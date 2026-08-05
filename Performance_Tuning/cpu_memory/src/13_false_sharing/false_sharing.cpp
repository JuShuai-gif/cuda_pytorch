// Experiment 13: False sharing.
//
// Two threads increment two separate counters. Version A puts both counters
// in one cache line; version B pads them so each is on its own cache line
// (padding derived from the detected line size, not hardcoded 64).
//
// Reference: PDF 6.4.1 (Figures 6.10-6.11), A.3.

#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#include "benchmark.h"
#include "cpu_info.h"

static constexpr long ITER = 200'000'000L;
static constexpr int kRounds = 3;

int main() {
    long line = cpuinfo::l1d_line_size();
    if (line <= 0) line = 64;
    std::printf("Experiment 13: false sharing (line size=%ld)\n", line);

    struct Shared {
        long a;
        long b;   // same cache line as a
    };
    struct alignas(64) Padded {
        long val;
    };

    auto run_shared = [&] {
        Shared s{0, 0};
        std::thread t1([&] { for (long i = 0; i < ITER; ++i) ++s.a; });
        std::thread t2([&] { for (long i = 0; i < ITER; ++i) ++s.b; });
        t1.join();
        t2.join();
        bm::do_not_optimize(s.a + s.b);
    };

    auto run_padded = [&] {
        std::vector<Padded> c(2);
        c[0].val = 0;
        c[1].val = 0;
        std::thread t1([&] { for (long i = 0; i < ITER; ++i) ++c[0].val; });
        std::thread t2([&] { for (long i = 0; i < ITER; ++i) ++c[1].val; });
        t1.join();
        t2.join();
        bm::do_not_optimize(c[0].val + c[1].val);
    };

    run_shared();
    run_padded();

    auto r_shared = bm::time_rounds(kRounds, run_shared);
    auto r_padded = bm::time_rounds(kRounds, run_padded);

    std::printf("shared_same_line: mean=%.3f ms\n", r_shared.mean_ms);
    std::printf("padded_separate : mean=%.3f ms\n", r_padded.mean_ms);
    std::printf("speedup padded/shared = %.2fx\n", r_shared.mean_ms / r_padded.mean_ms);
    return 0;
}
