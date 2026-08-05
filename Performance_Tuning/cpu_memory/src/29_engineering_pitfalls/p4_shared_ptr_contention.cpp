// Pitfall P4: hidden atomic traffic in shared_ptr / std::atomic.
//
// Copying a std::shared_ptr increments its atomic control-block refcount.
// Many threads copying the same shared_ptr generate atomic RMW traffic on
// the same cache line -- a real, easy-to-miss false-sharing-like cost.
// Compare copying a raw pointer vs a shared_ptr.
//
// Related PDF: 6.4.2 (atomicity optimizations), 9.

#include <cstdint>
#include <cstdio>
#include <memory>
#include <thread>
#include <vector>

#include "benchmark.h"

static constexpr int kThreads = 4;
static constexpr long kIters = 50'000'000L;
static constexpr int kRounds = 3;

int main() {
    std::printf("Pitfall P4: shared_ptr refcount contention\n");

    auto data = std::make_shared<int>(42);

    auto run_shared = [&] {
        std::vector<std::thread> pool;
        for (int t = 0; t < kThreads; ++t)
            pool.emplace_back([&] {
                std::shared_ptr<int> sp = data;  // copy once
                long long s = 0;
                for (long i = 0; i < kIters; ++i) {
                    std::shared_ptr<int> local = sp;  // atomic refcount++/-- every copy
                    s += *local;
                }
                bm::do_not_optimize(s);
            });
        for (auto& th : pool) th.join();
    };

    auto run_raw = [&] {
        const int* raw = data.get();
        std::vector<std::thread> pool;
        for (int t = 0; t < kThreads; ++t)
            pool.emplace_back([&] {
                long long s = 0;
                for (long i = 0; i < kIters; ++i) s += *raw;  // no atomics
                bm::do_not_optimize(s);
            });
        for (auto& th : pool) th.join();
    };

    run_shared();
    run_raw();

    auto r_shared = bm::time_rounds(kRounds, run_shared);
    auto r_raw = bm::time_rounds(kRounds, run_raw);

    std::printf("shared_ptr copies : mean=%.2f ms\n", r_shared.mean_ms);
    std::printf("raw pointer reads : mean=%.2f ms\n", r_raw.mean_ms);
    std::printf("shared_ptr is %.1fx slower (atomic refcount on one cache line).\n",
                r_shared.mean_ms / r_raw.mean_ms);
    std::printf("Fix: copy shared_ptr once per thread, pass raw/borrowed refs.\n");
    return 0;
}
