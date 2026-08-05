// Pitfall P7: "just use std::atomic with default ordering" is a hidden cost,
// but seq_cst vs relaxed on a single counter is often NOT the real problem.
//
// The honest lesson from x86: fetch_add has nearly the same cost whether
// seq_cst or relaxed (x86 lock prefix dominates). What actually hurts is
// the ATOMIC OPERATION ITSELF on a contended line (see p4) and doing
// per-element atomics at all. This experiment shows that:
//   - seq_cst vs relaxed on one counter: ~same cost on x86 (don't chase it);
//   - the big win is replacing per-item atomics with a per-thread local
//     sum + one final reduce.
//
// Related PDF: 6.4.2 (atomicity optimizations: prefer local reductions).

#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

#include "benchmark.h"

static constexpr int kThreads = 4;
static constexpr long kIters = 50'000'000L;
static constexpr int kRounds = 3;

int main() {
    std::printf("Pitfall P7: atomics -- which cost is real?\n");
    std::printf("Platform note: on x86, lock-prefixed atomics dominate, so\n"
                "seq_cst vs relaxed for a plain fetch_add differ little.\n\n");

    std::atomic<long long> shared{0};
    std::vector<long long> per(kThreads);

    auto run_shared = [&](std::memory_order mo) {
        std::vector<std::thread> pool;
        for (int t = 0; t < kThreads; ++t)
            pool.emplace_back([&] {
                for (long i = 0; i < kIters; ++i) shared.fetch_add(1, mo);
            });
        for (auto& th : pool) th.join();
        bm::do_not_optimize(shared.load(mo));
    };

    auto run_local = [&] {
        std::vector<std::thread> pool;
        for (int t = 0; t < kThreads; ++t)
            pool.emplace_back([&, t] {
                long long s = 0;
                for (long i = 0; i < kIters; ++i) ++s;  // no atomics at all
                per[(size_t)t] = s;
            });
        for (auto& th : pool) th.join();
        long long sum = 0;
        for (auto v : per) sum += v;
        bm::do_not_optimize(sum);
    };

    run_shared(std::memory_order_seq_cst);
    run_shared(std::memory_order_relaxed);
    run_local();

    auto r_seq = bm::time_rounds(kRounds, [&] { run_shared(std::memory_order_seq_cst); });
    auto r_rel = bm::time_rounds(kRounds, [&] { run_shared(std::memory_order_relaxed); });
    auto r_local = bm::time_rounds(kRounds, run_local);

    std::printf("shared atomic, seq_cst : mean=%.2f ms\n", r_seq.mean_ms);
    std::printf("shared atomic, relaxed : mean=%.2f ms\n", r_rel.mean_ms);
    std::printf("per-thread local + sum : mean=%.2f ms\n", r_local.mean_ms);
    std::printf("\nLesson: don't micro-optimize ordering on x86; instead\n"
                "remove unnecessary atomic operations (local reduce).\n");
    return 0;
}
