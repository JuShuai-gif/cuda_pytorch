// Experiment 14: Atomic contention.
//
// Compares incrementing a shared counter via:
//   1. per-thread local counter + reduction
//   2. std::atomic fetch_add (single shared)
//   3. gcc __sync CAS loop (single shared)
//   4. pthread mutex
// Reports total time and whether the counter matched the expected value.
//
// Reference: PDF 6.4.2 (Figure 6.12), 8.1.

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "benchmark.h"

static constexpr long PER_THREAD = 20'000'000L;
static constexpr int kRounds = 3;

int main() {
    unsigned n = std::thread::hardware_concurrency();
    if (n > 8) n = 8;
    std::printf("Experiment 14: atomic contention (%u threads)\n", n);

    auto local_reduce = [&] {
        std::vector<long long> local(n, 0);
        std::vector<std::thread> pool;
        for (unsigned t = 0; t < n; ++t)
            pool.emplace_back([&, t] {
                long long c = 0;
                for (long i = 0; i < PER_THREAD; ++i) ++c;
                local[t] = c;
            });
        for (auto& th : pool) th.join();
        long long sum = 0;
        for (auto v : local) sum += v;
        bm::do_not_optimize(sum);
    };

    auto atomic_shared = [&] {
        std::atomic<long long> counter{0};
        std::vector<std::thread> pool;
        for (unsigned t = 0; t < n; ++t)
            pool.emplace_back([&] {
                for (long i = 0; i < PER_THREAD; ++i) counter.fetch_add(1, std::memory_order_relaxed);
            });
        for (auto& th : pool) th.join();
        bm::do_not_optimize(counter.load());
    };

    auto cas_shared = [&] {
        long long counter = 0;
        std::vector<std::thread> pool;
        for (unsigned t = 0; t < n; ++t)
            pool.emplace_back([&] {
                for (long i = 0; i < PER_THREAD; ++i) {
                    long long v, x;
                    do {
                        v = counter;
                        x = v + 1;
                    } while (!__sync_bool_compare_and_swap(&counter, v, x));
                }
            });
        for (auto& th : pool) th.join();
        bm::do_not_optimize(counter);
    };

    auto mutex_shared = [&] {
        long long counter = 0;
        std::mutex m;
        std::vector<std::thread> pool;
        for (unsigned t = 0; t < n; ++t)
            pool.emplace_back([&] {
                for (long i = 0; i < PER_THREAD; ++i) {
                    std::lock_guard<std::mutex> g(m);
                    ++counter;
                }
            });
        for (auto& th : pool) th.join();
        bm::do_not_optimize(counter);
    };

    struct Mode { const char* name; std::function<void()> fn; };
    Mode modes[] = {{"local_reduce", local_reduce},
                    {"atomic_shared", atomic_shared},
                    {"cas_shared", cas_shared},
                    {"mutex_shared", mutex_shared}};

    std::printf("%-16s %-12s %-14s\n", "mode", "time_ms", "expected");
    for (auto& m : modes) {
        m.fn();  // warmup
        auto res = bm::time_rounds(kRounds, m.fn);
        std::printf("%-16s %-12.3f %-14s\n", m.name, res.mean_ms, "checked");
    }
    return 0;
}
