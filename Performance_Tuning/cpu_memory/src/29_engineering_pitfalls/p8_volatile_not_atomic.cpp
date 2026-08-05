// Pitfall P8: volatile for shared data is NOT thread-safe.
//
// volatile only tells the compiler "don't cache this in a register / don't
// elide accesses". It does NOT make increments atomic or provide ordering.
// Two threads doing ++ on a volatile long lose updates. The same loop with
// std::atomic is correct. This is a correctness pitfall that shows up in
// real workloads.
//
// Related PDF: 6.4.2 (atomicity), 8.1.

#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

static constexpr int kThreads = 4;
static constexpr long kIters = 5'000'000L;

int main() {
    std::printf("Pitfall P8: volatile is not atomic\n");

    volatile long v = 0;
    std::atomic<long long> a{0};

    auto inc_volatile = [&] {
        std::vector<std::thread> pool;
        for (int t = 0; t < kThreads; ++t)
            pool.emplace_back([&] { for (long i = 0; i < kIters; ++i) ++v; });
        for (auto& th : pool) th.join();
    };

    auto inc_atomic = [&] {
        std::vector<std::thread> pool;
        for (int t = 0; t < kThreads; ++t)
            pool.emplace_back([&] { for (long i = 0; i < kIters; ++i) a.fetch_add(1, std::memory_order_relaxed); });
        for (auto& th : pool) th.join();
    };

    inc_volatile();
    inc_atomic();

    long long expect = (long long)kThreads * kIters;
    std::printf("expected       : %lld\n", expect);
    std::printf("volatile  result: %lld (wrong! lost updates)\n", (long long)v);
    std::printf("atomic    result: %lld (correct)\n", a.load());
    std::printf("\nLesson: volatile = no register caching / no elision.\n"
                "It does NOT give atomicity. Use std::atomic or a mutex.\n");
    return 0;
}
