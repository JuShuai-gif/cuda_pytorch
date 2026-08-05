// 11_false_sharing: measure the cache-line ping-pong cost and the fix.
//
// PDF 10 (p112). Both versions do identical work; the only difference is
// whether the two hot counters share a cache line.
#include <cstdio>
#include <thread>

#include "common/benchmark.h"

long long same_line() {
    long long a = 0, b = 0;
    std::thread t1([&] { for (int i = 0; i < 100'000'000; ++i) a += 1; });
    std::thread t2([&] { for (int i = 0; i < 100'000'000; ++i) b += 1; });
    t1.join();
    t2.join();
    return a + b;
}

struct alignas(64) Counter { long long val; };

long long padded() {
    Counter a{0}, b{0};
    std::thread t1([&] { for (int i = 0; i < 100'000'000; ++i) a.val += 1; });
    std::thread t2([&] { for (int i = 0; i < 100'000'000; ++i) b.val += 1; });
    t1.join();
    t2.join();
    return a.val + b.val;
}

int main() {
    bench("same_line (false sharing)", [&] { return same_line(); });
    bench("padded (own cache line)",   [&] { return padded(); });
    std::printf("\nresults equal: %lld %lld\n", same_line(), padded());
    return 0;
}
