// 11_false_sharing: optimized -- two counters padded to separate cache lines.
//
// PDF 10 (p112): align thread-specific data by the cache line size (64 B).
#include <cstdio>
#include <thread>

struct alignas(64) Counter {   // each counter occupies its own cache line
    long long val;
};

int main() {
    Counter a{0}, b{0};
    std::thread t1([&] { for (int i = 0; i < 200'000'000; ++i) a.val += 1; });
    std::thread t2([&] { for (int i = 0; i < 200'000'000; ++i) b.val += 1; });
    t1.join();
    t2.join();
    std::printf("a=%lld b=%lld\n", a.val, b.val);
    return 0;
}
