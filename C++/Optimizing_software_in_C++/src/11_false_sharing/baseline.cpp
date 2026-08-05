// 11_false_sharing: baseline -- two threads increment adjacent counters in
// the SAME cache line.
//
// PDF 10 (p112): threads writing to the same cache line invalidate each
// other's cache; every increment forces a cache-line transfer.
#include <cstdio>
#include <thread>

int main() {
    long long a = 0, b = 0;   // adjacent in memory -> same cache line
    std::thread t1([&] { for (int i = 0; i < 200'000'000; ++i) a += 1; });
    std::thread t2([&] { for (int i = 0; i < 200'000'000; ++i) b += 1; });
    t1.join();
    t2.join();
    std::printf("a=%lld b=%lld\n", a, b);
    return 0;
}
