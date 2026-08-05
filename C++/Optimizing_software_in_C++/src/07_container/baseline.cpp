// 07_container: baseline -- std::list without reserve-style optimization,
// and vector filled by naive push_back.
//
// PDF 9.6-9.7 (p95-105): linked lists allocate per element and are cache-hostile;
// std::vector reallocates when full (10 elements -> 7 allocations, PDF p98).
#include <cstdio>
#include <list>
#include <vector>

int main() {
    // (a) push_back without reserve: repeated reallocations (PDF p98).
    std::vector<int> v;
    for (int i = 0; i < 4'000'000; ++i) v.push_back(i);
    long long sum = 0;
    for (int x : v) sum += x;

    // (b) std::list: per-element allocation + pointer chasing.
    std::list<int> l;
    for (int i = 0; i < 100'000; ++i) l.push_back(i);
    long long lsum = 0;
    for (int x : l) lsum += x;

    std::printf("vector sum=%lld list sum=%lld\n", sum, lsum);
    return 0;
}
