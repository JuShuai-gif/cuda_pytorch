// 08_memory_cache: AoS vs SoA data layout.
//
// PDF 9.4 (p93-95, Example 9.1): when items are accessed together, keep them
// adjacent (AoS). SoA helps vectorization (PDF 12.9 p133) but separates data.
#include <cstdio>
#include <vector>

#include "common/benchmark.h"

struct Item { int a; int b; };   // AoS

int main() {
    const size_t n = 16'000'000;

    // AoS: a and b adjacent per item.
    std::vector<Item> aos(n);
    for (size_t i = 0; i < n; ++i) { aos[i].a = (int)i; aos[i].b = (int)-i; }

    // SoA: two separate arrays.
    std::vector<int> a(n), b(n);
    for (size_t i = 0; i < n; ++i) { a[i] = (int)i; b[i] = (int)-i; }

    // Access both fields of every item together.
    bench("AoS_sum", [&] {
        long long s = 0;
        for (size_t i = 0; i < n; ++i) s += aos[i].a + aos[i].b;
        return s;
    });

    bench("SoA_sum", [&] {
        long long s = 0;
        for (size_t i = 0; i < n; ++i) s += a[i] + b[i];
        return s;
    });

    // Sanity: same math.
    long long s1 = 0, s2 = 0;
    for (size_t i = 0; i < n; ++i) { s1 += aos[i].a + aos[i].b; s2 += a[i] + b[i]; }
    std::printf("\naos=%lld soa=%lld equal=%d\n", s1, s2, s1 == s2);
    return 0;
}
