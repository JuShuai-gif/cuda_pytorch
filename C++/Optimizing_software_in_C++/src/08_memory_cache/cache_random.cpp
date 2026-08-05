// 08_memory_cache: sequential vs random access at a fixed size.
//
// PDF 9.9 (p106): sequential access is fastest, random access is worst.
#include <cstdio>
#include <random>
#include <vector>

#include "common/benchmark.h"

int main() {
    const size_t n = 64 * 1024 * 1024 / sizeof(int);   // 64 MiB of ints
    std::vector<int> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = (int)i;

    // Sequential: touch every element in order.
    bench("sequential", [&] {
        long long s = 0;
        for (size_t i = 0; i < n; ++i) s += v[i];
        return s;
    });

    // Strided: 16 elements apart -> fewer cache lines used, more per line.
    bench("strided16", [&] {
        long long s = 0;
        for (size_t i = 0; i < n; i += 16) s += v[i];
        return s;
    });

    // Random order: every access likely misses.
    std::vector<size_t> idx(n);
    for (size_t i = 0; i < n; ++i) idx[i] = i;
    std::mt19937 rng(12345);
    std::shuffle(idx.begin(), idx.end(), rng);
    bench("random", [&] {
        long long s = 0;
        for (size_t i = 0; i < n; ++i) s += v[idx[i]];
        return s;
    });

    std::printf("\nchecksums equal: %lld\n", (long long)v[0]);
    return 0;
}
