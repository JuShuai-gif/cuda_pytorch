// 15_lookup_table: compute vs table, and table size vs cache.
//
// PDF 14.1 (p144-146): lookup is fast only while the table stays in cache;
// a big table gets evicted and the wins disappear.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "common/benchmark.h"

int factorial_loop(int n) {
    int f = 1;
    for (int i = 2; i <= n; ++i) f *= i;
    return f;
}

static const int fact_table[13] = {1, 1, 2, 6, 24, 120, 720, 5040,
    40320, 362880, 3628800, 39916800, 479001600};
int factorial_table(int n) {
    if ((unsigned)n < 13) return fact_table[n];
    return 0;
}

// A sin-like function computed with a series vs. a table of 2^k entries.
double compute_series(double x) {
    // ~8 terms of the Taylor series for sin
    double x2 = x * x;
    double term = x, sum = x;
    for (int k = 1; k < 4; ++k) {
        term *= -x2 / ((2 * k) * (2 * k + 1));
        sum += term;
    }
    return sum;
}

int main() {
    // (a) factorial: loop vs table
    bench("factorial_loop",  [&] { long long s=0; for (int i=0;i<20'000'000;++i) s+=factorial_loop(i%13); return s; });
    bench("factorial_table", [&] { long long s=0; for (int i=0;i<20'000'000;++i) s+=factorial_table(i%13); return s; });

    // (b) table size vs cache: small table (fits L1) vs big table (evicted)
    const int small = 1024, big = 64 * 1024 * 1024 / 8;   // 8 bytes/double
    std::vector<double> tsmall(small), tbig(big);
    for (int i = 0; i < small; ++i) tsmall[i] = (double)i * 0.001;
    for (int i = 0; i < big; ++i) tbig[i] = (double)i * 0.001;

    std::srand(42);
    std::vector<int> idx_small(8'000'000), idx_big(8'000'000);
    for (int& x : idx_small) x = std::rand() % small;
    for (int& x : idx_big) x = std::rand() % big;

    bench("small_table_lookup", [&] {
        double s = 0.0; for (int i : idx_small) s += tsmall[i]; return s;
    });
    bench("big_table_lookup", [&] {
        double s = 0.0; for (int i : idx_big) s += tbig[i]; return s;
    });

    std::printf("\nchecksums: %lld %lld\n",
                (long long)factorial_loop(12), (long long)factorial_table(12));
    return 0;
}
