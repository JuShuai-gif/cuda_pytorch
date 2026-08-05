// 15_lookup_table: optimized -- static const lookup tables.
//
// PDF 14.1 (p144-146): a table read from L1 cache costs a few cycles.
// static const: initialized once, and the compiler knows it never changes
// (PDF p26, Example 7.1).
#include <cstdio>
#include <vector>

static const int fact_table[13] = {1, 1, 2, 6, 24, 120, 720, 5040,
    40320, 362880, 3628800, 39916800, 479001600};   // PDF Example 14.1b

static const double phase_table[4] = {1.0, 1.5, 2.0, 2.5};

int factorial_table(int n) {
    if ((unsigned)n < 13) return fact_table[n];   // bounds check + lookup
    return 0;
}

double phase_table_lookup(int i) { return phase_table[i & 3]; }

int main() {
    const int n = 100'000'000;
    long long s = 0;
    for (int i = 0; i < n; ++i) s += factorial_table(i % 13);
    std::printf("factorial checksum = %lld\n", s);

    double d = 0.0;
    for (int i = 0; i < n; ++i) d += phase_table_lookup(i);
    std::printf("phase checksum = %.1f\n", d);
    return 0;
}
