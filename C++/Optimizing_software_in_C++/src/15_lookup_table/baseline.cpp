// 15_lookup_table: baseline -- compute factorial and a periodic function
// with a loop / library call instead of a table.
//
// PDF 14.1 (p144-146): computing in a loop vs reading a cached table.
#include <cstdio>
#include <vector>

int factorial_loop(int n) {   // PDF Example 14.1a
    int f = 1;
    for (int i = 2; i <= n; ++i) f *= i;
    return f;
}

double phase_loop(int i) {    // a "cheap-ish" function we could tabulate
    return (i % 4 == 0) ? 1.0 : (i % 4 == 1) ? 1.5 : (i % 4 == 2) ? 2.0 : 2.5;
}

int main() {
    const int n = 100'000'000;
    long long s = 0;
    for (int i = 0; i < n; ++i) s += factorial_loop(i % 13);
    std::printf("factorial checksum = %lld\n", s);

    double d = 0.0;
    for (int i = 0; i < n; ++i) d += phase_loop(i);
    std::printf("phase checksum = %.1f\n", d);
    return 0;
}
