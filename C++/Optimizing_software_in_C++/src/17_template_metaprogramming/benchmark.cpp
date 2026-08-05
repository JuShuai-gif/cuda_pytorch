// 17_template_metaprogramming: std::pow vs compile-time integerPower, and
// constexpr table generation used as a compile-time constant.
#include <cmath>
#include <cstdio>

#include "common/benchmark.h"

template <int n>
inline double ipow_step(double x, double y) {
    if constexpr ((n & 1) == 1) y *= x;
    constexpr int n1 = n >> 1;
    if constexpr (n1 == 0) return y;
    else return ipow_step<n1>(x * x, y);
}

template <int n>
double integerPower(double x) {
    if constexpr (n == 0) return 1.0;
    else if constexpr (n < 0) return 1.0 / integerPower<-n>(x);
    else return ipow_step<n>(x, 1.0);
}

// constexpr table: all 13 factorials computed at compile time (PDF p167).
constexpr int fact(int n) { int r = 1; for (int i = 2; i <= n; ++i) r *= i; return r; }
constexpr int fact_table[13] = { fact(0), fact(1), fact(2), fact(3), fact(4),
    fact(5), fact(6), fact(7), fact(8), fact(9), fact(10), fact(11), fact(12) };

int main() {
    bench("std::pow(x,10)",  [&] { double s=0; for (int i=0;i<5'000'000;++i) s+=std::pow(1.5,10); return s; });
    bench("integerPower<10>",[&] { double s=0; for (int i=0;i<5'000'000;++i) s+=integerPower<10>(1.5); return s; });

    // compile-time table used at runtime
    long long s = 0;
    for (int i = 0; i < 5'000'000; ++i) s += fact_table[i % 13];
    std::printf("\nchecksums: %.6f %.6f  fact_table=%lld\n",
                std::pow(1.5,10), integerPower<10>(1.5), s);
    return 0;
}
