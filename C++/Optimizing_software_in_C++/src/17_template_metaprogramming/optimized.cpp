// 17_template_metaprogramming: optimized -- compile-time power with
// if constexpr, and a constexpr bit scan.
//
// PDF 15.2/15.3 (p166-167): the compiler expands this into plain
// multiplications; false branches are never instantiated.
#include <cstdio>

template <int n>
inline double ipow_step(double x, double y) {
    if constexpr ((n & 1) == 1) y *= x;
    constexpr int n1 = n >> 1;
    if constexpr (n1 == 0) {
        return y;
    } else {
        return ipow_step<n1>(x * x, y);
    }
}

template <int n>
double integerPower(double x) {
    if constexpr (n == 0) return 1.0;
    else if constexpr (n < 0) return 1.0 / integerPower<-n>(x);
    else return ipow_step<n>(x, 1.0);
}

constexpr int bit_scan_reverse(unsigned long long n) {   // PDF Example 15.3
    if (n == 0) return -1;
    unsigned long long a = n, b = 0, j = 64, k = 0;
    do {
        j >>= 1;
        k = 1ULL << j;
        if (a >= k) { a >>= j; b += j; }
    } while (j > 0);
    return (int)b;
}

int main() {
    volatile double r = integerPower<10>(1.5);
    // compile-time computation: used as a constant
    constexpr int msb = bit_scan_reverse(0xF0F0F0F0ULL);
    volatile int b = msb;
    std::printf("checksums: %.4f %d\n", r, b);
    return 0;
}
