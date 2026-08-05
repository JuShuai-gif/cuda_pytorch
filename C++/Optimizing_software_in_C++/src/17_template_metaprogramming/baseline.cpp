// 17_template_metaprogramming: baseline -- runtime pow() and runtime bit scan.
//
// PDF 15 (p163-167): pow(x,10) with an integer exponent goes through a
// general (log/exp) path; a loop-based integer power is already better.
#include <cmath>
#include <cstdio>

double xpow10(double x) { return std::pow(x, 10); }   // general path

unsigned runtime_msb(unsigned long long n) {          // bit scan reverse
    unsigned b = 0;
    while (n >>= 1) ++b;
    return b;
}

int main() {
    volatile double r = xpow10(1.5);
    volatile unsigned b = runtime_msb(0xF0F0F0F0ULL);
    std::printf("checksums: %.4f %u\n", r, b);
    return 0;
}
