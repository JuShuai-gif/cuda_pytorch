// 16_division_optimization: optimized -- constant divisor (mul+shift),
// unsigned, power of two, float reciprocal.
//
// PDF 14.5-14.6 (p150-153): constant > variable, unsigned > signed,
// power of two = shift, float division -> multiply by reciprocal.
#include <cstdio>
#include <vector>

int sum_div_const(const std::vector<int>& v) {
    int s = 0;
    for (int x : v) s += x / 7;             // constant: compiler uses mul
    return s;
}

int sum_div_unsigned(const std::vector<int>& v) {
    int s = 0;
    for (int x : v) s += (int)((unsigned)x / 7u);   // unsigned: fewer fixes
    return s;
}

int sum_shift(const std::vector<int>& v) {
    int s = 0;
    for (int x : v) s += x >> 3;            // divide by 8 = shift (careful: negative)
    return s;
}

double sum_reciprocal(const std::vector<double>& v) {
    double s = 0.0;
    double inv = 1.0 / 7.0;                 // PDF p152: multiply by reciprocal
    for (double x : v) s += x * inv;
    return s;
}

int main() {
    std::vector<int> vi(8'000'000, 1000);
    std::vector<double> vd(8'000'000, 1000.0);
    volatile int r = sum_div_const(vi) + sum_shift(vi);
    volatile double d = sum_reciprocal(vd);
    std::printf("checksums = %d %.1f\n", r, d);
    return 0;
}
