// 16_division_optimization: measure all division variants and verify the
// compiler replaced the constant division with a multiply.
#include <cstdio>
#include <vector>

#include "common/benchmark.h"

int div_var(const std::vector<int>& v, int d) {
    int s = 0;
    for (int x : v) s += x / d;
    return s;
}
int div_const(const std::vector<int>& v) {
    int s = 0;
    for (int x : v) s += x / 7;
    return s;
}
int div_unsigned(const std::vector<int>& v) {
    int s = 0;
    for (int x : v) s += (int)((unsigned)x / 7u);
    return s;
}
int div_pow2(const std::vector<int>& v) {
    int s = 0;
    for (int x : v) s += x / 8;
    return s;
}
double fdiv_var(const std::vector<double>& v, double d) {
    double s = 0.0;
    for (double x : v) s += x / d;
    return s;
}
double fdiv_recip(const std::vector<double>& v) {
    double s = 0.0;
    double inv = 1.0 / 7.0;
    for (double x : v) s += x * inv;
    return s;
}

int main() {
    std::vector<int> vi(8'000'000, 1000);
    std::vector<double> vd(8'000'000, 1000.0);

    bench("int_div_variable",  [&] { return div_var(vi, 7); });
    bench("int_div_const",     [&] { return div_const(vi); });
    bench("int_div_unsigned",  [&] { return div_unsigned(vi); });
    bench("int_div_pow2",      [&] { return div_pow2(vi); });
    bench("float_div_variable",[&] { return fdiv_var(vd, 7.0); });
    bench("float_div_recip",   [&] { return fdiv_recip(vd); });

    std::printf("\nresults equal (int): %d %d %d %d\n",
                div_var(vi,7), div_const(vi), div_unsigned(vi), div_pow2(vi));
    std::printf("results equal (float): %.4f %.4f\n",
                fdiv_var(vd,7.0), fdiv_recip(vd));
    return 0;
}
