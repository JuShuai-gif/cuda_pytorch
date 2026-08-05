// 02_integer_float: compare int32/int64, float/double, division, conversions.
//
// PDF 7.2 (p29-31) and 7.3 (p31-33):
//   * non-vector float vs double: same speed for +,-,*; division is slower
//     in double on some CPUs;
//   * integer division by a variable is expensive;
//   * avoid int<->float conversions in hot loops.
#include <cstdint>
#include <cstdio>
#include <vector>

#include "common/benchmark.h"

// --- integer arithmetic -----------------------------------------------------
int32_t sum_int32(const std::vector<int32_t>& v, int32_t d) {
    int32_t s = 0;
    for (int32_t x : v) s += x / d;   // variable divisor: slow idiv
    return s;
}

int32_t sum_int32_const(const std::vector<int32_t>& v) {
    int32_t s = 0;
    for (int32_t x : v) s += x / 10;  // constant: compiler emits mul+shift
    return s;
}

int64_t sum_int64(const std::vector<int64_t>& v) {
    int64_t s = 0;
    for (int64_t x : v) s += x;
    return s;
}

// --- float vs double --------------------------------------------------------
double sum_double(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x * 1.0001;
    return s;
}

float sum_float(const std::vector<float>& v) {
    float s = 0.0f;
    for (float x : v) s += x * 1.0001f;
    return s;
}

// --- conversions in the hot loop -------------------------------------------
double conv_loop(const std::vector<int>& v) {
    double s = 0.0;
    for (int x : v) s += static_cast<double>(x);   // int->double every step
    return s;
}

double no_conv_loop(const std::vector<int>& v) {
    double s = 0.0;
    int i = 0;
    for (int x : v) {                              // avoid per-element cast
        s += x;                                    // promoted once, not cast
        (void)i;
    }
    return s;
}

int main() {
    const size_t n = 8'000'000;
    std::vector<int32_t> vi32(n, 1000);
    std::vector<int64_t> vi64(n, 1000);
    std::vector<double> vd(n, 1.0);
    std::vector<float> vf(n, 1.0f);
    std::vector<int> vint(n, 1000);

    bench("sum_int32_div_var",  [&] { return sum_int32(vi32, 10); });
    bench("sum_int32_div_const",[&] { return sum_int32_const(vi32); });
    bench("sum_int64_add",      [&] { return sum_int64(vi64); });
    bench("sum_double_mul",     [&] { return sum_double(vd); });
    bench("sum_float_mul",      [&] { return sum_float(vf); });
    bench("conv_loop",          [&] { return conv_loop(vint); });
    bench("no_conv_loop",       [&] { return no_conv_loop(vint); });

    std::printf("\nresults: %d %d %lld %.3f %.3f %.3f %.3f\n",
                sum_int32(vi32, 10), sum_int32_const(vi32),
                (long long)sum_int64(vi64), sum_double(vd), sum_float(vf),
                conv_loop(vint), no_conv_loop(vint));
    return 0;
}
