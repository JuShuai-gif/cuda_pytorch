// 05_function: optimized -- template + inline function in the hot loop.
//
// PDF 7.14 (p48-50): an inlineable template call lets the compiler inline
// the body and remove the call/return overhead entirely.
#include <cstdio>
#include <vector>

template <class F>
double apply_template(const std::vector<double>& v, F f) {
    double s = 0.0;
    for (double x : v) s += f(x);   // f is inlined: no call at all
    return s;
}

static inline double times2_plus1(double x) { return x * 2.0 + 1.0; }

int main() {
    std::vector<double> v(8'000'000, 1.0);
    volatile double r = apply_template(v, times2_plus1);
    std::printf("checksum = %.0f\n", r);
    return 0;
}
