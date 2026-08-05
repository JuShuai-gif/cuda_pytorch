// 05_function: baseline -- calls through std::function in a hot loop.
//
// PDF 7.14 (p48-50): function call overhead; std::function adds indirection
// (type-erased call) that the compiler often cannot inline away.
#include <cstdio>
#include <functional>
#include <vector>

double apply_std_function(const std::vector<double>& v,
                          const std::function<double(double)>& f) {
    double s = 0.0;
    for (double x : v) s += f(x);
    return s;
}

int main() {
    std::vector<double> v(8'000'000, 1.0);
    std::function<double(double)> f = [](double x) { return x * 2.0 + 1.0; };
    volatile double r = apply_std_function(v, f);
    std::printf("checksum = %.0f\n", r);
    return 0;
}
