// 05_function: compare call mechanisms -- plain, inline, static, virtual,
// function pointer, std::function, lambda/template.
//
// PDF 7.14-7.17 (p48-52): call overhead sources and ways to reduce them.
#include <cstdio>
#include <functional>
#include <vector>

#include "common/benchmark.h"

double plain_call(double x) { return x * 2.0 + 1.0; }
static double static_call(double x) { return x * 2.0 + 1.0; }
inline double inline_call(double x) { return x * 2.0 + 1.0; }

struct Base {
    virtual ~Base() = default;
    virtual double f(double x) const { return x * 2.0 + 1.0; }
};
struct Derived : Base {
    double f(double x) const override { return x * 2.0 + 1.0; }
};

template <class F>
double loop(const std::vector<double>& v, F f) {
    double s = 0.0;
    for (double x : v) s += f(x);
    return s;
}

int main() {
    std::vector<double> v(8'000'000, 1.0);

    bench("plain_call",      [&] { return loop(v, plain_call); });
    bench("static_call",     [&] { return loop(v, static_call); });
    bench("inline_call",     [&] { return loop(v, inline_call); });

    double (*fp)(double) = plain_call;
    bench("func_pointer",    [&] { return loop(v, fp); });

    std::function<double(double)> sf = plain_call;
    bench("std_function",    [&] { return loop(v, sf); });

    Derived d;
    const Base& b = d;
    bench("virtual_call",    [&] { return loop(v, [&](double x) { return b.f(x); }); });

    auto lambda = [](double x) { return x * 2.0 + 1.0; };
    bench("lambda_template", [&] { return loop(v, lambda); });

    std::printf("\nresults equal: %.0f\n", loop(v, plain_call));
    return 0;
}
