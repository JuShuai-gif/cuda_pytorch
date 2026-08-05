// 06_class_virtual: member functions, virtual vs non-virtual, RTTI cost.
//
// PDF 7.19-7.25 (p52-57). Note: the compiler may devirtualize some calls
// (PDF p76); check the assembly.
#include <cstdio>
#include <vector>

#include "common/benchmark.h"

// Non-virtual member function.
struct Plain {
    double a, b;
    double sum() const { return a + b; }
};

// Virtual member function hierarchy.
struct Base {
    virtual ~Base() = default;
    virtual double f(double x) const = 0;
};
struct Impl : Base {
    double a = 1.0;
    double f(double x) const override { return a + x; }
};

// RTTI check cost: dynamic_cast on a known type.
double rtti_check(const Base* b, double x) {
    if (auto* p = dynamic_cast<const Impl*>(b)) return p->f(x);
    return 0.0;
}

int main() {
    std::vector<Plain> plains(4'000'000, Plain{1.0, 2.0});
    bench("plain_method", [&] {
        double s = 0.0;
        for (const auto& p : plains) s += p.sum();
        return s;
    });

    Impl impl;
    const Base* b = &impl;
    bench("virtual_call", [&] { return b->f(3.0); });
    bench("rtti_dyncast", [&] { return rtti_check(b, 3.0); });

    std::printf("\ncheck: %.2f %.2f %.2f\n",
                plains[0].sum(), b->f(3.0), rtti_check(b, 3.0));
    return 0;
}
