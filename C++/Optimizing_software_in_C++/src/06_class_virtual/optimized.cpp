// 06_class_virtual: optimized -- compile-time polymorphism with templates.
//
// PDF 7.30 (p59-61) and 8.1 devirtualization (p76): a template parameter is
// resolved at compile time; the call is a plain (inlineable) function.
#include <cstdio>
#include <vector>

template <class Shape>
double template_sum(const std::vector<Shape>& shapes, double x) {
    double s = 0.0;
    for (const auto& sh : shapes) s += sh.area(x);  // inlined, no vtable
    return s;
}

struct Square { double area(double x) const { return x * x; } };
struct Circle { double area(double x) const { return x * x * 3.14159; } };

int main() {
    std::vector<Square> squares(1'000'000);
    std::vector<Circle> circles(1'000'000);
    volatile double r1 = template_sum(squares, 3.0);
    volatile double r2 = template_sum(circles, 3.0);
    std::printf("sum_sq=%.2f sum_ci=%.2f\n", r1, r2);
    return 0;
}
