// 06_class_virtual: baseline -- virtual dispatch + dynamic_cast + RTTI.
//
// PDF 7.22-7.24 (p55-57): virtual calls go through the vtable; dynamic_cast
// performs a runtime check; RTTI adds bookkeeping.
#include <cstdio>
#include <vector>

struct Shape {
    virtual ~Shape() = default;
    virtual double area(double x) const = 0;
};
struct Square : Shape {
    double area(double x) const override { return x * x; }
};
struct Circle : Shape {
    double area(double x) const override { return x * x * 3.14159; }
};

double virtual_sum(const std::vector<Shape*>& shapes, double x) {
    double s = 0.0;
    for (const Shape* sh : shapes) s += sh->area(x);   // vtable dispatch
    return s;
}

double dynamic_cast_sum(const std::vector<Shape*>& shapes, double x) {
    double s = 0.0;
    for (const Shape* sh : shapes) {
        if (auto* sq = dynamic_cast<const Square*>(sh)) s += sq->area(x);
        else if (auto* ci = dynamic_cast<const Circle*>(sh)) s += ci->area(x);
    }
    return s;
}

int main() {
    std::vector<Shape*> shapes;
    for (int i = 0; i < 2'000'000; ++i) {
        shapes.push_back(i % 2 ? (Shape*)new Square : (Shape*)new Circle);
    }
    volatile double r1 = virtual_sum(shapes, 3.0);
    volatile double r2 = dynamic_cast_sum(shapes, 3.0);
    std::printf("vsum=%.2f dsum=%.2f\n", r1, r2);
    for (Shape* s : shapes) delete s;
    return 0;
}
