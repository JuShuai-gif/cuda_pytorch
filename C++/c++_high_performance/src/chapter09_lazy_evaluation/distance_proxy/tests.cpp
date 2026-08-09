// Correctness checks for distance_proxy.

#include <cmath>
#include <cstdio>

#include "test_utils.hpp"

namespace {

class DistProxy {
public:
    DistProxy(float x0, float y0, float x1, float y1)
        : dist_sqrd_{std::pow(x0 - x1, 2.0F) + std::pow(y0 - y1, 2.0F)} {}
    auto operator==(const DistProxy& other) const {
        return dist_sqrd_ == other.dist_sqrd_;
    }
    auto operator<(const DistProxy& other) const {
        return dist_sqrd_ < other.dist_sqrd_;
    }
    auto operator<(float dist) const { return dist_sqrd_ < dist * dist; }
    operator float() const&& { return std::sqrt(dist_sqrd_); }

private:
    float dist_sqrd_{};
};

class Point {
public:
    Point(float x, float y) : x_(x), y_(y) {}
    auto distance(const Point& p) const {
        return DistProxy{x_, y_, p.x_, p.y_};
    }
    float x() const { return x_; }
    float y() const { return y_; }

private:
    float x_{};
    float y_{};
};

float distance_sqrt(const Point& a, const Point& b) {
    const float dx = a.x() - b.x();
    const float dy = a.y() - b.y();
    return std::sqrt(dx * dx + dy * dy);
}

}  // namespace

int main() {
    const Point target{3.0F, 5.0F};
    const Point a{6.0F, 9.0F};
    const Point b{7.0F, 4.0F};

    // Proxy comparisons agree with sqrt-based distances.
    CHP_CHECK((a.distance(target) < b.distance(target)) ==
              (distance_sqrt(a, target) < distance_sqrt(b, target)));

    // Threshold comparison uses squared distance under the hood.
    const float dist_ab = distance_sqrt(a, b);
    CHP_CHECK(a.distance(b) < dist_ab + 1.0F);
    CHP_CHECK(!(a.distance(b) < dist_ab - 1.0F));

    // Actual distance via implicit conversion on a temporary.
    const float d = a.distance(b);
    CHP_CHECK(std::abs(d - dist_ab) < 1e-5F);

    return chp::test_summary("distance_proxy");
}
