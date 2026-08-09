// Postponing std::sqrt when comparing distances (PDF p.265-273).
//
// Point::distance() returns a DistProxy holding the squared distance.
// Comparisons only involve dist*dist (no sqrt); the actual distance is
// computed via an implicit conversion to float that only works on r-values.

#include <cmath>
#include <cstdio>
#include <vector>

namespace chp9 {

class DistProxy {
public:
    DistProxy(float x0, float y0, float x1, float y1)
        : dist_sqrd_{std::pow(x0 - x1, 2.0F) + std::pow(y0 - y1, 2.0F)} {}

    // Comparisons operate on squared distances; no sqrt needed.
    auto operator==(const DistProxy& other) const {
        return dist_sqrd_ == other.dist_sqrd_;
    }
    auto operator<(const DistProxy& other) const {
        return dist_sqrd_ < other.dist_sqrd_;
    }
    auto operator<(float dist) const { return dist_sqrd_ < dist * dist; }

    // r-value only: converting a proxy stored in a variable is a compile
    // error, so std::sqrt is invoked at most once per proxy (PDF p.272).
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

}  // namespace chp9

namespace {

// Baseline: compare distances via std::sqrt (PDF p.265).
float distance_sqrt(const chp9::Point& a, const chp9::Point& b) {
    const float dx = a.x() - b.x();
    const float dy = a.y() - b.y();
    return std::sqrt(dx * dx + dy * dy);
}

}  // namespace

int main() {
    std::printf("== distance_proxy ==\n");

    const chp9::Point bingo{31.0F, 11.0F};
    const chp9::Point a{23.0F, 42.0F};
    const chp9::Point b{33.0F, 12.0F};

    // Same syntax as the naive version, no sqrt executed here.
    const bool a_is_nearest = a.distance(bingo) < b.distance(bingo);
    std::printf("a nearest to bingo: %d\n", a_is_nearest);
    std::printf("sqrt baseline: %f\n", distance_sqrt(a, bingo));

    // Actual distance: implicit conversion on a temporary, exactly once.
    const float dist = a.distance(b);
    std::printf("distance a-b: %f\n", dist);
    std::printf("consistent with baseline: %s\n",
                std::abs(dist - distance_sqrt(a, b)) < 1e-5 ? "yes" : "NO");

    // Threshold comparison: is the pair nearer than 20 units?
    std::printf("a-b within 20: %d\n", a.distance(b) < 20.0F);

    return 0;
}
