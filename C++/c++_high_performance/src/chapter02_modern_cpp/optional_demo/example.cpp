// std::optional: a stack-allocated container of at most one element.
//
// The book (PDF p.80-82) points out that std::optional<T> is a small
// stack-allocated wrapper whose memory overhead over T is one bool (plus
// padding), and that empty optionals sort before non-empty ones.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

namespace {

struct Point {
    double x = 0.0;
    double y = 0.0;
};

// Two lines intersect in at most one point -> return an optional.
std::optional<Point> intersection(const Point& a1, const Point& b1,
                                  const Point& a2, const Point& b2) {
    // Simplified: compute the line from a to b and return a point only if
    // the second line crosses it (mock logic for the demo).
    const double denom = (b2.y - a2.y) * (b1.x - a1.x) -
                         (b2.x - a2.x) * (b1.y - a1.y);
    if (denom == 0.0) {
        return std::nullopt;  // parallel lines, no intersection
    }
    return Point{a1.x + (b1.x - a1.x) * 0.5, a1.y + (b1.y - a1.y) * 0.5};
}

struct Hat {
    std::string color;
};

class Head {
public:
    void set_hat(const Hat& h) { hat_ = h; }
    bool has_hat() const { return hat_.has_value(); }
    const Hat& get_hat() const {
        assert(hat_.has_value());
        return *hat_;
    }
    void remove_hat() { hat_ = {}; }

private:
    std::optional<Hat> hat_;
};

}  // namespace

int main() {
    std::printf("== optional_demo ==\n");
    std::printf("sizeof(optional<int>) = %zu, sizeof(int) = %zu\n",
                sizeof(std::optional<int>), sizeof(int));
    std::printf("sizeof(optional<Point>) = %zu, sizeof(Point) = %zu\n",
                sizeof(std::optional<Point>), sizeof(Point));

    // Optional return values.
    {
        auto p = intersection(Point{0, 0}, Point{1, 1}, Point{0, 0}, Point{0, 1});
        if (p.has_value()) {
            std::printf("intersection: (%g, %g)\n", p->x, p->y);
        } else {
            std::printf("no intersection\n");
        }
        auto q = intersection(Point{0, 0}, Point{1, 1}, Point{1, 1}, Point{2, 2});
        std::printf("q has value: %d\n", q.has_value());
    }

    // Optional member variables.
    {
        Head head;
        std::printf("head has hat: %d\n", head.has_hat());
        head.set_hat(Hat{"red"});
        std::printf("head has hat: %d, color %s\n", head.has_hat(),
                    head.get_hat().color.c_str());
        head.remove_hat();
        std::printf("head has hat after removal: %d\n", head.has_hat());
    }

    // Sorting: empty optionals come first (book PDF p.82).
    {
        std::vector<std::optional<int>> c{{3}, {}, {1}, {}, {2}};
        std::sort(c.begin(), c.end());
        std::printf("sorted optionals:");
        for (const auto& v : c) {
            std::printf(" %s", v ? std::to_string(*v).c_str() : "_");
        }
        std::printf("\n");
    }

    return 0;
}
