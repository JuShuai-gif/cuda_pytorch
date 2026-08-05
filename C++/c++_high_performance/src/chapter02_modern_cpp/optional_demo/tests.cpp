#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

#include "test_utils.hpp"

namespace {

struct Point {
    double x = 0.0;
    double y = 0.0;
};

std::optional<Point> intersection(const Point& a1, const Point& b1,
                                  const Point& a2, const Point& b2) {
    const double denom = (b2.y - a2.y) * (b1.x - a1.x) -
                         (b2.x - a2.x) * (b1.y - a1.y);
    if (denom == 0.0) {
        return std::nullopt;
    }
    return Point{a1.x + (b1.x - a1.x) * 0.5, a1.y + (b1.y - a1.y) * 0.5};
}

}  // namespace

int main() {
    // Empty vs non-empty.
    std::optional<int> empty;
    std::optional<int> value{4};
    CHP_CHECK(!empty.has_value());
    CHP_CHECK(value.has_value());
    CHP_CHECK(*value == 4);

    // Accessing an empty optional throws std::bad_optional_access.
    // (Note: operator* on an empty optional is UB; only value() throws.)
    bool threw = false;
    try {
        (void)empty.value();
    } catch (const std::bad_optional_access&) {
        threw = true;
    }
    CHP_CHECK(threw);

    // Comparison rules (book PDF p.82).
    CHP_CHECK(std::optional<int>{} == std::optional<int>{});
    CHP_CHECK(std::optional<int>{} != std::optional<int>{4});
    CHP_CHECK(std::optional<int>{} < std::optional<int>{4});
    CHP_CHECK(std::optional<int>{4} < std::optional<int>{5});

    // Sorting puts empty optionals first.
    std::vector<std::optional<int>> c{{3}, {}, {1}, {}, {2}};
    std::sort(c.begin(), c.end());
    CHP_CHECK(!c[0].has_value());
    CHP_CHECK(!c[1].has_value());
    CHP_CHECK(c[2].has_value() && *c[2] == 1);
    CHP_CHECK(c[4].has_value() && *c[4] == 3);

    // Optional return value.
    auto p = intersection(Point{0, 0}, Point{1, 1}, Point{0, 0}, Point{0, 1});
    CHP_CHECK(p.has_value());
    auto q = intersection(Point{0, 0}, Point{1, 1}, Point{2, 2}, Point{3, 3});
    CHP_CHECK(!q.has_value());

    // Memory overhead is one bool (plus padding).
    CHP_CHECK(sizeof(std::optional<int>) == sizeof(int) + 1 ||
              sizeof(std::optional<int>) == 2 * sizeof(int));

    return chp::test_summary("optional_demo");
}
