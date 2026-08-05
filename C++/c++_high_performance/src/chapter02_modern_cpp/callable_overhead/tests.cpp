#include <functional>

#include "test_utils.hpp"

namespace {

struct TimesThree {
    int operator()(int v) const { return v * 3; }
};

}  // namespace

int main() {
    // A plain lambda assigned into a std::function (book PDF p.57).
    std::function<int(int)> f = [](int v) { return v * 3; };
    CHP_CHECK(f(7) == 21);

    // A functor assigned into a std::function computes the same result.
    std::function<int(int)> g = TimesThree{};
    CHP_CHECK(g(7) == 21);

    // Reassignment at runtime works for std::function (book PDF p.57).
    g = [](int v) { return v * 5; };
    CHP_CHECK(g(7) == 35);
    g = TimesThree{};
    CHP_CHECK(g(7) == 21);

    // Lambda vs functor equivalence for the same computation.
    auto l = [](int v) { return v * 3; };
    CHP_CHECK(l(9) == TimesThree{}(9));

    return chp::test_summary("callable_overhead");
}
