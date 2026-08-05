#include <cstddef>
#include <cstdio>
#include <string>
#include <tuple>

#include "test_utils.hpp"

namespace {

template <std::size_t Index, typename Tuple, typename Functor>
void tuple_at(const Tuple& tpl, const Functor& func) {
    func(std::get<Index>(tpl));
}

template <typename Tuple, typename Functor, std::size_t Index = 0>
void tuple_for_each(const Tuple& tpl, const Functor& f) {
    constexpr auto tuple_size = std::tuple_size_v<Tuple>;
    if constexpr (Index < tuple_size) {
        tuple_at<Index>(tpl, f);
        tuple_for_each<Tuple, Functor, Index + 1>(tpl, f);
    }
}

template <typename Tuple, typename Functor, std::size_t Index = 0>
bool tuple_any_of(const Tuple& tpl, const Functor& f) {
    constexpr auto tuple_size = std::tuple_size_v<Tuple>;
    if constexpr (Index < tuple_size) {
        return f(std::get<Index>(tpl))
                   ? true
                   : tuple_any_of<Tuple, Functor, Index + 1>(tpl, f);
    } else {
        return false;
    }
}

}  // namespace

int main() {
    // tuple access.
    auto t = std::make_tuple(42, std::string{"hi"}, true);
    CHP_CHECK(std::get<0>(t) == 42);
    CHP_CHECK(std::get<1>(t) == "hi");
    CHP_CHECK(std::get<2>(t) == true);
    static_assert(std::tuple_size_v<decltype(t)> == 3, "size 3");

    // Structured bindings.
    const auto& [n, s, b] = t;
    CHP_CHECK(n == 42 && s == "hi" && b == true);

    // tuple_for_each visits all elements (compile-time unrolled).
    int visited = 0;
    tuple_for_each(t, [&visited](const auto&) { ++visited; });
    CHP_CHECK(visited == 3);

    // tuple_any_of short-circuits.
    auto t2 = std::make_tuple(1, 2, 3);
    CHP_CHECK(tuple_any_of(t2, [](auto v) { return v == 2; }));
    CHP_CHECK(!tuple_any_of(t2, [](auto v) { return v == 99; }));

    // Empty tuple for_each is a no-op.
    std::tuple<> empty;
    int count = 0;
    tuple_for_each(empty, [&count](const auto&) { ++count; });
    CHP_CHECK(count == 0);

    return chp::test_summary("tuple_variadic");
}
