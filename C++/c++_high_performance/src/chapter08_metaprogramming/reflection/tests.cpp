#include <cstddef>
#include <cstdio>
#include <string>
#include <tuple>
#include <utility>

#include <experimental/type_traits>

#include "test_utils.hpp"

namespace {

template <typename Tuple, typename Functor, std::size_t Index = 0>
void tuple_for_each(const Tuple& tpl, const Functor& f) {
    constexpr auto tuple_size = std::tuple_size_v<Tuple>;
    if constexpr (Index < tuple_size) {
        f(std::get<Index>(tpl));
        tuple_for_each<Tuple, Functor, Index + 1>(tpl, f);
    }
}

class Town {
public:
    Town(std::size_t houses, std::size_t settlers, std::string name)
        : houses_(houses), settlers_(settlers), name_(std::move(name)) {}

    auto reflect() const { return std::tie(houses_, settlers_, name_); }

private:
    std::size_t houses_;
    std::size_t settlers_;
    std::string name_;
};

template <typename T>
using has_reflect_member = decltype(&T::reflect);
namespace exp = std::experimental;
template <typename T>
constexpr bool is_reflectable_v =
    exp::is_detected<has_reflect_member, T>::value;

template <typename T, bool IsReflectable = is_reflectable_v<T>>
auto operator==(const T& a, const T& b)
    -> std::enable_if_t<IsReflectable, bool> {
    return a.reflect() == b.reflect();
}

template <typename T, bool IsReflectable = is_reflectable_v<T>>
auto operator!=(const T& a, const T& b)
    -> std::enable_if_t<IsReflectable, bool> {
    return a.reflect() != b.reflect();
}

template <typename T, bool IsReflectable = is_reflectable_v<T>>
auto operator<(const T& a, const T& b)
    -> std::enable_if_t<IsReflectable, bool> {
    return a.reflect() < b.reflect();
}

}  // namespace

int main() {
    const Town a{100, 200, "Shire"};
    const Town b{1000, 2000, "Mordor"};
    const Town c{100, 200, "Shire"};

    CHP_CHECK(a == c);
    CHP_CHECK(a != b);
    CHP_CHECK(a < b);
    CHP_CHECK(!(b < a));

    // A non-reflectable type does not get the operators.
    static_assert(is_reflectable_v<Town>, "Town is reflectable");
    static_assert(!is_reflectable_v<int>, "int is not reflectable");

    return chp::test_summary("reflection");
}
