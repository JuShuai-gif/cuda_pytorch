// Minimal reflection: iterate a class's members via std::tie.
//
// The book (PDF p.242-248): a class exposes its members through reflect().
// Generic functions (operator==, operator<, operator<<) are then generated
// for every "reflectable" type using is_detected + enable_if_t.

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>

#include <experimental/type_traits>

namespace {

// tuple_for_each (from the tuple_variadic experiment) unrolls the tuple.
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

    // Reflection: expose members as a tuple of references (PDF p.243).
    auto reflect() const { return std::tie(houses_, settlers_, name_); }

private:
    std::size_t houses_;
    std::size_t settlers_;
    std::string name_;
};

// Detect reflect() (PDF p.246).
template <typename T>
using has_reflect_member = decltype(&T::reflect);
namespace exp = std::experimental;
template <typename T>
constexpr bool is_reflectable_v =
    exp::is_detected<has_reflect_member, T>::value;

// Generic operators for every reflectable type (PDF p.246-247).
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

template <typename T, bool IsReflectable = is_reflectable_v<T>>
auto operator<<(std::ostream& ostr, const T& v)
    -> std::enable_if_t<IsReflectable, std::ostream&> {
    tuple_for_each(v.reflect(), [&ostr](const auto& m) {
        ostr << m << " ";
    });
    return ostr;
}

}  // namespace

int main() {
    std::printf("== reflection ==\n");

    const Town shire{100, 200, "Shire"};
    const Town mordor{1000, 2000, "Mordor"};

    std::printf("shire: ");
    std::cout << shire;
    std::printf("\nmordor: ");
    std::cout << mordor;
    std::printf("\n");

    std::printf("shires == mordor? %d\n", shire == mordor);
    std::printf("shire < mordor?  %d\n", shire < mordor);
    std::printf("shire == copy?   %d\n", shire == Town{100, 200, "Shire"});

    return 0;
}
