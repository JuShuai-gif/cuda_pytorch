#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "test_utils.hpp"

namespace {

template <typename T>
T pow_n(const T& v, int n) {
    T product = T{1};
    for (int i = 0; i < n; ++i) {
        product *= v;
    }
    return product;
}

template <typename T, int N>
constexpr T const_pow_n(const T& v) {
    static_assert(N >= 0, "N must be non-negative");
    T product = T{1};
    for (int i = 0; i < N; ++i) {
        product *= v;
    }
    return product;
}

}  // namespace

int main() {
    // pow_n works for both float and int.
    CHP_CHECK((pow_n(2.0F, 3)) == 8.0F);
    CHP_CHECK((pow_n(3, 3)) == 27);

    // Non-type template parameter: same value, different N.
    CHP_CHECK((const_pow_n<float, 2>(4.0F)) == 16.0F);
    CHP_CHECK((const_pow_n<float, 3>(4.0F)) == 64.0F);

    // static_assert conditions hold.
    static_assert(std::is_floating_point<float>::value, "float is fp");

    // Template instantiation is checked at compile time.
    static_assert(const_pow_n<int, 2>(3) == 9, "compile-time compute");

    // A compile-time value can feed a template argument.
    constexpr int kExponent = 2;
    CHP_CHECK((const_pow_n<int, kExponent>(5)) == 25);

    return chp::test_summary("templates_basics");
}
