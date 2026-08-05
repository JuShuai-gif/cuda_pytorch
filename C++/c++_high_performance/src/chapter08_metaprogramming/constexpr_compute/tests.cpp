#include <cmath>
#include <cstdio>
#include <type_traits>

#include "test_utils.hpp"

namespace {

constexpr int sum(int x, int y, int z) { return x + y + z; }

template <typename T>
T generic_mod(const T& v, const T& n) {
    if constexpr (std::is_floating_point_v<T>) {
        return static_cast<T>(std::fmod(v, n));
    } else {
        return v % n;
    }
}

}  // namespace

int main() {
    // constexpr with compile-time inputs is a constant.
    static_assert(sum(3, 4, 5) == 12, "compile-time sum");
    CHP_CHECK(sum(3, 4, 5) == 12);

    // Compile-time values can feed template/static_assert.
    constexpr int kSum = sum(1, 2, 3);
    static_assert(kSum == 6, "integral constant style");

    // generic_mod: int -> %, float -> fmod.
    static_assert(std::is_same_v<decltype(generic_mod(17, 5)), int>,
                  "int overload");
    static_assert(std::is_same_v<decltype(generic_mod(17.5F, 5.0F)), float>,
                  "float overload");
    CHP_CHECK(generic_mod(17, 5) == 2);
    CHP_CHECK(std::fabs(generic_mod(17.5F, 5.0F) - 2.5F) < 1e-6F);

    // float generic_mod must NOT contain the % branch: verifies if constexpr.
    static_assert(std::is_floating_point_v<decltype(generic_mod(1.0, 2.0))>,
                  "double");

    return chp::test_summary("constexpr_compute");
}
