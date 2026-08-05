#include <cstdint>
#include <cstdio>
#include <string>
#include <type_traits>
#include <vector>

#include <experimental/type_traits>

#include "test_utils.hpp"

namespace {

struct Octopus {
    void mess_with_arms() {}
};
struct Whale {
    void blow_a_fountain() {}
};

template <typename T>
using has_mess_with_arms = decltype(&T::mess_with_arms);
template <typename T>
using has_blow_a_fountain = decltype(&T::blow_a_fountain);

// sign_func: unsigned always positive, signed checks sign (PDF p.215).
template <typename T>
int sign_func(const T& v) {
    if (std::is_unsigned_v<T>) {
        return 1;
    }
    return v < 0 ? -1 : 1;
}

}  // namespace

int main() {
    // Boolean traits.
    static_assert(std::is_same_v<std::uint8_t, unsigned char>, "uint8_t");
    static_assert(std::is_floating_point_v<float>, "float is fp");
    static_assert(!std::is_floating_point_v<int>, "int is not fp");

    // Type transforms.
    static_assert(std::is_same_v<std::remove_pointer_t<int*>, int>, "rp");
    static_assert(std::is_same_v<std::add_pointer_t<float>, float*>, "ap");
    static_assert(std::is_same_v<std::decay_t<int&>, int>, "decay");

    // is_detected.
    static_assert(std::experimental::is_detected<has_mess_with_arms,
                                                 Octopus>::value,
                  "Octopus messes with arms");
    static_assert(!std::experimental::is_detected<has_mess_with_arms,
                                                  Whale>::value,
                  "Whale does not");

    // sign_func.
    CHP_CHECK(sign_func(std::uint32_t{32}) == 1);
    CHP_CHECK(sign_func(std::int32_t{-32}) == -1);
    CHP_CHECK(sign_func(std::int32_t{32}) == 1);

    return chp::test_summary("type_traits");
}
