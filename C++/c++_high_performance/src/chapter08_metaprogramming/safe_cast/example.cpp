// Generic safe cast: type-checked casts with if constexpr.
//
// The book (PDF p.249-251): safe_cast() performs different verification per
// cast kind, and fails to compile for unsupported casts (via static_assert
// + make_false<T> so the error is delayed until instantiation).

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <type_traits>
#include <typeinfo>

namespace {

template <typename T>
constexpr bool make_false() {
    return false;
}

template <typename Dst, typename Src>
Dst safe_cast(const Src& v) {
    using namespace std;
    constexpr auto is_same_type = is_same_v<Src, Dst>;
    constexpr auto is_pointer_to_pointer = is_pointer_v<Src> && is_pointer_v<Dst>;
    constexpr auto is_float_to_float =
        is_floating_point_v<Src> && is_floating_point_v<Dst>;
    constexpr auto is_number_to_number =
        is_arithmetic_v<Src> && is_arithmetic_v<Dst>;
    constexpr auto is_ptr_to_intptr =
        is_pointer_v<Src> && (is_same_v<uintptr_t, Dst> || is_same_v<intptr_t, Dst>);

    if constexpr (is_same_type) {
        return v;
    } else if constexpr (is_ptr_to_intptr) {
        return reinterpret_cast<Dst>(v);
    } else if constexpr (is_pointer_to_pointer) {
        return static_cast<Dst>(v);
    } else if constexpr (is_float_to_float) {
        auto casted = static_cast<Dst>(v);
        auto casted_back = static_cast<Src>(casted);
        assert(casted_back == casted_back);  // reject NaN
        (void)casted_back;
        return casted;
    } else if constexpr (is_number_to_number) {
        auto casted = static_cast<Dst>(v);
        auto casted_back = static_cast<Src>(casted);
        assert(casted == casted_back);
        (void)casted_back;
        return casted;
    } else {
        static_assert(make_false<Src>(), "Unsupported cast");
        return Dst{};
    }
}

}  // namespace

int main() {
    std::printf("== safe_cast ==\n");

    // Same type: identity.
    const int same = safe_cast<int>(42);
    std::printf("same type: %d\n", same);

    // Number to number (checked round-trip).
    const int n = safe_cast<int>(42.0F);
    std::printf("float->int: %d\n", n);

    // int to int64_t: no precision loss.
    const std::int64_t big = safe_cast<std::int64_t>(123456);
    std::printf("int->int64: %lld\n", static_cast<long long>(big));

    // Pointer to uintptr_t (the only guaranteed integer that holds an address).
    int x = 7;
    const std::uintptr_t addr = safe_cast<std::uintptr_t>(&x);
    std::printf("ptr->uintptr_t: %p\n", reinterpret_cast<void*>(addr));

    // Unsupported cast (e.g. pointer to non-uintptr int) does not compile.
    // (comment out to demonstrate)
    // auto bad = safe_cast<int>(&x);  // static_assert fails

    return 0;
}
