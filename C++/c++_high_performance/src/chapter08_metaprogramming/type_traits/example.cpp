// Type traits, decltype, enable_if_t, and is_detected.
//
// The book (PDF p.214-221): type traits answer questions about types at
// compile time. enable_if_t conditionally enables a template function.
// is_detected (experimental) inspects whether a type has a given member.

#include <cstdint>
#include <cstdio>
#include <string>
#include <type_traits>
#include <vector>

#include <experimental/type_traits>

namespace exp = std::experimental;

namespace {

// --- is_detected: does the type have this member? (PDF p.219-220) ---
struct Octopus {
    void mess_with_arms() {}
};
struct Whale {
    void blow_a_fountain() {}
};
struct Shark {
    using fin_type = float;
};
struct Eel {
    int electricity_ = 0;
};

template <typename T>
using has_mess_with_arms = decltype(&T::mess_with_arms);
template <typename T>
using has_blow_a_fountain = decltype(&T::blow_a_fountain);
template <typename T>
using has_fin_type = typename T::fin_type;
template <typename T>
using has_electricity = decltype(T::electricity_);

// --- enable_if_t + is_detected combined (PDF p.220-221) ---
template <typename T>
using has_to_string = decltype(&T::to_string);
template <typename T>
using has_name_member = decltype(T::name_);

struct Squid {
    std::string to_string() const { return "Steve the Squid"; }
};
struct Salmon {
    Salmon() : name_("Jeff the Salmon") {}
    std::string name_;
};

// Print via to_string() if present.
template <typename T,
          bool HasToString = exp::is_detected<has_to_string, T>::value,
          bool HasName = exp::is_detected<has_name_member, T>::value>
auto print(const T& v) -> std::enable_if_t<HasToString && !HasName, void> {
    std::printf("%s\n", v.to_string().c_str());
}

// Print via name_ if present.
template <typename T,
          bool HasToString = exp::is_detected<has_to_string, T>::value,
          bool HasName = exp::is_detected<has_name_member, T>::value>
auto print(const T& v) -> std::enable_if_t<HasName && !HasToString, void> {
    std::printf("%s\n", v.name_.c_str());
}

}  // namespace

int main() {
    std::printf("== type_traits ==\n");

    // Boolean type traits.
    std::printf("is_same_v<uint8_t, unsigned char> = %d\n",
                std::is_same_v<std::uint8_t, unsigned char>);
    std::printf("is_floating_point_v<float> = %d\n",
                std::is_floating_point_v<float>);

    // Type-transforming traits.
    using value_type = std::remove_pointer_t<int*>;
    static_assert(std::is_same_v<value_type, int>, "remove_pointer");
    using ptr_type = std::add_pointer_t<float>;
    static_assert(std::is_same_v<ptr_type, float*>, "add_pointer");

    // decltype + remove_reference on a lambda parameter (PDF p.216-217).
    auto sign_func = [](const auto& v) -> int {
        using RefType = decltype(v);
        using ValueType = std::remove_reference_t<RefType>;
        if constexpr (std::is_unsigned_v<ValueType>) {
            return 1;
        }
        return v < 0 ? -1 : 1;
    };
    const std::uint32_t u = 32;
    const std::int32_t i = -32;
    std::printf("sign(u32)=%d sign(i32)=%d\n", sign_func(u), sign_func(i));

    // is_detected on classes (PDF p.219-220).
    static_assert(exp::is_detected<has_mess_with_arms, Octopus>::value, "");
    static_assert(!exp::is_detected<has_mess_with_arms, Whale>::value, "");
    static_assert(exp::is_detected<has_fin_type, Shark>::value, "");
    static_assert(exp::is_detected<has_electricity, Eel>::value, "");

    // enable_if + is_detected: print() dispatches (PDF p.220-221).
    print(Squid{});
    print(Salmon{});

    std::printf("type_traits checks passed\n");
    return 0;
}
