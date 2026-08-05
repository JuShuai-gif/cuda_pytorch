// constexpr, if constexpr, and verifying compile-time evaluation.
//
// The book (PDF p.222-227): constexpr functions can run at compile time when
// their inputs are known. if constexpr eradicates dead branches (unlike a
// plain if, which still compiles every branch). std::integral_constant can
// PROVE a computation happened at compile time.

#include <cmath>
#include <cstdio>
#include <type_traits>

namespace {

constexpr int sum(int x, int y, int z) { return x + y + z; }

// if constexpr: the else branch is eradicated for non-matching types
// (book PDF p.226 generic_mod example).
template <typename T>
T generic_mod(const T& v, const T& n) {
    if constexpr (std::is_floating_point_v<T>) {
        return static_cast<T>(std::fmod(v, n));
    } else {
        return v % n;
    }
}

// A compile-time hash (sum of chars). This is a BAD real-world hash, but a
// good teaching example (book PDF p.254).
constexpr std::size_t hash_function(const char* str) {
    std::size_t sum = 0;
    for (auto ptr = str; *ptr != '\0'; ++ptr) {
        sum += static_cast<std::size_t>(*ptr);
    }
    return sum;
}

}  // namespace

int main() {
    std::printf("== constexpr_compute ==\n");

    // Compile-time evaluation when inputs are known.
    constexpr int s = sum(3, 4, 5);
    static_assert(s == 12, "sum evaluated at compile time");

    // std::integral_constant proves compile-time evaluation (PDF p.223):
    // the template argument must be a constant expression.
    const auto kcompile = std::integral_constant<int, sum(1, 2, 3)>{};
    static_assert(kcompile.value == 6, "compile-time constant");
    std::printf("integral_constant<sum(1,2,3)> = %d (compiled)\n",
                kcompile.value);

    // constexpr also works at runtime when inputs are not known.
    int a = 10, b = 20, c = 30;
    const auto runtime = sum(a, b, c);
    std::printf("runtime sum=%d\n", runtime);

    // if constexpr: generic_mod works for both int and float.
    const int mi = generic_mod(17, 5);
    const float mf = generic_mod(17.5F, 5.0F);
    std::printf("generic_mod(17,5)=%d generic_mod(17.5,5)=%g\n", mi, mf);

    // Compile-time hash: verify value, then check assembly to see it folded.
    constexpr std::size_t h = hash_function("abc");
    static_assert(h == 294, "97+98+99");
    std::printf("hash_function(\"abc\") = %zu (compile-time constant)\n", h);

    return 0;
}
