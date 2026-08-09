// The pipe operator as an extension method (PDF p.274-275).
//
// C++ has no extension methods, but overloading operator| lets a library
// user write `range | contains(value)` instead of a free function call.

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace chp9 {

// Right-hand operand of the pipe expression. Holding a reference is safe
// because the proxy is consumed immediately by operator|.
template <typename T>
struct ContainsProxy {
    const T& value_;
};

// The pipe operator does the actual work: a linear search over the range.
template <typename Range, typename T>
auto operator|(const Range& range, const ContainsProxy<T>& proxy) {
    return std::find(range.begin(), range.end(), proxy.value_) != range.end();
}

// Convenience factory so callers never write ContainsProxy<Type> explicitly.
template <typename T>
auto contains(const T& value) {
    return ContainsProxy<T>{value};
}

}  // namespace chp9

int main() {
    std::printf("== pipe_operator ==\n");

    const std::vector<int> numbers{1, 3, 5, 7, 9};
    std::printf("7 in numbers? %d\n", numbers | chp9::contains(7));
    std::printf("8 in numbers? %d\n", numbers | chp9::contains(8));

    // Works for any element type and any range with begin()/end().
    const std::vector<std::string> penguins{"Ping", "Roy", "Silo"};
    std::printf("\"Silo\" in penguins? %d\n",
                penguins | chp9::contains(std::string{"Silo"}));
    std::printf("\"Kowalski\" in penguins? %d\n",
                penguins | chp9::contains(std::string{"Kowalski"}));

    return 0;
}
