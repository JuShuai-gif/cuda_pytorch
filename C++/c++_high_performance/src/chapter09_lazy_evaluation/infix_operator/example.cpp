// The infix operator: emulate Python's `in` keyword (PDF p.275-277).
//
// Overloading operator< and operator> on a pair of tag/proxy types makes
// `"Botswana" <in> africa` legal C++, expanding to two ordinary calls.

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace chp9 {

// Left-hand operand captured by the operator< overload.
template <typename T>
struct InProxy {
    const T& val_;
};

// Tag type + constexpr instance so the syntax is <in> instead of <InTag{}>.
struct InTag {};
constexpr static auto in = InTag{};

// First half of the trick: v < in turns the value into an InProxy.
template <typename T>
auto operator<(const T& value, const InTag&) {
    return InProxy<T>{value};
}

// Second half: p > range performs the containment check.
template <typename T, typename Range>
auto operator>(const InProxy<T>& proxy, const Range& range) {
    return std::find(range.begin(), range.end(), proxy.val_) != range.end();
}

}  // namespace chp9

int main() {
    std::printf("== infix_operator ==\n");

    const std::vector<std::string> asia{"Korea", "Philippines", "Macau"};
    const std::vector<std::string> africa{"Senegal", "Botswana", "Guinea"};

    std::printf("\"Botswana\" in asia?  %d\n", "Botswana" <chp9::in> asia);
    std::printf("\"Botswana\" in africa? %d\n", "Botswana" <chp9::in> africa);

    const std::vector<int> digits{1, 2, 3, 4, 5};
    std::printf("3 in digits? %d\n", 3 <chp9::in> digits);
    std::printf("9 in digits? %d\n", 9 <chp9::in> digits);

    return 0;
}
