// Correctness checks for pipe_operator.

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "test_utils.hpp"

namespace {

template <typename T>
struct ContainsProxy {
    const T& value_;
};

template <typename Range, typename T>
auto operator|(const Range& range, const ContainsProxy<T>& proxy) {
    return std::find(range.begin(), range.end(), proxy.value_) != range.end();
}

template <typename T>
auto contains(const T& value) {
    return ContainsProxy<T>{value};
}

}  // namespace

int main() {
    const std::vector<int> numbers{1, 3, 5, 7, 9};
    CHP_CHECK(numbers | contains(7));
    CHP_CHECK(numbers | contains(1));
    CHP_CHECK(!(numbers | contains(8)));
    CHP_CHECK(!(numbers | contains(0)));

    const std::vector<std::string> penguins{"Ping", "Roy", "Silo"};
    CHP_CHECK(penguins | contains(std::string{"Silo"}));
    CHP_CHECK(!(penguins | contains(std::string{"Kowalski"})));

    // Works on non-vector ranges too.
    const std::string text = "high performance";
    CHP_CHECK(text | contains('p'));
    CHP_CHECK(!(text | contains('z')));

    return chp::test_summary("pipe_operator");
}
