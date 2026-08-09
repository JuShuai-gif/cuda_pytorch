// Correctness checks for infix_operator.

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "test_utils.hpp"

namespace {

template <typename T>
struct InProxy {
    const T& val_;
};

struct InTag {};
constexpr static auto in = InTag{};

template <typename T>
auto operator<(const T& value, const InTag&) {
    return InProxy<T>{value};
}

template <typename T, typename Range>
auto operator>(const InProxy<T>& proxy, const Range& range) {
    return std::find(range.begin(), range.end(), proxy.val_) != range.end();
}

}  // namespace

int main() {
    const std::vector<std::string> asia{"Korea", "Philippines", "Macau"};
    const std::vector<std::string> africa{"Senegal", "Botswana", "Guinea"};

    CHP_CHECK("Botswana" <in> africa);
    CHP_CHECK(!("Botswana" <in> asia));
    CHP_CHECK("Korea" <in> asia);
    CHP_CHECK(!("Korea" <in> africa));

    const std::vector<int> digits{1, 2, 3, 4, 5};
    CHP_CHECK(3 <in> digits);
    CHP_CHECK(5 <in> digits);
    CHP_CHECK(!(6 <in> digits));

    return chp::test_summary("infix_operator");
}
