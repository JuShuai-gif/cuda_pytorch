#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "test_utils.hpp"

namespace {

[[gnu::noinline]] std::size_t count_loop(const std::vector<int>& v,
                                         int needle) {
    std::size_t n = 0;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (v[i] == needle) {
            ++n;
        }
    }
    return n;
}

[[gnu::noinline]] std::size_t count_algo(const std::vector<int>& v,
                                         int needle) {
    return static_cast<std::size_t>(std::count(v.begin(), v.end(), needle));
}

[[gnu::noinline]] std::size_t count_strings(const std::vector<std::string>& v,
                                            const std::string& needle) {
    return static_cast<std::size_t>(std::count(v.begin(), v.end(), needle));
}

}  // namespace

int main() {
    const std::vector<int> values = {1, 5, 2, 5, 3, 5, 4, 5};
    CHP_CHECK(count_loop(values, 5) == 4);
    CHP_CHECK(count_algo(values, 5) == 4);
    CHP_CHECK(count_loop(values, 5) == count_algo(values, 5));

    const std::vector<std::string> strings = {"a", "b", "a", "c", "a"};
    CHP_CHECK(count_strings(strings, "a") == 3);
    CHP_CHECK(count_strings(strings, "z") == 0);

    return chp::test_summary("zero_cost_asm");
}
