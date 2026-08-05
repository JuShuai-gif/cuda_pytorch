#include <algorithm>
#include <cstdio>
#include <vector>

#include "find.hpp"
#include "test_utils.hpp"

int main() {
    std::vector<int> v = {1, 5, 2, 5, 3, 5, 4};

    // find_slow and find_fast must agree with std::find on present values.
    for (int needle : {1, 5, 4}) {
        const auto s = chp::fu::find_slow(v.begin(), v.end(), needle);
        const auto f = chp::fu::find_fast(v.begin(), v.end(), needle);
        const auto st = std::find(v.begin(), v.end(), needle);
        CHP_CHECK(s == st);
        CHP_CHECK(f == st);
    }

    // Missing value returns end for both.
    {
        const auto s = chp::fu::find_slow(v.begin(), v.end(), 99);
        const auto f = chp::fu::find_fast(v.begin(), v.end(), 99);
        CHP_CHECK(s == v.end());
        CHP_CHECK(f == v.end());
    }

    // Short ranges (length 0..3) must be handled by the tail switch.
    for (std::size_t n = 0; n <= 3; ++n) {
        std::vector<int> small(n);
        for (std::size_t i = 0; i < n; ++i) {
            small[i] = static_cast<int>(i);
        }
        for (int needle = -1; needle <= 3; ++needle) {
            const auto s = chp::fu::find_slow(small.begin(), small.end(),
                                              needle);
            const auto f = chp::fu::find_fast(small.begin(), small.end(),
                                              needle);
            const auto st = std::find(small.begin(), small.end(), needle);
            CHP_CHECK(s == st);
            CHP_CHECK(f == st);
        }
    }

    return chp::test_summary("find_unroll");
}
