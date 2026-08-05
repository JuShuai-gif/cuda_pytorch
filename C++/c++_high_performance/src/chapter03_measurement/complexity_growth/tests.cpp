#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <vector>

#include "baseline.hpp"
#include "test_utils.hpp"

int main() {
    // Linear search basics.
    {
        const std::vector<int> v = {1, 2, 3, 4, 5};
        CHP_CHECK(chp::cg::linear_search_int(v, 3));
        CHP_CHECK(!chp::cg::linear_search_int(v, 9));
        CHP_CHECK(!chp::cg::linear_search_int({}, 1));
    }

    // Point linear search.
    {
        const std::vector<chp::cg::Point> v = {{0, 0}, {1, 1}, {2, 2}};
        CHP_CHECK(chp::cg::linear_search_point(v, chp::cg::Point{1, 1}));
        CHP_CHECK(!chp::cg::linear_search_point(v, chp::cg::Point{1, 2}));
    }

    // Binary search on sorted data.
    {
        std::vector<int> v(1000);
        for (std::size_t i = 0; i < v.size(); ++i) {
            v[i] = static_cast<int>(i * 2);  // even numbers
        }
        CHP_CHECK(chp::cg::binary_search_int(v, 0));
        CHP_CHECK(chp::cg::binary_search_int(v, 998));
        CHP_CHECK(!chp::cg::binary_search_int(v, 999));
        CHP_CHECK(!chp::cg::binary_search_int(v, -1));
        CHP_CHECK(!chp::cg::binary_search_int({}, 5));
        // Binary search requires sorted input: verify against a sorted copy.
        const std::vector<int> unsorted = {5, 1, 4, 2, 3};
        std::vector<int> sorted = unsorted;
        std::sort(sorted.begin(), sorted.end());
        for (int key = 0; key < 6; ++key) {
            const bool expected = std::binary_search(sorted.begin(),
                                                     sorted.end(), key);
            CHP_CHECK(chp::cg::binary_search_int(sorted, key) == expected);
        }
    }

    // All three agree on a large sorted input.
    {
        std::vector<int> v(100'000);
        for (std::size_t i = 0; i < v.size(); ++i) {
            v[i] = static_cast<int>(i);
        }
        for (int key = 0; key < 100; ++key) {
            const bool lin = chp::cg::linear_search_int(v, key);
            const bool bin = chp::cg::binary_search_int(v, key);
            CHP_CHECK(lin == bin);
        }
        CHP_CHECK(!chp::cg::binary_search_int(v, -1));
    }

    return chp::test_summary("complexity_growth");
}
