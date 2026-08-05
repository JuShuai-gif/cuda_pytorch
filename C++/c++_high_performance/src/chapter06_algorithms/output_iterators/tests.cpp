#include <algorithm>
#include <cstdio>
#include <iterator>
#include <set>
#include <vector>

#include "test_utils.hpp"

int main() {
    const std::vector<int> vals = {1, 2, 3, 4};
    const std::vector<int> expected = {1, 4, 9, 16};

    // Preallocated destination.
    {
        std::vector<int> out(4, 0);
        std::transform(vals.begin(), vals.end(), out.begin(),
                       [](int v) { return v * v; });
        CHP_CHECK(out == expected);
    }

    // back_inserter.
    {
        std::vector<int> out;
        std::transform(vals.begin(), vals.end(), std::back_inserter(out),
                       [](int v) { return v * v; });
        CHP_CHECK(out == expected);
    }

    // inserter into a set (squares are unique here).
    {
        std::set<int> out;
        std::transform(vals.begin(), vals.end(),
                       std::inserter(out, out.end()),
                       [](int v) { return v * v; });
        CHP_CHECK(out == std::set<int>({1, 4, 9, 16}));
    }

    // reserve + back_inserter.
    {
        std::vector<int> out;
        out.reserve(vals.size());
        std::transform(vals.begin(), vals.end(), std::back_inserter(out),
                       [](int v) { return v * v; });
        CHP_CHECK(out == expected);
        CHP_CHECK(out.capacity() == 4);  // reserve avoided reallocation
    }

    return chp::test_summary("output_iterators");
}
