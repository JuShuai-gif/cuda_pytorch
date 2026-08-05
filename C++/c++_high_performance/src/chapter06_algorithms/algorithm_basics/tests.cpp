#include <algorithm>
#include <cstdio>
#include <vector>

#include "test_utils.hpp"

namespace {

template <typename Iterator, typename T>
bool contains(Iterator begin, Iterator end, const T& v) {
    for (auto it = begin; it != end; ++it) {
        if (*it == v) {
            return true;
        }
    }
    return false;
}

}  // namespace

int main() {
    // Generic contains() on vector and list.
    {
        std::vector<int> v = {1, 2, 3};
        CHP_CHECK(contains(v.begin(), v.end(), 2));
        CHP_CHECK(!contains(v.begin(), v.end(), 9));
        std::vector<int> empty;
        CHP_CHECK(!contains(empty.begin(), empty.end(), 1));
    }

    // remove + erase idiom.
    {
        std::vector<int> v = {1, 1, 2, 2, 3, 3};
        const auto end = std::remove(v.begin(), v.end(), 2);
        v.erase(end, v.end());
        const std::vector<int> expected = {1, 1, 3, 3};
        CHP_CHECK(v == expected);
    }

    // unique + erase idiom.
    {
        std::vector<int> v = {1, 1, 2, 2, 3, 3};
        const auto end = std::unique(v.begin(), v.end());
        v.erase(end, v.end());
        const std::vector<int> expected = {1, 2, 3};
        CHP_CHECK(v == expected);
    }

    // transform with a preallocated destination.
    {
        std::vector<int> in = {1, 2, 3};
        std::vector<int> out;
        out.resize(in.size());
        std::transform(in.begin(), in.end(), out.begin(),
                       [](int v) { return v * v; });
        const std::vector<int> expected = {1, 4, 9};
        CHP_CHECK(out == expected);
    }

    return chp::test_summary("algorithm_basics");
}
