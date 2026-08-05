#include <algorithm>
#include <cstdio>
#include <vector>

#include "test_utils.hpp"

int main() {
    // nth_element partitions so that elements before the nth are <= it and
    // elements after are >= it.
    {
        std::vector<int> v = {7, 2, 9, 1, 5, 3, 8, 6, 4, 0};
        const auto nth = v.begin() + 5;  // element 5 (0-based) in sorted order
        std::nth_element(v.begin(), nth, v.end());
        const int pivot = v[5];
        for (std::size_t i = 0; i < 5; ++i) {
            CHP_CHECK(v[i] <= pivot);
        }
        for (std::size_t i = 6; i < v.size(); ++i) {
            CHP_CHECK(v[i] >= pivot);
        }
        // The full sorted array's 5th element is 5.
        std::vector<int> sorted = {7, 2, 9, 1, 5, 3, 8, 6, 4, 0};
        std::sort(sorted.begin(), sorted.end());
        CHP_CHECK(pivot == sorted[5]);
    }

    // partial_sort: the first m elements are the smallest m, sorted.
    {
        std::vector<int> v = {5, 3, 9, 1, 7, 2, 8, 4, 6, 0};
        std::partial_sort(v.begin(), v.begin() + 3, v.end());
        CHP_CHECK(v[0] == 0);
        CHP_CHECK(v[1] == 1);
        CHP_CHECK(v[2] == 2);
    }

    // Median via nth_element (book PDF p.161).
    {
        std::vector<int> v = {3, 1, 4, 1, 5, 9, 2};
        std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
        std::vector<int> sorted = v;
        std::sort(sorted.begin(), sorted.end());
        CHP_CHECK(v[v.size() / 2] == sorted[sorted.size() / 2]);
    }

    return chp::test_summary("partial_sorting");
}
