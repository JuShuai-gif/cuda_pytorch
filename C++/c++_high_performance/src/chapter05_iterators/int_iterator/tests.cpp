#include <algorithm>
#include <cstdio>
#include <iterator>
#include <vector>

#include "iterators.hpp"
#include "test_utils.hpp"

using chp::iter::BidirectionalIntIterator;
using chp::iter::IntIterator;

int main() {
    // Forward iteration produces the expected sequence.
    {
        std::vector<int> out;
        std::copy(IntIterator{0}, IntIterator{5}, std::back_inserter(out));
        const std::vector<int> expected = {0, 1, 2, 3, 4};
        CHP_CHECK(out == expected);
    }

    // Empty range: begin == end.
    {
        std::vector<int> out;
        std::copy(IntIterator{3}, IntIterator{3}, std::back_inserter(out));
        CHP_CHECK(out.empty());
    }

    // Bidirectional iteration in reverse (manual -- loop).
    {
        std::vector<int> out;
        for (BidirectionalIntIterator it{5}; it != BidirectionalIntIterator{0};
             --it) {
            out.push_back(*it);
        }
        const std::vector<int> expected = {5, 4, 3, 2, 1};
        CHP_CHECK(out == expected);
    }

    // Post-increment matches pre-increment value sequence.
    {
        std::vector<int> out;
        IntIterator it{1};
        out.push_back(*it++);
        out.push_back(*it);
        const std::vector<int> expected = {1, 2};
        CHP_CHECK(out == expected);
    }

    // std::distance works with the forward iterator.
    {
        const auto n = std::distance(IntIterator{10}, IntIterator{20});
        CHP_CHECK(n == 10);
    }

    return chp::test_summary("int_iterator");
}
