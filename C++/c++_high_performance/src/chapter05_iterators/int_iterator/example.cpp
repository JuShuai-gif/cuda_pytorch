// Custom iterators that generate integers on the fly (book PDF p.133-136).
//
// The IntIterator generates values without backing storage, yet works with
// range-based for-loops and STL algorithms because it exposes the same
// pointer-mimicking operators as a real container iterator.

#include "iterators.hpp"

#include <algorithm>
#include <cstdio>
#include <vector>

using chp::iter::BidirectionalIntIterator;
using chp::iter::IntIterator;

int main() {
    std::printf("== int_iterator ==\n");

    // Forward iteration (book PDF p.133): prints 12 13 14 15.
    std::printf("forward :");
    for (IntIterator it{12}; it != IntIterator{16}; ++it) {
        std::printf(" %d", *it);
    }
    std::printf("\n");

    // Bidirectional iteration (book PDF p.136): 12 .. 0.
    std::printf("reverse :");
    for (BidirectionalIntIterator it{12}; it != BidirectionalIntIterator{-1};
         --it) {
        std::printf(" %d", *it);
    }
    std::printf("\n");

    // Custom iterator works with std::copy (book PDF p.134).
    std::vector<int> numbers;
    std::copy(IntIterator{5}, IntIterator{12},
              std::back_inserter(numbers));
    std::printf("copied :");
    for (int n : numbers) {
        std::printf(" %d", n);
    }
    std::printf("\n");

    // Iterator category is advertised via iterator_traits.
    static_assert(
        std::is_same<std::iterator_traits<IntIterator>::iterator_category,
                     std::forward_iterator_tag>::value,
        "IntIterator is a forward iterator");
    static_assert(
        std::is_same<std::iterator_traits<BidirectionalIntIterator>::
                         iterator_category,
                     std::bidirectional_iterator_tag>::value,
        "BidirectionalIntIterator is bidirectional");

    return 0;
}
