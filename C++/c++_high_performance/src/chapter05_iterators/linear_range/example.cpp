// Iterating floating point ranges without precision errors (book PDF p.136-142).
//
// Naive `for (float t = 0; t <= 1; t += 0.1f)` fails because 0.1 is not
// exactly representable. The book solves this by iterating over an INDEX and
// computing value = start + step*idx, wrapping it in an iterator + range.

#include "linear_range.hpp"

#include <algorithm>
#include <cstdio>
#include <iterator>
#include <set>
#include <vector>

using chp::lr::LinearRange;
using chp::lr::LinearRangeIterator;
using chp::lr::make_linear_range;

int main() {
    std::printf("== linear_range ==\n");

    // 11 values: 0.0 .. 1.0 with step 0.1 (book PDF p.137).
    std::printf("0..1 in 11 values: ");
    for (auto t : make_linear_range(0.0F, 1.0F, 11)) {
        std::printf("%.1f ", t);
    }
    std::printf("\n");

    // 4 values: 0.0, 0.33, 0.66, 1.0 (book PDF p.141).
    std::printf("0..1 in 4 values:  ");
    for (auto t : make_linear_range(0.0, 1.0, 4)) {
        std::printf("%.2f ", t);
    }
    std::printf("\n");

    // Reverse order (book PDF p.142).
    std::printf("1..0 in 4 values:  ");
    for (auto t : make_linear_range(1.0, 0.0, 4)) {
        std::printf("%.2f ", t);
    }
    std::printf("\n");

    // Works with STL algorithms (book PDF p.140).
    std::vector<float> vec;
    std::copy(make_linear_range(0.0F, 1.0F, 4).begin(),
              make_linear_range(0.0F, 1.0F, 4).end(),
              std::back_inserter(vec));
    std::printf("copied: ");
    for (float v : vec) {
        std::printf("%.2f ", v);
    }
    std::printf("\n");

    // Standalone iterators into a std::set (book PDF p.140).
    std::set<float> s;
    const float start = 0.0F;
    const float stop = 1.0F;
    const std::size_t num = 6;
    const float step = chp::lr::get_step_size(start, stop, num);
    LinearRangeIterator<float> first{start, step, 0};
    LinearRangeIterator<float> last{start, step, num};
    std::copy(first, last, std::inserter(s, s.end()));
    std::printf("set from iterators: ");
    for (float v : s) {
        std::printf("%.2f ", v);
    }
    std::printf("\n");

    // C++17 class template argument deduction makes the factory optional
    // (book PDF p.142).
    LinearRange r{0.0F, 1.0F, 4};
    std::printf("CTAD LinearRange:   ");
    for (auto t : r) {
        std::printf("%.2f ", t);
    }
    std::printf("\n");

    return 0;
}
