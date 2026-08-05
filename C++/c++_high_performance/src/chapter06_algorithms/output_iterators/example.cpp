// Output iterators: where the algorithm's output goes.
//
// The book (PDF p.148-149): algorithms with output (std::copy, std::transform)
// need already-allocated space OR an insert iterator. Writing to an empty
// container's begin() is undefined behavior (crash). Options:
//   1. preallocate with resize();
//   2. std::back_inserter (vector), std::inserter (set);
//   3. reserve() when the size is known, to avoid reallocations.

#include <algorithm>
#include <cstdio>
#include <iterator>
#include <set>
#include <vector>

namespace {

int square(int v) { return v * v; }

}  // namespace

int main() {
    std::printf("== output_iterators ==\n");

    const std::vector<int> vals = {1, 2, 3, 4};

    // 1. Preallocated destination.
    std::vector<int> prealloc(4, 0);
    std::transform(vals.begin(), vals.end(), prealloc.begin(), square);
    std::printf("preallocated : ");
    for (int x : prealloc) {
        std::printf("%d ", x);
    }
    std::printf("\n");

    // 2a. back_inserter into an empty vector.
    std::vector<int> with_back;
    std::transform(vals.begin(), vals.end(), std::back_inserter(with_back),
                   square);
    std::printf("back_inserter: ");
    for (int x : with_back) {
        std::printf("%d ", x);
    }
    std::printf("\n");

    // 2b. inserter into a std::set.
    std::set<int> with_set;
    std::transform(vals.begin(), vals.end(),
                   std::inserter(with_set, with_set.end()), square);
    std::printf("inserter(set): ");
    for (int x : with_set) {
        std::printf("%d ", x);
    }
    std::printf("\n");

    // 3. reserve() before back_inserter avoids reallocations.
    std::vector<int> with_reserve;
    with_reserve.reserve(vals.size());
    std::transform(vals.begin(), vals.end(),
                   std::back_inserter(with_reserve), square);
    std::printf("reserve+back  : ");
    for (int x : with_reserve) {
        std::printf("%d ", x);
    }
    std::printf("\n");

    return 0;
}
