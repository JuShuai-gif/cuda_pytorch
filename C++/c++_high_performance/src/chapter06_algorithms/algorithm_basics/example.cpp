// STL algorithm fundamentals.
//
// The book (PDF p.145-148) covers the core concepts:
//  - algorithms operate on iterators, not containers (contains(), Grid rows);
//  - algorithms do not change the container size: remove()/unique() just
//    shuffle elements and return a new end iterator;
//  - the caller must combine them with erase().

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <vector>

namespace {

// A generic algorithm usable with any container exposing iterators
// (book PDF p.145).
template <typename Iterator, typename T>
bool contains(Iterator begin, Iterator end, const T& v) {
    for (auto it = begin; it != end; ++it) {
        if (*it == v) {
            return true;
        }
    }
    return false;
}

// A 2D grid over a 1D vector; rows are exposed as iterator pairs
// (book PDF p.146).
struct Grid {
    Grid(std::size_t w, std::size_t h) : w_(w), h_(h) { data_.resize(w * h); }

    std::pair<std::vector<int>::iterator, std::vector<int>::iterator> get_row(
        std::size_t y) {
        auto l = data_.begin() + static_cast<std::ptrdiff_t>(w_ * y);
        auto r = l + static_cast<std::ptrdiff_t>(w_);
        return {l, r};
    }

    std::vector<int> data_;
    std::size_t w_;
    std::size_t h_;
};

}  // namespace

int main() {
    std::printf("== algorithm_basics ==\n");

    // contains() with vector and list iterators.
    const std::vector<int> v = {1, 2, 3, 4, 5};
    std::printf("contains(v, 3): %d\n", contains(v.begin(), v.end(), 3));
    std::printf("contains(v, 9): %d\n", contains(v.begin(), v.end(), 9));

    // Grid rows work with STL algorithms (book PDF p.146).
    Grid grid(10, 10);
    auto row = grid.get_row(3);
    std::generate(row.first, row.second, []() { return 42; });
    std::printf("row 3 has %zu fives\n",
                static_cast<std::size_t>(
                    std::count(row.first, row.second, 5)));
    std::printf("row 3 all 42s: %d\n",
                std::all_of(row.first, row.second,
                            [](int x) { return x == 42; }));

    // remove() does NOT shrink the container; erase() completes the job
    // (book PDF p.147).
    std::vector<int> a = {1, 1, 2, 2, 3, 3};
    const auto new_end = std::remove(a.begin(), a.end(), 2);
    a.erase(new_end, a.end());
    std::printf("after remove+erase 2: ");
    for (int x : a) {
        std::printf("%d ", x);
    }
    std::printf("\n");

    // unique() + erase (book PDF p.147-148).
    std::vector<int> b = {1, 1, 2, 2, 3, 3};
    const auto new_end_b = std::unique(b.begin(), b.end());
    b.erase(new_end_b, b.end());
    std::printf("after unique+erase: ");
    for (int x : b) {
        std::printf("%d ", x);
    }
    std::printf("\n");

    return 0;
}
