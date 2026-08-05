// Moving the first n elements to the back: three approaches.
//
// The book (PDF p.156-158) shows a naive for-loop that invalidates iterators
// when the vector reallocates, a safer index-based loop that is O(n^2) on
// std::list (std::next is O(n) per call), and finally std::rotate which is
// O(n) on every container and does not allocate.

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <list>
#include <vector>

namespace {

// Approach 1 (book p.156): copies while iterating, then erases.
// BUG: if c reallocates during emplace_back, `it` becomes invalid.
// We demonstrate this only on a std::list (no reallocation) to stay safe.
template <typename Container>
void move_n_to_back_unsafe(Container& c, std::size_t n) {
    for (auto it = c.begin(); it != std::next(c.begin(),
                                              static_cast<std::ptrdiff_t>(n));
         ++it) {
        c.emplace_back(std::move(*it));
    }
    c.erase(c.begin(), std::next(c.begin(), static_cast<std::ptrdiff_t>(n)));
}

// Approach 2 (book p.157): index-based, safe but O(n^2) on std::list because
// std::next(c.begin(), i) is O(i) there.
template <typename Container>
void move_n_to_back_indexed(Container& c, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        auto value = std::move(*std::next(
            c.begin(), static_cast<std::ptrdiff_t>(i)));
        c.emplace_back(std::move(value));
    }
    c.erase(c.begin(), std::next(c.begin(), static_cast<std::ptrdiff_t>(n)));
}

// Approach 3 (book p.157): std::rotate, O(n), no allocation, works with
// fixed-size containers too.
template <typename Container>
void move_n_to_back_rotate(Container& c, std::size_t n) {
    auto new_begin =
        std::next(c.begin(), static_cast<std::ptrdiff_t>(n));
    std::rotate(c.begin(), new_begin, c.end());
}

}  // namespace

int main() {
    std::printf("== rotate_example ==\n");

    // On a list, all three approaches produce the same result.
    std::list<int> l1 = {1, 2, 3, 4, 5};
    std::list<int> l2 = {1, 2, 3, 4, 5};
    std::list<int> l3 = {1, 2, 3, 4, 5};
    move_n_to_back_unsafe(l1, 2);
    move_n_to_back_indexed(l2, 2);
    move_n_to_back_rotate(l3, 2);

    std::printf("unsafe : ");
    for (int x : l1) std::printf("%d ", x);
    std::printf("\nindexed: ");
    for (int x : l2) std::printf("%d ", x);
    std::printf("\nrotate : ");
    for (int x : l3) std::printf("%d ", x);
    std::printf("\n");

    // std::rotate works on fixed-size arrays too (book p.157).
    int arr[] = {1, 2, 3, 4, 5};
    std::rotate(arr, arr + 2, arr + 5);
    std::printf("array  : ");
    for (int x : arr) std::printf("%d ", x);
    std::printf("\n");

    return 0;
}
