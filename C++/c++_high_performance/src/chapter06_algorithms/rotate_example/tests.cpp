#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <list>
#include <vector>

#include "test_utils.hpp"

namespace {

template <typename Container>
void move_n_to_back_indexed(Container& c, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        auto value = std::move(*std::next(
            c.begin(), static_cast<std::ptrdiff_t>(i)));
        c.emplace_back(std::move(value));
    }
    c.erase(c.begin(), std::next(c.begin(), static_cast<std::ptrdiff_t>(n)));
}

template <typename Container>
void move_n_to_back_rotate(Container& c, std::size_t n) {
    auto new_begin =
        std::next(c.begin(), static_cast<std::ptrdiff_t>(n));
    std::rotate(c.begin(), new_begin, c.end());
}

}  // namespace

int main() {
    const std::vector<int> expected = {3, 4, 5, 1, 2};

    {
        std::vector<int> v = {1, 2, 3, 4, 5};
        move_n_to_back_indexed(v, 2);
        CHP_CHECK(v == expected);
    }
    {
        std::vector<int> v = {1, 2, 3, 4, 5};
        move_n_to_back_rotate(v, 2);
        CHP_CHECK(v == expected);
    }
    {
        std::list<int> l = {1, 2, 3, 4, 5};
        move_n_to_back_rotate(l, 2);
        CHP_CHECK(l == std::list<int>({3, 4, 5, 1, 2}));
    }
    // n == 0 and n == size are no-ops.
    {
        std::vector<int> v = {1, 2, 3};
        move_n_to_back_rotate(v, 0);
        CHP_CHECK(v == std::vector<int>({1, 2, 3}));
        move_n_to_back_rotate(v, 3);
        CHP_CHECK(v == std::vector<int>({1, 2, 3}));
    }

    return chp::test_summary("rotate_example");
}
