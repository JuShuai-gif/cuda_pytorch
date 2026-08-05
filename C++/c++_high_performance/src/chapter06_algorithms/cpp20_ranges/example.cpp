// C++20 std::ranges: the modern realization of the book's "ranges library".
//
// The book (PDF p.163-173) covered the then-proposed ranges library
// (range-v3). Since C++20 the standard provides std::ranges: range-based
// algorithms, views, and the pipe operator. This example mirrors the book's
// Warrior example and the view/action/algorithm concepts.
//
// Build is guarded by the ENABLE_CPP20_EXAMPLES CMake option (requires a
// C++20-capable compiler).

#include <algorithm>
#include <cstdio>
#include <ranges>
#include <string>
#include <vector>

namespace {

enum class EAbility { Fencing, Archery };

struct Warrior {
    EAbility ability{};
    int level{};
    std::string name{};
};

}  // namespace

int main() {
    // --- Book's Warrior example (PDF p.163-165) via std::ranges ---
    const std::vector<Warrior> warriors = {
        {EAbility::Fencing, 12, "Zorro"},
        {EAbility::Archery, 10, "Legolas"},
        {EAbility::Archery, 7, "Link"},
    };

    auto is_archer = [](const Warrior& w) { return w.ability == EAbility::Archery; };
    auto level_of = [](const Warrior& w) { return w.level; };

    // filter | transform | max: all lazy views, no intermediate vector.
    auto archer_levels = warriors | std::views::filter(is_archer) |
                         std::views::transform(level_of);
    const auto max_it = std::ranges::max_element(archer_levels);
    std::printf("max archer level (ranges): %d\n",
                max_it != archer_levels.end() ? *max_it : -1);

    // --- Views: lazy transform/filter (book PDF p.166-168) ---
    std::vector<int> numbers = {1, 2, 3, 4, 5, 6, 7};
    auto odd_squares = numbers | std::views::transform([](int v) { return v * v; }) |
                       std::views::filter([](int v) { return (v % 2) == 1; });
    std::printf("odd squares:");
    for (int s : odd_squares) {
        std::printf(" %d", s);
    }
    std::printf("\n");

    // --- Range algorithms (book PDF p.173) ---
    std::printf("ranges::count(7): %td\n",
                std::ranges::count(numbers, 7));

    // --- join: flatten a list of lists (book PDF p.167) ---
    std::vector<std::vector<int>> list_of_lists = {{1, 2}, {3, 4, 5}, {5}, {4, 3, 2, 1}};
    auto flattened = list_of_lists | std::views::join;
    std::printf("joined:");
    for (int v : flattened) {
        std::printf(" %d", v);
    }
    std::printf("\n");

    return 0;
}
