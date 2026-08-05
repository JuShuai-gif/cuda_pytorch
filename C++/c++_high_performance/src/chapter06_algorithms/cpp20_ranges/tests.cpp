#include <algorithm>
#include <cstdio>
#include <ranges>
#include <string>
#include <vector>

#include "test_utils.hpp"

namespace {

enum class EAbility { Fencing, Archery };

struct Warrior {
    EAbility ability{};
    int level{};
    std::string name{};
};

}  // namespace

int main() {
    const std::vector<Warrior> warriors = {
        {EAbility::Fencing, 12, "Zorro"},
        {EAbility::Archery, 10, "Legolas"},
        {EAbility::Archery, 7, "Link"},
    };

    // Ranges pipeline matches a hand-written loop result.
    auto archer_levels =
        warriors |
        std::views::filter([](const Warrior& w) {
            return w.ability == EAbility::Archery;
        }) |
        std::views::transform([](const Warrior& w) { return w.level; });

    int max_hand = 0;
    for (const auto& w : warriors) {
        if (w.ability == EAbility::Archery) {
            max_hand = std::max(max_hand, w.level);
        }
    }
    const auto max_it = std::ranges::max_element(archer_levels);
    CHP_CHECK(max_it != archer_levels.end());
    CHP_CHECK(*max_it == max_hand);
    CHP_CHECK(max_hand == 10);

    // Views are lazy: transform is applied on access, no vector built.
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    auto squares = numbers | std::views::transform([](int v) { return v * v; });
    int idx = 0;
    for (int s : squares) {
        CHP_CHECK(s == (idx + 1) * (idx + 1));
        ++idx;
    }
    CHP_CHECK(idx == 5);

    // join view flattens.
    std::vector<std::vector<int>> lol = {{1, 2}, {3, 4}};
    auto flat = lol | std::views::join;
    const std::vector<int> expected = {1, 2, 3, 4};
    std::vector<int> collected(flat.begin(), flat.end());
    CHP_CHECK(collected == expected);

    return chp::test_summary("cpp20_ranges");
}
