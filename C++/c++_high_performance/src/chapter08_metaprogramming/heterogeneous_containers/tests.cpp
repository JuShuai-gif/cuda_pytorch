#include <algorithm>
#include <any>
#include <cstdio>
#include <string>
#include <variant>
#include <vector>

#include "test_utils.hpp"

using VariantType = std::variant<int, std::string, bool>;

int main() {
    // --- variant basics (PDF p.238) ---
    VariantType v;
    CHP_CHECK(std::holds_alternative<int>(v));  // default: first alternative
    v = 7;
    CHP_CHECK(std::holds_alternative<int>(v) && std::get<int>(v) == 7);
    v = std::string{"Bjarne"};
    CHP_CHECK(std::holds_alternative<std::string>(v));
    v = false;
    CHP_CHECK(std::holds_alternative<bool>(v));

    // std::get by type throws on wrong alternative.
    v = 42;
    bool threw = false;
    try {
        (void)std::get<std::string>(v);
    } catch (const std::bad_variant_access&) {
        threw = true;
    }
    CHP_CHECK(threw);

    // --- vector<variant> heterogeneous container (PDF p.240-242) ---
    std::vector<VariantType> c{42, std::string{"needle"}, true};
    CHP_CHECK(c.size() == 3);

    // Count by type.
    const auto num_bools = static_cast<int>(std::count_if(
        c.begin(), c.end(),
        [](const auto& x) { return std::holds_alternative<bool>(x); }));
    CHP_CHECK(num_bools == 1);

    // Find by type and value.
    const bool has_needle = std::any_of(
        c.begin(), c.end(), [](const auto& x) {
            return std::holds_alternative<std::string>(x) &&
                   std::get<std::string>(x) == "needle";
        });
    CHP_CHECK(has_needle);

    // vector<any> behaves equivalently for the three known types.
    std::vector<std::any> ac{42, std::string{"needle"}, true};
    CHP_CHECK(ac.size() == 3);
    CHP_CHECK(std::any_cast<int>(ac[0]) == 42);
    CHP_CHECK(std::any_cast<std::string>(ac[1]) == "needle");
    CHP_CHECK(std::any_cast<bool>(ac[2]) == true);

    return chp::test_summary("heterogeneous_containers");
}
