// Comparing four ways to represent a value that may be absent:
//   1. a sentinel value (e.g. -1)
//   2. a pointer (nullptr means "absent")
//   3. std::optional<T>
//   4. std::variant<std::monostate, T>
//
// The book discusses std::optional (PDF p.80-82) and std::variant is covered
// in Chapter 8 (PDF p.238); this example compares the representation costs.

#include <cstddef>
#include <cstdio>
#include <optional>
#include <variant>

namespace {

// Sentinel value approach: 0 means "no measurement".
int lookup_sentinel(int id, int* values) { return values[id]; }

// Pointer approach.
const int* lookup_pointer(int id, const int* values) { return &values[id]; }

// Optional approach.
std::optional<int> lookup_optional(int id, const int* values) {
    return std::optional<int>{values[id]};
}

// Variant approach.
std::variant<std::monostate, int> lookup_variant(int id, const int* values) {
    return std::variant<std::monostate, int>{std::in_place_index<1>,
                                             values[id]};
}

}  // namespace

int main() {
    std::printf("== optional_variant_pointer ==\n");

    int values[] = {1, 2, 3};

    std::printf("sizeof(int)                        = %zu\n", sizeof(int));
    std::printf("sizeof(int*)                       = %zu\n", sizeof(int*));
    std::printf("sizeof(optional<int>)              = %zu\n",
                sizeof(std::optional<int>));
    std::printf("sizeof(variant<monostate, int>)    = %zu\n",
                sizeof(std::variant<std::monostate, int>));

    const int id = 1;
    const int sentinel = lookup_sentinel(id, values);
    const int* ptr = lookup_pointer(id, values);
    auto opt = lookup_optional(id, values);
    auto var = lookup_variant(id, values);

    std::printf("sentinel: %d\n", sentinel);
    std::printf("pointer : %d (null when absent)\n", ptr ? *ptr : -1);
    std::printf("optional: %d (has_value=%d)\n",
                opt.has_value() ? *opt : -1, opt.has_value());
    std::printf("variant : %d (holds int=%d)\n",
                std::holds_alternative<int>(var) ? std::get<int>(var) : -1,
                std::holds_alternative<int>(var));
    return 0;
}
