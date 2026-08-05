#include <cstddef>
#include <cstdio>
#include <optional>
#include <variant>

#include "test_utils.hpp"

int main() {
    // Sentinel representation: -1 means absent.
    {
        int v = -1;  // absent
        CHP_CHECK(v == -1);
        v = 5;  // present
        CHP_CHECK(v == 5);
    }

    // Pointer representation.
    {
        int data = 7;
        const int* p = &data;  // present
        CHP_CHECK(p != nullptr && *p == 7);
        p = nullptr;  // absent
        CHP_CHECK(p == nullptr);
    }

    // std::optional representation.
    {
        std::optional<int> o;
        CHP_CHECK(!o.has_value());
        o = 9;
        CHP_CHECK(o.has_value() && *o == 9);
    }

    // std::variant representation.
    {
        std::variant<std::monostate, int> v;
        CHP_CHECK(std::holds_alternative<std::monostate>(v));  // absent
        v = 11;
        CHP_CHECK(std::holds_alternative<int>(v));
        CHP_CHECK(std::get<int>(v) == 11);
    }

    // Size comparison (documented, platform dependent).
    std::printf("sizes: int=%zu int*= %zu optional<int>=%zu "
                "variant<monostate,int>=%zu\n",
                sizeof(int), sizeof(int*), sizeof(std::optional<int>),
                sizeof(std::variant<std::monostate, int>));

    return chp::test_summary("optional_variant_pointer");
}
