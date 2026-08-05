#include <any>
#include <cstdio>
#include <string>
#include <typeinfo>

#include "test_utils.hpp"

int main() {
    std::any a;
    CHP_CHECK(!a.has_value());

    a = std::string{"hello"};
    CHP_CHECK(a.has_value());
    CHP_CHECK(a.type() == typeid(std::string));
    CHP_CHECK(std::any_cast<std::string&>(a) == "hello");

    // Copying an any copies the stored value.
    std::any b = a;
    std::any_cast<std::string&>(b) = "changed";
    CHP_CHECK(std::any_cast<std::string>(a) == "hello");
    CHP_CHECK(std::any_cast<std::string>(b) == "changed");

    // Changing the held type.
    a = 42;
    CHP_CHECK(a.type() == typeid(int));
    CHP_CHECK(std::any_cast<int>(a) == 42);

    // Wrong cast throws.
    bool threw = false;
    try {
        (void)std::any_cast<float>(a);
    } catch (const std::bad_any_cast&) {
        threw = true;
    }
    CHP_CHECK(threw);

    // std::any stores a pointer/handle for heap-allocated state; its inline
    // buffer (SBO) on libstdc++ is smaller than std::string (16 < 32 bytes).
    CHP_CHECK(sizeof(std::any) < sizeof(std::string));

    return chp::test_summary("any_demo");
}
