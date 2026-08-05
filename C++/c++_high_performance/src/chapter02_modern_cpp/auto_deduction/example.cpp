#include <cstdio>
#include <string>
#include <type_traits>
#include <vector>

#include "test_utils.hpp"

namespace {

struct Foo {
    int m_ = 42;

    auto val() const { return m_; }
    auto& cref() const { return m_; }
    auto& mref() { return m_; }
};

// A temporary string: proves that const auto& and auto&& extend lifetime.
std::string make_string() { return std::string(64, 'x'); }

}  // namespace

int main() {
    // --- auto in function signatures (return type deduction) ---
    auto foo = Foo{};
    static_assert(std::is_same<decltype(foo.val()), int>::value,
                  "val() returns by value");
    static_assert(std::is_same<decltype(foo.cref()), const int&>::value,
                  "cref() returns const int&");
    static_assert(std::is_same<decltype(foo.mref()), int&>::value,
                  "mref() returns int&");

    // --- auto variables: value, const reference, mutable reference ---
    auto v = foo.val();          // copy: mutating v does not affect foo
    auto& mr = foo.mref();       // mutable reference
    const auto& cr = foo.cref(); // const reference

    mr = 100;                    // changes foo.m_
    static_assert(std::is_same<decltype(v), int>::value, "v is int");
    static_assert(std::is_same<decltype(mr), int&>::value, "mr is int&");
    static_assert(std::is_same<decltype(cr), const int&>::value, "cr is const int&");

    CHP_CHECK(v == 42);      // v was copied before the mutation
    CHP_CHECK(mr == 100);    // mr aliases foo.m_
    CHP_CHECK(cr == 100);    // cr sees the same object

    // --- const auto& / auto&& bind to temporaries and extend lifetime ---
    {
        const auto& s = make_string();   // temporary's lifetime extended
        CHP_CHECK(s.size() == 64);
    }
    {
        auto&& s = make_string();        // forwarding reference binds to temporary
        CHP_CHECK(s.size() == 64);
    }

    // --- mutable reference cannot bind to a temporary (compile error), so
    //     we only demonstrate the const and forwarding cases above. ---

    std::printf("auto_deduction: checks finished\n");
    return chp::test_summary("auto_deduction");
}
