#include <cstddef>
#include <cstdio>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "test_utils.hpp"

namespace {

// A tracked resource so we can count copies vs moves and verify destruction.
class Counted {
public:
    explicit Counted(int v = 0) : value_(v) { ++constructed; }

    Counted(const Counted&) { ++constructed; ++copies; }
    Counted& operator=(const Counted&) { ++copies; return *this; }

    Counted(Counted&&) noexcept { ++constructed; ++moves; }
    Counted& operator=(Counted&&) noexcept { ++moves; return *this; }

    ~Counted() { ++destroyed; }

    int value() const { return value_; }

    static void reset() {
        constructed = 0;
        destroyed = 0;
        copies = 0;
        moves = 0;
    }
    static int constructed;
    static int destroyed;
    static int copies;
    static int moves;

private:
    int value_;
};

int Counted::constructed = 0;
int Counted::destroyed = 0;
int Counted::copies = 0;
int Counted::moves = 0;

}  // namespace

int main() {
    // Rule of Three: a class managing a resource needs all three special
    // members. Here we just verify the Rule of Five via a counted type.
    Counted::reset();
    {
        std::vector<Counted> v;
        v.reserve(4);
        v.push_back(Counted{1});  // move into the vector (empty -> move)
        v.emplace_back(2);
        v.emplace_back(3);
        v.emplace_back(4);
        v.emplace_back(5);        // triggers reallocation -> moves
    }
    CHP_CHECK(Counted::constructed == Counted::destroyed);
    CHP_CHECK(Counted::moves > 0);

    // Rule of Zero: a class whose members are all RAII works with the
    // compiler-generated special members.
    struct RuleOfZero {
        std::vector<int> values;
        std::string name;
    };
    static_assert(std::is_copy_constructible<RuleOfZero>::value, "copyable");
    static_assert(std::is_move_constructible<RuleOfZero>::value, "movable");
    RuleOfZero a{{1, 2, 3}, "hi"};
    RuleOfZero b = std::move(a);
    CHP_CHECK(b.values.size() == 3);
    CHP_CHECK(a.values.empty());  // moved-from vector is empty (libstdc++)

    return chp::test_summary("rule_of_three_five_zero");
}
