#include <cstdio>
#include <type_traits>
#include <vector>

#include "baseline.hpp"
#include "test_utils.hpp"

namespace {

using chp::nomv::MoveNoexcept;
using chp::nomv::MoveThrowing;

}  // namespace

int main() {
    // Trait assertions encode the whole point of the experiment.
    static_assert(std::is_nothrow_move_constructible<MoveNoexcept>::value,
                  "MoveNoexcept move must be noexcept");
    static_assert(!std::is_nothrow_move_constructible<MoveThrowing>::value,
                  "MoveThrowing move must NOT be noexcept");

    // During vector growth, a noexcept move leads to moves; a throwing move
    // forces the vector to copy existing elements instead.
    MoveNoexcept::copies = 0;
    MoveNoexcept::moves = 0;
    {
        std::vector<MoveNoexcept> v;
        for (int i = 0; i < 100; ++i) {
            v.emplace_back(i);
        }
    }
    CHP_CHECK(MoveNoexcept::moves > 0);

    MoveThrowing::copies = 0;
    MoveThrowing::moves = 0;
    {
        std::vector<MoveThrowing> v;
        for (int i = 0; i < 100; ++i) {
            v.emplace_back(i);
        }
    }
    // Reallocation copies the existing elements (no safe move available).
    CHP_CHECK(MoveThrowing::copies > 0);

    return chp::test_summary("noexcept_move");
}
