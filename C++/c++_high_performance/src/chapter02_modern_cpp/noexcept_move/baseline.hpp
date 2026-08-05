#pragma once

#include <cstddef>

namespace chp {
namespace nomv {

// A type whose move constructor is noexcept: vector reallocation will use
// moves.
struct MoveNoexcept {
    explicit MoveNoexcept(int v) : value(v) {}
    MoveNoexcept(const MoveNoexcept& other) : value(other.value) { ++copies; }
    MoveNoexcept& operator=(const MoveNoexcept&) {
        ++copies;
        return *this;
    }
    MoveNoexcept(MoveNoexcept&& other) noexcept : value(other.value) {
        ++moves;
    }
    MoveNoexcept& operator=(MoveNoexcept&&) noexcept {
        ++moves;
        return *this;
    }
    int value = 0;
    static int copies;
    static int moves;
};

// A type whose move constructor may throw: vector reallocation must fall
// back to copies (std::move_if_noexcept).
struct MoveThrowing {
    explicit MoveThrowing(int v) : value(v) {}
    MoveThrowing(const MoveThrowing& other) : value(other.value) { ++copies; }
    MoveThrowing& operator=(const MoveThrowing&) {
        ++copies;
        return *this;
    }
    MoveThrowing(MoveThrowing&& other) : value(other.value) { ++moves; }
    MoveThrowing& operator=(MoveThrowing&&) { ++moves; return *this; }
    int value = 0;
    static int copies;
    static int moves;
};

// Trivial type: std::copy on a range should become memmove.
struct PointPlain {
    int x = 0;
    int y = 0;
};

// Same layout, but a user-declared (empty) destructor. This makes the type
// non-trivially-copyable, so std::copy cannot be replaced by memmove, and
// the implicit move constructor is suppressed (book PDF p.77-78).
struct PointEmptyDtor {
    int x = 0;
    int y = 0;
    ~PointEmptyDtor() {}
};

}  // namespace nomv
}  // namespace chp
