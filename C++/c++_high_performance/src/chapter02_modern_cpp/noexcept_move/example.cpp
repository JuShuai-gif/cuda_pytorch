// noexcept and the empty-destructor trap.
//
// The book (PDF p.72) warns: "Do not forget to mark your move-constructors
// and move-assignment operators as noexcept ... Not marking them noexcept
// prevents STL containers and algorithms from utilizing them."
//
// The book (PDF p.77-78) also warns that a user-declared empty destructor
// `~Point(){}` blocks optimizations (std::copy becomes a loop instead of
// memmove) and suppresses the implicit move constructor.

#include "baseline.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <type_traits>
#include <vector>

namespace {

using chp::nomv::MoveNoexcept;
using chp::nomv::MoveThrowing;
using chp::nomv::PointEmptyDtor;
using chp::nomv::PointPlain;

// Inlined copy helpers for assembly inspection:
//   g++ -std=c++17 -O3 -S src/chapter02_modern_cpp/noexcept_move/example.cpp
[[gnu::noinline]] void copy_plain(PointPlain* src, PointPlain* dst,
                                  std::size_t n) {
    std::copy(src, src + n, dst);
}

[[gnu::noinline]] void copy_empty_dtor(PointEmptyDtor* src, PointEmptyDtor* dst,
                                       std::size_t n) {
    std::copy(src, src + n, dst);
}

}  // namespace

int main() {
    std::printf("== noexcept_move ==\n");
    std::printf("is_nothrow_move_constructible: MoveNoexcept=%d "
                "MoveThrowing=%d\n",
                std::is_nothrow_move_constructible<MoveNoexcept>::value,
                std::is_nothrow_move_constructible<MoveThrowing>::value);
    std::printf("is_trivially_copyable: PointPlain=%d PointEmptyDtor=%d\n",
                std::is_trivially_copyable<PointPlain>::value,
                std::is_trivially_copyable<PointEmptyDtor>::value);

    // --- vector reallocation: noexcept move vs throwing move ---
    std::vector<MoveNoexcept> vne;
    MoveNoexcept::copies = 0;
    MoveNoexcept::moves = 0;
    for (int i = 0; i < 100; ++i) {
        vne.emplace_back(i);
    }
    std::printf("vector<MoveNoexcept>  growth: copies=%d moves=%d\n",
                MoveNoexcept::copies, MoveNoexcept::moves);

    std::vector<MoveThrowing> vt;
    MoveThrowing::copies = 0;
    MoveThrowing::moves = 0;
    for (int i = 0; i < 100; ++i) {
        vt.emplace_back(i);
    }
    std::printf("vector<MoveThrowing>  growth: copies=%d moves=%d\n",
                MoveThrowing::copies, MoveThrowing::moves);

    std::printf("\nIf MoveThrowing reallocated with moves the moves counter\n");
    std::printf("would dominate; a non-noexcept move forces copies instead.\n");

    // --- assembly inspection helpers are referenced to keep them emitted ---
    PointPlain src_plain[4]{};
    PointEmptyDtor src_empty[4]{};
    PointPlain dst_plain[4]{};
    PointEmptyDtor dst_empty[4]{};
    copy_plain(src_plain, dst_plain, 4);
    copy_empty_dtor(src_empty, dst_empty, 4);
    std::printf("copy helpers referenced.\n");
    return 0;
}
