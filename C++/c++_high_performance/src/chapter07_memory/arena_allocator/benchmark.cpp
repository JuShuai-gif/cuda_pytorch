#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <new>
#include <set>
#include <vector>

#include "arena.hpp"
#include "benchmark.hpp"

namespace {

constexpr std::size_t kIterations = 5;
constexpr std::size_t kRounds = 7;
constexpr std::size_t kWarmup = 2;

}  // namespace

int main() {
    std::printf("== arena_allocator benchmark ==\n");
    std::printf("Inserting 40k unique ints into std::set.\n\n");

    using chp::arena::Arena;
    using chp::arena::ShortAlloc;

    // Default allocator: every node from the global heap.
    const auto r_default = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            std::set<int> s;
            for (int i = 0; i < 100'000; ++i) {
                s.insert(i);
            }
            acc += static_cast<std::uint64_t>(s.size());
        });

    // Arena allocator: nodes come from a big stack buffer.
    const auto r_arena = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            // 40k nodes * ~40 B (int key + rb-tree overhead) ~ 1.6 MB, which
            // fits a 2 MB stack arena. (A larger arena risks a stack overflow.)
            constexpr std::size_t kArenaSize = 2 * 1024 * 1024;
            Arena<kArenaSize> arena;
            std::set<int, std::less<int>, ShortAlloc<int, kArenaSize>> s{
                ShortAlloc<int, kArenaSize>{arena}};
            for (int i = 0; i < 40'000; ++i) {
                s.insert(i);
            }
            acc += static_cast<std::uint64_t>(s.size());
        });

    chp::print_result("std::set with global heap allocator", r_default);
    chp::print_result("std::set with stack arena allocator", r_arena);

    const double ratio = r_default.mean_ns / r_arena.mean_ns;
    std::printf("default/arena time ratio: %.2fx\n", ratio);
    return 0;
}
