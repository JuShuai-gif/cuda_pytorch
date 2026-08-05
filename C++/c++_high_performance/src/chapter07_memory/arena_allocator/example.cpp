// Arena + custom STL allocator.
//
// The book (PDF p.199-208) builds an Arena<N> bump allocator and a stateful
// ShortAlloc<T,N> that containers (std::set, std::vector) can use to draw
// memory from a stack buffer instead of the global heap.

#include "arena.hpp"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <new>
#include <set>
#include <vector>

namespace {

long g_heap_allocs = 0;
bool g_tracking = false;

}  // namespace

void* operator new(std::size_t size) {
    if (g_tracking) {
        ++g_heap_allocs;
    }
    if (void* p = std::malloc(size)) {
        return p;
    }
    throw std::bad_alloc();
}

void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

namespace {

struct TrackingGuard {
    long a0 = g_heap_allocs;
    TrackingGuard() { g_tracking = true; }
    ~TrackingGuard() { g_tracking = false; }
    long heap_allocs() const { return g_heap_allocs - a0; }
};

}  // namespace

int main() {
    std::printf("== arena_allocator ==\n");

    using chp::arena::Arena;
    using chp::arena::ShortAlloc;

    // --- std::set with a stack arena (book PDF p.208) ---
    using SmallSet = std::set<int, std::less<int>, ShortAlloc<int, 512>>;
    Arena<512> stack_arena;
    SmallSet unique_numbers{ShortAlloc<int, 512>{stack_arena}};

    TrackingGuard g;
    for (int i = 0; i < 10; ++i) {
        unique_numbers.insert(i);  // 10 unique values fit in the arena
    }
    std::printf("set with arena: size=%zu arena used=%zu heap allocs=%ld\n",
                unique_numbers.size(), stack_arena.used(), g.heap_allocs());
    // All node allocations come from the arena; no heap allocation happens.
    // (A larger set would overflow the arena and fall back to operator new.)

    // --- direct Arena use with placement new (book PDF p.203) ---
    Arena<1024> user_arena;
    auto* p1 = new (user_arena.allocate(sizeof(int))) int{1};
    auto* p2 = new (user_arena.allocate(sizeof(int))) int{2};
    std::printf("arena ints: %d %d, used=%zu\n", *p1, *p2,
                user_arena.used());

    return 0;
}
