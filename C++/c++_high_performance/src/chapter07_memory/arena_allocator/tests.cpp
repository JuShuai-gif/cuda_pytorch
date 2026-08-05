#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <new>
#include <set>
#include <type_traits>
#include <vector>

#include "arena.hpp"
#include "test_utils.hpp"

using chp::arena::Arena;
using chp::arena::ShortAlloc;

int main() {
    // --- Arena basics ---
    {
        Arena<64> a;
        CHP_CHECK(a.size() == 64);
        CHP_CHECK(a.used() == 0);

        char* p = a.allocate(8);
        CHP_CHECK(p != nullptr);
        // Allocations are rounded up to alignof(max_align_t) (16 on x86-64),
        // so an 8-byte request consumes 16 bytes of the arena.
        CHP_CHECK(a.used() == alignof(std::max_align_t));
        // Allocation is max-aligned.
        CHP_CHECK(reinterpret_cast<std::uintptr_t>(p) %
                      alignof(std::max_align_t) ==
                  0);

        // Alignment is honored: a small request rounds up.
        char* q = a.allocate(1);
        CHP_CHECK(reinterpret_cast<std::uintptr_t>(q) %
                      alignof(std::max_align_t) ==
                  0);
        CHP_CHECK(a.used() == 2 * alignof(std::max_align_t));

        // Deallocate the top block reclaims it.
        a.deallocate(q, 1);
        CHP_CHECK(a.used() == alignof(std::max_align_t));

        // With three blocks, deallocating a NON-top block is a no-op.
        char* r = a.allocate(1);
        char* s = a.allocate(1);
        CHP_CHECK(a.used() == 3 * alignof(std::max_align_t));
        a.deallocate(r, 1);  // r is in the middle -> ignored
        CHP_CHECK(a.used() == 3 * alignof(std::max_align_t));
        a.deallocate(s, 1);  // s is the top -> reclaimed
        CHP_CHECK(a.used() == 2 * alignof(std::max_align_t));
        a.deallocate(p, 8);  // p is now NOT the top (r is) -> no-op
        CHP_CHECK(a.used() == 2 * alignof(std::max_align_t));

        // reset() hands the whole buffer back.
        a.reset();
        CHP_CHECK(a.used() == 0);
    }

    // --- Overflow falls back to operator new ---
    {
        Arena<16> a;
        char* big = a.allocate(100);  // does not fit -> heap
        CHP_CHECK(big != nullptr);
        a.deallocate(big, 100);  // forwarded to ::operator delete
        CHP_CHECK(a.used() == 0);
    }

    // --- Arena is not copyable / movable ---
    static_assert(!std::is_copy_constructible<Arena<16>>::value,
                  "Arena must not be copyable");
    static_assert(!std::is_copy_assignable<Arena<16>>::value,
                  "Arena must not be copy-assignable");

    // --- ShortAlloc with std::set ---
    {
        Arena<512> arena;
        std::set<int, std::less<int>, ShortAlloc<int, 512>> s{
            ShortAlloc<int, 512>{arena}};
        for (int i = 0; i < 20; ++i) {
            s.insert(i);
        }
        CHP_CHECK(s.size() == 20);
        CHP_CHECK(arena.used() > 0);   // nodes drawn from the arena
        CHP_CHECK(arena.used() <= 512);
        for (int i = 0; i < 20; ++i) {
            CHP_CHECK(s.count(i) == 1);
        }
        s.clear();
        CHP_CHECK(s.empty());
    }

    // --- ShortAlloc with std::vector ---
    {
        Arena<256> arena;
        std::vector<int, ShortAlloc<int, 256>> v{ShortAlloc<int, 256>{arena}};
        v.reserve(32);
        for (int i = 0; i < 32; ++i) {
            v.push_back(i);
        }
        CHP_CHECK(v.size() == 32);
        CHP_CHECK(v.front() == 0 && v.back() == 31);
    }

    // --- allocator equality ---
    {
        Arena<128> arena;
        ShortAlloc<int, 128> a1{arena};
        ShortAlloc<int, 128> a2{arena};
        CHP_CHECK(a1 == a2);
        Arena<128> arena2;
        ShortAlloc<int, 128> a3{arena2};
        CHP_CHECK(a1 != a3);
    }

    return chp::test_summary("arena_allocator");
}
