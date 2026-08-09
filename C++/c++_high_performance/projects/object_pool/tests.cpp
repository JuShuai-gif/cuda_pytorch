// Correctness checks for ObjectPool.

#include <cstddef>
#include <cstdio>
#include <memory>
#include <new>

#include "object_pool.hpp"

namespace chp {
inline int g_test_failures = 0;
}  // namespace chp

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::fprintf(stderr, "[FAIL] %s:%d: %s\n", __FILE__, __LINE__,   \
                         #cond);                                             \
            ++chp::g_test_failures;                                          \
        }                                                                    \
    } while (0)

namespace {

struct IntBox {
    IntBox(int v) : value(v) { ++g_ctor; }
    ~IntBox() { ++g_dtor; }
    int value;
    inline static int g_ctor = 0;
    inline static int g_dtor = 0;
};

}  // namespace

int main() {
    {
        chp::ObjectPool pool(sizeof(int), 4);
        CHECK(pool.capacity() == 4);
        CHECK(pool.in_use() == 0);
        CHECK(pool.free_count() == 4);

        // Allocate all blocks.
        void* blocks[4];
        for (int i = 0; i < 4; ++i) {
            blocks[i] = pool.allocate();
            CHECK(blocks[i] != nullptr);
            CHECK(pool.contains(blocks[i]));
        }
        // Exhausted.
        CHECK(pool.allocate() == nullptr);
        CHECK(pool.in_use() == 4);

        // Return blocks and reuse them.
        pool.deallocate(blocks[1]);
        CHECK(pool.free_count() == 1);
        void* again = pool.allocate();
        CHECK(again == blocks[1]);  // LIFO free list reuses the same block

        // Alignment: every block is max_align_t aligned.
        for (int i = 0; i < 4; ++i) {
            CHECK(reinterpret_cast<std::uintptr_t>(blocks[i]) %
                      alignof(std::max_align_t) == 0);
        }
    }

    // RAII wrapper: construct in pool, destroy returns the block.
    {
        chp::ObjectPool pool(sizeof(IntBox), 2);
        {
            chp::Pooled<IntBox> a(pool, 42);
            chp::Pooled<IntBox> b(pool, 7);
            CHECK(a->value == 42);
            CHECK(b->value == 7);
            CHECK(pool.in_use() == 2);
        }
        CHECK(IntBox::g_ctor == 2);
        CHECK(IntBox::g_dtor == 2);
        CHECK(pool.in_use() == 0);
        CHECK(pool.free_count() == 2);
    }

    // Exhaustion throws bad_alloc instead of UB.
    {
        chp::ObjectPool pool(sizeof(int), 1);
        bool threw = false;
        try {
            chp::Pooled<int> a(pool, 1);
            chp::Pooled<int> b(pool, 2);
        } catch (const std::bad_alloc&) {
            threw = true;
        }
        CHECK(threw);
    }

    if (chp::g_test_failures == 0) {
        std::printf("[PASS] object_pool: all checks passed\n");
        return 0;
    }
    std::printf("[FAIL] object_pool: %d check(s) failed\n",
                chp::g_test_failures);
    return 1;
}
