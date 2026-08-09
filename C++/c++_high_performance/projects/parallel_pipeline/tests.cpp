// Correctness checks for parallel_pipeline.

#include <cstdio>
#include <cstddef>
#include <numeric>
#include <vector>

#include "parallel_pipeline.hpp"

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

bool even(long v) { return (v % 2) == 0; }
bool all_true(long) { return true; }
bool none(long) { return false; }
long plus(long a, long b) { return a + b; }

}  // namespace

int main() {

    std::vector<int> src(10'000);
    std::iota(src.begin(), src.end(), 0);

    // map = x*x, filter = even, reduce = sum.
    {
        const auto result = chp::parallel_pipeline<int, long>(
            src, [](int v) { return static_cast<long>(v) * v; }, even,
            plus, 0L);
        // Sum of squares of even numbers 0..9999.
        long expect = 0;
        for (int v : src) {
            if (v % 2 == 0) {
                expect += static_cast<long>(v) * v;
            }
        }
        CHECK(result == expect);
    }

    // filter keeps everything: reduce == plain sum of squares.
    {
        const auto result = chp::parallel_pipeline<int, long>(
            src, [](int v) { return static_cast<long>(v) * v; }, all_true,
            plus, 0L);
        long expect = 0;
        for (int v : src) {
            expect += static_cast<long>(v) * v;
        }
        CHECK(result == expect);
    }

    // filter keeps nothing: empty reduce == init.
    {
        const auto result = chp::parallel_pipeline<int, long>(
            src, [](int v) { return static_cast<long>(v) * v; }, none,
            plus, 100L);
        CHECK(result == 100L);
    }

    // Non-commutative-safe usage: multiplication over 1..n (commutative here).
    {
        const auto result = chp::parallel_pipeline<int, long>(
            src, [](int v) { return static_cast<long>(v) + 1; }, all_true,
            [](long a, long b) { return a * b; }, 1L);
        long expect = 1;
        for (int v : src) {
            expect *= static_cast<long>(v) + 1;
        }
        CHECK(result == expect);
    }

    if (chp::g_test_failures == 0) {
        std::printf("[PASS] parallel_pipeline: all checks passed\n");
        return 0;
    }
    std::printf("[FAIL] parallel_pipeline: %d check(s) failed\n",
                chp::g_test_failures);
    return 1;
}
