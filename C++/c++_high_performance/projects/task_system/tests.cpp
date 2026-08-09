// Correctness checks for TaskSystem.

#include <atomic>
#include <cstdio>
#include <future>
#include <stdexcept>
#include <thread>
#include <vector>

#include "task_system.hpp"

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

int main() {

    // 1. Return values via future.
    {
        chp::TaskSystem pool(4);
        auto f = pool.submit([] { return 2 + 3; });
        CHECK(f.get() == 5);
    }

    // 2. Multiple tasks all execute and return.
    {
        chp::TaskSystem pool(4);
        std::vector<std::future<int>> futures;
        for (int i = 0; i < 100; ++i) {
            futures.push_back(pool.submit([i] { return i * i; }));
        }
        int sum = 0;
        for (auto& f : futures) {
            sum += f.get();
        }
        // sum of squares 0..99 = 99*100*199/6
        CHECK(sum == 99 * 100 * 199 / 6);
    }

    // 3. More tasks than workers (queueing works).
    {
        chp::TaskSystem pool(2);
        std::vector<std::future<void>> futures;
        for (int i = 0; i < 50; ++i) {
            futures.push_back(pool.submit([] {}));
        }
        for (auto& f : futures) {
            f.get();
        }
    }

    // 4. Exceptions propagate through the future.
    {
        chp::TaskSystem pool(2);
        auto f = pool.submit([]() -> int { throw std::runtime_error{"boom"}; });
        bool caught = false;
        try {
            (void)f.get();
        } catch (const std::runtime_error&) {
            caught = true;
        }
        CHECK(caught);
    }

    // 5. Tasks with arguments are forwarded.
    {
        chp::TaskSystem pool(2);
        auto f = pool.submit([](int a, int b) { return a * b; }, 6, 7);
        CHECK(f.get() == 42);
    }

    if (chp::g_test_failures == 0) {
        std::printf("[PASS] task_system: all checks passed\n");
        return 0;
    }
    std::printf("[FAIL] task_system: %d check(s) failed\n",
                chp::g_test_failures);
    return 1;
}
