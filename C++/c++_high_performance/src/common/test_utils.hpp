#pragma once

#include <cstdio>

namespace chp {

inline int g_test_failures = 0;

// Prints the verdict for a test suite and returns the process exit code.
inline int test_summary(const char* suite) {
    if (g_test_failures == 0) {
        std::printf("[PASS] %s: all checks passed\n", suite);
        return 0;
    }
    std::printf("[FAIL] %s: %d check(s) failed\n", suite, g_test_failures);
    return 1;
}

}  // namespace chp

#define CHP_CHECK(cond)                                                       \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::fprintf(stderr, "[FAIL] %s:%d: %s\n", __FILE__, __LINE__,    \
                         #cond);                                              \
            ++chp::g_test_failures;                                           \
        }                                                                     \
    } while (0)
