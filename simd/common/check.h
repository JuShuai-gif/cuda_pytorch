#pragma once

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * check.h -- Lightweight test assertion framework.
 *
 * Each macro checks a condition, prints PASS or FAIL, and exits on failure.
 * Designed for quick correctness validation of SIMD kernels against scalar
 * reference implementations.
 *
 * Usage:
 *   CHECK_EQ(a, b, "message");
 *   CHECK_NEAR(a, b, 1e-5, "message");
 *   CHECK_NEAR_ARRAY(a, b, n, 1e-5f, "message");
 *   CHECK_TRUE(cond, "message");
 */

#ifdef __cplusplus
extern "C" {
#endif

#define CHECK_EQ(a, b, msg) do {                                                \
    if ((a) == (b)) {                                                           \
        printf("  [PASS] %s\n", msg);                                           \
    } else {                                                                    \
        printf("  [FAIL] %s: expected %lld, got %lld (%s:%d)\n",                \
               msg, (long long)(intptr_t)(b), (long long)(intptr_t)(a),         \
               __FILE__, __LINE__);                                             \
        exit(1);                                                                \
    }                                                                           \
} while (0)

#define CHECK_NEAR(a, b, tol, msg) do {                                         \
    double _ck_diff = fabs((double)(a) - (double)(b));                          \
    if (_ck_diff <= (double)(tol)) {                                            \
        printf("  [PASS] %s\n", msg);                                           \
    } else {                                                                    \
        printf("  [FAIL] %s: |%g - %g| = %g > %g (%s:%d)\n",                    \
               msg, (double)(a), (double)(b), _ck_diff, (double)(tol),           \
               __FILE__, __LINE__);                                             \
        exit(1);                                                                \
    }                                                                           \
} while (0)

#define CHECK_NEAR_ARRAY(a, b, n, tol, msg) do {                                \
    int _ck_ok = 1;                                                             \
    size_t _ck_fail_i = 0;                                                      \
    double _ck_fail_a = 0, _ck_fail_b = 0;                                      \
    for (size_t _ck_i = 0; _ck_i < (size_t)(n); ++_ck_i) {                     \
        double _ck_d = fabs((double)(a)[_ck_i] - (double)(b)[_ck_i]);           \
        if (_ck_d > (double)(tol)) {                                            \
            _ck_ok = 0;                                                         \
            _ck_fail_i = _ck_i;                                                 \
            _ck_fail_a = (double)(a)[_ck_i];                                    \
            _ck_fail_b = (double)(b)[_ck_i];                                    \
            break;                                                              \
        }                                                                       \
    }                                                                           \
    if (_ck_ok) {                                                               \
        printf("  [PASS] %s (%zu elements)\n", msg, (size_t)(n));               \
    } else {                                                                    \
        printf("  [FAIL] %s: mismatch at index %zu: expected %g, got %g "       \
               "(tolerance %g) (%s:%d)\n",                                      \
               msg, _ck_fail_i, _ck_fail_b, _ck_fail_a, (double)(tol),          \
               __FILE__, __LINE__);                                             \
        exit(1);                                                                \
    }                                                                           \
} while (0)

#define CHECK_TRUE(cond, msg) do {                                              \
    if (cond) {                                                                 \
        printf("  [PASS] %s\n", msg);                                           \
    } else {                                                                    \
        printf("  [FAIL] %s: condition is false (%s:%d)\n",                     \
               msg, __FILE__, __LINE__);                                        \
        exit(1);                                                                \
    }                                                                           \
} while (0)

#ifdef __cplusplus
}
#endif
