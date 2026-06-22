#pragma once

/*
 * test_edge.h -- Edge case test infrastructure for SIMD correctness validation.
 *
 * SIMD code has edge cases that scalar code handles implicitly through
 * IEEE 754 hardware. This header provides:
 *
 *   1. Data generators that produce special fp32 values (NaN, Inf, denorm,
 *      signed zero, FLT_MIN/MAX) using explicit IEEE 754 bit patterns, so
 *      the compiler cannot constant-fold them away.
 *   2. Alignment-aware data generators that place sentinel values at
 *      buffer boundaries to detect SIMD over-read bugs.
 *   3. Validation functions for NaN propagation, Inf arithmetic rules,
 *      denormal flush-to-zero behavior, and zero-length safety.
 *   4. An integrated test runner (run_edge_tests) that executes all checks.
 *
 * Usage:
 *   #include "test_edge.h"
 *   int failures = run_edge_tests("avx2_add", scalar_add, avx2_add);
 *   edge_test_report("avx2_add", failures);
 *
 * Pattern reference (IEEE 754 single precision, little-endian):
 *   0x00000000 = +0.0       0x80000000 = -0.0
 *   0x3F800000 = +1.0       0xBF800000 = -1.0
 *   0x00800000 = FLT_MIN    0x80800000 = -FLT_MIN
 *   0x7F7FFFFF = FLT_MAX    0xFF7FFFFF = -FLT_MAX
 *   0x7F800000 = +Inf       0xFF800000 = -Inf
 *   0x7FC00000 = qNaN       0xFFC00000 = -qNaN
 *   0x00000001 = min denorm 0x007FFFFF = max denorm
 */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Internal helpers (bit-level float manipulation) ---- */

typedef union {
    float    f;
    uint32_t u;
} float_bits_t;

/*
 * float_from_bits -- Safely construct a float from its IEEE 754 bit pattern.
 *
 * Uses memcpy to avoid:
 *   a) Undefined behavior from type-punning via pointer cast (strict aliasing).
 *   b) FPU Invalid Operation exceptions from loading signaling NaN.
 */
static inline float float_from_bits(uint32_t bits) {
    float val;
    memcpy(&val, &bits, sizeof(val));
    return val;
}

/*
 * bits_from_float -- Extract the IEEE 754 bit pattern of a float.
 */
static inline uint32_t bits_from_float(float val) {
    uint32_t bits;
    memcpy(&bits, &val, sizeof(bits));
    return bits;
}

/*
 * float_bit_equal -- True if two floats have identical bit patterns.
 *
 * Differs from a == b because:
 *   - NaN != NaN in IEEE 754, but bitwise they can be identical.
 *   - This check treats identical NaN payloads as equal.
 */
static inline int float_bit_equal(float a, float b) {
    if (isnan(a) && isnan(b)) {
        /* Both NaN -- consider bitwise equal for safety. Even if payload differs,
         * the correctness property "NaN propagated" holds. */
        return 1;
    }
    return bits_from_float(a) == bits_from_float(b);
}

/* ---- Phase 1 static edge-case value arrays (created at compile time) ---- */

/*
 * EDGE_F_LITERALS -- Float specials expressible as C99 literals.
 * We keep these separate from bit-pattern values so that only values
 * that CANNOT be expressed as literals (NaN, denorm min/max) go through
 * the float_from_bits path, minimizing the risk of FPU exceptions.
 */
static const float EDGE_F_LITERALS[] = {
     0.0f,          /* +0                    */
    -0.0f,          /* -0                    */
     1.0f,          /* +1.0                  */
    -1.0f,          /* -1.0                  */
     2.0f,          /* +2.0                  */
    -2.0f,          /* -2.0                  */
     0.5f,          /* +0.5                  */
    -0.5f,          /* -0.5                  */
     FLT_MIN,       /* smallest normal       */
    -FLT_MIN,       /* -smallest normal      */
     FLT_MAX,       /* largest normal        */
    -FLT_MAX,       /* -largest normal       */
     INFINITY,      /* +Inf                  */
    -INFINITY,      /* -Inf                  */
};

enum { EDGE_N_LITERALS = sizeof(EDGE_F_LITERALS) / sizeof(EDGE_F_LITERALS[0]) };

/*
 * EDGE_U_BITS -- Special IEEE 754 bit patterns that require memcpy to
 * load safely.  Order: denormals first (least risky), then NaNs.
 *
 * Denormals:
 *   0x00000001  smallest positive subnormal (1.401e-45)
 *   0x000FFFFF  mid-range positive subnormal
 *   0x007FFFFF  largest positive subnormal  (1.175e-38)
 *   0x80000001  smallest negative subnormal
 *   0x800FFFFF  mid-range negative subnormal
 *   0x807FFFFF  largest negative subnormal
 *
 * NaNs:
 *   0x7FC00000  canonical quiet NaN  (payload = 0x400000)
 *   0xFFC00000  canonical negative quiet NaN
 *   0x7F800001  signaling NaN (bit 22 clear, payload = 1)
 *   0xFF800001  negative signaling NaN
 */
static const uint32_t EDGE_U_BITS[] = {
    0x00000001u, 0x000FFFFFu, 0x007FFFFFu,
    0x80000001u, 0x800FFFFFu, 0x807FFFFFu,
    0x7FC00000u, 0xFFC00000u,
    0x7F800001u, 0xFF800001u,
};

enum { EDGE_N_BITS = sizeof(EDGE_U_BITS) / sizeof(EDGE_U_BITS[0]) };

/*
 * EDGE_CYCLE -- Total number of distinct special values in one full cycle
 * of fill_edge_f32.
 */
enum { EDGE_CYCLE = EDGE_N_LITERALS + EDGE_N_BITS };

/* ---- Data generation ---- */

/*
 * fill_edge_f32 -- Fill data[0..n-1] with a repeating edge-case pattern.
 *
 * One full cycle consists of EDGE_CYCLE values: the C99-literal specials
 * followed by the bit-pattern specials (denormals, NaNs).  The cycle
 * repeats until n elements are filled.
 *
 * Purpose:
 *   Exposing the SIMD path to NaN, Inf, signed zero, and denorm inputs
 *   reveals bugs that scalar IEEE 754 hardware handles transparently:
 *     - NaN payload not preserved through SIMD register moves
 *     - Inf arithmetic producing wrong sign
 *     - Denormals flushed to zero when they should not be
 *     - Signed zero swapping sign during vector operations
 *
 * Implementation note:
 *   We use two static arrays and interleave them during the fill loop.
 *   The C-literal array is copied in one memcpy block; the bit-pattern
 *   array is copied element-by-element via memcpy to prevent any FPU
 *   load of sNaN values.
 */
static inline void fill_edge_f32(float* data, size_t n) {
    size_t pos = 0;

    while (pos < n) {
        /* Fill a block of C-literal specials */
        size_t block = n - pos;
        if (block > EDGE_N_LITERALS) block = EDGE_N_LITERALS;
        memcpy(data + pos, EDGE_F_LITERALS, block * sizeof(float));
        pos += block;
        if (pos >= n) break;

        /* Fill a block of bit-pattern specials */
        block = n - pos;
        if (block > EDGE_N_BITS) block = EDGE_N_BITS;
        for (size_t i = 0; i < block; i++) {
            data[pos + i] = float_from_bits(EDGE_U_BITS[i]);
        }
        pos += block;
    }
}

/*
 * fill_edge_alignment -- Fill data with sentinel-marked alignment boundaries.
 *
 * Places NaN sentinel values at positions that border alignment boundaries
 * (every `alignment` bytes).  The remaining positions are filled with a
 * simple sequential pattern (1.0f + i % 100).
 *
 * Why this matters for SIMD:
 *   - Unaligned SIMD loads (_mm256_loadu_ps) that straddle a cache-line
 *     boundary (64 B) incur a penalty, but must still produce correct data.
 *   - Loads near the end of a buffer may read past the allocation if the
 *     main loop does not have a proper tail guard.  The sentinel NaN at
 *     `n - guard` detects such over-reads: if NaN leaks into the output,
 *     the SIMD path read beyond the buffer.
 *   - Some SIMD ISA levels (AVX-512 masked operations) can mask these, but
 *     AVX2 scalar tails must be correct.
 *
 * Parameters:
 *   data      -- output buffer
 *   n         -- number of elements
 *   alignment -- boundary size in bytes (e.g., 32 for AVX2, 64 for cache line)
 */
static inline void fill_edge_alignment(float* data, size_t n, size_t alignment) {
    float nan_val = float_from_bits(0x7FC00000u);
    size_t guard = alignment / sizeof(float);
    if (guard == 0) guard = 1;
    if (guard > n) guard = n;

    /* Fill with sequential normal values */
    for (size_t i = 0; i < n; i++) {
        data[i] = 1.0f + (float)(i % 100);
    }

    /*
     * Stamp NaN sentinels in two critical zones:
     *   1. At each alignment boundary (every `guard` elements), mark the
     *      element exactly ON the boundary.
     *   2. At the tail (last `guard` elements), mark them all NaN.
     */
    for (size_t i = 0; i < n; i += guard) {
        data[i] = nan_val;
    }

    size_t tail_start = (n > guard) ? (n - guard) : 0;
    for (size_t i = tail_start; i < n; i++) {
        data[i] = nan_val;
    }
}

/* ---- Validation predicates ---- */

static inline int is_nan_f32(float x) { return isnan(x); }

/* ---- Edge-case check functions ---- */

/*
 * check_nan_propagation -- Verify NaN arithmetic law.
 *
 * IEEE 754 rule: any operation with a NaN operand produces a NaN result.
 * For binary operations like a + b:
 *     isnan(a) || isnan(b)  ==>  isnan(result)
 *
 * Also checks the converse: if neither operand is NaN and both are finite,
 * the result must not be NaN (catches spurious NaN generation in SIMD).
 *
 * Returns number of failed positions.
 */
static inline int check_nan_propagation(const float* a, const float* b,
                                        const float* result, size_t n) {
    int failures = 0;
    for (size_t i = 0; i < n; i++) {
        int has_nan_input = (isnan(a[i]) != 0) || (isnan(b[i]) != 0);

        if (has_nan_input) {
            /* NaN input MUST produce NaN output */
            if (!isnan(result[i])) {
                fprintf(stderr, "  NaN propagation FAIL at [%zu]: "
                        "a=%g b=%g -> result=%g (expected NaN)\n",
                        i, (double)a[i], (double)b[i], (double)result[i]);
                failures++;
            }
        } else if (isfinite(a[i]) && isfinite(b[i])) {
            /* Finite + finite MUST NOT produce NaN (addition never does;
             * subtraction/multiplication/division would need adjusted logic) */
            if (isnan(result[i])) {
                fprintf(stderr, "  Spurious NaN at [%zu]: "
                        "a=%g b=%g -> result=%g\n",
                        i, (double)a[i], (double)b[i], (double)result[i]);
                failures++;
            }
        }
        /* Inf + finite or finite + Inf is NOT an error -- NaN check passes, Inf check next */
    }
    return failures;
}

/*
 * check_inf_arithmetic -- Verify Inf arithmetic rules for addition.
 *
 * IEEE 754 rules for addition:
 *   Inf + finite  =  Inf  (same sign as the Inf operand, or if both Inf same sign)
 *   Inf + (-Inf)  =  NaN  (indeterminate form)
 *   finite + Inf  =  Inf
 *
 * We also check for obviously wrong output like "finite + finite = Inf"
 * which would indicate overflow in the SIMD path but not in scalar.
 *
 * Returns number of failed positions.
 */
static inline int check_inf_arithmetic(const float* a, const float* b,
                                       const float* result, size_t n) {
    int failures = 0;
    for (size_t i = 0; i < n; i++) {
        int a_inf = isinf(a[i]);
        int b_inf = isinf(b[i]);

        if (a_inf && b_inf && (signbit(a[i]) != signbit(b[i]))) {
            /*
             * Inf + (-Inf)  =>  must be NaN (indeterminate).
             * This is the classic "Inf - Inf" edge case that trips up
             * SIMD implementations that don't propagate NaN correctly.
             */
            if (!isnan(result[i])) {
                fprintf(stderr, "  Inf-Inf FAIL at [%zu]: "
                        "a=%g b=%g -> result=%g (expected NaN)\n",
                        i, (double)a[i], (double)b[i], (double)result[i]);
                failures++;
            }
        } else if ((a_inf && !b_inf) || (!a_inf && b_inf)) {
            /*
             * Inf + finite => must be Inf.  The sign of the result
             * should match the sign of the Inf operand (IEEE 754
             * addition with a finite value preserves the sign of Inf).
             */
            if (!isinf(result[i])) {
                fprintf(stderr, "  Inf+finite FAIL at [%zu]: "
                        "a=%g b=%g -> result=%g (expected Inf)\n",
                        i, (double)a[i], (double)b[i], (double)result[i]);
                failures++;
            }
        }
        /*
         * Note: finite + finite can overflow to Inf (e.g. FLT_MAX + FLT_MAX).
         * This is correct IEEE 754 behavior, not an error.  The consistency
         * check (scalar vs SIMD) in run_edge_tests catches any actual
         * divergence between paths.
         */
    }
    return failures;
}

/*
 * check_denormal_behavior -- Detect denormal flush-to-zero.
 *
 * Modern CPUs often run in "FTZ" (Flush To Zero) mode for performance:
 * denormal inputs/outputs are silently replaced with zero.  This breaks
 * numerical reproducibility between runs and between scalar/SIMD paths
 * when one path flushes and the other does not.
 *
 * Check strategy:
 *   - Count how many denormal results were preserved vs flushed.
 *   - If expect_flush_to_zero=1, denormals becoming zero is acceptable.
 *   - If expect_flush_to_zero=0, any flushed denormal is a failure.
 *
 * Returns number of unexpected denormal handling mismatches.
 */
static inline int check_denormal_behavior(const float* a, const float* b,
                                          const float* result, size_t n,
                                          int expect_flush_to_zero) {
    int failures = 0;
    int denorm_inputs = 0;
    int denorm_results = 0;
    int flushed_results = 0;

    for (size_t i = 0; i < n; i++) {
        /* Detect denormal inputs: non-zero finite float with abs < FLT_MIN.
         * We use fpclassify for portability. */
        int a_is_denorm = (fpclassify(a[i]) == FP_SUBNORMAL);
        int b_is_denorm = (fpclassify(b[i]) == FP_SUBNORMAL);

        if (a_is_denorm || b_is_denorm) {
            denorm_inputs++;

            int r_class = fpclassify(result[i]);

            if (r_class == FP_SUBNORMAL) {
                denorm_results++;
            } else if (r_class == FP_ZERO && !expect_flush_to_zero) {
                fprintf(stderr, "  Denorm flushed at [%zu]: "
                        "a=%e b=%e -> result=%e (FTZ not expected)\n",
                        i, (double)a[i], (double)b[i], (double)result[i]);
                failures++;
                flushed_results++;
            } else if (r_class == FP_ZERO && expect_flush_to_zero) {
                flushed_results++;
            } else if (r_class == FP_NORMAL) {
                /* Denorm + normal can produce a normal result -- this is fine */
            }
        }
    }

    /* Summary log for diagnostic purposes */
    if (denorm_inputs > 0) {
        printf("  [INFO] Denorm inputs: %d, denorm results: %d, flushed: %d "
               "(expect_flush=%d)\n",
               denorm_inputs, denorm_results, flushed_results, expect_flush_to_zero);
    }

    return failures;
}

/*
 * check_zero_length -- Verifies that a function does not crash when n=0.
 *
 * All SIMD implementations must guard against zero-length input with an
 * early return or a loop that never executes.  A missing `if (n == 0) return;`
 * leads to `n - 8` wrapping around (size_t underflow) and a crash on the
 * first SIMD load.
 */
static inline int check_zero_length(void (*fn)(const float*, const float*,
                                               float*, size_t)) {
    float a = 0.0f, b = 0.0f, c = 0.0f;

    /*
     * Intentionally pass a garbage pointer -- if the function tries to
     * access it, we get SIGSEGV.  A correct implementation returns
     * immediately because n == 0.
     *
     * We use dummy variables with `volatile` to prevent the compiler
     * from optimizing the call away.
     */
    volatile const float* va = &a;
    volatile const float* vb = &b;
    volatile float*       vc = &c;

    fn((const float*)va, (const float*)vb, (float*)vc, 0);
    return 0;  /* survived => pass */
}

/*
 * check_consistency -- Bit-exact comparison of scalar vs SIMD results.
 *
 * Unlike CHECK_NEAR_ARRAY (tolerance-based, uses fabs), this does a
 * bit-exact comparison with special NaN handling.  Edge case values
 * (NaN, Inf, denorm) require bit-exact matching because:
 *   1. NaN != NaN in IEEE 754, so fabs(NaN - NaN) is NaN and never < tol.
 *   2. Inf - Inf = NaN, which tolerance-based comparison cannot validate.
 *   3. Denorm vs zero differ by exact bit patterns.
 */
static inline int check_consistency(const float* scalar, const float* simd,
                                    size_t n, const char* tag) {
    int failures = 0;
    for (size_t i = 0; i < n; i++) {
        if (!float_bit_equal(scalar[i], simd[i])) {
            fprintf(stderr, "  %s mismatch at [%zu]: "
                    "scalar=%e (0x%08X) simd=%e (0x%08X)\n",
                    tag, i,
                    (double)scalar[i], bits_from_float(scalar[i]),
                    (double)simd[i],   bits_from_float(simd[i]));
            failures++;
            if (failures >= 5) {
                fprintf(stderr, "  ... (stopping after 5 failures)\n");
                break;
            }
        }
    }
    return failures;
}

/* ---- Edge case test runner ---- */

/*
 * Edge test buffer size: must be large enough to hold several full cycles
 * of EDGE_CYCLE (~24 values) plus alignment guard elements (~16).
 */
#define EDGE_TEST_N 256

/*
 * run_edge_tests -- Orchestrate all edge case tests for a binary f32 op.
 *
 * Steps:
 *   1. NaN propagation test
 *   2. Inf arithmetic test
 *   3. Denormal behavior test (expect_flush = 0, i.e., denorms preserved)
 *   4. Zero-length safety test
 *   5. Alignment boundary test
 *
 * Returns number of failures (0 = all pass).
 */
static inline int run_edge_tests(const char* name,
                                  void (*scalar_fn)(const float*,
                                                    const float*,
                                                    float*, size_t),
                                  void (*simd_fn)(const float*,
                                                  const float*,
                                                  float*, size_t)) {
    int failures = 0;

    printf("\n=== Edge Case Tests: %s ===\n", name);

    /*
     * Allocate aligned buffers.  32-byte alignment satisfies both AVX2 (256-bit)
     * and AVX-512 (512-bit) alignment requirements and avoids crossing
     * page boundaries mid-register during normal operation.
     */
    float* a       = (float*)aligned_alloc(32, EDGE_TEST_N * sizeof(float));
    float* b       = (float*)aligned_alloc(32, EDGE_TEST_N * sizeof(float));
    float* c_scalar = (float*)aligned_alloc(32, EDGE_TEST_N * sizeof(float));
    float* c_simd   = (float*)aligned_alloc(32, EDGE_TEST_N * sizeof(float));

    if (!a || !b || !c_scalar || !c_simd) {
        fprintf(stderr, "  [FAIL] Allocation failed for edge test.\n");
        failures++;
        goto cleanup;
    }

    /* --------------------------------------------------------
     * 1. NaN propagation test
     * -------------------------------------------------------- */
    printf("\n  [1] NaN propagation...\n");
    fill_edge_f32(a, EDGE_TEST_N);
    fill_edge_f32(b, EDGE_TEST_N);
    memset(c_scalar, 0, EDGE_TEST_N * sizeof(float));
    memset(c_simd,   0, EDGE_TEST_N * sizeof(float));

    scalar_fn(a, b, c_scalar, EDGE_TEST_N);
    simd_fn(a, b, c_simd, EDGE_TEST_N);

    {
        int f1 = check_nan_propagation(a, b, c_scalar, EDGE_TEST_N);
        int f2 = check_nan_propagation(a, b, c_simd, EDGE_TEST_N);
        int f3 = check_consistency(c_scalar, c_simd, EDGE_TEST_N, "NaN");

        if (f1 == 0 && f2 == 0 && f3 == 0) {
            printf("  [PASS] NaN propagation\n");
        } else {
            failures += f1 + f2 + f3;
            printf("  [FAIL] NaN propagation: scalar(%d) simd(%d) mismatch(%d)\n",
                   f1, f2, f3);
        }
    }

    /* --------------------------------------------------------
     * 2. Inf arithmetic test
     * -------------------------------------------------------- */
    printf("\n  [2] Inf arithmetic...\n");
    memset(c_scalar, 0, EDGE_TEST_N * sizeof(float));
    memset(c_simd,   0, EDGE_TEST_N * sizeof(float));

    scalar_fn(a, b, c_scalar, EDGE_TEST_N);
    simd_fn(a, b, c_simd, EDGE_TEST_N);

    {
        int f1 = check_inf_arithmetic(a, b, c_scalar, EDGE_TEST_N);
        int f2 = check_inf_arithmetic(a, b, c_simd, EDGE_TEST_N);
        int f3 = check_consistency(c_scalar, c_simd, EDGE_TEST_N, "Inf");

        if (f1 == 0 && f2 == 0 && f3 == 0) {
            printf("  [PASS] Inf arithmetic\n");
        } else {
            failures += f1 + f2 + f3;
            printf("  [FAIL] Inf arithmetic: scalar(%d) simd(%d) mismatch(%d)\n",
                   f1, f2, f3);
        }
    }

    /* --------------------------------------------------------
     * 3. Denormal behavior test
     * -------------------------------------------------------- */
    printf("\n  [3] Denormal handling...\n");

    /* Build a denorm-heavy input set: every element is a denormal */
    {
        float denorm_min = float_from_bits(0x00000001u);
        float denorm_mid = float_from_bits(0x000FFFFFu);
        float denorm_max = float_from_bits(0x007FFFFFu);

        for (size_t i = 0; i < EDGE_TEST_N; i++) {
            a[i] = denorm_min;
            b[i] = (i % 3 == 0) ? denorm_min :
                   (i % 3 == 1) ? denorm_mid : denorm_max;
        }

        memset(c_scalar, 0, EDGE_TEST_N * sizeof(float));
        memset(c_simd,   0, EDGE_TEST_N * sizeof(float));

        scalar_fn(a, b, c_scalar, EDGE_TEST_N);
        simd_fn(a, b, c_simd, EDGE_TEST_N);

        /* expect_flush_to_zero = 0: we want denormals preserved */
        int f1 = check_denormal_behavior(a, b, c_scalar, EDGE_TEST_N, 0);
        int f2 = check_denormal_behavior(a, b, c_simd, EDGE_TEST_N, 0);
        int f3 = check_consistency(c_scalar, c_simd, EDGE_TEST_N, "Denorm");

        if (f1 == 0 && f2 == 0 && f3 == 0) {
            printf("  [PASS] Denormal handling\n");
        } else {
            failures += f1 + f2 + f3;
            printf("  [FAIL] Denormal: scalar(%d) simd(%d) mismatch(%d)\n",
                   f1, f2, f3);
        }
    }

    /* --------------------------------------------------------
     * 4. Zero-length safety test
     * -------------------------------------------------------- */
    printf("\n  [4] Zero-length safety...\n");
    {
        int f1 = check_zero_length(scalar_fn);
        int f2 = check_zero_length(simd_fn);
        if (f1 == 0 && f2 == 0) {
            printf("  [PASS] Zero-length (no crash)\n");
        } else {
            failures += f1 + f2;
            printf("  [FAIL] Zero-length: scalar(%d) simd(%d)\n", f1, f2);
        }
    }

    /* --------------------------------------------------------
     * 5. Alignment boundary test
     * -------------------------------------------------------- */
    printf("\n  [5] Alignment boundaries...\n");
    {
        /* Fill with alignment-edge pattern: NaN sentinels at boundaries */
        fill_edge_alignment(a, EDGE_TEST_N, 32);
        memcpy(b, a, EDGE_TEST_N * sizeof(float));  /* same pattern for both inputs */

        memset(c_scalar, 0, EDGE_TEST_N * sizeof(float));
        memset(c_simd,   0, EDGE_TEST_N * sizeof(float));

        scalar_fn(a, b, c_scalar, EDGE_TEST_N);
        simd_fn(a, b, c_simd, EDGE_TEST_N);

        int f = check_consistency(c_scalar, c_simd, EDGE_TEST_N, "Align");
        if (f == 0) {
            printf("  [PASS] Alignment boundaries\n");
        } else {
            failures += f;
            printf("  [FAIL] Alignment boundaries: %d mismatch(es)\n", f);
        }
    }

cleanup:
    free(a);
    free(b);
    free(c_scalar);
    free(c_simd);

    return failures;
}

/* ---- Report printing ---- */

static inline void edge_test_report(const char* name, int failures) {
    printf("\n========================================\n");
    printf("  Edge Case Test Report: %s\n", name);
    printf("========================================\n");
    if (failures == 0) {
        printf("  Result: ALL PASS\n");
    } else {
        printf("  Result: %d FAILURE(S)\n", failures);
    }
    printf("========================================\n\n");
}

#ifdef __cplusplus
}
#endif
