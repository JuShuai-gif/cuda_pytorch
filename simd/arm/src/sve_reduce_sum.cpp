/**
 * sve_reduce_sum.cpp -- Horizontal Reduction using ARM SVE
 *
 * Computes sum of all elements in a float32 array using SVE's svaddv
 * (horizontal reduce-add across vector) instruction.
 *
 * Strategy:
 *   1. Whilelt loop: accumulate into SVE vector register
 *   2. svaddv_f32: horizontal add all lanes in one instruction
 *   3. No scalar tail loop needed (predicate handles edge)
 *
 * Also demonstrates the "len-based" pattern as an alternative to whilelt.
 * Compares against NEON reduce and scalar baseline to show speedup.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif

#include <arm_neon.h>

extern "C" {
#include "../../common/benchmark.h"
#include "../../common/cpu_features.h"
#include "../../common/check.h"
#include "../../common/random_data.h"
#include "../../common/aligned_buffer.h"
}

static const int N_REDUCE = 1000000;

// ============================================================================
// Scalar baseline
// ============================================================================

static float scalar_reduce_sum(const float* __restrict a, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += a[i];
    }
    return sum;
}

// ============================================================================
// NEON reduce sum
// ============================================================================
//
// Process 4 floats per iteration (NEON 128-bit), accumulate in float32x4_t,
// then use vaddvq_f32 for final horizontal reduction. Scalar tail for
// non-multiple-of-4 remainder.

static float neon_reduce_sum(const float* __restrict a, int n) {
    float32x4_t vacc0 = vdupq_n_f32(0.0f);
    float32x4_t vacc1 = vdupq_n_f32(0.0f);
    float32x4_t vacc2 = vdupq_n_f32(0.0f);
    float32x4_t vacc3 = vdupq_n_f32(0.0f);

    int i = 0;
    // 4 independent accumulators to hide vaddq latency (~3 cycles)
    for (; i + 15 < n; i += 16) {
        vacc0 = vaddq_f32(vacc0, vld1q_f32(&a[i]));
        vacc1 = vaddq_f32(vacc1, vld1q_f32(&a[i + 4]));
        vacc2 = vaddq_f32(vacc2, vld1q_f32(&a[i + 8]));
        vacc3 = vaddq_f32(vacc3, vld1q_f32(&a[i + 12]));
    }

    for (; i + 3 < n; i += 4) {
        vacc0 = vaddq_f32(vacc0, vld1q_f32(&a[i]));
    }

    // Combine accumulators, then horizontal reduce
    vacc0 = vaddq_f32(vaddq_f32(vacc0, vacc1), vaddq_f32(vacc2, vacc3));
    float sum = vaddvq_f32(vacc0);  // h-add all 4 lanes

    // Scalar tail for NEON
    for (; i < n; i++) sum += a[i];

    return sum;
}

// ============================================================================
// SVE reduce sum: whilelt pattern + svaddv horizontal reduction
// ============================================================================
//
// Key insight: SVE's svaddv_f32(pg, v) adds only the active lanes in v
// (those where pg is true). This means we don't need separate handling
// for the tail -- even if the vector is partially filled, svaddv only
// adds the valid elements.

#ifdef __ARM_FEATURE_SVE
static float sve_reduce_sum_whilelt(const float* __restrict a, int n) {
    svfloat32_t vsum = svdup_f32(0.0f);

    int i = 0;
    while (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);
        // Predicated addition: active lanes = vsum[lane] + a[i+lane]
        // Inactive lanes keep their previous value (merge with vsum)
        vsum = svadd_f32_m(pg, vsum, svld1(pg, &a[i]));

        i += svcntw();
    }

    // Horizontal reduce: sum all active lanes in vsum.
    // svptrue_b32() = all-true predicate for the current vector length.
    float result = svaddv_f32(svptrue_b32(), vsum);
    return result;
}

// ============================================================================
// SVE reduce sum: len-based pattern (alternative)
// ============================================================================
//
// Uses svlen-based loop which precomputes the number of iterations
// and unrolls with a fixed step. The predicate at each iteration
// handles partial vectors automatically.

static float sve_reduce_sum_len(const float* __restrict a, int n) {
    // Determine vector length in 32-bit elements
    uint64_t vl = svcntw();

    // Precompute number of full-vector iterations
    int i_lim = (int)((uint64_t)n / vl * vl);

    svfloat32_t vsum0 = svdup_f32(0.0f);
    svfloat32_t vsum1 = svdup_f32(0.0f);

    int i = 0;
    // Process full vectors (all lanes active)
    for (; i < i_lim; i += (int)(vl * 2)) {
        vsum0 = svadd_f32_m(svptrue_b32(), vsum0, svld1(svptrue_b32(), &a[i]));
        vsum1 = svadd_f32_m(svptrue_b32(), vsum1,
                            svld1(svptrue_b32(), &a[i + (int)vl]));
    }

    // Combine the two accumulators
    vsum0 = svadd_f32_m(svptrue_b32(), vsum0, vsum1);

    // Process tail with predicate (partially active vector)
    if (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);
        vsum0 = svadd_f32_m(pg, vsum0, svld1(pg, &a[i]));
    }

    float result = svaddv_f32(svptrue_b32(), vsum0);
    return result;
}
#endif // __ARM_FEATURE_SVE

// ============================================================================
// Benchmark wrappers
// ============================================================================

static float* g_rs_a = nullptr;
static int    g_rs_n = 0;
static volatile float g_rs_result = 0.0f;

__attribute__((noinline))
static void bench_scalar_reduce() {
    g_rs_result = scalar_reduce_sum(g_rs_a, g_rs_n);
}

__attribute__((noinline))
static void bench_neon_reduce() {
    g_rs_result = neon_reduce_sum(g_rs_a, g_rs_n);
}

#ifdef __ARM_FEATURE_SVE
__attribute__((noinline))
static void bench_sve_whilelt() {
    g_rs_result = sve_reduce_sum_whilelt(g_rs_a, g_rs_n);
}

__attribute__((noinline))
static void bench_sve_len() {
    g_rs_result = sve_reduce_sum_len(g_rs_a, g_rs_n);
}
#endif

// ============================================================================
// Main
// ============================================================================

int main() {
    cpu_print_features();

    int n = N_REDUCE;
    g_rs_n = n;
    g_rs_a = ALIGNED_ALLOC(float, n, 64);
    fill_random_f32(g_rs_a, n);

    // --- Correctness ---
    printf("\n=== Correctness Checks ===\n");
    float ref = scalar_reduce_sum(g_rs_a, n);
    printf("  Scalar reference sum: %.6f\n", (double)ref);

    float neon_val = neon_reduce_sum(g_rs_a, n);
    CHECK_NEAR(neon_val, ref, 1e-2f, "NEON reduce sum vs scalar");

#ifdef __ARM_FEATURE_SVE
    if (cpu_has_sve()) {
        float sve_wl = sve_reduce_sum_whilelt(g_rs_a, n);
        CHECK_NEAR(sve_wl, ref, 1e-2f, "SVE whilelt reduce sum vs scalar");

        float sve_len = sve_reduce_sum_len(g_rs_a, n);
        CHECK_NEAR(sve_len, ref, 1e-2f, "SVE len-based reduce sum vs scalar");

        // Print SVE vector info
        int sve_width = svcntw();
        printf("\n=== SVE Reduce Analysis ===\n");
        printf("  SVE vector width: %d x f32 lanes\n", sve_width);
        printf("  NEON: %d iterations with 4-lane accumulators\n",
               (n + 15) / 16 * 4); // 4 accumulators, 16 elems/iter
        printf("  SVE whilelt: %d iterations, no tail loop\n",
               (n + sve_width - 1) / sve_width);
        printf("  SVE svaddv: horizontal reduction in 1 instruction\n");
        printf("  NEON vaddvq: horizontal reduction in 1 instruction (but on fixed 4 lanes)\n");
    }
#endif

    // --- Benchmark ---
    printf("\n=== Benchmark: Reduce Sum (N=%d) ===\n", n);

#ifdef __ARM_FEATURE_SVE
    int num_results = cpu_has_sve() ? 4 : 2;
    benchmark_result_t results[4];
    int ri = 0;

    BENCH_COMPUTE(bench_scalar_reduce(), n, (size_t)n * sizeof(float), 30,
                  results[ri]);
    results[ri++].name = "scalar reduce sum";

    BENCH_COMPUTE(bench_neon_reduce(), n, (size_t)n * sizeof(float), 30,
                  results[ri]);
    results[ri++].name = "NEON reduce (4 acc)";

    if (cpu_has_sve()) {
        BENCH_COMPUTE(bench_sve_whilelt(), n, (size_t)n * sizeof(float), 30,
                      results[ri]);
        results[ri++].name = "SVE whilelt reduce";

        BENCH_COMPUTE(bench_sve_len(), n, (size_t)n * sizeof(float), 30,
                      results[ri]);
        results[ri++].name = "SVE len-based reduce";
    }

    bench_report(results, (size_t)ri);

    printf("=== Analysis ===\n");
    printf("  SVE advantage in reductions:\n");
    printf("  1. Wider vectors = fewer loop iterations = less loop overhead\n");
    printf("  2. svaddv reduces all SVE-width lanes in one instruction\n");
    printf("  3. Predicated loop = no scalar tail overhead\n");
    printf("  4. For N=%d (memory-bound), the key benefit is fewer loads\n", n);
#else
    benchmark_result_t results[2];
    BENCH_COMPUTE(bench_scalar_reduce(), n, (size_t)n * sizeof(float), 30,
                  results[0]);
    results[0].name = "scalar reduce sum";
    BENCH_COMPUTE(bench_neon_reduce(), n, (size_t)n * sizeof(float), 30,
                  results[1]);
    results[1].name = "NEON reduce (4 acc)";
    bench_report(results, 2);
    printf("  SVE code not compiled in (use -march=armv8-a+sve)\n");
#endif

    printf("\n  Checksum: %.6f\n", (double)g_rs_result);

    ALIGNED_FREE(g_rs_a);
    return 0;
}
