/*
 * avx2_dot_product.cpp -- AVX2 dot product: sum(a[i] * b[i])
 *
 * SIMD width: 256-bit = 8x f32 per register
 * N = 1000000
 *
 * Variants:
 *   1. FMA-based:   _mm256_fmadd_ps(a, b, acc) -- 1 instruction for mul+acc
 *   2. Mul+Add:     _mm256_mul_ps  + _mm256_add_ps -- 2 instructions
 *   3. 4-way unrolled FMA: 4 independent accumulators for ILP
 *
 * Horizontal reduction:
 *   - Fast path (permute+add): faster than hadd
 *   - _mm256_hadd_ps: 2-cycle latency but limited to port 5 on Skylake
 *
 * The permute+add approach uses:
 *   hi128 = _mm256_permute2f128_ps(v, v, 0x01)
 *   sum   = _mm256_add_ps(v, hi128)
 *   sum   = _mm256_hadd_ps(sum, sum)    (still need 2 hadd for final 128->scalar)
 *   sum   = _mm256_hadd_ps(sum, sum)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include "../../common/benchmark.h"
#include "../../common/check.h"
#include "../../common/random_data.h"
#include "../../common/aligned_buffer.h"
#include "../../common/cpu_features.h"

static const size_t N = 1000000;

/* ================================================================
 * Horizontal reduction helpers
 * ================================================================ */

/* Reduce __m256 to a single float.
 * Uses permute+add to combine lo/hi 128 lanes, then 2x hadd. */
static inline float hsum_ps(__m256 v) {
    /* v = [A0, A1, A2, A3, B0, B1, B2, B3] */
    __m256 v_perm = _mm256_permute2f128_ps(v, v, 0x01);
    /* v_perm = [B0, B1, B2, B3, A0, A1, A2, A3] */
    __m256 sum = _mm256_add_ps(v, v_perm);
    /* sum = [A0+B0, A1+B1, A2+B2, A3+B3, B0+A0, B1+A1, B2+A2, B3+A3] */
    /* Now 2 hadd steps on the 128-bit lanes */
    sum = _mm256_hadd_ps(sum, sum);
    /* after 1st hadd: lo = [s0+s1, s2+s3, s0+s1, s2+s3] */
    sum = _mm256_hadd_ps(sum, sum);
    /* after 2nd hadd: lo = [s0+s1+s2+s3, ...] */
    return _mm256_cvtss_f32(sum);
}

/*
 * Alternative reduction using only SSE-level operations on the low 128 bits
 * after combining the two 128-bit halves.
 *
 * KEY PITFALL: _mm256_hadd_ps only operates within 128-bit lanes.
 *   v = [a0,a1,a2,a3, b0,b1,b2,b3]
 *   hadd(v,v) = [a0+a1,a2+a3,a0+a1,a2+a3, b0+b1,b2+b3,b0+b1,b2+b3]
 *   hadd again  = [a0+..+a3, a0+..+a3, a0+..+a3, a0+..+a3, ...]
 *   _mm256_cvtss_f32 gets only the LOW lane: a0+..+a3 (half sum!)
 * So you MUST combine lo/hi 128-bit halves first.
 */
static inline float hsum_ps_hadd_only(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

/* ================================================================
 * Scalar baseline
 * ================================================================ */

static float scalar_dot(const float* a, const float* b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

/* ================================================================
 * AVX2 FMA-based dot product (single accumulator)
 * ================================================================ */

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Waggressive-loop-optimizations"
static float avx2_dot_fma(const float* a, const float* b, size_t n) {
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    float sum = hsum_ps(acc);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}
#pragma GCC diagnostic pop

/* ================================================================
 * AVX2 Mul+Add (less efficient, kept for comparison)
 * ================================================================ */

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Waggressive-loop-optimizations"
static float avx2_dot_mul_add(const float* a, const float* b, size_t n) {
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 prod = _mm256_mul_ps(va, vb);
        acc = _mm256_add_ps(acc, prod);
    }
    float sum = hsum_ps(acc);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}
#pragma GCC diagnostic pop

/* ================================================================
 * AVX2 FMA 4-way unrolled (4 independent accumulators for ILP)
 *
 * Using 4 accumulators reduces the dependency chain depth by 4x,
 * allowing the CPU to better exploit instruction-level parallelism.
 * Modern x86 cores have 2 FMA units (ports 0,1 on Skylake+), so
 * 2 FMAs can issue per cycle. 4 accumulators keep both pipes fed.
 * ================================================================ */

static float avx2_dot_fma_unroll4(const float* a, const float* b, size_t n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    size_t i = 0;
    /* 4-way unrolled loop: process 4*8 = 32 elements per iteration */
    for (; i + 32 <= n; i += 32) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        acc0 = _mm256_fmadd_ps(va0, vb0, acc0);

        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        acc1 = _mm256_fmadd_ps(va1, vb1, acc1);

        __m256 va2 = _mm256_loadu_ps(a + i + 16);
        __m256 vb2 = _mm256_loadu_ps(b + i + 16);
        acc2 = _mm256_fmadd_ps(va2, vb2, acc2);

        __m256 va3 = _mm256_loadu_ps(a + i + 24);
        __m256 vb3 = _mm256_loadu_ps(b + i + 24);
        acc3 = _mm256_fmadd_ps(va3, vb3, acc3);
    }

    /* Process remaining full vectors with a single accumulator */
    __m256 acc = acc0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }

    /* Merge all 4 accumulators */
    acc = _mm256_add_ps(acc, acc1);
    acc = _mm256_add_ps(acc, acc2);
    acc = _mm256_add_ps(acc, acc3);

    float sum = hsum_ps(acc);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

/* ================================================================
 * Show both reduction methods
 * ================================================================ */

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Waggressive-loop-optimizations"
static float avx2_dot_hadd_only(const float* a, const float* b, size_t n) {
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    float sum = hsum_ps_hadd_only(acc);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}
#pragma GCC diagnostic pop

/* ================================================================
 * main
 * ================================================================ */

int main() {
    cpu_print_features();

    if (!cpu_has_avx2()) {
        printf("AVX2 not supported on this CPU. Exiting.\n");
        return 1;
    }

    printf("\n=== AVX2 Dot Product (N = %zu) ===\n\n", N);

    /* Allocate and fill */
    float* a = ALIGNED_ALLOC(float, N, 32);
    float* b = ALIGNED_ALLOC(float, N, 32);

    if (!a || !b) {
        fprintf(stderr, "Allocation failed.\n");
        return 1;
    }

    rand_xorshift64_seed(42);
    fill_random_f32(a, N);
    rand_xorshift64_seed(99);
    fill_random_f32(b, N);

    /* ---- Correctness ---- */

    printf("--- Correctness ---\n");

    float ref = scalar_dot(a, b, N);
    float fma_val = avx2_dot_fma(a, b, N);
    float mul_add_val = avx2_dot_mul_add(a, b, N);
    float unroll4_val = avx2_dot_fma_unroll4(a, b, N);

    /* Dot product loses precision due to summation order; use 5e-5 tolerance */
    const float tol = 5e-5f * (float)N;
    CHECK_NEAR(fma_val, ref, tol, "avx2_dot_fma matches scalar");
    CHECK_NEAR(mul_add_val, ref, tol, "avx2_dot_mul_add matches scalar");
    CHECK_NEAR(unroll4_val, ref, tol, "avx2_dot_unroll4 matches scalar");

    /* Also verify hadd-only gives same result */
    float hadd_val = avx2_dot_hadd_only(a, b, N);
    CHECK_NEAR(hadd_val, ref, tol, "avx2_dot_hadd_only matches scalar");

    printf("  scalar      = %.6f\n", (double)ref);
    printf("  FMA         = %.6f\n", (double)fma_val);
    printf("  mul+add     = %.6f\n", (double)mul_add_val);
    printf("  FMA unroll4 = %.6f\n", (double)unroll4_val);

    /* ---- Benchmark ---- */

    /* Read a + b (2 * N * sizeof(float)), write 1 float */
    const size_t bytes_rw = N * 2 * sizeof(float);

    benchmark_result_t results[4];
    memset(results, 0, sizeof(results));

    BENCH_COMPUTE(volatile float _r0 = scalar_dot(a, b, N); (void)_r0;,
                  N, bytes_rw, 20, results[0]);
    results[0].name = "scalar_dot";

    BENCH_COMPUTE(volatile float _r1 = avx2_dot_mul_add(a, b, N); (void)_r1;,
                  N, bytes_rw, 20, results[1]);
    results[1].name = "avx2_dot_mul_add";

    BENCH_COMPUTE(volatile float _r2 = avx2_dot_fma(a, b, N); (void)_r2;,
                  N, bytes_rw, 20, results[2]);
    results[2].name = "avx2_dot_fma";

    BENCH_COMPUTE(volatile float _r3 = avx2_dot_fma_unroll4(a, b, N); (void)_r3;,
                  N, bytes_rw, 20, results[3]);
    results[3].name = "avx2_dot_fma_unroll4";

    printf("\n--- Benchmark Results ---\n");
    printf("SIMD width: 256-bit (8x f32)\n");
    bench_report(results, 4);

    printf("Notes:\n");
    printf("  - FMA (_mm256_fmadd_ps) fuses multiply+add into 1 instruction,\n");
    printf("    reducing both latency (4 cycles vs 4+3=7 for mul+add) and\n");
    printf("    uop count (1 vs 2).\n");
    printf("  - 4-way unrolling uses 4 independent accumulators to expose ILP.\n");
    printf("    On Skylake/X with 2 FMA units, this can double throughput by\n");
    printf("    overlapping the latency of 4 independent FMA chains.\n");
    printf("  - Horizontal reduction (hsum_ps):\n");
    printf("    * _mm256_hadd_ps latency: 2 cycles, but only on port 5 (Skylake).\n");
    printf("    * Better: permute+add (port 5 + port 0/1) + 2x hadd.\n");
    printf("    * The permute+add approach avoids port 5 saturation when the\n");
    printf("      reduction is in a tight loop.\n");

    ALIGNED_FREE(a);
    ALIGNED_FREE(b);

    return 0;
}
