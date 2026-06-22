/*
 * avx2_reduce_sum.cpp -- AVX2 array reduction: sum(data[i])
 *
 * SIMD width: 256-bit = 8x f32 per register
 * N = 1000000
 *
 * Reduction is a classic "horizontal" operation where the bottleneck
 * is the dependency chain through a single accumulator:
 *   acc = acc + (next 8 floats)
 *
 * Unrolling with multiple accumulators breaks this dependency:
 *   acc0 += group 0, acc1 += group 1, ...
 *   final = acc0 + acc1 + ... + (merge + hadd)
 *
 * Variants: 1x (no unroll), 2x, 4x unrolling
 *
 * Final horizontal reduction:
 *   1. _mm256_permute2f128_ps(v, v, 0x01)  // swap lo/hi 128 lanes
 *   2. _mm256_add_ps(v, swapped)            // pairwise add across lanes
 *   3. _mm256_hadd_ps(x2)                   // reduce to scalar
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
 * Horizontal reduction helper
 *
 * Reduce __m256 to a single float.
 * Strategy: permute lo/hi halves -> add -> 2x hadd -> extract.
 * This distributes port pressure across port 5 (permute) and
 * execution ports 0/1 (add) instead of saturating port 5 with
 * a chain of hadd instructions.
 * ================================================================ */

static inline float reduce256_ps(__m256 v) {
    /* v = [a0, a1, a2, a3,  b0, b1, b2, b3] */
    /* swap lo/hi 128-bit lanes */
    __m256 swapped = _mm256_permute2f128_ps(v, v, 0x01);
    /* swapped = [b0, b1, b2, b3,  a0, a1, a2, a3] */
    v = _mm256_add_ps(v, swapped);
    /* v = [a0+b0, a1+b1, a2+b2, a3+b3, same again] */
    v = _mm256_hadd_ps(v, v);
    /* hadd within lanes: [(a0+b0)+(a1+b1), (a2+b2)+(a3+b3), ...] */
    v = _mm256_hadd_ps(v, v);
    /* final: all 4 pairs summed */
    return _mm256_cvtss_f32(v);
}

/* ================================================================
 * Scalar baseline
 * ================================================================ */

static float scalar_sum(const float* data, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) sum += data[i];
    return sum;
}

/* ================================================================
 * AVX2 reduce sum -- 1x (no unrolling)
 *
 * Single __m256 accumulator. The dependency chain is:
 *   acc = acc + load
 * Each FMA/add has ~4 cycle latency, so we're limited by this
 * serial dependency. Unrolling helps by having multiple chains.
 * ================================================================ */

static float avx2_sum_1x(const float* data, size_t n) {
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(data + i);
        acc = _mm256_add_ps(acc, v);
    }
    float sum = reduce256_ps(acc);
    for (; i < n; i++) sum += data[i];
    return sum;
}

/* ================================================================
 * AVX2 reduce sum -- 2x unrolling
 *
 * Two independent accumulators halve the dependency chain depth.
 * Each accumulator handles every-other block of 8 floats.
 * ================================================================ */

static float avx2_sum_2x(const float* data, size_t n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    size_t i = 0;

    for (; i + 16 <= n; i += 16) {
        __m256 v0 = _mm256_loadu_ps(data + i);
        __m256 v1 = _mm256_loadu_ps(data + i + 8);
        acc0 = _mm256_add_ps(acc0, v0);
        acc1 = _mm256_add_ps(acc1, v1);
    }

    /* Merge remaining */
    __m256 acc = _mm256_add_ps(acc0, acc1);
    for (; i + 8 <= n; i += 8) {
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(data + i));
    }

    float sum = reduce256_ps(acc);
    for (; i < n; i++) sum += data[i];
    return sum;
}

/* ================================================================
 * AVX2 reduce sum -- 4x unrolling
 *
 * Four independent accumulators = 4x reduction in dependency chain.
 * On Skylake+ with 2 FMA/ADD units (ports 0,1), 4 accumulators
 * can keep both pipes fully utilized (2 inflight chains per pipe).
 * ================================================================ */

static float avx2_sum_4x(const float* data, size_t n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    size_t i = 0;

    for (; i + 32 <= n; i += 32) {
        __m256 v0 = _mm256_loadu_ps(data + i);
        acc0 = _mm256_add_ps(acc0, v0);

        __m256 v1 = _mm256_loadu_ps(data + i + 8);
        acc1 = _mm256_add_ps(acc1, v1);

        __m256 v2 = _mm256_loadu_ps(data + i + 16);
        acc2 = _mm256_add_ps(acc2, v2);

        __m256 v3 = _mm256_loadu_ps(data + i + 24);
        acc3 = _mm256_add_ps(acc3, v3);
    }

    /* Merge all accumulators */
    __m256 acc01 = _mm256_add_ps(acc0, acc1);
    __m256 acc23 = _mm256_add_ps(acc2, acc3);
    __m256 acc   = _mm256_add_ps(acc01, acc23);

    /* Tail full vectors */
    for (; i + 8 <= n; i += 8) {
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(data + i));
    }

    float sum = reduce256_ps(acc);
    for (; i < n; i++) sum += data[i];
    return sum;
}

/* ================================================================
 * main
 * ================================================================ */

int main() {
    cpu_print_features();

    if (!cpu_has_avx2()) {
        printf("AVX2 not supported on this CPU. Exiting.\n");
        return 1;
    }

    printf("\n=== AVX2 Reduce Sum (N = %zu) ===\n\n", N);

    /* Allocate */
    float* data = ALIGNED_ALLOC(float, N, 32);
    if (!data) {
        fprintf(stderr, "Allocation failed.\n");
        return 1;
    }

    rand_xorshift64_seed(42);
    fill_random_f32(data, N);

    /* ---- Correctness ---- */

    printf("--- Correctness ---\n");

    float ref  = scalar_sum(data, N);
    float s1x  = avx2_sum_1x(data, N);
    float s2x  = avx2_sum_2x(data, N);
    float s4x  = avx2_sum_4x(data, N);

    /* Reduction order affects precision; use tolerance scaled by N */
    const float tol = 1e-4f * (float)N;

    CHECK_NEAR(s1x, ref, tol, "avx2_sum_1x matches scalar");
    CHECK_NEAR(s2x, ref, tol, "avx2_sum_2x matches scalar");
    CHECK_NEAR(s4x, ref, tol, "avx2_sum_4x matches scalar");

    printf("  scalar    = %.6f\n", (double)ref);
    printf("  1x unroll = %.6f\n", (double)s1x);
    printf("  2x unroll = %.6f\n", (double)s2x);
    printf("  4x unroll = %.6f\n", (double)s4x);

    /* ---- Benchmark ---- */

    const size_t bytes_rw = N * sizeof(float); /* read data */

    benchmark_result_t results[4];
    memset(results, 0, sizeof(results));

    BENCH_COMPUTE(volatile float _r0 = scalar_sum(data, N); (void)_r0;,
                  N, bytes_rw, 20, results[0]);
    results[0].name = "scalar_sum";

    BENCH_COMPUTE(volatile float _r1 = avx2_sum_1x(data, N); (void)_r1;,
                  N, bytes_rw, 20, results[1]);
    results[1].name = "avx2_sum_1x (no unroll)";

    BENCH_COMPUTE(volatile float _r2 = avx2_sum_2x(data, N); (void)_r2;,
                  N, bytes_rw, 20, results[2]);
    results[2].name = "avx2_sum_2x";

    BENCH_COMPUTE(volatile float _r3 = avx2_sum_4x(data, N); (void)_r3;,
                  N, bytes_rw, 20, results[3]);
    results[3].name = "avx2_sum_4x";

    printf("\n--- Benchmark Results ---\n");
    printf("SIMD width: 256-bit (8x f32)\n");
    bench_report(results, 4);

    printf("Notes:\n");
    printf("  - Reduction is dominated by the dependency chain through the\n");
    printf("    accumulator. With 1 accumulator, the CPU must wait for each\n");
    printf("    _mm256_add_ps to complete before starting the next (4+ cycle\n");
    printf("    latency on most x86 cores).\n");
    printf("  - 2x unrolling: halves the chain depth; good if the CPU can\n");
    printf("    schedule across 2 accumulator chains.\n");
    printf("  - 4x unrolling: 4 independent chains. On Skylake+ (2 add units),\n");
    printf("    this allows 2 adds per cycle in steady state, fully utilizing\n");
    printf("    both execution ports.\n");
    printf("  - Beyond 4x: diminishing returns due to register pressure (16 regs\n");
    printf("    total on x86-64) and front-end decode bandwidth.\n");
    printf("  - Final reduction uses permute+add+2x hadd for balanced port usage.\n");

    ALIGNED_FREE(data);

    return 0;
}
