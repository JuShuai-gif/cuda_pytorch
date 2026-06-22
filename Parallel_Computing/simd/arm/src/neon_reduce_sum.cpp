#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <arm_neon.h>
#include "../../common/benchmark.h"
#include "../../common/check.h"
#include "../../common/random_data.h"
#include "../../common/aligned_buffer.h"

// =============================================================================
// neon_reduce_sum -- Array sum with NEON, demonstrating ILP via unrolling
//   sum = sum(data[i])
//   SIMD width: 4x f32 per 128-bit NEON register
//   N = 1000000
//
//   Variant 1: simple NEON (1 accumulator vector)
//   Variant 2: NEON with 4-way unrolling (4 accumulator vectors for ILP)
//   Baseline:  scalar loop
//
//   On out-of-order ARM cores (e.g. Cortex-A76, X1), unrolling with multiple
//   accumulator registers hides floating-point add latency by overlapping
//   independent chains of vaddq_f32.
// =============================================================================

static const size_t N = 1000000;
static const int    BENCH_ITERS = 10;

// ---- scalar sum ----
static float scalar_sum_f32(const float* data, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum;
}

// ---- NEON sum: 1 accumulator (simple) ----
// single float32x4_t accumulator, vertical add per iteration
static float neon_sum_simple_f32(const float* data, size_t n) {
    float32x4_t vacc = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        vacc = vaddq_f32(vacc, v);
    }
    float sum = vaddvq_f32(vacc);
    for (; i < n; i++) {
        sum += data[i];
    }
    return sum;
}

// ---- NEON sum: 4-way unrolled (4 accumulator vectors for ILP) ----
// 4 independent accumulator chains hide FADD latency on OoO cores.
// Each chain depends only on itself, so the CPU can interleave them.
static float neon_sum_unrolled4_f32(const float* data, size_t n) {
    float32x4_t vacc0 = vdupq_n_f32(0.0f);
    float32x4_t vacc1 = vdupq_n_f32(0.0f);
    float32x4_t vacc2 = vdupq_n_f32(0.0f);
    float32x4_t vacc3 = vdupq_n_f32(0.0f);

    size_t i = 0;
    // Main loop: process 16 elements (4 vectors) per iteration
    for (; i + 16 <= n; i += 16) {
        float32x4_t v0 = vld1q_f32(data + i + 0);
        float32x4_t v1 = vld1q_f32(data + i + 4);
        float32x4_t v2 = vld1q_f32(data + i + 8);
        float32x4_t v3 = vld1q_f32(data + i + 12);

        vacc0 = vaddq_f32(vacc0, v0);
        vacc1 = vaddq_f32(vacc1, v1);
        vacc2 = vaddq_f32(vacc2, v2);
        vacc3 = vaddq_f32(vacc3, v3);
    }
    // Partial loop: single vectors for remaining multiples of 4
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        vacc0 = vaddq_f32(vacc0, v);
    }

    // Combine the 4 accumulator vectors
    float32x4_t vsum01 = vaddq_f32(vacc0, vacc1);
    float32x4_t vsum23 = vaddq_f32(vacc2, vacc3);
    float32x4_t vsum   = vaddq_f32(vsum01, vsum23);
    float sum = vaddvq_f32(vsum);

    // Scalar tail
    for (; i < n; i++) {
        sum += data[i];
    }
    return sum;
}

// =============================================================================
// main
// =============================================================================
int main(void) {
    printf("================================================================\n");
    printf("  NEON Array Reduction (Sum) -- ILP via Unrolling\n");
    printf("  SIMD width: 4x f32 per 128-bit NEON register\n");
    printf("  N = %zu\n", N);
    printf("================================================================\n");

    float* data = ALIGNED_ALLOC(float, N, 16);
    CHECK_TRUE(is_aligned(data, 16), "data buffer is 16-byte aligned");
    fill_random_f32(data, N);

    // ---- Correctness ----
    printf("\n-- Correctness --\n");

    float scalar_result = scalar_sum_f32(data, N);
    float neon_simple   = neon_sum_simple_f32(data, N);
    float neon_unrolled = neon_sum_unrolled4_f32(data, N);

    printf("  scalar       = %.6f\n", (double)scalar_result);
    printf("  neon_simple  = %.6f  (1 accumulator)\n", (double)neon_simple);
    printf("  neon_unroll4 = %.6f  (4 accumulators, ILP)\n",
           (double)neon_unrolled);

    // Use a tolerance proportional to N since FP rounding accumulates
    float tol = N * 1e-6f;
    CHECK_NEAR(scalar_result, neon_simple,   (double)tol,
               "neon_simple matches scalar");
    CHECK_NEAR(scalar_result, neon_unrolled, (double)tol,
               "neon_unrolled matches scalar");

    // ---- Benchmarks ----
    printf("\n-- Benchmarks (%d timed iterations) --\n", BENCH_ITERS);

    volatile float sink = 0.0f;
    size_t bytes = N * sizeof(float); // read only

    benchmark_result_t results[3];

    BENCH_COMPUTE(sink = scalar_sum_f32(data, N), N, bytes, BENCH_ITERS, results[0]);
    results[0].name = "scalar_sum_f32";

    BENCH_COMPUTE(sink = neon_sum_simple_f32(data, N), N, bytes, BENCH_ITERS, results[1]);
    results[1].name = "neon_sum_simple (1-acc)";

    BENCH_COMPUTE(sink = neon_sum_unrolled4_f32(data, N), N, bytes, BENCH_ITERS, results[2]);
    results[2].name = "neon_sum_unroll4 (4-acc)";

    bench_report(results, 3);

    // ---- Analysis ----
    printf("Analysis:\n");
    printf("  simple speedup:  %.2fx vs scalar\n",
           results[0].elapsed_ns / results[1].elapsed_ns);
    printf("  unrolled speedup: %.2fx vs scalar\n",
           results[0].elapsed_ns / results[2].elapsed_ns);
    printf("  unroll vs simple: %.2fx -- extra ILP from 4 accumulator chains\n",
           results[1].elapsed_ns / results[2].elapsed_ns);
    printf("  On in-order cores (A53/A55), expect ~1x unroll benefit;\n");
    printf("  on OoO cores (A76/X1), expect ~1.3-2.0x benefit from ILP.\n");

    (void)sink;

    ALIGNED_FREE(data);

    printf("\nAll tests passed.\n");
    return 0;
}
