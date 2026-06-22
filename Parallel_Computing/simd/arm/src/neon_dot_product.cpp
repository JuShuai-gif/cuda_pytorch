#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <arm_neon.h>
#include "../../common/benchmark.h"
#include "../../common/check.h"
#include "../../common/random_data.h"
#include "../../common/aligned_buffer.h"

// =============================================================================
// neon_dot_product -- Dot product with NEON, two reduction strategies
//   dot = sum(a[i] * b[i])
//   SIMD width: 4x f32 per 128-bit NEON register
//   N = 1000000
//
//   Variant 1 (naive): vmulq_f32 -> vaddvq_f32 each iteration (horizontal add
//                      on every step -- high latency)
//   Variant 2 (fast):  vmulq_f32 -> accumulate in 4 float32x4_t regs
//                      -> final reduction with pairwise adds + vaddvq
// =============================================================================

static const size_t N = 1000000;
static const int    BENCH_ITERS = 10;

// ---- scalar dot product ----
static float scalar_dot_f32(const float* a, const float* b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// ---- NEON dot product variant 1: vaddvq_f32 every iteration ----
// Slower: each vaddvq_f32 is a horizontal reduction with ~4-7 cycles
// latency on most ARM cores (Cortex-A53/A72/A76/X1 etc).
static float neon_dot_naive_f32(const float* a, const float* b, size_t n) {
    float sum = 0.0f;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va   = vld1q_f32(a + i);
        float32x4_t vb   = vld1q_f32(b + i);
        float32x4_t vmul = vmulq_f32(va, vb);
        sum += vaddvq_f32(vmul); // horizontal reduction per 4 elements
    }
    for (; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// ---- NEON dot product variant 2: 4 accumulator registers + final reduction ----
// Faster: accumulates partial sums in 4 NEON registers, reduces at the end.
// Uses 4-way unrolling for instruction-level parallelism (ILP).
static float neon_dot_fast_f32(const float* a, const float* b, size_t n) {
    // 4 accumulator vectors for ILP
    float32x4_t vacc0 = vdupq_n_f32(0.0f);
    float32x4_t vacc1 = vdupq_n_f32(0.0f);
    float32x4_t vacc2 = vdupq_n_f32(0.0f);
    float32x4_t vacc3 = vdupq_n_f32(0.0f);

    size_t i = 0;
    // Main loop: 4x unrolled (16 elements per iteration)
    for (; i + 16 <= n; i += 16) {
        float32x4_t va0 = vld1q_f32(a + i + 0);
        float32x4_t vb0 = vld1q_f32(b + i + 0);
        float32x4_t va1 = vld1q_f32(a + i + 4);
        float32x4_t vb1 = vld1q_f32(b + i + 4);
        float32x4_t va2 = vld1q_f32(a + i + 8);
        float32x4_t vb2 = vld1q_f32(b + i + 8);
        float32x4_t va3 = vld1q_f32(a + i + 12);
        float32x4_t vb3 = vld1q_f32(b + i + 12);

        vacc0 = vmlaq_f32(vacc0, va0, vb0); // vacc += va * vb
        vacc1 = vmlaq_f32(vacc1, va1, vb1);
        vacc2 = vmlaq_f32(vacc2, va2, vb2);
        vacc3 = vmlaq_f32(vacc3, va3, vb3);
    }
    // Partial loop: 4 elements per iteration
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vacc0 = vmlaq_f32(vacc0, va, vb);
    }

    // Horizontal reduction of the 4 accumulators
    float32x4_t vsum01 = vaddq_f32(vacc0, vacc1);
    float32x4_t vsum23 = vaddq_f32(vacc2, vacc3);
    float32x4_t vsum   = vaddq_f32(vsum01, vsum23);
    float sum = vaddvq_f32(vsum);

    // Scalar tail
    for (; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// =============================================================================
// main
// =============================================================================
int main(void) {
    printf("================================================================\n");
    printf("  NEON Dot Product -- Two Reduction Strategies\n");
    printf("  SIMD width: 4x f32 per 128-bit NEON register\n");
    printf("  N = %zu\n", N);
    printf("================================================================\n");

    float* a = ALIGNED_ALLOC(float, N, 16);
    float* b = ALIGNED_ALLOC(float, N, 16);
    CHECK_TRUE(is_aligned(a, 16), "buffer a is 16-byte aligned");

    fill_random_f32(a, N);
    fill_random_f32(b, N);

    // ---- Correctness ----
    printf("\n-- Correctness --\n");

    float scalar_result = scalar_dot_f32(a, b, N);
    float neon_naive    = neon_dot_naive_f32(a, b, N);
    float neon_fast     = neon_dot_fast_f32(a, b, N);

    printf("  scalar   = %.6f\n", (double)scalar_result);
    printf("  neon_v1   = %.6f  (naive: vaddvq every iteration)\n",
           (double)neon_naive);
    printf("  neon_v2   = %.6f  (fast: 4-way accum + final reduction)\n",
           (double)neon_fast);

    CHECK_NEAR(scalar_result, neon_naive, 1e-3, "neon_naive matches scalar");
    CHECK_NEAR(scalar_result, neon_fast,  1e-3, "neon_fast matches scalar");

    // ---- Benchmarks ----
    printf("\n-- Benchmarks (%d timed iterations) --\n", BENCH_ITERS);

    // We wrap dot product calls in a tiny inline kernel that writes result
    // to a volatile to prevent the compiler from optimizing away the call.
    // For BENCH_COMPUTE we use a lambda-like macro pattern.
    volatile float sink = 0.0f;

    size_t bytes = N * 2 * sizeof(float); // read a + read b

    benchmark_result_t results[3];

    BENCH_COMPUTE(sink = scalar_dot_f32(a, b, N), N, bytes, BENCH_ITERS, results[0]);
    results[0].name = "scalar_dot_f32";

    BENCH_COMPUTE(sink = neon_dot_naive_f32(a, b, N), N, bytes, BENCH_ITERS, results[1]);
    results[1].name = "neon_dot_naive (vaddvq/iter)";

    BENCH_COMPUTE(sink = neon_dot_fast_f32(a, b, N), N, bytes, BENCH_ITERS, results[2]);
    results[2].name = "neon_dot_fast (4-acc+unroll)";

    bench_report(results, 3);

    // ---- Analysis ----
    printf("Analysis:\n");
    printf("  naive speedup: %.2fx  -- vaddvq_f32 per iteration adds latency\n",
           results[0].elapsed_ns / results[1].elapsed_ns);
    printf("  fast speedup:  %.2fx  -- 4 accumulators hide FMUL+FMA latency\n",
           results[0].elapsed_ns / results[2].elapsed_ns);
    printf("  The 4-way unrolled variant typically 2-3x faster than naive\n");

    (void)sink;

    ALIGNED_FREE(a);
    ALIGNED_FREE(b);

    printf("\nAll tests passed.\n");
    return 0;
}
