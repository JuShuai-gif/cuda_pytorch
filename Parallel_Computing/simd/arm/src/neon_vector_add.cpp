#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <arm_neon.h>
#include "../../common/benchmark.h"
#include "../../common/check.h"
#include "../../common/random_data.h"
#include "../../common/aligned_buffer.h"

// =============================================================================
// neon_vector_add -- Element-wise addition with NEON (f32 and i32)
//   c[i] = a[i] + b[i]
//   SIMD width: 4x f32 per 128-bit NEON register, 4x i32 per NEON register
//   N = 1000003 (not multiple of 4, ensures tail handling is tested)
//   Alignment: 16-byte aligned buffers via ALIGNED_ALLOC
// =============================================================================

static const size_t N = 1000003;
static const int    BENCH_ITERS = 10;

// ---- f32 scalar baseline ----
static void scalar_add_f32(const float* a, const float* b, float* c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// ---- f32 NEON SIMD (4x f32 per 128-bit register) ----
static void neon_add_f32(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0;
    // Main loop: process 4x f32 per iteration
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vc = vaddq_f32(va, vb);
        vst1q_f32(c + i, vc);
    }
    // Tail: scalar fallback for remaining elements (0..3)
    for (; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// ---- i32 scalar baseline ----
static void scalar_add_i32(const int32_t* a, const int32_t* b, int32_t* c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// ---- i32 NEON SIMD (4x i32 per 128-bit register) ----
static void neon_add_i32(const int32_t* a, const int32_t* b, int32_t* c, size_t n) {
    size_t i = 0;
    // Main loop: process 4x i32 per iteration
    for (; i + 4 <= n; i += 4) {
        int32x4_t va = vld1q_s32(a + i);
        int32x4_t vb = vld1q_s32(b + i);
        int32x4_t vc = vaddq_s32(va, vb);
        vst1q_s32(c + i, vc);
    }
    // Tail: scalar fallback
    for (; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// =============================================================================
// main
// =============================================================================
int main(void) {
    printf("================================================================\n");
    printf("  NEON Vector Addition -- f32 & i32\n");
    printf("  SIMD width: 4x per 128-bit NEON register\n");
    printf("  N = %zu (not multiple of 4)\n", N);
    printf("================================================================\n");

    // Allocate 16-byte aligned buffers
    float*   a_f32  = ALIGNED_ALLOC(float, N, 16);
    float*   b_f32  = ALIGNED_ALLOC(float, N, 16);
    float*   ref_f32 = ALIGNED_ALLOC(float, N, 16);
    float*   neon_f32 = ALIGNED_ALLOC(float, N, 16);

    int32_t* a_i32  = ALIGNED_ALLOC(int32_t, N, 16);
    int32_t* b_i32  = ALIGNED_ALLOC(int32_t, N, 16);
    int32_t* ref_i32 = ALIGNED_ALLOC(int32_t, N, 16);
    int32_t* neon_i32 = ALIGNED_ALLOC(int32_t, N, 16);

    // Verify alignment
    CHECK_TRUE(is_aligned(a_f32, 16), "a_f32 is 16-byte aligned");
    CHECK_TRUE(is_aligned(b_f32, 16), "b_f32 is 16-byte aligned");
    CHECK_TRUE(is_aligned(neon_f32, 16), "neon_f32 is 16-byte aligned");

    // Fill with random data
    fill_random_f32(a_f32, N);
    fill_random_f32(b_f32, N);
    fill_random_i32(a_i32, N);
    fill_random_i32(b_i32, N);

    // ---- f32 correctness ----
    printf("\n-- f32 Correctness --\n");

    memset(ref_f32, 0, N * sizeof(float));
    memset(neon_f32, 0, N * sizeof(float));

    scalar_add_f32(a_f32, b_f32, ref_f32, N);
    neon_add_f32(a_f32, b_f32, neon_f32, N);

    CHECK_NEAR_ARRAY(ref_f32, neon_f32, N, 1e-6, "f32 addition matches scalar");

    // ---- i32 correctness ----
    printf("\n-- i32 Correctness --\n");

    memset(ref_i32, 0, N * sizeof(int32_t));
    memset(neon_i32, 0, N * sizeof(int32_t));

    scalar_add_i32(a_i32, b_i32, ref_i32, N);
    neon_add_i32(a_i32, b_i32, neon_i32, N);

    CHECK_NEAR_ARRAY(ref_i32, neon_i32, N, 1e-10, "i32 addition matches scalar");

    // ---- Benchmark ----
    printf("\n-- Benchmarks (%d timed iterations) --\n", BENCH_ITERS);

    // f32 benchmarks
    // bytes_processed = 3 arrays * N * sizeof(float) (read a,b + write c)
    size_t f32_bytes = N * 3 * sizeof(float);

    benchmark_result_t results_f32[2];
    BENCH_COMPUTE(scalar_add_f32(a_f32, b_f32, ref_f32, N), N, f32_bytes, BENCH_ITERS, results_f32[0]);
    results_f32[0].name = "scalar_add_f32";

    BENCH_COMPUTE(neon_add_f32(a_f32, b_f32, neon_f32, N), N, f32_bytes, BENCH_ITERS, results_f32[1]);
    results_f32[1].name = "neon_add_f32 (4x)";

    bench_report(results_f32, 2);

    // i32 benchmarks
    size_t i32_bytes = N * 3 * sizeof(int32_t);

    benchmark_result_t results_i32[2];
    BENCH_COMPUTE(scalar_add_i32(a_i32, b_i32, ref_i32, N), N, i32_bytes, BENCH_ITERS, results_i32[0]);
    results_i32[0].name = "scalar_add_i32";

    BENCH_COMPUTE(neon_add_i32(a_i32, b_i32, neon_i32, N), N, i32_bytes, BENCH_ITERS, results_i32[1]);
    results_i32[1].name = "neon_add_i32 (4x)";

    bench_report(results_i32, 2);

    // ---- Summary ----
    double f32_speedup = results_f32[0].elapsed_ns / results_f32[1].elapsed_ns;
    double i32_speedup = results_i32[0].elapsed_ns / results_i32[1].elapsed_ns;
    printf("Summary: f32 NEON speedup = %.2fx, i32 NEON speedup = %.2fx\n",
           f32_speedup, i32_speedup);

    ALIGNED_FREE(a_f32);
    ALIGNED_FREE(b_f32);
    ALIGNED_FREE(ref_f32);
    ALIGNED_FREE(neon_f32);
    ALIGNED_FREE(a_i32);
    ALIGNED_FREE(b_i32);
    ALIGNED_FREE(ref_i32);
    ALIGNED_FREE(neon_i32);

    printf("\nAll tests passed.\n");
    return 0;
}
