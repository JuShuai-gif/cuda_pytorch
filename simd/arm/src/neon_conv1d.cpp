/**
 * neon_conv1d.cpp -- 1D Convolution with ARM NEON
 *
 * Computes out[i] = sum(kernel[j] * input[i+j]) for j=0..K-1
 *
 * NEON strategies demonstrated:
 *   - k=3: "Register rotation" -- shift window by 1, reuse 2 of 4 values
 *   - k=5: Vector multiply-accumulate per output position
 *   - Edge handling via zero-padding
 */

#include <arm_neon.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>

#include "../../common/benchmark.h"
#include "../../common/cpu_features.h"
#include "../../common/check.h"
#include "../../common/random_data.h"
#include "../../common/aligned_buffer.h"

// ============================================================================
// Constants
// ============================================================================

static const float KERNEL_3[3] = {0.25f, 0.50f, 0.25f}; // smoothing kernel
static const float KERNEL_5[5] = {0.1f, 0.2f, 0.4f, 0.2f, 0.1f};
static const int   N = 1000000;

// ============================================================================
// Scalar baselines
// ============================================================================

static void scalar_conv1d_k3(float* __restrict out,
                              const float* __restrict in, int n) {
    const float k0 = KERNEL_3[0], k1 = KERNEL_3[1], k2 = KERNEL_3[2];
    for (int i = 0; i <= n - 3; i++) {
        out[i] = k0 * in[i] + k1 * in[i + 1] + k2 * in[i + 2];
    }
}

static void scalar_conv1d_k5(float* __restrict out,
                              const float* __restrict in, int n) {
    const float* k = KERNEL_5;
    for (int i = 0; i <= n - 5; i++) {
        out[i] = k[0] * in[i] + k[1] * in[i+1] + k[2] * in[i+2]
               + k[3] * in[i+3] + k[4] * in[i+4];
    }
}

static void scalar_conv1d_k3_zeropad(float* __restrict out,
                                      const float* __restrict in, int n) {
    const float k0 = KERNEL_3[0], k1 = KERNEL_3[1], k2 = KERNEL_3[2];
    for (int i = 0; i < n; i++) {
        float s = 0.0f;
        if (i >= 0 && i < n)       s += k0 * in[i];
        if (i+1 >= 0 && i+1 < n)   s += k1 * in[i+1];
        if (i+2 >= 0 && i+2 < n)   s += k2 * in[i+2];
        out[i] = s;
    }
}

// ============================================================================
// NEON k=3: Register Rotation (2 outputs per iteration)
// ============================================================================
//
// Strategy: Load 4 values [x0, x1, x2, x3] into a single register.
//   Position i:     dot([k0,k1,k2,0], [x0,x1,x2,x3])
//   Position i+1:   vextq by 1 lane -> [x1,x2,x3,0], dot with kernel
// This demonstrates reusing 2 values (x1, x2) between consecutive outputs
// without reloading from memory.

static void neon_conv1d_k3_rotate(float* __restrict out,
                                   const float* __restrict in, int n) {
    const float k0 = KERNEL_3[0], k1 = KERNEL_3[1], k2 = KERNEL_3[2];
    // Pack kernel into a vector: fourth lane is zero (no contribution)
    float32x4_t kvec = {k0, k1, k2, 0.0f};
    float32x4_t vzero = vdupq_n_f32(0.0f);

    int i = 0;
    // Process 2 outputs per iteration: uses vextq to reuse data in registers
    for (; i <= n - 4; i += 2) {
        // Load window [in[i], in[i+1], in[i+2], in[i+3]]
        float32x4_t v = vld1q_f32(&in[i]);

        // Output i: dot product with kernel
        // Since kvec[3]=0, vaddvq sums k0*in[i] + k1*in[i+1] + k2*in[i+2]
        out[i] = vaddvq_f32(vmulq_f32(v, kvec));

        // Output i+1: shift window by 1, "rotate" the register
        // vextq_f32(a, b, n): extract 4 lanes from concatenation a:b
        //   starting at lane n of a (0-indexed)
        // vextq_f32(v, vzero, 1) -> [v[1], v[2], v[3], 0]
        float32x4_t v_shifted = vextq_f32(v, vzero, 1);
        out[i + 1] = vaddvq_f32(vmulq_f32(v_shifted, kvec));
    }

    // Scalar tail for remaining elements
    for (; i <= n - 3; i++) {
        out[i] = k0 * in[i] + k1 * in[i + 1] + k2 * in[i + 2];
    }
}

// ============================================================================
// NEON k=5: Vector multiply-accumulate (1 output per iteration)
// ============================================================================
//
// Kernel size 5 exceeds NEON's 4-wide register. Strategy:
//   Load [in[i]..in[i+3]] into v0, multiply by [k0..k3]
//   Horizontal-reduce v0, then add k4 * in[i+4] separately.

static void neon_conv1d_k5(float* __restrict out,
                            const float* __restrict in, int n) {
    const float* k = KERNEL_5;
    // Pack first 4 kernel values; 5th handled separately
    float32x4_t kvec_lo = vld1q_f32(k); // [k0, k1, k2, k3]

    int i = 0;
    for (; i <= n - 5; i++) {
        float32x4_t v = vld1q_f32(&in[i]);                // [in[i],..,in[i+3]]
        float32x4_t prod = vmulq_f32(v, kvec_lo);          // [k0*x0,..,k3*x3]
        float sum = vaddvq_f32(prod) + k[4] * in[i + 4];  // add k4*x4
        out[i] = sum;
    }
}

// ============================================================================
// NEON k=3 with zero-padding (full output length, including edges)
// ============================================================================

static void neon_conv1d_k3_zeropad(float* __restrict out,
                                    const float* __restrict in, int n) {
    const float k0 = KERNEL_3[0], k1 = KERNEL_3[1], k2 = KERNEL_3[2];
    float32x4_t kvec = {k0, k1, k2, 0.0f};
    float32x4_t vzero = vdupq_n_f32(0.0f);

    int i = 0;
    for (; i <= n - 4; i += 2) {
        float32x4_t v = vld1q_f32(&in[i]);
        out[i]     = vaddvq_f32(vmulq_f32(v, kvec));
        float32x4_t vs = vextq_f32(v, vzero, 1);
        out[i + 1] = vaddvq_f32(vmulq_f32(vs, kvec));
    }

    // Zero-padded tail: last 2 outputs (i = n-2, n-1)
    for (; i < n; i++) {
        float s = 0.0f;
        if (i < n)      s += k0 * in[i];
        if (i + 1 < n)  s += k1 * in[i + 1];
        if (i + 2 < n)  s += k2 * in[i + 2];
        out[i] = s;
    }
}

// ============================================================================
// Correctness checks
// ============================================================================

static void check_k3(const float* in, int n_out) {
    float* ref  = ALIGNED_ALLOC(float, n_out, 64);
    float* test = ALIGNED_ALLOC(float, n_out, 64);
    float* in_pad  = ALIGNED_ALLOC(float, n_out, 64);
    float* test_pad = ALIGNED_ALLOC(float, n_out, 64);

    scalar_conv1d_k3(ref, in, n_out);
    neon_conv1d_k3_rotate(test, in, n_out);
    CHECK_NEAR_ARRAY(test, ref, n_out - 3, 1e-5f, "NEON conv1d k=3 vs scalar");

    scalar_conv1d_k3_zeropad(in_pad, in, n_out);
    neon_conv1d_k3_zeropad(test_pad, in, n_out);
    CHECK_NEAR_ARRAY(test_pad, in_pad, n_out, 1e-5f, "NEON conv1d k=3 zeropad vs scalar");

    ALIGNED_FREE(ref);
    ALIGNED_FREE(test);
    ALIGNED_FREE(in_pad);
    ALIGNED_FREE(test_pad);
}

static void check_k5(const float* in, int n_out) {
    float* ref  = ALIGNED_ALLOC(float, n_out, 64);
    float* test = ALIGNED_ALLOC(float, n_out, 64);

    scalar_conv1d_k5(ref, in, n_out);
    neon_conv1d_k5(test, in, n_out);
    CHECK_NEAR_ARRAY(test, ref, n_out - 5, 1e-5f, "NEON conv1d k=5 vs scalar");

    ALIGNED_FREE(ref);
    ALIGNED_FREE(test);
}

// ============================================================================
// Benchmark wrappers (operate on global input/output for noinline effect)
// ============================================================================

static float* g_in  = nullptr;
static float* g_out = nullptr;
static float* g_res = nullptr;
static int    g_n   = 0;
static int    g_n_out = 0;

__attribute__((noinline))
static void bench_scalar_k3() { scalar_conv1d_k3(g_out, g_in, g_n); }
__attribute__((noinline))
static void bench_neon_k3()   { neon_conv1d_k3_rotate(g_out, g_in, g_n); }
__attribute__((noinline))
static void bench_scalar_k5() { scalar_conv1d_k5(g_out, g_in, g_n); }
__attribute__((noinline))
static void bench_neon_k5()   { neon_conv1d_k5(g_out, g_in, g_n); }

// ============================================================================
// Main
// ============================================================================

int main() {
    cpu_print_features();

    // Allocate aligned buffers
    int n       = N;
    int n_out_k3  = n - 2;  /* valid output size for k=3 */
    int n_out_k5  = n - 4;  /* valid output size for k=5 */
    g_in  = ALIGNED_ALLOC(float, n, 64);
    g_out = ALIGNED_ALLOC(float, std::max(n_out_k3, n_out_k5), 64);
    g_n   = n;

    fill_random_f32(g_in, n);

    // --- Correctness ---
    printf("\n=== Correctness Checks ===\n");
    check_k3(g_in, n);
    check_k5(g_in, n);

    // --- Benchmark: k=3 ---
    printf("\n=== Benchmark: Conv1D k=3 (N=%d) ===\n", n);
    benchmark_result_t results_k3[2];

    g_n_out = n_out_k3;
    BENCH_COMPUTE(bench_scalar_k3(), n_out_k3,
        n_out_k3 * sizeof(float) + n * sizeof(float), 30, results_k3[0]);
    results_k3[0].name = "scalar k=3";

    BENCH_COMPUTE(bench_neon_k3(), n_out_k3,
        n_out_k3 * sizeof(float) + n * sizeof(float), 30, results_k3[1]);
    results_k3[1].name = "NEON k=3 (rotate)";

    bench_report(results_k3, 2);

    // --- Benchmark: k=5 ---
    printf("\n=== Benchmark: Conv1D k=5 (N=%d) ===\n", n);
    benchmark_result_t results_k5[2];

    g_n_out = n_out_k5;
    BENCH_COMPUTE(bench_scalar_k5(), n_out_k5,
        n_out_k5 * sizeof(float) + n * sizeof(float), 30, results_k5[0]);
    results_k5[0].name = "scalar k=5";

    BENCH_COMPUTE(bench_neon_k5(), n_out_k5,
        n_out_k5 * sizeof(float) + n * sizeof(float), 30, results_k5[1]);
    results_k5[1].name = "NEON k=5 (vmla)";

    bench_report(results_k5, 2);

    ALIGNED_FREE(g_in);
    ALIGNED_FREE(g_out);
    return 0;
}
