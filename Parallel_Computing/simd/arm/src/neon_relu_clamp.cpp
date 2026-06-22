#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <arm_neon.h>
#include "../../common/benchmark.h"
#include "../../common/check.h"
#include "../../common/random_data.h"
#include "../../common/aligned_buffer.h"

// =============================================================================
// neon_relu_clamp -- ReLU, Clamp, and LeakyReLU with NEON
//   ReLU:       out[i] = max(0, in[i])              -> vmaxq_f32
//   Clamp:      out[i] = clamp(in[i], lo, hi)       -> vminq_f32 + vmaxq_f32
//   LeakyReLU:  out[i] = in[i] > 0 ? in[i] : alpha*in[i]  -> vbslq_f32 mask
//   SIMD width: 4x f32 per 128-bit NEON register
//   N = 1000000
// =============================================================================

static const size_t N        = 1000000;
static const float  CLAMP_LO = -0.3f;
static const float  CLAMP_HI =  0.7f;
static const float  LEAKY_ALPHA = 0.01f;
static const int    BENCH_ITERS = 10;

// ---- ReLU scalar ----
static void scalar_relu_f32(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; i++) {
        out[i] = (in[i] > 0.0f) ? in[i] : 0.0f;
    }
}

// ---- ReLU NEON: vmaxq_f32 with zero vector ----
static void neon_relu_f32(const float* in, float* out, size_t n) {
    const float32x4_t vzero = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t vin = vld1q_f32(in + i);
        float32x4_t vout = vmaxq_f32(vin, vzero);
        vst1q_f32(out + i, vout);
    }
    for (; i < n; i++) {
        out[i] = (in[i] > 0.0f) ? in[i] : 0.0f;
    }
}

// ---- Clamp scalar ----
static void scalar_clamp_f32(const float* in, float* out, size_t n,
                             float lo, float hi) {
    for (size_t i = 0; i < n; i++) {
        float v = in[i];
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        out[i] = v;
    }
}

// ---- Clamp NEON: vminq_f32(vmaxq_f32(in, lo), hi) ----
static void neon_clamp_f32(const float* in, float* out, size_t n,
                           float lo, float hi) {
    const float32x4_t vlo = vdupq_n_f32(lo);
    const float32x4_t vhi = vdupq_n_f32(hi);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t vin  = vld1q_f32(in + i);
        float32x4_t vmax = vmaxq_f32(vin, vlo);
        float32x4_t vout = vminq_f32(vmax, vhi);
        vst1q_f32(out + i, vout);
    }
    for (; i < n; i++) {
        float v = in[i];
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        out[i] = v;
    }
}

// ---- LeakyReLU scalar ----
static void scalar_leaky_relu_f32(const float* in, float* out, size_t n,
                                  float alpha) {
    for (size_t i = 0; i < n; i++) {
        out[i] = (in[i] > 0.0f) ? in[i] : (alpha * in[i]);
    }
}

// ---- LeakyReLU NEON: vbslq_f32(mask, in, alpha*in) ----
static void neon_leaky_relu_f32(const float* in, float* out, size_t n,
                                float alpha) {
    const float32x4_t vzero  = vdupq_n_f32(0.0f);
    const float32x4_t valpha = vdupq_n_f32(alpha);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t vin      = vld1q_f32(in + i);
        float32x4_t vscaled  = vmulq_f32(vin, valpha);
        uint32x4_t  vmask    = vcgtq_f32(vin, vzero);
        float32x4_t vout     = vbslq_f32(vmask, vin, vscaled);
        vst1q_f32(out + i, vout);
    }
    for (; i < n; i++) {
        out[i] = (in[i] > 0.0f) ? in[i] : (alpha * in[i]);
    }
}

// =============================================================================
// main
// =============================================================================
int main(void) {
    printf("================================================================\n");
    printf("  NEON ReLU / Clamp / LeakyReLU\n");
    printf("  SIMD width: 4x f32 per 128-bit NEON register\n");
    printf("  Clamp range: [%.2f, %.2f], LeakyReLU alpha: %.2f\n",
           CLAMP_LO, CLAMP_HI, LEAKY_ALPHA);
    printf("  N = %zu\n", N);
    printf("================================================================\n");

    float* in    = ALIGNED_ALLOC(float, N, 16);
    float* ref   = ALIGNED_ALLOC(float, N, 16);
    float* neon  = ALIGNED_ALLOC(float, N, 16);

    CHECK_TRUE(is_aligned(in, 16), "input buffer is 16-byte aligned");

    // Generate inputs with negative, zero, and positive values
    fill_random_f32(in, N);

    // ---- ReLU Correctness ----
    printf("\n-- ReLU Correctness --\n");
    memset(ref, 0, N * sizeof(float));
    memset(neon, 0, N * sizeof(float));
    scalar_relu_f32(in, ref, N);
    neon_relu_f32(in, neon, N);
    CHECK_NEAR_ARRAY(ref, neon, N, 1e-6, "ReLU matches scalar");

    // ---- Clamp Correctness ----
    printf("\n-- Clamp Correctness --\n");
    memset(ref, 0, N * sizeof(float));
    memset(neon, 0, N * sizeof(float));
    scalar_clamp_f32(in, ref, N, CLAMP_LO, CLAMP_HI);
    neon_clamp_f32(in, neon, N, CLAMP_LO, CLAMP_HI);
    CHECK_NEAR_ARRAY(ref, neon, N, 1e-6, "Clamp matches scalar");

    // ---- LeakyReLU Correctness ----
    printf("\n-- LeakyReLU Correctness --\n");
    memset(ref, 0, N * sizeof(float));
    memset(neon, 0, N * sizeof(float));
    scalar_leaky_relu_f32(in, ref, N, LEAKY_ALPHA);
    neon_leaky_relu_f32(in, neon, N, LEAKY_ALPHA);
    CHECK_NEAR_ARRAY(ref, neon, N, 1e-6, "LeakyReLU matches scalar");

    // ---- Benchmarks ----
    printf("\n-- Benchmarks (%d timed iterations) --\n", BENCH_ITERS);
    size_t bytes = N * 2 * sizeof(float); // read + write

    // ReLU
    benchmark_result_t res_relu[2];
    BENCH_COMPUTE(scalar_relu_f32(in, ref, N), N, bytes, BENCH_ITERS, res_relu[0]);
    res_relu[0].name = "scalar_relu_f32";
    BENCH_COMPUTE(neon_relu_f32(in, neon, N), N, bytes, BENCH_ITERS, res_relu[1]);
    res_relu[1].name = "neon_relu_f32 (4x)";
    printf("\n>>> ReLU\n");
    bench_report(res_relu, 2);

    // Clamp
    benchmark_result_t res_clamp[2];
    BENCH_COMPUTE(scalar_clamp_f32(in, ref, N, CLAMP_LO, CLAMP_HI),
                  N, bytes, BENCH_ITERS, res_clamp[0]);
    res_clamp[0].name = "scalar_clamp_f32";
    BENCH_COMPUTE(neon_clamp_f32(in, neon, N, CLAMP_LO, CLAMP_HI),
                  N, bytes, BENCH_ITERS, res_clamp[1]);
    res_clamp[1].name = "neon_clamp_f32 (4x)";
    printf("\n>>> Clamp\n");
    bench_report(res_clamp, 2);

    // LeakyReLU
    benchmark_result_t res_leaky[2];
    BENCH_COMPUTE(scalar_leaky_relu_f32(in, ref, N, LEAKY_ALPHA),
                  N, bytes, BENCH_ITERS, res_leaky[0]);
    res_leaky[0].name = "scalar_leaky_relu";
    BENCH_COMPUTE(neon_leaky_relu_f32(in, neon, N, LEAKY_ALPHA),
                  N, bytes, BENCH_ITERS, res_leaky[1]);
    res_leaky[1].name = "neon_leaky_relu (4x)";
    printf("\n>>> LeakyReLU\n");
    bench_report(res_leaky, 2);

    // Summary
    printf("Summary:\n");
    printf("  ReLU speedup:      %.2fx\n",
           res_relu[0].elapsed_ns / res_relu[1].elapsed_ns);
    printf("  Clamp speedup:     %.2fx\n",
           res_clamp[0].elapsed_ns / res_clamp[1].elapsed_ns);
    printf("  LeakyReLU speedup: %.2fx\n",
           res_leaky[0].elapsed_ns / res_leaky[1].elapsed_ns);

    ALIGNED_FREE(in);
    ALIGNED_FREE(ref);
    ALIGNED_FREE(neon);

    printf("\nAll tests passed.\n");
    return 0;
}
