/*
 * avx2_relu_clamp.cpp -- AVX2 activation functions: ReLU, Clamp, LeakyReLU, GELU
 *
 * SIMD width: 256-bit = 8x f32 per register
 * N = 1000000
 *
 * Activations:
 *   ReLU(x)      = max(x, 0)
 *   Clamp(x)     = min(max(x, lo), hi)
 *   LeakyReLU(x) = x if x > 0 else alpha * x   (alpha = 0.01)
 *   GELU approx  = x * 0.5 * (1 + tanh(0.79788456 * (x + 0.044715 * x^3)))
 *
 * GELU tanh uses Pade (3,2) rational approximation:
 *   tanh(x) ~ x * (27 + x^2) / (27 + 9*x^2)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <immintrin.h>
#include "../../common/benchmark.h"
#include "../../common/check.h"
#include "../../common/random_data.h"
#include "../../common/aligned_buffer.h"
#include "../../common/cpu_features.h"

static const size_t N = 1000000;

/* ================================================================
 * Scalar baselines
 * ================================================================ */

static float scalar_relu(float x) {
    return (x > 0.0f) ? x : 0.0f;
}

static float scalar_clamp(float x, float lo, float hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static float scalar_leaky_relu(float x, float alpha) {
    return (x > 0.0f) ? x : alpha * x;
}

static float scalar_gelu_approx(float x) {
    /* GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
    const float c1 = 0.7978845608f; /* sqrt(2/pi) */
    const float c2 = 0.044715f;

    float x3 = x * x * x;
    float inner = c1 * (x + c2 * x3);

    /* Pade (3,2) tanh approximation */
    float i2 = inner * inner;
    float tanh_approx = inner * (27.0f + i2) / (27.0f + 9.0f * i2);

    return 0.5f * x * (1.0f + tanh_approx);
}

static void scalar_relu_array(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = scalar_relu(in[i]);
}

static void scalar_clamp_array(const float* in, float* out, size_t n, float lo, float hi) {
    for (size_t i = 0; i < n; i++) out[i] = scalar_clamp(in[i], lo, hi);
}

static void scalar_leaky_relu_array(const float* in, float* out, size_t n, float alpha) {
    for (size_t i = 0; i < n; i++) out[i] = scalar_leaky_relu(in[i], alpha);
}

static void scalar_gelu_array(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = scalar_gelu_approx(in[i]);
}

/* ================================================================
 * AVX2 ReLU: max(x, 0)
 * ================================================================ */

static void avx2_relu(const float* in, float* out, size_t n) {
    const __m256 zero = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vin = _mm256_loadu_ps(in + i);
        _mm256_storeu_ps(out + i, _mm256_max_ps(vin, zero));
    }
    for (; i < n; i++) out[i] = scalar_relu(in[i]);
}

/* ================================================================
 * AVX2 Clamp: min(max(x, lo), hi)
 * ================================================================ */

static void avx2_clamp(const float* in, float* out, size_t n, float lo, float hi) {
    const __m256 vlo = _mm256_set1_ps(lo);
    const __m256 vhi = _mm256_set1_ps(hi);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vin = _mm256_loadu_ps(in + i);
        __m256 vtmp = _mm256_max_ps(vin, vlo);
        _mm256_storeu_ps(out + i, _mm256_min_ps(vtmp, vhi));
    }
    for (; i < n; i++) out[i] = scalar_clamp(in[i], lo, hi);
}

/* ================================================================
 * AVX2 LeakyReLU: blend with mask from comparison
 * ================================================================ */

static void avx2_leaky_relu(const float* in, float* out, size_t n, float alpha) {
    const __m256 zero  = _mm256_setzero_ps();
    const __m256 valpha = _mm256_set1_ps(alpha);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vin = _mm256_loadu_ps(in + i);
        /* mask = vin > 0.0f ? 0xFFFFFFFF : 0x00000000 */
        __m256 mask = _mm256_cmp_ps(vin, zero, _CMP_GT_OQ);
        /* negative_branch = alpha * vin */
        __m256 neg = _mm256_mul_ps(vin, valpha);
        /* blend: if mask set -> keep vin, else use neg */
        __m256 res = _mm256_blendv_ps(neg, vin, mask);
        _mm256_storeu_ps(out + i, res);
    }
    for (; i < n; i++) out[i] = scalar_leaky_relu(in[i], alpha);
}

/* ================================================================
 * AVX2 GELU approximation
 *
 * GELU(x) = 0.5 * x * (1 + tanh(c1 * (x + c2 * x^3)))
 *
 * We use Pade (3,2) for tanh: t * (27 + t^2) / (27 + 9*t^2)
 * where t = c1 * (x + c2 * x^3)
 * ================================================================ */

static void avx2_gelu_approx(const float* in, float* out, size_t n) {
    const __m256 c1    = _mm256_set1_ps(0.7978845608f);  /* sqrt(2/pi) */
    const __m256 c2    = _mm256_set1_ps(0.044715f);
    const __m256 half  = _mm256_set1_ps(0.5f);
    const __m256 one   = _mm256_set1_ps(1.0f);
    const __m256 v27   = _mm256_set1_ps(27.0f);
    const __m256 v9    = _mm256_set1_ps(9.0f);

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(in + i);

        /* x^2 */
        __m256 x2 = _mm256_mul_ps(x, x);
        /* x^3 */
        __m256 x3 = _mm256_mul_ps(x2, x);
        /* t = c1 * (x + c2 * x^3) */
        __m256 t  = _mm256_fmadd_ps(c2, x3, x);
        t = _mm256_mul_ps(t, c1);

        /* Pade (3,2) tanh approx: t * (27 + t^2) / (27 + 9*t^2) */
        __m256 t2 = _mm256_mul_ps(t, t);
        __m256 num = _mm256_add_ps(v27, t2);
        __m256 den = _mm256_fmadd_ps(v9, t2, v27);
        __m256 tanh_approx = _mm256_div_ps(_mm256_mul_ps(t, num), den);

        /* 0.5 * x * (1 + tanh_approx) */
        __m256 g = _mm256_add_ps(one, tanh_approx);
        g = _mm256_mul_ps(g, x);
        g = _mm256_mul_ps(g, half);

        _mm256_storeu_ps(out + i, g);
    }
    for (; i < n; i++) out[i] = scalar_gelu_approx(in[i]);
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

    printf("\n=== AVX2 Activation Functions (N = %zu) ===\n\n", N);

    /* Allocate */
    float* in  = ALIGNED_ALLOC(float, N, 32);
    float* ref = ALIGNED_ALLOC(float, N, 32);
    float* simd = ALIGNED_ALLOC(float, N, 32);

    if (!in || !ref || !simd) {
        fprintf(stderr, "Allocation failed.\n");
        return 1;
    }

    /* Fill with values in [-3, 3] to cover negative, zero, positive */
    rand_xorshift64_seed(42);
    fill_range_f32(in, N, -3.0f, 3.0f);

    const float clamp_lo = -1.0f;
    const float clamp_hi = 1.0f;
    const float leaky_alpha = 0.01f;
    const size_t bytes_2arr = N * 2 * sizeof(float); /* read in + write out */

    /* ---- Correctness ---- */

    printf("--- Correctness: ReLU ---\n");
    memset(ref, 0, N * sizeof(float));
    memset(simd, 0, N * sizeof(float));
    scalar_relu_array(in, ref, N);
    avx2_relu(in, simd, N);
    CHECK_NEAR_ARRAY(simd, ref, N, 1e-6, "avx2_relu matches scalar");

    printf("\n--- Correctness: Clamp [%.1f, %.1f] ---\n", clamp_lo, clamp_hi);
    memset(ref, 0, N * sizeof(float));
    memset(simd, 0, N * sizeof(float));
    scalar_clamp_array(in, ref, N, clamp_lo, clamp_hi);
    avx2_clamp(in, simd, N, clamp_lo, clamp_hi);
    CHECK_NEAR_ARRAY(simd, ref, N, 1e-6, "avx2_clamp matches scalar");

    printf("\n--- Correctness: LeakyReLU (alpha=%.2f) ---\n", leaky_alpha);
    memset(ref, 0, N * sizeof(float));
    memset(simd, 0, N * sizeof(float));
    scalar_leaky_relu_array(in, ref, N, leaky_alpha);
    avx2_leaky_relu(in, simd, N, leaky_alpha);
    CHECK_NEAR_ARRAY(simd, ref, N, 1e-6, "avx2_leaky_relu matches scalar");

    printf("\n--- Correctness: GELU (approx) ---\n");
    memset(ref, 0, N * sizeof(float));
    memset(simd, 0, N * sizeof(float));
    scalar_gelu_array(in, ref, N);
    avx2_gelu_approx(in, simd, N);
    /* GELU approximation has limited precision; use 1e-4 tolerance */
    CHECK_NEAR_ARRAY(simd, ref, N, 1e-4, "avx2_gelu_approx matches scalar");

    /* ---- Benchmark ---- */

    benchmark_result_t results[8];
    memset(results, 0, sizeof(results));

    BENCH_COMPUTE(scalar_relu_array(in, ref, N),
                  N, bytes_2arr, 20, results[0]);
    results[0].name = "scalar_relu";

    BENCH_COMPUTE(avx2_relu(in, simd, N),
                  N, bytes_2arr, 20, results[1]);
    results[1].name = "avx2_relu";

    BENCH_COMPUTE(scalar_clamp_array(in, ref, N, clamp_lo, clamp_hi),
                  N, bytes_2arr, 20, results[2]);
    results[2].name = "scalar_clamp";

    BENCH_COMPUTE(avx2_clamp(in, simd, N, clamp_lo, clamp_hi),
                  N, bytes_2arr, 20, results[3]);
    results[3].name = "avx2_clamp";

    BENCH_COMPUTE(scalar_leaky_relu_array(in, ref, N, leaky_alpha),
                  N, bytes_2arr, 20, results[4]);
    results[4].name = "scalar_leaky_relu";

    BENCH_COMPUTE(avx2_leaky_relu(in, simd, N, leaky_alpha),
                  N, bytes_2arr, 20, results[5]);
    results[5].name = "avx2_leaky_relu";

    BENCH_COMPUTE(scalar_gelu_array(in, ref, N),
                  N, bytes_2arr, 20, results[6]);
    results[6].name = "scalar_gelu_approx";

    BENCH_COMPUTE(avx2_gelu_approx(in, simd, N),
                  N, bytes_2arr, 20, results[7]);
    results[7].name = "avx2_gelu_approx";

    printf("\n--- Benchmark Results ---\n");
    printf("SIMD width: 256-bit (8x f32)\n");
    bench_report(results, 8);

    printf("Notes:\n");
    printf("  - ReLU is a simple max(x,0) -- very fast with _mm256_max_ps\n");
    printf("  - Clamp chains min+max -- 2 SIMD ops per 8 elements\n");
    printf("  - LeakyReLU uses blendv based on comparison mask; same cost as\n");
    printf("    a conditional move in scalar\n");
    printf("  - GELU is compute-bound (mul, add, div, fmadd). The Pade tanh\n");
    printf("    approximant replaces the expensive tanh() with simple arithmetic.\n");
    printf("    _mm256_div_ps is high-latency (~13 cycles on Skylake);\n");
    printf("    polynomial-only approaches can avoid division.\n");

    ALIGNED_FREE(in);
    ALIGNED_FREE(ref);
    ALIGNED_FREE(simd);

    return 0;
}
