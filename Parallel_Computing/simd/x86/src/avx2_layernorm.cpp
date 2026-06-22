/**
 * AVX2 LayerNorm Implementation
 *
 * LayerNorm:  x_norm[i] = (x[i] - mean) / sqrt(var + eps) * gamma[i] + beta[i]
 *
 * Demonstrates:
 *   - 3-pass algorithm for mean, variance, and normalization
 *   - _mm256_rsqrt_ps: fast reciprocal sqrt approximation
 *   - Newton-Raphson refinement: y = y * (1.5 - 0.5 * x * y * y)
 *   - N = 1024 (typical ML hidden dim)
 *   - gamma = all 1s, beta = all 0s (simplified)
 *   - eps = 1e-5
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include "../../common/aligned_buffer.h"
#include "../../common/benchmark.h"
#include "../../common/check.h"
#include "../../common/cpu_features.h"
#include "../../common/random_data.h"

/* ------------------------------------------------------------------------ */
/*  Scalar LayerNorm                                                        */
/* ------------------------------------------------------------------------ */
__attribute__((noinline))
static void layernorm_scalar(const float *x, float *out,
                             const float *gamma, const float *beta,
                             int N, float eps) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++) sum += x[i];
    float mean = sum / (float)N;

    float var_sum = 0.0f;
    for (int i = 0; i < N; i++) {
        float diff = x[i] - mean;
        var_sum += diff * diff;
    }
    float var = var_sum / (float)N;
    float inv_std = 1.0f / sqrtf(var + eps);

    for (int i = 0; i < N; i++) {
        out[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

/* ------------------------------------------------------------------------ */
/*  AVX2 LayerNorm - 3-pass optimized                                       */
/* ------------------------------------------------------------------------ */
__attribute__((noinline))
static void layernorm_avx2(const float *x, float *out,
                           const float *gamma, const float *beta,
                           int N, float eps) {
    __m256 veps = _mm256_set1_ps(eps);

    /* Pass 1: compute mean via reduction sum */
    __m256 vsum = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < N; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        vsum = _mm256_add_ps(vsum, vx);
    }
    /* Horizontal reduce vsum */
    __m128 lo = _mm256_castps256_ps128(vsum);
    __m128 hi = _mm256_extractf128_ps(vsum, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float total_sum = _mm_cvtss_f32(sum128);
    /* Scalar tail for sum */
    for (; i < N; i++) total_sum += x[i];
    float mean = total_sum / (float)N;
    __m256 vmean = _mm256_set1_ps(mean);

    /* Pass 2: compute variance (sum of squared diffs from mean) */
    __m256 vvar_sum = _mm256_setzero_ps();
    i = 0;
    for (; i + 7 < N; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 diff = _mm256_sub_ps(vx, vmean);
        vvar_sum = _mm256_fmadd_ps(diff, diff, vvar_sum);
    }
    lo = _mm256_castps256_ps128(vvar_sum);
    hi = _mm256_extractf128_ps(vvar_sum, 1);
    sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float total_var = _mm_cvtss_f32(sum128);
    for (; i < N; i++) {
        float diff = x[i] - mean;
        total_var += diff * diff;
    }
    float var = total_var / (float)N;

    /* Compute 1/sqrt(var + eps) using rsqrt + Newton-Raphson refinement */
    __m256 vvar_eps = _mm256_add_ps(_mm256_set1_ps(var), veps);
    /* _mm256_rsqrt_ps gives ~11-bit precision */
    __m256 vrsqrt = _mm256_rsqrt_ps(vvar_eps);
    /* Newton-Raphson: y = y * (1.5 - 0.5 * x * y * y) */
    __m256 half    = _mm256_set1_ps(0.5f);
    __m256 three_half = _mm256_set1_ps(1.5f);
    __m256 y2 = _mm256_mul_ps(vrsqrt, vrsqrt);
    __m256 xy2 = _mm256_mul_ps(vvar_eps, y2);
    __m256 step = _mm256_sub_ps(three_half, _mm256_mul_ps(half, xy2));
    __m256 vinv_std = _mm256_mul_ps(vrsqrt, step);

    float inv_std_scalar = _mm_cvtss_f32(_mm256_castps256_ps128(vinv_std));
    __m256 vinv_std_vec = _mm256_set1_ps(inv_std_scalar);

    /* Pass 3: normalize (gamma=1, beta=0 for this demo) */
    i = 0;
    for (; i + 7 < N; i += 8) {
        __m256 vx   = _mm256_loadu_ps(x + i);
        __m256 vg   = _mm256_loadu_ps(gamma + i);
        __m256 vb   = _mm256_loadu_ps(beta + i);
        __m256 vnorm = _mm256_sub_ps(vx, vmean);
        vnorm = _mm256_mul_ps(vnorm, vinv_std_vec);
        vnorm = _mm256_fmadd_ps(vnorm, vg, vb);
        _mm256_storeu_ps(out + i, vnorm);
    }
    for (; i < N; i++) {
        out[i] = (x[i] - mean) * inv_std_scalar * gamma[i] + beta[i];
    }
}

/* ------------------------------------------------------------------------ */
/*  AVX2 LayerNorm - inline rsqrt per-element (alternative)                 */
/* ------------------------------------------------------------------------ */
__attribute__((noinline))
static void layernorm_avx2_rsqrt(const float *x, float *out,
                                 const float *gamma, const float *beta,
                                 int N, float eps) {
    /* Pass 1 & 2: mean and variance (same as above) */
    __m256 vsum = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < N; i += 8) {
        vsum = _mm256_add_ps(vsum, _mm256_loadu_ps(x + i));
    }
    __m128 lo = _mm256_castps256_ps128(vsum);
    __m128 hi = _mm256_extractf128_ps(vsum, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float total_sum = _mm_cvtss_f32(sum128);
    for (; i < N; i++) total_sum += x[i];
    float mean = total_sum / (float)N;
    __m256 vmean = _mm256_set1_ps(mean);

    __m256 vvar_sum = _mm256_setzero_ps();
    i = 0;
    for (; i + 7 < N; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 diff = _mm256_sub_ps(vx, vmean);
        vvar_sum = _mm256_fmadd_ps(diff, diff, vvar_sum);
    }
    lo = _mm256_castps256_ps128(vvar_sum);
    hi = _mm256_extractf128_ps(vvar_sum, 1);
    sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float total_var = _mm_cvtss_f32(sum128);
    for (; i < N; i++) {
        float diff = x[i] - mean;
        total_var += diff * diff;
    }
    float var = total_var / (float)N;

    /* Compute rsqrt once, broadcast for all lanes */
    float rcp = 1.0f / sqrtf(var + eps);
    __m256 vinv_std = _mm256_set1_ps(rcp);

    /* Pass 3 */
    i = 0;
    for (; i + 7 < N; i += 8) {
        __m256 vx   = _mm256_loadu_ps(x + i);
        __m256 vg   = _mm256_loadu_ps(gamma + i);
        __m256 vb   = _mm256_loadu_ps(beta + i);
        __m256 vnorm = _mm256_sub_ps(vx, vmean);
        vnorm = _mm256_mul_ps(vnorm, vinv_std);
        vnorm = _mm256_fmadd_ps(vnorm, vg, vb);
        _mm256_storeu_ps(out + i, vnorm);
    }
    for (; i < N; i++) {
        out[i] = (x[i] - mean) * rcp * gamma[i] + beta[i];
    }
}

/* ------------------------------------------------------------------------ */
/*  Main                                                                    */
/* ------------------------------------------------------------------------ */
int main() {
    cpu_print_features();

    if (!cpu_has_avx2()) {
        printf("AVX2 not supported on this CPU. Exiting.\n");
        return 1;
    }

    const int N = 1024;
    const float eps = 1e-5f;

    printf("\n=== AVX2 LayerNorm ===\n");
    printf("N = %d (typical ML hidden dim)\n", N);
    printf("eps = %.1e\n", eps);
    printf("gamma = all 1.0, beta = all 0.0\n\n");

    /* Allocate aligned buffers */
    float *x     = ALIGNED_ALLOC(float, N, 32);
    float *out   = ALIGNED_ALLOC(float, N, 32);
    float *ref   = ALIGNED_ALLOC(float, N, 32);
    float *gamma = ALIGNED_ALLOC(float, N, 32);
    float *beta  = ALIGNED_ALLOC(float, N, 32);

    /* Fill data deterministically */
    rand_xorshift64_seed(42);
    fill_range_f32(x, N, -10.0f, 10.0f);
    fill_constant_f32(gamma, N, 1.0f);
    fill_constant_f32(beta, N, 0.0f);

    /* Reference */
    layernorm_scalar(x, ref, gamma, beta, N, eps);

    /* ---- Correctness ---- */
    printf("--- Correctness ---\n");

    memset(out, 0, (size_t)N * sizeof(float));
    layernorm_avx2(x, out, gamma, beta, N, eps);
    CHECK_NEAR_ARRAY(out, ref, N, 1e-5f, "AVX2 3-pass (rsqrt+NR)");

    memset(out, 0, (size_t)N * sizeof(float));
    layernorm_avx2_rsqrt(x, out, gamma, beta, N, eps);
    CHECK_NEAR_ARRAY(out, ref, N, 1e-5f, "AVX2 scalar rcp");

    /* ---- Benchmark ---- */
    const size_t bytes_rw = (size_t)N * sizeof(float) * 4;  /* x(rd) + gamma(rd) + beta(rd) + out(wr) */

    benchmark_result_t results[3];
    memset(results, 0, sizeof(results));

    BENCH_COMPUTE(layernorm_scalar(x, out, gamma, beta, N, eps),
                  N, bytes_rw, 50, results[0]);
    results[0].name = "scalar";

    BENCH_COMPUTE(layernorm_avx2(x, out, gamma, beta, N, eps),
                  N, bytes_rw, 50, results[1]);
    results[1].name = "AVX2 (rsqrt+NR)";

    BENCH_COMPUTE(layernorm_avx2_rsqrt(x, out, gamma, beta, N, eps),
                  N, bytes_rw, 50, results[2]);
    results[2].name = "AVX2 (scalar rcp)";

    printf("\n--- Benchmark Results ---\n");
    printf("SIMD width: 256-bit (8x f32 per register)\n");
    bench_report(results, 3);

    printf("Notes:\n");
    printf("  LayerNorm: x_norm[i] = (x[i] - mean) / sqrt(var + eps) * gamma[i] + beta[i]\n");
    printf("  3-pass algorithm: pass1=mean, pass2=variance, pass3=normalize\n");
    printf("  _mm256_rsqrt_ps: fast reciprocal sqrt (~11-bit precision)\n");
    printf("  Newton-Raphson refinement: y = y * (1.5 - 0.5 * x * y * y)\n");
    printf("  NR step improves to ~23-bit precision (sufficient for LayerNorm)\n");
    printf("  Scalar rcp uses plain 1.0f / sqrtf(var+eps) broadcast to all lanes\n");

    ALIGNED_FREE(x);
    ALIGNED_FREE(out);
    ALIGNED_FREE(ref);
    ALIGNED_FREE(gamma);
    ALIGNED_FREE(beta);

    return 0;
}
