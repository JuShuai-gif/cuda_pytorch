/**
 * AVX2 Softmax - Numerator Computation
 *
 * softmax(x)[i] = exp(x[i] - max) / sum(exp(x[j] - max))
 *
 * This file shows:
 *   - 2-pass approach: find max, subtract, exp, sum
 *   - Online max tracking (1-pass streaming)
 *   - Polynomial exp approximation for AVX2 (no SVML)
 *   - skips final division (scalar phase, outside SIMD scope)
 *   - N = 1024
 */

#include "../../common/aligned_buffer.h"
#include "../../common/benchmark.h"
#include "../../common/check.h"
#include "../../common/cpu_features.h"
#include "../../common/random_data.h"

#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static inline __m256 _my_exp_ps(__m256 x) {
    const __m256 lower = _mm256_set1_ps(-87.0f);
    const __m256 upper = _mm256_set1_ps(87.0f);
    x = _mm256_max_ps(lower, _mm256_min_ps(upper, x));

    /* C1 = ln(2)  (exact single-precision) */
    const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
    const __m256 ln2_hi = _mm256_set1_ps(0.693359375f);
    const __m256 ln2_lo = _mm256_set1_ps(-2.12194440e-4f);
    const __m256 one = _mm256_set1_ps(1.0f);

    /* exp(r) Taylor coefficients for small r */
    const __m256 c2 = _mm256_set1_ps(0.5f);             /* 1/2! */
    const __m256 c3 = _mm256_set1_ps(1.6666666666e-1f); /* 1/3! */
    const __m256 c4 = _mm256_set1_ps(4.1666666666e-2f); /* 1/4! */
    const __m256 c5 = _mm256_set1_ps(8.3333333333e-3f); /* 1/5! */

    /* n = round(x * log2(e)) */
    __m256 n = _mm256_mul_ps(x, log2e);
    n = _mm256_round_ps(n, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    /* r = x - n * ln(2), split for precision */
    __m256 r = _mm256_fnmadd_ps(n, ln2_hi, x);  /* r_hi = remaining */
    r = _mm256_fnmadd_ps(n, ln2_lo, r);          /* r_lo = final remainder */

    /* exp(r) = 1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120 + ... */
    __m256 poly = _mm256_fmadd_ps(c5, r, c4);
    poly = _mm256_fmadd_ps(poly, r, c3);
    poly = _mm256_fmadd_ps(poly, r, c2);
    poly = _mm256_fmadd_ps(poly, r, one);
    poly = _mm256_fmadd_ps(poly, r, one);

    /* 2^n: shift exponent field. n as integer + 127 bias */
    __m256i ni = _mm256_cvtps_epi32(n);
    ni = _mm256_add_epi32(ni, _mm256_set1_epi32(127));
    ni = _mm256_slli_epi32(ni, 23);

    return _mm256_mul_ps(_mm256_castsi256_ps(ni), poly);
}

__attribute__((noinline))
void softmax_scalar(const float *x, float *num, float *denom, int N) {
    float max_val = x[0];
    for (int i = 1; i < N; i++)
        if (x[i] > max_val) max_val = x[i];

    float sum_exp = 0.0f;
    for (int i = 0; i < N; i++) {
        float e = expf(x[i] - max_val);
        num[i] = e;
        sum_exp += e;
    }
    *denom = sum_exp;
}

/* 2-pass approach */
__attribute__((noinline))
void softmax_avx2_2pass(const float *x, float *num, float *denom, int N) {
    /* Pass 1: find max */
    __m256 vmax = _mm256_set1_ps(-1e30f);
    int i = 0;
    for (; i + 7 < N; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        vmax = _mm256_max_ps(vmax, vx);
    }
    __m128 lo = _mm256_castps256_ps128(vmax);
    __m128 hi = _mm256_extractf128_ps(vmax, 1);
    __m128 m128 = _mm_max_ps(lo, hi);
    m128 = _mm_max_ps(m128, _mm_permute_ps(m128, 0x4E));
    m128 = _mm_max_ps(m128, _mm_permute_ps(m128, 0xB1));
    float max_val = _mm_cvtss_f32(m128);
    for (; i < N; i++) if (x[i] > max_val) max_val = x[i];

    __m256 vmax_val = _mm256_set1_ps(max_val);
    __m256 vsum = _mm256_setzero_ps();

    /* Pass 2: exp and accumulate */
    i = 0;
    for (; i + 7 < N; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 diff = _mm256_sub_ps(vx, vmax_val);
        __m256 ve = _my_exp_ps(diff);
        _mm256_storeu_ps(num + i, ve);
        vsum = _mm256_add_ps(vsum, ve);
    }
    lo = _mm256_castps256_ps128(vsum);
    hi = _mm256_extractf128_ps(vsum, 1);
    m128 = _mm_add_ps(lo, hi);
    m128 = _mm_hadd_ps(m128, m128);
    m128 = _mm_hadd_ps(m128, m128);
    float sum_exp = _mm_cvtss_f32(m128);

    for (; i < N; i++) {
        float e = expf(x[i] - max_val);
        num[i] = e;
        sum_exp += e;
    }
    *denom = sum_exp;
}

/* Online max tracking (1-pass streaming) */
__attribute__((noinline))
void softmax_avx2_online(const float *x, float *num, float *denom, int N) {
    /* Naive online: maintain running max, but this is not true 1-pass
     * because we'd need to renormalize. For the demo, we show the
     * streaming max approach often used in flash attention style kernels. */

    __m256 vmax_running = _mm256_set1_ps(-1e30f);
    __m256 vsum = _mm256_setzero_ps();

    int i = 0;
    for (; i + 7 < N; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        vmax_running = _mm256_max_ps(vmax_running, vx);

        __m256 diff = _mm256_sub_ps(vx, vmax_running);
        __m256 ve = _my_exp_ps(diff);
        _mm256_storeu_ps(num + i, ve);
        vsum = _mm256_add_ps(vsum, ve);
    }

    __m128 lo = _mm256_castps256_ps128(vmax_running);
    __m128 hi = _mm256_extractf128_ps(vmax_running, 1);
    __m128 m128 = _mm_max_ps(lo, hi);
    m128 = _mm_max_ps(m128, _mm_permute_ps(m128, 0x4E));
    m128 = _mm_max_ps(m128, _mm_permute_ps(m128, 0xB1));
    float max_val = _mm_cvtss_f32(m128);

    lo = _mm256_castps256_ps128(vsum);
    hi = _mm256_extractf128_ps(vsum, 1);
    m128 = _mm_add_ps(lo, hi);
    m128 = _mm_hadd_ps(m128, m128);
    m128 = _mm_hadd_ps(m128, m128);
    float sum_exp = _mm_cvtss_f32(m128);

    for (; i < N; i++) {
        if (x[i] > max_val) max_val = x[i];
        float e = expf(x[i] - max_val);
        num[i] = e;
        sum_exp += e;
    }
    *denom = sum_exp;
}

/* ---- Main ------------------------------------------------------------- */
int main() {
    cpu_print_features();

    if (!cpu_has_avx2()) {
        printf("AVX2 not supported on this CPU. Exiting.\n");
        return 0;
    }

    const int N = 1024;

    printf("\n=== AVX2 Softmax (Numerator + Denominator) ===\n");
    printf("N = %d\n", N);
    printf("SIMD width = 256 bits (8 f32 per register)\n");
    printf("exp polynomial: 2^f via degree-5 approximation + int scaling\n\n");

    float *x       = ALIGNED_ALLOC(float, N, 32);
    float *num     = ALIGNED_ALLOC(float, N, 32);
    float *ref_num = ALIGNED_ALLOC(float, N, 32);
    float *num2    = ALIGNED_ALLOC(float, N, 32);

    /* Fill with range -10 to 10, then set specific outliers */
    fill_range_f32(x, N, -10.0f, 10.0f);
    x[0]     = 8.0f;
    x[N/2]   = 12.0f;
    x[N-1]   = -5.0f;

    /* Reference */
    float denom_ref;
    softmax_scalar(x, ref_num, &denom_ref, N);
    printf("Reference denominator = %.6f\n", denom_ref);

    /* AVX2 2-pass */
    float denom_avx2;
    softmax_avx2_2pass(x, num, &denom_avx2, N);
    printf("\nAVX2 2-pass:\n  denominator=%.6f (ref=%.6f, diff=%.2e)\n",
           denom_avx2, denom_ref, fabsf(denom_avx2 - denom_ref));
    CHECK_NEAR_ARRAY(num, ref_num, N, 0.02f, "AVX2 2-pass numerator vs reference");

    /* AVX2 online (approximate algorithm; correctness not guaranteed) */
    float denom_online;
    softmax_avx2_online(x, num2, &denom_online, N);
    printf("\nAVX2 online-max (approximate):\n  denominator=%.6f (ref=%.6f, diff=%.2e)\n",
           denom_online, denom_ref, fabsf(denom_online - denom_ref));

    /* Benchmark */
    const int iters = 200000;
    const int bytes_per_call = (N * 2 + 1) * (int)sizeof(float);
    benchmark_result_t results[3];

    BENCH_COMPUTE(softmax_scalar(x, num, &denom_avx2, N),
                  N, bytes_per_call, iters, results[0]);
    results[0].name = "scalar";

    BENCH_COMPUTE(softmax_avx2_2pass(x, num, &denom_avx2, N),
                  N, bytes_per_call, iters, results[1]);
    results[1].name = "AVX2 2-pass";

    BENCH_COMPUTE(softmax_avx2_online(x, num2, &denom_online, N),
                  N, bytes_per_call, iters, results[2]);
    results[2].name = "AVX2 online";

    bench_report(results, 3);

    printf("--- 2-pass vs Online Max Tracking ---\n");
    printf("2-pass:  Find global max (O(N)), then exp+sum (O(N)).\n");
    printf("Online:  Track running max per vector, exp immediately.\n");
    printf("  Advantage: single pass, better cache locality.\n");
    printf("  Disadvantage: exp values computed before knowing true max,\n");
    printf("  so they are approximate (slightly wrong when max changes).\n");
    printf("Note: True online softmax requires rescaling: when new max found,\n");
    printf("multiply all previous exp values by exp(old_max - new_max).\n");

    printf("\n--- Polynomial exp approximation ---\n");
    printf("Without SVML (_mm256_exp_ps), we compute:\n");
    printf("  exp(x) = 2^(x * log2(e))\n");
    printf("  Split: x*log2(e) = floor + frac\n");
    printf("  2^(frac) via polynomial: 1 + x*P(x)\n");
    printf("  Scale by 2^(floor) via integer exponent in float bits.\n");
    printf("  Relative error <= 0.02 (1.5-2%% for typical values).\n");

    ALIGNED_FREE(x);
    ALIGNED_FREE(num);
    ALIGNED_FREE(ref_num);
    ALIGNED_FREE(num2);
    return 0;
}
