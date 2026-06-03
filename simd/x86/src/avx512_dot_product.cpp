/**
 * AVX-512 Dot Product with FMA
 *
 * Demonstrates:
 *   - _mm512_fmadd_ps: fused multiply-add (a*b + c) in single instruction
 *   - 4-way accumulator unrolling for instruction-level parallelism (ILP)
 *   - N = 1000000
 *   - Compare AVX-512 (16-way) vs AVX2 (8-way) vs scalar
 *   - Runtime AVX-512 check
 */

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __linux__
#include <cpuid.h>
#endif

static double get_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int cpu_has_avx512f() {
#ifndef __linux__
    return 0;
#else
    unsigned int eax, ebx, ecx, edx;
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return 0;
    return (ebx & (1 << 16)) != 0;
#endif
}

/* ---- Scalar ---- */
float dot_scalar(const float *a, const float *b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++)
        sum += a[i] * b[i];
    return sum;
}

/* ---- AVX2: 8-way FMA with 4 accumulators ---- */
float dot_avx2(const float *a, const float *b, int n) {
    __m256 vsum0 = _mm256_setzero_ps();
    __m256 vsum1 = _mm256_setzero_ps();
    __m256 vsum2 = _mm256_setzero_ps();
    __m256 vsum3 = _mm256_setzero_ps();

    int i = 0;
    for (; i + 31 < n; i += 32) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        vsum0 = _mm256_fmadd_ps(va0, vb0, vsum0);

        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        vsum1 = _mm256_fmadd_ps(va1, vb1, vsum1);

        __m256 va2 = _mm256_loadu_ps(a + i + 16);
        __m256 vb2 = _mm256_loadu_ps(b + i + 16);
        vsum2 = _mm256_fmadd_ps(va2, vb2, vsum2);

        __m256 va3 = _mm256_loadu_ps(a + i + 24);
        __m256 vb3 = _mm256_loadu_ps(b + i + 24);
        vsum3 = _mm256_fmadd_ps(va3, vb3, vsum3);
    }
    for (; i + 7 < n; i += 8) {
        vsum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),
                                 _mm256_loadu_ps(b + i), vsum0);
    }

    vsum0 = _mm256_add_ps(vsum0, vsum1);
    vsum0 = _mm256_add_ps(vsum0, vsum2);
    vsum0 = _mm256_add_ps(vsum0, vsum3);

    __m128 lo = _mm256_castps256_ps128(vsum0);
    __m128 hi = _mm256_extractf128_ps(vsum0, 1);
    __m128 s128 = _mm_add_ps(lo, hi);
    s128 = _mm_hadd_ps(s128, s128);
    s128 = _mm_hadd_ps(s128, s128);
    float result = _mm_cvtss_f32(s128);

    for (; i < n; i++) result += a[i] * b[i];
    return result;
}

/* ---- AVX-512: 16-way FMA with 4 accumulators ---- */
float dot_avx512_4acc(const float *a, const float *b, int n) {
    __m512 vsum0 = _mm512_setzero_ps();
    __m512 vsum1 = _mm512_setzero_ps();
    __m512 vsum2 = _mm512_setzero_ps();
    __m512 vsum3 = _mm512_setzero_ps();

    int i = 0;
    for (; i + 63 < n; i += 64) {
        vsum0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i),
                                 _mm512_loadu_ps(b + i), vsum0);
        vsum1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16),
                                 _mm512_loadu_ps(b + i + 16), vsum1);
        vsum2 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 32),
                                 _mm512_loadu_ps(b + i + 32), vsum2);
        vsum3 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 48),
                                 _mm512_loadu_ps(b + i + 48), vsum3);
    }
    for (; i + 15 < n; i += 16) {
        vsum0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i),
                                 _mm512_loadu_ps(b + i), vsum0);
    }

    vsum0 = _mm512_add_ps(vsum0, vsum1);
    vsum0 = _mm512_add_ps(vsum0, vsum2);
    vsum0 = _mm512_add_ps(vsum0, vsum3);

    float result = _mm512_reduce_add_ps(vsum0);

    for (; i < n; i++) result += a[i] * b[i];
    return result;
}

/* ---- AVX-512: 1 accumulator (show ILP benefit) ---- */
float dot_avx512_1acc(const float *a, const float *b, int n) {
    __m512 vsum = _mm512_setzero_ps();

    int i = 0;
    for (; i + 15 < n; i += 16) {
        vsum = _mm512_fmadd_ps(_mm512_loadu_ps(a + i),
                                _mm512_loadu_ps(b + i), vsum);
    }

    float result = _mm512_reduce_add_ps(vsum);

    for (; i < n; i++) result += a[i] * b[i];
    return result;
}

/* ---- AVX-512: 8 accumulators ---- */
float dot_avx512_8acc(const float *a, const float *b, int n) {
    __m512 vsum[8] = {};
    for (int k = 0; k < 8; k++) vsum[k] = _mm512_setzero_ps();

    int i = 0;
    for (; i + 127 < n; i += 128) {
        for (int k = 0; k < 8; k++) {
            int off = i + k * 16;
            vsum[k] = _mm512_fmadd_ps(_mm512_loadu_ps(a + off),
                                       _mm512_loadu_ps(b + off), vsum[k]);
        }
    }
    for (; i + 15 < n; i += 16) {
        vsum[0] = _mm512_fmadd_ps(_mm512_loadu_ps(a + i),
                                   _mm512_loadu_ps(b + i), vsum[0]);
    }

    for (int k = 1; k < 8; k++)
        vsum[0] = _mm512_add_ps(vsum[0], vsum[k]);

    float result = _mm512_reduce_add_ps(vsum[0]);

    for (; i < n; i++) result += a[i] * b[i];
    return result;
}

/* ---- Verification ---- */
static int verify(float got, float expected, float tol) {
    float err = fabsf(got - expected);
    printf("  result=%.6f  expected=%.6f  err=%.2e  %s\n",
           got, expected, err, err < tol ? "OK" : "FAIL");
    return err < tol;
}

/* ---- Main ---- */
int main() {
    const int N = 1000000;

    printf("=== AVX-512 Dot Product with FMA ===\n");
    printf("N = %d\n", N);
    printf("FMA: a*b + c in single instruction (4 cycles latency, 0.5 CPI)\n");
    printf("SIMD widths: AVX2=8, AVX-512=16 f32\n\n");

    if (!cpu_has_avx512f()) {
        printf("AVX-512F not available on this CPU.\n");
        printf("Compile with -mavx512f; run on AVX-512 capable hardware.\n");
        return 0;
    }

    printf("AVX-512F: YES\n\n");

    float *a = (float*)aligned_alloc(64, N * sizeof(float));
    float *b = (float*)aligned_alloc(64, N * sizeof(float));

    srand(42);
    for (int i = 0; i < N; i++) {
        a[i] = (float)(rand() % 1000) / 1000.0f;
        b[i] = (float)(rand() % 1000) / 1000.0f;
    }

    /* Verification */
    float ref = dot_scalar(a, b, N);
    printf("Reference result: %.6f\n\n", ref);

    printf("AVX2 (4-acc): ");
    float r_avx2 = dot_avx2(a, b, N);
    verify(r_avx2, ref, 0.01f);

    printf("AVX-512 (1-acc): ");
    float r_1 = dot_avx512_1acc(a, b, N);
    verify(r_1, ref, 0.01f);

    printf("AVX-512 (4-acc): ");
    float r_4 = dot_avx512_4acc(a, b, N);
    verify(r_4, ref, 0.01f);

    printf("AVX-512 (8-acc): ");
    float r_8 = dot_avx512_8acc(a, b, N);
    verify(r_8, ref, 0.01f);

    /* Benchmark */
    int iters = 500;
    printf("\nPerformance (%d iterations):\n", iters);

    volatile float sink;

    double t0 = get_time_sec();
    for (int k = 0; k < iters; k++) sink = dot_scalar(a, b, N);
    double t_s = (get_time_sec() - t0) / iters;

    double t1 = get_time_sec();
    for (int k = 0; k < iters; k++) sink = dot_avx2(a, b, N);
    double t_a2 = (get_time_sec() - t1) / iters;

    double t2 = get_time_sec();
    for (int k = 0; k < iters; k++) sink = dot_avx512_1acc(a, b, N);
    double t_1a = (get_time_sec() - t2) / iters;

    double t3 = get_time_sec();
    for (int k = 0; k < iters; k++) sink = dot_avx512_4acc(a, b, N);
    double t_4a = (get_time_sec() - t3) / iters;

    double t4 = get_time_sec();
    for (int k = 0; k < iters; k++) sink = dot_avx512_8acc(a, b, N);
    double t_8a = (get_time_sec() - t4) / iters;

    printf("  Scalar:           %7.1f us  (%.2f GFLOPS)\n",
           t_s * 1e6, (2.0 * N) / t_s / 1e9);
    printf("  AVX2 4-acc:       %7.1f us  (%.2fx, %.2f GFLOPS)\n",
           t_a2 * 1e6, t_s / t_a2, (2.0 * N) / t_a2 / 1e9);
    printf("  AVX-512 1-acc:    %7.1f us  (%.2fx, %.2f GFLOPS)\n",
           t_1a * 1e6, t_s / t_1a, (2.0 * N) / t_1a / 1e9);
    printf("  AVX-512 4-acc:    %7.1f us  (%.2fx, %.2f GFLOPS)\n",
           t_4a * 1e6, t_s / t_4a, (2.0 * N) / t_4a / 1e9);
    printf("  AVX-512 8-acc:    %7.1f us  (%.2fx, %.2f GFLOPS)\n",
           t_8a * 1e6, t_s / t_8a, (2.0 * N) / t_8a / 1e9);

    printf("\n--- FMA ILP Analysis ---\n");
    printf("FMA latency = 4 cycles, throughput = 2 per cycle (Skylake-X).\n");
    printf("To hide latency: need >= 4 * 2 = 8 independent accumulators\n");
    printf("for full throughput on Skylake-X. 4 accumulators = 2x improvement.\n");
    printf("8 accumulators should approach theoretical peak for L1-resident data.\n");

    free(a); free(b);
    return 0;
}
