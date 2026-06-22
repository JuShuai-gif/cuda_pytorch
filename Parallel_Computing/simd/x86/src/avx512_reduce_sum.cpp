/**
 * AVX-512 Reduction Sum
 *
 * Demonstrates:
 *   - _mm512_reduce_add_ps: intrinsic shorthand for horizontal sum
 *   - Manual reduce via permute + add (shows what happens under the hood)
 *   - Multiple accumulator approach for ILP
 *   - N = 1000000
 *   - Compare AVX-512 (16-way) vs AVX2 (8-way) vs scalar
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
float reduce_sum_scalar(const float *a, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += a[i];
    return sum;
}

/* ---- AVX2 (8-way) ---- */
float reduce_sum_avx2(const float *a, int n) {
    __m256 vsum0 = _mm256_setzero_ps();
    __m256 vsum1 = _mm256_setzero_ps();
    __m256 vsum2 = _mm256_setzero_ps();
    __m256 vsum3 = _mm256_setzero_ps();

    int i = 0;
    for (; i + 31 < n; i += 32) {
        vsum0 = _mm256_add_ps(vsum0, _mm256_loadu_ps(a + i));
        vsum1 = _mm256_add_ps(vsum1, _mm256_loadu_ps(a + i + 8));
        vsum2 = _mm256_add_ps(vsum2, _mm256_loadu_ps(a + i + 16));
        vsum3 = _mm256_add_ps(vsum3, _mm256_loadu_ps(a + i + 24));
    }
    for (; i + 7 < n; i += 8)
        vsum0 = _mm256_add_ps(vsum0, _mm256_loadu_ps(a + i));

    vsum0 = _mm256_add_ps(vsum0, vsum1);
    vsum0 = _mm256_add_ps(vsum0, vsum2);
    vsum0 = _mm256_add_ps(vsum0, vsum3);

    __m128 lo = _mm256_castps256_ps128(vsum0);
    __m128 hi = _mm256_extractf128_ps(vsum0, 1);
    __m128 s128 = _mm_add_ps(lo, hi);
    s128 = _mm_hadd_ps(s128, s128);
    s128 = _mm_hadd_ps(s128, s128);
    float result = _mm_cvtss_f32(s128);

    for (; i < n; i++) result += a[i];
    return result;
}

/* ---- AVX-512: _mm512_reduce_add_ps ---- */
float reduce_sum_avx512_reduce(const float *a, int n) {
    __m512 vsum = _mm512_setzero_ps();
    __m512 vsum1 = _mm512_setzero_ps();
    __m512 vsum2 = _mm512_setzero_ps();
    __m512 vsum3 = _mm512_setzero_ps();

    int i = 0;
    for (; i + 63 < n; i += 64) {
        vsum  = _mm512_add_ps(vsum,  _mm512_loadu_ps(a + i));
        vsum1 = _mm512_add_ps(vsum1, _mm512_loadu_ps(a + i + 16));
        vsum2 = _mm512_add_ps(vsum2, _mm512_loadu_ps(a + i + 32));
        vsum3 = _mm512_add_ps(vsum3, _mm512_loadu_ps(a + i + 48));
    }
    for (; i + 15 < n; i += 16)
        vsum = _mm512_add_ps(vsum, _mm512_loadu_ps(a + i));

    vsum = _mm512_add_ps(vsum, vsum1);
    vsum = _mm512_add_ps(vsum, vsum2);
    vsum = _mm512_add_ps(vsum, vsum3);

    float result = _mm512_reduce_add_ps(vsum);

    for (; i < n; i++) result += a[i];
    return result;
}

/* ---- AVX-512: manual reduce via permute+add ---- */
float reduce_sum_avx512_manual(const float *a, int n) {
    __m512 vsum = _mm512_setzero_ps();
    __m512 vsum1 = _mm512_setzero_ps();
    __m512 vsum2 = _mm512_setzero_ps();
    __m512 vsum3 = _mm512_setzero_ps();

    int i = 0;
    for (; i + 63 < n; i += 64) {
        vsum  = _mm512_add_ps(vsum,  _mm512_loadu_ps(a + i));
        vsum1 = _mm512_add_ps(vsum1, _mm512_loadu_ps(a + i + 16));
        vsum2 = _mm512_add_ps(vsum2, _mm512_loadu_ps(a + i + 32));
        vsum3 = _mm512_add_ps(vsum3, _mm512_loadu_ps(a + i + 48));
    }
    for (; i + 15 < n; i += 16)
        vsum = _mm512_add_ps(vsum, _mm512_loadu_ps(a + i));

    vsum = _mm512_add_ps(vsum, vsum1);
    vsum = _mm512_add_ps(vsum, vsum2);
    vsum = _mm512_add_ps(vsum, vsum3);

    /* Manual horizontal reduce of a 512-bit register (16 floats) */
    /* Step 1: shuffle high 256 to low 256, add */
    __m256 lo256 = _mm512_castps512_ps256(vsum);
    __m256 hi256 = _mm512_extractf32x8_ps(vsum, 1);
    __m256 sum256 = _mm256_add_ps(lo256, hi256);

    /* Step 2: 128-bit shuffles */
    __m128 lo128 = _mm256_castps256_ps128(sum256);
    __m128 hi128 = _mm256_extractf128_ps(sum256, 1);
    __m128 sum128 = _mm_add_ps(lo128, hi128);

    /* Step 3: hadd (2+2, then 1+1) */
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float result = _mm_cvtss_f32(sum128);

    for (; i < n; i++) result += a[i];
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

    printf("=== AVX-512 Reduction Sum ===\n");
    printf("N = %d\n", N);
    printf("Widths: scalar=1, AVX2=8, AVX-512=16 f32\n\n");

    if (!cpu_has_avx512f()) {
        printf("AVX-512F not available on this CPU.\n");
        printf("Compile with -mavx512f; run on AVX-512 capable hardware.\n");
        printf("(Try: Xeon Scalable, AMD Zen4 Epyc, or emulation)\n");
        return 0;
    }

    printf("AVX-512F: YES\n\n");

    float *a = (float*)aligned_alloc(64, N * sizeof(float));
    for (int i = 0; i < N; i++)
        a[i] = (float)(i % 1000) * 0.001f;

    /* Verification */
    float ref = reduce_sum_scalar(a, N);
    printf("Reference result: %.6f\n\n", ref);

    printf("AVX2 (4-acc): ");
    float r_avx2 = reduce_sum_avx2(a, N);
    verify(r_avx2, ref, 1e-3f);

    printf("AVX-512 reduce_add: ");
    float r_red = reduce_sum_avx512_reduce(a, N);
    verify(r_red, ref, 1e-3f);

    printf("AVX-512 manual: ");
    float r_man = reduce_sum_avx512_manual(a, N);
    verify(r_man, ref, 1e-3f);

    /* Benchmark */
    int iters = 500;
    printf("\nPerformance (%d iterations):\n", iters);

    volatile float sink;

    double t0 = get_time_sec();
    for (int k = 0; k < iters; k++) sink = reduce_sum_scalar(a, N);
    double t_scalar = (get_time_sec() - t0) / iters;

    double t1 = get_time_sec();
    for (int k = 0; k < iters; k++) sink = reduce_sum_avx2(a, N);
    double t_avx2 = (get_time_sec() - t1) / iters;

    double t2 = get_time_sec();
    for (int k = 0; k < iters; k++) sink = reduce_sum_avx512_reduce(a, N);
    double t_512r = (get_time_sec() - t2) / iters;

    double t3 = get_time_sec();
    for (int k = 0; k < iters; k++) sink = reduce_sum_avx512_manual(a, N);
    double t_512m = (get_time_sec() - t3) / iters;

    printf("  Scalar:               %7.1f us\n", t_scalar * 1e6);
    printf("  AVX2 (8-way 4-acc):   %7.1f us  (%.2fx vs scalar)\n",
           t_avx2 * 1e6, t_scalar / t_avx2);
    printf("  AVX-512 reduce_add:   %7.1f us  (%.2fx vs scalar, %.2fx vs AVX2)\n",
           t_512r * 1e6, t_scalar / t_512r, t_avx2 / t_512r);
    printf("  AVX-512 manual:       %7.1f us  (%.2fx vs scalar, %.2fx vs AVX2)\n",
           t_512m * 1e6, t_scalar / t_512m, t_avx2 / t_512m);

    printf("\n  Throughput (AVX-512): %.2f GB/s (read bandwidth)\n",
           (double)N * sizeof(float) / t_512r / 1e9);

    printf("\n--- _mm512_reduce_add_ps ---\n");
    printf("This intrinsic hides the horizontal reduction:\n");
    printf("  val = _mm512_reduce_add_ps(v);\n");
    printf("It expands to ~8 instructions: shuffle+add cascade.\n");
    printf("Manual reduction shows the actual pattern for learning purposes.\n");

    free(a);
    return 0;
}
