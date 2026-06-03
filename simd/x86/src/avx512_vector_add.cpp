/**
 * AVX-512 Vector Add with Masked Tail
 *
 * Demonstrates:
 *   - _mm512_add_ps: 16 f32 per register
 *   - Masked tail: __mmask16 mask = (1 << (N % 16)) - 1
 *   - _mm512_mask_add_ps
 *   - Compare masked tail vs scalar tail
 *   - N = 1000003 (prime, forces non-trivial tail)
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

/* ---- CPU feature detection ---- */
static int cpu_has_avx512f() {
#ifndef __linux__
    return 0;
#else
    unsigned int eax, ebx, ecx, edx;
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return 0;
    return (ebx & (1 << 16)) != 0;  /* AVX-512F */
#endif
}

static int cpu_has_avx512bw() {
#ifndef __linux__
    return 0;
#else
    unsigned int eax, ebx, ecx, edx;
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return 0;
    return (ebx & (1 << 30)) != 0;  /* AVX-512BW */
#endif
}

static int cpu_has_avx512dq() {
#ifndef __linux__
    return 0;
#else
    unsigned int eax, ebx, ecx, edx;
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return 0;
    return (ebx & (1 << 17)) != 0;  /* AVX-512DQ */
#endif
}

static int cpu_has_avx512vl() {
#ifndef __linux__
    return 0;
#else
    unsigned int eax, ebx, ecx, edx;
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return 0;
    return (ebx & (1 << 31)) != 0;  /* AVX-512VL */
#endif
}

static void print_cpu_features() {
    printf("CPU AVX-512 features:\n");
    printf("  AVX-512F:  %s\n", cpu_has_avx512f()  ? "YES" : "NO");
    printf("  AVX-512BW: %s\n", cpu_has_avx512bw() ? "YES" : "NO");
    printf("  AVX-512DQ: %s\n", cpu_has_avx512dq() ? "YES" : "NO");
    printf("  AVX-512VL: %s\n", cpu_has_avx512vl() ? "YES" : "NO");
}

/* ---- Scalar baseline ---- */
void vadd_scalar(const float *a, const float *b, float *c, int n) {
    for (int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

/* ---- AVX-512 with scalar tail ---- */
void vadd_avx512_scalar_tail(const float *a, const float *b, float *c, int n) {
    int i = 0;
    for (; i + 15 < n; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        _mm512_storeu_ps(c + i, _mm512_add_ps(va, vb));
    }
    for (; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

/* ---- AVX-512 with masked tail ---- */
void vadd_avx512_masked_tail(const float *a, const float *b, float *c, int n) {
    int i = 0;
    for (; i + 15 < n; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        _mm512_storeu_ps(c + i, _mm512_add_ps(va, vb));
    }
    /* Masked tail */
    int remainder = n - i;
    if (remainder > 0) {
        __mmask16 mask = (1U << remainder) - 1;
        __m512 va = _mm512_maskz_loadu_ps(mask, a + i);
        __m512 vb = _mm512_maskz_loadu_ps(mask, b + i);
        __m512 vc = _mm512_add_ps(va, vb);
        _mm512_mask_storeu_ps(c + i, mask, vc);
    }
}

/* ---- AVX-512 with mask_add (single instruction for tail) ---- */
void vadd_avx512_mask_add(const float *a, const float *b, float *c, int n) {
    int i = 0;
    for (; i + 15 < n; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        _mm512_storeu_ps(c + i, _mm512_add_ps(va, vb));
    }
    int remainder = n - i;
    if (remainder > 0) {
        __mmask16 mask = (1U << remainder) - 1;
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        /* _mm512_mask_add_ps: c = (mask) ? a+b : c */
        __m512 vc = _mm512_mask_add_ps(vb, mask, va, vb);
        _mm512_mask_storeu_ps(c + i, mask, vc);
    }
}

/* ---- Verification ---- */
static int verify(const float *got, const float *ref, int n) {
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float err = fabsf(got[i] - ref[i]);
        if (err > max_err) max_err = err;
        if (err > 1e-7f) {
            fprintf(stderr, "mismatch[%d]: %f vs %f\n", i, got[i], ref[i]);
            return 0;
        }
    }
    return 1;
}

/* ---- Main ---- */
int main() {
    const int N = 1000003;

    printf("=== AVX-512 Vector Add with Masked Tail ===\n");
    printf("N = %d (prime, remainder = %d)\n", N, N % 16);
    printf("SIMD width = 512 bits (16 f32 per register)\n\n");

    print_cpu_features();
    printf("\n");

    if (!cpu_has_avx512f()) {
        printf("AVX-512F not available. Compile with -mavx512f for supported CPU.\n");
        printf("This CPU lacks AVX-512; code will not execute.\n");
        printf("(Intel 12th/13th gen consumer CPUs have AVX-512 fused off.)\n");
        printf("For AVX-512, use server Xeon or AMD Zen4+ (with proper flags).\n");
        return 0;
    }

    float *a = (float*)aligned_alloc(64, N * sizeof(float));
    float *b = (float*)aligned_alloc(64, N * sizeof(float));
    float *ref = (float*)aligned_alloc(64, N * sizeof(float));
    float *out = (float*)aligned_alloc(64, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        a[i] = (float)(i % 1000);
        b[i] = (float)((i * 3) % 1000);
    }

    vadd_scalar(a, b, ref, N);

    /* Verify scalar tail approach */
    memset(out, 0, N * sizeof(float));
    vadd_avx512_scalar_tail(a, b, out, N);
    printf("Scalar tail check:  %s\n", verify(out, ref, N) ? "OK" : "FAIL");

    /* Verify masked tail approach */
    memset(out, 0, N * sizeof(float));
    vadd_avx512_masked_tail(a, b, out, N);
    printf("Masked tail check:  %s\n", verify(out, ref, N) ? "OK" : "FAIL");

    /* Verify mask_add approach */
    memset(out, 0, N * sizeof(float));
    vadd_avx512_mask_add(a, b, out, N);
    printf("Mask add check:     %s\n", verify(out, ref, N) ? "OK" : "FAIL");

    /* Benchmark */
    int iters = 500;
    printf("\nPerformance (%d iterations):\n", iters);

    double t0 = get_time_sec();
    for (int k = 0; k < iters; k++) vadd_scalar(a, b, out, N);
    double t_scalar = (get_time_sec() - t0) / iters;

    double t1 = get_time_sec();
    for (int k = 0; k < iters; k++) vadd_avx512_scalar_tail(a, b, out, N);
    double t_stail = (get_time_sec() - t1) / iters;

    double t2 = get_time_sec();
    for (int k = 0; k < iters; k++) vadd_avx512_masked_tail(a, b, out, N);
    double t_mtail = (get_time_sec() - t2) / iters;

    double t3 = get_time_sec();
    for (int k = 0; k < iters; k++) vadd_avx512_mask_add(a, b, out, N);
    double t_madd = (get_time_sec() - t3) / iters;

    printf("  Scalar:              %7.1f us  (%.2f GB/s)\n",
           t_scalar * 1e6, (3.0 * N * sizeof(float)) / t_scalar / 1e9);
    printf("  AVX-512 scalar tail: %7.1f us  (%.2fx speedup, %.2f GB/s)\n",
           t_stail * 1e6, t_scalar / t_stail, (3.0 * N * sizeof(float)) / t_stail / 1e9);
    printf("  AVX-512 masked tail: %7.1f us  (%.2fx speedup, %.2f GB/s)\n",
           t_mtail * 1e6, t_scalar / t_mtail, (3.0 * N * sizeof(float)) / t_mtail / 1e9);
    printf("  AVX-512 mask_add:    %7.1f us  (%.2fx speedup, %.2f GB/s)\n",
           t_madd * 1e6, t_scalar / t_madd, (3.0 * N * sizeof(float)) / t_madd / 1e9);

    printf("\n--- Masked Tail Explanation ---\n");
    printf("Traditional:      for(i=0; i<n%%16; i++) out[base+i] = ...\n");
    printf("AVX-512 masked:   __mmask16 mask = (1<<rem)-1;\n");
    printf("                  _mm512_mask_add_ps(dst, mask, a, b);\n");
    printf("                  _mm512_mask_storeu_ps(out, mask, v);\n");
    printf("Benefits: no branches, no scalar code, compact.\n");

    free(a); free(b); free(ref); free(out);
    return 0;
}
