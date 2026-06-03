/**
 * AVX-512 Masked Tail Handling Comparison
 *
 * Demonstrates 3 approaches for handling non-multiple-of-16 tails:
 *   1. Scalar tail (traditional)
 *   2. Masked operation on full width
 *   3. Zero-padded (allocate extra, pad with zeros)
 *
 * Uses vector add as the example operation.
 * N varies from 1 to 256 (small N stress test).
 * Shows that masked approach has no branches and is cleanest.
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

/* ======================================================================== */
/*  Approach 1: Scalar tail                                                 */
/* ======================================================================== */
void vadd_scalar_tail(const float *a, const float *b, float *c, int n) {
    int i = 0;
    for (; i + 15 < n; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        _mm512_storeu_ps(c + i, _mm512_add_ps(va, vb));
    }
    /* Scalar tail: up to 15 elements in a loop with branch */
    for (int j = i; j < n; j++) {
        c[j] = a[j] + b[j];
    }
}

/* ======================================================================== */
/*  Approach 2: Masked operation (zero mask for non-active lanes)           */
/* ======================================================================== */
void vadd_masked(const float *a, const float *b, float *c, int n) {
    int i = 0;
    for (; i + 15 < n; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        _mm512_storeu_ps(c + i, _mm512_add_ps(va, vb));
    }
    /* Masked tail: single instruction, no loop */
    int rem = n - i;
    if (rem > 0) {
        __mmask16 mask = (1U << rem) - 1;
        __m512 va = _mm512_maskz_loadu_ps(mask, a + i);
        __m512 vb = _mm512_maskz_loadu_ps(mask, b + i);
        __m512 vc = _mm512_add_ps(va, vb);
        _mm512_mask_storeu_ps(c + i, mask, vc);
    }
}

/* ======================================================================== */
/*  Approach 3: Zero-padded (round up allocation, pad with zeros)           */
/* ======================================================================== */
void vadd_zeropad(const float *a, const float *b, float *c, int n) {
    /* Requires a, b, c to be allocated with extra space and zero-padded.
     * For this demo, we zero-pad in-place using a copy. */
    int padded_n = (n + 15) & ~15;

    /* Allocate temporary padded arrays */
    float *ap = (float*)aligned_alloc(64, padded_n * sizeof(float));
    float *bp = (float*)aligned_alloc(64, padded_n * sizeof(float));
    float *cp = (float*)aligned_alloc(64, padded_n * sizeof(float));

    memcpy(ap, a, n * sizeof(float));
    memcpy(bp, b, n * sizeof(float));
    memset(ap + n, 0, (padded_n - n) * sizeof(float));
    memset(bp + n, 0, (padded_n - n) * sizeof(float));

    for (int i = 0; i < padded_n; i += 16) {
        __m512 va = _mm512_load_ps(ap + i);
        __m512 vb = _mm512_load_ps(bp + i);
        _mm512_store_ps(cp + i, _mm512_add_ps(va, vb));
    }

    memcpy(c, cp, n * sizeof(float));

    free(ap); free(bp); free(cp);
}

/* ======================================================================== */
/*  Verification                                                            */
/* ======================================================================== */
static int verify(const float *got, const float *ref, int n) {
    for (int i = 0; i < n; i++) {
        if (fabsf(got[i] - ref[i]) > 1e-7f) {
            fprintf(stderr, "mismatch[%d]: got=%f ref=%f\n", i, got[i], ref[i]);
            return 0;
        }
    }
    return 1;
}

static void vadd_scalar(const float *a, const float *b, float *c, int n) {
    for (int i = 0; i < n; i++) c[i] = a[i] + b[i];
}

/* ======================================================================== */
/*  Code size analysis (approximate instruction count)                      */
/* ======================================================================== */
static void analyze_code_size() {
    printf("\n--- Code Size Comparison (approximate instructions) ---\n");
    printf("Approach         | Main Loop | Tail   | Total (est)\n");
    printf("-----------------|-----------|--------|------------\n");
    printf("Scalar tail      |  3 instr  | 4 instr/ele | O(N/16*3 + 15*4)\n");
    printf("Masked           |  3 instr  | 5 instr | O(N/16*3 + 5)\n");
    printf("Zero-padded      |  3 instr  | 0 (memcpy overhead) | O(padded/16*3)\n");
    printf("\nMasked approach: constant tail cost (no loop, no branching).\n");
    printf("This is ideal for very small N where tail overhead dominates.\n");
}

/* ======================================================================== */
/*  Main                                                                    */
/* ======================================================================== */
int main() {
    printf("=== AVX-512 Masked Tail Handling Comparison ===\n");
    printf("Comparing 3 tail-handling strategies for AVX-512 (16 f32)\n\n");

    if (!cpu_has_avx512f()) {
        printf("AVX-512F not available on this CPU.\n");
        return 0;
    }

    printf("AVX-512F: YES\n\n");

    /* --- Correctness test for various N --- */
    int test_sizes[] = {1, 5, 15, 16, 17, 31, 32, 33, 47, 48, 100, 127, 128, 255, 256};
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);

    printf("--- Correctness Verification (N=1 to 256) ---\n");
    int all_ok = 1;
    int max_n = 256;

    float *a  = (float*)aligned_alloc(64, max_n * sizeof(float));
    float *b  = (float*)aligned_alloc(64, max_n * sizeof(float));
    float *c1 = (float*)aligned_alloc(64, max_n * sizeof(float));
    float *c2 = (float*)aligned_alloc(64, max_n * sizeof(float));
    float *c3 = (float*)aligned_alloc(64, max_n * sizeof(float));
    float *ref_val = (float*)aligned_alloc(64, max_n * sizeof(float));

    for (int ti = 0; ti < num_tests; ti++) {
        int n = test_sizes[ti];

        for (int i = 0; i < n; i++) {
            a[i] = (float)(i + 1);
            b[i] = (float)(i * 2 + 1);
        }

        vadd_scalar(a, b, ref_val, n);

        memset(c1, 0, n * sizeof(float));
        vadd_scalar_tail(a, b, c1, n);
        int ok1 = verify(c1, ref_val, n);

        memset(c2, 0, n * sizeof(float));
        vadd_masked(a, b, c2, n);
        int ok2 = verify(c2, ref_val, n);

        memset(c3, 0, n * sizeof(float));
        vadd_zeropad(a, b, c3, n);
        int ok3 = verify(c3, ref_val, n);

        printf("  N=%3d: scalar_tail=%s  masked=%s  zeropad=%s\n",
               n, ok1 ? "OK" : "XX", ok2 ? "OK" : "XX", ok3 ? "OK" : "XX");
        if (!ok1 || !ok2 || !ok3) all_ok = 0;
    }
    printf("  ALL: %s\n\n", all_ok ? "PASSED" : "FAILED");

    /* --- Micro-benchmark: small N stress test --- */
    printf("--- Small-N Latency (N=1 to 256, %d iterations) ---\n", 500000);
    int iters = 500000;

    for (int ti = 0; ti < num_tests; ti++) {
        int n = test_sizes[ti];

        for (int i = 0; i < n; i++) {
            a[i] = (float)i;
            b[i] = (float)(i * 3);
        }

        double t1 = get_time_sec();
        for (int k = 0; k < iters; k++) vadd_scalar_tail(a, b, c1, n);
        t1 = (get_time_sec() - t1) / iters;

        double t2 = get_time_sec();
        for (int k = 0; k < iters; k++) vadd_masked(a, b, c2, n);
        t2 = (get_time_sec() - t2) / iters;

        double t3 = get_time_sec();
        for (int k = 0; k < iters; k++) vadd_zeropad(a, b, c3, n);
        t3 = (get_time_sec() - t3) / iters;

        double best = t1;
        if (t2 < best) best = t2;
        if (t3 < best) best = t3;

        printf("  N=%3d: scalar_tail=%7.1f ns  masked=%7.1f ns  zeropad=%7.1f ns  "
               "best=%s\n",
               n, t1 * 1e9, t2 * 1e9, t3 * 1e9,
               (t2 == best) ? "masked" : (t1 == best) ? "scalar" : "zeropad");
    }

    /* --- Large N benchmark --- */
    printf("\n--- Large-N Throughput (N=1000000) ---\n");
    int N_big = 1000000;
    float *ab = (float*)aligned_alloc(64, (N_big + 16) * sizeof(float));
    float *bb = (float*)aligned_alloc(64, (N_big + 16) * sizeof(float));
    float *cb = (float*)aligned_alloc(64, (N_big + 16) * sizeof(float));

    for (int i = 0; i < N_big; i++) {
        ab[i] = (float)(i % 1000);
        bb[i] = (float)((i * 3) % 1000);
    }

    iters = 200;
    double ts = get_time_sec();
    for (int k = 0; k < iters; k++) vadd_scalar_tail(ab, bb, cb, N_big);
    double t_st = (get_time_sec() - ts) / iters;

    double tm = get_time_sec();
    for (int k = 0; k < iters; k++) vadd_masked(ab, bb, cb, N_big);
    double t_msk = (get_time_sec() - tm) / iters;

    printf("  Scalar tail: %7.1f us  (%.2f GB/s)\n",
           t_st * 1e6, (3.0 * N_big * sizeof(float)) / t_st / 1e9);
    printf("  Masked:      %7.1f us  (%.2f GB/s)\n",
           t_msk * 1e6, (3.0 * N_big * sizeof(float)) / t_msk / 1e9);
    printf("  Speedup:     %.2fx\n", t_st / t_msk);

    analyze_code_size();

    free(a); free(b); free(c1); free(c2); free(c3); free(ref_val);
    free(ab); free(bb); free(cb);
    return 0;
}
