/**
 * AVX-512 Gather and Scatter Operations
 *
 * Demonstrates:
 *   - _mm512_i32gather_ps: out[i] = table[index[i]]
 *   - _mm512_i32scatter_ps: table[index[i]] = in[i]
 *   - Comparison with manual scalar loop
 *   - Why gather is slow (micro-coded, not true vector load)
 *   - Use case: sparse embedding lookup in recommendation systems
 *   - N = 100000
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

#define TABLE_SIZE 100000

/* ======================================================================== */
/*  Scalar gather                                                           */
/* ======================================================================== */
void gather_scalar(const float *table, const int32_t *indices, float *out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = table[indices[i]];
    }
}

/* ======================================================================== */
/*  Scalar scatter                                                          */
/* ======================================================================== */
void scatter_scalar(float *table, const int32_t *indices, const float *in, int n) {
    for (int i = 0; i < n; i++) {
        table[indices[i]] = in[i];
    }
}

/* ======================================================================== */
/*  AVX-512 gather: _mm512_i32gather_ps                                     */
/* ======================================================================== */
void gather_avx512(const float *table, const int32_t *indices, float *out, int n) {
    int i = 0;
    for (; i + 15 < n; i += 16) {
        __m512i v_idx = _mm512_loadu_si512((const __m512i*)(indices + i));
        __m512 v_out = _mm512_i32gather_ps(v_idx, table, 4);
        _mm512_storeu_ps(out + i, v_out);
    }
    for (; i < n; i++) {
        out[i] = table[indices[i]];
    }
}

/* ======================================================================== */
/*  AVX-512 scatter: _mm512_i32scatter_ps                                   */
/* ======================================================================== */
void scatter_avx512(float *table, const int32_t *indices, const float *in, int n) {
    int i = 0;
    for (; i + 15 < n; i += 16) {
        __m512i v_idx = _mm512_loadu_si512((const __m512i*)(indices + i));
        __m512 v_in = _mm512_loadu_ps(in + i);
        _mm512_i32scatter_ps(table, v_idx, v_in, 4);
    }
    for (; i < n; i++) {
        table[indices[i]] = in[i];
    }
}

/* ======================================================================== */
/*  AVX-512 gather with clustered indices (cache-friendly)                  */
/* ======================================================================== */
void gather_avx512_clustered(const float *table, const int32_t *indices,
                              float *out, int n) {
    int i = 0;
    for (; i + 15 < n; i += 16) {
        __m512i v_idx = _mm512_loadu_si512((const __m512i*)(indices + i));
        __m512 v_out = _mm512_i32gather_ps(v_idx, table, 4);
        _mm512_storeu_ps(out + i, v_out);
    }
    for (; i < n; i++) {
        out[i] = table[indices[i]];
    }
}

/* ======================================================================== */
/*  Verification                                                            */
/* ======================================================================== */
static int verify_gather(const float *got, const int32_t *indices,
                          const float *table, int n) {
    int errs = 0;
    for (int i = 0; i < n; i++) {
        float expected = table[indices[i]];
        if (fabsf(got[i] - expected) > 1e-7f) {
            if (errs < 3)
                fprintf(stderr, "gather[%d]: got=%f ref=%f idx=%d\n",
                        i, got[i], expected, indices[i]);
            errs++;
        }
    }
    printf("  errors=%d %s\n", errs, errs == 0 ? "OK" : "FAIL");
    return errs == 0;
}

/* ======================================================================== */
/*  Main                                                                    */
/* ======================================================================== */
int main() {
    const int N = 100000;

    printf("=== AVX-512 Gather and Scatter ===\n");
    printf("N = %d lookups\n", N);
    printf("Table size = %d\n", TABLE_SIZE);
    printf("SIMD width = 512 bits (16 f32 per register)\n\n");

    if (!cpu_has_avx512f()) {
        printf("AVX-512F not available on this CPU.\n");
        printf("Compile with -mavx512f; run on AVX-512 capable hardware.\n");
        return 0;
    }

    printf("AVX-512F: YES\n\n");

    /* Allocate table (embedding table equivalent) */
    float *table = (float*)aligned_alloc(64, TABLE_SIZE * sizeof(float));
    for (int i = 0; i < TABLE_SIZE; i++)
        table[i] = (float)(i * 0.5f);

    /* --- Test 1: Random indices (worst case for gather) --- */
    printf("--- Test 1: Random indices (worst case) ---\n");
    int32_t *indices_rand = (int32_t*)aligned_alloc(64, N * sizeof(int32_t));
    float *out_r = (float*)aligned_alloc(64, N * sizeof(float));
    float *ref_r = (float*)aligned_alloc(64, N * sizeof(float));

    srand(12345);
    for (int i = 0; i < N; i++)
        indices_rand[i] = rand() % TABLE_SIZE;

    /* Reference */
    gather_scalar(table, indices_rand, ref_r, N);

    /* AVX-512 gather */
    memset(out_r, 0, N * sizeof(float));
    gather_avx512(table, indices_rand, out_r, N);
    printf("AVX-512 gather random: ");
    verify_gather(out_r, indices_rand, table, N);

    /* --- Test 2: Clustered/sequential indices (best case) --- */
    printf("\n--- Test 2: Clustered indices (cache-friendly) ---\n");
    int32_t *indices_clust = (int32_t*)aligned_alloc(64, N * sizeof(int32_t));
    float *out_c = (float*)aligned_alloc(64, N * sizeof(float));
    float *ref_c = (float*)aligned_alloc(64, N * sizeof(float));

    for (int i = 0; i < N; i++)
        indices_clust[i] = (i / 16) * 16;  /* 16 consecutive same indices */

    gather_scalar(table, indices_clust, ref_c, N);
    memset(out_c, 0, N * sizeof(float));
    gather_avx512_clustered(table, indices_clust, out_c, N);
    printf("AVX-512 gather clustered: ");
    verify_gather(out_c, indices_clust, table, N);

    /* --- Test 3: Scatter --- */
    printf("\n--- Test 3: Scatter ---\n");
    float *table_copy = (float*)aligned_alloc(64, TABLE_SIZE * sizeof(float));
    float *in_data = (float*)aligned_alloc(64, N * sizeof(float));

    for (int i = 0; i < N; i++)
        in_data[i] = (float)(i + 1);

    /* Scalar scatter */
    memcpy(table_copy, table, TABLE_SIZE * sizeof(float));
    scatter_scalar(table_copy, indices_rand, in_data, N);
    /* AVX-512 scatter */
    float *table_avx512 = (float*)aligned_alloc(64, TABLE_SIZE * sizeof(float));
    memcpy(table_avx512, table, TABLE_SIZE * sizeof(float));
    scatter_avx512(table_avx512, indices_rand, in_data, N);

    int scatter_errs = 0;
    for (int i = 0; i < TABLE_SIZE; i++) {
        if (fabsf(table_copy[i] - table_avx512[i]) > 1e-7f) {
            if (scatter_errs < 3)
                fprintf(stderr, "scatter[%d]: scalar=%f avx512=%f\n",
                        i, table_copy[i], table_avx512[i]);
            scatter_errs++;
        }
    }
    printf("  errors=%d %s\n", scatter_errs, scatter_errs == 0 ? "OK" : "FAIL");

    /* --- Benchmark --- */
    printf("\n--- Performance (%d iterations) ---\n", 2000);
    int iters = 2000;

    /* Gather: random */
    double t0 = get_time_sec();
    for (int k = 0; k < iters; k++) gather_scalar(table, indices_rand, out_r, N);
    double t_sg = (get_time_sec() - t0) / iters;

    double t1 = get_time_sec();
    for (int k = 0; k < iters; k++) gather_avx512(table, indices_rand, out_r, N);
    double t_avg = (get_time_sec() - t1) / iters;

    /* Gather: clustered */
    double t2 = get_time_sec();
    for (int k = 0; k < iters; k++) gather_scalar(table, indices_clust, out_c, N);
    double t_sgc = (get_time_sec() - t2) / iters;

    double t3 = get_time_sec();
    for (int k = 0; k < iters; k++) gather_avx512_clustered(table, indices_clust, out_c, N);
    double t_avgc = (get_time_sec() - t3) / iters;

    printf("\nGather (random indices):\n");
    printf("  Scalar:   %7.1f us\n", t_sg * 1e6);
    printf("  AVX-512:  %7.1f us  (%.2fx)\n", t_avg * 1e6, t_sg / t_avg);

    printf("\nGather (clustered indices):\n");
    printf("  Scalar:   %7.1f us\n", t_sgc * 1e6);
    printf("  AVX-512:  %7.1f us  (%.2fx)\n", t_avgc * 1e6, t_sgc / t_avgc);

    /* Scatter benchmark */
    double t4 = get_time_sec();
    for (int k = 0; k < iters; k++) {
        memcpy(table_copy, table, TABLE_SIZE * sizeof(float));
        scatter_scalar(table_copy, indices_rand, in_data, N);
    }
    double t_ss = (get_time_sec() - t4) / iters;

    double t5 = get_time_sec();
    for (int k = 0; k < iters; k++) {
        memcpy(table_avx512, table, TABLE_SIZE * sizeof(float));
        scatter_avx512(table_avx512, indices_rand, in_data, N);
    }
    double t_as = (get_time_sec() - t5) / iters;

    printf("\nScatter:\n");
    printf("  Scalar:   %7.1f us\n", t_ss * 1e6);
    printf("  AVX-512:  %7.1f us  (%.2fx)\n", t_as * 1e6, t_ss / t_as);

    /* --- Explanation --- */
    printf("\n--- Why Gather is Slow ---\n");
    printf("Gather is micro-coded (not a true vector load):\n");
    printf("  1. Each lane generates an independent load request.\n");
    printf("  2. No guaranteed coalescing - each load is a separate cache access.\n");
    printf("  3. 16 independent L1 accesses can take dozens of cycles.\n");
    printf("  4. TLB pressure: 16 entries potentially accessed per instruction.\n\n");

    printf("When is gather beneficial?\n");
    printf("  - Indices are clustered (multiple lanes hit same cache line): 2-4x speedup.\n");
    printf("  - Indices are sequential (same as contiguous load): near-native speed.\n");
    printf("  - Code simplicity: single instruction replaces 16 scalar loads.\n");
    printf("  - Random indices: often NO faster than scalar, sometimes slower.\n\n");

    printf("Use cases:\n");
    printf("  - Sparse embedding lookup in recommendation systems (DLRM, etc.).\n");
    printf("  - Sparse matrix multiply (SpMM) with indexed columns.\n");
    printf("  - Gather from LUT tables in signal processing.\n");
    printf("  - Hash table probing.\n\n");

    printf("Alternatives for random access:\n");
    printf("  - Software prefetch (_mm_prefetch) before gather block.\n");
    printf("  - Reorder indices for cache locality (sort/block).\n");
    printf("  - Use quantization LUT with smaller range (fewer cache lines).\n");

    free(table); free(indices_rand); free(out_r); free(ref_r);
    free(indices_clust); free(out_c); free(ref_c);
    free(table_copy); free(table_avx512); free(in_data);
    return 0;
}
