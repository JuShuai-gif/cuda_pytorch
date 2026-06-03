/*
 * avx2_vector_add.cpp -- AVX2 vector addition (f32, i32, aligned vs unaligned)
 *
 * SIMD width: 256-bit = 8x f32 or 8x i32 per register
 * N = 1000003 (prime, to expose tail handling)
 *
 * Variants:
 *   scalar_f32, scalar_i32       -- scalar baselines
 *   avx2_f32_unaligned           -- _mm256_loadu_ps / _mm256_storeu_ps
 *   avx2_f32_aligned             -- _mm256_load_ps  / _mm256_store_ps
 *   avx2_i32                     -- _mm256_add_epi32
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include "../../common/benchmark.h"
#include "../../common/check.h"
#include "../../common/random_data.h"
#include "../../common/aligned_buffer.h"
#include "../../common/cpu_features.h"

static const size_t N = 1000003;

/* ================================================================
 * Scalar baselines
 * ================================================================ */

static void scalar_add_f32(const float* a, const float* b, float* c, size_t n) {
    for (size_t i = 0; i < n; i++) c[i] = a[i] + b[i];
}

static void scalar_add_i32(const int32_t* a, const int32_t* b, int32_t* c, size_t n) {
    for (size_t i = 0; i < n; i++) c[i] = a[i] + b[i];
}

/* ================================================================
 * AVX2 f32 -- unaligned (modern default)
 * ================================================================ */

static void avx2_add_f32_unaligned(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0;
    /* Main loop: 8 floats per iteration */
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(c + i, _mm256_add_ps(va, vb));
    }
    /* Scalar tail */
    for (; i < n; i++) c[i] = a[i] + b[i];
}

/* ================================================================
 * AVX2 f32 -- aligned (requires 32-byte aligned buffers)
 * ================================================================ */

static void avx2_add_f32_aligned(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_load_ps(a + i);
        __m256 vb = _mm256_load_ps(b + i);
        _mm256_store_ps(c + i, _mm256_add_ps(va, vb));
    }
    for (; i < n; i++) c[i] = a[i] + b[i];
}

/* ================================================================
 * AVX2 i32
 * ================================================================ */

static void avx2_add_i32(const int32_t* a, const int32_t* b, int32_t* c, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        _mm256_storeu_si256((__m256i*)(c + i), _mm256_add_epi32(va, vb));
    }
    for (; i < n; i++) c[i] = a[i] + b[i];
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

    printf("\n=== AVX2 Vector Add (N = %zu) ===\n\n", N);

    /* Allocate buffers */
    float* a_f32  = ALIGNED_ALLOC(float, N, 32);
    float* b_f32  = ALIGNED_ALLOC(float, N, 32);
    float* c_ref  = ALIGNED_ALLOC(float, N, 32);
    float* c_unal = ALIGNED_ALLOC(float, N, 32);
    float* c_aln  = ALIGNED_ALLOC(float, N, 32);

    int32_t* a_i32  = ALIGNED_ALLOC(int32_t, N, 32);
    int32_t* b_i32  = ALIGNED_ALLOC(int32_t, N, 32);
    int32_t* c_i32_ref = ALIGNED_ALLOC(int32_t, N, 32);
    int32_t* c_i32_avx = ALIGNED_ALLOC(int32_t, N, 32);

    if (!a_f32 || !b_f32 || !c_ref || !c_unal || !c_aln ||
        !a_i32 || !b_i32 || !c_i32_ref || !c_i32_avx) {
        fprintf(stderr, "Allocation failed.\n");
        return 1;
    }

    /* Fill input data */
    rand_xorshift64_seed(42);
    fill_random_f32(a_f32, N);
    rand_xorshift64_seed(99);
    fill_random_f32(b_f32, N);

    rand_xorshift64_seed(42);
    fill_random_i32(a_i32, N);
    rand_xorshift64_seed(99);
    fill_random_i32(b_i32, N);

    /* Verify alignment */
    CHECK_TRUE(is_aligned(a_f32, 32),  "a_f32 is 32-byte aligned");
    CHECK_TRUE(is_aligned(b_f32, 32),  "b_f32 is 32-byte aligned");
    CHECK_TRUE(is_aligned(c_aln, 32),  "c_aln is 32-byte aligned");

    /* ---- Correctness checks ---- */

    printf("\n--- Correctness: f32 ---\n");

    memset(c_ref,  0, N * sizeof(float));
    memset(c_unal, 0, N * sizeof(float));
    memset(c_aln,  0, N * sizeof(float));

    scalar_add_f32(a_f32, b_f32, c_ref, N);
    avx2_add_f32_unaligned(a_f32, b_f32, c_unal, N);
    avx2_add_f32_aligned(a_f32, b_f32, c_aln, N);

    CHECK_NEAR_ARRAY(c_unal, c_ref, N, 1e-6, "avx2_add_f32_unaligned matches scalar");
    CHECK_NEAR_ARRAY(c_aln,  c_ref, N, 1e-6, "avx2_add_f32_aligned matches scalar");

    printf("\n--- Correctness: i32 ---\n");

    memset(c_i32_ref, 0, N * sizeof(int32_t));
    memset(c_i32_avx, 0, N * sizeof(int32_t));

    scalar_add_i32(a_i32, b_i32, c_i32_ref, N);
    avx2_add_i32(a_i32, b_i32, c_i32_avx, N);

    CHECK_NEAR_ARRAY(c_i32_avx, c_i32_ref, N, 0, "avx2_add_i32 matches scalar");

    /* ---- Benchmark ---- */

    const size_t bytes_f32 = N * 3 * sizeof(float); /* read a + b, write c */

    benchmark_result_t results[6];
    memset(results, 0, sizeof(results));

    BENCH_COMPUTE(scalar_add_f32(a_f32, b_f32, c_ref, N),
                  N, bytes_f32, 20, results[0]);
    results[0].name = "scalar_add_f32";

    BENCH_COMPUTE(avx2_add_f32_unaligned(a_f32, b_f32, c_unal, N),
                  N, bytes_f32, 20, results[1]);
    results[1].name = "avx2_add_f32 (unaligned)";

    BENCH_COMPUTE(avx2_add_f32_aligned(a_f32, b_f32, c_aln, N),
                  N, bytes_f32, 20, results[2]);
    results[2].name = "avx2_add_f32 (aligned)";

    BENCH_COMPUTE(scalar_add_i32(a_i32, b_i32, c_i32_ref, N),
                  N, bytes_f32, 20, results[3]);
    results[3].name = "scalar_add_i32";

    BENCH_COMPUTE(avx2_add_i32(a_i32, b_i32, c_i32_avx, N),
                  N, bytes_f32, 20, results[4]);
    results[4].name = "avx2_add_i32";

    /* Custom speedup for aligned vs unaligned */
    {
        volatile int dummy = 0;
        double t0, t1, best_un, best_al;
        best_un = best_al = 1e18;

        for (int k = 0; k < 20; k++) {
            t0 = get_time_ns();
            avx2_add_f32_unaligned(a_f32, b_f32, c_unal, N);
            t1 = get_time_ns() - t0;
            if (t1 < best_un) best_un = t1;

            t0 = get_time_ns();
            avx2_add_f32_aligned(a_f32, b_f32, c_aln, N);
            t1 = get_time_ns() - t0;
            if (t1 < best_al) best_al = t1;
            (void)dummy;
        }

        results[5].name          = "aligned vs unaligned ratio";
        results[5].elapsed_ns    = 0;
        results[5].ns_per_element = 0;
        results[5].gb_per_sec    = 0;
        results[5].speedup       = (best_un > 0) ? best_un / best_al : 0;
        results[5].iterations    = 20;
        results[5].num_elements  = N;
    }

    printf("\n--- Benchmark Results ---\n");
    printf("SIMD width: 256-bit (8x f32 / 8x i32)\n");
    bench_report(results, 6);

    printf("Notes:\n");
    printf("  - unaligned loads (_mm256_loadu_ps) are the modern default.\n");
    printf("  - aligned loads (_mm256_load_ps) require 32-byte aligned buffers\n");
    printf("    or they will segfault.\n");
    printf("  - On Haswell and later, unaligned load latency is identical to\n");
    printf("    aligned when data IS aligned. The penalty only applies when\n");
    printf("    a load crosses a cache-line boundary (64 bytes).\n");
    printf("  - i32 and f32 addition have the same throughput on AVX2.\n");

    /* Cleanup */
    ALIGNED_FREE(a_f32);
    ALIGNED_FREE(b_f32);
    ALIGNED_FREE(c_ref);
    ALIGNED_FREE(c_unal);
    ALIGNED_FREE(c_aln);
    ALIGNED_FREE(a_i32);
    ALIGNED_FREE(b_i32);
    ALIGNED_FREE(c_i32_ref);
    ALIGNED_FREE(c_i32_avx);

    return 0;
}
