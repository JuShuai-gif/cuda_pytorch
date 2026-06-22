/*
 * edge_cases_demo.cpp -- Demonstrate edge case testing for AVX2 vector add.
 *
 * Validates that the AVX2 SIMD path produces bit-identical results to the
 * scalar path for edge cases that scalar IEEE 754 hardware handles implicitly:
 *   - NaN propagation
 *   - Inf arithmetic (Inf + finite, Inf - Inf)
 *   - Denormal preservation (FTZ check)
 *   - Zero-length input safety
 *   - Alignment boundary correctness
 *
 * Also benchmarks both paths with regular data vs edge case data to measure
 * whether edge case handling introduces any performance regression.
 *
 * Build:
 *   g++ -O3 -mavx2 -std=c++11 edge_cases_demo.cpp -o edge_cases_demo
 *
 * Dependencies:
 *   ../../common/test_edge.h
 *   ../../common/check.h
 *   ../../common/cpu_features.h
 *   ../../common/benchmark.h
 *   ../../common/random_data.h
 *   ../../common/timer.h
 *   ../../common/aligned_buffer.h
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <immintrin.h>

#include "../../common/test_edge.h"
#include "../../common/check.h"
#include "../../common/cpu_features.h"
#include "../../common/benchmark.h"
#include "../../common/random_data.h"
#include "../../common/timer.h"
#include "../../common/aligned_buffer.h"

/* ================================================================
 * Scalar baseline
 * ================================================================ */

static void scalar_add_f32(const float* a, const float* b, float* c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

/* ================================================================
 * AVX2 vector add -- unaligned (intrinsics)
 * ================================================================ */

static void avx2_add_f32(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0;
    /* Main loop: 8 floats per iteration (256-bit) */
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(c + i, _mm256_add_ps(va, vb));
    }
    /* Scalar tail for remaining elements */
    for (; i < n; i++) {
        c[i] = a[i] + b[i];
    }
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

    printf("\n======================================================\n");
    printf("  AVX2 Vector Add -- Edge Case Demo\n");
    printf("======================================================\n");

    /* ================================================================
     * Phase 0: Normal random data (sanity check)
     * ================================================================ */
    {
        printf("\n--- [0] Normal random data (sanity) ---\n");

        const size_t N = 1000;
        float* a      = ALIGNED_ALLOC(float, N, 32);
        float* b      = ALIGNED_ALLOC(float, N, 32);
        float* c_ref  = ALIGNED_ALLOC(float, N, 32);
        float* c_simd = ALIGNED_ALLOC(float, N, 32);

        if (!a || !b || !c_ref || !c_simd) {
            fprintf(stderr, "Allocation failed.\n");
            return 1;
        }

        rand_xorshift64_seed(123);
        fill_random_f32(a, N);
        rand_xorshift64_seed(456);
        fill_random_f32(b, N);

        scalar_add_f32(a, b, c_ref, N);
        avx2_add_f32(a, b, c_simd, N);

        CHECK_NEAR_ARRAY(c_simd, c_ref, N, 1e-6f,
                         "AVX2 matches scalar on random data");

        ALIGNED_FREE(a);
        ALIGNED_FREE(b);
        ALIGNED_FREE(c_ref);
        ALIGNED_FREE(c_simd);
    }

    /* ================================================================
     * Phases 1-5: Edge case tests (via test_edge.h runner)
     * ================================================================ */
    int edge_failures = run_edge_tests("avx2_add_f32", scalar_add_f32, avx2_add_f32);

    /* ================================================================
     * Phase 6: Manual alignment stress test with shifted pointer
     *
     * Allocate a buffer slightly larger than needed, then use an
     * interior pointer at a misaligned offset.  This tests whether
     * _mm256_loadu_ps handles misaligned data correctly at the
     * tail boundary.
     * ================================================================ */
    printf("\n--- [6] Misaligned pointer test ---\n");
    {
        /*
         * Allocate raw buffer with extra space so we can offset the
         * pointer without going out of bounds.
         */
        const size_t N = 257;               /* odd: not a multiple of 8 */
        const size_t extra = 8;             /* headroom for pointer shifting */
        const size_t total_elems = N + extra;

        float* raw_a = ALIGNED_ALLOC(float, total_elems, 64);
        float* raw_b = ALIGNED_ALLOC(float, total_elems, 64);
        float* c_ref  = ALIGNED_ALLOC(float, N, 32);
        float* c_simd = ALIGNED_ALLOC(float, N, 32);

        if (!raw_a || !raw_b || !c_ref || !c_simd) {
            fprintf(stderr, "Allocation failed.\n");
            return 1;
        }

        /*
         * Fill all memory including the "extra" region with a known
         * pattern so that any over-read is detectable.
         */
        rand_xorshift64_seed(789);
        for (size_t i = 0; i < total_elems; i++) raw_a[i] = 1.0f;
        rand_xorshift64_seed(321);
        for (size_t i = 0; i < total_elems; i++) raw_b[i] = 2.0f;

        /* Now test at every possible offset 0..7 in the first 8-byte group */
        int all_offset_pass = 1;
        for (size_t offset = 0; offset < 8; offset++) {
            const float* a_shifted = raw_a + offset;
            const float* b_shifted = raw_b + offset;

            memset(c_ref,  0, N * sizeof(float));
            memset(c_simd, 0, N * sizeof(float));

            scalar_add_f32(a_shifted, b_shifted, c_ref, N);
            avx2_add_f32(a_shifted, b_shifted, c_simd, N);

            for (size_t i = 0; i < N; i++) {
                if (!float_bit_equal(c_ref[i], c_simd[i])) {
                    fprintf(stderr, "  Misaligned fail: offset=%zu, i=%zu: "
                            "ref=%e (0x%08X) simd=%e (0x%08X)\n",
                            offset, i,
                            (double)c_ref[i],  bits_from_float(c_ref[i]),
                            (double)c_simd[i], bits_from_float(c_simd[i]));
                    all_offset_pass = 0;
                    break;
                }
            }
            if (!all_offset_pass) break;
        }

        if (all_offset_pass) {
            printf("  [PASS] Misaligned pointers (offsets 0..7)\n");
        } else {
            edge_failures++;
            printf("  [FAIL] Misaligned pointers\n");
        }

        ALIGNED_FREE(raw_a);
        ALIGNED_FREE(raw_b);
        ALIGNED_FREE(c_ref);
        ALIGNED_FREE(c_simd);
    }

    /* ================================================================
     * Phase 7: Benchmark -- regular data vs edge case data
     *
     * Compares performance of the AVX2 path on normal data vs data
     * full of NaNs/Infs/denorms.  Some CPUs have slower NaN/denorm
     * paths in the FPU (microcode assists), which can affect SIMD
     * throughput.
     * ================================================================ */
    {
        printf("\n--- [7] Benchmark: regular vs edge-case data ---\n");

        const size_t N_BENCH = 1000003;  /* prime, to expose tail handling */
        const int    ITERS   = 50;

        float* a    = ALIGNED_ALLOC(float, N_BENCH, 32);
        float* b    = ALIGNED_ALLOC(float, N_BENCH, 32);
        float* c_out = ALIGNED_ALLOC(float, N_BENCH, 32);

        if (!a || !b || !c_out) {
            fprintf(stderr, "Benchmark allocation failed.\n");
            return 1;
        }

        const size_t bytes_total = N_BENCH * 3 * sizeof(float);

        /*
         * Benchmark 1: regular random data
         */
        printf("\n  -- Regular random data --\n");
        rand_xorshift64_seed(555);
        fill_random_f32(a, N_BENCH);
        rand_xorshift64_seed(666);
        fill_random_f32(b, N_BENCH);

        {
            benchmark_result_t r_scalar, r_simd;

            BENCH_COMPUTE(scalar_add_f32(a, b, c_out, N_BENCH),
                          N_BENCH, bytes_total, ITERS, r_scalar);
            r_scalar.name = "scalar (regular data)";

            BENCH_COMPUTE(avx2_add_f32(a, b, c_out, N_BENCH),
                          N_BENCH, bytes_total, ITERS, r_simd);
            r_simd.name = "AVX2 (regular data)";

            benchmark_result_t results[2] = { r_scalar, r_simd };
            bench_report(results, 2);
        }

        /*
         * Benchmark 2: edge case data (NaN, Inf, denorm)
         */
        printf("\n  -- Edge case data (NaN / Inf / denorm) --\n");
        fill_edge_f32(a, N_BENCH);
        fill_edge_f32(b, N_BENCH);

        {
            benchmark_result_t r_scalar_edge, r_simd_edge;

            BENCH_COMPUTE(scalar_add_f32(a, b, c_out, N_BENCH),
                          N_BENCH, bytes_total, ITERS, r_scalar_edge);
            r_scalar_edge.name = "scalar (edge data)";

            BENCH_COMPUTE(avx2_add_f32(a, b, c_out, N_BENCH),
                          N_BENCH, bytes_total, ITERS, r_simd_edge);
            r_simd_edge.name = "AVX2 (edge data)";

            benchmark_result_t results_edge[2] = { r_scalar_edge, r_simd_edge };
            bench_report(results_edge, 2);
        }

        ALIGNED_FREE(a);
        ALIGNED_FREE(b);
        ALIGNED_FREE(c_out);
    }

    /* ================================================================
     * Final report
     * ================================================================ */
    edge_test_report("avx2_add_f32 (with misaligned test)", edge_failures);

    return (edge_failures == 0) ? 0 : 1;
}
