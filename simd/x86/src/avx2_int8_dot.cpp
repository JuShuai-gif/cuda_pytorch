/**
 * AVX2 int8 Dot Product for Quantized ML Inference
 *
 * Demonstrates:
 *   - _mm256_maddubs_epi16: u8 * s8 -> s16 horizontal add pairs to s16
 *   - _mm256_madd_epi16:    s16 * s16 -> s32 (dot with ones vector)
 *   - _mm256_add_epi32:     s32 accumulate across vectors
 *   - Zero-point subtraction for asymmetric quantization
 *   - N = 1000000
 *   - input: uint8_t, weights: int8_t
 *   - Print equivalent FP32 ops per second
 *
 * The dot product S = sum(input[i] * weight[i]) for i=0..N-1.
 *
 * With u8 * s8 via maddubs:
 *   maddubs(a, b) = sum(a_i_lo * b_i_lo, a_i_hi * b_i_hi) into s16 elements
 *   We pair input bytes with int8 weight bytes.
 *
 * For asymmetric zero-point:
 *   input_dequant = (input_raw - input_zero) * scale
 *   weight_dequant = (weight_raw - weight_zero) * scale
 *   We pre-subtract zero-point and accumulate correction terms.
 */

#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../common/aligned_buffer.h"
#include "../../common/benchmark.h"
#include "../../common/check.h"
#include "../../common/cpu_features.h"
#include "../../common/random_data.h"

/* ------------------------------------------------------------------------ */
/*  Scalar int32 dot product (reference)                                    */
/* ------------------------------------------------------------------------ */
__attribute__((noinline))
int32_t dot_int8_scalar(const uint8_t *input, const int8_t *weights, int N) {
    int32_t sum = 0;
    for (int i = 0; i < N; i++) {
        sum += (int32_t)input[i] * (int32_t)weights[i];
    }
    return sum;
}

__attribute__((noinline))
int32_t dot_int8_scalar_zp(const uint8_t *input, const int8_t *weights,
                           uint8_t input_zp, int8_t weight_zp, int N) {
    int32_t sum = 0;
    for (int i = 0; i < N; i++) {
        int32_t a = (int32_t)input[i] - (int32_t)input_zp;
        int32_t b = (int32_t)weights[i] - (int32_t)weight_zp;
        sum += a * b;
    }
    return sum;
}

/* ------------------------------------------------------------------------ */
/*  AVX2 int8 dot product - direct s16 conversion (clear and correct)       */
/*                                                                          */
/*  Use _mm256_cvtepu8_epi16 (16 u8 -> 16 s16) and                         */
/*  _mm256_cvtepi8_epi16 (16 s8 -> 16 s16) to convert lanes.               */
/*  Then madd_epi16 for pairwise multiply-add (16 s16 pairs -> 8 s32).      */
/*  Process 16 elements per inner block with 4 accumulators for ILP.        */
/* ------------------------------------------------------------------------ */
__attribute__((noinline))
int32_t dot_int8_avx2_direct(const uint8_t *input, const int8_t *weights, int N) {
    __m256i vsum0 = _mm256_setzero_si256();
    __m256i vsum1 = _mm256_setzero_si256();
    __m256i vsum2 = _mm256_setzero_si256();
    __m256i vsum3 = _mm256_setzero_si256();

    int i = 0;
    /* Process 64 elements per iteration (4 blocks of 16 u8/s8 each) */
    for (; i + 63 < N; i += 64) {
        for (int b = 0; b < 4; b++) {
            int off = i + b * 16;
            /* Load 16 u8 + 16 s8 */
            __m128i vin128  = _mm_loadu_si128((const __m128i*)(input + off));
            __m128i vwt128  = _mm_loadu_si128((const __m128i*)(weights + off));

            /* Zero-extend u8->s16, sign-extend s8->s16 */
            __m256i vin_s16 = _mm256_cvtepu8_epi16(vin128);
            __m256i vwt_s16 = _mm256_cvtepi8_epi16(vwt128);

            /* madd: pairwise multiply s16*s16 -> s32 horizontal sum */
            __m256i prod = _mm256_madd_epi16(vin_s16, vwt_s16);

            if (b == 0)      vsum0 = _mm256_add_epi32(vsum0, prod);
            else if (b == 1) vsum1 = _mm256_add_epi32(vsum1, prod);
            else if (b == 2) vsum2 = _mm256_add_epi32(vsum2, prod);
            else             vsum3 = _mm256_add_epi32(vsum3, prod);
        }
    }

    /* Handle remaining full blocks of 16 */
    for (; i + 15 < N; i += 16) {
        __m128i vin128  = _mm_loadu_si128((const __m128i*)(input + i));
        __m128i vwt128  = _mm_loadu_si128((const __m128i*)(weights + i));
        __m256i vin_s16 = _mm256_cvtepu8_epi16(vin128);
        __m256i vwt_s16 = _mm256_cvtepi8_epi16(vwt128);
        __m256i prod = _mm256_madd_epi16(vin_s16, vwt_s16);
        vsum0 = _mm256_add_epi32(vsum0, prod);
    }

    /* Reduce accumulators */
    vsum0 = _mm256_add_epi32(vsum0, vsum1);
    vsum0 = _mm256_add_epi32(vsum0, vsum2);
    vsum0 = _mm256_add_epi32(vsum0, vsum3);

    /* Horizontal reduction of 8 int32 lanes:
     * After permute+add, each lane has sum from its 128-bit half.
     * After 2x hadd, all 8 lanes contain the total sum. */
    __m256i perm = _mm256_permute2x128_si256(vsum0, vsum0, 0x01);
    vsum0 = _mm256_add_epi32(vsum0, perm);
    vsum0 = _mm256_hadd_epi32(vsum0, vsum0);
    vsum0 = _mm256_hadd_epi32(vsum0, vsum0);
    int32_t result = _mm_cvtsi128_si32(_mm256_castsi256_si128(vsum0));

    for (; i < N; i++)
        result += (int32_t)input[i] * (int32_t)weights[i];

    return result;
}

/* ------------------------------------------------------------------------ */
/*  AVX2 int8 dot product with zero-point subtraction                       */
/* ------------------------------------------------------------------------ */
__attribute__((noinline))
int32_t dot_int8_avx2_zp(const uint8_t *input, const int8_t *weights,
                         uint8_t input_zp, int8_t weight_zp, int N) {
    __m256i vsum0 = _mm256_setzero_si256();
    __m256i vsum1 = _mm256_setzero_si256();

    __m256i v_in_zp = _mm256_set1_epi16((int16_t)input_zp);
    __m256i v_wt_zp = _mm256_set1_epi16((int16_t)weight_zp);

    int i = 0;
    for (; i + 63 < N; i += 64) {
        for (int b = 0; b < 4; b++) {
            int off = i + b * 16;
            __m128i vin128 = _mm_loadu_si128((const __m128i*)(input + off));
            __m128i vwt128 = _mm_loadu_si128((const __m128i*)(weights + off));

            __m256i vin_s16 = _mm256_cvtepu8_epi16(vin128);
            __m256i vwt_s16 = _mm256_cvtepi8_epi16(vwt128);

            vin_s16 = _mm256_sub_epi16(vin_s16, v_in_zp);
            vwt_s16 = _mm256_sub_epi16(vwt_s16, v_wt_zp);

            __m256i prod = _mm256_madd_epi16(vin_s16, vwt_s16);

            if (b < 2) vsum0 = _mm256_add_epi32(vsum0, prod);
            else       vsum1 = _mm256_add_epi32(vsum1, prod);
        }
    }
    for (; i + 15 < N; i += 16) {
        __m128i vin128 = _mm_loadu_si128((const __m128i*)(input + i));
        __m128i vwt128 = _mm_loadu_si128((const __m128i*)(weights + i));
        __m256i vin_s16 = _mm256_cvtepu8_epi16(vin128);
        __m256i vwt_s16 = _mm256_cvtepi8_epi16(vwt128);
        vin_s16 = _mm256_sub_epi16(vin_s16, v_in_zp);
        vwt_s16 = _mm256_sub_epi16(vwt_s16, v_wt_zp);
        vsum0 = _mm256_add_epi32(vsum0, _mm256_madd_epi16(vin_s16, vwt_s16));
    }

    vsum0 = _mm256_add_epi32(vsum0, vsum1);

    __m256i perm = _mm256_permute2x128_si256(vsum0, vsum0, 0x01);
    vsum0 = _mm256_add_epi32(vsum0, perm);
    vsum0 = _mm256_hadd_epi32(vsum0, vsum0);
    vsum0 = _mm256_hadd_epi32(vsum0, vsum0);
    int32_t result = _mm_cvtsi128_si32(_mm256_castsi256_si128(vsum0));

    for (; i < N; i++) {
        int32_t a = (int32_t)input[i] - (int32_t)input_zp;
        int32_t b = (int32_t)weights[i] - (int32_t)weight_zp;
        result += a * b;
    }

    return result;
}

/* ------------------------------------------------------------------------ */
/*  HIGHLY OPTIMIZED AVX2 int8 dot using maddubs_epi16 + madd_epi16        */
/*                                                                          */
/*  The key chain:                                                          */
/*    maddubs_epi16(u8, s8) -> 16 s16 values (horizontal pair sums)        */
/*    madd_epi16(maddubs_out, ones_vector) -> s32 (sum all s16)            */
/*                                                                          */
/*  This is the standard VPMADDUBSW + VPMADDWD pattern used in production   */
/*  quantized inference (e.g., ARM NEON SMLAL + SADDLP analog).            */
/*                                                                          */
/*  To make maddubs work with unsigned u8: paired with int8_t weights.      */
/*  Each u8 value is paired with its neighbor (pairwise).                   */
/*  maddubs computes: s16[i] = u8[2i]*s8[2i] + u8[2i+1]*s8[2i+1]           */
/*                                                                          */
/*  Then madd_epi16 with {1,1,1,...} sums adjacent s16 pairs to s32.       */
/*  Finally, horizontal reduce the s32 partial sums.                        */
/* ------------------------------------------------------------------------ */
__attribute__((noinline))
int32_t dot_int8_avx2_maddubs(const uint8_t *input, const int8_t *weights, int N) {
    __m256i vsum0 = _mm256_setzero_si256();
    __m256i vsum1 = _mm256_setzero_si256();
    __m256i vsum2 = _mm256_setzero_si256();
    __m256i vsum3 = _mm256_setzero_si256();

    __m256i ones = _mm256_set1_epi16(1);

    int i = 0;
    /* Process 128 elements per iteration (4 registers * 32 bytes) */
    for (; i + 127 < N; i += 128) {
        for (int j = 0; j < 4; j++) {
            __m256i vin = _mm256_load_si256((const __m256i*)(input + i + j * 32));
            __m256i vwt = _mm256_load_si256((const __m256i*)(weights + i + j * 32));

            /* maddubs: pairwise u8*s8 -> s16 (16 elements) */
            __m256i madd = _mm256_maddubs_epi16(vin, vwt);

            /* madd epi16: s16 * ones -> s32 (sum adjacent pairs, 8 elements) */
            __m256i acc = _mm256_madd_epi16(madd, ones);

            if (j == 0) vsum0 = _mm256_add_epi32(vsum0, acc);
            else if (j == 1) vsum1 = _mm256_add_epi32(vsum1, acc);
            else if (j == 2) vsum2 = _mm256_add_epi32(vsum2, acc);
            else vsum3 = _mm256_add_epi32(vsum3, acc);
        }
    }

    /* Handle remaining full vectors */
    for (; i + 31 < N; i += 32) {
        __m256i vin = _mm256_load_si256((const __m256i*)(input + i));
        __m256i vwt = _mm256_load_si256((const __m256i*)(weights + i));
        __m256i madd = _mm256_maddubs_epi16(vin, vwt);
        __m256i acc  = _mm256_madd_epi16(madd, ones);
        vsum0 = _mm256_add_epi32(vsum0, acc);
    }

    /* Reduce accumulators */
    vsum0 = _mm256_add_epi32(vsum0, vsum1);
    vsum0 = _mm256_add_epi32(vsum0, vsum2);
    vsum0 = _mm256_add_epi32(vsum0, vsum3);

    /* Horizontal reduction of 8 int32 lanes:
     * After permute+add, each lane has sum from its 128-bit half.
     * After 2x hadd, all 8 lanes contain the total sum. */
    __m256i perm = _mm256_permute2x128_si256(vsum0, vsum0, 0x01);
    vsum0 = _mm256_add_epi32(vsum0, perm);
    vsum0 = _mm256_hadd_epi32(vsum0, vsum0);
    vsum0 = _mm256_hadd_epi32(vsum0, vsum0);
    int32_t result = _mm_cvtsi128_si32(_mm256_castsi256_si128(vsum0));

    /* Scalar tail */
    for (; i < N; i++) {
        result += (int32_t)input[i] * (int32_t)weights[i];
    }

    return result;
}

/* ------------------------------------------------------------------------ */
/*  Main                                                                    */
/* ------------------------------------------------------------------------ */
int main(void) {
    const int N = 1000000;

    cpu_print_features();

    if (!cpu_has_avx2()) {
        printf("\nAVX2 not supported on this CPU. Exiting.\n");
        return 0;
    }
    printf("\n");

    printf("=== AVX2 int8 Dot Product for Quantized ML ===\n");
    printf("N = %d\n", N);
    printf("SIMD width = 256 bits (8 f32, 32 u8/s8 per register)\n");
    printf("\nVPMADDUBSW+VPMADDWD production int8 dot pattern:\n");
    printf("  _mm256_maddubs_epi16: 32 u8 * 32 s8 -> 16 s16 partials\n");
    printf("  _mm256_madd_epi16:    s16 * {1,1,...} -> 8 s32 accumulators\n");
    printf("  _mm256_add_epi32:     accumulate s32 across vectors\n\n");

    /* Allocate aligned memory */
    uint8_t *input   = ALIGNED_ALLOC(uint8_t, N, 32);
    int8_t  *weights = ALIGNED_ALLOC(int8_t, N, 32);

    /* Fill with deterministic random data.
     * IMPORTANT: maddubs_epi16 sums 2 adjacent u8*s8 intermediate products
     * into a single s16. This s16 can overflow if products are too large.
     * Max safe: |u8*s8[i] + u8*s8[i+1]| <= 32767.
     * We limit weights to +/-63 so 255*63*2 = 32130 < 32767. */
    rand_xorshift64_seed(42);
    fill_random_u8(input, N);
    fill_random_i8(weights, N);
    /* Constrain weights to [-64, 63] to prevent maddubs intermediate overflow */
    for (int i = 0; i < N; i++) {
        weights[i] = (int8_t)(weights[i] / 2);
    }

    /* --- Verification --- */
    int32_t ref = dot_int8_scalar(input, weights, N);
    printf("Verification:\n");

    int32_t r_direct = dot_int8_avx2_direct(input, weights, N);
    CHECK_TRUE(r_direct == ref, "AVX2 direct (s16 conv) matches scalar");

    int32_t r_maddubs = dot_int8_avx2_maddubs(input, weights, N);
    CHECK_TRUE(r_maddubs == ref, "AVX2 maddubs+madd matches scalar");

    /* Zero-point test */
    uint8_t input_zp = 128;
    int8_t weight_zp = 0;
    int32_t ref_zp = dot_int8_scalar_zp(input, weights, input_zp, weight_zp, N);
    int32_t r_zp = dot_int8_avx2_zp(input, weights, input_zp, weight_zp, N);
    CHECK_TRUE(r_zp == ref_zp, "AVX2 with zero-point matches scalar");

    /* --- Benchmark --- */
    printf("\nBenchmark (minimum of %d iterations):\n", 200);
    volatile int32_t sink;

    benchmark_result_t results[4];
    int bytes_per_call = N * (int)(sizeof(uint8_t) + sizeof(int8_t));

    BENCH_COMPUTE(sink = dot_int8_scalar(input, weights, N),
                  N, bytes_per_call, 200, results[0]);
    results[0].name = "scalar (int8)";

    BENCH_COMPUTE(sink = dot_int8_avx2_direct(input, weights, N),
                  N, bytes_per_call, 200, results[1]);
    results[1].name = "AVX2 direct (s16 conv)";

    BENCH_COMPUTE(sink = dot_int8_avx2_zp(input, weights, input_zp, weight_zp, N),
                  N, bytes_per_call, 200, results[2]);
    results[2].name = "AVX2 zero-point";

    BENCH_COMPUTE(sink = dot_int8_avx2_maddubs(input, weights, N),
                  N, bytes_per_call, 200, results[3]);
    results[3].name = "AVX2 maddubs+madd";

    bench_report(results, 4);

    /* Equivalent FP32 ops per second and throughput */
    printf("Throughput & Equivalent GFLOPS:\n");
    for (int r = 0; r < 4; r++) {
        double ns = results[r].elapsed_ns;
        if (ns > 0.0) {
            double gelem_s = (double)results[r].num_elements / ns;  /* Gelem/s */
            double gflops  = 2.0 * gelem_s;  /* 2 flops per MAC (mul + add) */
            printf("  %-28s  %8.3f Gelem/s  %8.3f GFLOPS\n",
                   results[r].name, gelem_s, gflops);
        }
    }

    printf("\n--- Technique Explanation ---\n");
    printf("_mm256_maddubs_epi16:  32 bytes u8 * 32 bytes s8 -> 16 s16 acc\n");
    printf("_mm256_madd_epi16:     s16 * ones -> s32 (sum of 2 adjacent s16)\n");
    printf("_mm256_add_epi32:      accumulate 8-wide s32 partial sums\n");
    printf("This is the standard VPMADDUBSW+VPMADDWD pattern in production.\n");

    ALIGNED_FREE(input);
    ALIGNED_FREE(weights);
    (void)sink; /* suppress unused-but-set-volatile warning */
    return 0;
}
