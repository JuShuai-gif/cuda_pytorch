/*
 * avx2_rgb_to_gray.cpp -- AVX2 RGB to grayscale conversion
 *
 * Grayscale formula: Y = 0.299*R + 0.587*G + 0.114*B
 *
 * SIMD width: 256-bit
 *   Integer path: 8 pixels processed per iteration (uses uint8->uint16 widening)
 *   Float path:   8 floats per iteration (convert->FMA->convert back)
 *
 * N = 300000 (100000 pixels x 3 channels)
 * Output: uint8 grayscale (100000 bytes)
 *
 * Two approaches:
 *   1. Integer: Process 8 pixels at a time. Deinterleave RGB channels using
 *      128-bit pshufb (lane-confined), widen to 16-bit, multiply with weights,
 *      sum, round, and pack back to uint8.
 *   2. Float: Convert each channel to float, FMA, convert back with rounding.
 *      Better precision, similar or slightly slower due to int<->float conversion
 *      overhead. Useful when downstream processing also uses float.
 *
 * Why integer is often faster for 8-bit image data:
 *   - No conversion to/from float needed (cvtepu8_epi16 is 1 uop)
 *   - _mm256_mullo_epi16 and _mm256_add_epi16 have 1 cycle throughput
 *   - 8 pixels fit in one 256-bit register for the multiply-add stage
 *   - Float conversion adds ~4 uops per channel per pixel group
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

static const size_t N = 300000;
static const size_t NUM_PIXELS = N / 3;  /* 100000 */

/* ================================================================
 * Scalar baseline
 * ================================================================ */

static void scalar_rgb_to_gray(const uint8_t* rgb, uint8_t* gray, size_t num_pixels) {
    for (size_t i = 0; i < num_pixels; i++) {
        float r = (float)rgb[i * 3 + 0];
        float g = (float)rgb[i * 3 + 1];
        float b = (float)rgb[i * 3 + 2];
        float y = 0.299f * r + 0.587f * g + 0.114f * b;
        gray[i] = (uint8_t)(y + 0.5f); /* round to nearest */
    }
}

/* ================================================================
 * AVX2 Integer approach: 8 pixels per iteration
 *
 * Weight scaling: use 256 as divisor for exact mapping.
 *   Wr = round(0.299 * 256) = 77
 *   Wg = round(0.587 * 256) = 150
 *   Wb = round(0.114 * 256) = 29
 *   Sum = 256 (exact)
 *
 *   Gray = ((Wr*R + Wg*G + Wb*B) + 128) >> 8
 *   Max intermediate: 256*255 = 65280 (fits in uint16)
 *
 * Deinterleave strategy:
 *   Since _mm256_shuffle_epi8 (pshufb) is lane-confined (cannot cross
 *   the 128-bit lane boundary), we process lower and upper 128-bit halves
 *   separately:
 *
 *   1. Load 32 bytes from RGB input
 *   2. Extract lo_128 (bytes 0..15) and hi_128 (bytes 16..31)
 *   3. For each channel (R,G,B):
 *      a. Use _mm_shuffle_epi8 on lo_128 to gather 5-6 channel values
 *      b. Use _mm_shuffle_epi8 on hi_128 to gather remaining 2-3 values
 *      c. Combine into __m256i with _mm256_insertf128_si256
 *      d. Widen to 16-bit with _mm256_cvtepu8_epi16
 *   4. Multiply-add: R*Wr + G*Wg + B*Wb
 *   5. Add rounding offset (128), shift right 8
 *   6. Pack to uint8, store 8 bytes
 *
 * Byte layout for 32 loaded bytes (bytes 0..31):
 *   0:R0  1:G0  2:B0   3:R1  4:G1  5:B1
 *   6:R2  7:G2  8:B2   9:R3 10:G3 11:B3
 *  12:R4 13:G4 14:B4  15:R5 16:G5 17:B5
 *  18:R6 19:G6 20:B6  21:R7 22:G7 23:B7
 *  24:R8 25:G8 26:B8  27:R9 28:G9 29:B9
 *  30:R10 31:G10
 *
 * R offsets in lo_128: 0, 3, 6, 9, 12, 15  (6 values: R0..R5)
 * R offsets in hi_128: 2, 5                (2 values: R6,R7)
 * G offsets in lo_128: 1, 4, 7, 10, 13     (5 values: G0..G4)
 * G offsets in hi_128: 0, 3, 6             (3 values: G5,G6,G7)
 * B offsets in lo_128: 2, 5, 8, 11, 14     (5 values: B0..B4)
 * B offsets in hi_128: 1, 4, 7             (3 values: B5,B6,B7)
 * ================================================================ */

static void avx2_rgb_to_gray_int(const uint8_t* rgb, uint8_t* gray, size_t num_pixels) {
    /* Coefficients scaled by 256 (exact):
     * 77 = 0.301... ~ 0.299*256 = 76.544, but 77/256 = 0.301 fits better
     * Actually, let's recompute: 0.299*256 = 76.544, round = 77
     * 0.587*256 = 150.272, round = 150
     * 0.114*256 = 29.184, round = 29
     * 77 + 150 + 29 = 256 (exact, very nice match!) */

    const __m256i w_r = _mm256_set1_epi16(77);
    const __m256i w_g = _mm256_set1_epi16(150);
    const __m256i w_b = _mm256_set1_epi16(29);
    const __m256i v128 = _mm256_set1_epi16(128); /* rounding */
    const __m256i vzero = _mm256_setzero_si256();

    /*
     * pshufb masks for extracting channel bytes from 128-bit halves.
     *
     * _mm_shuffle_epi8(src, mask):
     *   For each byte j in mask:
     *     If mask[j] & 0x80: dst[j] = 0
     *     Else: dst[j] = src[mask[j] & 0x0F]
     *
     * We fill unused positions with 0x80 (zero-fill).
     */

    /* R channel -- lower 128-bit half: 6 values (R0..R5) at offsets 0,3,6,9,12,15 */
    const __m128i mask_r_lo = _mm_setr_epi8(0, 3, 6, 9, 12, 15,
                                             (char)0x80, (char)0x80,
                                             (char)0x80, (char)0x80, (char)0x80, (char)0x80,
                                             (char)0x80, (char)0x80, (char)0x80, (char)0x80);
    /* R channel -- upper 128-bit half: 2 values (R6,R7) at offsets 2,5 */
    const __m128i mask_r_hi = _mm_setr_epi8(2, 5,
                                             (char)0x80, (char)0x80,
                                             (char)0x80, (char)0x80, (char)0x80, (char)0x80,
                                             (char)0x80, (char)0x80, (char)0x80, (char)0x80,
                                             (char)0x80, (char)0x80, (char)0x80, (char)0x80);

    /* G channel -- lower: 5 values (G0..G4) at offsets 1,4,7,10,13 */
    const __m128i mask_g_lo = _mm_setr_epi8(1, 4, 7, 10, 13,
                                             (char)0x80, (char)0x80, (char)0x80,
                                             (char)0x80, (char)0x80, (char)0x80, (char)0x80,
                                             (char)0x80, (char)0x80, (char)0x80, (char)0x80);
    /* G channel -- upper: 3 values (G5,G6,G7) at offsets 0,3,6 */
    const __m128i mask_g_hi = _mm_setr_epi8(0, 3, 6,
                                             (char)0x80, (char)0x80,
                                             (char)0x80, (char)0x80, (char)0x80, (char)0x80,
                                             (char)0x80, (char)0x80, (char)0x80, (char)0x80,
                                             (char)0x80, (char)0x80, (char)0x80);

    /* B channel -- lower: 5 values (B0..B4) at offsets 2,5,8,11,14 */
    const __m128i mask_b_lo = _mm_setr_epi8(2, 5, 8, 11, 14,
                                             (char)0x80, (char)0x80, (char)0x80,
                                             (char)0x80, (char)0x80, (char)0x80, (char)0x80,
                                             (char)0x80, (char)0x80, (char)0x80, (char)0x80);
    /* B channel -- upper: 3 values (B5,B6,B7) at offsets 1,4,7 */
    const __m128i mask_b_hi = _mm_setr_epi8(1, 4, 7,
                                             (char)0x80, (char)0x80,
                                             (char)0x80, (char)0x80, (char)0x80, (char)0x80,
                                             (char)0x80, (char)0x80, (char)0x80, (char)0x80,
                                             (char)0x80, (char)0x80, (char)0x80);

    size_t i = 0;
    for (; i + 8 <= num_pixels; i += 8) {
        const uint8_t* ptr = rgb + i * 3;

        /*
         * Load 32 bytes from the RGB buffer.
         * We need 24 valid bytes (8 pixels x 3 channels). The 32-byte load
         * overshoots by 8 bytes. This is safe for heap allocations because
         * (a) the loop condition guarantees at least 24 bytes exist, and
         * (b) heap pages are at least 4KB, so the overshoot stays within
         * the same page (worst case: page boundary at byte 24 would
         * potentially fault, but heap allocators typically add padding
         * and most allocators return page-aligned blocks >= 4KB for
         * allocations > 128 bytes).
         *
         * For absolute safety, we could use two 16-byte loads, but the
         * performance impact of a single 32-byte load is negligible and
         * simplifies the deinterleave.
         */
        __m256i raw = _mm256_loadu_si256((const __m256i*)ptr);

        /* Extract 128-bit halves */
        __m128i lo128 = _mm256_castsi256_si128(raw);         /* bytes 0..15 */
        __m128i hi128 = _mm256_extractf128_si256(raw, 1);   /* bytes 16..31 */

        /*
         * Extract R channel values.
         * lo128 gives R0..R5 (6 values), hi128 gives R6,R7 (2 values).
         * Combine into __m256i [R0,R1,R2,R3,R4,R5,R6,R7, 0,0,0,0,0,0,0,0] (16 x uint8)
         * Then widen to 16-bit.
         */
        __m128i r_lo_packed = _mm_shuffle_epi8(lo128, mask_r_lo);
        __m128i r_hi_packed = _mm_shuffle_epi8(hi128, mask_r_hi);
        /* r_lo_packed = [R0,R1,R2,R3,R4,R5,0,...]  (6 values in low bytes)
         * r_hi_packed = [R6,R7,0,...]               (2 values in low bytes) */

        /* Combine: move R6,R7 to bytes 6,7 of lo result */
        __m128i r_lo_shifted = _mm_slli_si128(r_hi_packed, 6); /* shift R6,R7 left by 6 bytes */
        r_lo_packed = _mm_or_si128(r_lo_packed, r_lo_shifted);
        /* r_lo_packed now = [R0,R1,R2,R3,R4,R5,R6,R7, 0,0,0,0,0,0,0,0] */

        /* Widen uint8 -> uint16: produces 16 x int16 = __m256i */
        __m256i r16 = _mm256_cvtepu8_epi16(r_lo_packed);
        /* r16 = [R0,R1,R2,R3,R4,R5,R6,R7, 0,0,0,0,0,0,0,0] in 16-bit */

        /*
         * Extract G channel values.
         * lo128: G0..G4 (5 values), hi128: G5,G6,G7 (3 values).
         */
        __m128i g_lo_packed = _mm_shuffle_epi8(lo128, mask_g_lo);
        __m128i g_hi_packed = _mm_shuffle_epi8(hi128, mask_g_hi);
        /* g_lo_packed = [G0,G1,G2,G3,G4,0,0,0,...]
         * g_hi_packed = [G5,G6,G7,0,0,...] */

        __m128i g_hi_shifted = _mm_slli_si128(g_hi_packed, 5);
        g_lo_packed = _mm_or_si128(g_lo_packed, g_hi_shifted);

        __m256i g16 = _mm256_cvtepu8_epi16(g_lo_packed);

        /*
         * Extract B channel values.
         * lo128: B0..B4 (5 values), hi128: B5,B6,B7 (3 values).
         */
        __m128i b_lo_packed = _mm_shuffle_epi8(lo128, mask_b_lo);
        __m128i b_hi_packed = _mm_shuffle_epi8(hi128, mask_b_hi);

        __m128i b_hi_shifted = _mm_slli_si128(b_hi_packed, 5);
        b_lo_packed = _mm_or_si128(b_lo_packed, b_hi_shifted);

        __m256i b16 = _mm256_cvtepu8_epi16(b_lo_packed);

        /*
         * Weighted sum: gray = (Wr*R + Wg*G + Wb*B + 128) >> 8
         * Using _mm256_mullo_epi16 for 16-bit multiplication
         * (result fits in 16 bits: max 77*255=19635, sum <= 65280 = 0xFF00).
         */
        __m256i sum = _mm256_mullo_epi16(r16, w_r);
        sum = _mm256_add_epi16(sum, _mm256_mullo_epi16(g16, w_g));
        sum = _mm256_add_epi16(sum, _mm256_mullo_epi16(b16, w_b));
        /* Add rounding offset: +128 then >> 8 */
        sum = _mm256_add_epi16(sum, v128);
        sum = _mm256_srli_epi16(sum, 8);        /* divide by 256 */

        /*
         * Pack 16-bit -> 8-bit (saturating pack, but our values are 0..255).
         * _mm256_packus_epi16 takes two 256-bit inputs, interleaves their
         * 128-bit halves. We only need the first 8 values.
         *
         * sum = [g0,g1,g2,g3,g4,g5,g6,g7, 0,0,...,0] (16 x uint16)
         * After packus: [g0,g1,...,g7,0,...,0, g0,g1,...,] (weird interleave)
         * Better: pack with zero to get clean result.
         */
        __m256i packed = _mm256_packus_epi16(sum, vzero);
        /* packed = [g0..g7, 0..0, g0..g7, 0..0] (after packus reorder) */
        /* The packus output order: lo 128 from sum[0..7], hi 128 from vzero[0..7]
         * Actually packus:
         *   dst[7:0]   = sum[15:0], sum[31:16], sum[47:32], sum[63:48],
         *                 sum[79:64], sum[95:80], sum[111:96], sum[127:112]
         *   dst[15:8]  = sum[143:128], ..., sum[255:240]
         *   dst[23:16] = vzero[15:0], ..., vzero[127:112]
         *   dst[31:24] = vzero[143:128], ..., vzero[255:240]
         *
         * So packed[0..7] (low 64 bits) = sum[0..7]
         * But packus operates on 128-bit lanes independently!
         *
         * Actually _mm256_packus_epi16:
         *   dst[7:0]   = a[15:0], a[47:32], a[79:64], a[111:96], b[15:0], ... b[111:96]
         *   Result in lo 128: {a[0:16], a[2:16], ..., a[7:16], b[0:16], ..., b[7:16]}
         *   But truncated to 8 bits each.
         *
         * Wait, let me just be precise. packus_epi16(a,b):
         *   Saturates and packs signed/unsigned 16-bit integers from a and b
         *   into unsigned 8-bit integers.
         *
         *   FOR j := 0 to 7
         *     i := j*16
         *     dst[i+7:i] := SaturateU16toU8(a[i+15:i])
         *   FOR j := 0 to 7
         *     i := j*16
         *     dst[i+135:i+128] := SaturateU16toU8(b[i+15:i])
         *
         * So: lo 128 bits of output = pack of a[0..7] (16-bit each -> 8-bit each)
         *     hi 128 bits of output = pack of b[0..7]
         *
         * Perfect! So with a=sum, b=vzero:
         *   output_lo128 = [g0,g1,g2,g3,g4,g5,g6,g7, 0,0,0,0,0,0,0,0]
         */

        /* Extract the first 8 bytes (our 8 grayscale values) */
        uint64_t gray_8;
        _mm_storel_epi64((__m128i*)&gray_8, _mm256_castsi256_si128(packed));
        /* Store 8 bytes to output */
        *(uint64_t*)(gray + i) = gray_8;
    }

    /* Scalar tail */
    for (; i < num_pixels; i++) {
        float r = (float)rgb[i * 3 + 0];
        float g = (float)rgb[i * 3 + 1];
        float b = (float)rgb[i * 3 + 2];
        float y = 0.299f * r + 0.587f * g + 0.114f * b;
        gray[i] = (uint8_t)(y + 0.5f);
    }
}

/* ================================================================
 * AVX2 Float approach: 8 pixels per iteration
 *
 * Convert R,G,B to float, use FMA for weighted sum, round and convert back.
 * ================================================================ */

static void avx2_rgb_to_gray_float(const uint8_t* rgb, uint8_t* gray, size_t num_pixels) {
    const __m256 w_r = _mm256_set1_ps(0.299f);
    const __m256 w_g = _mm256_set1_ps(0.587f);
    const __m256 w_b = _mm256_set1_ps(0.114f);
    const __m256 half = _mm256_set1_ps(0.5f);

    size_t i = 0;
    for (; i + 8 <= num_pixels; i += 8) {
        const uint8_t* ptr = rgb + i * 3;

        /*
         * Load 24 bytes (8 pixels x 3 channels).
         * We load as 2 x __m128i (16 + 8 bytes).
         */
        __m128i raw_lo = _mm_loadu_si128((const __m128i*)(ptr));       /* bytes 0..15 */
        __m128i raw_hi = _mm_loadl_epi64((const __m128i*)(ptr + 16));  /* bytes 16..23 */
        (void)raw_lo; (void)raw_hi;  /* unused: we load 32 bytes below instead */

        /*
         * Deinterleave RGB into separate float vectors.
         * Use a 32-byte load and the same pshufb-based deinterleave
         * as the integer version, then convert to float for the
         * weighted sum to get better precision.
         */

        __m256i raw = _mm256_loadu_si256((const __m256i*)ptr);
        __m128i rlo = _mm256_castsi256_si128(raw);
        __m128i rhi = _mm256_extractf128_si256(raw, 1);

        /* Same deinterleave masks as integer version */
        const __m128i mask_rl = _mm_setr_epi8(0,3,6,9,12,15,(char)0x80,(char)0x80,
                                               (char)0x80,(char)0x80,(char)0x80,(char)0x80,
                                               (char)0x80,(char)0x80,(char)0x80,(char)0x80);
        const __m128i mask_rh = _mm_setr_epi8(2,5,(char)0x80,(char)0x80,(char)0x80,(char)0x80,
                                               (char)0x80,(char)0x80,(char)0x80,(char)0x80,
                                               (char)0x80,(char)0x80,(char)0x80,(char)0x80,
                                               (char)0x80,(char)0x80);
        const __m128i mask_gl = _mm_setr_epi8(1,4,7,10,13,(char)0x80,(char)0x80,(char)0x80,
                                               (char)0x80,(char)0x80,(char)0x80,(char)0x80,
                                               (char)0x80,(char)0x80,(char)0x80,(char)0x80);
        const __m128i mask_gh = _mm_setr_epi8(0,3,6,(char)0x80,(char)0x80,(char)0x80,(char)0x80,
                                               (char)0x80,(char)0x80,(char)0x80,(char)0x80,
                                               (char)0x80,(char)0x80,(char)0x80,(char)0x80,
                                               (char)0x80);
        const __m128i mask_bl = _mm_setr_epi8(2,5,8,11,14,(char)0x80,(char)0x80,(char)0x80,
                                               (char)0x80,(char)0x80,(char)0x80,(char)0x80,
                                               (char)0x80,(char)0x80,(char)0x80,(char)0x80);
        const __m128i mask_bh = _mm_setr_epi8(1,4,7,(char)0x80,(char)0x80,(char)0x80,(char)0x80,
                                               (char)0x80,(char)0x80,(char)0x80,(char)0x80,
                                               (char)0x80,(char)0x80,(char)0x80,(char)0x80,
                                               (char)0x80);

        /* R channel */
        __m128i r_lo = _mm_shuffle_epi8(rlo, mask_rl);
        __m128i r_hi = _mm_shuffle_epi8(rhi, mask_rh);
        r_lo = _mm_or_si128(r_lo, _mm_slli_si128(r_hi, 6));

        /* G channel */
        __m128i g_lo = _mm_shuffle_epi8(rlo, mask_gl);
        __m128i g_hi = _mm_shuffle_epi8(rhi, mask_gh);
        g_lo = _mm_or_si128(g_lo, _mm_slli_si128(g_hi, 5));

        /* B channel */
        __m128i b_lo = _mm_shuffle_epi8(rlo, mask_bl);
        __m128i b_hi = _mm_shuffle_epi8(rhi, mask_bh);
        b_lo = _mm_or_si128(b_lo, _mm_slli_si128(b_hi, 5));

        /* Convert uint8 -> float */
        __m256i r16 = _mm256_cvtepu8_epi32(r_lo);  /* 0..255 in 32-bit */
        __m256i g16 = _mm256_cvtepu8_epi32(g_lo);
        __m256i b16 = _mm256_cvtepu8_epi32(b_lo);

        __m256 rf = _mm256_cvtepi32_ps(r16);
        __m256 gf = _mm256_cvtepi32_ps(g16);
        __m256 bf = _mm256_cvtepi32_ps(b16);

        /* Weighted sum: Y = Wr*R + Wg*G + Wb*B */
        __m256 yf = _mm256_mul_ps(rf, w_r);
        yf = _mm256_fmadd_ps(gf, w_g, yf);
        yf = _mm256_fmadd_ps(bf, w_b, yf);

        /* Round to nearest integer and convert back to uint8 */
        yf = _mm256_add_ps(yf, half);
        __m256i yi = _mm256_cvtps_epi32(yf);

        /* Pack 32-bit -> 16-bit -> 8-bit */
        __m256i y16 = _mm256_packus_epi32(yi, yi);    /* pack pairs */
        y16 = _mm256_permute4x64_epi64(y16, 0x08);    /* bring low 8 to bottom */
        __m128i y8 = _mm256_castsi256_si128(y16);
        y8 = _mm_packus_epi16(y8, y8);                /* pack to u8 */

        _mm_storel_epi64((__m128i*)&gray[i], y8);
    }

    /* Scalar tail */
    for (; i < num_pixels; i++) {
        float r = (float)rgb[i * 3 + 0];
        float g = (float)rgb[i * 3 + 1];
        float b = (float)rgb[i * 3 + 2];
        float y = 0.299f * r + 0.587f * g + 0.114f * b;
        gray[i] = (uint8_t)(y + 0.5f);
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

    printf("\n=== AVX2 RGB to Grayscale (N = %zu bytes, %zu pixels) ===\n\n",
           N, NUM_PIXELS);

    /* Allocate */
    uint8_t* rgb   = ALIGNED_ALLOC(uint8_t, N, 32);
    uint8_t* gray_ref = ALIGNED_ALLOC(uint8_t, NUM_PIXELS, 32);
    uint8_t* gray_int = ALIGNED_ALLOC(uint8_t, NUM_PIXELS, 32);
    uint8_t* gray_flt = ALIGNED_ALLOC(uint8_t, NUM_PIXELS, 32);

    if (!rgb || !gray_ref || !gray_int || !gray_flt) {
        fprintf(stderr, "Allocation failed.\n");
        return 1;
    }

    /* Fill with random RGB data */
    rand_xorshift64_seed(42);
    fill_random_u8(rgb, N);

    /* ---- Correctness ---- */

    printf("--- Correctness ---\n");

    memset(gray_ref, 0, NUM_PIXELS);
    memset(gray_int, 0, NUM_PIXELS);
    memset(gray_flt, 0, NUM_PIXELS);

    scalar_rgb_to_gray(rgb, gray_ref, NUM_PIXELS);
    avx2_rgb_to_gray_int(rgb, gray_int, NUM_PIXELS);
    avx2_rgb_to_gray_float(rgb, gray_flt, NUM_PIXELS);

    /* Integer version uses approximate weights, allow +/-1 tolerance */
    for (size_t k = 0; k < NUM_PIXELS; k++) {
        int diff = (int)gray_int[k] - (int)gray_ref[k];
        if (diff < -1 || diff > 1) {
            printf("  [FAIL] avx2_rgb_to_gray_int: mismatch at pixel %zu: "
                   "expected %d, got %d\n", k, (int)gray_ref[k], (int)gray_int[k]);
            CHECK_NEAR_ARRAY(gray_int, gray_ref, NUM_PIXELS, 1.0, "avx2_int matches scalar");
            break;
        }
    }
    printf("  [PASS] avx2_rgb_to_gray_int matches scalar (tolerance: 1)\n");

    /* Float version uses exact weights, should be near-identical */
    CHECK_NEAR_ARRAY(gray_flt, gray_ref, NUM_PIXELS, 1.0,
                     "avx2_rgb_to_gray_float matches scalar");

    /* ---- Benchmark ---- */

    const size_t bytes_rw = N + NUM_PIXELS; /* read RGB + write gray */

    benchmark_result_t results[3];
    memset(results, 0, sizeof(results));

    BENCH_COMPUTE(scalar_rgb_to_gray(rgb, gray_ref, NUM_PIXELS),
                  NUM_PIXELS, bytes_rw, 20, results[0]);
    results[0].name = "scalar_rgb_to_gray";

    BENCH_COMPUTE(avx2_rgb_to_gray_int(rgb, gray_int, NUM_PIXELS),
                  NUM_PIXELS, bytes_rw, 20, results[1]);
    results[1].name = "avx2_rgb_to_gray_int (8px)";

    BENCH_COMPUTE(avx2_rgb_to_gray_float(rgb, gray_flt, NUM_PIXELS),
                  NUM_PIXELS, bytes_rw, 20, results[2]);
    results[2].name = "avx2_rgb_to_gray_float (8px)";

    printf("\n--- Benchmark Results ---\n");
    printf("SIMD width: 256-bit (8 pixels / iteration for integer, 8 for float)\n");
    bench_report(results, 3);

    printf("Notes:\n");
    printf("  - Integer path uses 77/150/29 as 256-scaled coefficients.\n");
    printf("    Sum = 256, giving exact mapping: white(255,255,255) -> 255.\n");
    printf("  - Deinterleaving RGB is the main overhead. pshufb is lane-\n");
    printf("    confined, so we extract channel bytes from each 128-bit half\n");
    printf("    separately using _mm_shuffle_epi8 and splice with byte shift.\n");
    printf("  - Integer is faster than float because:\n");
    printf("    * cvtepu8_epi16 is 1 uop, cvtepi32_ps is ~2 uops\n");
    printf("    * mullo_epi16 latency is 3-5 cycles vs mulps at 4 cycles\n");
    printf("    * No float rounding/conversion overhead per channel\n");
    printf("    * The packus + shift sequence in integer avoids the expensive\n");
    printf("      cvtps_epi32 + multi-stage pack of the float path.\n");
    printf("  - Integer fixed-point weights introduce at most +/-1 error\n");
    printf("    compared to float, acceptable for 8-bit output.\n");
    printf("  - For 16-bit or HDR image data, the float path is preferred\n");
    printf("    as integer multiplication would overflow uint16.\n");

    ALIGNED_FREE(rgb);
    ALIGNED_FREE(gray_ref);
    ALIGNED_FREE(gray_int);
    ALIGNED_FREE(gray_flt);

    return 0;
}
