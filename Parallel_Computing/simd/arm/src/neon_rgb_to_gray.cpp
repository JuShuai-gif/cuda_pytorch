#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <arm_neon.h>
#include "../../common/benchmark.h"
#include "../../common/check.h"
#include "../../common/random_data.h"
#include "../../common/aligned_buffer.h"

// =============================================================================
// neon_rgb_to_gray -- RGB (interleaved) to grayscale
//   gray = 0.299*R + 0.587*G + 0.114*B
//   N = 100000 pixels (300000 bytes of RGB input)
//
//   Approach 1 (direct): vld3q_u8 to load interleaved RGB, convert to float,
//                        compute formula, convert back to uint8, store
//   Approach 2 (2-pass): deinterleave RGB -> planar R,G,B arrays via vld3q_u8,
//                        then compute grayscale on planar data (more cache-friendly)
//
//   Both output uint8 grayscale using vector stores (vst1q_u8).
// =============================================================================

static const size_t N = 100000;
static const int    BENCH_ITERS = 10;

// BT.601 grayscale coefficients
static const float COEF_R = 0.299f;
static const float COEF_G = 0.587f;
static const float COEF_B = 0.114f;

// ============================================================================
// Scalar baseline
// ============================================================================

static void scalar_rgb_to_gray_u8(const uint8_t* rgb, uint8_t* gray, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float r = (float)rgb[i * 3 + 0];
        float g = (float)rgb[i * 3 + 1];
        float b = (float)rgb[i * 3 + 2];
        float y = COEF_R * r + COEF_G * g + COEF_B * b;
        if (y < 0.0f)   y = 0.0f;
        if (y > 255.0f) y = 255.0f;
        gray[i] = (uint8_t)(y + 0.5f);
    }
}

// ============================================================================
// NEON u8 <-> f32 conversion helpers
// ============================================================================

// Convert 4 uint8 values at byte offset `ofs` within a uint8x16 register
// to float32x4. Uses widen chain: u8 -> u16 -> u32 -> f32.
// `ofs` must be 0, 4, 8, or 12.
static inline float32x4_t neon_u8_at_ofs_to_f32(uint8x16_t v, int ofs) {
    // Extract the desired 8-byte half, then take the correct 4 from it
    uint8x8_t lo = vget_low_u8(v);
    uint8x8_t hi = vget_high_u8(v);
    uint8x8_t target = (ofs < 8) ? lo : hi;
    int sub = (ofs < 8) ? ofs : (ofs - 8);

    // Shift and widen: uint8x8 -> uint16x8 -> pick low 4 -> uint32x4 -> float
    uint16x8_t u16 = vmovl_u8(target);
    uint16x4_t u16_low = vget_low_u16(u16);
    uint16x4_t u16_hi  = vget_high_u16(u16);
    uint16x4_t u16_4 = (sub == 0) ? u16_low : u16_hi;
    uint32x4_t u32 = vmovl_u16(u16_4);
    return vcvtq_f32_u32(u32);
}

// Convert float32x4 (clamped 0..255) to uint16x4 for accumulation
static inline uint16x4_t neon_f32x4_to_u16x4(float32x4_t vf) {
    uint32x4_t u32 = vcvtq_u32_f32(vf);
    return vqmovn_u32(u32); // saturating narrow u32x4 -> u16x4
}

// ============================================================================
// Approach 1: Direct interleaved processing with vld3q_u8
// ============================================================================
// Uses vld3q_u8 to load 16 RGB pixels, converts in 4 groups of 4,
// then packs all 16 results into a uint8x16 vector for a single store.

static void neon_rgb_to_gray_direct_u8(const uint8_t* rgb, uint8_t* gray,
                                        size_t n) {
    const float32x4_t vcoef_r = vdupq_n_f32(COEF_R);
    const float32x4_t vcoef_g = vdupq_n_f32(COEF_G);
    const float32x4_t vcoef_b = vdupq_n_f32(COEF_B);
    const float32x4_t v255    = vdupq_n_f32(255.0f);
    const float32x4_t v0      = vdupq_n_f32(0.0f);
    const float32x4_t vhalf   = vdupq_n_f32(0.5f);

    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        uint8x16x3_t vrgb = vld3q_u8(rgb + i * 3);

        // Process 4 groups of 4 pixels each: offset 0, 4, 8, 12
        uint16x4_t u16_group[4];
        for (int g = 0; g < 4; g++) {
            int ofs = g * 4;
            float32x4_t rf = neon_u8_at_ofs_to_f32(vrgb.val[0], ofs);
            float32x4_t gf = neon_u8_at_ofs_to_f32(vrgb.val[1], ofs);
            float32x4_t bf = neon_u8_at_ofs_to_f32(vrgb.val[2], ofs);

            // gray = 0.299*R + 0.587*G + 0.114*B
            float32x4_t yf = vmlaq_f32(vmulq_f32(rf, vcoef_r), gf, vcoef_g);
            yf = vmlaq_f32(yf, bf, vcoef_b);

            // Clamp and round: [0.0, 255.0] + 0.5
            yf = vaddq_f32(yf, vhalf);
            yf = vmaxq_f32(yf, v0);
            yf = vminq_f32(yf, v255);

            u16_group[g] = neon_f32x4_to_u16x4(yf);
        }

        // Pack 4 x uint16x4 (16 values) -> uint8x16 -> single vector store
        uint16x8_t u16_01 = vcombine_u16(u16_group[0], u16_group[1]);
        uint16x8_t u16_23 = vcombine_u16(u16_group[2], u16_group[3]);
        uint8x8_t  u8_01  = vqmovn_u16(u16_01);
        uint8x8_t  u8_23  = vqmovn_u16(u16_23);
        uint8x16_t u8_out = vcombine_u8(u8_01, u8_23);
        vst1q_u8(gray + i, u8_out);
    }

    // Tail: scalar fallback
    for (; i < n; i++) {
        float r = (float)rgb[i * 3 + 0];
        float g = (float)rgb[i * 3 + 1];
        float b = (float)rgb[i * 3 + 2];
        float y = COEF_R * r + COEF_G * g + COEF_B * b;
        if (y < 0.0f)   y = 0.0f;
        if (y > 255.0f) y = 255.0f;
        gray[i] = (uint8_t)(y + 0.5f);
    }
}

// ============================================================================
// Approach 2: Deinterleave first, then compute on planar data
// ============================================================================

static void neon_deinterleave_rgb(const uint8_t* rgb, uint8_t* r_planar,
                                   uint8_t* g_planar, uint8_t* b_planar,
                                   size_t n) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        uint8x16x3_t vrgb = vld3q_u8(rgb + i * 3);
        vst1q_u8(r_planar + i, vrgb.val[0]);
        vst1q_u8(g_planar + i, vrgb.val[1]);
        vst1q_u8(b_planar + i, vrgb.val[2]);
    }
    for (; i < n; i++) {
        r_planar[i] = rgb[i * 3 + 0];
        g_planar[i] = rgb[i * 3 + 1];
        b_planar[i] = rgb[i * 3 + 2];
    }
}

static void neon_compute_gray_planar(const uint8_t* r, const uint8_t* g,
                                      const uint8_t* b, uint8_t* gray,
                                      size_t n) {
    const float32x4_t vcoef_r = vdupq_n_f32(COEF_R);
    const float32x4_t vcoef_g = vdupq_n_f32(COEF_G);
    const float32x4_t vcoef_b = vdupq_n_f32(COEF_B);
    const float32x4_t v255    = vdupq_n_f32(255.0f);
    const float32x4_t v0      = vdupq_n_f32(0.0f);
    const float32x4_t vhalf   = vdupq_n_f32(0.5f);

    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        uint8x16_t vr = vld1q_u8(r + i);
        uint8x16_t vg = vld1q_u8(g + i);
        uint8x16_t vb = vld1q_u8(b + i);

        // 4 groups x 4 pixels
        uint16x4_t u16_group[4];
        for (int g_idx = 0; g_idx < 4; g_idx++) {
            int ofs = g_idx * 4;
            float32x4_t rf = neon_u8_at_ofs_to_f32(vr, ofs);
            float32x4_t gf = neon_u8_at_ofs_to_f32(vg, ofs);
            float32x4_t bf = neon_u8_at_ofs_to_f32(vb, ofs);

            float32x4_t yf = vmlaq_f32(vmulq_f32(rf, vcoef_r), gf, vcoef_g);
            yf = vmlaq_f32(yf, bf, vcoef_b);

            yf = vaddq_f32(yf, vhalf);
            yf = vmaxq_f32(yf, v0);
            yf = vminq_f32(yf, v255);

            u16_group[g_idx] = neon_f32x4_to_u16x4(yf);
        }

        // Pack 4 groups into uint8x16 -> single vector store
        uint16x8_t u16_01 = vcombine_u16(u16_group[0], u16_group[1]);
        uint16x8_t u16_23 = vcombine_u16(u16_group[2], u16_group[3]);
        uint8x8_t  u8_01  = vqmovn_u16(u16_01);
        uint8x8_t  u8_23  = vqmovn_u16(u16_23);
        uint8x16_t u8_out = vcombine_u8(u8_01, u8_23);
        vst1q_u8(gray + i, u8_out);
    }

    for (; i < n; i++) {
        float y = COEF_R * (float)r[i] + COEF_G * (float)g[i]
                + COEF_B * (float)b[i];
        if (y < 0.0f)   y = 0.0f;
        if (y > 255.0f) y = 255.0f;
        gray[i] = (uint8_t)(y + 0.5f);
    }
}

// Approach 2 wrapper: deinterleave + compute (two-pass)
static void neon_rgb_to_gray_planar_u8(const uint8_t* rgb, uint8_t* gray,
                                        size_t n,
                                        uint8_t* r_buf, uint8_t* g_buf,
                                        uint8_t* b_buf) {
    neon_deinterleave_rgb(rgb, r_buf, g_buf, b_buf, n);
    neon_compute_gray_planar(r_buf, g_buf, b_buf, gray, n);
}

// ============================================================================
// main
// ============================================================================
int main(void) {
    printf("================================================================\n");
    printf("  NEON RGB -> Grayscale (BT.601)\n");
    printf("  Formula: gray = %.3f*R + %.3f*G + %.3f*B\n",
           COEF_R, COEF_G, COEF_B);
    printf("  N = %zu pixels\n", N);
    printf("================================================================\n");

    uint8_t* rgb     = ALIGNED_ALLOC(uint8_t, N * 3, 16);
    uint8_t* ref     = ALIGNED_ALLOC(uint8_t, N, 16);
    uint8_t* gray1   = ALIGNED_ALLOC(uint8_t, N, 16);
    uint8_t* gray2   = ALIGNED_ALLOC(uint8_t, N, 16);

    // Planar buffers for approach 2
    uint8_t* r_buf   = ALIGNED_ALLOC(uint8_t, N, 16);
    uint8_t* g_buf   = ALIGNED_ALLOC(uint8_t, N, 16);
    uint8_t* b_buf   = ALIGNED_ALLOC(uint8_t, N, 16);

    CHECK_TRUE(is_aligned(rgb, 16), "rgb buffer is 16-byte aligned");

    fill_random_u8(rgb, N * 3);

    // ---- Correctness ----
    printf("\n-- Correctness --\n");

    memset(ref, 0, N);
    scalar_rgb_to_gray_u8(rgb, ref, N);

    memset(gray1, 0, N);
    neon_rgb_to_gray_direct_u8(rgb, gray1, N);
    CHECK_EQ(memcmp(ref, gray1, N), 0,
             "Approach 1 (direct interleaved) matches scalar");

    memset(gray2, 0, N);
    neon_rgb_to_gray_planar_u8(rgb, gray2, N, r_buf, g_buf, b_buf);
    CHECK_EQ(memcmp(ref, gray2, N), 0,
             "Approach 2 (planar, 2-pass) matches scalar");

    // ---- Benchmarks ----
    printf("\n-- Benchmarks (%d timed iterations) --\n", BENCH_ITERS);

    // bytes_processed: read N*3 bytes (RGB) + write N bytes (gray)
    size_t bytes = N * 3 + N;

    benchmark_result_t results[3];

    BENCH_COMPUTE(scalar_rgb_to_gray_u8(rgb, gray1, N),
                  N, bytes, BENCH_ITERS, results[0]);
    results[0].name = "scalar_rgb_to_gray";

    BENCH_COMPUTE(neon_rgb_to_gray_direct_u8(rgb, gray1, N),
                  N, bytes, BENCH_ITERS, results[1]);
    results[1].name = "neon_direct (vld3q)";

    BENCH_COMPUTE(neon_rgb_to_gray_planar_u8(rgb, gray2, N, r_buf, g_buf, b_buf),
                  N, bytes, BENCH_ITERS, results[2]);
    results[2].name = "neon_planar (2-pass)";

    bench_report(results, 3);

    // ---- Analysis ----
    double spd_direct = results[0].elapsed_ns / results[1].elapsed_ns;
    double spd_planar = results[0].elapsed_ns / results[2].elapsed_ns;

    printf("Analysis:\n");
    printf("  Direct (vld3q) speedup:  %.2fx\n", spd_direct);
    printf("  Planar (2-pass) speedup: %.2fx\n", spd_planar);
    printf("\n");
    printf("  The direct approach uses vld3q_u8 to deinterleave on-the-fly\n");
    printf("  and computes grayscale in a single pass. Both implementations\n");
    printf("  now use proper vector stores (vst1q_u8) for maximum throughput.\n");
    printf("\n");
    printf("  The planar 2-pass approach first deinterleaves RGB to separate\n");
    printf("  R/G/B arrays (pass 1), then computes grayscale on contiguous\n");
    printf("  planar data (pass 2). This increases total work but improves\n");
    printf("  cache locality for the compute phase.\n");
    printf("\n");
    printf("  Which is faster depends on the core:\n");
    printf("  - In-order cores (A53/A55): direct approach often wins because\n");
    printf("    it avoids the extra memory traffic of storing/reloading planar data.\n");
    printf("  - OoO cores (A76/X1): planar can win if the deinterleave pass\n");
    printf("    uses bandwidth the core would otherwise waste waiting for RGB loads.\n");
    printf("  - The optimal strategy is often an integer-only approach using\n");
    printf("    vmulal/vmlal on uint16 intermediates, avoiding float entirely.\n");

    ALIGNED_FREE(rgb);
    ALIGNED_FREE(ref);
    ALIGNED_FREE(gray1);
    ALIGNED_FREE(gray2);
    ALIGNED_FREE(r_buf);
    ALIGNED_FREE(g_buf);
    ALIGNED_FREE(b_buf);

    printf("\nAll tests passed.\n");
    return 0;
}
