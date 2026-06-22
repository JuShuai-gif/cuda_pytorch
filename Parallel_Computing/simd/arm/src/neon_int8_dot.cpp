/**
 * neon_int8_dot.cpp -- int8 Dot Product for Quantized ML Inference
 *
 * Computes dot(input, weights) over int8 vectors.
 *
 * Two NEON strategies:
 *   1. vdotq_s32 (ARMv8.2+ dot-product instruction) -- 4x int8 dots/inst
 *   2. Widen to int16, multiply, accumulate (fallback for ARMv8.0/v8.1)
 *
 * Runtime detection of ARMv8.2 ASIMDDP extension selects the optimal path.
 * This maps directly to MLP/GEMV inference where activation-weight dot
 * products dominate compute in quantized models.
 */

#include <arm_neon.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

extern "C" {
#include "../../common/benchmark.h"
#include "../../common/cpu_features.h"
#include "../../common/check.h"
#include "../../common/random_data.h"
#include "../../common/aligned_buffer.h"
}

// ============================================================================
// ARMv8.2 Dot Product (ASIMDDP) detection
// ============================================================================

#ifdef __linux__
#include <sys/auxv.h>
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1UL << 20)
#endif

static int has_asimddp(void) {
    unsigned long hwcap = getauxval(AT_HWCAP);
    return (hwcap & HWCAP_ASIMDDP) ? 1 : 0;
}
#else
static int has_asimddp(void) { return 0; }
#endif

// ============================================================================
// Constants
// ============================================================================

static const int N_DOT = 1000000; // Number of elements (must be multiple of 16)

// ============================================================================
// Scalar baseline
// ============================================================================

static int32_t scalar_dot(const int8_t* __restrict a,
                           const int8_t* __restrict b, int n) {
    int32_t sum = 0;
    for (int i = 0; i < n; i++) {
        sum += (int32_t)a[i] * (int32_t)b[i];
    }
    return sum;
}

// ============================================================================
// NEON without dotprod (ARMv8.0): widen to int16, multiply, accumulate
// ============================================================================
//
// Strategy:
//   1. Load 16 x int8 -> int8x16_t
//   2. Widen: int8x16_t -> int16x8_t (lo), int16x8_t (hi) using vmovl
//   3. Multiply-accumulate: vmlal_s16(int32x4_t, int16x4_t, int16x4_t)
//   4. Combine 4 x int32 accumulators in parallel to hide latency

static int32_t neon_dot_widen(const int8_t* __restrict a,
                               const int8_t* __restrict b, int n) {
    int32x4_t acc0 = vdupq_n_s32(0);
    int32x4_t acc1 = vdupq_n_s32(0);
    int32x4_t acc2 = vdupq_n_s32(0);
    int32x4_t acc3 = vdupq_n_s32(0);

    int i = 0;
    for (; i + 63 < n; i += 64) {
        // Process 64 bytes per iteration, 4 independent accumulators
        for (int j = 0; j < 4; j++) {
            int8x16_t va = vld1q_s8(a + i + j * 16);
            int8x16_t vb = vld1q_s8(b + i + j * 16);

            // Widen to int16: vmull gives 2 x int16x8
            int16x8_t a_lo = vmovl_s8(vget_low_s8(va));
            int16x8_t a_hi = vmovl_s8(vget_high_s8(va));
            int16x8_t b_lo = vmovl_s8(vget_low_s8(vb));
            int16x8_t b_hi = vmovl_s8(vget_high_s8(vb));

            // Multiply-accumulate low and high halves into int32 accumulators
            int32x4_t* acc = (j == 0) ? &acc0 : (j == 1) ? &acc1
                            : (j == 2) ? &acc2 : &acc3;
            *acc = vmlal_s16(*acc, vget_low_s16(a_lo), vget_low_s16(b_lo));
            *acc = vmlal_s16(*acc, vget_high_s16(a_lo), vget_high_s16(b_lo));
            *acc = vmlal_s16(*acc, vget_low_s16(a_hi), vget_low_s16(b_hi));
            *acc = vmlal_s16(*acc, vget_high_s16(a_hi), vget_high_s16(b_hi));
        }
    }

    // Tail: process 16 bytes at a time
    for (; i + 15 < n; i += 16) {
        int8x16_t va = vld1q_s8(a + i);
        int8x16_t vb = vld1q_s8(b + i);
        int16x8_t a_lo = vmovl_s8(vget_low_s8(va));
        int16x8_t a_hi = vmovl_s8(vget_high_s8(va));
        int16x8_t b_lo = vmovl_s8(vget_low_s8(vb));
        int16x8_t b_hi = vmovl_s8(vget_high_s8(vb));

        acc0 = vmlal_s16(acc0, vget_low_s16(a_lo), vget_low_s16(b_lo));
        acc0 = vmlal_s16(acc0, vget_high_s16(a_lo), vget_high_s16(b_lo));
        acc0 = vmlal_s16(acc0, vget_low_s16(a_hi), vget_low_s16(b_hi));
        acc0 = vmlal_s16(acc0, vget_high_s16(a_hi), vget_high_s16(b_hi));
    }

    // Scalar tail
    int32_t sum = vaddvq_s32(acc0) + vaddvq_s32(acc1)
                + vaddvq_s32(acc2) + vaddvq_s32(acc3);
    for (; i < n; i++) {
        sum += (int32_t)a[i] * (int32_t)b[i];
    }
    return sum;
}

// ============================================================================
// NEON with vdotq_s32 (ARMv8.2+): accelerated dot product
// ============================================================================
//
// vdotq_s32(int32x4_t acc, int8x16_t a, int8x16_t b) performs:
//   acc[0] += a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
//   acc[1] += a[4]*b[4] + a[5]*b[5] + a[6]*b[6] + a[7]*b[7]
//   acc[2] += a[8]*b[8] + a[9]*b[9] + a[10]*b[10] + a[11]*b[11]
//   acc[3] += a[12]*b[12] + a[13]*b[13] + a[14]*b[14] + a[15]*b[15]
//
// Each instruction does 4 x (4 int8 multiply-adds) = 16 ops.
// Throughput: typically 1 per cycle on Cortex-A76/A78, ~32 int8 MACs/cycle.

#if defined(__ARM_FEATURE_DOTPROD) || defined(__ARM_NEON)
static int32_t neon_dot_vdotq(const int8_t* __restrict a,
                               const int8_t* __restrict b, int n) {
    int32x4_t acc0 = vdupq_n_s32(0);
    int32x4_t acc1 = vdupq_n_s32(0);
    int32x4_t acc2 = vdupq_n_s32(0);
    int32x4_t acc3 = vdupq_n_s32(0);

    int i = 0;
    // Unroll by 4: keep 4 independent accumulators to hide vdotq latency
    for (; i + 63 < n; i += 64) {
        acc0 = vdotq_s32(acc0, vld1q_s8(a + i),      vld1q_s8(b + i));
        acc1 = vdotq_s32(acc1, vld1q_s8(a + i + 16), vld1q_s8(b + i + 16));
        acc2 = vdotq_s32(acc2, vld1q_s8(a + i + 32), vld1q_s8(b + i + 32));
        acc3 = vdotq_s32(acc3, vld1q_s8(a + i + 48), vld1q_s8(b + i + 48));
    }

    for (; i + 15 < n; i += 16) {
        acc0 = vdotq_s32(acc0, vld1q_s8(a + i), vld1q_s8(b + i));
    }

    int32_t sum = vaddvq_s32(acc0) + vaddvq_s32(acc1)
                + vaddvq_s32(acc2) + vaddvq_s32(acc3);

    for (; i < n; i++) {
        sum += (int32_t)a[i] * (int32_t)b[i];
    }
    return sum;
}
#else
static int32_t neon_dot_vdotq(const int8_t*, const int8_t*, int) {
    fprintf(stderr, "vdotq_s32 requires __ARM_FEATURE_DOTPROD\n");
    return 0;
}
#endif

// ============================================================================
// Runtime dispatch
// ============================================================================

typedef int32_t (*dot_func_t)(const int8_t*, const int8_t*, int);

static int32_t dispatch_dot(const int8_t* a, const int8_t* b, int n) {
#if defined(__ARM_FEATURE_DOTPROD)
    if (has_asimddp()) {
        return neon_dot_vdotq(a, b, n);
    }
#endif
    return neon_dot_widen(a, b, n);
}

// ============================================================================
// Benchmark wrappers
// ============================================================================

static int8_t* g_a = nullptr;
static int8_t* g_b = nullptr;
static int      g_n_dot = 0;
static volatile int32_t g_result = 0; // volatile to prevent DCE

__attribute__((noinline))
static void bench_scalar() { g_result = scalar_dot(g_a, g_b, g_n_dot); }
__attribute__((noinline))
static void bench_widen() { g_result = neon_dot_widen(g_a, g_b, g_n_dot); }
__attribute__((noinline))
static void bench_vdotq() { g_result = neon_dot_vdotq(g_a, g_b, g_n_dot); }

// ============================================================================
// Main
// ============================================================================

int main() {
    cpu_print_features();

    // Dot-product extension detection
    printf("  ASIMDDP (ARMv8.2-DotProd): %s\n", has_asimddp() ? "YES" : "NO");

    int n = N_DOT;
    g_n_dot = n;
    g_a = ALIGNED_ALLOC(int8_t, n, 64);
    g_b = ALIGNED_ALLOC(int8_t, n, 64);

    fill_random_i8(g_a, n);
    fill_random_i8(g_b, n);

    // --- Correctness ---
    printf("\n=== Correctness Checks ===\n");
    int32_t ref = scalar_dot(g_a, g_b, n);
    int32_t res_widen = neon_dot_widen(g_a, g_b, n);
    CHECK_EQ(res_widen, ref, "NEON widen int16 dot vs scalar");

#if defined(__ARM_FEATURE_DOTPROD)
    if (has_asimddp()) {
        int32_t res_vdot = neon_dot_vdotq(g_a, g_b, n);
        CHECK_EQ(res_vdot, ref, "NEON vdotq_s32 dot vs scalar");
    } else {
        printf("  [SKIP] vdotq_s32 -- CPU lacks ASIMDDP\n");
    }
#else
    printf("  [SKIP] vdotq_s32 -- compiled without __ARM_FEATURE_DOTPROD\n");
#endif

    // --- Path selection ---
    printf("\n=== Runtime Path Selection ===\n");
#if defined(__ARM_FEATURE_DOTPROD)
    if (has_asimddp()) {
        printf("  Using: vdotq_s32 path (ARMv8.2+, 4x int8 dot per instruction)\n");
    } else {
        printf("  Using: widen-to-int16 path (ARMv8.0, 2x int16 MAC)\n");
    }
#else
    printf("  Using: widen-to-int16 path (compiled without dotprod support)\n");
#endif
    printf("  ML mapping: Each vdotq_s32 = 4 int8 MACs = 1 quantized neuron weight*activation\n");
    printf("              This is the core operation in quantized GEMV/MLP inference.\n");

    // --- Benchmark ---
    printf("\n=== Benchmark: int8 Dot Product (N=%d) ===\n", n);
    benchmark_result_t results[3];

    BENCH_COMPUTE(bench_scalar(), n,
        (size_t)n * 2 * sizeof(int8_t), 30, results[0]);
    results[0].name = "scalar int8 dot";

    BENCH_COMPUTE(bench_widen(), n,
        (size_t)n * 2 * sizeof(int8_t), 30, results[1]);
    results[1].name = "NEON widen(int16) dot";

#if defined(__ARM_FEATURE_DOTPROD)
    if (has_asimddp()) {
        BENCH_COMPUTE(bench_vdotq(), n,
            (size_t)n * 2 * sizeof(int8_t), 30, results[2]);
        results[2].name = "NEON vdotq_s32 dot";
        bench_report(results, 3);
    } else {
        bench_report(results, 2);
    }
#else
    bench_report(results, 2);
#endif

    // Avoid dead code elimination of g_result
    printf("  Verification checksum: %d\n", (int)g_result);

    ALIGNED_FREE(g_a);
    ALIGNED_FREE(g_b);
    return 0;
}
