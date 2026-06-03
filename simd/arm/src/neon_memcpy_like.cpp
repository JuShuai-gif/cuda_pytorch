#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <arm_neon.h>
#include "../../common/benchmark.h"
#include "../../common/check.h"
#include "../../common/random_data.h"
#include "../../common/aligned_buffer.h"

// =============================================================================
// neon_memcpy_like -- Memory copy with NEON, unrolling, and prefetch hints
//   dst = src (simple copy)
//   SIMD width: 16 bytes per 128-bit NEON register (vld1q_u8 / vst1q_u8)
//   N = 10000000 (10M elements = 10 MB of uint8)
//
//   Variants:
//     1. scalar byte loop
//     2. standard memcpy (libc)
//     3. NEON: vld1q + vst1q, 16 bytes per iteration
//     4. NEON unrolled: 4 loads + 4 stores, 64 bytes per iteration
//     5. NEON unrolled + prefetch hints (read-ahead on src, write-hint on dst)
// =============================================================================

static const size_t N = 10000000;
static const int    BENCH_ITERS = 5;

// ---- scalar byte copy ----
static void scalar_copy_u8(const uint8_t* src, uint8_t* dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = src[i];
    }
}

// ---- NEON copy: 16 bytes per iteration ----
static void neon_copy_u8(const uint8_t* src, uint8_t* dst, size_t n) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        uint8x16_t v = vld1q_u8(src + i);
        vst1q_u8(dst + i, v);
    }
    for (; i < n; i++) {
        dst[i] = src[i];
    }
}

// ---- NEON unrolled: 4x loads + 4x stores = 64 bytes per iteration ----
static void neon_copy_unrolled_u8(const uint8_t* src, uint8_t* dst, size_t n) {
    size_t i = 0;
    for (; i + 64 <= n; i += 64) {
        uint8x16_t v0 = vld1q_u8(src + i +  0);
        uint8x16_t v1 = vld1q_u8(src + i + 16);
        uint8x16_t v2 = vld1q_u8(src + i + 32);
        uint8x16_t v3 = vld1q_u8(src + i + 48);

        vst1q_u8(dst + i +  0, v0);
        vst1q_u8(dst + i + 16, v1);
        vst1q_u8(dst + i + 32, v2);
        vst1q_u8(dst + i + 48, v3);
    }
    // Partial: single 16-byte vector
    for (; i + 16 <= n; i += 16) {
        uint8x16_t v = vld1q_u8(src + i);
        vst1q_u8(dst + i, v);
    }
    for (; i < n; i++) {
        dst[i] = src[i];
    }
}

// ---- NEON unrolled + prefetch: read-ahead on src, write-hint on dst ----
// __builtin_prefetch(addr, 0=read, 3=non-temporal) on source to pull into cache
// __builtin_prefetch(addr, 1=write, 3=non-temporal) on dst to avoid cache pollution
static void neon_copy_prefetch_u8(const uint8_t* src, uint8_t* dst, size_t n) {
    size_t i = 0;
    // Prefetch distance: 512 bytes (8 cache lines on 64-byte cache line systems)
    const size_t PF_DIST = 512;
    for (; i + 64 <= n; i += 64) {
        // Prefetch ahead
        if (i + PF_DIST < n) {
            __builtin_prefetch(src + i + PF_DIST, 0, 3);
            __builtin_prefetch(dst + i + PF_DIST, 1, 3);
        }

        uint8x16_t v0 = vld1q_u8(src + i +  0);
        uint8x16_t v1 = vld1q_u8(src + i + 16);
        uint8x16_t v2 = vld1q_u8(src + i + 32);
        uint8x16_t v3 = vld1q_u8(src + i + 48);

        vst1q_u8(dst + i +  0, v0);
        vst1q_u8(dst + i + 16, v1);
        vst1q_u8(dst + i + 32, v2);
        vst1q_u8(dst + i + 48, v3);
    }
    for (; i + 16 <= n; i += 16) {
        uint8x16_t v = vld1q_u8(src + i);
        vst1q_u8(dst + i, v);
    }
    for (; i < n; i++) {
        dst[i] = src[i];
    }
}

// ---- memcpy wrapper for BENCH_COMPUTE compatibility ----
static void libc_memcpy_wrapper(const uint8_t* src, uint8_t* dst, size_t n) {
    memcpy(dst, src, n);
}

// =============================================================================
// main
// =============================================================================
int main(void) {
    printf("================================================================\n");
    printf("  NEON Memory Copy -- Unrolling and Prefetch\n");
    printf("  SIMD width: 16 bytes per 128-bit NEON register (uint8x16)\n");
    printf("  N = %zu bytes (%.2f MB)\n", N, (double)N / (1024.0 * 1024.0));
    printf("================================================================\n");

    uint8_t* src = ALIGNED_ALLOC(uint8_t, N, 16);
    uint8_t* ref = ALIGNED_ALLOC(uint8_t, N, 16);
    uint8_t* dst = ALIGNED_ALLOC(uint8_t, N, 16);

    CHECK_TRUE(is_aligned(src, 16), "src buffer is 16-byte aligned");
    CHECK_TRUE(is_aligned(dst, 16), "dst buffer is 16-byte aligned");

    fill_random_u8(src, N);

    // ---- Correctness ----
    printf("\n-- Correctness --\n");

    // scalar reference
    memset(ref, 0xAA, N);
    scalar_copy_u8(src, ref, N);

    // NEON basic
    memset(dst, 0xAA, N);
    neon_copy_u8(src, dst, N);
    CHECK_EQ(memcmp(ref, dst, N), 0, "neon_copy matches scalar");

    // NEON unrolled
    memset(dst, 0xAA, N);
    neon_copy_unrolled_u8(src, dst, N);
    CHECK_EQ(memcmp(ref, dst, N), 0, "neon_unrolled matches scalar");

    // NEON prefetch
    memset(dst, 0xAA, N);
    neon_copy_prefetch_u8(src, dst, N);
    CHECK_EQ(memcmp(ref, dst, N), 0, "neon_prefetch matches scalar");

    // libc memcpy
    memset(dst, 0xAA, N);
    libc_memcpy_wrapper(src, dst, N);
    CHECK_EQ(memcmp(ref, dst, N), 0, "libc_memcpy matches scalar");

    // ---- Benchmarks ----
    printf("\n-- Benchmarks (%d timed iterations) --\n", BENCH_ITERS);

    // bytes_processed = read N bytes + write N bytes = 2 * N
    size_t total_bytes = N * 2;

    benchmark_result_t results[5];

    BENCH_COMPUTE(scalar_copy_u8(src, dst, N), N, total_bytes, BENCH_ITERS, results[0]);
    results[0].name = "scalar_byte_copy";

    BENCH_COMPUTE(libc_memcpy_wrapper(src, dst, N), N, total_bytes, BENCH_ITERS, results[1]);
    results[1].name = "libc_memcpy";

    BENCH_COMPUTE(neon_copy_u8(src, dst, N), N, total_bytes, BENCH_ITERS, results[2]);
    results[2].name = "neon_copy (16B/iter)";

    BENCH_COMPUTE(neon_copy_unrolled_u8(src, dst, N), N, total_bytes, BENCH_ITERS, results[3]);
    results[3].name = "neon_unrolled (64B/iter)";

    BENCH_COMPUTE(neon_copy_prefetch_u8(src, dst, N), N, total_bytes, BENCH_ITERS, results[4]);
    results[4].name = "neon_prefetch+unroll";

    bench_report(results, 5);

    // ---- Bandwidth Summary ----
    printf("Bandwidth Summary (GB/s):\n");
    for (int i = 0; i < 5; i++) {
        printf("  %-30s %8.2f GB/s  (%s)\n",
               results[i].name,
               results[i].gb_per_sec,
               (i == 0) ? "baseline" : "---");
    }
    printf("\nAnalysis:\n");
    printf("  - libc memcpy is heavily optimized (may use STP/LDP or DC ZVA)\n");
    printf("  - NEON unrolling reduces loop overhead; 4x unroll = fewer branches\n");
    printf("  - __builtin_prefetch hints help on systems with lower\n"
           "    memory-level parallelism (MLP)\n");
    printf("  - On modern ARM cores, libc memcpy typically wins for large copies\n");

    ALIGNED_FREE(src);
    ALIGNED_FREE(ref);
    ALIGNED_FREE(dst);

    printf("\nAll tests passed.\n");
    return 0;
}
