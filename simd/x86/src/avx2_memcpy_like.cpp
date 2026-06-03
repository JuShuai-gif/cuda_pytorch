/*
 * avx2_memcpy_like.cpp -- AVX2 memory copy operations: src -> dst
 *
 * SIMD width: 256-bit = 32 bytes per register
 * N = 10,000,000 (10M floats = 40 MB)
 *
 * Variants:
 *   1. Scalar:           element-by-element copy
 *   2. Simple 32-byte:   1x _mm256_loadu_si256 + _mm256_storeu_si256
 *   3. 4x unrolled:      128 bytes per iteration
 *   4. Non-temporal:     _mm256_stream_si256 (bypass cache, write-combining)
 *   5. std::memcpy:      standard library for comparison
 *
 * Non-temporal stores are useful when the destination data will not be
 * read again soon. They bypass the cache hierarchy entirely and write
 * directly to memory using write-combining buffers (64B fill buffers).
 * This avoids:
 *   - Cache pollution (evicting useful data)
 *   - Read-for-ownership (RFO) on the destination cache line
 *
 * Use streaming stores when:
 *   - dst >> LLC size (won't be reused before eviction)
 *   - src fits in cache but dst doesn't
 *   - You're the sole producer of dst (no sharing)
 *
 * Report: GB/s throughput for each variant.
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

static const size_t N = 10000000; /* 10M floats = 40 MB */

/* ================================================================
 * Scalar baseline
 * ================================================================ */

static void scalar_copy(const float* src, float* dst, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = src[i];
}

/* ================================================================
 * Simple 32-byte AVX2 copy (1x)
 * ================================================================ */

static void avx2_copy_1x(const float* src, float* dst, size_t n) {
    size_t count = n * sizeof(float);
    const char* s = (const char*)src;
    char* d = (char*)dst;
    size_t i = 0;

    for (; i + 32 <= count; i += 32) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(s + i));
        _mm256_storeu_si256((__m256i*)(d + i), v);
    }

    /* Scalar tail: copy remaining bytes */
    if (i < count) {
        memcpy(d + i, s + i, count - i);
    }
}

/* ================================================================
 * 4x unrolled AVX2 copy (128 bytes per iteration)
 *
 * Unrolling reduces loop overhead (branch, counter update) and
 * amortizes instruction fetch across more data. On modern CPUs,
 * the loop stream detector (LSD) can handle small unrolled loops
 * efficiently, but explicit unrolling removes dependency on it.
 * ================================================================ */

static void avx2_copy_4x(const float* src, float* dst, size_t n) {
    size_t count = n * sizeof(float);
    const char* s = (const char*)src;
    char* d = (char*)dst;
    size_t i = 0;

    for (; i + 128 <= count; i += 128) {
        __m256i v0 = _mm256_loadu_si256((const __m256i*)(s + i));
        __m256i v1 = _mm256_loadu_si256((const __m256i*)(s + i + 32));
        __m256i v2 = _mm256_loadu_si256((const __m256i*)(s + i + 64));
        __m256i v3 = _mm256_loadu_si256((const __m256i*)(s + i + 96));

        _mm256_storeu_si256((__m256i*)(d + i), v0);
        _mm256_storeu_si256((__m256i*)(d + i + 32), v1);
        _mm256_storeu_si256((__m256i*)(d + i + 64), v2);
        _mm256_storeu_si256((__m256i*)(d + i + 96), v3);
    }

    /* Tail: use 1x copy */
    for (; i + 32 <= count; i += 32) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(s + i));
        _mm256_storeu_si256((__m256i*)(d + i), v);
    }

    if (i < count) {
        memcpy(d + i, s + i, count - i);
    }
}

/* ================================================================
 * Non-temporal (streaming) AVX2 copy
 *
 * _mm256_stream_si256 writes directly to memory, bypassing the cache.
 * Uses write-combining buffers (6-10 per core on Intel, typically
 * 4 on AMD). Data is held in WC buffer until a full cache line (64B)
 * is accumulated, then flushed.
 *
 * IMPORTANT: destination must be 32-byte aligned for streaming stores
 *            to work at full efficiency. On most CPUs, misaligned
 *            streaming stores can be significantly slower.
 *
 * WARNING: After streaming stores, an SFENCE (_mm_sfence) is needed
 *          before reading back the data to ensure visibility.
 * ================================================================ */

static void avx2_copy_stream(const float* src, float* dst, size_t n) {
    size_t count = n * sizeof(float);
    const char* s = (const char*)src;
    char* d = (char*)dst;
    size_t i = 0;

    /* Prefetch source data into L1 cache for streaming copy */
    for (size_t pf = 0; pf < count; pf += 256) {
        _mm_prefetch(s + pf, _MM_HINT_T0);
    }

    for (; i + 128 <= count; i += 128) {
        __m256i v0 = _mm256_loadu_si256((const __m256i*)(s + i));
        __m256i v1 = _mm256_loadu_si256((const __m256i*)(s + i + 32));
        __m256i v2 = _mm256_loadu_si256((const __m256i*)(s + i + 64));
        __m256i v3 = _mm256_loadu_si256((const __m256i*)(s + i + 96));

        _mm256_stream_si256((__m256i*)(d + i), v0);
        _mm256_stream_si256((__m256i*)(d + i + 32), v1);
        _mm256_stream_si256((__m256i*)(d + i + 64), v2);
        _mm256_stream_si256((__m256i*)(d + i + 96), v3);
    }

    for (; i + 32 <= count; i += 32) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(s + i));
        _mm256_stream_si256((__m256i*)(d + i), v);
    }

    /* SFENCE: ensure all streaming stores are visible before proceeding */
    _mm_sfence();

    if (i < count) {
        memcpy(d + i, s + i, count - i);
    }
}

/* ================================================================
 * Standard memcpy wrapper (for comparison)
 * ================================================================ */

static void std_memcpy_wrapper(const float* src, float* dst, size_t n) {
    memcpy(dst, src, n * sizeof(float));
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

    printf("\n=== AVX2 Memory Copy (N = %zu = %.1f MB) ===\n\n",
           N, (double)(N * sizeof(float)) / (1024.0 * 1024.0));

    /* Allocate -- use aligned buffers for fair comparison */
    float* src = ALIGNED_ALLOC(float, N, 32);
    float* dst = ALIGNED_ALLOC(float, N, 32);
    float* dst_ref = ALIGNED_ALLOC(float, N, 32);

    if (!src || !dst || !dst_ref) {
        fprintf(stderr, "Allocation failed.\n");
        return 1;
    }

    /* Fill source with random data */
    rand_xorshift64_seed(42);
    fill_random_f32(src, N);

    /* ---- Correctness ---- */

    printf("--- Correctness ---\n");

    /* Generate reference */
    memset(dst_ref, 0, N * sizeof(float));
    scalar_copy(src, dst_ref, N);

    memset(dst, 0, N * sizeof(float));
    avx2_copy_1x(src, dst, N);
    CHECK_NEAR_ARRAY(dst, dst_ref, N, 0.0f, "avx2_copy_1x matches scalar");

    memset(dst, 0, N * sizeof(float));
    avx2_copy_4x(src, dst, N);
    CHECK_NEAR_ARRAY(dst, dst_ref, N, 0.0f, "avx2_copy_4x matches scalar");

    memset(dst, 0, N * sizeof(float));
    avx2_copy_stream(src, dst, N);
    CHECK_NEAR_ARRAY(dst, dst_ref, N, 0.0f, "avx2_copy_stream matches scalar");

    memset(dst, 0, N * sizeof(float));
    std_memcpy_wrapper(src, dst, N);
    CHECK_NEAR_ARRAY(dst, dst_ref, N, 0.0f, "std_memcpy matches scalar");

    /* ---- Benchmark ---- */

    /*
     * Bytes processed = read src + write dst = 2 * N * sizeof(float)
     * For streaming stores, cache bypass means we measure pure
     * memory bandwidth (write-combining). Regular stores measure
     * cache hierarchy throughput.
     */

    const size_t bytes_copy = N * 2 * sizeof(float); /* 80 MB for 10M floats */

    benchmark_result_t results[5];
    memset(results, 0, sizeof(results));

    BENCH_MEMORY(scalar_copy(src, dst, N),
                 N, bytes_copy, 20, results[0]);
    results[0].name = "scalar_copy";

    BENCH_MEMORY(std_memcpy_wrapper(src, dst, N),
                 N, bytes_copy, 20, results[1]);
    results[1].name = "std::memcpy";

    BENCH_MEMORY(avx2_copy_1x(src, dst, N),
                 N, bytes_copy, 20, results[2]);
    results[2].name = "avx2_copy_1x (32B)";

    BENCH_MEMORY(avx2_copy_4x(src, dst, N),
                 N, bytes_copy, 20, results[3]);
    results[3].name = "avx2_copy_4x (128B)";

    BENCH_MEMORY(avx2_copy_stream(src, dst, N),
                 N, bytes_copy, 20, results[4]);
    results[4].name = "avx2_copy_stream (NT)";

    printf("\n--- Benchmark Results ---\n");
    printf("SIMD width: 256-bit (32 bytes)\n");
    bench_report(results, 5);

    printf("Notes:\n");
    printf("  - memcpy is highly optimized (glibc uses rep movsb with ERMS\n");
    printf("    on modern Intel, plus software fallback for small/large copies).\n");
    printf("    It is the gold standard for memory copy.\n");
    printf("  - 4x unrolling (128B/iter) reduces loop overhead vs 1x (32B/iter).\n");
    printf("  - Non-temporal stores (_mm256_stream_si256):\n");
    printf("    * Bypass L1/L2/L3 cache (no read-for-ownership).\n");
    printf("    * Use write-combining buffers (typically 64B per buffer).\n");
    printf("    * Best when dst data will NOT be reused soon (avoids cache\n");
    printf("      pollution and RFO overhead).\n");
    printf("    * Requires SFENCE before reading back destination.\n");
    printf("    * Can be slower for small copies or when data fits in cache,\n");
    printf("      because streaming stores have higher latency than cache\n");
    printf("      line fills.\n");
    printf("  - The measured GB/s approaches DRAM bandwidth for large (40MB)\n");
    printf("    copies that exceed LLC capacity.\n");

    ALIGNED_FREE(src);
    ALIGNED_FREE(dst);
    ALIGNED_FREE(dst_ref);

    return 0;
}
