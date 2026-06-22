/**
 * AVX-512 Byte Scanning (find position of specific byte in buffer)
 *
 * Demonstrates:
 *   - _mm512_cmpeq_epi8_mask: compare 64 bytes, produce 64-bit mask
 *   - _tzcnt_u32: count trailing zeros (find first match)
 *   - _mm512_mask_cmpeq_epi8_mask for tail handling
 *   - N = 10000000
 *   - Compare: AVX-512 byte scan vs memchr vs scalar loop
 *   - Use case: newline scanning, JSON parsing, string search
 *   - Report GB/s scanned
 *
 * Why AVX-512 byte ops are fast: 64 bytes per compare, single cycle,
 * mask-based branch prediction friendly.
 */

#include <immintrin.h>
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

static int cpu_has_avx512bw() {
#ifndef __linux__
    return 0;
#else
    unsigned int eax, ebx, ecx, edx;
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return 0;
    return (ebx & (1 << 30)) != 0;  /* AVX-512BW for byte/word ops */
#endif
}

/* ======================================================================== */
/*  Scalar byte scan: returns position of first `target` byte, or -1        */
/* ======================================================================== */
int64_t byte_scan_scalar(const uint8_t *data, int64_t n, uint8_t target) {
    for (int64_t i = 0; i < n; i++) {
        if (data[i] == target) return i;
    }
    return -1;
}

/* ======================================================================== */
/*  memchr wrapper (libc optimized)                                         */
/* ======================================================================== */
int64_t byte_scan_memchr(const uint8_t *data, int64_t n, uint8_t target) {
    const uint8_t *pos = (const uint8_t*)memchr(data, (int)target, (size_t)n);
    if (pos) return pos - data;
    return -1;
}

/* ======================================================================== */
/*  AVX-512 byte scan: 64 bytes per iteration                               */
/*                                                                          */
/*  K mask has 1 bit per byte. If mask != 0, target found.                  */
/*  _tzcnt_u64(mask) gives position within the 64-byte chunk.              */
/* ======================================================================== */
int64_t byte_scan_avx512(const uint8_t *data, int64_t n, uint8_t target) {
    __m512i vtarget = _mm512_set1_epi8((char)target);
    int64_t i = 0;

    /* Fast path: aligned blocks of 64 */
    for (; i + 63 < n; i += 64) {
        __m512i vdata = _mm512_loadu_si512((const __m512i*)(data + i));
        __mmask64 mask = _mm512_cmpeq_epi8_mask(vdata, vtarget);
        if (mask != 0) {
            /* Found! Count trailing zeros to find position */
            return i + __builtin_ctzll(mask);
        }
    }

    /* Tail: use masked compare to avoid reading past buffer */
    if (i < n) {
        int64_t rem = n - i;
        __mmask64 mask = (1ULL << rem) - 1;
        __m512i vdata = _mm512_maskz_loadu_epi8(mask, data + i);
        __mmask64 result = _mm512_mask_cmpeq_epi8_mask(mask, vdata, vtarget);
        if (result != 0) {
            return i + __builtin_ctzll(result);
        }
    }

    return -1;
}

/* ======================================================================== */
/*  AVX-512 scan: find newline (0x0a) - specialized version                 */
/* ======================================================================== */
int64_t newline_scan_avx512(const uint8_t *data, int64_t n) {
    return byte_scan_avx512(data, n, 0x0a);
}

int64_t newline_scan_scalar(const uint8_t *data, int64_t n) {
    return byte_scan_scalar(data, n, 0x0a);
}

/* ======================================================================== */
/*  AVX2 byte scan: 32 bytes per iteration (for comparison)                 */
/* ======================================================================== */
int64_t byte_scan_avx2(const uint8_t *data, int64_t n, uint8_t target) {
    __m256i vtarget = _mm256_set1_epi8((char)target);
    int64_t i = 0;

    for (; i + 31 < n; i += 32) {
        __m256i vdata = _mm256_loadu_si256((const __m256i*)(data + i));
        __m256i cmp = _mm256_cmpeq_epi8(vdata, vtarget);
        int mask = _mm256_movemask_epi8(cmp);
        if (mask != 0) {
            return i + __builtin_ctz(mask);
        }
    }

    if (i < n) {
        int64_t rem = n - i;
        /* Pad with non-matching bytes */
        uint8_t padded[32];
        memset(padded, target ^ 1, 32);  /* fill with != target */
        memcpy(padded, data + i, (size_t)rem);

        __m256i vdata = _mm256_loadu_si256((const __m256i*)padded);
        __m256i cmp = _mm256_cmpeq_epi8(vdata, vtarget);
        int mask = _mm256_movemask_epi8(cmp);
        if (mask != 0) {
            int64_t pos = __builtin_ctz(mask);
            if (pos < rem) return i + pos;
        }
    }

    return -1;
}

/* ======================================================================== */
/*  Count occurrences (demonstrate throughput, not just find-first)         */
/* ======================================================================== */
int64_t count_byte_scalar(const uint8_t *data, int64_t n, uint8_t target) {
    int64_t count = 0;
    for (int64_t i = 0; i < n; i++) {
        if (data[i] == target) count++;
    }
    return count;
}

int64_t count_byte_avx512(const uint8_t *data, int64_t n, uint8_t target) {
    __m512i vtarget = _mm512_set1_epi8((char)target);
    int64_t total = 0;
    int64_t i = 0;

    for (; i + 63 < n; i += 64) {
        __m512i vdata = _mm512_loadu_si512((const __m512i*)(data + i));
        __mmask64 mask = _mm512_cmpeq_epi8_mask(vdata, vtarget);
        total += __builtin_popcountll(mask);
    }

    if (i < n) {
        int64_t rem = n - i;
        __mmask64 tail_mask = (1ULL << rem) - 1;
        __m512i vdata = _mm512_maskz_loadu_epi8(tail_mask, data + i);
        __mmask64 mask = _mm512_cmpeq_epi8_mask(vdata, vtarget);
        mask &= tail_mask;
        total += __builtin_popcountll(mask);
    }

    return total;
}

/* ======================================================================== */
/*  Main                                                                    */
/* ======================================================================== */
int main() {
    const int64_t N = 10000000LL;  /* 10 million bytes */

    printf("=== AVX-512 Byte Scanning ===\n");
    printf("N = %lld bytes (%.2f MB)\n", (long long)N, (double)N / 1e6);
    printf("SIMD widths: AVX2=32 bytes, AVX-512=64 bytes per compare\n");
    printf("Use case: newline scanning, JSON parsing, string search\n\n");

    if (!cpu_has_avx512f() || !cpu_has_avx512bw()) {
        printf("AVX-512F or AVX-512BW not available on this CPU.\n");
        printf("AVX-512 byte operations require AVX-512BW.\n");
        printf("Falling back to AVX2 byte scan for comparison.\n");
    } else {
        printf("AVX-512F: YES, AVX-512BW: YES\n");
    }
    printf("\n");

    uint8_t *data = (uint8_t*)aligned_alloc(64, (size_t)(N + 64));

    /* Fill with random printable ASCII, insert targets periodically */
    srand(42);
    for (int64_t i = 0; i < N; i++) {
        data[i] = (uint8_t)(32 + rand() % 95);  /* printable */
    }

    /* Insert newline characters at known positions */
    data[100] = '\n';           /* early match */
    data[N / 2] = '\n';         /* mid match */
    data[N - 256] = '\n';       /* late match */

    /* Also insert some 'X' for testing generic byte search */
    data[42] = 'X';
    data[N - 100] = 'X';

    /* --- Find-first tests --- */

    /* Test 1: Find 'X' */
    printf("--- Test 1: Find first 'X' ---\n");
    int64_t pos_scalar = byte_scan_scalar(data, N, 'X');
    int64_t pos_memchr = byte_scan_memchr(data, N, 'X');
    int64_t pos_avx2   = byte_scan_avx2(data, N, 'X');
    int64_t pos_avx512 = byte_scan_avx512(data, N, 'X');

    printf("  Scalar:  pos=%lld  %s\n", (long long)pos_scalar,
           pos_scalar == 42 ? "OK" : "FAIL");
    printf("  memchr:  pos=%lld  %s\n", (long long)pos_memchr,
           pos_memchr == 42 ? "OK" : "FAIL");
    printf("  AVX2:    pos=%lld  %s\n", (long long)pos_avx2,
           pos_avx2 == 42 ? "OK" : "FAIL");
    printf("  AVX-512: pos=%lld  %s\n", (long long)pos_avx512,
           pos_avx512 == 42 ? "OK" : "FAIL");

    /* Test 2: Find newline at N/2 */
    printf("\n--- Test 2: Find newline at mid-point ---\n");
    int64_t nl_scalar = newline_scan_scalar(data, N);
    int64_t nl_memchr  = byte_scan_memchr(data, N, '\n');
    int64_t nl_avx512  = newline_scan_avx512(data, N);
    int64_t expected_nl = 100;

    printf("  Scalar:  pos=%lld  %s\n", (long long)nl_scalar,
           nl_scalar == expected_nl ? "OK" : "FAIL");
    printf("  memchr:  pos=%lld  %s\n", (long long)nl_memchr,
           nl_memchr == expected_nl ? "OK" : "FAIL");
    printf("  AVX-512: pos=%lld  %s\n", (long long)nl_avx512,
           nl_avx512 == expected_nl ? "OK" : "FAIL");

    /* Test 3: Count 'e' (letter e) */
    printf("\n--- Test 3: Count character 'e' ---\n");
    int64_t count_scalar = count_byte_scalar(data, N, 'e');
    int64_t count_avx512 = count_byte_avx512(data, N, 'e');
    printf("  Scalar:  count=%lld\n", (long long)count_scalar);
    printf("  AVX-512: count=%lld  %s\n", (long long)count_avx512,
           count_avx512 == count_scalar ? "OK" : "FAIL");

    /* --- Benchmark: find-first (data without target = scan entire buffer) --- */
    /* Create a buffer without 'X' to force full scan */
    uint8_t *no_target = (uint8_t*)aligned_alloc(64, (size_t)(N + 64));
    for (int64_t i = 0; i < N; i++)
        no_target[i] = (uint8_t)(32 + rand() % 94 + 1);  /* never 'X' or 0x00 */

    printf("\n--- Performance: find-first (scan entire buffer, no match) ---\n");
    int iters = 500;

    double t0 = get_time_sec();
    for (int k = 0; k < iters; k++)
        byte_scan_scalar(no_target, N, 0x00);  /* 0x00 not present */
    double t_scalar = (get_time_sec() - t0) / iters;

    double t1 = get_time_sec();
    for (int k = 0; k < iters; k++)
        byte_scan_memchr(no_target, N, 0x00);
    double t_memchr = (get_time_sec() - t1) / iters;

    double t2 = get_time_sec();
    for (int k = 0; k < iters; k++)
        byte_scan_avx2(no_target, N, 0x00);
    double t_avx2 = (get_time_sec() - t2) / iters;

    double t3 = get_time_sec();
    for (int k = 0; k < iters; k++)
        byte_scan_avx512(no_target, N, 0x00);
    double t_avx512 = (get_time_sec() - t3) / iters;

    printf("  Scalar:   %7.1f us  (%.2f GB/s)\n",
           t_scalar * 1e6, (double)N / t_scalar / 1e9);
    printf("  memchr:   %7.1f us  (%.2f GB/s)\n",
           t_memchr * 1e6, (double)N / t_memchr / 1e9);
    printf("  AVX2:     %7.1f us  (%.2f GB/s)\n",
           t_avx2 * 1e6, (double)N / t_avx2 / 1e9);
    printf("  AVX-512:  %7.1f us  (%.2f GB/s)\n",
           t_avx512 * 1e6, (double)N / t_avx512 / 1e9);

    /* --- Benchmark: count (always full scan) --- */
    printf("\n--- Performance: count byte (always full scan) ---\n");
    iters = 200;

    double tc0 = get_time_sec();
    for (int k = 0; k < iters; k++)
        count_byte_scalar(data, N, 'e');
    double tc_scalar = (get_time_sec() - tc0) / iters;

    double tc1 = get_time_sec();
    for (int k = 0; k < iters; k++)
        count_byte_avx512(data, N, 'e');
    double tc_avx512 = (get_time_sec() - tc1) / iters;

    printf("  Scalar:   %7.1f us  (%.2f GB/s)\n",
           tc_scalar * 1e6, (double)N / tc_scalar / 1e9);
    printf("  AVX-512:  %7.1f us  (%.2f GB/s, %.2fx speedup)\n",
           tc_avx512 * 1e6, (double)N / tc_avx512 / 1e9,
           tc_scalar / tc_avx512);

    /* --- Explanation --- */
    printf("\n--- Why AVX-512 Byte Operations Are Fast ---\n");
    printf("1. 64 bytes per compare: single instruction processes a cache line.\n");
    printf("2. _mm512_cmpeq_epi8_mask produces a 64-bit mask directly into \n");
    printf("   a general-purpose register (k0-k7). No movemask needed.\n");
    printf("3. _tzcnt_u64 / _popcnt64 on the mask is 1 cycle.\n");
    printf("4. The mask test + branch is highly predictable.\n");
    printf("5. AVX-512BW provides byte-granularity operations.\n");
    printf("6. No lane-crossing shuffle overhead like pre-AVX-512.\n\n");

    printf("Applications:\n");
    printf("  - JSON/XML parsing: find next quote, colon, brace.\n");
    printf("  - CSV parsing: find next comma or newline.\n");
    printf("  - Log processing: find timestamp boundaries.\n");
    printf("  - String search: faster than memchr for large scans.\n");
    printf("  - simdjson uses similar techniques (with ARM NEON too).\n");

    free(data); free(no_target);
    return 0;
}
