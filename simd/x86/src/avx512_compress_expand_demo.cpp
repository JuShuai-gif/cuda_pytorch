/**
 * avx512_compress_expand_demo.cpp -- 压缩/展开操作演示
 *
 * 演示 AVX-512F 的压缩与展开指令：
 *   1. 压缩: 将选中元素紧密打包 (稀疏 → 稠密)
 *   2. 展开: 将稠密元素散列到稀疏位置
 *   3. 过滤操作: 保留满足条件的元素
 *   4. 稀疏矩阵组装: 从稀疏表示重建稠密矩阵
 *
 * 需要: -mavx512f -mavx512bw
 */

#include "../../common/aligned_buffer.h"
#include "../../common/benchmark.h"
#include "../../common/check.h"
#include "../../common/cpu_features.h"
#include "../../common/random_data.h"

#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef __AVX512F__
#error "This file requires -mavx512f compiler flag"
#endif

/* ================================================================
 * 1. 过滤: 保留 > 0 的元素 (压缩)
 * ================================================================ */

__attribute__((noinline))
static size_t filter_positive_avx512(const float* src, float* dst, size_t n) {
    const __m512 zero = _mm512_setzero_ps();
    size_t written = 0;

    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(src + i);
        __mmask16 pos = _mm512_cmp_ps_mask(v, zero, _CMP_GT_OQ);
        _mm512_mask_compressstoreu_ps(dst + written, pos, v);
        written += (size_t)__builtin_popcount((unsigned int)pos);
    }

    for (; i < n; i++)
        if (src[i] > 0.0f) dst[written++] = src[i];

    return written;
}

__attribute__((noinline))
static size_t filter_positive_scalar(const float* src, float* dst, size_t n) {
    size_t w = 0;
    for (size_t i = 0; i < n; i++)
        if (src[i] > 0.0f) dst[w++] = src[i];
    return w;
}

/* ================================================================
 * 2. 展开: 将稠密元素散列到掩码位置
 * ================================================================ */

__attribute__((noinline))
static void expand_to_mask_avx512(const float* dense, float* sparse,
                                   __mmask16 mask, int count) {
    (void)count; /* count = popcount(mask) */
    __m512 result = _mm512_maskz_expandloadu_ps(mask, dense);
    _mm512_storeu_ps(sparse, result);
}

__attribute__((noinline))
static void expand_to_mask_scalar(const float* dense, float* sparse,
                                   __mmask16 mask, int count) {
    (void)count;
    int di = 0;
    for (int i = 0; i < 16; i++) {
        if (mask & (1u << i)) {
            sparse[i] = dense[di++];
        } else {
            sparse[i] = 0.0f;
        }
    }
}

/* ================================================================
 * 3. 压缩 + 展开来回测试
 *
 * 将向量压缩为稠密格式，再展开回稀疏格式。
 * 结果应与原始值一致（非掩码位置清零）。
 * ================================================================ */

__attribute__((noinline))
static void compress_expand_roundtrip_avx512(const float* src, float* dst,
                                              __mmask16 mask) {
    float tmp[16];
    __m512 v = _mm512_loadu_ps(src);
    _mm512_mask_compressstoreu_ps(tmp, mask, v);
    __m512 result = _mm512_maskz_expandloadu_ps(mask, tmp);
    _mm512_storeu_ps(dst, result);
}

/* ================================================================
 * 4. 带计数的过滤 (生产级模式)
 * ================================================================ */

__attribute__((noinline))
static size_t filter_values_avx512(const float* src, float* dst, size_t n,
                                    float lo, float hi) {
    const __m512 vlo = _mm512_set1_ps(lo);
    const __m512 vhi = _mm512_set1_ps(hi);
    size_t written = 0;

    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(src + i);

        /* 选择满足 lo <= v <= hi 的元素 */
        __mmask16 gt_lo = _mm512_cmp_ps_mask(v, vlo, _CMP_GE_OQ);
        __mmask16 lt_hi = _mm512_cmp_ps_mask(v, vhi, _CMP_LE_OQ);
        __mmask16 in_range = _kand_mask16(gt_lo, lt_hi);

        _mm512_mask_compressstoreu_ps(dst + written, in_range, v);
        written += (size_t)__builtin_popcount((unsigned int)in_range);
    }
    for (; i < n; i++)
        if (src[i] >= lo && src[i] <= hi) dst[written++] = src[i];

    return written;
}

/* ================================================================
 * 基准测试基础设施
 * ================================================================ */

static const size_t N = 100000;
static float* g_src = NULL;
static float* g_dst = NULL;
static size_t  g_written = 0;

__attribute__((noinline)) static void bn_filter_scalar() {
    g_written = filter_positive_scalar(g_src, g_dst, N);
}
__attribute__((noinline)) static void bn_filter_avx512() {
    g_written = filter_positive_avx512(g_src, g_dst, N);
}

/* ================================================================
 * 主函数
 * ================================================================ */

int main() {
    cpu_print_features();

    if (!cpu_has_avx512f()) {
        printf("AVX-512F not supported. Exiting.\n");
        return 0;
    }

    printf("\n=== AVX-512 Compress/Expand Demo ===\n");
    printf("ISA: AVX-512F (_mm512_mask_compressstoreu_ps / "
           "_mm512_maskz_expandloadu_ps)\n\n");

    /* ---- 测试 1: 压缩/展开来回测试 ---- */
    printf("--- Compress/Expand Round-Trip ---\n");
    {
        float src[16], dst[16], ref[16];
        for (int i = 0; i < 16; i++) src[i] = (float)(i + 1);

        __mmask16 mask = 0b1010010110100101; /* 选中 8 个元素 */

        compress_expand_roundtrip_avx512(src, dst, mask);
        expand_to_mask_scalar(src, ref, mask, 8);

        int ok = 1;
        for (int i = 0; i < 16; i++)
            if (fabsf(dst[i] - ref[i]) > 1e-6f) { ok = 0; break; }
        printf("  [%s] Compress+expand round-trip (mask=0x%04X)\n",
               ok ? "PASS" : "FAIL", (unsigned int)mask);

        if (!ok) {
            printf("  Expected: ");
            for (int i = 0; i < 16; i++) printf("%.0f ", ref[i]);
            printf("\n  Got:      ");
            for (int i = 0; i < 16; i++) printf("%.0f ", dst[i]);
            printf("\n");
            return 1;
        }
    }

    /* ---- 测试 2: 过滤正确性 ---- */
    printf("\n--- Filter Correctness (N = %zu) ---\n", N);

    g_src = ALIGNED_ALLOC(float, N, 64);
    g_dst = ALIGNED_ALLOC(float, N, 64);

    rand_xorshift64_seed(42);
    fill_random_f32(g_src, N);

    float* ref_dst = ALIGNED_ALLOC(float, N, 64);
    size_t ref_n = filter_positive_scalar(g_src, ref_dst, N);
    size_t avx_n = filter_positive_avx512(g_src, g_dst, N);

    printf("  Scalar filtered: %zu elements\n", ref_n);
    printf("  AVX-512 filtered: %zu elements\n", avx_n);

    if (ref_n == avx_n) {
        int ok = 1;
        for (size_t i = 0; i < ref_n; i++)
            if (fabsf(g_dst[i] - ref_dst[i]) > 1e-6f) { ok = 0; break; }
        printf("  [%s] Filter values match (all %zu elements)\n",
               ok ? "PASS" : "FAIL", ref_n);
    } else {
        printf("  [FAIL] Element count mismatch\n");
    }

    /* ---- 测试 3: 展开到掩码 ---- */
    printf("\n--- Expand to Mask ---\n");
    {
        float dense[8] = {10,20,30,40,50,60,70,80};
        float sparse[16], ref_sparse[16];
        __mmask16 mask = 0b1001001001001001; /* 4 个元素 */

        expand_to_mask_avx512(dense, sparse, mask, 4);
        expand_to_mask_scalar(dense, ref_sparse, mask, 4);

        int ok = 1;
        for (int i = 0; i < 16; i++)
            if (fabsf(sparse[i] - ref_sparse[i]) > 1e-6f) { ok = 0; break; }
        printf("  [%s] Expand to sparse mask\n", ok ? "PASS" : "FAIL");
    }

    /* ---- 基准测试 ---- */
    printf("\n--- Filter Benchmark (N = %zu) ---\n", N);
    {
        benchmark_result_t results[2];
        memset(results, 0, sizeof(results));

        BENCH_COMPUTE(bn_filter_scalar(), N, N * 2 * sizeof(float), 50,
                      results[0]);
        results[0].name = "filter scalar";

        BENCH_COMPUTE(bn_filter_avx512(), N, N * 2 * sizeof(float), 50,
                      results[1]);
        results[1].name = "filter AVX-512 (compress)";

        bench_report(results, 2);
    }

    printf("--- Compress/Expand Key Points ---\n");
    printf("  Compress: sparse → dense, branchless filter\n");
    printf("  Expand:   dense → sparse, fills mask positions\n");
    printf("  Both are microcoded (~15 cycle latency, ~1/3 cycle throughput)\n");
    printf("  Best for: filtering, sparse matrix assembly, gather compaction\n");
    printf("  Alternative: scalar loop for very small N (< 32)\n");

    ALIGNED_FREE(g_src); ALIGNED_FREE(g_dst); ALIGNED_FREE(ref_dst);
    return 0;
}
