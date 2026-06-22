/**
 * sse_basics_demo.cpp -- SSE/SSE2/SSE3/SSSE3/SSE4.1 基础演示
 *
 * 演示基本的 SSE 操作：
 *   - 向量加法 (SSE)
 *   - 点积 (SSE2)
 *   - 横向加法 (SSE3)
 *   - 字节混排 (SSSE3 PSHUFB)
 *   - 混合与选择 (SSE4.1)
 *   - 横向最大值 (SSE)
 *
 * 所有内核均经过标量参考实现验证。每颗 x86-64 CPU
 * 都支持 SSE2；本演示使用 SSE4.1 作为编译目标，
 * 以使用最有用的内置函数（blend、mullo_epi32、extract）。
 *
 * 构建方式：使用 -msse4.1（由 CMake 通过 X86_COMMON_OPTIONS 设置）
 */

#include "../../common/aligned_buffer.h"
#include "../../common/benchmark.h"
#include "../../common/check.h"
#include "../../common/cpu_features.h"
#include "../../common/random_data.h"

#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ================================================================
 * 1. SSE 向量加法：c[i] = a[i] + b[i]（一次处理 4 个 float）
 * ================================================================ */

__attribute__((noinline))
static void sse_vec_add(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        _mm_storeu_ps(c + i, _mm_add_ps(va, vb));
    }
    for (; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

/* 标量参考实现 */
__attribute__((noinline))
static void scalar_vec_add(const float* a, const float* b, float* c, size_t n) {
    for (size_t i = 0; i < n; i++) c[i] = a[i] + b[i];
}

/* ================================================================
 * 2. SSE 点积 (SSE2)：sum(a[i] * b[i])，通过 FMA 模拟实现
 *    使用 2 个累加器来隐藏乘法+加法的延迟。
 * ================================================================ */

__attribute__((noinline))
static float sse_dot_product(const float* a, const float* b, size_t n) {
    __m128 acc0 = _mm_setzero_ps();
    __m128 acc1 = _mm_setzero_ps();
    size_t i = 0;

    for (; i + 8 <= n; i += 8) {
        __m128 va0 = _mm_loadu_ps(a + i);
        __m128 vb0 = _mm_loadu_ps(b + i);
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(va0, vb0));

        __m128 va1 = _mm_loadu_ps(a + i + 4);
        __m128 vb1 = _mm_loadu_ps(b + i + 4);
        acc1 = _mm_add_ps(acc1, _mm_mul_ps(va1, vb1));
    }

    acc0 = _mm_add_ps(acc0, acc1);

    /* 横向归约：4 个 f32 → 1 个 f32 */
    /* SSE3 _mm_hadd_ps：相邻对求和 */
    __m128 h = _mm_hadd_ps(acc0, acc0);     /* [a0+a1, a2+a3, a0+a1, a2+a3] */
    h = _mm_hadd_ps(h, h);                  /* [总和, 总和, 总和, 总和] */
    float result = _mm_cvtss_f32(h);

    for (; i < n; i++) result += a[i] * b[i];
    return result;
}

__attribute__((noinline))
static float scalar_dot_product(const float* a, const float* b, size_t n) {
    float s = 0.0f;
    for (size_t i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

/* ================================================================
 * 3. SSE 横向最大值（含 argmax 索引追踪）
 * ================================================================ */

__attribute__((noinline))
static float sse_hmax(const float* a, size_t n) {
    __m128 vmax = _mm_set1_ps(-1e30f);
    size_t i = 0;

    for (; i + 4 <= n; i += 4) {
        __m128 v = _mm_loadu_ps(a + i);
        vmax = _mm_max_ps(vmax, v);
    }

    /* 归约 4 通道最大值 */
    __m128 shuf = _mm_shuffle_ps(vmax, vmax, _MM_SHUFFLE(2, 3, 0, 1));
    vmax = _mm_max_ps(vmax, shuf);
    shuf = _mm_shuffle_ps(vmax, vmax, _MM_SHUFFLE(1, 0, 3, 2));
    vmax = _mm_max_ps(vmax, shuf);

    float max_val = _mm_cvtss_f32(vmax);
    for (; i < n; i++) if (a[i] > max_val) max_val = a[i];
    return max_val;
}

__attribute__((noinline))
static float scalar_hmax(const float* a, size_t n) {
    float m = a[0];
    for (size_t i = 1; i < n; i++) if (a[i] > m) m = a[i];
    return m;
}

/* ================================================================
 * 4. SSSE3 PSHUFB：反转 128 位寄存器中的 16 个字节
 * ================================================================ */

__attribute__((noinline))
static void ssse3_reverse_bytes(const uint8_t* src, uint8_t* dst, size_t n) {
    /* 预加载反转混排模式：15,14,...,1,0 */
    const __m128i rev_pattern = _mm_setr_epi8(
        15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
    size_t i = 0;

    for (; i + 16 <= n; i += 16) {
        __m128i v = _mm_loadu_si128((const __m128i*)(src + i));
        __m128i rev = _mm_shuffle_epi8(v, rev_pattern);
        _mm_storeu_si128((__m128i*)(dst + i), rev);
    }
    for (; i < n; i++) dst[i] = src[i];
}

__attribute__((noinline))
static void scalar_reverse_chunks(const uint8_t* src, uint8_t* dst, size_t n) {
    /* 在每个 16 字节块内反转（与 PSHUFB 行为一致） */
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        for (int j = 0; j < 16; j++)
            dst[i + j] = src[i + 15 - j];
    }
    for (; i < n; i++) dst[i] = src[i];
}

/* ================================================================
 * 5. SSE4.1 混合：无分支条件选择
 *    result = a > 0 ? a : 0（通过 blend 实现 ReLU）
 * ================================================================ */

__attribute__((noinline))
static void sse41_relu_blend(const float* src, float* dst, size_t n) {
    const __m128 zero = _mm_setzero_ps();
    size_t i = 0;

    for (; i + 4 <= n; i += 4) {
        __m128 v = _mm_loadu_ps(src + i);
        __m128 mask = _mm_cmpgt_ps(v, zero);  /* v > 0 的位置全 1 */
        /* blendv：mask<0 时从 v 中选择，否则从 zero 中选择 */
        _mm_storeu_ps(dst + i, _mm_blendv_ps(zero, v, mask));
    }
    for (; i < n; i++) dst[i] = (src[i] > 0.0f) ? src[i] : 0.0f;
}

__attribute__((noinline))
static void scalar_relu(const float* src, float* dst, size_t n) {
    for (size_t i = 0; i < n; i++)
        dst[i] = (src[i] > 0.0f) ? src[i] : 0.0f;
}

/* ================================================================
 * 6. SSE4.1 整数乘法：4 × i32 乘法
 *    SSE4.1 新增了 _mm_mullo_epi32，SSE2 中没有。
 *    在 SSE4.1 之前这是一个主要的痛点。
 * ================================================================ */

__attribute__((noinline))
static void sse41_i32_mul(const int32_t* a, const int32_t* b,
                           int32_t* c, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128i va = _mm_loadu_si128((const __m128i*)(a + i));
        __m128i vb = _mm_loadu_si128((const __m128i*)(b + i));
        _mm_storeu_si128((__m128i*)(c + i), _mm_mullo_epi32(va, vb));
    }
    for (; i < n; i++) c[i] = a[i] * b[i];
}

__attribute__((noinline))
static void scalar_i32_mul(const int32_t* a, const int32_t* b,
                            int32_t* c, size_t n) {
    for (size_t i = 0; i < n; i++) c[i] = a[i] * b[i];
}

/* ================================================================
 * 基准测试基础设施
 * ================================================================ */

static const size_t N = 100003; /* 质数，用于尾部测试 */
static float*  g_a = NULL;
static float*  g_b = NULL;
static float*  g_c = NULL;
static int32_t* g_ia = NULL;
static int32_t* g_ib = NULL;
static int32_t* g_ic = NULL;
static uint8_t* g_src8 = NULL;
static uint8_t* g_dst8 = NULL;
static float g_dot_result = 0.0f;

__attribute__((noinline)) static void bn_scalar_add()  { scalar_vec_add(g_a, g_b, g_c, N); }
__attribute__((noinline)) static void bn_sse_add()      { sse_vec_add(g_a, g_b, g_c, N); }
__attribute__((noinline)) static void bn_scalar_dot()   { g_dot_result = scalar_dot_product(g_a, g_b, N); }
__attribute__((noinline)) static void bn_sse_dot()      { g_dot_result = sse_dot_product(g_a, g_b, N); }
__attribute__((noinline)) static void bn_scalar_relu()  { scalar_relu(g_a, g_c, N); }
__attribute__((noinline)) static void bn_sse41_relu()   { sse41_relu_blend(g_a, g_c, N); }
__attribute__((noinline)) static void bn_scalar_i32mul(){ scalar_i32_mul(g_ia, g_ib, g_ic, N); }
__attribute__((noinline)) static void bn_sse41_i32mul() { sse41_i32_mul(g_ia, g_ib, g_ic, N); }

/* ================================================================
 * 主函数
 * ================================================================ */

int main() {
    cpu_print_features();

    printf("\n=== SSE 基础演示 ===\n");
    printf("目标 ISA：SSE4.1（2008+，所有现代 x86-64 CPU）\n");
    printf("SIMD 宽度：128 位（每个寄存器 4 f32 / 2 f64 / 16 u8）\n");
    printf("数据规模 N = %zu\n\n", N);

    /* 分配对齐缓冲区 */
    g_a    = ALIGNED_ALLOC(float, N, 16);
    g_b    = ALIGNED_ALLOC(float, N, 16);
    g_c    = ALIGNED_ALLOC(float, N, 16);
    g_ia   = ALIGNED_ALLOC(int32_t, N, 16);
    g_ib   = ALIGNED_ALLOC(int32_t, N, 16);
    g_ic   = ALIGNED_ALLOC(int32_t, N, 16);
    g_src8 = ALIGNED_ALLOC(uint8_t, 256, 16);
    g_dst8 = ALIGNED_ALLOC(uint8_t, 256, 16);

    float* ref_c  = ALIGNED_ALLOC(float, N, 16);
    int32_t* ref_ic = ALIGNED_ALLOC(int32_t, N, 16);
    uint8_t* ref8  = ALIGNED_ALLOC(uint8_t, 256, 16);

    /* 填充数据 */
    rand_xorshift64_seed(42);
    fill_random_f32(g_a, N);
    rand_xorshift64_seed(99);
    fill_random_f32(g_b, N);
    rand_xorshift64_seed(7);
    fill_random_i32(g_ia, N);
    rand_xorshift64_seed(13);
    fill_random_i32(g_ib, N);

    /* 为 PSHUFB 测试填充字节模式 */
    for (int i = 0; i < 256; i++) g_src8[i] = (uint8_t)i;

    /* ---- 正确性验证 ---- */
    printf("--- 正确性验证 ---\n");

    /* 测试 1：向量加法 */
    scalar_vec_add(g_a, g_b, ref_c, N);
    sse_vec_add(g_a, g_b, g_c, N);
    CHECK_NEAR_ARRAY(g_c, ref_c, N, 1e-6f, "SSE vector add vs scalar");

    /* 测试 2：点积 */
    float ref_dot = scalar_dot_product(g_a, g_b, N);
    float sse_dot = sse_dot_product(g_a, g_b, N);
    CHECK_NEAR(sse_dot, ref_dot, 1e-3f, "SSE dot product vs scalar");

    /* 测试 3：横向最大值 */
    float ref_hmax = scalar_hmax(g_a, N);
    float sse_hmax_val = sse_hmax(g_a, N);
    CHECK_NEAR(sse_hmax_val, ref_hmax, 1e-6f, "SSE horizontal max vs scalar");

    /* 测试 4：PSHUFB 在每 16 字节块内反转 */
    scalar_reverse_chunks(g_src8, ref8, 256);
    ssse3_reverse_bytes(g_src8, g_dst8, 256);
    {
        int ok = 1;
        for (int i = 0; i < 256; i++)
            if (g_dst8[i] != ref8[i]) { ok = 0; break; }
        printf("  [%s] SSSE3 PSHUFB byte reverse (256 bytes)\n",
               ok ? "PASS" : "FAIL");
        if (!ok) exit(1);
    }

    /* 测试 5：SSE4.1 通过 blend 实现 ReLU */
    scalar_relu(g_a, ref_c, N);
    sse41_relu_blend(g_a, g_c, N);
    CHECK_NEAR_ARRAY(g_c, ref_c, N, 0.0f, "SSE4.1 ReLU (blendv) vs scalar");

    /* 测试 6：SSE4.1 i32 乘法 */
    scalar_i32_mul(g_ia, g_ib, ref_ic, N);
    sse41_i32_mul(g_ia, g_ib, g_ic, N);
    {
        int ok = 1;
        for (size_t i = 0; i < N; i++)
            if (g_ic[i] != ref_ic[i]) { ok = 0; break; }
        printf("  [%s] SSE4.1 i32 multiply (mullo_epi32)\n", ok ? "PASS" : "FAIL");
        if (!ok) exit(1);
    }

    /* ---- 基准测试 ---- */
    printf("\n--- 基准测试（N = %zu）---\n", N);

    {
        benchmark_result_t results[9];
        memset(results, 0, sizeof(results));
        int slot = 0;

        size_t bytes_add = N * 3 * sizeof(float);
        BENCH_COMPUTE(bn_scalar_add(), N, bytes_add, 50, results[slot]);
        results[slot].name = "vec_add scalar"; slot++;

        BENCH_COMPUTE(bn_sse_add(), N, bytes_add, 50, results[slot]);
        results[slot].name = "vec_add SSE (4x f32)"; slot++;

        size_t bytes_dot = N * 2 * sizeof(float);
        BENCH_COMPUTE(bn_scalar_dot(), N, bytes_dot, 50, results[slot]);
        results[slot].name = "dot_product scalar"; slot++;

        BENCH_COMPUTE(bn_sse_dot(), N, bytes_dot, 50, results[slot]);
        results[slot].name = "dot_product SSE (2-acc)"; slot++;

        size_t bytes_relu = N * 2 * sizeof(float);
        BENCH_COMPUTE(bn_scalar_relu(), N, bytes_relu, 50, results[slot]);
        results[slot].name = "ReLU scalar"; slot++;

        BENCH_COMPUTE(bn_sse41_relu(), N, bytes_relu, 50, results[slot]);
        results[slot].name = "ReLU SSE4.1 (blendv)"; slot++;

        size_t bytes_i32 = N * 3 * sizeof(int32_t);
        BENCH_COMPUTE(bn_scalar_i32mul(), N, bytes_i32, 50, results[slot]);
        results[slot].name = "i32_mul scalar"; slot++;

        BENCH_COMPUTE(bn_sse41_i32mul(), N, bytes_i32, 50, results[slot]);
        results[slot].name = "i32_mul SSE4.1 (4x i32)"; slot++;

        bench_report(results, (size_t)slot);
    }

    printf("--- SSE 核心要点 ---\n");
    printf("- SSE2 是所有 x86-64 CPU 的基线（float 参数通过 XMM 寄存器传递）\n");
    printf("- SSE 每条指令处理 4 f32 或 2 f64（标量吞吐量的 2 倍）\n");
    printf("- SSE 没有 FMA：mul+add = 2 条微操作（AVX2 的 vfmadd = 1 条微操作）\n");
    printf("- SSSE3 PSHUFB 是有史以来最通用的字节级混排指令\n");
    printf("- SSE4.1 新增了 _mm_mullo_epi32（32 位整数乘法）和 _mm_blendv_ps\n");
    printf("- 新代码请以 AVX2 为目标；SSE 用于旧版兼容路径\n");

    /* 清理 */
    ALIGNED_FREE(g_a);  ALIGNED_FREE(g_b);  ALIGNED_FREE(g_c);
    ALIGNED_FREE(g_ia); ALIGNED_FREE(g_ib); ALIGNED_FREE(g_ic);
    ALIGNED_FREE(g_src8); ALIGNED_FREE(g_dst8);
    ALIGNED_FREE(ref_c); ALIGNED_FREE(ref_ic); ALIGNED_FREE(ref8);
    return 0;
}
