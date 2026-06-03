/**
 * complete_softmax_avx2.cpp -- 完整的 AVX2 Softmax 实现
 *
 * softmax(x)[i] = exp(x[i] - max) / sum(exp(x[j] - max))
 *
 * 本文件演示了完整的 3-pass softmax：
 *   第 1 遍：找到全局最大值（为了数值稳定性）
 *   第 2 遍：计算 exp(x - max)，累加求和，存储 exp 值
 *   第 3 遍：将每个存储的 exp 值除以总和
 *
 * 同时还展示了：
 *   - 多项式 exp 近似（无需 SVML）
 *   - 带 rescaling 的在线最大值追踪（流式友好，2-pass）
 *   - 与标量参考实现的对比
 *   - 所有变体的基准测试
 *
 * 现有的 avx2_softmax_partial.cpp 只展示了分子计算。
 * 本文件通过最终的归一化步骤补全了整个流程。
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

#ifndef INFINITY
#define INFINITY (1.0f / 0.0f)
#endif

/* ================================================================
 * 基于 exp2 的快速多项式 exp(x) 近似
 *
 * exp(x) = exp2(x * log2(e))
 * 分解：x * log2(e) = N + f   其中 N = 整数部分，f = 小数部分
 * exp2(N + f) = 2^N * 2^f
 * 2^N = 对 float 指数位进行位操作
 * 2^f ≈ 1 + c1*f + c2*f² + c3*f³ + c4*f⁴ + c5*f⁵  （泰勒展开）
 *
 * 最大相对误差：对于 |x| < 10 约为 ~1.5%（机器学习推理可接受）
 * ================================================================ */

static inline __m256 exp_fast_avx2(__m256 x) {
    /* 截断以避免单精度溢出/下溢 */
    const __m256 lower = _mm256_set1_ps(-87.0f);
    const __m256 upper = _mm256_set1_ps(87.0f);
    x = _mm256_max_ps(lower, _mm256_min_ps(upper, x));

    const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
    const __m256 ln2_hi = _mm256_set1_ps(0.693359375f);
    const __m256 ln2_lo = _mm256_set1_ps(-2.12194440e-4f);
    const __m256 one = _mm256_set1_ps(1.0f);

    /* exp(r) 的泰勒系数：1 + r + r²/2! + r³/3! + ... */
    const __m256 c2 = _mm256_set1_ps(1.0f / 2.0f);
    const __m256 c3 = _mm256_set1_ps(1.0f / 6.0f);
    const __m256 c4 = _mm256_set1_ps(1.0f / 24.0f);
    const __m256 c5 = _mm256_set1_ps(1.0f / 120.0f);

    /* n = round(x * log2(e)) */
    __m256 n = _mm256_mul_ps(x, log2e);
    n = _mm256_round_ps(n, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    /* r = x - n * ln(2)  （拆分为两部分以提高精度：高 + 低） */
    __m256 r = _mm256_fnmadd_ps(n, ln2_hi, x);
    r = _mm256_fnmadd_ps(n, ln2_lo, r);

    /* 使用 Horner 方法计算 exp(r) */
    __m256 poly;
    poly = _mm256_fmadd_ps(c5, r, c4);
    poly = _mm256_fmadd_ps(poly, r, c3);
    poly = _mm256_fmadd_ps(poly, r, c2);
    poly = _mm256_fmadd_ps(poly, r, one);
    poly = _mm256_fmadd_ps(poly, r, one);

    /* 通过位操作计算 2^N：将 N + 偏移值(127) 移入指数位 */
    __m256i ni = _mm256_cvtps_epi32(n);
    ni = _mm256_add_epi32(ni, _mm256_set1_epi32(127));
    ni = _mm256_slli_epi32(ni, 23);

    return _mm256_mul_ps(_mm256_castsi256_ps(ni), poly);
}

/* ================================================================
 * 水平求和：8 个 f32 → 1 个 f32（基于 shuffle，避免 hadd 瓶颈）
 * ================================================================ */

static inline float hsum256_ps(__m256 v) {
    /* 交换高 128 位和低 128 位，相加 */
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    /* 在 128 位内：对交换位置，相加 */
    sum = _mm_add_ps(sum, _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(1, 0, 3, 2)));
    sum = _mm_add_ps(sum, _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtss_f32(sum);
}

/* ================================================================
 * 标量参考实现：使用 std::exp 的 3-pass softmax
 * ================================================================ */

__attribute__((noinline))
static void softmax_scalar(const float* x, float* y, size_t n) {
    /* 第 1 遍：找最大值 */
    float max_val = x[0];
    for (size_t i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    /* 第 2 遍：计算 exp 并求和 */
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float e = expf(x[i] - max_val);
        y[i] = e;
        sum += e;
    }

    /* 第 3 遍：归一化 */
    float inv_sum = 1.0f / sum;
    for (size_t i = 0; i < n; i++) {
        y[i] *= inv_sum;
    }
}

/* ================================================================
 * 使用多项式 exp 的 AVX2 3-pass softmax
 * ================================================================ */

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Waggressive-loop-optimizations"

__attribute__((noinline))
static void softmax_avx2_3pass(const float* x, float* y, size_t n) {
    if (n == 0) return;

    /* ---- 第 1 遍：找到全局最大值 ---- */
    __m256 vmax = _mm256_set1_ps(-INFINITY);
    size_t i = 0;

    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        vmax = _mm256_max_ps(vmax, v);
    }
    float max_val = hsum256_ps(vmax);
    max_val = -INFINITY;  /* 重新用标量计算以覆盖尾部元素 */
    for (i = 0; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    /* ---- 第 2 遍：计算 exp(x - max) 并累加求和 ---- */
    const __m256 vmax_val = _mm256_set1_ps(max_val);
    __m256 vsum = _mm256_setzero_ps();

    i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        __m256 diff = _mm256_sub_ps(v, vmax_val);
        __m256 ve = exp_fast_avx2(diff);
        _mm256_storeu_ps(y + i, ve);
        vsum = _mm256_add_ps(vsum, ve);
    }

    float sum = hsum256_ps(vsum);
    {
        size_t tail_i = i;
        for (; tail_i < n; tail_i++) {
            float e = expf(x[tail_i] - max_val);
            y[tail_i] = e;
            sum += e;
        }
    }

    /* ---- 第 3 遍：除以总和进行归一化 ---- */
    const __m256 inv_sum = _mm256_set1_ps(1.0f / sum);
    {
        size_t j = 0;
        for (; j + 8 <= n; j += 8) {
            __m256 v = _mm256_loadu_ps(y + j);
            _mm256_storeu_ps(y + j, _mm256_mul_ps(v, inv_sum));
        }
        float inv = 1.0f / sum;
        for (; j < n; j++) {
            y[j] *= inv;
        }
    }
}

#pragma GCC diagnostic pop

/* ================================================================
 * AVX2 2-pass "在线" softmax（标量最大值追踪 + 向量 exp）
 *
 * 维护一个标量运行最大值 m。每个数据块更新：
 *   m_new = max(m, max(chunk))         ← 标量归约
 *   sum = sum * exp(m - m_new) + Σ exp(chunk[i] - m_new)
 *
 * 这避免了第 3 遍：我们在单次流式遍历中存储每元素的 exp(x - m_final)
 * 和 sum。然后进行归一化。
 * ================================================================ */

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Waggressive-loop-optimizations"

__attribute__((noinline))
static void softmax_avx2_online(const float* x, float* y, size_t n) {
    if (n == 0) return;

    float max_val = -INFINITY;
    float sum_val = 0.0f;

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);

        /* 找当前块的最大值 */
        float chunk_max = -INFINITY;
        {
            float tmp[8];
            _mm256_storeu_ps(tmp, v);
            for (int j = 0; j < 8; j++)
                if (tmp[j] > chunk_max) chunk_max = tmp[j];
        }

        if (chunk_max > max_val) {
            /* 用 exp(旧最大值 - 新最大值) 重新缩放运行总和 */
            float rescale = expf(max_val - chunk_max);
            sum_val *= rescale;
            max_val = chunk_max;
        }

        /* 计算 exp(v - max_val) 并累加 */
        __m256 vshifted = _mm256_sub_ps(v, _mm256_set1_ps(max_val));
        __m256 vexp_sub = exp_fast_avx2(vshifted);

        float tmp_sum[8];
        _mm256_storeu_ps(tmp_sum, vexp_sub);
        for (int j = 0; j < 8; j++) sum_val += tmp_sum[j];
    }

    /* 标量处理尾部 */
    for (; i < n; i++) {
        float v = x[i];
        if (v > max_val) {
            sum_val *= expf(max_val - v);
            max_val = v;
        }
        sum_val += expf(v - max_val);
    }

    /* 最终遍历：计算 y[i] = exp(x[i] - max_val) / sum_val */
    float inv_sum = 1.0f / sum_val;
    const __m256 vmax = _mm256_set1_ps(max_val);
    const __m256 vinv = _mm256_set1_ps(inv_sum);
    i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        __m256 diff = _mm256_sub_ps(v, vmax);
        __m256 vexp = exp_fast_avx2(diff);
        _mm256_storeu_ps(y + i, _mm256_mul_ps(vexp, vinv));
    }
    for (; i < n; i++) {
        y[i] = expf(x[i] - max_val) * inv_sum;
    }
}

#pragma GCC diagnostic pop

/* ================================================================
 * 基准测试基础设施
 * ================================================================ */

static const size_t N = 1024;
static float* g_x = NULL;
static float* g_y = NULL;

__attribute__((noinline)) static void bn_scalar()  { softmax_scalar(g_x, g_y, N); }
__attribute__((noinline)) static void bn_avx2_3p() { softmax_avx2_3pass(g_x, g_y, N); }
__attribute__((noinline)) static void bn_avx2_onl(){ softmax_avx2_online(g_x, g_y, N); }

/* ================================================================
 * 主函数
 * ================================================================ */

int main() {
    cpu_print_features();

    if (!cpu_has_avx2()) {
        printf("AVX2 not supported on this CPU. Exiting.\n");
        return 0;
    }

    printf("\n=== Complete AVX2 Softmax (3-Pass + Online) ===\n");
    printf("N = %zu (L1-resident for clean compute benchmarking)\n", N);
    printf("exp approximation: polynomial degree-5, relative error < 2%%\n\n");

    /* 分配内存 */
    g_x = ALIGNED_ALLOC(float, N, 32);
    g_y = ALIGNED_ALLOC(float, N, 32);
    float* y_ref = ALIGNED_ALLOC(float, N, 32);

    /* 填充有趣的数据分布 */
    rand_xorshift64_seed(42);
    fill_range_f32(g_x, N, -5.0f, 5.0f);
    g_x[0]     = 10.0f;   /* 离群值 */
    g_x[N / 2] = -8.0f;   /* 离群值 */
    g_x[N - 1] = 7.0f;    /* 离群值 */

    /* ---- 正确性验证 ---- */
    printf("--- Correctness Verification ---\n");

    softmax_scalar(g_x, y_ref, N);

    /* 验证概率之和等于 1.0 */
    {
        float total = 0.0f;
        for (size_t i = 0; i < N; i++) total += y_ref[i];
        printf("  Scalar sum of softmax outputs: %.6f (should be 1.0)\n", total);
    }

    softmax_avx2_3pass(g_x, g_y, N);
    CHECK_NEAR_ARRAY(g_y, y_ref, N, 0.03f,
        "AVX2 3-pass softmax vs scalar (tol 0.03 for poly exp)");

    /* 同时验证 AVX2 输出的和 ≈ 1.0 */
    {
        float total = 0.0f;
        for (size_t i = 0; i < N; i++) total += g_y[i];
        printf("  AVX2 3-pass sum of outputs: %.6f (should be ~1.0)\n", total);
    }

    softmax_avx2_online(g_x, g_y, N);
    CHECK_NEAR_ARRAY(g_y, y_ref, N, 0.03f,
        "AVX2 online softmax vs scalar (tol 0.03)");

    /* ---- 基准测试 ---- */
    printf("\n--- Benchmarks (N = %zu) ---\n", N);

    {
        benchmark_result_t results[3];
        memset(results, 0, sizeof(results));

        /* 对于 softmax，字节数：3-pass 中 1 读(x) + 1 写(y) × 3 遍。
         * 但内存复用使此为近似值。使用元素计数来计算 ns/elem。 */
        size_t bytes_per_call = N * 2 * sizeof(float);

        BENCH_COMPUTE(bn_scalar(), N, bytes_per_call, 500, results[0]);
        results[0].name = "softmax scalar (std::exp)";

        BENCH_COMPUTE(bn_avx2_3p(), N, bytes_per_call, 500, results[1]);
        results[1].name = "softmax AVX2 3-pass (poly exp)";

        BENCH_COMPUTE(bn_avx2_onl(), N, bytes_per_call, 500, results[2]);
        results[2].name = "softmax AVX2 online (poly exp)";

        bench_report(results, 3);
    }

    printf("--- Softmax Performance Notes ---\n");
    printf("  3-pass approach: 3N reads + 2N writes = 5N memory ops\n");
    printf("  Online approach:  2N reads + 2N writes = 4N memory ops\n");
    printf("  For N <= 4096 (L1-resident), compute-bound (exp dominates)\n");
    printf("  For N >  4096, memory-bound (bandwidth limits throughput)\n");
    printf("\n");
    printf("  Numerical stability:\n");
    printf("    - Subtracting max prevents exp overflow (expf(>89) = +Inf)\n");
    printf("    - Polynomial exp: max relative error ~1.5%% @ |x|<5\n");
    printf("    - For production: use SVML _mm256_exp_ps or better approximation\n");
    printf("\n");
    printf("  FlashAttention connection:\n");
    printf("    - Online softmax is the foundation of FlashAttention\n");
    printf("    - Real FlashAttention applies this to Q*K^T attention scores\n");
    printf("    - Key trick: O(m_new - m_old) rescaling, not O(N) recomputation\n");

    /* 清理 */
    ALIGNED_FREE(g_x);
    ALIGNED_FREE(g_y);
    ALIGNED_FREE(y_ref);
    return 0;
}
