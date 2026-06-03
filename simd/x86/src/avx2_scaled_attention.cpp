/**
 * avx2_scaled_attention.cpp -- AVX2 缩放点积注意力机制
 *
 * Attention(Q, K, V) = softmax(Q * K^T / sqrt(dk)) * V
 *
 * 输入:
 *   Q[M][D]  -- 查询矩阵, M 个查询, 每个维度 D
 *   K[N][D]  -- 键矩阵, N 个键, 每个维度 D
 *   V[N][Dv] -- 值矩阵, N 个值, 每个维度 Dv
 *
 * 核心算法:
 *   - 在线 softmax (online softmax): 不存储 N 个中间 exp 值,
 *     使用 rescaling 技术: 当发现新的最大值时,
 *     将之前所有累积的 partial sum 乘以 exp(old_max - new_max)
 *   - AVX2 8 路内层循环用于 Q*K^T 点积计算
 *   - AVX2 8 路内层循环用于 attention 权重 × V 的累积
 *   - 4 路累加器展开以隐藏 FMA 延迟 (每次展开迭代处理 4 个 D 维度元素)
 *
 * 参考:
 *   - FlashAttention: Fast and Memory-Efficient Exact Attention (Dao et al.)
 *   - Online normalizer calculation for softmax (Milakov & Gimelshein, 2018)
 *
 * ~250 行, 包含与朴素实现的正确性对比、性能基准测试、
 * FlashAttention 风格的在线计算注释
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
 * 快速多项式 exp(x) 近似 (基于 exp2 分解)
 *
 * exp(x) = 2^(x * log2(e))
 * 分解: x * log2(e) = N + f, 其中 N 为整数部分, f 为小数部分
 * 2^N: 通过整数位移浮点指数字段实现
 * 2^f: 通过 5 阶泰勒多项式近似
 *
 * 最大相对误差: 对 |x| < 10 约 ~1.5%
 * ================================================================ */
static inline __m256 exp_fast_avx2(__m256 x) {
    /* 截断以避免单精度溢出/下溢 */
    const __m256 lower  = _mm256_set1_ps(-87.0f);
    const __m256 upper  = _mm256_set1_ps(87.0f);
    x = _mm256_max_ps(lower, _mm256_min_ps(upper, x));

    const __m256 log2e   = _mm256_set1_ps(1.44269504088896341f);
    const __m256 ln2_hi  = _mm256_set1_ps(0.693359375f);
    const __m256 ln2_lo  = _mm256_set1_ps(-2.12194440e-4f);
    const __m256 one     = _mm256_set1_ps(1.0f);

    /* 泰勒系数: c_k = 1/k! */
    const __m256 c2 = _mm256_set1_ps(0.5f);
    const __m256 c3 = _mm256_set1_ps(1.6666666666e-1f);
    const __m256 c4 = _mm256_set1_ps(4.1666666666e-2f);
    const __m256 c5 = _mm256_set1_ps(8.3333333333e-3f);

    /* N = round(x * log2(e)) */
    __m256 n = _mm256_mul_ps(x, log2e);
    n = _mm256_round_ps(n, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    /* r = x - N * ln(2), 分高低两部分以提高精度 */
    __m256 r = _mm256_fnmadd_ps(n, ln2_hi, x);
    r = _mm256_fnmadd_ps(n, ln2_lo, r);

    /* exp(r) ≈ 1 + r * (1 + r * (1/2 + r * (1/6 + r * (1/24 + r/120)))) */
    __m256 poly = _mm256_fmadd_ps(c5, r, c4);
    poly = _mm256_fmadd_ps(poly, r, c3);
    poly = _mm256_fmadd_ps(poly, r, c2);
    poly = _mm256_fmadd_ps(poly, r, one);
    poly = _mm256_fmadd_ps(poly, r, one);

    /* 2^N: 将 N+127 左移 23 位放入浮点指数字段 */
    __m256i ni = _mm256_cvtps_epi32(n);
    ni = _mm256_add_epi32(ni, _mm256_set1_epi32(127));
    ni = _mm256_slli_epi32(ni, 23);

    return _mm256_mul_ps(_mm256_castsi256_ps(ni), poly);
}

/* ================================================================
 * AVX2 横向求和: 将 __m256 的 8 个 f32 归约为标量
 * ================================================================ */
static inline float hsum_ps_avx2(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}

/* ================================================================
 * 标量参考实现: 朴素 softmax 注意力 (2-pass)
 *
 * Pass 1: 计算 Q*K^T 的每一行, 找到该行的最大值
 * Pass 2: 计算 exp(score - max) / sum(exp(...)) 并加权 V
 *
 * 时间复杂度: O(M * N * (D + Dv))
 * 空间复杂度: O(N) (存储 attention 权重)
 * ================================================================ */
__attribute__((noinline))
static void attention_scalar(
    const float* Q, const float* K, const float* V,
    float* O,
    int M, int N, int D, int Dv)
{
    float scale = 1.0f / sqrtf((float)D);

    for (int m = 0; m < M; m++) {
        const float* q_row = Q + (size_t)m * D;

        /* Pass 1: 计算 QK^T 并找每行最大值 */
        float max_val = -INFINITY;
        for (int n = 0; n < N; n++) {
            const float* k_row = K + (size_t)n * D;
            float dot = 0.0f;
            for (int d = 0; d < D; d++) {
                dot += q_row[d] * k_row[d];
            }
            dot *= scale;
            if (dot > max_val) max_val = dot;
        }

        /* Pass 2: 计算 softmax 并累积 V */
        float sum_exp = 0.0f;
        float* o_row = O + (size_t)m * Dv;
        memset(o_row, 0, (size_t)Dv * sizeof(float));

        for (int n = 0; n < N; n++) {
            const float* k_row = K + (size_t)n * D;
            float dot = 0.0f;
            for (int d = 0; d < D; d++) {
                dot += q_row[d] * k_row[d];
            }
            dot = (dot * scale) - max_val;
            float w = expf(dot);
            sum_exp += w;

            const float* v_row = V + (size_t)n * Dv;
            for (int dv = 0; dv < Dv; dv++) {
                o_row[dv] += w * v_row[dv];
            }
        }

        /* 归一化 */
        float inv_sum = 1.0f / sum_exp;
        for (int dv = 0; dv < Dv; dv++) {
            o_row[dv] *= inv_sum;
        }
    }
}

/* ================================================================
 * AVX2 在线 softmax 注意力实现
 *
 * 在线 softmax 的核心思想:
 *   传统方法需要两次遍历 (找 max, 然后 exp+sum),
 *   在线方法只需一次遍历, 使用 rescaling 校正:
 *     当发现新的最大值 new_max > old_max 时:
 *       sum_exp *= exp(old_max - new_max)
 *       output *= exp(old_max - new_max)
 *
 * 内层循环优化:
 *   - Q*K^T 点积: AVX2 8 路 FMA + 4 路展开
 *   - V 累积: AVX2 8 路 FMA + 4 路展开
 *   - 累加器展开: 处理 4 个 D 维度元素/次 (32 f32), 隐藏 4 周期 FMA 延迟
 *
 * FlashAttention 风格注释:
 *   - 分块 (tiling): 将 Q, K, V 分成小块放入 SRAM/缓存
 *   - 在线归一化: 在分块循环内维护 running max 和 running sum
 *   - 本实现是 single-query 版本 (M=1 或外层逐 query 循环)
 *     真正的 FlashAttention 会将多个 query 一起分块以共享 K, V
 * ================================================================ */
__attribute__((noinline))
static void attention_avx2_online(
    const float* Q, const float* K, const float* V,
    float* O,
    int M, int N, int D, int Dv)
{
    float scale = 1.0f / sqrtf((float)D);

    for (int m = 0; m < M; m++) {
        const float* q_row = Q + (size_t)m * D;
        float* o_row = O + (size_t)m * Dv;

        /* 在线 softmax 状态 */
        float running_max  = -INFINITY;
        float running_sum  = 0.0f;

        /* 清零输出行 */
        memset(o_row, 0, (size_t)Dv * sizeof(float));

        for (int n = 0; n < N; n++) {
            const float* k_row = K + (size_t)n * D;

            /* ---------- Q * K^T 点积 (AVX2 8-wide + 4-way unroll) ---------- */
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            int d = 0;
            /* 4 路展开: 每次处理 32 个 f32 (4 × 8 = 32 次标量运算) */
            for (; d + 31 < D; d += 32) {
                __m256 q0 = _mm256_loadu_ps(q_row + d);
                __m256 k0 = _mm256_loadu_ps(k_row + d);
                acc0 = _mm256_fmadd_ps(q0, k0, acc0);

                __m256 q1 = _mm256_loadu_ps(q_row + d + 8);
                __m256 k1 = _mm256_loadu_ps(k_row + d + 8);
                acc1 = _mm256_fmadd_ps(q1, k1, acc1);

                __m256 q2 = _mm256_loadu_ps(q_row + d + 16);
                __m256 k2 = _mm256_loadu_ps(k_row + d + 16);
                acc2 = _mm256_fmadd_ps(q2, k2, acc2);

                __m256 q3 = _mm256_loadu_ps(q_row + d + 24);
                __m256 k3 = _mm256_loadu_ps(k_row + d + 24);
                acc3 = _mm256_fmadd_ps(q3, k3, acc3);
            }

            /* 8 路处理剩余元素 */
            for (; d + 7 < D; d += 8) {
                __m256 qv = _mm256_loadu_ps(q_row + d);
                __m256 kv = _mm256_loadu_ps(k_row + d);
                acc0 = _mm256_fmadd_ps(qv, kv, acc0);
            }

            /* 合并 4 个累加器 */
            acc0 = _mm256_add_ps(acc0, acc1);
            acc2 = _mm256_add_ps(acc2, acc3);
            acc0 = _mm256_add_ps(acc0, acc2);

            /* 横向求和 */
            float dot = hsum_ps_avx2(acc0);

            /* 标量尾部 */
            for (; d < D; d++) {
                dot += q_row[d] * k_row[d];
            }

            dot *= scale;

            /* ---------- 在线 softmax rescaling ---------- */
            if (dot > running_max) {
                /* 发现新的最大值: 缩放旧的累积值 */
                float rescale = expf(running_max - dot);
                running_sum *= rescale;

                /* 缩放输出行 (V 累积) */
                __m256 vrescale = _mm256_set1_ps(rescale);
                int dv = 0;
                for (; dv + 31 < Dv; dv += 32) {
                    __m256 vo0 = _mm256_loadu_ps(o_row + dv);
                    __m256 vo1 = _mm256_loadu_ps(o_row + dv + 8);
                    __m256 vo2 = _mm256_loadu_ps(o_row + dv + 16);
                    __m256 vo3 = _mm256_loadu_ps(o_row + dv + 24);
                    _mm256_storeu_ps(o_row + dv,      _mm256_mul_ps(vo0, vrescale));
                    _mm256_storeu_ps(o_row + dv + 8,  _mm256_mul_ps(vo1, vrescale));
                    _mm256_storeu_ps(o_row + dv + 16, _mm256_mul_ps(vo2, vrescale));
                    _mm256_storeu_ps(o_row + dv + 24, _mm256_mul_ps(vo3, vrescale));
                }
                for (; dv + 7 < Dv; dv += 8) {
                    __m256 vo = _mm256_loadu_ps(o_row + dv);
                    _mm256_storeu_ps(o_row + dv, _mm256_mul_ps(vo, vrescale));
                }
                for (; dv < Dv; dv++) {
                    o_row[dv] *= rescale;
                }

                running_max = dot;
            }

            /* 计算 exp(score - max) 并累加 */
            float shifted = dot - running_max;
            __m256 vshifted = _mm256_set1_ps(shifted);
            __m256 vexp = exp_fast_avx2(vshifted);
            float weight = _mm_cvtss_f32(_mm256_castps256_ps128(vexp));
            running_sum += weight;

            /* ---------- 概率 × V 累积 (AVX2 8-wide + 4-way unroll) ---------- */
            __m256 vweight = _mm256_set1_ps(weight);
            const float* v_row = V + (size_t)n * Dv;
            int dv = 0;

            /* 4 路展开: 每次处理 32 个 Dv 元素 */
            for (; dv + 31 < Dv; dv += 32) {
                __m256 vv0 = _mm256_loadu_ps(v_row + dv);
                __m256 ov0 = _mm256_loadu_ps(o_row + dv);
                _mm256_storeu_ps(o_row + dv, _mm256_fmadd_ps(vweight, vv0, ov0));

                __m256 vv1 = _mm256_loadu_ps(v_row + dv + 8);
                __m256 ov1 = _mm256_loadu_ps(o_row + dv + 8);
                _mm256_storeu_ps(o_row + dv + 8, _mm256_fmadd_ps(vweight, vv1, ov1));

                __m256 vv2 = _mm256_loadu_ps(v_row + dv + 16);
                __m256 ov2 = _mm256_loadu_ps(o_row + dv + 16);
                _mm256_storeu_ps(o_row + dv + 16, _mm256_fmadd_ps(vweight, vv2, ov2));

                __m256 vv3 = _mm256_loadu_ps(v_row + dv + 24);
                __m256 ov3 = _mm256_loadu_ps(o_row + dv + 24);
                _mm256_storeu_ps(o_row + dv + 24, _mm256_fmadd_ps(vweight, vv3, ov3));
            }

            /* 8 路处理剩余 */
            for (; dv + 7 < Dv; dv += 8) {
                __m256 vv = _mm256_loadu_ps(v_row + dv);
                __m256 ov = _mm256_loadu_ps(o_row + dv);
                _mm256_storeu_ps(o_row + dv, _mm256_fmadd_ps(vweight, vv, ov));
            }

            /* 标量尾部 */
            for (; dv < Dv; dv++) {
                o_row[dv] += weight * v_row[dv];
            }
        }

        /* 最终归一化: output /= running_sum */
        float inv_sum = 1.0f / running_sum;
        __m256 vinv = _mm256_set1_ps(inv_sum);

        int dv = 0;
        for (; dv + 31 < Dv; dv += 32) {
            _mm256_storeu_ps(o_row + dv,      _mm256_mul_ps(_mm256_loadu_ps(o_row + dv),      vinv));
            _mm256_storeu_ps(o_row + dv + 8,  _mm256_mul_ps(_mm256_loadu_ps(o_row + dv + 8),  vinv));
            _mm256_storeu_ps(o_row + dv + 16, _mm256_mul_ps(_mm256_loadu_ps(o_row + dv + 16), vinv));
            _mm256_storeu_ps(o_row + dv + 24, _mm256_mul_ps(_mm256_loadu_ps(o_row + dv + 24), vinv));
        }
        for (; dv + 7 < Dv; dv += 8) {
            _mm256_storeu_ps(o_row + dv, _mm256_mul_ps(_mm256_loadu_ps(o_row + dv), vinv));
        }
        for (; dv < Dv; dv++) {
            o_row[dv] *= inv_sum;
        }
    }
}

/* ================================================================
 * 性能基准测试包装器
 * ================================================================ */

/* 全局测试参数 */
static const int M  = 4;    /* 查询数量 */
static const int N  = 64;   /* 键/值数量 */
static const int D  = 64;   /* 键维度 */
static const int Dv = 64;   /* 值维度 */

static float* g_Q = NULL;
static float* g_K = NULL;
static float* g_V = NULL;
static float* g_O = NULL;

__attribute__((noinline))
static void bn_attn_scalar() {
    attention_scalar(g_Q, g_K, g_V, g_O, M, N, D, Dv);
}

__attribute__((noinline))
static void bn_attn_avx2() {
    attention_avx2_online(g_Q, g_K, g_V, g_O, M, N, D, Dv);
}

/* ================================================================
 * 主函数
 * ================================================================ */
int main() {
    cpu_print_features();

    if (!cpu_has_avx2()) {
        printf("AVX2 not supported on this CPU. Exiting.\n");
        return 0;
    }

    printf("\n=== AVX2 缩放点积注意力 (Online Softmax) ===\n");
    printf("M=%d (查询数), N=%d (键/值数), D=%d (键维度), Dv=%d (值维度)\n", M, N, D, Dv);
    printf("SIMD 宽度: 256-bit (8 f32 per register)\n");
    printf("内层展开: 4-way (隐藏 ~4-cycle FMA 延迟)\n\n");

    /* 分配对齐内存 */
    g_Q = ALIGNED_ALLOC(float, (size_t)M * D, 32);
    g_K = ALIGNED_ALLOC(float, (size_t)N * D, 32);
    g_V = ALIGNED_ALLOC(float, (size_t)N * Dv, 32);
    g_O = ALIGNED_ALLOC(float, (size_t)M * Dv, 32);

    /* 填充确定性随机数据 (范围 [-1, 1]) */
    rand_xorshift64_seed(42);
    fill_random_f32(g_Q, (size_t)M * D);
    fill_random_f32(g_K, (size_t)N * D);
    fill_random_f32(g_V, (size_t)N * Dv);

    /* ---- 正确性验证 ---- */
    printf("--- 正确性验证 ---\n");

    float* ref = ALIGNED_ALLOC(float, (size_t)M * Dv, 32);
    float* out = ALIGNED_ALLOC(float, (size_t)M * Dv, 32);

    memset(ref, 0, (size_t)M * Dv * sizeof(float));
    memset(out, 0, (size_t)M * Dv * sizeof(float));

    attention_scalar(g_Q, g_K, g_V, ref, M, N, D, Dv);
    attention_avx2_online(g_Q, g_K, g_V, out, M, N, D, Dv);

    printf("  标量参考输出前 8 个值: ");
    for (int i = 0; i < 8 && i < M * Dv; i++) {
        printf("%.4f ", (double)ref[i]);
    }
    printf("\n");
    printf("  AVX2 在线输出前 8 个值: ");
    for (int i = 0; i < 8 && i < M * Dv; i++) {
        printf("%.4f ", (double)out[i]);
    }
    printf("\n");

    CHECK_NEAR_ARRAY(out, ref, (size_t)M * Dv, 0.05f,
                     "AVX2 在线 softmax 注意力 vs 标量参考");

    /* 输出第一个 query 的 softmax 权重和, 验证归一化 */
    printf("\n  每行输出前几个值 (目视检查一致性):\n");
    for (int m = 0; m < M; m++) {
        printf("    Q[%d]: ref=%.4f avx2=%.4f\n",
               m, (double)ref[m * Dv], (double)out[m * Dv]);
    }

    ALIGNED_FREE(ref);
    ALIGNED_FREE(out);

    /* ---- 性能基准测试 ---- */
    printf("\n--- 性能基准测试 ---\n");

    const size_t nelem = (size_t)M * Dv;
    const size_t bytes_rw =
        (size_t)M * D * sizeof(float)     /* Q read */
        + (size_t)N * D * sizeof(float)   /* K read */
        + (size_t)N * Dv * sizeof(float)  /* V read */
        + (size_t)M * Dv * sizeof(float); /* O write */

    benchmark_result_t results[2];
    memset(results, 0, sizeof(results));

    BENCH_COMPUTE(bn_attn_scalar(), nelem, bytes_rw, 500, results[0]);
    results[0].name = "标量 (scalar)";

    BENCH_COMPUTE(bn_attn_avx2(), nelem, bytes_rw, 500, results[1]);
    results[1].name = "AVX2 online-softmax";

    bench_report(results, 2);

    /* ---- FlashAttention 风格在线计算注释 ---- */
    printf("=== FlashAttention 风格在线计算注释 ===\n");
    printf("\n");
    printf("在线 softmax 算法:\n");
    printf("  传统 2-pass: Pass1 找 max, Pass2 计算 exp+sum, 最后归一化\n");
    printf("  在线 1-pass: 流式处理 K/V, 当发现更大 score 时 rescale 累积值\n");
    printf("\n");
    printf("Rescaling 公式:\n");
    printf("  设 current_max = m_old, new_max = m_new > m_old\n");
    printf("  sum_exp_new = sum_exp_old * exp(m_old - m_new) + exp(score_new - m_new)\n");
    printf("  output_new  = output_old  * exp(m_old - m_new) + weight_new * V[n]\n");
    printf("\n");
    printf("FlashAttention 扩展:\n");
    printf("  - 将 Q 分块 (外层循环), K/V 分块 (内层循环)\n");
    printf("  - 每个 K/V 块内计算 partial softmax, 跨块 rescale\n");
    printf("  - 将中间结果保存在 SRAM/寄存器中, 避免写入 HBM\n");
    printf("  - 本实现是单 query 版本, 对应于 FlashAttention 的 inner loop\n");
    printf("  - 真正的 FlashAttention 在多 query 间共享 K/V 块\n");
    printf("\n");
    printf("AVX2 优化要点:\n");
    printf("  - Q*K^T 点积: 4 路 FMA 展开 (32 f32/次), 隐藏 ~4 周期延迟\n");
    printf("  - V 累积: 4 路加载-计算-存储展开\n");
    printf("  - exp 近似: 5 阶多项式 (无 SVML 依赖), 误差 ~1.5%%\n");
    printf("  - 内存访问: 每次内循环遍历整个 D/Dv 维度\n");
    printf("  - 对小规模 (M,N<128): 数据完全驻留 L1 缓存, 延迟受限\n");
    printf("  - 对大规模: 需要 Q/K/V 分块以利用 L2/L3 缓存\n");

    ALIGNED_FREE(g_Q);
    ALIGNED_FREE(g_K);
    ALIGNED_FREE(g_V);
    ALIGNED_FREE(g_O);

    return 0;
}
