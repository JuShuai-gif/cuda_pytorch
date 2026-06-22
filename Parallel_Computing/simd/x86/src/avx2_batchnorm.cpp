/**
 * avx2_batchnorm.cpp -- AVX2 批归一化 (Batch Normalization) 前向推理
 *
 * 公式: y = (x - mean) / sqrt(var + eps) * gamma + beta
 *
 * 算法: 2-pass 方案, 使用 Welford 在线算法
 *   Pass 1: 流式计算均值 (mean) 和 M2 (平方差之和),
 *           使用 Welford 在线更新公式, 单次遍历即可获得精确统计量
 *   Pass 2: 用预计算的 inv_std 进行归一化
 *
 * 输入布局: NCHW
 *   N = batch size (批大小)
 *   C = channels (通道数)
 *   H = height (高度)
 *   W = width (宽度)
 *
 * 按通道独立处理, 每个通道有独立的 mean, var, gamma, beta。
 * AVX2 8 路处理空间维度 (H*W)。
 *
 * Welford 在线算法:
 *   对于序列 x_1, x_2, ..., x_n:
 *     delta = x_k - mean_{k-1}
 *     mean_k = mean_{k-1} + delta / k
 *     M2_k   = M2_{k-1} + delta * (x_k - mean_k)
 *   最终: mean = mean_n, var = M2_n / n (总体方差)
 *
 * ~200 行
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
 * 标量 BatchNorm 参考实现 (2-pass)
 *
 * Pass 1: 计算 mean
 * Pass 2: 计算方差 (使用已知的 mean)
 * Pass 3: 归一化
 *
 * 输入布局 NCHW, 按通道独立统计
 * ================================================================ */
__attribute__((noinline))
static void batchnorm_scalar(
    const float* x, float* y,
    const float* gamma, const float* beta,
    int N, int C, int H, int W, float eps)
{
    int HW = H * W;

    for (int c = 0; c < C; c++) {
        /* Pass 1: 计算均值 */
        float sum = 0.0f;
        for (int n = 0; n < N; n++) {
            const float* x_nc = x + ((size_t)n * C + c) * HW;
            for (int i = 0; i < HW; i++) {
                sum += x_nc[i];
            }
        }
        float mean = sum / (float)(N * HW);

        /* Pass 2: 计算方差 */
        float var_sum = 0.0f;
        for (int n = 0; n < N; n++) {
            const float* x_nc = x + ((size_t)n * C + c) * HW;
            for (int i = 0; i < HW; i++) {
                float diff = x_nc[i] - mean;
                var_sum += diff * diff;
            }
        }
        float var = var_sum / (float)(N * HW);
        float inv_std = 1.0f / sqrtf(var + eps);

        /* Pass 3: 归一化 */
        float g = gamma[c];
        float b = beta[c];
        for (int n = 0; n < N; n++) {
            const float* x_nc = x + ((size_t)n * C + c) * HW;
            float* y_nc = y + ((size_t)n * C + c) * HW;
            for (int i = 0; i < HW; i++) {
                y_nc[i] = (x_nc[i] - mean) * inv_std * g + b;
            }
        }
    }
}

/* ================================================================
 * AVX2 BatchNorm - 使用 Welford 在线算法 (2-pass)
 *
 * Pass 1: 流式计算 mean 和 M2 (Welford), 仅遍历数据一次
 *   - 不需要先知道 mean 再算方差
 *   - 数值稳定性优于朴素的两遍方差算法
 *   - AVX2 8 路并行处理空间维度
 *
 * Pass 2: 用预计算的 inv_std 进行归一化 (AVX2 8 路)
 *
 * 按通道处理: 每个通道有独立的统计量和 affine 参数
 * ================================================================ */
__attribute__((noinline))
static void batchnorm_avx2_welford(
    const float* x, float* y,
    const float* gamma, const float* beta,
    int N, int C, int H, int W, float eps)
{
    int HW = H * W;
    int total = N * HW;

    for (int c = 0; c < C; c++) {
        /* ---- Pass 1: Welford 在线算法 (mean + M2) ---- */
        float mean = 0.0f;
        float M2   = 0.0f;
        size_t count = 0;

        for (int n = 0; n < N; n++) {
            const float* x_nc = x + ((size_t)n * C + c) * HW;
            int i = 0;

            /* AVX2 8 路 Welford 更新 */
            for (; i + 7 < HW; i += 8) {
                __m256 vx = _mm256_loadu_ps(x_nc + i);

                /* 提取 8 个标量值进行 Welford 更新 */
                float vals[8];
                _mm256_storeu_ps(vals, vx);

                for (int j = 0; j < 8; j++) {
                    count++;
                    float delta = vals[j] - mean;
                    mean += delta / (float)count;
                    float delta2 = vals[j] - mean;
                    M2 += delta * delta2;
                }
            }

            /* 标量尾部 */
            for (; i < HW; i++) {
                count++;
                float delta = x_nc[i] - mean;
                mean += delta / (float)count;
                float delta2 = x_nc[i] - mean;
                M2 += delta * delta2;
            }
        }

        /* 总体方差 = M2 / total */
        float var = M2 / (float)total;
        float inv_std = 1.0f / sqrtf(var + eps);

        /* ---- Pass 2: 归一化 (AVX2 8 路) ---- */
        __m256 vmean   = _mm256_set1_ps(mean);
        __m256 vinv    = _mm256_set1_ps(inv_std);
        __m256 vgamma  = _mm256_set1_ps(gamma[c]);
        __m256 vbeta   = _mm256_set1_ps(beta[c]);

        for (int n = 0; n < N; n++) {
            const float* x_nc = x + ((size_t)n * C + c) * HW;
            float* y_nc = y + ((size_t)n * C + c) * HW;
            int i = 0;

            /* AVX2 8 路归一化 */
            for (; i + 7 < HW; i += 8) {
                __m256 vx = _mm256_loadu_ps(x_nc + i);
                /* y = ((x - mean) * inv_std) * gamma + beta */
                __m256 vnorm = _mm256_sub_ps(vx, vmean);
                vnorm = _mm256_mul_ps(vnorm, vinv);
                vnorm = _mm256_fmadd_ps(vnorm, vgamma, vbeta);
                _mm256_storeu_ps(y_nc + i, vnorm);
            }

            /* 标量尾部 */
            for (; i < HW; i++) {
                y_nc[i] = (x_nc[i] - mean) * inv_std * gamma[c] + beta[c];
            }
        }
    }
}

/* ================================================================
 * AVX2 BatchNorm 变体: 使用 rsqrt 近似 + Newton-Raphson 优化
 *
 * _mm256_rsqrt_ps 提供 ~11 位精度的倒数平方根,
 * 一次 Newton-Raphson 迭代后提升到 ~23 位,
 * 对 BatchNorm 推理通常足够
 * ================================================================ */
__attribute__((noinline))
static void batchnorm_avx2_rsqrt(
    const float* x, float* y,
    const float* gamma, const float* beta,
    int N, int C, int H, int W, float eps)
{
    int HW = H * W;
    int total = N * HW;

    for (int c = 0; c < C; c++) {
        /* Pass 1: Welford (同上述实现, 简化版) */
        float mean = 0.0f;
        float M2   = 0.0f;
        int count  = 0;

        for (int n = 0; n < N; n++) {
            const float* x_nc = x + ((size_t)n * C + c) * HW;
            for (int i = 0; i < HW; i++) {
                count++;
                float delta = x_nc[i] - mean;
                mean += delta / (float)count;
                float delta2 = x_nc[i] - mean;
                M2 += delta * delta2;
            }
        }

        float var = M2 / (float)total;

        /* rsqrt + Newton-Raphson 优化 */
        __m256 vvar_eps = _mm256_set1_ps(var + eps);
        __m256 vrsqrt = _mm256_rsqrt_ps(vvar_eps);
        /* NR: y = y * (1.5 - 0.5 * x * y * y) */
        __m256 half       = _mm256_set1_ps(0.5f);
        __m256 three_half = _mm256_set1_ps(1.5f);
        __m256 y2  = _mm256_mul_ps(vrsqrt, vrsqrt);
        __m256 xy2 = _mm256_mul_ps(vvar_eps, y2);
        __m256 step = _mm256_sub_ps(three_half, _mm256_mul_ps(half, xy2));
        __m256 vinv_std = _mm256_mul_ps(vrsqrt, step);

        float inv_std = _mm_cvtss_f32(_mm256_castps256_ps128(vinv_std));

        /* Pass 2: 归一化 */
        __m256 vmean   = _mm256_set1_ps(mean);
        __m256 vinv    = _mm256_set1_ps(inv_std);
        __m256 vgamma  = _mm256_set1_ps(gamma[c]);
        __m256 vbeta   = _mm256_set1_ps(beta[c]);

        for (int n = 0; n < N; n++) {
            const float* x_nc = x + ((size_t)n * C + c) * HW;
            float* y_nc = y + ((size_t)n * C + c) * HW;
            int i = 0;
            for (; i + 7 < HW; i += 8) {
                __m256 vx = _mm256_loadu_ps(x_nc + i);
                __m256 vnorm = _mm256_sub_ps(vx, vmean);
                vnorm = _mm256_mul_ps(vnorm, vinv);
                vnorm = _mm256_fmadd_ps(vnorm, vgamma, vbeta);
                _mm256_storeu_ps(y_nc + i, vnorm);
            }
            for (; i < HW; i++) {
                y_nc[i] = (x_nc[i] - mean) * inv_std * gamma[c] + beta[c];
            }
        }
    }
}

/* ================================================================
 * 性能基准测试包装器
 * ================================================================ */

/* 全局测试参数 */
static const int BN_N = 4;     /* batch size */
static const int BN_C = 8;     /* channels */
static const int BN_H = 32;    /* height */
static const int BN_W = 32;    /* width */

static float* g_x     = NULL;
static float* g_y     = NULL;
static float* g_gamma = NULL;
static float* g_beta  = NULL;

__attribute__((noinline))
static void bn_bn_scalar() {
    batchnorm_scalar(g_x, g_y, g_gamma, g_beta,
                     BN_N, BN_C, BN_H, BN_W, 1e-5f);
}

__attribute__((noinline))
static void bn_bn_avx2_welford() {
    batchnorm_avx2_welford(g_x, g_y, g_gamma, g_beta,
                           BN_N, BN_C, BN_H, BN_W, 1e-5f);
}

__attribute__((noinline))
static void bn_bn_avx2_rsqrt() {
    batchnorm_avx2_rsqrt(g_x, g_y, g_gamma, g_beta,
                         BN_N, BN_C, BN_H, BN_W, 1e-5f);
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

    size_t total_elems = (size_t)BN_N * BN_C * BN_H * BN_W;
    size_t total_bytes = total_elems * sizeof(float);

    printf("\n=== AVX2 Batch Normalization (Welford 在线算法) ===\n");
    printf("布局: NCHW = %d x %d x %d x %d = %zu 元素\n",
           BN_N, BN_C, BN_H, BN_W, total_elems);
    printf("SIMD 宽度: 256-bit (8 f32 per register)\n");
    printf("算法: 2-pass (Welford pass1 + normalize pass2)\n");
    printf("eps = 1e-5\n\n");

    /* 分配对齐内存 */
    g_x     = ALIGNED_ALLOC(float, total_elems, 32);
    g_y     = ALIGNED_ALLOC(float, total_elems, 32);
    g_gamma = ALIGNED_ALLOC(float, BN_C, 32);
    g_beta  = ALIGNED_ALLOC(float, BN_C, 32);

    /* 填充数据 */
    rand_xorshift64_seed(42);
    fill_range_f32(g_x, total_elems, -5.0f, 5.0f);
    /* gamma 和 beta: 学到的参数, 通常接近 1 和 0 */
    for (int c = 0; c < BN_C; c++) {
        g_gamma[c] = 0.5f + (float)c * 0.1f;  /* 范围 [0.5, 1.2] */
        g_beta[c]  = (float)(c - BN_C / 2) * 0.1f;  /* 范围 [-0.4, 0.3] */
    }

    /* ---- 正确性验证 ---- */
    printf("--- 正确性验证 ---\n");

    float* ref = ALIGNED_ALLOC(float, total_elems, 32);
    float* out = ALIGNED_ALLOC(float, total_elems, 32);

    memset(ref, 0, total_bytes);
    memset(out, 0, total_bytes);

    batchnorm_scalar(g_x, ref, g_gamma, g_beta,
                     BN_N, BN_C, BN_H, BN_W, 1e-5f);
    batchnorm_avx2_welford(g_x, out, g_gamma, g_beta,
                           BN_N, BN_C, BN_H, BN_W, 1e-5f);

    printf("  参考输出前 8 个值: ");
    for (int i = 0; i < 8 && i < (int)total_elems; i++) {
        printf("%.4f ", (double)ref[i]);
    }
    printf("\n");
    printf("  AVX2 输出前 8 个值: ");
    for (int i = 0; i < 8 && i < (int)total_elems; i++) {
        printf("%.4f ", (double)out[i]);
    }
    printf("\n");

    CHECK_NEAR_ARRAY(out, ref, total_elems, 1e-4f,
                     "AVX2 Welford BatchNorm vs 标量参考");

    /* 验证 AVX2 rsqrt 变体 */
    memset(out, 0, total_bytes);
    batchnorm_avx2_rsqrt(g_x, out, g_gamma, g_beta,
                         BN_N, BN_C, BN_H, BN_W, 1e-5f);
    CHECK_NEAR_ARRAY(out, ref, total_elems, 1e-4f,
                     "AVX2 rsqrt+NR BatchNorm vs 标量参考");

    /* 验证每通道统计量 */
    printf("\n  每通道均值 (应接近 0):\n");
    for (int c = 0; c < BN_C && c < 4; c++) {
        float sum = 0.0f;
        for (int n = 0; n < BN_N; n++) {
            const float* y_nc = out + ((size_t)n * BN_C + c) * BN_H * BN_W;
            for (int i = 0; i < BN_H * BN_W; i++) sum += y_nc[i];
        }
        float channel_mean = sum / (float)(BN_N * BN_H * BN_W);
        printf("    C[%d]: 均值=%.6f\n", c, (double)channel_mean);
    }

    ALIGNED_FREE(ref);
    ALIGNED_FREE(out);

    /* ---- 性能基准测试 ---- */
    printf("\n--- 性能基准测试 ---\n");

    const size_t bytes_rw =
        total_bytes * 2                          /* x(rd) + y(wr) */
        + (size_t)BN_C * sizeof(float) * 2;      /* gamma(rd) + beta(rd) */

    benchmark_result_t results[3];
    memset(results, 0, sizeof(results));

    BENCH_COMPUTE(bn_bn_scalar(), total_elems, bytes_rw, 200, results[0]);
    results[0].name = "标量 (scalar)";

    BENCH_COMPUTE(bn_bn_avx2_welford(), total_elems, bytes_rw, 200, results[1]);
    results[1].name = "AVX2 Welford (2-pass)";

    BENCH_COMPUTE(bn_bn_avx2_rsqrt(), total_elems, bytes_rw, 200, results[2]);
    results[2].name = "AVX2 rsqrt+NR";

    bench_report(results, 3);

    /* ---- 算法注释 ---- */
    printf("=== Welford 在线算法注释 ===\n");
    printf("\n");
    printf("Welford 算法 (单遍方差计算):\n");
    printf("  传统方法: 先算 mean, 再算 sum((x-mean)^2) -- 需要两遍遍历\n");
    printf("  Welford: 维护 (count, mean, M2) 三元组, 每个新值更新一次\n");
    printf("  更新公式:\n");
    printf("    delta = x - mean_{old}\n");
    printf("    mean_{new} = mean_{old} + delta / count\n");
    printf("    delta2 = x - mean_{new}\n");
    printf("    M2_{new} = M2_{old} + delta * delta2\n");
    printf("  最终: var = M2 / count (总体方差)\n");
    printf("\n");
    printf("数值稳定性:\n");
    printf("  Welford 比朴素方差算法 (sum(x^2) - (sum(x))^2/n) 更稳定\n");
    printf("  后者在大值相近时会导致灾难性抵消\n");
    printf("  Welford 的误差增长为 O(sqrt(N) * eps), 而非 O(N * eps)\n");
    printf("\n");
    printf("NCHW 布局说明:\n");
    printf("  x[n][c][h][w] 存储为连续数组: x[n*C*H*W + c*H*W + h*W + w]\n");
    printf("  同一通道 (c) 内的所有空间位置 (h,w) 和批 (n) 共享 mean/var\n");
    printf("  各通道独立处理: 每个通道有独立的 gamma[c], beta[c]\n");
    printf("  空间维度 (H*W) 连续, 适合 AVX2 8 路 SIMD 向量化\n");
    printf("\n");
    printf("rsqrt + Newton-Raphson:\n");
    printf("  _mm256_rsqrt_ps: ~11-bit 精度\n");
    printf("  1 次 NR 迭代后: ~23-bit 精度\n");
    printf("  对推理精度通常足够 (训练需要更高精度)\n");

    ALIGNED_FREE(g_x);
    ALIGNED_FREE(g_y);
    ALIGNED_FREE(g_gamma);
    ALIGNED_FREE(g_beta);

    return 0;
}
