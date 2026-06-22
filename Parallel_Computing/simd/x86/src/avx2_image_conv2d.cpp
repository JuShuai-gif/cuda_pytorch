/**
 * avx2_image_conv2d.cpp -- 使用 AVX2 进行二维图像卷积
 *
 * 演示多种卷积策略:
 *   1. 直接 3x3 卷积 (步长 1, 单通道)
 *   2. 可分离卷积 (水平 + 垂直两遍扫描)
 *   3. 高斯模糊 (5 抽头可分离核)
 *
 * 所有实现都在内层循环中使用 AVX2 8 路 SIMD。
 * 这是图像处理管线以及 CNN 中卷积层的基础。
 *
 * 参考: Modern X86 Assembly Language Programming 2nd Ed, 第 13 章
 */

#include "../../common/aligned_buffer.h"
#include "../../common/benchmark.h"
#include "../../common/check.h"
#include "../../common/cpu_features.h"
#include "../../common/random_data.h"

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ================================================================
 * 1. 直接 3x3 卷积 (步长 1, 单通道)
 *
 * out[y][x] = Σ Σ kernel[ky][kx] * in[y+ky][x+kx]
 *
 * AVX2 每次处理 8 个输出列。
 * 9 个系数 × 8 列 = 每次内层循环迭代 72 次 FMA。
 * ================================================================ */

__attribute__((noinline))
static void conv2d_3x3_avx2(const float* in, float* out,
                             const float* kernel,
                             int H, int W) {
    int OH = H - 2; /* H >= 3 */
    int OW = W - 2; /* W >= 3 */

    for (int oh = 0; oh < OH; oh++) {
        int ow = 0;
        for (; ow + 8 <= OW; ow += 8) {
            __m256 acc = _mm256_setzero_ps();

            for (int kh = 0; kh < 3; kh++) {
                const float* row = in + (oh + kh) * W + ow;

                /* 加载 3 个偏移窗口: 当前, +1, +2 */
                __m256 r0 = _mm256_loadu_ps(row);
                __m256 r1 = _mm256_loadu_ps(row + 1);
                __m256 r2 = _mm256_loadu_ps(row + 2);

                __m256 k0 = _mm256_set1_ps(kernel[kh * 3 + 0]);
                __m256 k1 = _mm256_set1_ps(kernel[kh * 3 + 1]);
                __m256 k2 = _mm256_set1_ps(kernel[kh * 3 + 2]);

                acc = _mm256_fmadd_ps(k0, r0, acc);
                acc = _mm256_fmadd_ps(k1, r1, acc);
                acc = _mm256_fmadd_ps(k2, r2, acc);
            }
            _mm256_storeu_ps(out + oh * OW + ow, acc);
        }
        /* 标量尾部处理 */
        for (; ow < OW; ow++) {
            float sum = 0.0f;
            for (int kh = 0; kh < 3; kh++)
                for (int kw = 0; kw < 3; kw++)
                    sum += in[(oh + kh) * W + ow + kw] * kernel[kh * 3 + kw];
            out[oh * OW + ow] = sum;
        }
    }
}

__attribute__((noinline))
static void conv2d_3x3_scalar(const float* in, float* out,
                               const float* kernel, int H, int W) {
    int OH = H - 2, OW = W - 2;
    for (int oh = 0; oh < OH; oh++) {
        for (int ow = 0; ow < OW; ow++) {
            float sum = 0.0f;
            for (int kh = 0; kh < 3; kh++)
                for (int kw = 0; kw < 3; kw++)
                    sum += in[(oh + kh) * W + ow + kw] * kernel[kh * 3 + kw];
            out[oh * OW + ow] = sum;
        }
    }
}

/* ================================================================
 * 2. 可分离卷积: 水平 → 垂直
 *
 * 如果 K(x,y) = Kx(x) * Ky(y), 则二维核 K(x,y) 是可分离的。
 * 这样可以将 O(k²) 的运算量降到 O(2k)。
 * 示例: 边缘检测, 高斯模糊。
 *
 * 高斯 5 抽头核: [1, 4, 6, 4, 1] / 16
 * ================================================================ */

__attribute__((noinline))
static void gaussian_h_avx2(const float* src, float* tmp,
                             int H, int W, const float* kernel_h, int K) {
    int half_k = K / 2;

    for (int y = 0; y < H; y++) {
        const float* row = src + (size_t)y * W;
        float* trow = tmp + (size_t)y * W;

        /* 标量左边界: x < half_k */
        int x = 0;
        for (; x < half_k && x < W; x++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                int sx = x + k - half_k;
                if (sx < 0) sx = 0;
                sum += kernel_h[k] * row[sx];
            }
            trow[x] = sum;
        }

        /* SIMD 内部区域: 所有窗口访问均有效 */
        for (; x + 8 + K - 1 - half_k <= W; x += 8) {
            __m256 acc = _mm256_setzero_ps();
            for (int k = 0; k < K; k++) {
                __m256 r = _mm256_loadu_ps(row + x + k - half_k);
                acc = _mm256_fmadd_ps(_mm256_set1_ps(kernel_h[k]), r, acc);
            }
            _mm256_storeu_ps(trow + x, acc);
        }

        /* 标量右边界 */
        for (; x < W; x++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                int sx = x + k - half_k;
                if (sx < 0) sx = 0;
                if (sx >= W) sx = W - 1;
                sum += kernel_h[k] * row[sx];
            }
            trow[x] = sum;
        }
    }
}

__attribute__((noinline))
static void gaussian_v_avx2(const float* tmp, float* dst,
                             int H, int W, const float* kernel_v, int K) {
    int half_k = K / 2;

    for (int y = 0; y < H; y++) {
        int x = 0;
        for (; x + 8 <= W; x += 8) {
            __m256 acc = _mm256_setzero_ps();
            for (int k = 0; k < K; k++) {
                int sy = y + k - half_k;
                if (sy < 0) sy = 0;
                if (sy >= H) sy = H - 1;
                __m256 r = _mm256_loadu_ps(tmp + (size_t)sy * W + x);
                acc = _mm256_fmadd_ps(_mm256_set1_ps(kernel_v[k]), r, acc);
            }
            _mm256_storeu_ps(dst + (size_t)y * W + x, acc);
        }
        for (; x < W; x++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                int sy = y + k - half_k;
                if (sy < 0) sy = 0;
                if (sy >= H) sy = H - 1;
                sum += kernel_v[k] * tmp[(size_t)sy * W + x];
            }
            dst[(size_t)y * W + x] = sum;
        }
    }
}

__attribute__((noinline))
static void gaussian_blur_avx2(const float* src, float* dst,
                                int H, int W, const float* kernel, int K) {
    float* tmp = ALIGNED_ALLOC(float, H * W, 32);
    gaussian_h_avx2(src, tmp, H, W, kernel, K);
    gaussian_v_avx2(tmp, dst, H, W, kernel, K);
    ALIGNED_FREE(tmp);
}

__attribute__((noinline))
static void gaussian_blur_scalar(const float* src, float* dst,
                                  int H, int W, const float* kernel, int K) {
    /* 可分离二维卷积，与 AVX2 实现对应 */
    float* tmp = (float*)malloc((size_t)H * W * sizeof(float));

    /* 水平扫描 (边缘钳位) */
    for (int y = 0; y < H; y++) {
        const float* row = src + y * W;
        float* trow = tmp + y * W;
        for (int x = 0; x < W; x++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                int sx = x + k - K / 2;
                if (sx < 0) sx = 0;
                if (sx >= W) sx = W - 1;
                sum += kernel[k] * row[sx];
            }
            trow[x] = sum;
        }
    }

    /* 垂直扫描 (边缘钳位) */
    for (int y = 0; y < H; y++) {
        float* drow = dst + y * W;
        for (int x = 0; x < W; x++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                int sy = y + k - K / 2;
                if (sy < 0) sy = 0;
                if (sy >= H) sy = H - 1;
                sum += kernel[k] * tmp[sy * W + x];
            }
            drow[x] = sum;
        }
    }

    free(tmp);
}

/* ================================================================
 * 性能测试基础设施
 * ================================================================ */

static const int H = 128, W = 128; /* 用于测试的小图像 */
static float* g_in  = NULL;
static float* g_out = NULL;

static float g_kernel_3x3[9] = {
    1.0f/9, 1.0f/9, 1.0f/9,
    1.0f/9, 1.0f/9, 1.0f/9,
    1.0f/9, 1.0f/9, 1.0f/9,
};

static float g_kernel_gauss[5] = {
    1.0f/16, 4.0f/16, 6.0f/16, 4.0f/16, 1.0f/16
};

__attribute__((noinline)) static void bn_conv3_scalar() {
    conv2d_3x3_scalar(g_in, g_out, g_kernel_3x3, H, W);
}
__attribute__((noinline)) static void bn_conv3_avx2() {
    conv2d_3x3_avx2(g_in, g_out, g_kernel_3x3, H, W);
}
__attribute__((noinline)) static void bn_gauss_scalar() {
    gaussian_blur_scalar(g_in, g_out, H, W, g_kernel_gauss, 5);
}
__attribute__((noinline)) static void bn_gauss_avx2() {
    gaussian_blur_avx2(g_in, g_out, H, W, g_kernel_gauss, 5);
}

/* ================================================================
 * 主函数
 * ================================================================ */

int main() {
    cpu_print_features();

    if (!cpu_has_avx2()) {
        printf("AVX2 not supported. Exiting.\n");
        return 0;
    }

    printf("\n=== AVX2 2D Image Convolution Demo ===\n");
    printf("Image: %d x %d = %d pixels\n", H, W, H * W);
    printf("SIMD width: 256-bit (8 f32 per register)\n\n");

    /* 分配内存 */
    g_in  = ALIGNED_ALLOC(float, H * W, 32);
    g_out = ALIGNED_ALLOC(float, H * W, 32);

    /* 填充随机图像数据 */
    rand_xorshift64_seed(42);
    fill_random_f32(g_in, H * W);

    size_t npix = (size_t)(H - 2) * (size_t)(W - 2);

    /* ---- 测试 1: 3x3 卷积 ---- */
    printf("--- 3x3 卷积 (盒式模糊) ---\n");
    {
        float* ref = ALIGNED_ALLOC(float, H * W, 32);
        memset(ref,    0, H * W * sizeof(float));
        memset(g_out, 0, H * W * sizeof(float));

        conv2d_3x3_scalar(g_in, ref, g_kernel_3x3, H, W);
        conv2d_3x3_avx2(g_in, g_out, g_kernel_3x3, H, W);

        int OH = H - 2, OW = W - 2;
        int ok = 1;
        for (int y = 0; y < OH; y++) {
            for (int x = 0; x < OW; x++) {
                if (fabsf(g_out[y * OW + x] - ref[y * OW + x]) > 1e-5f) {
                    ok = 0;
                    goto done_3x3;
                }
            }
        }
    done_3x3:
        printf("  [%s] 3x3 卷积与标量结果一致\n",
               ok ? "通过" : "失败");

        ALIGNED_FREE(ref);
    }

    /* ---- 测试 2: 高斯模糊 ---- */
    printf("\n--- 5x5 高斯模糊 (可分离) ---\n");
    {
        float* ref = ALIGNED_ALLOC(float, H * W, 32);

        /* 填充棋盘格图案以便观察效果 */
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                g_in[y * W + x] = ((x + y) % 2) ? 1.0f : 0.0f;

        gaussian_blur_scalar(g_in, ref, H, W, g_kernel_gauss, 5);
        gaussian_blur_avx2(g_in, g_out, H, W, g_kernel_gauss, 5);

        /* 检查所有像素 */
        int errors = 0;
        for (int y = 0; y < H && errors < 10; y++) {
            for (int x = 0; x < W && errors < 10; x++) {
                float diff = fabsf(g_out[y * W + x] - ref[y * W + x]);
                if (diff > 1e-4f) {
                    printf("    不匹配 (%d,%d): 参考值=%.6f avx2=%.6f 差值=%.2e\n",
                           y, x, ref[y * W + x], g_out[y * W + x], diff);
                    errors++;
                }
            }
        }
        printf("  [%s] 高斯模糊与标量结果一致 (%d 个错误)\n",
               errors == 0 ? "通过" : "失败", errors);

        ALIGNED_FREE(ref);
    }

    /* ---- 性能测试 ---- */
    {
        /* 重置数据为随机值 */
        rand_xorshift64_seed(42);
        fill_random_f32(g_in, H * W);

        benchmark_result_t results[4];
        memset(results, 0, sizeof(results));

        size_t bytes = H * W * sizeof(float) * 2;
        BENCH_COMPUTE(bn_conv3_scalar(), npix, bytes, 200, results[0]);
        results[0].name = "3x3 conv scalar";

        BENCH_COMPUTE(bn_conv3_avx2(), npix, bytes, 200, results[1]);
        results[1].name = "3x3 conv AVX2 (8 cols)";

        BENCH_COMPUTE(bn_gauss_scalar(), npix, bytes, 100, results[2]);
        results[2].name = "Gauss5x5 scalar (direct)";

        BENCH_COMPUTE(bn_gauss_avx2(), npix, bytes, 100, results[3]);
        results[3].name = "Gauss5x5 AVX2 (separable)";

        bench_report(results, 4);
    }

    printf("--- 卷积要点 ---\n");
    printf("  直接 3x3:    每个输出像素 9 次 FMA, 8 列并行\n");
    printf("  可分离 5x5: 将每个像素 O(25) 次运算降为 O(10)\n");
    printf("  FMA 主导:    每条 vfmadd231ps = 1 微操作, 8 次浮点运算\n");
    printf("  对于更大图像: 缓存分块至关重要 (按 L1/L2 大小分片)\n");
    printf("  CNN 关联:    im2col + GEMM 在多通道时通常更快\n");
    printf("  Winograd F(2,3): 3x3 卷积的乘法次数减少 2.25 倍\n");

    ALIGNED_FREE(g_in);
    ALIGNED_FREE(g_out);
    return 0;
}
