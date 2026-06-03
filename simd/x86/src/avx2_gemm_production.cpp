/*****************************************************************
 * AVX2 GEMM -- 生产级实现 (Production-Grade Implementation)
 *
 * 遵循 BLIS / oneDNN / XNNPACK 的工业级 GEMM 设计模式：
 *
 *   1. 多级缓存分块 (Multi-level Cache Blocking):
 *      - L1 微内核: 6×8 (MR=6 行, NR=8 列)，使用 YMM 寄存器
 *      - L2 宏内核: KC 维度分块，确保 B 面板常驻 L2 缓存
 *      - L3 外部分块: M 和 N 维度分块，用于大矩阵的 L3 缓存复用
 *
 *   2. 打包策略 (Packing Strategy):
 *      - B 打包为 KC×NC 布局 (行主序), 使得 SIMD 加载连续
 *      - A 打包为 MC×KC 布局 (行主序), 为广播访问优化
 *
 *   3. 微内核 (6×8, AVX2, 8x K 展开):
 *      - 6 个 ymm 累加器 (c0..c5, 对应 C 的每行)
 *      - 8x K 维度展开: 每次内循环处理 8 个 k 值
 *      - 2 个 B 寄存器交替加载以隐藏加载延迟
 *      - 总计约 9 个 ymm 寄存器 (6 C + 2 B + 1 A 广播)
 *
 *   4. 边界处理: 支持任意 M, N, K (不限于 MR, NR 的倍数)
 *
 *   5. 所有注释使用中文
 *
 * 理论峰值 (单核, AVX2+FMA):
 *   2 FMA 单元 × 8 flops/周期 × ~3 GHz ≈ 48 GFLOPS
 *   实际可达 70-85% 峰值效率
 *****************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <immintrin.h>

#include "../../common/aligned_buffer.h"
#include "../../common/benchmark.h"
#include "../../common/check.h"
#include "../../common/cpu_features.h"
#include "../../common/random_data.h"

/* =================================================================
 * 分块参数 (Block Size Constants)
 *
 * 这些参数针对典型桌面/服务器 CPU 的缓存大小调优:
 *   - L1 数据缓存: 32 KB (每核)
 *   - L2 缓存: 256 KB ~ 1 MB (每核)
 *   - L3 缓存: 8 MB ~ 32 MB (共享)
 *
 * MR=6, NR=8: 微内核尺寸，对应 BLIS Haswell 的参数选择
 *   - MR=6: 每行 C 一个累加器 + YMM 寄存器有 16 个，留足 K 展开空间
 *   - NR=8: 匹配 256-bit SIMD 宽度 (8×f32 = 256 bit)
 *
 * KC=256: K 面板尺寸
 *   - A_packed[MR][KC]: 6×256×4 = 6 KB
 *   - B_packed[KC][NR]: 256×8×4 = 8 KB
 *   - 总计约 14 KB, 可以在 L1 缓存中驻留
 *
 * MC=384: M 面板尺寸
 *   - A_packed[MC][KC]: 384×256×4 = 384 KB (在 L2 缓存边界附近)
 *   - 可根据实际缓存大小调整
 *
 * NC≈4096: N 面板尺寸
 *   - B_packed[KC][NC]: 256×NC×4 = NC KB
 *   - 需要 NC×KC×4 + MC×KC×4 < L3 缓存
 * ================================================================= */

namespace {
constexpr int MR = 6;     /* 微内核行数: C 的 6 行在寄存器中累加 */
constexpr int NR = 8;     /* 微内核列数: 匹配 AVX2 256-bit 宽度 */
constexpr int KC = 256;   /* K 面板尺寸: 使 A+B 面板常驻 L1/L2 */
constexpr int MC = 384;   /* M 面板尺寸: 使 A 面板常驻 L2 */
constexpr int NC = 4096;  /* N 面板尺寸: 用于 L3 缓存复用 (大矩阵) */

/* 理论峰值: 2 FMA 单元 × 8 flops/周期 × 3.0 GHz ≈ 48 GFLOPS */
constexpr double THEORETICAL_PEAK_GFLOPS = 48.0;
}  // namespace

/* =================================================================
 * 辅助函数: __m256 水平求和
 *
 * 将所有 8 个 lane 的 float 值归约为单一标量。
 * 策略: 交换高/低 128-bit 半部 → 加法 → 两次 hadd → 提取。
 * 端口分配: permute (port 5) + add (port 0/1), 避免端口 5 饱和。
 * ================================================================= */

static inline float hsum_ps(__m256 v) {
    /* v = [a0, a1, a2, a3,  b0, b1, b2, b3] */
    __m256 swapped = _mm256_permute2f128_ps(v, v, 0x01);
    /* swapped = [b0, b1, b2, b3,  a0, a1, a2, a3] */
    v = _mm256_add_ps(v, swapped);
    v = _mm256_hadd_ps(v, v);
    v = _mm256_hadd_ps(v, v);
    return _mm256_cvtss_f32(v);
}

/* =================================================================
 * 第 0 层: 标量基线 GEMM (Scalar Baseline)
 *
 * 标准三循环 GEMM: C[i][j] += sum_k A[i][k] * B[k][j]
 * 复杂度: O(MNK) 次浮点运算, O(MNK) 次内存访问 (无数据复用)
 *
 * 作为正确性验证的参考实现。
 * ================================================================= */

static void scalar_gemm(int M_p, int N_p, int K_p,
                        const float* A, int lda,
                        const float* B, int ldb,
                        float* C, int ldc) {
    for (int i = 0; i < M_p; ++i) {
        for (int j = 0; j < N_p; ++j) {
            float sum = C[i * ldc + j];
            for (int k = 0; k < K_p; ++k) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] = sum;
        }
    }
}

/* =================================================================
 * 第 1 层: 朴素 SIMD GEMM (Naive SIMD GEMM)
 *
 * 对 K 维度做 SIMD 向量化，但不做数据打包。
 * B 的加载需要 gather (跨 N 列的非连续访问) —— 这是性能瓶颈。
 *
 * 目的: 展示为什么数据打包是必要的。
 * ================================================================= */

static void gemm_naive_simd(int M_p, int N_p, int K_p,
                            const float* A, int lda,
                            const float* B, int ldb,
                            float* C, int ldc) {
    for (int i = 0; i < M_p; ++i) {
        for (int j = 0; j < N_p; ++j) {
            __m256 acc = _mm256_setzero_ps();
            int k = 0;
            for (; k + 8 <= K_p; k += 8) {
                /* A[i][k..k+7]: 连续的, 一次 SIMD 加载 */
                __m256 a_vec = _mm256_loadu_ps(&A[i * lda + k]);

                /*
                 * B[k..k+7][j]: 跨步为 N_p 的 8 个值, 需要手动 gather.
                 * _mm256_set_ps 元素顺序为高 lane 到低 lane (反向).
                 */
                __m256 b_vec = _mm256_set_ps(
                    B[(k + 7) * ldb + j],
                    B[(k + 6) * ldb + j],
                    B[(k + 5) * ldb + j],
                    B[(k + 4) * ldb + j],
                    B[(k + 3) * ldb + j],
                    B[(k + 2) * ldb + j],
                    B[(k + 1) * ldb + j],
                    B[(k + 0) * ldb + j]
                );

                acc = _mm256_fmadd_ps(a_vec, b_vec, acc);
            }
            float sum = hsum_ps(acc);
            /* 处理尾部 (K 不是 8 的倍数的情况) — 使用指针算术避免循环优化误报 */
            {
                const float* a_ptr = &A[i * lda + k];
                const float* b_ptr = &B[k * ldb + j];
                int rem = K_p - k;
                while (rem-- > 0) {
                    sum += (*a_ptr++) * (*b_ptr);
                    b_ptr += ldb;
                }
            }
            C[i * ldc + j] += sum;
        }
    }
}

/* =================================================================
 * 打包函数: pack_B (Pack B Matrix Panel)
 *
 * 将 B 矩阵 (行主序, K×N) 的子面板打包到连续缓冲区中。
 * 布局: B_packed[k * nc + j] = B[(kp + k) * ldb + (np + j)]
 *
 * 这使得对 B 的访问变为连续 (沿列方向), 从而可以用一次 SIMD
 * 加载读取 8 个 B 值, 消除朴素 SIMD 中的 gather 开销。
 *
 * 参数:
 *   B_packed  - 输出: 打包后的 B 面板 (kc × nc_padded, 行主序)
 *   B         - 输入: 原始 B 矩阵
 *   ldb       - B 的 leading dimension (通常 = N)
 *   kp        - B 中 K 维度的起始行
 *   np        - B 中 N 维度的起始列
 *   kc        - K 面板的实际大小 (<= KC)
 *   nc        - N 面板的实际大小 (<= NC)
 *   nc_padded - nc 向上对齐到 NR 的值 (用于 SIMD 访问)
 * ================================================================= */

static void pack_B(float* B_packed,
                   const float* B, int ldb,
                   int kp, int np,
                   int kc, int nc, int nc_padded) {
    for (int k = 0; k < kc; ++k) {
        int j = 0;
        /* 主循环: 每次处理 NR 个元素, 手动展开以提高吞吐 */
        for (; j + 7 < nc; j += 8) {
            __m256 b_row = _mm256_loadu_ps(&B[(kp + k) * ldb + np + j]);
            _mm256_storeu_ps(&B_packed[k * nc_padded + j], b_row);
        }
        /* 处理尾部不足 8 个元素的情况 */
        for (; j < nc; ++j) {
            B_packed[k * nc_padded + j] = B[(kp + k) * ldb + np + j];
        }
        /* 填充区 (padding) 设为零, 保证 SIMD 加载安全 */
        for (; j < nc_padded; ++j) {
            B_packed[k * nc_padded + j] = 0.0f;
        }
    }
}

/* =================================================================
 * 打包函数: pack_A (Pack A Matrix Panel)
 *
 * 将 A 矩阵 (行主序, M×K) 的子面板复制到连续缓冲区中。
 * 布局: A_packed[i * kc + k] = A[(mp + i) * lda + (kp + k)]
 *
 * A 已经是行主序 (K 是最内维), 所以打包主要是做连续复制。
 * 对于边缘行 (mp + i >= M), 填充零。
 *
 * 参数:
 *   A_packed   - 输出: 打包后的 A 面板 (mc_padded × kc, 行主序)
 *   A          - 输入: 原始 A 矩阵
 *   lda        - A 的 leading dimension (通常 = K)
 *   mp         - A 中 M 维度的起始行
 *   kp         - A 中 K 维度的起始列
 *   mc         - M 面板的实际大小 (<= MC)
 *   kc         - K 面板的实际大小 (<= KC)
 *   mc_padded  - mc 向上对齐到 MR 的值
 * ================================================================= */

static void pack_A(float* A_packed,
                   const float* A, int lda,
                   int mp, int kp,
                   int mc, int kc, int mc_padded) {
    for (int i = 0; i < mc; ++i) {
        int k = 0;
        /* 主循环: 每次用 SIMD 复制 8 个元素 */
        for (; k + 7 < kc; k += 8) {
            __m256 a_row = _mm256_loadu_ps(&A[(mp + i) * lda + kp + k]);
            _mm256_storeu_ps(&A_packed[i * kc + k], a_row);
        }
        /* 处理尾部 */
        for (; k < kc; ++k) {
            A_packed[i * kc + k] = A[(mp + i) * lda + kp + k];
        }
    }
    /* 填充行 (padding): 当 mc < mc_padded 时, 额外行设为零 */
    for (int i = mc; i < mc_padded; ++i) {
        for (int k = 0; k < kc; ++k) {
            A_packed[i * kc + k] = 0.0f;
        }
    }
}

/* =================================================================
 * 微内核: gemm_micro_6x8 (6×8 Micro-Kernel, 8x K 展开)
 *
 * 核心计算: C[0:mr][0:nr] += A_packed[0:mr][kc] × B_packed[kc][0:nr]
 *
 * 这是整个 GEMM 实现的核心, 对应 BLIS 中的微内核。
 *
 * 寄存器分配 (AVX2: 总共 16 个 ymm 寄存器):
 *   c0..c5: 6 个 ymm 累加器, 对应 C 的 6 行      (6 regs)
 *   b0, b1: 2 个 ymm 寄存器用于交替加载 B 面板    (2 regs)
 *   a_brd:  1 个 ymm 寄存器用于广播 A 值           (1 regs)
 *   ---------------------------------------------------
 *   总计: 9 个寄存器 (56% 的 ymm 寄存器文件)
 *
 * 8x K 维度展开:
 *   每次内层迭代处理 8 个连续的 k 值。
 *   使用 2 个 B 寄存器交替加载, 将 B 加载延迟与 FMA 计算重叠:
 *     加载 b0 → 对 6 行做 FMA → 加载 b1 → 对 6 行做 FMA → ...
 *   6 行 × 8 个 k 值 = 48 条 FMA 指令, 它们使用不同的目标寄存器,
 *   因此可以 pipelined 执行, 隐藏每条 FMA 的 4 周期延迟。
 *
 * 内循环分析 (每 8 个 k 值):
 *   8 次 B 加载 (每次 32 字节, 连续)
 *   48 次 A 标量广播 + FMA
 *   浮点运算数: 48 × 2 = 96 flops (每条 FMA 做了乘法+加法)
 *   字节数: 8×32 + 48×4 = 256 + 192 = 448 bytes (近似)
 *
 * 参数:
 *   K_len  - K 维度的长度 (可能 < KC, 边缘情况)
 *   A      - 打包后的 A 面板指针 (指向当前 MR 行的开始)
 *   lda    - A 打包后的 stride (= kc, 即 K 面板长度)
 *   B      - 打包后的 B 面板指针 (指向当前 NR 列的开始)
 *   ldb    - B 打包后的 stride (= nc_padded)
 *   C      - 输出矩阵的指针 (指向当前 tile 的 C[mi][nj])
 *   ldc    - C 的 leading dimension
 *   mr_use - 实际需要计算的行数 (1..MR, 边缘可能 < MR)
 *   nr_use - 实际需要计算的列数 (1..NR, 边缘可能 < NR)
 * ================================================================= */

static void gemm_micro_6x8(int K_len,
                           const float* A, int lda,
                           const float* B, int ldb,
                           float* C, int ldc,
                           int mr_use, int nr_use) {
    /*
     * 预加载 C 的当前值到累加器。
     *
     * 对于 nr_use < NR 的情况, 我们只加载有效列数 (用掩码操作),
     * 但为了避免分支, 始终加载全部 8 个值, 无效列的 store 会被跳过。
     * 这在工业实践中是常见的: 使用掩码 store 处理列边界。
     */
    __m256 c0 = _mm256_loadu_ps(&C[0 * ldc]);
    __m256 c1 = (mr_use > 1) ? _mm256_loadu_ps(&C[1 * ldc]) : _mm256_setzero_ps();
    __m256 c2 = (mr_use > 2) ? _mm256_loadu_ps(&C[2 * ldc]) : _mm256_setzero_ps();
    __m256 c3 = (mr_use > 3) ? _mm256_loadu_ps(&C[3 * ldc]) : _mm256_setzero_ps();
    __m256 c4 = (mr_use > 4) ? _mm256_loadu_ps(&C[4 * ldc]) : _mm256_setzero_ps();
    __m256 c5 = (mr_use > 5) ? _mm256_loadu_ps(&C[5 * ldc]) : _mm256_setzero_ps();

    /*
     * 主循环: K 维度每次前进 8 步。
     *
     * 使用 2 个 B 寄存器 (b0, b1) 交替加载, 与 FMA 计算交错执行:
     *   加载 b(k+0), b(k+1) → 6 行 × 2 FMAs →
     *   加载 b(k+2), b(k+3) → 6 行 × 2 FMAs →
     *   加载 b(k+4), b(k+5) → 6 行 × 2 FMAs →
     *   加载 b(k+6), b(k+7) → 6 行 × 2 FMAs
     *
     * 这种交错模式将 B 加载 (4-5 周期延迟) 与计算重叠,
     * 避免流水线停顿。
     *
     * 注意: 每个 FMA 显式使用 k+offset 索引 A, 不使用宏中的隐式 k 引用,
     * 以避免在循环体内重用宏时产生错误的偏移量。
     */

/* 辅助宏: 对 6 行分别执行 FMA, 使用显式的 k 偏移量和 B 寄存器参数 */
#define MICRO_K2_PAIR(row, k_off_a, reg_a, k_off_b, reg_b)                   \
    do {                                                                      \
        c##row = _mm256_fmadd_ps(                                             \
            _mm256_set1_ps(A[(row) * lda + (k) + (k_off_a)]), reg_a, c##row);\
        c##row = _mm256_fmadd_ps(                                             \
            _mm256_set1_ps(A[(row) * lda + (k) + (k_off_b)]), reg_b, c##row);\
    } while (0)

/* 对全部 6 行应用相同的 k 偏移/B 寄存器组合 */
#define MICRO_K2_ALL(k_off_a, reg_a, k_off_b, reg_b)                          \
    do {                                                                      \
        MICRO_K2_PAIR(0, k_off_a, reg_a, k_off_b, reg_b);                    \
        MICRO_K2_PAIR(1, k_off_a, reg_a, k_off_b, reg_b);                    \
        MICRO_K2_PAIR(2, k_off_a, reg_a, k_off_b, reg_b);                    \
        MICRO_K2_PAIR(3, k_off_a, reg_a, k_off_b, reg_b);                    \
        MICRO_K2_PAIR(4, k_off_a, reg_a, k_off_b, reg_b);                    \
        MICRO_K2_PAIR(5, k_off_a, reg_a, k_off_b, reg_b);                    \
    } while (0)

    int k = 0;

    /* 展开 8x 的 K 循环: 每次处理 8 个 k 值, 使用 b0/b1 交替加载 */
    for (; k + 8 <= K_len; k += 8) {
        /* 第 0-1 次迭代: B[k+0], B[k+1] */
        __m256 b0 = _mm256_loadu_ps(&B[(k + 0) * ldb]);
        __m256 b1 = _mm256_loadu_ps(&B[(k + 1) * ldb]);
        MICRO_K2_ALL(0, b0, 1, b1);

        /* 第 2-3 次迭代: B[k+2], B[k+3] */
        b0 = _mm256_loadu_ps(&B[(k + 2) * ldb]);
        b1 = _mm256_loadu_ps(&B[(k + 3) * ldb]);
        MICRO_K2_ALL(2, b0, 3, b1);

        /* 第 4-5 次迭代: B[k+4], B[k+5] */
        b0 = _mm256_loadu_ps(&B[(k + 4) * ldb]);
        b1 = _mm256_loadu_ps(&B[(k + 5) * ldb]);
        MICRO_K2_ALL(4, b0, 5, b1);

        /* 第 6-7 次迭代: B[k+6], B[k+7] */
        b0 = _mm256_loadu_ps(&B[(k + 6) * ldb]);
        b1 = _mm256_loadu_ps(&B[(k + 7) * ldb]);
        MICRO_K2_ALL(6, b0, 7, b1);
    }

#undef MICRO_K2_PAIR
#undef MICRO_K2_ALL

    /* 处理 K 维度展开后的剩余尾部 (< 8 个 k 值) */
    for (; k < K_len; ++k) {
        __m256 b_vec = _mm256_loadu_ps(&B[k * ldb]);
        c0 = _mm256_fmadd_ps(_mm256_set1_ps(A[0 * lda + k]), b_vec, c0);
        c1 = _mm256_fmadd_ps(_mm256_set1_ps(A[1 * lda + k]), b_vec, c1);
        c2 = _mm256_fmadd_ps(_mm256_set1_ps(A[2 * lda + k]), b_vec, c2);
        c3 = _mm256_fmadd_ps(_mm256_set1_ps(A[3 * lda + k]), b_vec, c3);
        c4 = _mm256_fmadd_ps(_mm256_set1_ps(A[4 * lda + k]), b_vec, c4);
        c5 = _mm256_fmadd_ps(_mm256_set1_ps(A[5 * lda + k]), b_vec, c5);
    }

    /*
     * 存储累加器结果回 C 矩阵。
     *
     * 对于 nr_use < NR 的情况, 使用掩码 store 只写入有效的列。
     * 对于 mr_use < MR 的情况, 只执行有效行的 store。
     *
     * 掩码生成: (1 << nr_use) - 1 产生低 nr_use 位为 1 的位掩码。
     * 例如 nr_use=5 → mask=0b00011111 (bit 0-4 set).
     * _mm256_maskstore_ps 只写入掩码位为 1 的 lane。
     */
    if (nr_use == NR) {
        /* 快速路径: 完整的 8 列, 直接用 _mm256_storeu_ps */
        if (mr_use >= 1) _mm256_storeu_ps(&C[0 * ldc], c0);
        if (mr_use >= 2) _mm256_storeu_ps(&C[1 * ldc], c1);
        if (mr_use >= 3) _mm256_storeu_ps(&C[2 * ldc], c2);
        if (mr_use >= 4) _mm256_storeu_ps(&C[3 * ldc], c3);
        if (mr_use >= 5) _mm256_storeu_ps(&C[4 * ldc], c4);
        if (mr_use >= 6) _mm256_storeu_ps(&C[5 * ldc], c5);
    } else {
        /*
         * 部分列路径: 使用掩码 store。
         *
         * 创建掩码: mask[i] = (i < nr_use) ? ~0 : 0
         * 注意: AVX2 maskstore 期望 32-bit 整数掩码, 0xFFFFFFFF 表示写入,
         * 0x00000000 表示跳过。掩码向量本身占一个完整的 ymm 寄存器。
         */
        int mask_arr[8];
        for (int j = 0; j < NR; ++j) {
            mask_arr[j] = (j < nr_use) ? -1 : 0;  /* -1 = 0xFFFFFFFF */
        }
        __m256i mask = _mm256_loadu_si256((const __m256i*)mask_arr);

        if (mr_use >= 1) _mm256_maskstore_ps(&C[0 * ldc], mask, c0);
        if (mr_use >= 2) _mm256_maskstore_ps(&C[1 * ldc], mask, c1);
        if (mr_use >= 3) _mm256_maskstore_ps(&C[2 * ldc], mask, c2);
        if (mr_use >= 4) _mm256_maskstore_ps(&C[3 * ldc], mask, c3);
        if (mr_use >= 5) _mm256_maskstore_ps(&C[4 * ldc], mask, c4);
        if (mr_use >= 6) _mm256_maskstore_ps(&C[5 * ldc], mask, c5);
    }
}

/* =================================================================
 * 宏内核: macro_kernel (Macro-Kernel)
 *
 * 处理一个 M 面板 × N 面板 × K 面板的计算:
 *   C[mp:mp+mc][np:np+nc] += A[mp:mp+mc][kp:kp+kc] × B[kp:kp+kc][np:np+nc]
 *
 * 流程:
 *   1. 打包 B 面板 (kc × nc) → B_packed
 *   2. 打包 A 面板 (mc × kc) → A_packed
 *   3. 在 M 和 N 维度上按 MR×NR 微内核 tile 迭代:
 *      - 对每个 tile 调用 gemm_micro_6x8
 *
 * 此函数封装了一个 "KC 迭代" 的计算逻辑。
 * 打包缓冲区在外部 (main_tiled_loop) 分配和重用。
 *
 * 参数:
 *   mc, nc, kc       - 当前面板的实际尺寸
 *   A, lda, B, ldb   - 原始矩阵及其 leading dimensions
 *   C, ldc           - 输出矩阵
 *   mp, np, kp       - 当前面板在全局矩阵中的偏移
 *   A_packed         - A 打包缓冲区 (mc_padded × kc)
 *   B_packed         - B 打包缓冲区 (kc × nc_padded)
 * ================================================================= */

static void macro_kernel(int mc, int nc, int kc,
                         const float* A, int lda,
                         const float* B, int ldb,
                         float* C, int ldc,
                         int mp, int np, int kp,
                         float* A_packed, float* B_packed) {
    /* 向上对齐到微内核尺寸, 用于打包缓冲区布局 */
    int mc_padded = ((mc + MR - 1) / MR) * MR;
    int nc_padded = ((nc + NR - 1) / NR) * NR;

    /* ---- 打包 A 面板 ---- */
    pack_A(A_packed, A, lda, mp, kp, mc, kc, mc_padded);

    /* ---- 打包 B 面板 ---- */
    pack_B(B_packed, B, ldb, kp, np, kc, nc, nc_padded);

    /*
     * ---- 在 M 和 N 维度上按微内核 tile 迭代 ----
     *
     * A_packed 的 stride: lda_packed = kc (因为 A_packed 是 mc_padded × kc)
     * B_packed 的 stride: ldb_packed = nc_padded (因为 B_packed 是 kc × nc_padded)
     *
     * 对于 B, 微内核需要从列偏移 nj_local 开始读取:
     *   &B_packed[nj_local] 指向第 0 行的第 nj_local 列
     *   前进一行需要 stride = nc_padded
     *   所以第 k 行的 nj_local 列是 B_packed[k * nc_padded + nj_local] ✓
     */
    for (int mi = 0; mi < mc; mi += MR) {
        int mr_use = (mi + MR <= mc) ? MR : (mc - mi);
        for (int nj = 0; nj < nc; nj += NR) {
            int nr_use = (nj + NR <= nc) ? NR : (nc - nj);

            gemm_micro_6x8(
                kc,
                &A_packed[mi * kc],   /* A 面板: 指向第 mi 行 */
                kc,                   /* lda: A_packed stride = kc */
                &B_packed[nj],        /* B 面板: 指向第 nj 列 */
                nc_padded,            /* ldb: B_packed stride = nc_padded */
                &C[(mp + mi) * ldc + np + nj],  /* C 子矩阵的开始 */
                ldc,                  /* C 的 leading dimension */
                mr_use, nr_use        /* 边缘 tile 的实际尺寸 */
            );
        }
    }
}

/* =================================================================
 * 主分块循环: gemm_tiled_impl (Main Tiled Loop)
 *
 * 三层缓存分块结构:
 *   第 1 层 (L3 缓存, M/N 维度): 在 M 和 N 上做大粒度分块
 *   第 2 层 (L2 缓存, K 维度): 在 K 上做面板分块
 *   第 3 层 (L1 缓存, 微内核): 6×8 微内核在寄存器中累加
 *
 * 循环嵌套顺序 (遵循 BLIS 的标准顺序):
 *   for mp in 0..M step MC:          # M 面板 (L3)
 *     for np in 0..N step NC:        # N 面板 (L3)
 *       for kp in 0..K step KC:      # K 面板 (L2)
 *         打包 B (KC × NC)           # B 面板常驻 L2 → 复用所有 M 微 tile
 *         for mi in 0..MC step MR:   # 微内核行
 *           for nj in 0..NC step NR: # 微内核列
 *             调用 gemm_micro_6x8
 *
 * 注意: 在工业实现中, B 面板在外层 (K 循环) 打包, 然后在 M/N 循环中
 * 被多次复用。A 面板在 K 循环内打包 (每次 K 迭代都需要不同的 A 面板)。
 * 但为了简化实现, 我们的 macro_kernel 同时打包 A 和 B。
 *
 * 参数:
 *   M, N, K  - 矩阵维度
 *   A, lda, B, ldb, C, ldc - 矩阵数据和 leading dimensions
 * ================================================================= */

static void gemm_tiled_impl(int M, int N, int K,
                            const float* A, int lda,
                            const float* B, int ldb,
                            float* C, int ldc) {
    /*
     * 预分配打包缓冲区。
     *
     * A_packed: mc_padded_max × kc_max
     *   = MR×ceil(MC/MR) × KC = 6×ceil(384/6)×256 = 6×64×256 floats
     *   ≈ 6 × 64 × 256 × 4 = 384 KB
     *
     * B_packed: kc_max × nc_padded_max
     *   = KC × NR×ceil(NC/NR) = 256 × 8×ceil(NC/8)
     *   对于 NC=4096: 256 × 4096 × 4 = 4 MB
     *   对于小矩阵 (N < NC): 256 × NR×ceil(N/NR) × 4
     *
     * 实际分配按当前 N 面板的最大需求。
     */
    int mc_max = (MC < M) ? MC : M;
    int mc_padded_max = ((mc_max + MR - 1) / MR) * MR;
    int kc_max = (KC < K) ? KC : K;

    int nc_max = (NC < N) ? NC : N;
    int nc_padded_max = ((nc_max + NR - 1) / NR) * NR;

    size_t a_packed_size = (size_t)mc_padded_max * (size_t)kc_max;
    size_t b_packed_size = (size_t)kc_max * (size_t)nc_padded_max;

    float* A_packed = ALIGNED_ALLOC(float, a_packed_size, 32);
    float* B_packed = ALIGNED_ALLOC(float, b_packed_size, 32);

    if (!A_packed || !B_packed) {
        fprintf(stderr, "gemm_tiled_impl: 打包缓冲区分配失败\n");
        ALIGNED_FREE(A_packed);
        ALIGNED_FREE(B_packed);
        return;
    }

    /*
     * 三层分块循环。
     *
     * 注意: mp, np, kp 使用 int (32-bit) 就足够,
     * 但为了与 size_t 的算术兼容, 使用 int 类型。
     */
    for (int mp = 0; mp < M; mp += MC) {
        int mc = (mp + MC <= M) ? MC : (M - mp);

        for (int np = 0; np < N; np += NC) {
            int nc = (np + NC <= N) ? NC : (N - np);

            for (int kp = 0; kp < K; kp += KC) {
                int kc = (kp + KC <= K) ? KC : (K - kp);

                /*
                 * 调用宏内核处理一个 (mc × nc × kc) 的面板。
                 *
                 * 打包缓冲区大小足够的条件:
                 *   A_packed: mc_padded × kc (< mc_padded_max × kc_max)
                 *   B_packed: kc × nc_padded (< kc_max × nc_padded_max)
                 *
                 * 注意: nc_padded 每次可能不同 (当 N 不是 NC 的倍数时).
                 * B_packed 的 stride 由 nc_padded 决定, 在 macro_kernel 中使用。
                 */
                macro_kernel(mc, nc, kc,
                             A, lda, B, ldb, C, ldc,
                             mp, np, kp,
                             A_packed, B_packed);
            }
        }
    }

    ALIGNED_FREE(A_packed);
    ALIGNED_FREE(B_packed);
}

/* =================================================================
 * 公共 API: gemm_production_avx2 (Main Entry Point)
 *
 * 生产级 AVX2 GEMM 的入口函数。
 *
 * 计算: C[M][N] += A[M][K] × B[K][N]
 * 所有矩阵均为行主序 (row-major), 支持任意维度。
 *
 * 参数:
 *   M, N, K  - 矩阵维度
 *   A         - 左矩阵 (M × K), 行主序
 *   lda       - A 的 leading dimension (≥ K)
 *   B         - 右矩阵 (K × N), 行主序
 *   ldb       - B 的 leading dimension (≥ N)
 *   C         - 输出矩阵 (M × N), 行主序, 累加模式 (C += A×B)
 *   ldc       - C 的 leading dimension (≥ N)
 *
 * 使用示例:
 *   float* A = ...; // M × K
 *   float* B = ...; // K × N
 *   float* C = ...; // M × N, 初始化为 0
 *   gemm_production_avx2(M, N, K, A, K, B, N, C, N);
 * ================================================================= */

extern "C"
void gemm_production_avx2(int M, int N, int K,
                           const float* A, int lda,
                           const float* B, int ldb,
                           float* C, int ldc) {
    gemm_tiled_impl(M, N, K, A, lda, B, ldb, C, ldc);
}

/* =================================================================
 * 辅助函数: 计算 GFLOPS
 * ================================================================= */

static double compute_gflops(double elapsed_ns, int M, int N, int K) {
    /* GEMM 浮点运算量: 2 × M × N × K (每次 FMA 算 2 次浮点运算) */
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double seconds = elapsed_ns * 1e-9;
    return (seconds > 0.0) ? flops / seconds / 1e9 : 0.0;
}

/* =================================================================
 * main: 正确性验证 + 性能基准测试
 *
 * 测试流程:
 *   1. 检测 AVX2 支持
 *   2. 用小矩阵验证正确性 (对比标量 GEMM)
 *   3. 用大矩阵做性能基准测试 (对比标量和朴素 SIMD)
 * ================================================================= */

int main() {
    cpu_print_features();

    if (!cpu_has_avx2()) {
        printf("当前 CPU 不支持 AVX2. 程序退出.\n");
        return 1;
    }

    /* =============================================================
     * 第一阶段: 正确性验证 (小矩阵)
     * ============================================================= */

    printf("\n========== 第一阶段: 正确性验证 (Correctness) ==========\n\n");

    /* 使用非对称维度测试边缘情况 */
    const int TEST_M = 17;  /* 非 MR 倍数, 测试行边界 */
    const int TEST_N = 23;  /* 非 NR 倍数, 测试列边界 */
    const int TEST_K = 31;  /* 非 8 倍数, 测试 K 尾部 */

    printf("测试维度: M=%d, N=%d, K=%d\n", TEST_M, TEST_N, TEST_K);
    printf("  注意: 维度不是 MR=%d/NR=%d 的倍数, 用于测试边缘处理\n\n", MR, NR);

    /* 分配小矩阵 */
    size_t nelem_a = (size_t)TEST_M * (size_t)TEST_K;
    size_t nelem_b = (size_t)TEST_K * (size_t)TEST_N;
    size_t nelem_c = (size_t)TEST_M * (size_t)TEST_N;

    float* A_small = ALIGNED_ALLOC(float, nelem_a, 32);
    float* B_small = ALIGNED_ALLOC(float, nelem_b, 32);
    float* C_small = ALIGNED_ALLOC(float, nelem_c, 32);
    float* C_ref   = ALIGNED_ALLOC(float, nelem_c, 32);

    if (!A_small || !B_small || !C_small || !C_ref) {
        fprintf(stderr, "小矩阵分配失败.\n");
        return 1;
    }

    /* 填充随机数据 */
    rand_xorshift64_seed(42);
    fill_random_f32(A_small, nelem_a);
    rand_xorshift64_seed(99);
    fill_random_f32(B_small, nelem_b);

    /* 初始化为零 (C += A×B 从零开始) */
    memset(C_small, 0, nelem_c * sizeof(float));
    memset(C_ref,   0, nelem_c * sizeof(float));

    /* 计算标量参考结果 */
    scalar_gemm(TEST_M, TEST_N, TEST_K,
                A_small, TEST_K, B_small, TEST_N, C_ref, TEST_N);

    /* 使用生产级 GEMM 计算 (C_small 初始化为 0, 然后 gemm 做 +=) */
    gemm_production_avx2(TEST_M, TEST_N, TEST_K,
                         A_small, TEST_K, B_small, TEST_N, C_small, TEST_N);

    /*
     * 验证: 比较生产级 GEMM 与标量参考结果。
     *
     * 浮点加法不满足结合律, 不同的累加顺序会产生微小的数值差异。
     * 容差需要足够大以适应 K 次累加的误差累积。
     * 对于 K=31, 每个输出元素进行了 31 次 FMA。
     */
    float tol = 1e-4f * (float)TEST_K;  /* ≈ 0.0031 */
    CHECK_NEAR_ARRAY(C_small, C_ref, nelem_c, tol,
                     "生产级 AVX2 GEMM 与标量参考一致");

    /* 额外验证: 使用完美对齐的维度 (MR/NR 的倍数) */
    {
        const int TEST_M2 = MR * 10;  /* 60 */
        const int TEST_N2 = NR * 10;  /* 80 */
        const int TEST_K2 = 8 * 10;   /* 80, 也是 8 的倍数 */

        printf("\n  额外验证: M=%d, N=%d, K=%d (维度对齐)\n",
               TEST_M2, TEST_N2, TEST_K2);

        size_t n_a2 = (size_t)TEST_M2 * (size_t)TEST_K2;
        size_t n_b2 = (size_t)TEST_K2 * (size_t)TEST_N2;
        size_t n_c2 = (size_t)TEST_M2 * (size_t)TEST_N2;

        float* A2 = ALIGNED_ALLOC(float, n_a2, 32);
        float* B2 = ALIGNED_ALLOC(float, n_b2, 32);
        float* C2 = ALIGNED_ALLOC(float, n_c2, 32);
        float* R2 = ALIGNED_ALLOC(float, n_c2, 32);

        rand_xorshift64_seed(123);
        fill_random_f32(A2, n_a2);
        rand_xorshift64_seed(456);
        fill_random_f32(B2, n_b2);
        memset(C2, 0, n_c2 * sizeof(float));
        memset(R2, 0, n_c2 * sizeof(float));

        scalar_gemm(TEST_M2, TEST_N2, TEST_K2,
                    A2, TEST_K2, B2, TEST_N2, R2, TEST_N2);
        gemm_production_avx2(TEST_M2, TEST_N2, TEST_K2,
                             A2, TEST_K2, B2, TEST_N2, C2, TEST_N2);

        float tol2 = 2e-4f * (float)TEST_K2;
        CHECK_NEAR_ARRAY(C2, R2, n_c2, tol2,
                         "对齐维度: 生产级 AVX2 GEMM 正确");

        ALIGNED_FREE(A2);
        ALIGNED_FREE(B2);
        ALIGNED_FREE(C2);
        ALIGNED_FREE(R2);
    }

    ALIGNED_FREE(A_small);
    ALIGNED_FREE(B_small);
    ALIGNED_FREE(C_small);
    ALIGNED_FREE(C_ref);

    /* =============================================================
     * 第二阶段: 性能基准测试 (大矩阵)
     * ============================================================= */

    printf("\n========== 第二阶段: 性能基准测试 (Benchmark) ==========\n\n");

    /*
     * 选择中等大小的矩阵进行基准测试:
     *   M=384, N=512, K=256
     *
     * 这些维度可以触发多层缓存分块:
     *   - M=384 匹配 MC (M 面板适合 L2)
     *   - K=256 匹配 KC (K 面板适合 L1/L2)
     *   - N=512 适中, 不会超过 L3
     */
    const int BENCH_M = 384;
    const int BENCH_N = 512;
    const int BENCH_K = 256;

    printf("基准测试维度: M=%d, N=%d, K=%d\n", BENCH_M, BENCH_N, BENCH_K);
    printf("SIMD 宽度: 256-bit (8×f32 每 ymm 寄存器)\n");
    printf("理论峰值: %.0f GFLOPS/核 "
           "(2 FMA 单元 × 8 flops/周期 × ~3 GHz)\n\n",
           THEORETICAL_PEAK_GFLOPS);

    /* 分配大矩阵 */
    size_t bn_a = (size_t)BENCH_M * (size_t)BENCH_K;
    size_t bn_b = (size_t)BENCH_K * (size_t)BENCH_N;
    size_t bn_c = (size_t)BENCH_M * (size_t)BENCH_N;

    float* A_bench = ALIGNED_ALLOC(float, bn_a, 32);
    float* B_bench = ALIGNED_ALLOC(float, bn_b, 32);
    float* C_bench = ALIGNED_ALLOC(float, bn_c, 32);
    float* C_work  = ALIGNED_ALLOC(float, bn_c, 32);

    if (!A_bench || !B_bench || !C_bench || !C_work) {
        fprintf(stderr, "基准测试矩阵分配失败.\n");
        return 1;
    }

    /* 填充随机数据 */
    rand_xorshift64_seed(42);
    fill_random_f32(A_bench, bn_a);
    rand_xorshift64_seed(99);
    fill_random_f32(B_bench, bn_b);

    /*
     * 内存字节数 (读+写):
     *   A 读取: M×K 次
     *   B 读取: K×N 次
     *   C 读取 + 写入: 2×M×N 次
     */
    size_t nelem_bench = (size_t)BENCH_M * (size_t)BENCH_N;
    size_t bytes_rw = (bn_a + bn_b + bn_c * 2) * sizeof(float);

    benchmark_result_t results[3];
    memset(results, 0, sizeof(results));

    int bench_iters = 20;

    /* ---- 基准 1: 标量 GEMM ---- */
    printf("  运行标量基准测试 (%d 次迭代)...\n", bench_iters);
    {
        float* Ab = A_bench; float* Bb = B_bench; float* Cb = C_work;
        BENCH_COMPUTE(
            memset(Cb, 0, bn_c * sizeof(float));
            scalar_gemm(BENCH_M, BENCH_N, BENCH_K,
                        Ab, BENCH_K, Bb, BENCH_N, Cb, BENCH_N);
            volatile float* _v = Cb; (void)_v;,
            nelem_bench, bytes_rw, bench_iters, results[0]);
        results[0].name = "标量 GEMM (scalar)";
    }

    /* ---- 基准 2: 朴素 SIMD GEMM ---- */
    printf("  运行朴素 SIMD 基准测试 (%d 次迭代)...\n", bench_iters);
    {
        float* Ab = A_bench; float* Bb = B_bench; float* Cb = C_work;
        BENCH_COMPUTE(
            memset(Cb, 0, bn_c * sizeof(float));
            gemm_naive_simd(BENCH_M, BENCH_N, BENCH_K,
                            Ab, BENCH_K, Bb, BENCH_N, Cb, BENCH_N);
            volatile float* _v = Cb; (void)_v;,
            nelem_bench, bytes_rw, bench_iters, results[1]);
        results[1].name = "朴素 SIMD GEMM (naive)";
    }

    /* ---- 基准 3: 生产级 AVX2 GEMM ---- */
    printf("  运行生产级 AVX2 GEMM 基准测试 (%d 次迭代)...\n", bench_iters);
    {
        float* Ab = A_bench; float* Bb = B_bench; float* Cb = C_work;
        BENCH_COMPUTE(
            memset(Cb, 0, bn_c * sizeof(float));
            gemm_production_avx2(BENCH_M, BENCH_N, BENCH_K,
                                 Ab, BENCH_K, Bb, BENCH_N, Cb, BENCH_N);
            volatile float* _v = Cb; (void)_v;,
            nelem_bench, bytes_rw, bench_iters, results[2]);
        results[2].name = "生产级 AVX2 GEMM (production)";
    }

    /* ---- 打印基准测试结果 ---- */
    bench_report(results, 3);

    /* ---- GFLOPS 分析 ---- */
    printf("--- GFLOPS 分析 ---\n\n");
    for (int i = 0; i < 3; ++i) {
        double gflops = compute_gflops(results[i].elapsed_ns,
                                       BENCH_M, BENCH_N, BENCH_K);
        double pct_peak = (gflops / THEORETICAL_PEAK_GFLOPS) * 100.0;
        printf("  %-35s  %8.3f GFLOPS  (%5.1f%% 峰值效率)\n",
               results[i].name, gflops, pct_peak);
    }

    /* =============================================================
     * 第三阶段: 大矩阵测试 (超出 L3 缓存)
     * ============================================================= */

    printf("\n--- 大矩阵测试 (M=1024, N=1024, K=1024) ---\n\n");

    const int BIG_M = 1024;
    const int BIG_N = 1024;
    const int BIG_K = 1024;

    size_t big_a = (size_t)BIG_M * BIG_K;
    size_t big_b = (size_t)BIG_K * BIG_N;
    size_t big_c = (size_t)BIG_M * BIG_N;

    float* Abig = ALIGNED_ALLOC(float, big_a, 32);
    float* Bbig = ALIGNED_ALLOC(float, big_b, 32);
    float* Cbig = ALIGNED_ALLOC(float, big_c, 32);

    if (!Abig || !Bbig || !Cbig) {
        fprintf(stderr, "大矩阵分配失败 (内存不足). 跳过大矩阵测试.\n");
    } else {
        rand_xorshift64_seed(42);
        fill_random_f32(Abig, big_a);
        rand_xorshift64_seed(99);
        fill_random_f32(Bbig, big_b);

        size_t big_nelem = (size_t)BIG_M * BIG_N;
        size_t big_bytes = (big_a + big_b + big_c * 2) * sizeof(float);

        benchmark_result_t big_results[1];
        memset(big_results, 0, sizeof(big_results));

        printf("  运行生产级 AVX2 GEMM (5 次迭代, 可能较慢)...\n");
        {
            float* Ab = Abig; float* Bb = Bbig; float* Cb = Cbig;
            BENCH_COMPUTE(
                memset(Cb, 0, big_c * sizeof(float));
                gemm_production_avx2(BIG_M, BIG_N, BIG_K,
                                     Ab, BIG_K, Bb, BIG_N, Cb, BIG_N);
                volatile float* _v = Cb; (void)_v;,
                big_nelem, big_bytes, 5, big_results[0]);
            big_results[0].name = "生产级 AVX2 GEMM (1024³)";
        }

        bench_report(big_results, 1);

        double big_gflops = compute_gflops(big_results[0].elapsed_ns,
                                           BIG_M, BIG_N, BIG_K);
        double big_pct = (big_gflops / THEORETICAL_PEAK_GFLOPS) * 100.0;
        printf("  %-35s  %8.3f GFLOPS  (%5.1f%% 峰值效率)\n\n",
               big_results[0].name, big_gflops, big_pct);
    }

    ALIGNED_FREE(Abig);
    ALIGNED_FREE(Bbig);
    ALIGNED_FREE(Cbig);

    /* =============================================================
     * 总结
     * ============================================================= */

    printf("--- 实现说明 ---\n\n");

    printf("多层缓存分块策略:\n");
    printf("  - L1 微内核: %d×%d (6×MR 行, 8×NR 列) YMM 寄存器累加\n", MR, NR);
    printf("  - L2 宏内核: KC=%d K 面板, 使 A+B 面板驻留 L1/L2\n", KC);
    printf("  - L3 外部分块: MC=%d, NC=%d 用于大矩阵缓存复用\n\n", MC, NC);

    printf("微内核寄存器使用 (共 16 个 ymm):\n");
    printf("  - 6 个 C 累加器 (c0..c5)\n");
    printf("  - 2 个 B 交替加载寄存器 (b0, b1)\n");
    printf("  - 1 个 A 广播寄存器 (a_brd)\n");
    printf("  - 总计 9/16 寄存器 (56%%)\n\n");

    printf("8x K 展开展开策略:\n");
    printf("  - B 加载与 FMA 计算交错执行\n");
    printf("  - 每次 K 迭代处理 8 个连续 k 值\n");
    printf("  - 6 行 × 8 FMAs = 48 条独立 FMA, 隐藏 4 周期延迟\n\n");

    printf("打包策略:\n");
    printf("  - B 面板打包为 KC×NC (列连续) 以支持 SIMD 连续加载\n");
    printf("  - A 面板打包为 MC×KC (行为主序) 以支持广播访问\n");
    printf("  - 边缘用零填充, 支持任意维度 (非 MR/NR 倍数)\n");

    ALIGNED_FREE(A_bench);
    ALIGNED_FREE(B_bench);
    ALIGNED_FREE(C_bench);
    ALIGNED_FREE(C_work);

    printf("\n所有测试完成.\n");
    return 0;
}
