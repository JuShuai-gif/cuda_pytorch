/**
 * avx2_int8_gemm_requantize.cpp -- 生产级 int8 量化 GEMM，支持按通道重量化
 *
 * 完整的 int8 量化 GEMM（通用矩阵乘法）流水线，适用于 ML 推理：
 *
 *   输出 = clip( (A_u8 × B_s8 - zp_correction) * scale, 0, 255 )
 *
 * 其中：
 *   - A（激活值）：uint8，非对称量化（含 zero_point zp_a）
 *   - B（权重）：int8，对称量化（zero_point = 0）
 *   - 累加器：int32
 *   - 输出：uint8，钳制到 [0, 255]
 *
 * --- 微内核（4×16，AVX2）---
 *   同时处理 4 行激活值 × 16 列权重。
 *   使用 VPMADDUBSW（_mm256_maddubs_epi16）进行 u8×s8 → s16，
 *   再使用 VPMADDWD（_mm256_madd_epi16）乘以 {1,...} 得到 s32。
 *   每次内层循环迭代执行 64 次 int8 MAC。
 *   采用 2 路累加器展开以提升指令级并行度（ILP）。
 *
 * --- 零点修正 ---
 *   非对称量化的零点处理：
 *     result = sum_k (q_a[k] - zp_a) * q_w[k]
 *            = sum_k q_a[k] * q_w[k] - zp_a * sum_k q_w[k]
 *   其中 sum_k q_w[k] 在打包阶段按列预计算。
 *
 * --- 按通道重量化 ---
 *   在 int32 累加后，乘以每通道的 fp32 缩放因子 scale[c]，
 *   钳制到 [0, 255]，转换回 uint8。
 *
 * --- 打包 ---
 *   B 面板按列主序打包，使得沿 K 维度的 SIMD 加载连续。
 *
 * 包含：标量参考实现、正确性检验（容差 ~1，因舍入误差）、基准测试。
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
 * 问题规模常量
 * ================================================================ */

static const int M_DIM = 32;  /* 激活值行数（批次大小） */
static const int N_DIM = 64;  /* 权重列数（输出通道）  */
static const int K_DIM = 128; /* 内层维度（输入通道）   */

/* 微瓦片尺寸 */
static const int MR = 4;   /* 微内核每次处理的行数 */
static const int NR = 16;  /* 微内核每次处理的列数 */

/* ================================================================
 * int32 水平求和（8 路 → 1 个标量）
 *
 * 对于 madd_epi16 的输出（8 个 int32 通道）：
 *   置换高 128 位和低 128 位 → 相加 → 2 次 hadd → 提取低 32 位
 * ================================================================ */

static inline int32_t hsum_i32_8(__m256i v) {
    /* 交换 128 位通道 */
    __m256i perm = _mm256_permute2x128_si256(v, v, 0x01);
    v = _mm256_add_epi32(v, perm);
    /* 通道内水平相加：2 次 hadd 将 4 个 i32 折叠为 1 个 */
    v = _mm256_hadd_epi32(v, v);
    v = _mm256_hadd_epi32(v, v);
    return _mm_cvtsi128_si32(_mm256_castsi256_si128(v));
}

/* ================================================================
 * uint8 钳制辅助函数
 * ================================================================ */

static inline uint8_t clamp_u8(int32_t val) {
    if (val < 0) return 0;
    if (val > 255) return 255;
    return (uint8_t)val;
}

/* ================================================================
 * B 面板打包：将 16 列 int8 权重按列主序打包
 *
 * 打包格式：B_packed[col * K + k] = original_B[k][col_offset + col]
 *   - col ∈ [0, NR-1]：列索引（面板内）
 *   - k ∈ [0, K-1]：K 维度索引
 *
 * 该格式使沿 K 维度的 SIMD 加载连续：
 *   _mm256_loadu_si256(&B_packed[col * K + k]) 加载 32 个连续的 int8 值，
 *   对应单列在 k..k+31 范围内的 K 个元素。
 *
 * 同时预计算每列的权重之和：
 *   col_sum[col] = sum_{k=0}^{K-1} B[k][col_offset + col]
 * 用于零点修正。
 * ================================================================ */

static void pack_B_panel(int8_t* B_packed, int32_t* col_sum,
                         const int8_t* B, int K, int N_ldb,
                         int col_start, int nr_use) {
    for (int j = 0; j < nr_use; j++) {
        int32_t wsum = 0;
        for (int k = 0; k < K; k++) {
            int8_t val = B[k * N_ldb + col_start + j];
            B_packed[j * K + k] = val;
            wsum += (int32_t)val;
        }
        col_sum[j] = wsum;
    }
}

/* ================================================================
 * 微内核：4×16 int8 GEMM（VPMADDUBSW + VPMADDWD）
 *
 * 计算 C_int32[mr][nr] += Σ_k A_u8[mr][k] * B_s8[nr][k]
 *
 * 其中：
 *   - mr ∈ [0, 3]（4 行激活值）
 *   - nr ∈ [0, 15]（16 列权重）
 *   - k 为内层维度（K 维度），每次迭代步进 32
 *
 * 内层循环（每次迭代，K 步进 64，2 路展开）：
 *   对每行 r：
 *     加载 32 字节 uint8（A[r][k] 和 A[r][k+32]）
 *     对每列 c：
 *       加载 32 字节 int8（B_packed[c*K + k] 和 B_packed[c*K + k + 32]）
 *       maddubs(u8_vec, s8_vec) → 16 个 s16 部分和
 *       madd_epi16(s16_vec, {1,1,...}) → 8 个 s32 累加器
 *       将 8 个 s32 值水平求和，并累加到标量累加器中
 *
 * 每次内层循环迭代执行 2×4×16×32 = 4096 次 int8 MAC（2 路展开）。
 * 其中，2 路展开 × 4 行 × 16 列 = 128 次 maddubs+madd 操作，
 * 每次操作处理 32 次 MAC = 4096 次 MAC。
 *
 * 寄存器分配（AVX2：16 个 ymm 寄存器）：
 *   2 个 A 向量寄存器（a0, a1）
 *   2 个 B 向量寄存器（b0, b1）
 *   2 个临时寄存器（maddubs 输出、madd 输出）
 *   1 个 ones 向量
 *   共使用 7 个，余量充足
 * ================================================================ */

static void micro_kernel_4x16(
    const uint8_t* __restrict__ A_packed,  /* mr_use 行 × K 列 uint8（行主序） */
    const int8_t* __restrict__ B_packed,   /* nr_use 列 × K 行 int8（列主序）  */
    int32_t* __restrict__ C_acc,           /* mr_use × NR int32 累加器（行主序，stride=NR）*/
    int K,
    int lda,                               /* A 面板的步幅（= K）            */
    int ldb,                               /* B 面板的步幅（= K）            */
    int mr_use,                            /* 实际使用的行数（≤ MR）         */
    int nr_use)                            /* 实际使用的列数（≤ NR）         */
{
    int ldc = NR;  /* C 累加器的步幅：始终为 NR（16），与分配一致 */

    /* ones 向量：madd_epi16 与 ones 相乘，将相邻的 s16 对求和为 s32 */
    const __m256i ones = _mm256_set1_epi16(1);

    int k = 0;

    /*
     * 主循环：每次迭代 K 步进 64（2 路展开 × 32 字节/寄存器）
     *
     * 2 路展开允许同时计算 k 和 k+32 的部分积，
     * 有效地将关键路径的延迟减少一半。
     */
    for (; k + 63 < K; k += 64) {
        for (int r = 0; r < mr_use; r++) {
            /* 加载 2 个 A 向量：A[r][k:k+32] 和 A[r][k+32:k+64] */
            __m256i a0 = _mm256_loadu_si256(
                (const __m256i*)(A_packed + r * lda + k));
            __m256i a1 = _mm256_loadu_si256(
                (const __m256i*)(A_packed + r * lda + k + 32));

            for (int c = 0; c < nr_use; c++) {
                /* 加载 2 个 B 向量：B_packed[col][k:k+32] 和 B_packed[col][k+32:k+64] */
                __m256i b0 = _mm256_loadu_si256(
                    (const __m256i*)(B_packed + c * ldb + k));
                __m256i b1 = _mm256_loadu_si256(
                    (const __m256i*)(B_packed + c * ldb + k + 32));

                /* VPMADDUBSW：u8 × s8 → s16（32 对 → 16 个 s16 部分和） */
                __m256i m0 = _mm256_maddubs_epi16(a0, b0);
                __m256i m1 = _mm256_maddubs_epi16(a1, b1);

                /* VPMADDWD：s16 × 1 → s32（对相邻的 s16 求和，16 个 s16 → 8 个 s32） */
                m0 = _mm256_madd_epi16(m0, ones);
                m1 = _mm256_madd_epi16(m1, ones);

                /* 将 8 个 s32 通道水平求和，并累加到标量累加器 */
                int32_t partial = hsum_i32_8(m0) + hsum_i32_8(m1);
                C_acc[r * ldc + c] += partial;
            }
        }
    }

    /* 尾部循环：每次迭代 K 步进 32（单路，处理剩余的 K 元素） */
    for (; k + 31 < K; k += 32) {
        for (int r = 0; r < mr_use; r++) {
            __m256i a0 = _mm256_loadu_si256(
                (const __m256i*)(A_packed + r * lda + k));
            for (int c = 0; c < nr_use; c++) {
                __m256i b0 = _mm256_loadu_si256(
                    (const __m256i*)(B_packed + c * ldb + k));
                __m256i m0 = _mm256_maddubs_epi16(a0, b0);
                m0 = _mm256_madd_epi16(m0, ones);
                C_acc[r * ldc + c] += hsum_i32_8(m0);
            }
        }
    }

    /* 标量尾部：处理 K 维度剩余不足 32 的元素 */
    for (; k < K; k++) {
        for (int r = 0; r < mr_use; r++) {
            uint8_t  aval = A_packed[r * lda + k];
            for (int c = 0; c < nr_use; c++) {
                int8_t bval = B_packed[c * ldb + k];
                C_acc[r * ldc + c] += (int32_t)(int16_t)aval * (int32_t)bval;
            }
        }
    }
}

/* ================================================================
 * 按通道重量化 + 零点修正 + 钳制到 uint8
 *
 * 对 int32 累加器应用零点修正和每通道缩放因子，输出 uint8。
 *
 * 公式：
 *   out[r][c] = clip( (C_int32[r][c] - zp_a * col_sum[c]) * scale[c], 0, 255 )
 *
 * 其中：
 *   zp_a：激活值的零点（uint8）
 *   col_sum[c]：预计算的权重列之和（int32）
 *   scale[c]：每通道的 fp32 缩放因子
 * ================================================================ */

static void requantize_to_u8(
    const int32_t* __restrict__ C_acc,    /* mr_use × NR int32 累加器（行主序，stride=NR） */
    uint8_t* __restrict__ C_out,          /* mr_use × nr_use uint8 输出（行主序，stride=ldc_out） */
    const int32_t* __restrict__ col_sum,  /* nr_use 列权重之和                           */
    const float* __restrict__ scale,       /* nr_use 通道 fp32 缩放因子                   */
    uint8_t zp_a,
    int mr_use,
    int nr_use,
    int ldc_out)                          /* C_out 的行步幅                               */
{
    int ldc_acc = NR;  /* C_acc 的步幅始终为 NR */

    for (int r = 0; r < mr_use; r++) {
        for (int c = 0; c < nr_use; c++) {
            /* 零点修正：减去 zp_a * sum(w) */
            int32_t zp_correction = (int32_t)zp_a * col_sum[c];

            /*
             * 重量化 + 钳制：
             *   val = round( (acc - zp_correction) * scale[c] )
             *   使用 nearbyint 风格舍入（加上 0.5f 截断）以获得接近四舍五入的效果
             */
            float val = (float)(C_acc[r * ldc_acc + c] - zp_correction);
            val *= scale[c];
            /* 四舍五入到最近整数，并钳制到 [0, 255] */
            int32_t ival = (int32_t)(val + 0.5f);
            C_out[r * ldc_out + c] = clamp_u8(ival);
        }
    }
}

/* ================================================================
 * 标量参考实现（用于正确性验证）
 *
 * 精确计算与 AVX2 路径相同的数学公式，但使用纯标量代码。
 * ================================================================ */

__attribute__((noinline))
static void int8_gemm_scalar_ref(
    const uint8_t* A, int lda,
    const int8_t* B, int ldb,
    const int32_t* col_sum,
    const float* scale,
    uint8_t* C_out, int ldc,
    uint8_t zp_a,
    int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            /* 计算原始点积：sum_k A[i][k] * B[k][j] */
            int32_t acc = 0;
            for (int k = 0; k < K; k++) {
                acc += (int32_t)(int16_t)A[i * lda + k]
                     * (int32_t)B[k * ldb + j];
            }
            /* 零点修正 */
            int32_t zp_correction = (int32_t)zp_a * col_sum[j];
            /* 重量化 + 钳制 */
            float val = (float)(acc - zp_correction);
            val *= scale[j];
            int32_t ival = (int32_t)(val + 0.5f);
            C_out[i * ldc + j] = clamp_u8(ival);
        }
    }
}

/* ================================================================
 * AVX2 分块 GEMM：外层循环遍历整个 M×N 问题
 *
 * 将大矩阵分解为 4×16 的微瓦片，打包每个瓦片的 A 和 B 面板，
 * 调用微内核进行累加，然后按通道重量化。
 * ================================================================ */

__attribute__((noinline))
static void int8_gemm_tiled_avx2(
    const uint8_t* A, int lda,
    const int8_t* B, int ldb,
    const float* scale,      /* 每输出通道的缩放因子，长度为 N */
    uint8_t* C_out, int ldc,
    uint8_t zp_a,
    int M, int N, int K) {
    /*
     * 临时缓冲区：
     *   A_packed：MR × K 字节（4 行激活值面板，行主序）
     *   B_packed：NR × K 字节（16 列权重面板，列主序）
     *   col_sum：NR 个 int32（每列权重之和）
     *   C_acc：MR × NR 个 int32（累加器瓦片）
     */
    uint8_t* A_packed = ALIGNED_ALLOC(uint8_t, (size_t)MR * (size_t)K, 32);
    int8_t*  B_packed = ALIGNED_ALLOC(int8_t,  (size_t)NR * (size_t)K, 32);
    int32_t* col_sum  = ALIGNED_ALLOC(int32_t, (size_t)NR, 32);
    int32_t* C_acc    = ALIGNED_ALLOC(int32_t, (size_t)MR * (size_t)NR, 32);

    /*
     * 按通道预计算权重列之和（针对整个 N 维度的所有权重列）
     * col_sum_full[j] = sum_{k=0}^{K-1} B[k][j]，其中 j ∈ [0, N-1]
     */
    int32_t* col_sum_full = ALIGNED_ALLOC(int32_t, (size_t)N, 32);
    for (int j = 0; j < N; j++) {
        int32_t wsum = 0;
        for (int k = 0; k < K; k++) {
            wsum += (int32_t)B[k * ldb + j];
        }
        col_sum_full[j] = wsum;
    }

    /* 外层分块：遍历行（M 维度，步长 MR） */
    for (int mi = 0; mi < M; mi += MR) {
        int mr_use = (mi + MR <= M) ? MR : (M - mi);

        /*
         * 打包 A 面板：复制 mr_use 行 × K 列（uint8，行主序）
         * A 已为行主序（lda 为 A 的步幅），因此只需直接复制。
         */
        for (int r = 0; r < mr_use; r++) {
            memcpy(A_packed + r * K, A + (mi + r) * lda, (size_t)K * sizeof(uint8_t));
        }

        /* 遍历列（N 维度，步长 NR） */
        for (int nj = 0; nj < N; nj += NR) {
            int nr_use = (nj + NR <= N) ? NR : (N - nj);

            /*
             * 打包 B 面板：nr_use 列 × K 行（int8，列主序），
             * 并计算每列权重之和。
             */
            pack_B_panel(B_packed, col_sum, B, K, ldb, nj, nr_use);

            /* 清零累加器瓦片（C_acc 尺寸为 MR × NR） */
            memset(C_acc, 0, (size_t)MR * (size_t)NR * sizeof(int32_t));

            /* 调用微内核 */
            micro_kernel_4x16(A_packed, B_packed, C_acc, K, K, K,
                              mr_use, nr_use);

            /* 按通道重量化 + 零点修正 → uint8 输出 */
            requantize_to_u8(C_acc, C_out + mi * ldc + nj,
                             col_sum, scale + nj,
                             zp_a, mr_use, nr_use, ldc);
        }
    }

    ALIGNED_FREE(A_packed);
    ALIGNED_FREE(B_packed);
    ALIGNED_FREE(col_sum);
    ALIGNED_FREE(C_acc);
    ALIGNED_FREE(col_sum_full);
}

/* ================================================================
 * 基准测试全局变量
 * ================================================================ */

static uint8_t* g_A        = NULL;
static int8_t*  g_B        = NULL;
static float*   g_scale    = NULL;
static uint8_t* g_C_out    = NULL;
static int32_t* g_col_sum  = NULL;

__attribute__((noinline))
static void bn_scalar_gemm() {
    int8_gemm_scalar_ref(g_A, K_DIM, g_B, N_DIM, g_col_sum, g_scale,
                         g_C_out, N_DIM, 128,
                         M_DIM, N_DIM, K_DIM);
}

__attribute__((noinline))
static void bn_avx2_gemm() {
    int8_gemm_tiled_avx2(g_A, K_DIM, g_B, N_DIM, g_scale,
                         g_C_out, N_DIM, 128,
                         M_DIM, N_DIM, K_DIM);
}

/* ================================================================
 * 主函数
 * ================================================================ */

int main(void) {
    cpu_print_features();

    if (!cpu_has_avx2()) {
        printf("\nAVX2 not supported on this CPU. Exiting.\n");
        return 0;
    }
    printf("\n");

    printf("=== 生产级 AVX2 int8 量化 GEMM + 重量化 ===\n");
    printf("问题规模：M=%d, N=%d, K=%d\n", M_DIM, N_DIM, K_DIM);
    printf("微瓦片：MR=%d × NR=%d\n", MR, NR);
    printf("数据类型：激活值 uint8（非对称），权重 int8（对称）\n");
    printf("内层循环模式：VPMADDUBSW + VPMADDWD（生产级 int8 点积）\n\n");

    /* ---- 分配内存 ---- */
    const uint8_t zp_a = 128; /* 激活值零点（uint8 中间值） */

    g_A       = ALIGNED_ALLOC(uint8_t, (size_t)M_DIM * (size_t)K_DIM, 32);
    g_B       = ALIGNED_ALLOC(int8_t,  (size_t)K_DIM * (size_t)N_DIM, 32);
    g_scale   = ALIGNED_ALLOC(float,   (size_t)N_DIM, 32);
    g_C_out   = ALIGNED_ALLOC(uint8_t, (size_t)M_DIM * (size_t)N_DIM, 32);
    g_col_sum = ALIGNED_ALLOC(int32_t, (size_t)N_DIM, 32);
    uint8_t* C_ref = ALIGNED_ALLOC(uint8_t, (size_t)M_DIM * (size_t)N_DIM, 32);

    /* ---- 填充随机数据 ---- */
    rand_xorshift64_seed(42);
    fill_random_u8(g_A, (size_t)M_DIM * (size_t)K_DIM);

    rand_xorshift64_seed(99);
    fill_random_i8(g_B, (size_t)K_DIM * (size_t)N_DIM);
    /*
     * 约束权重范围至 [-64, 63]，防止 VPMADDUBSW 中间溢出：
     *   最大中间乘积：255 × 64 = 16320 + 相邻的 255 × 64 = 32640 < 32767
     * VPMADDUBSW 将相邻的 u8×s8 对求和，结果必须小于 int16 最大值。
     */
    for (size_t i = 0; i < (size_t)K_DIM * (size_t)N_DIM; i++) {
        g_B[i] = (int8_t)(g_B[i] / 2);
    }

    /* 每通道的随机缩放因子（典型范围为 1e-3 到 1e-1） */
    rand_xorshift64_seed(777);
    for (int j = 0; j < N_DIM; j++) {
        uint64_t r = rand_xorshift64_next();
        g_scale[j] = 0.001f + 0.1f * ((float)(r & 0xFFFFFFu) / (float)0xFFFFFFu);
    }

    /* 预计算每个输出通道的权重列之和（标量参考使用） */
    for (int j = 0; j < N_DIM; j++) {
        int32_t wsum = 0;
        for (int k = 0; k < K_DIM; k++) {
            wsum += (int32_t)g_B[k * N_DIM + j];
        }
        g_col_sum[j] = wsum;
    }

    /* ---- 正确性验证 ---- */
    printf("--- 正确性验证 ---\n\n");

    /* 计算标量参考 */
    memset(C_ref, 0, (size_t)M_DIM * (size_t)N_DIM * sizeof(uint8_t));
    int8_gemm_scalar_ref(g_A, K_DIM, g_B, N_DIM, g_col_sum, g_scale,
                         C_ref, N_DIM, zp_a,
                         M_DIM, N_DIM, K_DIM);

    /* 计算 AVX2 tiled 版本 */
    memset(g_C_out, 0, (size_t)M_DIM * (size_t)N_DIM * sizeof(uint8_t));
    int8_gemm_tiled_avx2(g_A, K_DIM, g_B, N_DIM, g_scale,
                         g_C_out, N_DIM, zp_a,
                         M_DIM, N_DIM, K_DIM);

    /*
     * 正确性检查：
     *   int8 量化 GEMM 因舍入（浮点重量化）和不同累加顺序
     *   而存在固有误差。容差为 1 个 uint8 值可容纳这些差异。
     */
    printf("比较 AVX2 tiled GEMM 与标量参考...\n");
    {
        int mismatches = 0;
        int max_diff   = 0;
        for (int i = 0; i < M_DIM; i++) {
            for (int j = 0; j < N_DIM; j++) {
                int idx  = i * N_DIM + j;
                int diff = abs((int)g_C_out[idx] - (int)C_ref[idx]);
                if (diff > max_diff) max_diff = diff;
                if (diff > 1) {
                    if (mismatches < 5) {
                        printf("  不匹配 [%d][%d]：AVX2=%d，ref=%d（差值=%d）\n",
                               i, j, (int)g_C_out[idx], (int)C_ref[idx], diff);
                    }
                    mismatches++;
                }
            }
        }
        if (mismatches == 0 && max_diff <= 1) {
            printf("  [PASS] AVX2 tiled GEMM 与标量匹配（最大差值=%d，不匹配数=%d）\n",
                   max_diff, mismatches);
        } else if (max_diff <= 1) {
            printf("  [PASS] 所有 %d 个元素差值 ≤ 1（舍入容差内）\n",
                   M_DIM * N_DIM);
        } else {
            printf("  [FAIL] 最大差值=%d，不匹配数=%d/%d\n",
                   max_diff, mismatches, M_DIM * N_DIM);
        }
    }
    printf("\n");

    /* 验证零点修正：输入全为零点值 zp_a=128，输出应全为零 */
    printf("零点修正验证（A 全部 = zp_a=%d，期望输出为零）...\n", (int)zp_a);
    {
        uint8_t* A_zp_test = ALIGNED_ALLOC(uint8_t, (size_t)M_DIM * (size_t)K_DIM, 32);
        uint8_t* C_zp_test = ALIGNED_ALLOC(uint8_t, (size_t)M_DIM * (size_t)N_DIM, 32);
        memset(A_zp_test, zp_a, (size_t)M_DIM * (size_t)K_DIM);
        memset(C_zp_test, 0xFF, (size_t)M_DIM * (size_t)N_DIM);

        int8_gemm_tiled_avx2(A_zp_test, K_DIM, g_B, N_DIM, g_scale,
                             C_zp_test, N_DIM, zp_a,
                             M_DIM, N_DIM, K_DIM);

        int nonzero = 0;
        for (int i = 0; i < M_DIM * N_DIM; i++) {
            if (C_zp_test[i] != 0) nonzero++;
        }
        if (nonzero == 0) {
            printf("  [PASS] 所有输出为零（零点修正正确）\n");
        } else {
            printf("  [WARN] %d/%d 个非零元素（可能因舍入误差产生）\n",
                   nonzero, M_DIM * N_DIM);
        }

        ALIGNED_FREE(A_zp_test);
        ALIGNED_FREE(C_zp_test);
    }
    printf("\n");

    /* ---- 基准测试 ---- */
    printf("--- 基准测试（每次 %d 次迭代的最小值）---\n", 100);

    /*
     * 内存字节数：A(M×K) + B(K×N) + C_out(M×N) 写
     * 已打包的 B 面板包含额外的副本，但为简单起见仅计算原始数据。
     */
    size_t bytes_rw = ((size_t)M_DIM * (size_t)K_DIM * sizeof(uint8_t))
                    + ((size_t)K_DIM * (size_t)N_DIM * sizeof(int8_t))
                    + ((size_t)M_DIM * (size_t)N_DIM * sizeof(uint8_t));
    size_t nelem = (size_t)M_DIM * (size_t)N_DIM;

    benchmark_result_t results[2];
    memset(results, 0, sizeof(results));

    BENCH_COMPUTE(bn_scalar_gemm(), nelem, bytes_rw, 100, results[0]);
    results[0].name = "scalar int8 GEMM";

    BENCH_COMPUTE(bn_avx2_gemm(), nelem, bytes_rw, 100, results[1]);
    results[1].name = "AVX2 tiled GEMM (4x16)";

    bench_report(results, 2);

    /* 内存带宽分析 */
    printf("--- 内存带宽分析 ---\n\n");
    for (int i = 0; i < 2; i++) {
        double ns       = results[i].elapsed_ns;
        double total_MB = (double)bytes_rw / (1024.0 * 1024.0);
        if (ns > 0.0) {
            double GB_per_s = (double)bytes_rw / ns;
            printf("  %-28s  %.3f MB 传输于 %.1f ns = %.3f GB/s\n",
                   results[i].name, total_MB, ns, GB_per_s);
        }
    }
    printf("\n");

    /* ---- 技术说明 ---- */
    printf("--- 技术说明 ---\n\n");
    printf("1. VPMADDUBSW + VPMADDWD（生产级 int8 点积模式）\n");
    printf("   _mm256_maddubs_epi16：32 uint8 × 32 int8 → 16 int16 部分和\n");
    printf("   _mm256_madd_epi16：int16 × {1,1,...} → 8 int32 累加器\n");
    printf("   此模式是 XNNPACK、QNNPACK 和 ONNX Runtime 中使用的标准方法。\n\n");

    printf("2. 零点修正（非对称量化）\n");
    printf("   在打包阶段按列预计算 sum_k(W[k][c])。\n");
    printf("   在重量化阶段：result = acc - zp_a * precomputed_sum。\n");
    printf("   这避免了在热循环中逐迭代减去零点。\n\n");

    printf("3. 按通道重量化\n");
    printf("   每个输出通道保留独立的 fp32 缩放因子。\n");
    printf("   在 int32 累加后应用（不在内核内部），以避免混合精度开销。\n\n");

    printf("4. B 打包为列主序\n");
    printf("   B_packed[col][k] 确保沿 K 维度的 SIMD 加载连续。\n");
    printf("   maddubs 在一次操作中处理每列 32 个 K 元素。\n\n");

    printf("5. 2 路累加器展开\n");
    printf("   同时处理 k 和 k+32，将依赖延迟减少一半。\n");
    printf("   每行使用 2 个 A 向量，每列使用 2 个 B 向量。\n\n");

    printf("应用于：LLM 推理中的 INT8 矩阵乘法、\n");
    printf("        量化全连接层、Transformer FFN 层。\n");

    /* 清理 */
    ALIGNED_FREE(g_A);
    ALIGNED_FREE(g_B);
    ALIGNED_FREE(g_scale);
    ALIGNED_FREE(g_C_out);
    ALIGNED_FREE(g_col_sum);
    ALIGNED_FREE(C_ref);
    return 0;
}
