#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>
#include <torch/torch.h>
#include <torch/types.h>
#include <torch/extension.h>
#include "gemm_kernels.cuh"
#include "utils.cuh"

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

using namespace nvcuda;

#define WARP_SIZE 32
#define DEVICE_INLINE __device__ inline
#define HOST_DEVICE_INLINE __device__ __host__ inline
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST32BITS(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// gmem -> smem
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))

// ca(cache all, L1 + L2): support 4, 8, 16 bytes, cg(cache global, L2): only support 16 bytes.
#define CP_ASYNC_CA(dst, src, bytes) asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes) asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))

// smem -> gmem: requires sm_90 or higher.
#define CP_ASYNC_BULK_COMMIT_GROUP() asm volatile("cp.async.bulk.commit_group;\n" ::)
#define CP_ASYNC_BULK_WAIT_ALL() asm volatile("cp.async.bulk.wait_all;\n" ::)
#define CP_ASYNC_BULK_WAIT_GROUP(n) asm volatile("cp.async.bulk.wait_group %0;\n" ::"n"(n))
#define CP_ASYNC_BULK(dst, src, bytes) asm volatile("cp.async.bulk.global.shared::cta.bulk_group.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))

// ldmatrix
#define LDMATRIX_X1(R, addr) asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n" : "=r"(R) : "r"(addr))
#define LDMATRIX_X2(R0, R1, addr) asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))
#define LDMATRIX_X4(R0, R1, R2, R3, addr) asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "r"(addr))
#define LDMATRIX_X1_T(R, addr) asm volatile("ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];\n" : "=r"(R) : "r"(addr))
#define LDMATRIX_X2_T(R0, R1, addr) asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))
#define LDMATRIX_X4_T(R0, R1, R2, R3, addr) asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "r"(addr))

// stmatrix: requires sm_90 or higher.
#define STMATRIX_X1(addr, R) asm volatile("stmatrix.sync.aligned.x1.m8n8.shared.b16 [%0], {%1};\n" ::"r"(addr), "r"(R))
#define STMATRIX_X2(addr, R0, R1) asm volatile("stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};\n" ::"r"(addr), "r"(R0), "r"(R1))
#define STMATRIX_X4(addr, R0, R1, R2, R3) asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(addr), "r"(R0), "r"(R1), "r"(R2), "r"(R3))
#define STMATRIX_X1_T(addr, R) asm volatile("stmatrix.sync.aligned.x1.trans.m8n8.shared.b16 [%0], {%1};\n" ::"r"(addr), "r"(R))
#define STMATRIX_X2_T(addr, R0, R1) asm volatile("stmatrix.sync.aligned.x2.trans.m8n8.shared.b16 [%0], {%1, %2};\n" ::"r"(addr), "r"(R0), "r"(R1))
#define STMATRIX_X4_T(addr, R0, R1, R2, R3) asm volatile("stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(addr), "r"(R0), "r"(R1), "r"(R2), "r"(R3))
// mma m16n8k16
#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1) asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" : "=r"(RD0), "=r"(RD1) : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1))

HOST_DEVICE_INLINE
int div_ceil(int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

template <const int MMA_M = 16, const int MMA_N = 8, const int MMA_K = 16>
__global__ void hgemm_mma_m16n8k16_naive_kernel(half *A, half *B, half *C,
                                                int M, int N, int K) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int NUM_K_TILES = div_ceil(K, MMA_K);

    constexpr int BM = MMA_M;

    // A 块的高度
    constexpr int BM = MMA_M; // 16
    // B 块的宽度
    constexpr int BN = MMA_N; // 8
    // A 块的宽度，B 块的高度
    constexpr int BK = MMA_K; // 16

    // A 块 16 * 16
    __shared__ half s_a[MMA_M][MMA_K];
    // B 块 16 * 8
    __shared__ half s_b[MMA_K][MMA_N];

    // C 块 16 * 8
    __shared__ half s_c[MMA_M][MMA_N]; // 16x8

    // 块内线程索引
    const int tid = threadIdx.y * blockDim.x + threadIdx.x; // within block
    // warp 内 id
    const int lane_id = tid % WARP_SIZE; // 0~31

    // 表示行数，0-15
    const int load_smem_a_m = tid / 2; // row 0~15
    // 表示列数，0，8
    const int load_smem_a_k = (tid % 2) * 8; // col 0,8

    // 只使用前 15 个
    const int load_smem_b_k = tid; // row 0~31, but only use 0~15
    // 列数只能是 0
    const int load_smem_b_n = 0; // col 0

    // A 块矩阵在全局内存中的行位置
    const int load_gmem_a_m = by * BM + load_smem_a_m; // global m
    // B 块矩阵在全局内存中的列位置
    const int load_gmem_b_n = bx * BN + load_smem_b_n; // global n

    if (load_gmem_a_m >= M && load_gmem_b_n >= N) return;

    // 存放结果
    // 每个线程都有 2 个 uint32_t
    uint32_t RC[2] = {0, 0};

#pragma unroll
    for (int k = 0; k < NUM_K_TILES; ++k) {
        // gmem_a -> smem_a，全局内存到共享内存
        int load_gmem_a_k = k * BK + load_smem_a_k; // global col of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        // 加载到共享内存中，行主序,一次读8个
        LDST128BITS(s_a[load_smem_a_m][load_smem_a_k]) = (LDST128BITS(A[load_gmem_a_addr]));

        // gmem_b -> smem_b
        // 将B块读到共享内存中，行主序
        if (lane_id < MMA_K) {
            int load_gmem_b_k = k * MMA_K + load_smem_b_k; // global row of b
            int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
            LDST128BITS(s_b[load_smem_b_k][load_smem_b_n]) = (LDST128BITS(B[load_gmem_b_addr]));
        }
        __syncthreads();

        // 每个线程所拥有的寄存器，每个线程是一个 uint32_t,也就是两个half,由于 A 是 x4,所以共4个寄存器
        uint32_t RA[4]; // 寄存器数组，存储从s_a加载的4个8x8矩阵片段（共16x16矩阵）
        // 2个寄存器
        uint32_t RB[2]; // 寄存器数组，存储从s_b加载的2个8x8转置矩阵片段（共16x8矩阵）

        uint32_t load_smem_a_ptr = __cvta_generic_to_shared(
            &s_a[lane_id % 16][(lane_id / 16) * 8]);
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], load_smem_a_ptr);

        uint32_t load_smem_b_ptr = __cvta_generic_to_shared(
            &s_b[lane_id % 16][0]);

        LDMATRIX_X2_T(RB[0], RB[1], load_smem_b_ptr);

        HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);

        // 同步所有线程，确保所有线程都完成了当前K tile的计算
        // __syncthreads()：块内线程同步，在加载下一个K tile的数据之前需要这个同步
        __syncthreads();
    }

    LDST32BITS(s_c[lane_id / 4][(lane_id % 4) * 2]) = LDST32BITS(RC[0]);
    LDST32BITS(s_c[lane_id / 4 + 8][(lane_id % 4) * 2]) = LDST32BITS(RC[1]);

    __syncthreads();

    // store s_c[16][8]
    // 共享内存写回全局内存
    if (lane_id < MMA_M) {
        // store 128 bits per memory issue.
        int store_gmem_c_m = by * BM + lane_id;
        int store_gmem_c_n = bx * BN;
        int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
        LDST128BITS(C[store_gmem_c_addr]) = (LDST128BITS(s_c[lane_id][0]));
    }
}

template <const int MMA_M = 16,
          const int MMA_N = 8,
          const int MMA_K = 16,
          const int MMA_TILE_M = 2,
          const int MMA_TILE_N = 4,
          const int WARP_TILE_M = 4,
          const int WARP_TILE_N = 4,
          const int A_PAD = 0,
          const int B_PAD = 0>
__global__ void __launch_bounds__(256)
    hgemm_mma_m16n8k16_mma2x4_warp4x4_kernel(
        half *A, half *B, half *C, int M, int N, int K) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int NUM_K_TILES = div_ceil(K, MMA_K);

    constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M;
    constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N;
    constexpr int BK = MMA_K;

    __shared__ half s_a[BM][BK + A_PAD];
    __shared__ half s_b[BK][BN + B_PAD];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x; // within block
    const int warp_id = tid / WARP_SIZE;                    // 0~7 warp_id within block
    const int lane_id = tid % WARP_SIZE;                    // 0~31
    const int warp_m = warp_id % 2;                         // 0,1
    const int warp_n = warp_id / 2;                         // 0,1,2,3

    int load_smem_a_m = tid / 2;                // row 0~127
    int load_smem_a_k = (tid % 2 == 0) ? 0 : 8; // col 0,8

    int load_smem_b_k = tid / 16;       // row 0~15
    int load_smem_b_n = (tid % 16) * 8; // col 0,8,...,120

    int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
    int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
    if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

    uint32_t RC[WARP_TILE_M][WARP_TILE_N][2];
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            RC[i][j][0] = 0;
            RC[i][j][1] = 0;
        }
    }

#pragma unroll
    for (int k = 0; k < NUM_K_TILES; ++k) {
        // gmem -> smem
        int load_gmem_a_k = k * BK + load_smem_a_k; // global col of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = k * BK + load_smem_b_k; // global row of b
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        LDST128BITS(s_b[load_smem_b_k][load_smem_b_n]) = (LDST128BITS(B[load_gmem_b_addr]));
        LDST128BITS(s_a[load_smem_a_m][load_smem_a_k]) = (LDST128BITS(A[load_gmem_a_addr]));
        __syncthreads();

        // ldmatrix for s_a, ldmatrix.trans for s_b.
        uint32_t RA[WARP_TILE_M][4];
        uint32_t RB[WARP_TILE_N][2];

// smem -> reg
#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
            int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
            int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
            int lane_smem_a_k = (lane_id / 16) * 8;           // 0,8
            uint32_t lane_smem_a_ptr = __cvta_generic_to_shared(
                &s_a[lane_smem_a_m][lane_smem_a_k]);
            LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], lane_smem_a_ptr);
        }

#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
            int lane_smem_b_k = lane_id % 16;  // 0~15
            int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
            uint32_t lane_smem_b_ptr = __cvta_generic_to_shared(
                &s_b[lane_smem_b_k][lane_smem_b_n]);
            LDMATRIX_X2_T(RB[j][0], RB[j][1], lane_smem_b_ptr);
        }

// MMA compute
#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j) {
                HMMA16816(RC[i][j][0], RC[i][j][1],
                          RA[i][0], RA[i][1], RA[i][2], RA[i][3],
                          RB[j][0], RB[j][1],
                          RC[i][j][0], RC[i][j][1]);
            }
        }
        __syncthreads();
    }

// reg -> gmem, MMA_MxMMA_N=16x8
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            int store_warp_smem_c_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
            int store_warp_smem_c_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
            // mapping lane smem index -> global index.
            // [16][8], https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
            // #matrix-fragments-for-mma-m16n8k16-with-floating-point-type
            // [0~7][0~3 u32 -> 0~7 f16], [8~15][0~3 u32 -> 0~7 f16]
            int store_lane_gmem_c_m = by * BM + store_warp_smem_c_m + lane_id / 4;
            int store_lane_gmem_c_n = bx * BN + store_warp_smem_c_n + (lane_id % 4) * 2;
            int store_gmem_c_addr_0 = store_lane_gmem_c_m * N + store_lane_gmem_c_n;
            int store_gmem_c_addr_1 = (store_lane_gmem_c_m + 8) * N + store_lane_gmem_c_n;
            // TODO: how to use LDST128BITS here ? reverse the loop order ?
            LDST32BITS(C[store_gmem_c_addr_0]) = LDST32BITS(RC[i][j][0]);
            LDST32BITS(C[store_gmem_c_addr_1]) = LDST32BITS(RC[i][j][1]);
        }
    }
}

// 128x128, mma2x4, warp4x4(64,32,16), stages, block swizzle
template <const int MMA_M = 16,
          const int MMA_N = 8,
          const int MMA_K = 16,
          const int MMA_TILE_M = 2,
          const int MMA_TILE_N = 4,
          const int WARP_TILE_M = 4,
          const int WARP_TILE_N = 4,
          const int A_PAD = 0,
          const int B_PAD = 0,
          const int K_STAGE = 2,
          const bool BLOCK_SWIZZLE = true>
__global__ void __launch_bounds__(256)
    hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_kernel(
        half *A, half *B, half *C, int M, int N, int K) {
    // BLOCK_SWIZZLE 0/1 control use block swizzle or not.
    //
    const int bx = ((int)BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
    const int by = blockIdx.y;
    const int NUM_K_TILES = div_ceil(K, MMA_K);

    // 2. 这里的 BM, BN 是 Thread Block 处理的矩阵块大小 (128x128)
    constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M; // 16*2*4=128
    constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N; // 8*4*4=128
    constexpr int BK = MMA_K;                            // 16

    // 3. 定义共享内存 (Shared Memory)，使用 K_STAGE 实现多级缓冲（默认为 2，即 Double Buffering）
    // 存储矩阵 A 的分块
    __shared__ half s_a[K_STAGE][BM][BK + A_PAD]; // 128*16*2=4KB
    // 存储矩阵 B 的分块
    __shared__ half s_b[K_STAGE][BK][BN + B_PAD]; // 16*128*2=4KB, 16*(128+16)*2=4.5KB

    constexpr int s_a_stage_offset = BM * (BK + A_PAD);
    constexpr int s_b_stage_offset = BK * (BN + B_PAD);

    // 计算线程在 Warp 中的身份
    const int tid = threadIdx.y * blockDim.x + threadIdx.x; // within block
    const int warp_id = tid / WARP_SIZE;                    // 0~7 warp_id within block
    const int lane_id = tid % WARP_SIZE;                    // 0~31

    const int warp_m = warp_id % 2; // 0,1
    const int warp_n = warp_id / 2; // 0,1,2,3

    // 确定从全局内存加载数据到共享内存时的坐标
    // 每个线程负责 A 的一行中的一部分
    int load_smem_a_m = tid / 2;                // row 0~127
    int load_smem_a_k = (tid % 2 == 0) ? 0 : 8; // col 0,8
    // 每个线程负责 B 的一行
    int load_smem_b_k = tid / 16;       // row 0~15
    int load_smem_b_n = (tid % 16) * 8; // col 0,8,...,120

    int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
    int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
    if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

    // 初始化累加寄存器(C),用于存放 MMA 的计算结果
    uint32_t RC[WARP_TILE_M][WARP_TILE_N][2];
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            RC[i][j][0] = 0;
            RC[i][j][1] = 0;
        }
    }

    uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
    uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

// 进入主循环前，先异步加载前(K_STAGE - 1) 个阶段的数据
#pragma unroll
    for (int k = 0; k < (K_STAGE - 1); ++k) { // 0, 1
        // k * WMMA_K, WMMA_K=16 -> (k << 4)
        // 计算全局内存地址并使用 CP_ASYNC 直接从 GMEM 拷贝到 SMEM，绕过寄存器
        int load_gmem_a_k = k * BK + load_smem_a_k; // global col of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = k * BK + load_smem_b_k; // global row of b
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

        uint32_t load_smem_a_ptr = (smem_a_base_ptr + (k * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) * sizeof(half));
        CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

        uint32_t load_smem_b_ptr = (smem_b_base_ptr + (k * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + load_smem_b_n) * sizeof(half));
        CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);

        CP_ASYNC_COMMIT_GROUP();
    }

    CP_ASYNC_WAIT_GROUP(K_STAGE - 2); // s2->0, s3->1, s4->2
    __syncthreads();

    // 主计算循环
#pragma unroll
    for (int k = (K_STAGE - 1); k < NUM_K_TILES; ++k) {
        // gmem -> smem
        // s2/4 can use bitwise ops but s3 can not, so, we use mod
        // ops for all stages kernel. s2: (k + 1)&1, s4: (k + 1)&3
        // s3: (k + 1) % 3
        // 流水线步骤1：发起下一轮数据的异步拷贝
        int smem_sel = (k + 1) % K_STAGE; // s3 k 2->0, k 3->1, k 4->2...
        int smem_sel_next = k % K_STAGE;  // s3 k 2->2, k 3->0, k 4->1...

        int load_gmem_a_k = k * BK + load_smem_a_k; // global col of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = k * BK + load_smem_b_k; // global row of b
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

        uint32_t load_smem_a_ptr = (smem_a_base_ptr + (smem_sel_next * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) * sizeof(half));
        CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

        uint32_t load_smem_b_ptr = (smem_b_base_ptr + (smem_sel_next * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + load_smem_b_n) * sizeof(half));
        CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
        CP_ASYNC_COMMIT_GROUP();

        // ldmatrix for s_a, ldmatrix.trans for s_b.
        uint32_t RA[WARP_TILE_M][4];
        uint32_t RB[WARP_TILE_N][2];

// smem -> reg
// 【流水线步骤 2】：从共享内存通过 LDMATRIX 指令加载数据到寄存器 (RA, RB)
// LDMATRIX 指令利用了 Warp 协作，可以高效加载 Tensor Core 所需的矩阵格式
#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
            int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
            int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
            int lane_smem_a_k = (lane_id / 16) * 8;           // 0,8
            uint32_t lane_smem_a_ptr = __cvta_generic_to_shared(
                &s_a[smem_sel][lane_smem_a_m][lane_smem_a_k]);
            LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], lane_smem_a_ptr);
        }

#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
            int lane_smem_b_k = lane_id % 16;  // 0~15
            int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
            uint32_t lane_smem_b_ptr = __cvta_generic_to_shared(
                &s_b[smem_sel][lane_smem_b_k][lane_smem_b_n]);
            LDMATRIX_X2_T(RB[j][0], RB[j][1], lane_smem_b_ptr);
        }

// MMA compute
// 【流水线步骤 3】：Tensor Core 矩阵乘加计算 (MMA)
// 执行 D = A * B + C
#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j) {
                HMMA16816(RC[i][j][0], RC[i][j][1],
                          RA[i][0], RA[i][1], RA[i][2], RA[i][3],
                          RB[j][0], RB[j][1],
                          RC[i][j][0], RC[i][j][1]);
            }
        }

        CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
        __syncthreads();
    }
    // make sure all memory issues ready.
    if ((K_STAGE - 2) > 0) {
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
    }

    // processing last (K_STAGE-1) k iters.
    {
#pragma unroll
        for (int k = 0; k < (K_STAGE - 1); k++) {
            int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
            // ldmatrix for s_a, ldmatrix.trans for s_b.
            uint32_t RA[WARP_TILE_M][4];
            uint32_t RB[WARP_TILE_N][2];

// smem -> reg
#pragma unroll
            for (int i = 0; i < WARP_TILE_M; ++i) {
                int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
                int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
                int lane_smem_a_k = (lane_id / 16) * 8;           // 0,8
                uint32_t lane_smem_a_ptr = __cvta_generic_to_shared(
                    &s_a[stage_sel][lane_smem_a_m][lane_smem_a_k]);
                LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], lane_smem_a_ptr);
            }

#pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j) {
                int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
                int lane_smem_b_k = lane_id % 16;  // 0~15
                int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
                uint32_t lane_smem_b_ptr = __cvta_generic_to_shared(
                    &s_b[stage_sel][lane_smem_b_k][lane_smem_b_n]);
                LDMATRIX_X2_T(RB[j][0], RB[j][1], lane_smem_b_ptr);
            }

// MMA compute
#pragma unroll
            for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
                for (int j = 0; j < WARP_TILE_N; ++j) {
                    HMMA16816(RC[i][j][0], RC[i][j][1],
                              RA[i][0], RA[i][1], RA[i][2], RA[i][3],
                              RB[j][0], RB[j][1],
                              RC[i][j][0], RC[i][j][1]);
                }
            }
        }
    }

    // reg -> gmem, MMA_MxMMA_N=16x8
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            int store_warp_smem_c_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
            int store_warp_smem_c_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
            // mapping lane smem index -> global index.
            // [16][8], https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
            // #matrix-fragments-for-mma-m16n8k16-with-floating-point-type
            // [0~7][0~3 u32 -> 0~7 f16], [8~15][0~3 u32 -> 0~7 f16]
            int store_lane_gmem_c_m = by * BM + store_warp_smem_c_m + lane_id / 4;
            int store_lane_gmem_c_n = bx * BN + store_warp_smem_c_n + (lane_id % 4) * 2;
            int store_gmem_c_addr_0 = store_lane_gmem_c_m * N + store_lane_gmem_c_n;
            int store_gmem_c_addr_1 = (store_lane_gmem_c_m + 8) * N + store_lane_gmem_c_n;
            LDST32BITS(C[store_gmem_c_addr_0]) = LDST32BITS(RC[i][j][0]);
            LDST32BITS(C[store_gmem_c_addr_1]) = LDST32BITS(RC[i][j][1]);
        }
    }
}

// 128x128, mma2x4, warp4x4(64,32,16), stages, block swizzle, dsmem
template <const int MMA_M = 16,
          const int MMA_N = 8,
          const int MMA_K = 16,
          const int MMA_TILE_M = 2,
          const int MMA_TILE_N = 4,
          const int WARP_TILE_M = 4,
          const int WARP_TILE_N = 4,
          const int A_PAD = 0,
          const int B_PAD = 0,
          const int K_STAGE = 2,
          const bool BLOCK_SWIZZLE = true,
          const bool COLLECTIVE_STORE = false>
__global__ void __launch_bounds__(256)
    hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_kernel(
        half *A, half *B, half *C, int M, int N, int K) {
    // BLOCK_SWIZZLE 0/1 control use block swizzle or not.
    // COLLECTIVE_STORE true/false control use stmatrix or not.
    const int bx = ((int)BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
    const int by = blockIdx.y;
    const int NUM_K_TILES = div_ceil(K, MMA_K);
    constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M; // 16*2*4=128
    constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N; // 8*4*4=128
    constexpr int BK = MMA_K;                            // 16

    extern __shared__ half smem[];
    half *s_a = smem;
    half *s_b = smem + K_STAGE * BM * (BK + A_PAD);
    constexpr int s_a_stage_offset = BM * (BK + A_PAD);
    constexpr int s_b_stage_offset = BK * (BN + B_PAD);

    const int tid = threadIdx.y * blockDim.x + threadIdx.x; // within block
    const int warp_id = tid / WARP_SIZE;                    // 0~7 warp_id within block
    const int lane_id = tid % WARP_SIZE;                    // 0~31
    const int warp_m = warp_id % 2;                         // 0,1
    const int warp_n = warp_id / 2;                         // 0,1,2,3

    int load_smem_a_m = tid / 2;                 // row 0~127
    int load_smem_a_k = (tid % 2 == 0) ? 0 : 8;  // col 0,8
    int load_smem_b_k = tid / 16;                // row 0~15
    int load_smem_b_n = (tid % 16) * 8;          // col 0,8,...,120
    int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
    int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
    if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

    uint32_t RC[WARP_TILE_M][WARP_TILE_N][2];
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            RC[i][j][0] = 0;
            RC[i][j][1] = 0;
        }
    }

    uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
    uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

#pragma unroll
    for (int k = 0; k < (K_STAGE - 1); ++k) { // 0, 1
        // k * WMMA_K, WMMA_K=16 -> (k << 4)
        int load_gmem_a_k = k * BK + load_smem_a_k; // global col of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = k * BK + load_smem_b_k; // global row of b
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

        uint32_t load_smem_a_ptr = (smem_a_base_ptr + (k * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) * sizeof(half));
        CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

        uint32_t load_smem_b_ptr = (smem_b_base_ptr + (k * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + load_smem_b_n) * sizeof(half));
        CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);

        CP_ASYNC_COMMIT_GROUP();
    }

    CP_ASYNC_WAIT_GROUP(K_STAGE - 2); // s2->0, s3->1, s4->2
    __syncthreads();

#pragma unroll
    for (int k = (K_STAGE - 1); k < NUM_K_TILES; ++k) {
        // gmem -> smem
        // s2/4 can use bitwise ops but s3 can not, so, we use mod
        // ops for all stages kernel. s2: (k + 1)&1, s4: (k + 1)&3
        // s3: (k + 1) % 3
        int smem_sel = (k + 1) % K_STAGE; // s3 k 2->0, k 3->1, k 4->2...
        int smem_sel_next = k % K_STAGE;  // s3 k 2->2, k 3->0, k 4->1...

        int load_gmem_a_k = k * BK + load_smem_a_k; // global col of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = k * BK + load_smem_b_k; // global row of b
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

        uint32_t load_smem_a_ptr = (smem_a_base_ptr + (smem_sel_next * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) * sizeof(half));
        CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

        uint32_t load_smem_b_ptr = (smem_b_base_ptr + (smem_sel_next * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + load_smem_b_n) * sizeof(half));
        CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
        CP_ASYNC_COMMIT_GROUP();

        uint32_t RA[WARP_TILE_M][4];
        uint32_t RB[WARP_TILE_N][2];
// ldmatrix for s_a, ldmatrix.trans for s_b.
// smem -> reg
#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
            int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
            int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
            int lane_smem_a_k = (lane_id / 16) * 8;           // 0,8
            uint32_t lane_smem_a_ptr = (smem_a_base_ptr + (smem_sel * s_a_stage_offset + lane_smem_a_m * (BK + A_PAD) + lane_smem_a_k) * sizeof(half));
            LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], lane_smem_a_ptr);
        }

#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
            int lane_smem_b_k = lane_id % 16;  // 0~15
            int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
            uint32_t lane_smem_b_ptr = (smem_b_base_ptr + (smem_sel * s_b_stage_offset + lane_smem_b_k * (BN + B_PAD) + lane_smem_b_n) * sizeof(half));
            LDMATRIX_X2_T(RB[j][0], RB[j][1], lane_smem_b_ptr);
        }

// MMA compute
#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j) {
                HMMA16816(RC[i][j][0], RC[i][j][1],
                          RA[i][0], RA[i][1], RA[i][2], RA[i][3],
                          RB[j][0], RB[j][1],
                          RC[i][j][0], RC[i][j][1]);
            }
        }

        CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
        __syncthreads();
    }

    // make sure all memory issues ready.
    if ((K_STAGE - 2) > 0) {
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
    }

    // processing last (K_STAGE-1) k iters.
    {
#pragma unroll
        for (int k = 0; k < (K_STAGE - 1); k++) {
            uint32_t RA[WARP_TILE_M][4];
            uint32_t RB[WARP_TILE_N][2];

            int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
// ldmatrix for s_a, ldmatrix.trans for s_b.
// smem -> reg
#pragma unroll
            for (int i = 0; i < WARP_TILE_M; ++i) {
                int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
                int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
                int lane_smem_a_k = (lane_id / 16) * 8;           // 0,8
                uint32_t lane_smem_a_ptr = (smem_a_base_ptr + (stage_sel * s_a_stage_offset + lane_smem_a_m * (BK + A_PAD) + lane_smem_a_k) * sizeof(half));
                LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], lane_smem_a_ptr);
            }

#pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j) {
                int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
                int lane_smem_b_k = lane_id % 16;  // 0~15
                int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
                uint32_t lane_smem_b_ptr = (smem_b_base_ptr + (stage_sel * s_b_stage_offset + lane_smem_b_k * (BN + B_PAD) + lane_smem_b_n) * sizeof(half));
                LDMATRIX_X2_T(RB[j][0], RB[j][1], lane_smem_b_ptr);
            }

// MMA compute
#pragma unroll
            for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
                for (int j = 0; j < WARP_TILE_N; ++j) {
                    HMMA16816(RC[i][j][0], RC[i][j][1],
                              RA[i][0], RA[i][1], RA[i][2], RA[i][3],
                              RB[j][0], RB[j][1],
                              RC[i][j][0], RC[i][j][1]);
                }
            }
        }
    }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 90)
    if (COLLECTIVE_STORE) {
        // The following code has not been tested because I do not have a GPU with sm>=90
        // reg -> smem(stmatrix) -> gmem(cp.async.bulk), MMA_MxMMA_N=16x8
        // NOTE: need [MMA_M][MMA_N] per warp to avoid overlap between warps.
        __shared__ half s_c[MMA_TILE_M][MMA_TILE_N][MMA_M][MMA_N]; // (2*4)*16*8*2=2KB
        uint32_t smem_c_base_ptr = __cvta_generic_to_shared(&s_c[warp_m][warp_n]);
#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j) {
                // store (i,j) warp tile -> smem c, 16x8
                uint32_t lane_smem_c_ptr = (smem_c_base_ptr + (lane_id % 16) * MMA_N * sizeof(half)); // (0~15)*8
                STMATRIX_X2(lane_smem_c_ptr, RC[i][j][0], RC[i][j][1]);
                // smem -> gmem, may use cp.async.bulk.global.share::cta?
                int store_warp_gmem_c_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
                int store_warp_gmem_c_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
                int store_lane_gmem_c_m = by * BM + store_warp_gmem_c_m;
                int store_lane_gmem_c_n = bx * BN + store_warp_gmem_c_n;
                // send 16 memory issues with 128 bits within lower half lanes.
                // TODO: use cp.async.bulk and wait outside the inner loop.
                if (lane_id < 16) {
                    int store_gmem_c_addr = (store_lane_gmem_c_m + lane_id) * N + store_lane_gmem_c_n;
                    LDST128BITS(C[store_gmem_c_addr]) = LDST128BITS(
                        s_c[warp_m][warp_n][lane_id][0]);
                }
                __syncwarp();
            }
        }
    } else {
// reg -> gmem, MMA_MxMMA_N=16x8
#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j) {
                int store_warp_smem_c_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
                int store_warp_smem_c_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
                int store_lane_gmem_c_m = by * BM + store_warp_smem_c_m + lane_id / 4;
                int store_lane_gmem_c_n = bx * BN + store_warp_smem_c_n + (lane_id % 4) * 2;
                int store_gmem_c_addr_0 = store_lane_gmem_c_m * N + store_lane_gmem_c_n;
                int store_gmem_c_addr_1 = (store_lane_gmem_c_m + 8) * N + store_lane_gmem_c_n;
                LDST32BITS(C[store_gmem_c_addr_0]) = LDST32BITS(RC[i][j][0]);
                LDST32BITS(C[store_gmem_c_addr_1]) = LDST32BITS(RC[i][j][1]);
            }
        }
    }
#else
    // #warning "stmatrix need sm>=90, force use __shfl_sync for collective store!"
    {
        for (int i = 0; i < WARP_TILE_M; ++i) {
            // How to use LDST128BITS here? __shfl_sync -> lane 0 -> store 8 half.
            // thus, we only need 8 memory issues with 128 bits after shfl_sync.
            // may reuse RA[4][4] as RC0 ? only new RC1[4][4].
            uint32_t RC0[WARP_TILE_N][4];
            uint32_t RC1[WARP_TILE_N][4];
#pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j) {
                // How to use LDST128BITS here? __shfl_sync -> lane 0 -> store 8 half.
                // thus, we only need 8 memory issues with 128 bits after shfl_sync.
                RC0[j][0] = RC[i][j][0];
                RC1[j][0] = RC[i][j][1];
                RC0[j][1] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 1);
                RC0[j][2] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 2);
                RC0[j][3] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 3);
                RC1[j][1] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 1);
                RC1[j][2] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 2);
                RC1[j][3] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 3);
            }

            if (lane_id % 4 == 0) {
                int store_warp_smem_c_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
                int store_lane_gmem_c_m = by * BM + store_warp_smem_c_m + lane_id / 4;
#pragma unroll
                for (int j = 0; j < WARP_TILE_N; ++j) {
                    int store_warp_smem_c_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
                    int store_lane_gmem_c_n = bx * BN + store_warp_smem_c_n;
                    int store_gmem_c_addr_0 = store_lane_gmem_c_m * N + store_lane_gmem_c_n;
                    int store_gmem_c_addr_1 = (store_lane_gmem_c_m + 8) * N + store_lane_gmem_c_n;
                    LDST128BITS(C[store_gmem_c_addr_0]) = LDST128BITS(RC0[j][0]);
                    LDST128BITS(C[store_gmem_c_addr_1]) = LDST128BITS(RC1[j][0]);
                }
            }
        }
    }
#endif
}

// In order to reduce bank conflicts, we will save the K(16x2=32) 
// dimension by half according to the stage dimension. For example, 
// stages=3, warp_tile_k=2, it will be saved as [3*2][BM][16].
// 128x128, mma2x4, warp4x4(64,32,32), stages, block swizzle, dsmem, 
// k32 with reg double buffers
template<const int MMA_M=16, 
         const int MMA_N=8, 
         const int MMA_K=16,
         const int MMA_TILE_M=2,
         const int MMA_TILE_N=4,
         const int WARP_TILE_M=4,
         const int WARP_TILE_N=4,
         const int WARP_TILE_K=2,
         const int A_PAD=0, 
         const int B_PAD=0,
         const int K_STAGE=2, 
         const bool BLOCK_SWIZZLE=true,
         const bool WARP_SWIZZLE=true>
__global__ void  __launch_bounds__(256) 
hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_kernel(
  const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C, 
  int M, int N, int K) {
  // BLOCK_SWIZZLE 0/1 control use block swizzle or not.
  const int bx = ((int) BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, MMA_K * WARP_TILE_K);
  constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M; // 16*2*4=128
  constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N; // 8*4*4=128
  constexpr int BK = MMA_K; // 16x2=32

  extern __shared__ half smem[]; 
  half* s_a = smem;
  half* s_b = smem + K_STAGE * BM * (BK + A_PAD) * WARP_TILE_K;
  constexpr int s_a_stage_offset = BM * (BK + A_PAD); // 128x16 
  constexpr int s_b_stage_offset = BK * (BN + B_PAD); // 16x128
  constexpr int s_a_mma_k_store_offset = K_STAGE * BM * (BK + A_PAD);
  constexpr int s_b_mma_k_store_offset = K_STAGE * BK * (BN + B_PAD);

  const int tid = threadIdx.y * blockDim.x + threadIdx.x; // within block
  const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
  const int lane_id = tid % WARP_SIZE; // 0~31
  const int warp_m = warp_id % 2; // 0,1
  const int warp_n = warp_id / 2; // 0,1,2,3

  int load_smem_a_m = tid / 2; // row 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 8; // col 0,8
  int load_smem_b_k = tid / 16; // row 0~15
  int load_smem_b_n = (tid % 16) * 8; // col 0,8,16,...
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
  if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

  uint32_t RC[WARP_TILE_M][WARP_TILE_N][2];
  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      RC[i][j][0] = 0;
      RC[i][j][1] = 0;
    }
  }

  uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
  uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

  #pragma unroll
  for (int k = 0; k < (K_STAGE - 1); ++k) { // 0, 1
    // k * WMMA_K, WMMA_K=16 -> (k << 4)
    int load_gmem_a_k = k * BK * WARP_TILE_K + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * BK * WARP_TILE_K + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 

    uint32_t load_smem_a_ptr = (
      smem_a_base_ptr + (k * s_a_stage_offset + 
                         load_smem_a_m * (BK + A_PAD) + 
                         load_smem_a_k) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16); // MMA_K 0
    uint32_t load_smem_a_mma_k_ptr = (
      smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(half) + 
      (k * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + 
      load_smem_a_k) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_a_mma_k_ptr, &A[load_gmem_a_addr + 16], 16); // MMA_K 1

    uint32_t load_smem_b_ptr = (
      smem_b_base_ptr + (k * s_b_stage_offset + 
                         load_smem_b_k * (BN + B_PAD) + 
                         load_smem_b_n) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
    
    int load_gmem_b_k_mma_k = k * BK * WARP_TILE_K + MMA_K + load_smem_b_k;
    int load_gmem_b_addr_mma_k = load_gmem_b_k_mma_k * N + load_gmem_b_n; 
    uint32_t load_smem_b_mma_k_ptr = (
      smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) + 
      (k * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + 
      load_smem_b_n) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_b_mma_k_ptr, &B[load_gmem_b_addr_mma_k], 16);

    CP_ASYNC_COMMIT_GROUP();
  }

  CP_ASYNC_WAIT_GROUP(K_STAGE-2); // s2->0, s3->1, s4->2
  __syncthreads(); 

  uint32_t RA[2][WARP_TILE_M][4];
  uint32_t RB[2][WARP_TILE_N][2];

  int reg_store_idx = 0;
  int reg_load_idx = 1;

  { 
    // ldmatrix for s_a, ldmatrix.trans for s_b.
    // smem -> reg buffers 0, first MMA_K, 0~15
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
      int lane_smem_a_k = (lane_id / 16) * 8; // 0,8
      uint32_t lane_smem_a_ptr = (
        smem_a_base_ptr + 
        (0 * s_a_stage_offset + lane_smem_a_m * (BK + A_PAD) + 
        lane_smem_a_k) * sizeof(half)
      );
      LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], 
                  RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], 
                  lane_smem_a_ptr);
    }

    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
      int lane_smem_b_k = lane_id % 16;  // 0~15, 0~15
      int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
      uint32_t lane_smem_b_ptr = (
        smem_b_base_ptr + 
        (0 * s_b_stage_offset + lane_smem_b_k * (BN + B_PAD) + 
        lane_smem_b_n) * sizeof(half)
      );
      // may use .x4.trans to load 4 matrix for reg double buffers at once?
      LDMATRIX_X2_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], 
                    lane_smem_b_ptr);
    }
  }
  
  #pragma unroll
  for (int k = (K_STAGE - 1); k < NUM_K_TILES; ++k) {
    reg_store_idx ^= 1; // 0->1
    reg_load_idx ^= 1; // 1->0
    int smem_sel = (k + 1) % K_STAGE; // s3 k 2->0, k 3->1, k 4->2...
    int smem_sel_next = k % K_STAGE;  // s3 k 2->2, k 3->0, k 4->1...

    // stage gmem -> smem
    int load_gmem_a_k = k * BK * WARP_TILE_K + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * BK * WARP_TILE_K + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 

    uint32_t load_smem_a_ptr = (
      smem_a_base_ptr + (smem_sel_next * s_a_stage_offset + 
                         load_smem_a_m * (BK + A_PAD) + 
                         load_smem_a_k) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16); // MMA_K 0
    uint32_t load_smem_a_mma_k_ptr = (
      smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(half) + 
      (smem_sel_next * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + 
      load_smem_a_k) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_a_mma_k_ptr, &A[load_gmem_a_addr + 16], 16); // MMA_K 1

    uint32_t load_smem_b_ptr = (
      smem_b_base_ptr + (smem_sel_next * s_b_stage_offset + 
                         load_smem_b_k * (BN + B_PAD) + 
                         load_smem_b_n) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
    
    int load_gmem_b_k_mma_k = k * BK * WARP_TILE_K + MMA_K + load_smem_b_k;
    int load_gmem_b_addr_mma_k = load_gmem_b_k_mma_k * N + load_gmem_b_n; 
    uint32_t load_smem_b_mma_k_ptr = (
      smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) + 
      (smem_sel_next * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + 
      load_smem_b_n) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_b_mma_k_ptr, &B[load_gmem_b_addr_mma_k], 16);
    CP_ASYNC_COMMIT_GROUP();
    
    // ldmatrix for s_a, ldmatrix.trans for s_b.
    // smem -> reg buffers 1, second MMA_K, 16~31
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
      int lane_smem_a_k = (lane_id / 16) * 8; // 0,8
      uint32_t lane_smem_a_ptr = (
        smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(half) + 
        (smem_sel * s_a_stage_offset + lane_smem_a_m * (BK + A_PAD) + 
        lane_smem_a_k) * sizeof(half)
      );
      LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], 
                  RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], 
                  lane_smem_a_ptr);
    }

    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
      int lane_smem_b_k = lane_id % 16; // 0~15
      int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
      uint32_t lane_smem_b_ptr = (
        smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) + 
        (smem_sel * s_b_stage_offset + lane_smem_b_k * (BN + B_PAD) + 
        lane_smem_b_n) * sizeof(half)
      );
      // may use .x4.trans to load 4 matrix for reg double buffers at once?
      LDMATRIX_X2_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], 
                    lane_smem_b_ptr);
    }
    
    // MMA compute, first MMA_K
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // Warp swizzle: Right -> Left -> Right -> Left
        int j_s = ((i % 2) && WARP_SWIZZLE)? (WARP_TILE_N - j - 1) : j;
        HMMA16816(RC[i][j_s][0], RC[i][j_s][1], 
                  RA[reg_load_idx][i][0], RA[reg_load_idx][i][1], 
                  RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], 
                  RB[reg_load_idx][j_s][0], RB[reg_load_idx][j_s][1], 
                  RC[i][j_s][0], RC[i][j_s][1]);
      }
    }
    
    reg_store_idx ^= 1; // 1 -> 0
    reg_load_idx ^= 1; // 0 -> 1
    // MMA compute, second MMA_K 
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // Warp swizzle: Right -> Left -> Right -> Left
        int j_s = ((i % 2) && WARP_SWIZZLE)? (WARP_TILE_N - j - 1) : j;
        HMMA16816(RC[i][j_s][0], RC[i][j_s][1], 
                  RA[reg_load_idx][i][0], RA[reg_load_idx][i][1], 
                  RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], 
                  RB[reg_load_idx][j_s][0], RB[reg_load_idx][j_s][1], 
                  RC[i][j_s][0], RC[i][j_s][1]);
      }
    }

    CP_ASYNC_WAIT_GROUP(K_STAGE-2);
    __syncthreads(); 

    // load next k iters to reg buffers.
    // smem -> reg buffers 0, first MMA_K, 0~15
    // int smem_sel_reg = (k + 2) % K_STAGE; // vs smem_sel k=2->(0)1, k=3->(1)2
    int smem_sel_reg = (smem_sel + 1) % K_STAGE; // vs smem_sel k=2->(0)1, k=3->(1)2
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
      int lane_smem_a_k = (lane_id / 16) * 8; // 0,8
      uint32_t lane_smem_a_ptr = (
        smem_a_base_ptr + (smem_sel_reg * s_a_stage_offset + 
                           lane_smem_a_m * (BK + A_PAD) + 
                           lane_smem_a_k) * sizeof(half)
      );
      LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], 
                  RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], 
                  lane_smem_a_ptr);
    }

    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
      int lane_smem_b_k = lane_id % 16;  // 0~15, 0~15
      int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
      uint32_t lane_smem_b_ptr = (
        smem_b_base_ptr + (smem_sel_reg * s_b_stage_offset + 
                           lane_smem_b_k * (BN + B_PAD) + 
                           lane_smem_b_n) * sizeof(half)
      );
      // may use .x4.trans to load 4 matrix for reg double buffers at once?
      LDMATRIX_X2_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], 
                    lane_smem_b_ptr);
    }
  }

  // make sure all memory issues ready.
  if ((K_STAGE - 2) > 0) {
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads(); 
  }

  // processing last (K_STAGE-1) k iters.
  {
    #pragma unroll
    for (int k = 0; k < (K_STAGE - 1); k++) {
      reg_store_idx ^= 1; // 0->1
      reg_load_idx ^= 1; // 1->0

      int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
      // ldmatrix for s_a, ldmatrix.trans for s_b.
      // smem -> reg buffers 1, second MMA_K
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
        int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
        int lane_smem_a_k = (lane_id / 16) * 8; // 0,8
        uint32_t lane_smem_a_ptr = (
          smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(half) +
          (stage_sel * s_a_stage_offset + lane_smem_a_m * (BK + A_PAD) + 
          lane_smem_a_k) * sizeof(half)
        );
        LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], 
                    RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], 
                    lane_smem_a_ptr);
      }

      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
        int lane_smem_b_k = lane_id % 16; // 0~15
        int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
        uint32_t lane_smem_b_ptr = (
          smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) + 
          (stage_sel * s_b_stage_offset + lane_smem_b_k * (BN + B_PAD) + 
          lane_smem_b_n) * sizeof(half)
        );
        LDMATRIX_X2_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], 
                      lane_smem_b_ptr);
      }

      // MMA compute, first MMA_K
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          // Warp swizzle: Right -> Left -> Right -> Left
          int j_s = ((i % 2) && WARP_SWIZZLE)? (WARP_TILE_N - j - 1) : j;
          HMMA16816(RC[i][j_s][0], RC[i][j_s][1], 
                    RA[reg_load_idx][i][0], RA[reg_load_idx][i][1], 
                    RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], 
                    RB[reg_load_idx][j_s][0], RB[reg_load_idx][j_s][1], 
                    RC[i][j_s][0], RC[i][j_s][1]);
        }
      }

      reg_store_idx ^= 1; // 1 -> 0
      reg_load_idx ^= 1; // 0 -> 1

      // MMA compute, second MMA_K
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          // Warp swizzle: Right -> Left -> Right -> Left
          int j_s = ((i % 2) && WARP_SWIZZLE)? (WARP_TILE_N - j - 1) : j;
          HMMA16816(RC[i][j_s][0], RC[i][j_s][1], 
                    RA[reg_load_idx][i][0], RA[reg_load_idx][i][1], 
                    RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], 
                    RB[reg_load_idx][j_s][0], RB[reg_load_idx][j_s][1], 
                    RC[i][j_s][0], RC[i][j_s][1]);
        }
      }
      
      // load next k iters to reg buffers.
      // smem -> reg buffers 0, first MMA_K, 0~15
      // int stage_sel_reg = ((NUM_K_TILES - K_STAGE + k) % K_STAGE); 
      int stage_sel_reg = (stage_sel + 1) % K_STAGE; 
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
        int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
        int lane_smem_a_k = (lane_id / 16) * 8; // 0,8
        uint32_t lane_smem_a_ptr = (
          smem_a_base_ptr + (stage_sel_reg * s_a_stage_offset + 
                             lane_smem_a_m * (BK + A_PAD) + 
                             lane_smem_a_k) * sizeof(half)
        );
        LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], 
                    RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], 
                    lane_smem_a_ptr);
      }

      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
        int lane_smem_b_k = lane_id % 16;  // 0~15, 0~15
        int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
        uint32_t lane_smem_b_ptr = (
          smem_b_base_ptr + (stage_sel_reg * s_b_stage_offset + 
                             lane_smem_b_k * (BN + B_PAD) + 
                             lane_smem_b_n) * sizeof(half)
        );
        LDMATRIX_X2_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], 
                      lane_smem_b_ptr);
      }
    }
  }

  // collective store with reg reuse & warp shuffle
  for (int i = 0; i < WARP_TILE_M; ++i) {
    // reuse RA[2][4][4] reg here, this may boost 0.3~0.5 TFLOPS up.
    // may not put 'if' in N loop, it will crash the 'pragma unroll' hint ?
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      // How to use LDST128BITS here? __shfl_sync -> lane 0 -> store 8 half.
      // thus, we only need 8 memory issues with 128 bits after shfl_sync.
      RA[0][j][0] = RC[i][j][0]; 
      RA[1][j][0] = RC[i][j][1];
      RA[0][j][1] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 1);
      RA[0][j][2] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 2);
      RA[0][j][3] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 3);
      RA[1][j][1] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 1);
      RA[1][j][2] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 2);
      RA[1][j][3] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 3);
    }

    if (lane_id % 4 == 0) {
      int store_warp_smem_c_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int store_lane_gmem_c_m = by * BM + store_warp_smem_c_m + lane_id / 4;
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        int store_warp_smem_c_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
        int store_lane_gmem_c_n = bx * BN + store_warp_smem_c_n;
        int store_gmem_c_addr_0 = store_lane_gmem_c_m * N + store_lane_gmem_c_n;
        int store_gmem_c_addr_1 = (store_lane_gmem_c_m + 8) * N + store_lane_gmem_c_n;
        LDST128BITS(C[store_gmem_c_addr_0]) = LDST128BITS(RA[0][j][0]); 
        LDST128BITS(C[store_gmem_c_addr_1]) = LDST128BITS(RA[1][j][0]); 
      }
    }
  }
}

// NOTE: use ldmatrix.x4.trans for matrix B smem -> reg
// In order to reduce bank conflicts, we will save the K(16x2=32) 
// dimension by half according to the stage dimension. For example, 
// stages=3, warp_tile_k=2, it will be saved as [3*2][BM][16].
// 128x128, mma2x4, warp4x4(64,32,32), stages, block swizzle, dsmem, 
// k32 with reg double buffers
template<const int MMA_M=16, 
         const int MMA_N=8, 
         const int MMA_K=16,
         const int MMA_TILE_M=2,
         const int MMA_TILE_N=4,
         const int WARP_TILE_M=4,
         const int WARP_TILE_N=4,
         const int WARP_TILE_K=2,
         const int A_PAD=0, 
         const int B_PAD=0,
         const int K_STAGE=2, 
         const bool BLOCK_SWIZZLE=true,
         const bool WARP_SWIZZLE=true>
__global__ void  __launch_bounds__(256) 
hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_x4_kernel(
  const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C, 
  int M, int N, int K) {
  // BLOCK_SWIZZLE 0/1 control use block swizzle or not.
  const int bx = ((int) BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, MMA_K * WARP_TILE_K);
  constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M; // 16*2*4=128
  constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N; // 8*4*4=128
  constexpr int BK = MMA_K; // 16x2=32

  extern __shared__ half smem[]; 
  half* s_a = smem;
  half* s_b = smem + K_STAGE * BM * (BK + A_PAD) * WARP_TILE_K;
  constexpr int s_a_stage_offset = BM * (BK + A_PAD); // 128x16 
  constexpr int s_b_stage_offset = BK * (BN + B_PAD); // 16x128
  constexpr int s_a_mma_k_store_offset = K_STAGE * BM * (BK + A_PAD);
  constexpr int s_b_mma_k_store_offset = K_STAGE * BK * (BN + B_PAD);

  const int tid = threadIdx.y * blockDim.x + threadIdx.x; // within block
  const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
  const int lane_id = tid % WARP_SIZE; // 0~31
  const int warp_m = warp_id % 2; // 0,1
  const int warp_n = warp_id / 2; // 0,1,2,3

  int load_smem_a_m = tid / 2; // row 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 8; // col 0,8
  int load_smem_b_k = tid / 16; // row 0~15
  int load_smem_b_n = (tid % 16) * 8; // col 0,8,16,...
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
  if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

  uint32_t RC[WARP_TILE_M][WARP_TILE_N][2];
  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      RC[i][j][0] = 0;
      RC[i][j][1] = 0;
    }
  }

  uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
  uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

  #pragma unroll
  for (int k = 0; k < (K_STAGE - 1); ++k) { // 0, 1
    // k * WMMA_K, WMMA_K=16 -> (k << 4)
    int load_gmem_a_k = k * BK * WARP_TILE_K + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * BK * WARP_TILE_K + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 

    uint32_t load_smem_a_ptr = (
      smem_a_base_ptr + (k * s_a_stage_offset + 
                         load_smem_a_m * (BK + A_PAD) + 
                         load_smem_a_k) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16); // MMA_K 0
    uint32_t load_smem_a_mma_k_ptr = (
      smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(half) + 
      (k * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + 
      load_smem_a_k) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_a_mma_k_ptr, &A[load_gmem_a_addr + 16], 16); // MMA_K 1

    uint32_t load_smem_b_ptr = (
      smem_b_base_ptr + (k * s_b_stage_offset + 
                         load_smem_b_k * (BN + B_PAD) + 
                         load_smem_b_n) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
    
    int load_gmem_b_k_mma_k = k * BK * WARP_TILE_K + MMA_K + load_smem_b_k;
    int load_gmem_b_addr_mma_k = load_gmem_b_k_mma_k * N + load_gmem_b_n; 
    uint32_t load_smem_b_mma_k_ptr = (
      smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) + 
      (k * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + 
      load_smem_b_n) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_b_mma_k_ptr, &B[load_gmem_b_addr_mma_k], 16);

    CP_ASYNC_COMMIT_GROUP();
  }

  CP_ASYNC_WAIT_GROUP(K_STAGE-2); // s2->0, s3->1, s4->2
  __syncthreads(); 

  uint32_t RA[2][WARP_TILE_M][4];
  uint32_t RB[2][WARP_TILE_N][2];

  int reg_store_idx = 0;
  int reg_load_idx = 1;

  { 
    // ldmatrix for s_a, ldmatrix.trans for s_b.
    // smem -> reg buffers 0, first MMA_K, 0~15
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
      int lane_smem_a_k = (lane_id / 16) * 8; // 0,8
      uint32_t lane_smem_a_ptr = (
        smem_a_base_ptr + 
        (0 * s_a_stage_offset + lane_smem_a_m * (BK + A_PAD) + 
        lane_smem_a_k) * sizeof(half)
      );
      LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], 
                  RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], 
                  lane_smem_a_ptr);
    }

    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
      int lane_smem_b_k = lane_id % 16;  // 0~15, 0~15
      int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
      uint32_t lane_smem_b_ptr = (
        smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) * (lane_id / 16) +
        (0 * s_b_stage_offset + lane_smem_b_k * (BN + B_PAD) + 
        lane_smem_b_n) * sizeof(half)
      );
      // TRICK: I use .x4.trans to load 4 matrix for reg double buffers at once.
      LDMATRIX_X4_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], 
                    RB[reg_load_idx][j][0],  RB[reg_load_idx][j][1],
                    lane_smem_b_ptr);
    }
  }
  
  #pragma unroll
  for (int k = (K_STAGE - 1); k < NUM_K_TILES; ++k) {
    reg_store_idx ^= 1; // 0->1
    reg_load_idx ^= 1; // 1->0
    int smem_sel = (k + 1) % K_STAGE; // s3 k 2->0, k 3->1, k 4->2...
    int smem_sel_next = k % K_STAGE;  // s3 k 2->2, k 3->0, k 4->1...

    // stage gmem -> smem
    int load_gmem_a_k = k * BK * WARP_TILE_K + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * BK * WARP_TILE_K + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 

    uint32_t load_smem_a_ptr = (
      smem_a_base_ptr + (smem_sel_next * s_a_stage_offset + 
                         load_smem_a_m * (BK + A_PAD) + 
                         load_smem_a_k) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16); // MMA_K 0
    uint32_t load_smem_a_mma_k_ptr = (
      smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(half) + 
      (smem_sel_next * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + 
      load_smem_a_k) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_a_mma_k_ptr, &A[load_gmem_a_addr + 16], 16); // MMA_K 1

    uint32_t load_smem_b_ptr = (
      smem_b_base_ptr + (smem_sel_next * s_b_stage_offset + 
                         load_smem_b_k * (BN + B_PAD) + 
                         load_smem_b_n) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
    
    int load_gmem_b_k_mma_k = k * BK * WARP_TILE_K + MMA_K + load_smem_b_k;
    int load_gmem_b_addr_mma_k = load_gmem_b_k_mma_k * N + load_gmem_b_n; 
    uint32_t load_smem_b_mma_k_ptr = (
      smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) + 
      (smem_sel_next * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + 
      load_smem_b_n) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_b_mma_k_ptr, &B[load_gmem_b_addr_mma_k], 16);
    CP_ASYNC_COMMIT_GROUP();
    
    // ldmatrix for s_a, ldmatrix.trans for s_b.
    // smem -> reg buffers 1, second MMA_K, 16~31
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
      int lane_smem_a_k = (lane_id / 16) * 8; // 0,8
      uint32_t lane_smem_a_ptr = (
        smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(half) + 
        (smem_sel * s_a_stage_offset + lane_smem_a_m * (BK + A_PAD) + 
        lane_smem_a_k) * sizeof(half)
      );
      LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], 
                  RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], 
                  lane_smem_a_ptr);
    }
    
    // MMA compute, first MMA_K
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // Warp swizzle: Right -> Left -> Right -> Left
        int j_s = ((i % 2) && WARP_SWIZZLE)? (WARP_TILE_N - j - 1) : j;
        HMMA16816(RC[i][j_s][0], RC[i][j_s][1], 
                  RA[reg_load_idx][i][0], RA[reg_load_idx][i][1], 
                  RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], 
                  RB[reg_load_idx][j_s][0], RB[reg_load_idx][j_s][1], 
                  RC[i][j_s][0], RC[i][j_s][1]);
      }
    }
    
    reg_store_idx ^= 1; // 1 -> 0
    reg_load_idx ^= 1; // 0 -> 1
    // MMA compute, second MMA_K 
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // Warp swizzle: Right -> Left -> Right -> Left
        int j_s = ((i % 2) && WARP_SWIZZLE)? (WARP_TILE_N - j - 1) : j;
        HMMA16816(RC[i][j_s][0], RC[i][j_s][1], 
                  RA[reg_load_idx][i][0], RA[reg_load_idx][i][1], 
                  RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], 
                  RB[reg_load_idx][j_s][0], RB[reg_load_idx][j_s][1], 
                  RC[i][j_s][0], RC[i][j_s][1]);
      }
    }

    CP_ASYNC_WAIT_GROUP(K_STAGE-2);
    __syncthreads(); 

    // load next k iters to reg buffers.
    // smem -> reg buffers 0, first MMA_K, 0~15
    // int smem_sel_reg = (k + 2) % K_STAGE; // vs smem_sel k=2->(0)1, k=3->(1)2
    int smem_sel_reg = (smem_sel + 1) % K_STAGE; // vs smem_sel k=2->(0)1, k=3->(1)2
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
      int lane_smem_a_k = (lane_id / 16) * 8; // 0,8
      uint32_t lane_smem_a_ptr = (
        smem_a_base_ptr + (smem_sel_reg * s_a_stage_offset + 
                           lane_smem_a_m * (BK + A_PAD) + 
                           lane_smem_a_k) * sizeof(half)
      );
      LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], 
                  RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], 
                  lane_smem_a_ptr);
    }

    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
      int lane_smem_b_k = lane_id % 16;  // 0~15, 0~15
      int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
      uint32_t lane_smem_b_ptr = (
        smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) * (lane_id / 16) +
        (smem_sel_reg * s_b_stage_offset + lane_smem_b_k * (BN + B_PAD) + 
        lane_smem_b_n) * sizeof(half)
      );
      // may use .x4.trans to load 4 matrix for reg double buffers at once?
      LDMATRIX_X4_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], 
                    RB[reg_load_idx][j][0],  RB[reg_load_idx][j][1],
                    lane_smem_b_ptr);
    }
  }

  // make sure all memory issues ready.
  if ((K_STAGE - 2) > 0) {
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads(); 
  }

  // processing last (K_STAGE-1) k iters.
  {
    #pragma unroll
    for (int k = 0; k < (K_STAGE - 1); k++) {
      reg_store_idx ^= 1; // 0->1
      reg_load_idx ^= 1; // 1->0

      int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
      // ldmatrix for s_a, ldmatrix.trans for s_b.
      // smem -> reg buffers 1, second MMA_K
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
        int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
        int lane_smem_a_k = (lane_id / 16) * 8; // 0,8
        uint32_t lane_smem_a_ptr = (
          smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(half) +
          (stage_sel * s_a_stage_offset + lane_smem_a_m * (BK + A_PAD) + 
          lane_smem_a_k) * sizeof(half)
        );
        LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], 
                    RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], 
                    lane_smem_a_ptr);
      }

      // MMA compute, first MMA_K
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          // Warp swizzle: Right -> Left -> Right -> Left
          int j_s = ((i % 2) && WARP_SWIZZLE)? (WARP_TILE_N - j - 1) : j;
          HMMA16816(RC[i][j_s][0], RC[i][j_s][1], 
                    RA[reg_load_idx][i][0], RA[reg_load_idx][i][1], 
                    RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], 
                    RB[reg_load_idx][j_s][0], RB[reg_load_idx][j_s][1], 
                    RC[i][j_s][0], RC[i][j_s][1]);
        }
      }

      reg_store_idx ^= 1; // 1 -> 0
      reg_load_idx ^= 1; // 0 -> 1

      // MMA compute, second MMA_K
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          // Warp swizzle: Right -> Left -> Right -> Left
          int j_s = ((i % 2) && WARP_SWIZZLE)? (WARP_TILE_N - j - 1) : j;
          HMMA16816(RC[i][j_s][0], RC[i][j_s][1], 
                    RA[reg_load_idx][i][0], RA[reg_load_idx][i][1], 
                    RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], 
                    RB[reg_load_idx][j_s][0], RB[reg_load_idx][j_s][1], 
                    RC[i][j_s][0], RC[i][j_s][1]);
        }
      }
      
      // load next k iters to reg buffers.
      // smem -> reg buffers 0, first MMA_K, 0~15
      // int stage_sel_reg = ((NUM_K_TILES - K_STAGE + k) % K_STAGE); 
      int stage_sel_reg = (stage_sel + 1) % K_STAGE; 
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
        int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
        int lane_smem_a_k = (lane_id / 16) * 8; // 0,8
        uint32_t lane_smem_a_ptr = (
          smem_a_base_ptr + (stage_sel_reg * s_a_stage_offset + 
                             lane_smem_a_m * (BK + A_PAD) + 
                             lane_smem_a_k) * sizeof(half)
        );
        LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], 
                    RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], 
                    lane_smem_a_ptr);
      }

      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
        int lane_smem_b_k = lane_id % 16;  // 0~15, 0~15
        int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
        uint32_t lane_smem_b_ptr = (
          smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) * (lane_id / 16) +
          (stage_sel_reg * s_b_stage_offset + lane_smem_b_k * (BN + B_PAD) + 
          lane_smem_b_n) * sizeof(half)
        );
        // may use .x4.trans to load 4 matrix for reg double buffers at once?
        LDMATRIX_X4_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], 
                      RB[reg_load_idx][j][0],  RB[reg_load_idx][j][1],
                      lane_smem_b_ptr);
      }
    }
  }

  // collective store with reg reuse & warp shuffle
  for (int i = 0; i < WARP_TILE_M; ++i) {
    // reuse RA[2][4][4] reg here, this may boost 0.3~0.5 TFLOPS up.
    // may not put 'if' in N loop, it will crash the 'pragma unroll' hint ?
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      // How to use LDST128BITS here? __shfl_sync -> lane 0 -> store 8 half.
      // thus, we only need 8 memory issues with 128 bits after shfl_sync.
      RA[0][j][0] = RC[i][j][0]; 
      RA[1][j][0] = RC[i][j][1];
      RA[0][j][1] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 1);
      RA[0][j][2] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 2);
      RA[0][j][3] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 3);
      RA[1][j][1] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 1);
      RA[1][j][2] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 2);
      RA[1][j][3] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 3);
    }

    if (lane_id % 4 == 0) {
      int store_warp_smem_c_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int store_lane_gmem_c_m = by * BM + store_warp_smem_c_m + lane_id / 4;
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        int store_warp_smem_c_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
        int store_lane_gmem_c_n = bx * BN + store_warp_smem_c_n;
        int store_gmem_c_addr_0 = store_lane_gmem_c_m * N + store_lane_gmem_c_n;
        int store_gmem_c_addr_1 = (store_lane_gmem_c_m + 8) * N + store_lane_gmem_c_n;
        LDST128BITS(C[store_gmem_c_addr_0]) = LDST128BITS(RA[0][j][0]); 
        LDST128BITS(C[store_gmem_c_addr_1]) = LDST128BITS(RA[1][j][0]); 
      }
    }
  }
}

// NOTE: reduce registers usage.
// In order to reduce bank conflicts, we will save the K(16x2=32) 
// dimension by half according to the stage dimension. For example, 
// stages=3, warp_tile_k=2, it will be saved as [3*2][BM][16].
// 128x128, mma2x4, warp4x4(64,32,32), stages, block swizzle, dsmem, 
// k32 with reg double buffers
template<const int MMA_M=16, 
         const int MMA_N=8, 
         const int MMA_K=16,
         const int MMA_TILE_M=2,
         const int MMA_TILE_N=4,
         const int WARP_TILE_M=4,
         const int WARP_TILE_N=4,
         const int WARP_TILE_K=2,
         const int A_PAD=0, 
         const int B_PAD=0,
         const int K_STAGE=2, 
         const bool BLOCK_SWIZZLE=true,
         const bool WARP_SWIZZLE=true>
__global__ void  __launch_bounds__(256) 
hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_rr_kernel(
  const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C, 
  int M, int N, int K) {
  // BLOCK_SWIZZLE 0/1 control use block swizzle or not.
  const int bx = ((int) BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, MMA_K * WARP_TILE_K);
  constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M; // 16*2*4=128
  constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N; // 8*4*4=128
  constexpr int BK = MMA_K; // 16x2=32

  extern __shared__ half smem[]; 
  half* s_a = smem;
  half* s_b = smem + K_STAGE * BM * (BK + A_PAD) * WARP_TILE_K;
  constexpr int s_a_stage_offset = BM * (BK + A_PAD); // 128x16 
  constexpr int s_b_stage_offset = BK * (BN + B_PAD); // 16x128
  constexpr int s_a_mma_k_store_offset = K_STAGE * BM * (BK + A_PAD);
  constexpr int s_b_mma_k_store_offset = K_STAGE * BK * (BN + B_PAD);

  const int tid = threadIdx.y * blockDim.x + threadIdx.x; // within block
  const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
  const int lane_id = tid % WARP_SIZE; // 0~31
  const int warp_m = warp_id % 2; // 0,1
  const int warp_n = warp_id / 2; // 0,1,2,3

  const int load_smem_a_m = tid / 2; // row 0~127
  const int load_smem_a_k = (tid % 2 == 0) ? 0 : 8; // col 0,8
  const int load_smem_b_k = tid / 16; // row 0~15
  const int load_smem_b_n = (tid % 16) * 8; // col 0,8,16,...
  const int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  const int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
  if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;
  // 16 reg for pre-defined vars.

  uint32_t RC[WARP_TILE_M][WARP_TILE_N][2]; // 4*4*2=32 reg
  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      RC[i][j][0] = 0;
      RC[i][j][1] = 0;
    }
  }

  uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
  uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

  #pragma unroll
  for (int k = 0; k < (K_STAGE - 1); ++k) { 
    // reduce 9 registers -> 4 registers.
    // a gmem -> a smem
    uint32_t load_smem_a_ptr = (
      smem_a_base_ptr + (k * s_a_stage_offset + 
                         load_smem_a_m * (BK + A_PAD) + 
                         load_smem_a_k) * sizeof(half)
    );
    CP_ASYNC_CG(
      load_smem_a_ptr, 
      &A[load_gmem_a_m * K + k * BK * WARP_TILE_K + load_smem_a_k], 
      16
    ); // MMA_K 0
    uint32_t load_smem_a_mma_k_ptr = (
      smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(half) + 
      (k * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + 
      load_smem_a_k) * sizeof(half)
    );
    CP_ASYNC_CG(
      load_smem_a_mma_k_ptr, 
      &A[load_gmem_a_m * K + k * BK * WARP_TILE_K + load_smem_a_k + 16], 
      16
    ); // MMA_K 1

    // b gmem -> b smem
    uint32_t load_smem_b_ptr = (
      smem_b_base_ptr + (k * s_b_stage_offset + 
                         load_smem_b_k * (BN + B_PAD) + 
                         load_smem_b_n) * sizeof(half)
    );
    CP_ASYNC_CG(
      load_smem_b_ptr, 
      &B[(k * BK * WARP_TILE_K + load_smem_b_k) * N + load_gmem_b_n], 
      16
    ); // MMA_K 0
    uint32_t load_smem_b_mma_k_ptr = (
      smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) + 
      (k * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + 
      load_smem_b_n) * sizeof(half)
    );
    CP_ASYNC_CG(
      load_smem_b_mma_k_ptr, 
      &B[(k * BK * WARP_TILE_K + MMA_K + load_smem_b_k) * N + load_gmem_b_n], 
      16
    ); // MMA_K 1

    CP_ASYNC_COMMIT_GROUP();
  }

  CP_ASYNC_WAIT_GROUP(K_STAGE-2); // s2->0, s3->1, s4->2
  __syncthreads(); 

  uint32_t RA[2][WARP_TILE_M][4]; // 2*4*4=32 reg
  uint32_t RB[2][WARP_TILE_N][2]; // 2*4*2=16 reg

  // 16+32+32+16=96 reg

  int reg_store_idx = 0;
  int reg_load_idx = 1;

  { 
    // ldmatrix for s_a, ldmatrix.trans for s_b.
    // smem -> reg buffers 0, first MMA_K, 0~15
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      // reduce 4 registers -> 1 registers.
      uint32_t lane_smem_a_ptr = (
        smem_a_base_ptr + (
                    (warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M + lane_id % 16) * 
                    (BK + A_PAD) + (lane_id / 16) * 8
                  ) * sizeof(half)
      );
      LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], 
                  RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], 
                  lane_smem_a_ptr
      );
    }

    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      // reduce 4 registers -> 1 registers.
      uint32_t lane_smem_b_ptr = (
        smem_b_base_ptr + (
                      (lane_id % 16) * (BN + B_PAD) + 
                      (warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N)
                    ) * sizeof(half)
      );
      LDMATRIX_X2_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], 
                    lane_smem_b_ptr
      );
    }
  }
  
  int smem_sel = 0; // s3 k 2->0, k 3->1, k 4->2...
  int smem_sel_next = (K_STAGE - 1) % K_STAGE;  // s3 k 2->2, k 3->0, k 4->1...
  #pragma unroll
  for (int k = (K_STAGE - 1); k < NUM_K_TILES; ++k) {
    reg_store_idx ^= 1; // 0->1
    reg_load_idx ^= 1; // 1->0
    smem_sel = (k + 1) % K_STAGE; // s3 k 2->0, k 3->1, k 4->2...
    smem_sel_next = k % K_STAGE;  // s3 k 2->2, k 3->0, k 4->1...

    // reduce 9 registers -> 4 registers.
    // a gmem -> a smem
    uint32_t load_smem_a_ptr = (
      smem_a_base_ptr + (smem_sel_next * s_a_stage_offset + 
                         load_smem_a_m * (BK + A_PAD) + 
                         load_smem_a_k) * sizeof(half)
    );
    CP_ASYNC_CG(
      load_smem_a_ptr, 
      &A[load_gmem_a_m * K + k * BK * WARP_TILE_K + load_smem_a_k], 
      16
    ); // MMA_K 0
    uint32_t load_smem_a_mma_k_ptr = (
      smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(half) + 
      (smem_sel_next * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + 
      load_smem_a_k) * sizeof(half)
    );
    CP_ASYNC_CG(
      load_smem_a_mma_k_ptr, 
      &A[load_gmem_a_m * K + k * BK * WARP_TILE_K + load_smem_a_k + 16], 
      16
    ); // MMA_K 1

    // b gmem -> b smem
    uint32_t load_smem_b_ptr = (
      smem_b_base_ptr + (smem_sel_next * s_b_stage_offset + 
                         load_smem_b_k * (BN + B_PAD) + 
                         load_smem_b_n) * sizeof(half)
    );
    CP_ASYNC_CG(
      load_smem_b_ptr, 
      &B[(k * BK * WARP_TILE_K + load_smem_b_k) * N + load_gmem_b_n], 
      16
    ); // MMA_K 0
    uint32_t load_smem_b_mma_k_ptr = (
      smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) + 
      (smem_sel_next * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + 
      load_smem_b_n) * sizeof(half)
    );
    CP_ASYNC_CG(
      load_smem_b_mma_k_ptr, 
      &B[(k * BK * WARP_TILE_K + MMA_K + load_smem_b_k) * N + load_gmem_b_n], 
      16
    ); // MMA_K 1

    CP_ASYNC_COMMIT_GROUP();
    
    // ldmatrix for s_a, ldmatrix.trans for s_b.
    // smem -> reg buffers 1, second MMA_K, 16~31
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      // reduce 4 registers -> 1 registers.
      uint32_t lane_smem_a_ptr = (
        smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(half) + (
                    smem_sel * s_a_stage_offset + 
                    (warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M + lane_id % 16) * 
                    (BK + A_PAD) + (lane_id / 16) * 8
                  ) * sizeof(half)
      );
      LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], 
                  RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], 
                  lane_smem_a_ptr
      );
    }

    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      // reduce 4 registers -> 1 registers.
      uint32_t lane_smem_b_ptr = (
        smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) + (
                      smem_sel * s_b_stage_offset + 
                      (lane_id % 16) * (BN + B_PAD) + 
                      (warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N)
                    ) * sizeof(half)
      );
      LDMATRIX_X2_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], 
                    lane_smem_b_ptr
      );
    }
    
    // MMA compute, first MMA_K
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // Warp swizzle: Right -> Left -> Right -> Left
        int j_s = ((i % 2) && WARP_SWIZZLE)? (WARP_TILE_N - j - 1) : j;
        HMMA16816(RC[i][j_s][0], RC[i][j_s][1], 
                  RA[reg_load_idx][i][0], RA[reg_load_idx][i][1], 
                  RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], 
                  RB[reg_load_idx][j_s][0], RB[reg_load_idx][j_s][1], 
                  RC[i][j_s][0], RC[i][j_s][1]);
      }
    }
    
    reg_store_idx ^= 1; // 1 -> 0
    reg_load_idx ^= 1; // 0 -> 1
    // MMA compute, second MMA_K 
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // Warp swizzle: Right -> Left -> Right -> Left
        int j_s = ((i % 2) && WARP_SWIZZLE)? (WARP_TILE_N - j - 1) : j;
        HMMA16816(RC[i][j_s][0], RC[i][j_s][1], 
                  RA[reg_load_idx][i][0], RA[reg_load_idx][i][1], 
                  RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], 
                  RB[reg_load_idx][j_s][0], RB[reg_load_idx][j_s][1], 
                  RC[i][j_s][0], RC[i][j_s][1]);
      }
    }

    CP_ASYNC_WAIT_GROUP(K_STAGE-2);
    __syncthreads(); 

    // load next k iters to reg buffers.
    // smem -> reg buffers 0, first MMA_K, 0~15
    // int smem_sel_reg = (smem_sel + 1) % K_STAGE; // vs smem_sel k=2->(0)1, k=3->(1)2
    smem_sel = (smem_sel + 1) % K_STAGE;
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      // reduce 4 registers -> 1 registers.
      uint32_t lane_smem_a_ptr = (
        smem_a_base_ptr + (
                    smem_sel * s_a_stage_offset + 
                    (warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M + lane_id % 16) * 
                    (BK + A_PAD) + (lane_id / 16) * 8
                  ) * sizeof(half)
      );
      LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], 
                  RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], 
                  lane_smem_a_ptr
      );
    }

    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      // reduce 4 registers -> 1 registers.
      uint32_t lane_smem_b_ptr = (
        smem_b_base_ptr + (
                      smem_sel * s_b_stage_offset + 
                      (lane_id % 16) * (BN + B_PAD) + 
                      (warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N)
                    ) * sizeof(half)
      );
      LDMATRIX_X2_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], 
                    lane_smem_b_ptr
      );
    }
  }

  // make sure all memory issues ready.
  if ((K_STAGE - 2) > 0) {
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads(); 
  }

  // processing last (K_STAGE-1) k iters.
  {
    #pragma unroll
    for (int k = 0; k < (K_STAGE - 1); k++) {
      reg_store_idx ^= 1; // 0->1
      reg_load_idx ^= 1; // 1->0

      // int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
      smem_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
      // ldmatrix for s_a, ldmatrix.trans for s_b.
      // smem -> reg buffers 1, second MMA_K
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        // reduce 4 registers -> 1 registers.
        uint32_t lane_smem_a_ptr = (
          smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(half) + (
                      smem_sel * s_a_stage_offset + 
                      (warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M + lane_id % 16) * 
                      (BK + A_PAD) + (lane_id / 16) * 8
                    ) * sizeof(half)
        );
        LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], 
                    RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], 
                    lane_smem_a_ptr
        );
      }

      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // reduce 4 registers -> 1 registers.
        uint32_t lane_smem_b_ptr = (
          smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) + (
                        smem_sel * s_b_stage_offset + 
                        (lane_id % 16) * (BN + B_PAD) + 
                        (warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N)
                      ) * sizeof(half)
        );
        LDMATRIX_X2_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], 
                      lane_smem_b_ptr
        );
      }

      // MMA compute, first MMA_K
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          // Warp swizzle: Right -> Left -> Right -> Left
          int j_s = ((i % 2) && WARP_SWIZZLE)? (WARP_TILE_N - j - 1) : j;
          HMMA16816(RC[i][j_s][0], RC[i][j_s][1], 
                    RA[reg_load_idx][i][0], RA[reg_load_idx][i][1], 
                    RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], 
                    RB[reg_load_idx][j_s][0], RB[reg_load_idx][j_s][1], 
                    RC[i][j_s][0], RC[i][j_s][1]);
        }
      }

      reg_store_idx ^= 1; // 1 -> 0
      reg_load_idx ^= 1; // 0 -> 1

      // MMA compute, second MMA_K
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          // Warp swizzle: Right -> Left -> Right -> Left
          int j_s = ((i % 2) && WARP_SWIZZLE)? (WARP_TILE_N - j - 1) : j;
          HMMA16816(RC[i][j_s][0], RC[i][j_s][1], 
                    RA[reg_load_idx][i][0], RA[reg_load_idx][i][1], 
                    RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], 
                    RB[reg_load_idx][j_s][0], RB[reg_load_idx][j_s][1], 
                    RC[i][j_s][0], RC[i][j_s][1]);
        }
      }
      
      // load next k iters to reg buffers.
      // smem -> reg buffers 0, first MMA_K, 0~15
      // int stage_sel_reg = (stage_sel + 1) % K_STAGE; 
      smem_sel = (smem_sel + 1) % K_STAGE; 
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        // reduce 4 registers -> 1 registers.
        uint32_t lane_smem_a_ptr = (
          smem_a_base_ptr + (
                      smem_sel * s_a_stage_offset + 
                      (warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M + lane_id % 16) * 
                      (BK + A_PAD) + (lane_id / 16) * 8
                    ) * sizeof(half)
        );
        LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], 
                    RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], 
                    lane_smem_a_ptr
        );
      }

      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // reduce 4 registers -> 1 registers.
        uint32_t lane_smem_b_ptr = (
          smem_b_base_ptr + (
                        smem_sel * s_b_stage_offset + 
                        (lane_id % 16) * (BN + B_PAD) + 
                        (warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N)
                      ) * sizeof(half)
        );
        LDMATRIX_X2_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], 
                      lane_smem_b_ptr
        );
      }
    }
  }

  // collective store with reg reuse & warp shuffle
  for (int i = 0; i < WARP_TILE_M; ++i) {
    // reuse RA[2][4][4] reg here, this may boost 0.3~0.5 TFLOPS up.
    // may not put 'if' in N loop, it will crash the 'pragma unroll' hint ?
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      // How to use LDST128BITS here? __shfl_sync -> lane 0 -> store 8 half.
      // thus, we only need 8 memory issues with 128 bits after shfl_sync.
      RA[0][j][0] = RC[i][j][0]; 
      RA[1][j][0] = RC[i][j][1];
      RA[0][j][1] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 1);
      RA[0][j][2] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 2);
      RA[0][j][3] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 3);
      RA[1][j][1] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 1);
      RA[1][j][2] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 2);
      RA[1][j][3] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 3);
    }

    if (lane_id % 4 == 0) {
      // reduce 6 registers -> 0 registers.
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        LDST128BITS(
          C[
            (by * BM + warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M + lane_id / 4) * N + 
            (bx * BN + warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N)
          ]
        ) = (LDST128BITS(RA[0][j][0])); 
        LDST128BITS(
          C[
            (by * BM + warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M + lane_id / 4 + 8) * N + 
            (bx * BN + warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N)
          ]
        ) = (LDST128BITS(RA[1][j][0])); 
      }
    }
  }
}



// NN: A/B/C All row major
// TN: A row major MxK, B col major NxK, C row major MxN
// 128x128, mma2x4, warp4x4(64,32,16), stages, block swizzle, dsmem
template<const int MMA_M=16, 
         const int MMA_N=8, 
         const int MMA_K=16,
         const int MMA_TILE_M=2,
         const int MMA_TILE_N=4,
         const int WARP_TILE_M=4,
         const int WARP_TILE_N=4,
         const int A_PAD=0, 
         const int B_PAD=0,
         const int K_STAGE=2, 
         const bool BLOCK_SWIZZLE=false>
__global__ void  __launch_bounds__(256) 
hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn_kernel(
  half* A, half* B, half* C, int M, int N, int K) {
  // BLOCK_SWIZZLE 0/1 control use block swizzle or not.
  const int bx = ((int) BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, MMA_K);
  constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M; // 16*2*4=128
  constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N; // 8*4*4=128
  constexpr int BK = MMA_K; // 16

  extern __shared__ half smem[]; 
  half* s_a = smem;
  half* s_b = smem + K_STAGE * BM * (BK + A_PAD);
  constexpr int s_a_stage_offset = BM * (BK + A_PAD); // BMxBK 128*16
  constexpr int s_b_stage_offset = BN * (BK + B_PAD); // BNxBK 128*16

  const int tid = threadIdx.y * blockDim.x + threadIdx.x; // within block
  const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
  const int lane_id = tid % WARP_SIZE; // 0~31
  const int warp_m = warp_id % 2; // 0,1
  const int warp_n = warp_id / 2; // 0,1,2,3

  int load_smem_a_m = tid / 2; // row 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 8; // col 0,8
  int load_smem_b_n = tid / 2; // row 0~127
  int load_smem_b_k = (tid % 2 == 0) ? 0 : 8; // col 0,8
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of c
  if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

  uint32_t RC[WARP_TILE_M][WARP_TILE_N][2];
  #pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      RC[i][j][0] = 0;
      RC[i][j][1] = 0;
    }
  }
  
  // may avoid cvta overhead ? only cvta smem base ptr once for cp.async.
  uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
  uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

  #pragma unroll
  for (int k = 0; k < (K_STAGE - 1); ++k) { // 0, 1
    int load_gmem_a_k = k * BK + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * BK + load_smem_b_k; // global col of b
    int load_gmem_b_addr = load_gmem_b_n * K + load_gmem_b_k; 

    uint32_t load_smem_a_ptr = (
      smem_a_base_ptr + (k * s_a_stage_offset + 
                         load_smem_a_m * (BK + A_PAD) + 
                         load_smem_a_k) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

    uint32_t load_smem_b_ptr = (
      smem_b_base_ptr + (k * s_b_stage_offset + 
                         load_smem_b_n * (BK + B_PAD) + 
                         load_smem_b_k) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);

    CP_ASYNC_COMMIT_GROUP();
  }

  CP_ASYNC_WAIT_GROUP(K_STAGE-2); // s2->0, s3->1, s4->2
  __syncthreads(); 

  #pragma unroll
  for (int k = (K_STAGE - 1); k < NUM_K_TILES; ++k) {
    // gmem -> smem
    // s2/4 can use bitwise ops but s3 can not, so, we use mod
    // ops for all stages kernel. s2: (k + 1)&1, s4: (k + 1)&3
    // s3: (k + 1) % 3
    int smem_sel = (k + 1) % K_STAGE; // s3 k 2->0, k 3->1, k 4->2...
    int smem_sel_next = k % K_STAGE;  // s3 k 2->2, k 3->0, k 4->1...

    int load_gmem_a_k = k * BK + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * BK + load_smem_b_k; // global col of b
    int load_gmem_b_addr = load_gmem_b_n * K + load_gmem_b_k; 

    uint32_t load_smem_a_ptr = (
      smem_a_base_ptr + (smem_sel_next * s_a_stage_offset + 
                         load_smem_a_m * (BK + A_PAD) + 
                         load_smem_a_k) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

    uint32_t load_smem_b_ptr = (
      smem_b_base_ptr + (smem_sel_next * s_b_stage_offset + 
                         load_smem_b_n * (BK + B_PAD) + 
                         load_smem_b_k) * sizeof(half)
    );
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);

    CP_ASYNC_COMMIT_GROUP();
    
    uint32_t RA[WARP_TILE_M][4];
    uint32_t RB[WARP_TILE_N][2];
    // smem -> reg
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
      int lane_smem_a_k = (lane_id / 16) * 8; // 0,8
      uint32_t lane_smem_a_ptr = (
        smem_a_base_ptr + (smem_sel * s_a_stage_offset + 
                           lane_smem_a_m * (BK + A_PAD) + 
                           lane_smem_a_k) * sizeof(half)
      );
      LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], lane_smem_a_ptr);
    }

    #pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
      int lane_smem_b_n = warp_smem_b_n + lane_id % 8; // 0~7, MMA_N=8
      int lane_smem_b_k = ((lane_id / 8) % 2) * 8; // 0,8
      uint32_t lane_smem_b_ptr = (
        smem_b_base_ptr + (smem_sel * s_b_stage_offset + 
                           lane_smem_b_n * (BK + B_PAD) + 
                           lane_smem_b_k) * sizeof(half)
      );
      LDMATRIX_X2(RB[j][0], RB[j][1], lane_smem_b_ptr);
    }
    
    // MMA compute
    #pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        HMMA16816(RC[i][j][0], RC[i][j][1], 
                  RA[i][0], RA[i][1], RA[i][2], RA[i][3], 
                  RB[j][0], RB[j][1], 
                  RC[i][j][0], RC[i][j][1]);
      }
    }

    CP_ASYNC_WAIT_GROUP(K_STAGE-2);
    __syncthreads(); 
  }

  // make sure all memory issues ready.
  if ((K_STAGE - 2) > 0) {
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads(); 
  }
  
  // processing last (K_STAGE-1) k iters.
  {
    #pragma unroll
    for (int k = 0; k < (K_STAGE - 1); k++) {
      uint32_t RA[WARP_TILE_M][4];
      uint32_t RB[WARP_TILE_N][2];

      int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
      // ldmatrix for s_a, ldmatrix.trans for s_b.
      // smem -> reg
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
        int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
        int lane_smem_a_k = (lane_id / 16) * 8; // 0,8
        uint32_t lane_smem_a_ptr = (
          smem_a_base_ptr + (stage_sel * s_a_stage_offset + 
                             lane_smem_a_m * (BK + A_PAD) + 
                             lane_smem_a_k) * sizeof(half)
        );
        LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], lane_smem_a_ptr);
      }

      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
        int lane_smem_b_n = warp_smem_b_n + lane_id % 8; // 0~7, MMA_N=8
        int lane_smem_b_k = ((lane_id / 8) % 2) * 8; // 0,8
        uint32_t lane_smem_b_ptr = (
          smem_b_base_ptr + (stage_sel * s_b_stage_offset + 
                            lane_smem_b_n * (BK + B_PAD) + 
                            lane_smem_b_k) * sizeof(half)
        );
        LDMATRIX_X2(RB[j][0], RB[j][1], lane_smem_b_ptr);
      }

      // MMA compute
      #pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          HMMA16816(RC[i][j][0], RC[i][j][1], 
                    RA[i][0], RA[i][1], RA[i][2], RA[i][3], 
                    RB[j][0], RB[j][1], 
                    RC[i][j][0], RC[i][j][1]);
        }
      }
    }
  }

  {
    for (int i = 0; i < WARP_TILE_M; ++i) {
      // How to use LDST128BITS here? __shfl_sync -> lane 0 -> store 8 half.
      // thus, we only need 8 memory issues with 128 bits after shfl_sync.
      // may reuse RA[4][4] as RC0 ? only new RC1[4][4].
      uint32_t RC0[WARP_TILE_N][4];
      uint32_t RC1[WARP_TILE_N][4];
      #pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // How to use LDST128BITS here? __shfl_sync -> lane 0 -> store 8 half.
        // thus, we only need 8 memory issues with 128 bits after shfl_sync.
        RC0[j][0] = RC[i][j][0];
        RC1[j][0] = RC[i][j][1];
        RC0[j][1] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 1);
        RC0[j][2] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 2);
        RC0[j][3] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 3);
        RC1[j][1] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 1);
        RC1[j][2] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 2);
        RC1[j][3] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 3);
      }

      if (lane_id % 4 == 0) {
        int store_warp_smem_c_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
        int store_lane_gmem_c_m = by * BM + store_warp_smem_c_m + lane_id / 4;
        #pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          int store_warp_smem_c_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
          int store_lane_gmem_c_n = bx * BN + store_warp_smem_c_n;
          int store_gmem_c_addr_0 = store_lane_gmem_c_m * N + store_lane_gmem_c_n;
          int store_gmem_c_addr_1 = (store_lane_gmem_c_m + 8) * N + store_lane_gmem_c_n;
          LDST128BITS(C[store_gmem_c_addr_0]) = LDST128BITS(RC0[j][0]); 
          LDST128BITS(C[store_gmem_c_addr_1]) = LDST128BITS(RC1[j][0]); 
        }
      }
    }
  }
}

















template <typename InputType>
__global__ void
sgemm_tensorcore_naive_kernel(int num_rows_a, int num_cols_b, int num_cols_a,
                              float alpha, const InputType *matrix_a,
                              const InputType *matrix_b, float beta,
                              float *matrix_c) {
    // Each warp computes one 16x16 WMMA tile of the output
    const size_t warp_row = blockIdx.y * WMMA_M;
    const size_t warp_col = blockIdx.x * WMMA_N;

    // Accumulator fragment for output (FP32)
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    const size_t K_tiles = (num_cols_a + WMMA_K - 1) / WMMA_K;

    for (size_t k = 0; k < K_tiles; ++k) {
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, InputType, nvcuda::wmma::row_major> a_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, InputType, nvcuda::wmma::row_major> b_frag;

        const InputType *a_ptr = matrix_a + warp_row * num_cols_a + k * WMMA_K;
        nvcuda::wmma::load_matrix_sync(a_frag, a_ptr, num_cols_a);

        const InputType *b_ptr = matrix_b + k * WMMA_K * num_cols_b + warp_col;
        nvcuda::wmma::load_matrix_sync(b_frag, b_ptr, num_cols_b);

        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Load current C tile for beta scaling
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_load_frag;
    float *c_ptr = matrix_c + warp_row * num_cols_b + warp_col;
    nvcuda::wmma::load_matrix_sync(c_load_frag, c_ptr, num_cols_b, nvcuda::wmma::mem_row_major);

    // Perform C = alpha * C_computed + beta * C_original
#pragma unroll
    for (int t = 0; t < c_frag.num_elements; ++t) {
        c_frag.x[t] = alpha * c_frag.x[t] + beta * c_load_frag.x[t];
    }

    // Store result
    nvcuda::wmma::store_matrix_sync(c_ptr, c_frag, num_cols_b, nvcuda::wmma::mem_row_major);
}

template <typename InputType>
void sgemm_tensorcore_naive_impl(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                                 torch::Tensor &output_matrix, float alpha, float beta,
                                 torch::ScalarType expected_dtype) {
    TORCH_CHECK(matrix_a.device().is_cuda() && matrix_b.device().is_cuda(), "Matrices must be on CUDA device");
    TORCH_CHECK(matrix_a.dtype() == expected_dtype && matrix_b.dtype() == expected_dtype, "Input dtype mismatch");
    TORCH_CHECK(output_matrix.dtype() == torch::kFloat32, "Matrix C must be float32");
    TORCH_CHECK(matrix_a.dim() == 2 && matrix_b.dim() == 2, "Matrices must be 2D");

    const int num_rows_a = static_cast<int>(matrix_a.size(0));
    const int num_cols_a = static_cast<int>(matrix_a.size(1));
    const int num_cols_b = static_cast<int>(matrix_b.size(1));

    TORCH_CHECK(matrix_b.size(0) == num_cols_a && output_matrix.size(0) == num_rows_a && output_matrix.size(1) == num_cols_b,
                "Matrix dimensions must match");

    const InputType *d_matrix_a = reinterpret_cast<const InputType *>(matrix_a.data_ptr());
    const InputType *d_matrix_b = reinterpret_cast<const InputType *>(matrix_b.data_ptr());
    float *d_output_matrix = output_matrix.data_ptr<float>();

    // Each block is a single warp (32 threads)
    // Grid dimensions: (num_cols_b / WMMA_N, num_rows_a / WMMA_M)
    dim3 grid_dim(ceil_div(num_cols_b, WMMA_N), ceil_div(num_rows_a, WMMA_M));
    dim3 block_dim(32); // Single warp per block

    sgemm_tensorcore_naive_kernel<InputType><<<grid_dim, block_dim>>>(
        num_rows_a, num_cols_b, num_cols_a,
        alpha, d_matrix_a, d_matrix_b, beta, d_output_matrix);
}

void sgemm_tensorcore_naive_fp16(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                                 torch::Tensor &output_matrix, float alpha, float beta) {
    sgemm_tensorcore_naive_impl<half>(matrix_a, matrix_b, output_matrix, alpha, beta, torch::kFloat16);
}

void sgemm_tensorcore_naive_bf16(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                                 torch::Tensor &output_matrix, float alpha, float beta) {
    sgemm_tensorcore_naive_impl<nv_bfloat16>(matrix_a, matrix_b, output_matrix, alpha, beta, torch::kBFloat16);
}

// --------------------- PyTorch bindings for custom kernel -----------------------
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
    m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                       \
    if (((T).options().dtype() != (th_type))) {                    \
        std::cout << "Tensor Info:" << (T).options() << std::endl; \
        throw std::runtime_error("values must be " #th_type);      \
    }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                \
    if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {  \
        throw std::runtime_error("Tensor size mismatch!"); \
    }

// only 1 warp per block(32 threads), m16n8k16. A, B, C: all row_major.
// ============================================================================
// 函数: hgemm_mma_m16n8k16_naive
// 功能: PyTorch绑定的基础半精度矩阵乘法
// 用法:
//   - 输入: a(M×K), b(K×N), c(M×N) 都是torch.half类型
//   - 输出: c = a × b (原地更新)
//   - 调用方式: hgemm_mma_m16n8k16_naive(a, b, c)
// 注意事项:
//   - a, b, c必须是torch.half类型
//   - a的列数必须等于b的行数
//   - c的行数必须等于a的行数，列数必须等于b的列数
//   - 使用基础的naive kernel，每个block只包含1个warp
// 示例:
//   torch::Tensor a = torch::randn({M, K}, torch::kHalf).cuda();
//   torch::Tensor b = torch::randn({K, N}, torch::kHalf).cuda();
//   torch::Tensor c = torch::zeros({M, N}, torch::kHalf).cuda();
//   hgemm_mma_m16n8k16_naive(a, b, c);
// ============================================================================
void hgemm_mma_m16n8k16_naive(
    torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)
    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 16;

    dim3 block(WARP_SIZE);
    dim3 grid(div_ceil(N, MMA_N), div_ceil(M, MMA_M));

    hgemm_mma_m16n8k16_naive_kernel<
        MMA_M, MMA_N, MMA_K><<<grid, block>>>(
        reinterpret_cast<half *>(a.data_ptr()),
        reinterpret_cast<half *>(b.data_ptr()),
        reinterpret_cast<half *>(c.data_ptr()),
        M, N, K);
}

// 128x128, mma2x4, warp4x4(64,32,16)
// ============================================================================
// 函数: hgemm_mma_m16n8k16_mma2x4_warp4x4
// 功能: PyTorch绑定的优化半精度矩阵乘法
// 用法:
//   - 输入: a(M×K), b(K×N), c(M×N) 都是torch.half类型
//   - 输出: c = a × b (原地更新)
//   - 调用方式: hgemm_mma_m16n8k16_mma2x4_warp4x4(a, b, c)
// 特点:
//   - 使用优化的kernel，每个block包含8个warp（256线程）
//   - 每个block计算128x128的输出块
//   - 使用共享内存padding避免bank冲突
//   - 性能比naive版本更高
// 注意事项:
//   - a, b, c必须是torch.half类型
//   - a的列数必须等于b的行数
//   - c的行数必须等于a的行数，列数必须等于b的列数
// 示例:
//   torch::Tensor a = torch::randn({M, K}, torch::kHalf).cuda();
//   torch::Tensor b = torch::randn({K, N}, torch::kHalf).cuda();
//   torch::Tensor c = torch::zeros({M, N}, torch::kHalf).cuda();
//   hgemm_mma_m16n8k16_mma2x4_warp4x4(a, b, c);
// ============================================================================
void hgemm_mma_m16n8k16_mma2x4_warp4x4(
    torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)
    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 16;
    constexpr int MMA_TILE_M = 2;
    constexpr int MMA_TILE_N = 4;
    constexpr int WARP_TILE_M = 4;
    constexpr int WARP_TILE_N = 4;
    constexpr int A_PAD = 0;
    constexpr int B_PAD = 16;
    constexpr int NUM_THREADS = (MMA_TILE_M * MMA_TILE_N * WARP_SIZE); // 2 * 4 * 32 = 256

    dim3 block(NUM_THREADS);
    dim3 grid(div_ceil(N, MMA_N * MMA_TILE_N * WARP_TILE_N),
              div_ceil(M, MMA_M * MMA_TILE_M * WARP_TILE_M));

    hgemm_mma_m16n8k16_mma2x4_warp4x4_kernel<
        MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N,
        WARP_TILE_M, WARP_TILE_N, A_PAD, B_PAD><<<grid, block>>>(
        reinterpret_cast<half *>(a.data_ptr()),
        reinterpret_cast<half *>(b.data_ptr()),
        reinterpret_cast<half *>(c.data_ptr()),
        M, N, K);
}


// 128x128, mma2x4, warp4x4(64,32,16), stages, block swizzle, dsmem, TN
#define LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4_DSMEM_TN_KERNEL(stages, stride)   \
{                                                                                   \
  const int smem_max_size = (                                                       \
    (stages) * BM * (BK + A_PAD) * sizeof(half) +                                   \
    (stages) * BN * (BK + B_PAD) * sizeof(half));                                   \
  cudaFuncSetAttribute(                                                             \
    hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn_kernel<                       \
      MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N,                                  \
      WARP_TILE_M, WARP_TILE_N, A_PAD, B_PAD, (stages), true>,                      \
    cudaFuncAttributeMaxDynamicSharedMemorySize,                                    \
    98304);                                                                         \
  const int N_SWIZZLE = (N + (stride) - 1) / (stride);                              \
  dim3 block(NUM_THREADS);                                                          \
  dim3 grid((div_ceil(N, BN) + N_SWIZZLE - 1) / N_SWIZZLE,                          \
             div_ceil(M, BM),                                                       \
             N_SWIZZLE);                                                            \
  hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn_kernel<                         \
    MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N,                                    \
    WARP_TILE_M, WARP_TILE_N, A_PAD, B_PAD, (stages), true><<<                      \
    grid, block, smem_max_size>>>(                                                  \
    reinterpret_cast<half*>(a.data_ptr()),                                          \
    reinterpret_cast<half*>(b.data_ptr()),                                          \
    reinterpret_cast<half*>(c.data_ptr()),                                          \
    M, N, K                                                                         \
  );                                                                                \
}

#define LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4_DSMEM_TN_KERNEL(stages)  \
{                                                                             \
  const int smem_max_size = (                                                 \
    (stages) * BM * (BK + A_PAD) * sizeof(half) +                             \
    (stages) * BN * (BK + B_PAD) * sizeof(half));                             \
  cudaFuncSetAttribute(                                                       \
    hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn_kernel<                 \
      MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N,                            \
      WARP_TILE_M, WARP_TILE_N, A_PAD, B_PAD, (stages), false>,               \
    cudaFuncAttributeMaxDynamicSharedMemorySize,                              \
    98304);                                                                   \
  dim3 block(NUM_THREADS);                                                    \
  dim3 grid(div_ceil(N, BN), div_ceil(M, BM));                                \
  hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn_kernel<                   \
    MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N,                              \
    WARP_TILE_M, WARP_TILE_N, A_PAD, B_PAD, (stages), false><<<               \
    grid, block, smem_max_size>>>(                                            \
    reinterpret_cast<half*>(a.data_ptr()),                                    \
    reinterpret_cast<half*>(b.data_ptr()),                                    \
    reinterpret_cast<half*>(c.data_ptr()),                                    \
    M, N, K                                                                   \
  );                                                                          \
}

// 128x128, mma2x4, warp4x4(64,32,16), stages, block swizzle, dsmem
void hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn(
  torch::Tensor a, torch::Tensor b, torch::Tensor c, 
  int stages, bool swizzle, int swizzle_stride) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1); 
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int MMA_K = 16;
  constexpr int MMA_TILE_M = 2;
  constexpr int MMA_TILE_N = 4; 
  constexpr int WARP_TILE_M = 4;
  constexpr int WARP_TILE_N = 4;
  constexpr int A_PAD = 0; // 0,8,16
  constexpr int B_PAD = 0; // 0,8,16
  constexpr int NUM_THREADS= (
    MMA_TILE_M * MMA_TILE_N * WARP_SIZE); // 2 * 4 * 32 = 256
  constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M;    
  constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N;    
  constexpr int BK = MMA_K;   
    
  if (swizzle) {
    // assert(swizzle_stride % 256 == 0);
    switch (stages)
    {
    case 2: 
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4_DSMEM_TN_KERNEL(2, swizzle_stride);
      break;
    case 3: 
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4_DSMEM_TN_KERNEL(3, swizzle_stride);
      break;
    case 4: 
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4_DSMEM_TN_KERNEL(4, swizzle_stride);
      break;
    case 5: 
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4_DSMEM_TN_KERNEL(5, swizzle_stride);
      break;
    default:
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4_DSMEM_TN_KERNEL(2, swizzle_stride);
      break;
    }
  } else {
    switch (stages)
    {
    case 2:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4_DSMEM_TN_KERNEL(2);
      break;
    case 3:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4_DSMEM_TN_KERNEL(3);
      break;
    case 4:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4_DSMEM_TN_KERNEL(4);
      break;
    case 5:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4_DSMEM_TN_KERNEL(5);
      break;
    default:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4_DSMEM_TN_KERNEL(2);
      break;
    }
  }
}

