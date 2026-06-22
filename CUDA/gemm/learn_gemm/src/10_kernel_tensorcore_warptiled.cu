#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <cstdio>
#include <float.h>
#include <vector>
#include <cublas_v2.h>
#include <algorithm>
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
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
// ca(cache all, L1 + L2): support 4, 8, 16 bytes, cg(cache global, L2): only support 16 bytes.
#define CP_ASYNC_CA(dst, src, bytes) asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes) asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))
// Support A and B matrix with row-major inorder to compare with the kernels using CUDA Cores in
// hgemm.cu and hgemm_async.cu.

HOST_DEVICE_INLINE
int div_ceil(int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

// Block tiling: BM = WARP_ROW_TILES * BLOCK_ROW_WARPS * WMMA_M (default 256)
//               BN = WARP_COL_TILES * BLOCK_COL_WARPS * WMMA_N (default 128)
// Warp tiling: Each warp computes WARP_ROW_TILES x WARP_COL_TILES WMMA tiles (default 4x2)
template <typename InputType,
          const int BLOCK_ROW_WARPS = 4,
          const int BLOCK_COL_WARPS = 4,
          const int WARP_ROW_TILES = 4,
          const int WARP_COL_TILES = 2,
          const int WMMA_M = 16,
          const int WMMA_N = 16,
          const int WMMA_K = 16>
__global__ void
sgemm_tensorcore_warptiled_kernel(int num_cols_b, int num_cols_a,
                                  float alpha, const InputType *matrix_a,
                                  const InputType *matrix_b, float beta,
                                  float *matrix_c) {
    const uint warp_id = threadIdx.x / 32;
    const uint warp_row = warp_id / BLOCK_COL_WARPS;
    const uint warp_col = warp_id % BLOCK_COL_WARPS;

    constexpr int BLOCK_ROW_TILES = WARP_ROW_TILES * BLOCK_ROW_WARPS;
    constexpr int BLOCK_COL_TILES = WARP_COL_TILES * BLOCK_COL_WARPS;

    constexpr int BM = BLOCK_ROW_TILES * WMMA_M;
    constexpr int BN = BLOCK_COL_TILES * WMMA_N;
    constexpr int BK = WMMA_K;

    // Shared memory: tile_a (BM x BK, row-major), tile_b (BK x BN, column-major)
    __shared__ InputType tile_a[BM * BK];
    __shared__ InputType tile_b[BK * BN];

    const InputType *global_a = matrix_a;
    const InputType *global_b = matrix_b;
    float *global_c = matrix_c;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, InputType, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, InputType, nvcuda::wmma::col_major> b_frag;

    // Accumulator fragments (FP32): each warp maintains WARP_ROW_TILES x WARP_COL_TILES tiles
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag[WARP_ROW_TILES][WARP_COL_TILES];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
#pragma unroll
    for (int i = 0; i < WARP_ROW_TILES; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_COL_TILES; ++j) {
            nvcuda::wmma::fill_fragment(acc_frag[i][j], 0.0f);
        }
    }

    constexpr int NUM_THREADS = BLOCK_ROW_WARPS * BLOCK_COL_WARPS * 32; // warps per block * threads per warp

    // K-loop: iterate by BK, load A and B tiles, compute WMMA operations
    for (int block_k_idx = 0; block_k_idx < num_cols_a; block_k_idx += BK) {
        // Load A tile (BM x BK, row-major)
        // TODO: Vectorize loads if possible - getting numerics issues
        for (int idx = threadIdx.x; idx < BM * BK; idx += NUM_THREADS) {
            int row = idx / BK;
            int col = idx % BK;
            int global_row = blockIdx.y * BM + row;
            int global_col = block_k_idx + col;
            tile_a[row * BK + col] = global_a[global_row * num_cols_a + global_col];
        }

        // Load B tile (BK x BN, column-major for WMMA)
        // TODO: Vectorize loads if possible - getting numerics issues
        for (int idx = threadIdx.x; idx < BK * BN; idx += NUM_THREADS) {
            int row = idx / BN;
            int col = idx % BN;
            int global_row = block_k_idx + row;
            int global_col = blockIdx.x * BN + col;
            tile_b[col * BK + row] = global_b[global_row * num_cols_b + global_col];
        }

        __syncthreads();

        // Warp-level tiling: each warp computes WARP_ROW_TILES x WARP_COL_TILES WMMA tiles
#pragma unroll
        for (int i = 0; i < WARP_ROW_TILES; ++i) {
#pragma unroll
            for (int j = 0; j < WARP_COL_TILES; ++j) {
                int a_tile_row = warp_row * WARP_ROW_TILES + i;
                int b_tile_col = warp_col * WARP_COL_TILES + j;

                InputType const *a_tile_ptr = tile_a + (a_tile_row * WMMA_M) * BK;
                InputType const *b_tile_ptr = tile_b + (b_tile_col * WMMA_N) * BK;

                nvcuda::wmma::load_matrix_sync(a_frag, a_tile_ptr, BK);
                nvcuda::wmma::load_matrix_sync(b_frag, b_tile_ptr, BK);
                nvcuda::wmma::mma_sync(acc_frag[i][j], a_frag, b_frag, acc_frag[i][j]);
            }
        }

        __syncthreads();
    }

    // Store results: C = alpha * (A * B) + beta * C
#pragma unroll
    for (int i = 0; i < WARP_ROW_TILES; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_COL_TILES; ++j) {
            int c_tile_row = warp_row * WARP_ROW_TILES + i;
            int c_tile_col = warp_col * WARP_COL_TILES + j;

            int global_row = blockIdx.y * BM + c_tile_row * WMMA_M;
            int global_col = blockIdx.x * BN + c_tile_col * WMMA_N;

            float *c_ptr = global_c + global_row * num_cols_b + global_col;

            nvcuda::wmma::load_matrix_sync(c_frag, c_ptr, num_cols_b, nvcuda::wmma::mem_row_major);

#pragma unroll
            for (int t = 0; t < c_frag.num_elements; ++t) {
                c_frag.x[t] = alpha * acc_frag[i][j].x[t] + beta * c_frag.x[t];
            }

            nvcuda::wmma::store_matrix_sync(c_ptr, c_frag, num_cols_b, nvcuda::wmma::mem_row_major);
        }
    }
}

// only 1 warp per block(32 threads), m16n16k16. A, B, C: all row_major.
template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16>
__global__ void hgemm_wmma_m16n16k16_naive_kernel(half *A, half *B, half *C,
                                                  int M, int N, int K) {
    const int NUM_K_TILES = div_ceil(K, WMMA_K);
    const int load_gmem_a_m = blockIdx.y * WMMA_M;
    const int load_gmem_b_n = blockIdx.x * WMMA_N;
    if (load_gmem_a_m >= M && load_gmem_b_n >= N) return;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag;
    wmma::fill_fragment(C_frag, 0.0);

#pragma unroll
    for (int k = 0; k < NUM_K_TILES; ++k) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> B_frag;

        wmma::load_matrix_sync(A_frag, A + load_gmem_a_m * K + k * WMMA_K, K);
        wmma::load_matrix_sync(B_frag, B + (k * WMMA_K) * N + load_gmem_b_n, N);

        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);

        __syncthreads();
    }
    wmma::store_matrix_sync(C + load_gmem_a_m * N + load_gmem_b_n, C_frag, N,
                            wmma::mem_row_major);
}

// m16n16k16 wmma  + tile MMA with smem,  A, B, C: all row_major.
template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16,
          const int WMMA_TILE_M = 4, const int WMMA_TILE_N = 2>
__global__ void hgemm_wmma_m16n16k16_mma4x2_kernel(
    half *A, half *B, half *C, int M, int N, int K) {
    // 256 threads(8 warps) per block.
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int NUM_K_TILES = div_ceil(K, WMMA_K);
    constexpr int BM = WMMA_M * WMMA_TILE_M;      // 16x4=64
    constexpr int BN = WMMA_N * WMMA_TILE_N;      // 16x2=32
    constexpr int BK = WMMA_K;                    // 16
    __shared__ half s_a[BM][BK], s_b[WMMA_K][BN]; // 64x16x2=2KB, 16x32x2=1KB

    // 要保证相同的warp下thread执行相同的指令
    // warp_id 0 -> warp_m 0, warp_n 0
    // warp_id 1 -> warp_m 0, warp_n 1
    // warp_id 2 -> warp_m 1, warp_n 0
    // warp_id 3 -> warp_m 1, warp_n 1
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
    const int lane_id = tid % WARP_SIZE; // 0~31
    const int warp_m = warp_id / 2;      // 0,1,2,3
    const int warp_n = warp_id % 2;      // 0,1

    // 256线程分别load s_a=64x16, s_b=16x32
    // 64*16/256=4, half4, 16x32/256=2, half2
    // s_a, 64*16, 每个线程load 4 half, 每行需要4线程，64行，共256线程
    const int load_smem_a_m = tid / 4;       // 0~63
    const int load_smem_a_k = (tid % 4) * 4; // 0,4,12,...
    // s_b, 16x32, 每个线程load 2 half, 每行需要8线程，32行，共256线程
    const int load_smem_b_k = tid / 16;                // 0~16
    const int load_smem_b_n = (tid % 16) * 2;          // 0,2,4,...,32
    const int load_gmem_a_m = by * BM + load_smem_a_m; // global m
    const int load_gmem_b_n = bx * BN + load_smem_b_n; // global n

    if (load_gmem_a_m >= M && load_gmem_b_n >= N) return;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag;
    wmma::fill_fragment(C_frag, 0.0);

#pragma unroll
    for (int k = 0; k < NUM_K_TILES; ++k) {
        int load_gmem_a_k = k * WMMA_K + load_smem_a_k; // global col of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = k * WMMA_K + load_smem_b_k; // global row of b
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        // 64 bits sync memory issues gmem_a -> smem_a.
        LDST64BITS(s_a[load_smem_a_m][load_smem_a_k]) = (LDST64BITS(A[load_gmem_a_addr]));
        // 32 bits sync memory issues gmem_b -> smem_b.
        LDST32BITS(s_b[load_smem_b_k][load_smem_b_n]) = (LDST32BITS(B[load_gmem_b_addr]));
        __syncthreads();

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> B_frag;

        wmma::load_matrix_sync(A_frag, &s_a[warp_m * WMMA_M][0], BK); // BM*BK, BK=WMMA_K
        wmma::load_matrix_sync(B_frag, &s_b[0][warp_n * WMMA_N], BN); // BK=BN, BK=WMMA_K

        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);

        __syncthreads();
    }

    const int store_gmem_a_m = by * BM + warp_m * WMMA_M;
    const int store_gmem_a_n = bx * BN + warp_n * WMMA_N;
    wmma::store_matrix_sync(C + store_gmem_a_m * N + store_gmem_a_n, C_frag, N,
                            wmma::mem_row_major);
}

// m16n16k16 wmma  + tile MMA with smem,  A, B, C: all row_major.
template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16,
          const int WMMA_TILE_M = 4, const int WMMA_TILE_N = 2,
          const int WARP_TILE_M = 2, const int WARP_TILE_N = 4>
__global__ void hgemm_wmma_m16n16k16_mma4x2_warp2x4_kernel(
    half *A, half *B, half *C, int M, int N, int K) {
    // 256 threads(8 warps) per block.
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int NUM_K_TILES = div_ceil(K, WMMA_K);
    constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; // 16x4*2=128
    constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 16x2*4=128
    constexpr int BK = WMMA_K;                             // 16
    __shared__ half s_a[BM][BK], s_b[BK][BN];              // 16x128x2=4KB

    // 要保证相同的warp下thread执行相同的指令
    // warp_id 0 -> warp_m 0, warp_n 0
    // warp_id 1 -> warp_m 0, warp_n 1
    // warp_id 2 -> warp_m 1, warp_n 0
    // warp_id 3 -> warp_m 1, warp_n 1
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
    const int lane_id = tid % WARP_SIZE; // 0~31
    const int warp_m = warp_id / 2;      // 0,1,2,3
    const int warp_n = warp_id % 2;      // 0,1

    // 0. 先计算shared memory中的索引
    // tid和需要加载的smem s_a[BM][BK] 之间的索引关系 BM=128 BK=8 按行读取 A行主序
    // 对于s_a每行16个数据，每个线程读取8个，需要2个线程；总共128行，需要128x2刚好256线程
    int load_smem_a_m = tid / 2;                // row 0~127
    int load_smem_a_k = (tid % 2 == 0) ? 0 : 8; // col 0,8
    // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=16 BN=128 按行读取 B行主序
    // 对于s_b每行128个数据，每个线程读8个数据，需要16个线程；总共16行，需要16x16=256个线程
    int load_smem_b_k = tid / 16;       // row 0~15
    int load_smem_b_n = (tid % 16) * 8; // col 0,8,...,120
    // 1. 再计算全局内存中的索引
    // 要加载到s_a中的元素对应到A全局内存中的行数 每个block负责出C中大小为BM*BN的块
    int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
    int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
    if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

    wmma::fragment<wmma::accumulator,
                   WMMA_M, WMMA_N, WMMA_K,
                   half>
        C_frag[WARP_TILE_M][WARP_TILE_N];

#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            wmma::fill_fragment(C_frag[i][j], 0.0);
        }
    }

#pragma unroll
    for (int k = 0; k < NUM_K_TILES; ++k) {
        int load_gmem_a_k = k * WMMA_K + load_smem_a_k; // global col of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = k * WMMA_K + load_smem_b_k; // global row of b
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        LDST128BITS(s_b[load_smem_b_k][load_smem_b_n]) = (LDST128BITS(B[load_gmem_b_addr]));
        LDST128BITS(s_a[load_smem_a_m][load_smem_a_k]) = (LDST128BITS(A[load_gmem_a_addr]));
        __syncthreads();

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                       wmma::row_major>
            A_frag[WARP_TILE_M];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                       wmma::row_major>
            B_frag[WARP_TILE_N];

#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
            // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
            const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
            wmma::load_matrix_sync(A_frag[i], &s_a[warp_smem_a_m][0], BK); // BM*BK, BK=WMMA_K
        }

#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
            const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
            wmma::load_matrix_sync(B_frag[j], &s_b[0][warp_smem_b_n], BN); // BM*BK, BK=WMMA_K
        }

#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j) {
                wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            const int store_gmem_a_m = by * BM + warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
            const int store_gmem_a_n = bx * BN + warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
            wmma::store_matrix_sync(C + store_gmem_a_m * N + store_gmem_a_n, C_frag[i][j], N,
                                    wmma::mem_row_major);
        }
    }
}

// Double buffers
template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16,
          const int WMMA_TILE_M = 4, const int WMMA_TILE_N = 2,
          const int WARP_TILE_M = 2, const int WARP_TILE_N = 4,
          const int OFFSET = 0>
__global__ void hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async_kernel(
    half *A, half *B, half *C, int M, int N, int K) {
    // 256 threads(8 warps) per block.
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int NUM_K_TILES = div_ceil(K, WMMA_K);
    constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; // 16x4*2=128
    constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 16x2*4=128
    constexpr int BK = WMMA_K;                             // 16
    // 16x128x2=4KB, 4+4=8KB, padding to reduce bank conflicts.
    __shared__ half s_a[2][BM][BK + OFFSET], s_b[2][BK][BN + OFFSET];

    // 要保证相同的warp下thread执行相同的指令
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
    const int lane_id = tid % WARP_SIZE; // 0~31
    const int warp_m = warp_id / 2;      // 0,1,2,3
    const int warp_n = warp_id % 2;      // 0,1

    // 0. 先计算shared memory中的索引
    // tid和需要加载的smem s_a[BM][BK] 之间的索引关系 BM=128 BK=8 按行读取 A行主序
    // 对于s_a每行16个数据，每个线程读取8个，需要2个线程；总共128行，需要128x2刚好256线程
    int load_smem_a_m = tid / 2;                // row 0~127
    int load_smem_a_k = (tid % 2 == 0) ? 0 : 8; // col 0,8
    // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=16 BN=128 按行读取 B行主序
    // 对于s_b每行128个数据，每个线程读8个数据，需要16个线程；总共16行，需要16x16=256个线程
    int load_smem_b_k = tid / 16;       // row 0~15
    int load_smem_b_n = (tid % 16) * 8; // col 0,8,...,120
    // 1. 再计算全局内存中的索引
    // 要加载到s_a中的元素对应到A全局内存中的行数 每个block负责出C中大小为BM*BN的块
    int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
    int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
    if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

    wmma::fragment<wmma::accumulator,
                   WMMA_M, WMMA_N, WMMA_K,
                   half>
        C_frag[WARP_TILE_M][WARP_TILE_N];

#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            wmma::fill_fragment(C_frag[i][j], 0.0);
        }
    }

    // k = 0 is loading here, buffer 0
    {
        int load_gmem_a_k = load_smem_a_k; // global col of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = load_smem_b_k; // global row of b
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

        uint32_t load_smem_a_ptr = __cvta_generic_to_shared(
            &s_a[0][load_smem_a_m][load_smem_a_k]);
        CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

        uint32_t load_smem_b_ptr = __cvta_generic_to_shared(
            &s_b[0][load_smem_b_k][load_smem_b_n]);
        CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);

        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP(0);
    }
    __syncthreads();

#pragma unroll
    for (int k = 1; k < NUM_K_TILES; ++k) { // start from 1
        int smem_sel = (k - 1) & 1;         // k 1->0, k 2->1, k 3->0, ...
        int smem_sel_next = k & 1;          // k 1->1, k 2->0, k 3->1, ...

        int load_gmem_a_k = k * WMMA_K + load_smem_a_k; // global col of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = k * WMMA_K + load_smem_b_k; // global row of b
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

        uint32_t load_smem_a_ptr = __cvta_generic_to_shared(
            &s_a[smem_sel_next][load_smem_a_m][load_smem_a_k]);
        CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

        uint32_t load_smem_b_ptr = __cvta_generic_to_shared(
            &s_b[smem_sel_next][load_smem_b_k][load_smem_b_n]);
        CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                       wmma::row_major>
            A_frag[WARP_TILE_M];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                       wmma::row_major>
            B_frag[WARP_TILE_N];

#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
            // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
            const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
            wmma::load_matrix_sync(A_frag[i], &s_a[smem_sel][warp_smem_a_m][0], BK + OFFSET);
        }

#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
            const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
            wmma::load_matrix_sync(B_frag[j], &s_b[smem_sel][0][warp_smem_b_n], BN + OFFSET);
        }

#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j) {
                wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
            }
        }

        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP(0);

        __syncthreads();
    }

    // processing last k tile
    {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                       wmma::row_major>
            A_frag[WARP_TILE_M];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                       wmma::row_major>
            B_frag[WARP_TILE_N];

#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
            // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
            const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
            wmma::load_matrix_sync(A_frag[i], &s_a[1][warp_smem_a_m][0], BK + OFFSET);
        }

#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
            const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
            wmma::load_matrix_sync(B_frag[j], &s_b[1][0][warp_smem_b_n], BN + OFFSET);
        }

#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j) {
                wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
            }
        }
    }

// finally, store back to C matrix.
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            const int store_gmem_a_m = by * BM + warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
            const int store_gmem_a_n = bx * BN + warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
            wmma::store_matrix_sync(C + store_gmem_a_m * N + store_gmem_a_n, C_frag[i][j], N,
                                    wmma::mem_row_major);
        }
    }
}

// m32n8k16/m8n32k16 kernel
template <const int WMMA_M = 32, const int WMMA_N = 8, const int WMMA_K = 16,
          const int WMMA_TILE_M = 2, const int WMMA_TILE_N = 4,
          const int WARP_TILE_M = 2, const int WARP_TILE_N = 4,
          const int OFFSET = 0>
__global__ void hgemm_wmma_m32n8k16_mma2x4_warp2x4_dbuf_async_kernel(
    half *A, half *B, half *C, int M, int N, int K) {
    // 256 threads(8 warps) per block.
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int NUM_K_TILES = div_ceil(K, WMMA_K);
    constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; // 32x2*2=128
    constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 8x4*4=128
    constexpr int BK = WMMA_K;                             // 16
    // 16x128x2=4KB, 4+4=8KB, padding to reduce bank conflicts.
    __shared__ half s_a[2][BM][BK + OFFSET], s_b[2][BK][BN + OFFSET];

    // 要保证相同的warp下thread执行相同的指令
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
    const int lane_id = tid % WARP_SIZE; // 0~31
    const int warp_m = warp_id / 4;      // 0,1
    const int warp_n = warp_id % 4;      // 0,1,2,3

    // 0. 先计算shared memory中的索引
    // tid和需要加载的smem s_a[BM][BK] 之间的索引关系 BM=128 BK=8 按行读取 A行主序
    // 对于s_a每行16个数据，每个线程读取8个，需要2个线程；总共128行，需要128x2刚好256线程
    int load_smem_a_m = tid / 2;                // row 0~127
    int load_smem_a_k = (tid % 2 == 0) ? 0 : 8; // col 0,8
    // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=16 BN=128 按行读取 B行主序
    // 对于s_b每行128个数据，每个线程读8个数据，需要16个线程；总共16行，需要16x16=256个线程
    int load_smem_b_k = tid / 16;       // row 0~15
    int load_smem_b_n = (tid % 16) * 8; // col 0,8,...,120
    // 1. 再计算全局内存中的索引
    // 要加载到s_a中的元素对应到A全局内存中的行数 每个block负责出C中大小为BM*BN的块
    int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
    int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
    if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

    wmma::fragment<wmma::accumulator,
                   WMMA_M, WMMA_N, WMMA_K,
                   half>
        C_frag[WARP_TILE_M][WARP_TILE_N];

#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            wmma::fill_fragment(C_frag[i][j], 0.0);
        }
    }

    // k = 0 is loading here, buffer 0
    {
        int load_gmem_a_k = load_smem_a_k; // global col of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = load_smem_b_k; // global row of b
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

        uint32_t load_smem_a_ptr = __cvta_generic_to_shared(
            &s_a[0][load_smem_a_m][load_smem_a_k]);
        CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

        uint32_t load_smem_b_ptr = __cvta_generic_to_shared(
            &s_b[0][load_smem_b_k][load_smem_b_n]);
        CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);

        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP(0);
    }
    __syncthreads();

#pragma unroll
    for (int k = 1; k < NUM_K_TILES; ++k) { // start from 1
        int smem_sel = (k - 1) & 1;         // k 1->0, k 2->1, k 3->0, ...
        int smem_sel_next = k & 1;          // k 1->1, k 2->0, k 3->1, ...

        int load_gmem_a_k = k * WMMA_K + load_smem_a_k; // global col of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = k * WMMA_K + load_smem_b_k; // global row of b
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

        uint32_t load_smem_a_ptr = __cvta_generic_to_shared(
            &s_a[smem_sel_next][load_smem_a_m][load_smem_a_k]);
        CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

        uint32_t load_smem_b_ptr = __cvta_generic_to_shared(
            &s_b[smem_sel_next][load_smem_b_k][load_smem_b_n]);
        CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                       wmma::row_major>
            A_frag[WARP_TILE_M];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                       wmma::row_major>
            B_frag[WARP_TILE_N];

#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
            // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
            const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
            wmma::load_matrix_sync(A_frag[i], &s_a[smem_sel][warp_smem_a_m][0], BK + OFFSET);
        }

#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
            const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
            wmma::load_matrix_sync(B_frag[j], &s_b[smem_sel][0][warp_smem_b_n], BN + OFFSET);
        }

#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j) {
                wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
            }
        }

        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP(0);

        __syncthreads();
    }

    // processing last k tile
    {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                       wmma::row_major>
            A_frag[WARP_TILE_M];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                       wmma::row_major>
            B_frag[WARP_TILE_N];

#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
            // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
            const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
            wmma::load_matrix_sync(A_frag[i], &s_a[1][warp_smem_a_m][0], BK + OFFSET);
        }

#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
            const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
            wmma::load_matrix_sync(B_frag[j], &s_b[1][0][warp_smem_b_n], BN + OFFSET);
        }

#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j) {
                wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
            }
        }
    }

// finally, store back to C matrix.
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            const int store_gmem_a_m = by * BM + warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
            const int store_gmem_a_n = bx * BN + warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
            wmma::store_matrix_sync(C + store_gmem_a_m * N + store_gmem_a_n, C_frag[i][j], N,
                                    wmma::mem_row_major);
        }
    }
}

// stage2/3/4 (stage2=double buffers+copy async), 128x128, mma4x2, warp2x4(32,64,16)
// 1. When using shared memory exceeds 48 KB, dynamic shared memory needs to be used,
// i.e., declare a block of dynamic shared memory with extern shared half smem[];.
// When calling the kernel, the size of the dynamic shared memory needs to be specified,
// and smem addressing should be used in a one-dimensional array manner.
// 2. Improve L2 Cache locality (Thread Block Swizzle): https://zhuanlan.zhihu.com/p/555339335
// 3. __launch_bounds__: avoid error 'too many resources required for launch'
// reference: https://blog.csdn.net/feng__shuai/article/details/124395023
template <const int WMMA_M = 16,
          const int WMMA_N = 16,
          const int WMMA_K = 16,
          const int WMMA_TILE_M = 4,
          const int WMMA_TILE_N = 2,
          const int WARP_TILE_M = 2,
          const int WARP_TILE_N = 4,
          const int A_PAD = 0,
          const int B_PAD = 0,
          const int K_STAGE = 2,
          const bool BLOCK_SWIZZLE = false>
__global__ void __launch_bounds__(256)
    hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_kernel(
        half *A, half *B, half *C, int M, int N, int K) {
    // 256 threads(8 warps) per block.
    // const int bx = blockIdx.x;
    // BLOCK_SWIZZLE 0/1 control use block swizzle or not.
    const int bx = ((int)BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
    const int by = blockIdx.y;
    const int NUM_K_TILES = div_ceil(K, WMMA_K);
    constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; // 16x4*2=128
    constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 16x2*4=128
    constexpr int BK = WMMA_K;                             // 16
    // s2: 2*128*(16+8)*2=12KB, 2*16*(128+8)*2=8.50KB,  ~21KB
    // s3: 3*128*(16+8)*2=18KB, 3*16*(128+8)*2=12.75KB, ~31KB
    // s4: 4*128*(16+8)*2=24KB, 4*16*(128+8)*2=17KB,    ~41KB
    __shared__ half s_a[K_STAGE][BM][BK + A_PAD], s_b[K_STAGE][BK][BN + B_PAD];
    constexpr int s_a_stage_offset = BM * (BK + A_PAD);
    constexpr int s_b_stage_offset = BK * (BN + B_PAD);

    // 要保证相同的warp下thread执行相同的指令
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / WARP_SIZE; // tid >> 5; // 0~7 warp_id within block
    const int warp_m = warp_id / 2;      // warp_id >> 1; // 0,1,2,3
    const int warp_n = warp_id % 2;      // 0,1

    // 先计算shared memory中的索引
    // tid和需要加载的smem s_a[BM][BK] 之间的索引关系 BM=128 BK=16 按行读取 A行主序
    // 对于s_a每行16个数据，每个线程读取8个，需要2个线程；总共128行，需要128x2刚好256线程
    int load_smem_a_m = tid / 2;                // tid >> 1; // row 0~127
    int load_smem_a_k = (tid % 2 == 0) ? 0 : 8; // col 0,8
    // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=16 BN=128 按行读取 B行主序
    // 对于s_b每行128个数据，每个线程读8个数据，需要16个线程；总共16行，需要16x16=256个线程
    int load_smem_b_k = tid / 16;       // tid >> 4; // row 0~15
    int load_smem_b_n = (tid % 16) * 8; // ((tid & 0xF) << 3); // col 0,8,...,120
    // 再计算全局内存中的索引
    // 要加载到s_a中的元素对应到A全局内存中的行数 每个block负责出C中大小为BM*BN的块
    int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
    int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
    if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>
        C_frag[WARP_TILE_M][WARP_TILE_N];

#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            wmma::fill_fragment(C_frag[i][j], 0.0);
        }
    }

    // may avoid cvta overhead ? only cvta smem base ptr once for cp.async.
    uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
    uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

#pragma unroll
    for (int k = 0; k < (K_STAGE - 1); ++k) { // 0, 1
        // k * WMMA_K, WMMA_K=16 -> (k << 4)
        int load_gmem_a_k = k * WMMA_K + load_smem_a_k; // global col of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = k * WMMA_K + load_smem_b_k; // global row of b
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
    for (int k = (K_STAGE - 1); k < NUM_K_TILES; k++) {
        // s2/4 can use bitwise ops but s3 can not, so, we use mod
        // ops for all stages kernel. s2: (k + 1)&1, s4: (k + 1)&3
        // s3: (k + 1) % 3
        int smem_sel = (k + 1) % K_STAGE; // s3 k 2->0, k 3->1, k 4->2...
        int smem_sel_next = k % K_STAGE;  // s3 k 2->2, k 3->0, k 4->1...

        // k * WMMA_K, WMMA_K=16 -> (k << 4)
        int load_gmem_a_k = k * WMMA_K + load_smem_a_k; // global col of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = k * WMMA_K + load_smem_b_k; // global row of b
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

        uint32_t load_smem_a_ptr = (smem_a_base_ptr + (smem_sel_next * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) * sizeof(half));
        CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

        uint32_t load_smem_b_ptr = (smem_b_base_ptr + (smem_sel_next * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + load_smem_b_n) * sizeof(half));
        CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
        CP_ASYNC_COMMIT_GROUP();

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                       wmma::row_major>
            A_frag[WARP_TILE_M];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                       wmma::row_major>
            B_frag[WARP_TILE_N];

// compute stage 0
#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
            // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
            const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
            wmma::load_matrix_sync(A_frag[i], &s_a[smem_sel][warp_smem_a_m][0], BK + A_PAD);
        }

#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
            const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
            wmma::load_matrix_sync(B_frag[j], &s_b[smem_sel][0][warp_smem_b_n], BN + B_PAD);
        }

#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j) {
                wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
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
            const int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                           wmma::row_major>
                A_frag[WARP_TILE_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                           wmma::row_major>
                B_frag[WARP_TILE_N];

#pragma unroll
            for (int i = 0; i < WARP_TILE_M; ++i) {
                // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
                const int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
                wmma::load_matrix_sync(A_frag[i], &s_a[stage_sel][warp_smem_a_m][0], BK + A_PAD);
            }

#pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j) {
                // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
                const int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
                wmma::load_matrix_sync(B_frag[j], &s_b[stage_sel][0][warp_smem_b_n], BN + B_PAD);
            }

#pragma unroll
            for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
                for (int j = 0; j < WARP_TILE_N; ++j) {
                    wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
                }
            }
        }
    }

// finally, store back to C matrix.
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            const int store_gmem_a_m = by * BM + warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
            const int store_gmem_a_n = bx * BN + warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
            wmma::store_matrix_sync(C + store_gmem_a_m * N + store_gmem_a_n, C_frag[i][j], N,
                                    wmma::mem_row_major);
        }
    }
}

// stage2/3/4 (stage2=double buffers+copy async), 128x128,mma4x2, warp2x4(32,64,16)
// 1. When using shared memory exceeds 48 KB, dynamic shared memory needs to be used,
// i.e., declare a block of dynamic shared memory with extern shared half smem[];.
// When calling the kernel, the size of the dynamic shared memory needs to be specified,
// and smem addressing should be used in a one-dimensional array manner.
// 2. Improve L2 Cache locality (Thread Block Swizzle): https://zhuanlan.zhihu.com/p/555339335
// 3. __launch_bounds__: avoid error 'too many resources required for launch'
// reference: https://blog.csdn.net/feng__shuai/article/details/124395023
template <const int WMMA_M = 16,
          const int WMMA_N = 16,
          const int WMMA_K = 16,
          const int WMMA_TILE_M = 4,
          const int WMMA_TILE_N = 2,
          const int WARP_TILE_M = 2,
          const int WARP_TILE_N = 4,
          const int A_PAD = 0,
          const int B_PAD = 0,
          const int K_STAGE = 2,
          const bool BLOCK_SWIZZLE = false>
__global__ void __launch_bounds__(256)
    hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem_kernel(
        half *A, half *B, half *C, int M, int N, int K) {
    // 256 threads(8 warps) per block.
    // const int bx = blockIdx.x;
    // BLOCK_SWIZZLE 0/1 control use block swizzle or not.
    const int bx = ((int)BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
    const int by = blockIdx.y;
    const int NUM_K_TILES = div_ceil(K, WMMA_K);
    constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; // 16x4*2=128
    constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 16x2*4=128
    constexpr int BK = WMMA_K;                             // 16
    // s2: 2*128*(16+8)*2=12KB, 2*16*(128+8)*2=8.50KB,  ~21KB
    // s3: 3*128*(16+8)*2=18KB, 3*16*(128+8)*2=12.75KB, ~31KB
    // s4: 4*128*(16+8)*2=24KB, 4*16*(128+8)*2=17KB,    ~41KB
    // s5: 5*128*(16+8)*2=30KB, 5*16*(128+8)*2=21.25KB, ~52KB > 48KB
    extern __shared__ half smem[];
    half *s_a = smem;
    half *s_b = smem + K_STAGE * BM * (BK + A_PAD);
    constexpr int s_a_stage_offset = BM * (BK + A_PAD);
    constexpr int s_b_stage_offset = BK * (BN + B_PAD);

    // 要保证相同的warp下thread执行相同的指令
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
    const int warp_m = warp_id / 2;      // 0,1,2,3
    const int warp_n = warp_id % 2;      // 0,1

    // 先计算shared memory中的索引
    // tid和需要加载的smem s_a[BM][BK] 之间的索引关系 BM=128 BK=16 按行读取 A行主序
    // 对于s_a每行16个数据，每个线程读取8个，需要2个线程；总共128行，需要128x2刚好256线程
    int load_smem_a_m = tid / 2;                // row 0~127
    int load_smem_a_k = (tid % 2 == 0) ? 0 : 8; // col 0,8
    // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=16 BN=128 按行读取 B行主序
    // 对于s_b每行128个数据，每个线程读8个数据，需要16个线程；总共16行，需要16x16=256个线程
    int load_smem_b_k = tid / 16;       // row 0~15
    int load_smem_b_n = (tid % 16) * 8; // col 0,8,...,120
    // 再计算全局内存中的索引
    // 要加载到s_a中的元素对应到A全局内存中的行数 每个block负责出C中大小为BM*BN的块
    int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
    int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
    if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>
        C_frag[WARP_TILE_M][WARP_TILE_N];

#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            wmma::fill_fragment(C_frag[i][j], 0.0);
        }
    }

    // only cvta smem base ptr once for cp.async.
    uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
    uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

#pragma unroll
    for (int k = 0; k < (K_STAGE - 1); ++k) { // 0, 1
        // k * WMMA_K, WMMA_K=16 -> (k << 4)
        int load_gmem_a_k = k * WMMA_K + load_smem_a_k; // global col of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = k * WMMA_K + load_smem_b_k; // global row of b
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
    for (int k = (K_STAGE - 1); k < NUM_K_TILES; k++) {
        // s2/4 can use bitwise ops but s3 can not, so, we use mod
        // ops for all stages kernel. s2: (k + 1)&1, s4: (k + 1)&3
        // s3: (k + 1) % 3
        int smem_sel = (k + 1) % K_STAGE; // s3 k 2->0, k 3->1, k 4->2...
        int smem_sel_next = k % K_STAGE;  // s3 k 2->2, k 3->0, k 4->1...

        // k * WMMA_K, WMMA_K=16 -> (k << 4)
        int load_gmem_a_k = k * WMMA_K + load_smem_a_k; // global col of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = k * WMMA_K + load_smem_b_k; // global row of b
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

        // load stage 2, k start from 2
        uint32_t load_smem_a_ptr = (smem_a_base_ptr + (smem_sel_next * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) * sizeof(half));
        CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

        uint32_t load_smem_b_ptr = (smem_b_base_ptr + (smem_sel_next * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + load_smem_b_n) * sizeof(half));
        CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
        CP_ASYNC_COMMIT_GROUP();

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                       wmma::row_major>
            A_frag[WARP_TILE_M];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                       wmma::row_major>
            B_frag[WARP_TILE_N];

// compute stage 0
#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
            // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
            int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
            half *load_smem_a_frag_ptr = (s_a + smem_sel * s_a_stage_offset + warp_smem_a_m * (BK + A_PAD)
                                          + 0); // BK=WMMA_K=16
            wmma::load_matrix_sync(A_frag[i], load_smem_a_frag_ptr, BK + A_PAD);
        }

#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
            int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
            half *load_smem_b_frag_ptr = (s_b + smem_sel * s_b_stage_offset + 0 * (BN + B_PAD) + warp_smem_b_n); // BK=WMMA_K=16
            wmma::load_matrix_sync(B_frag[j], load_smem_b_frag_ptr, BN + B_PAD);
        }

#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j) {
                wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
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
            const int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                           wmma::row_major>
                A_frag[WARP_TILE_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                           wmma::row_major>
                B_frag[WARP_TILE_N];

#pragma unroll
            for (int i = 0; i < WARP_TILE_M; ++i) {
                // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
                int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
                half *load_smem_a_frag_ptr = (s_a + stage_sel * s_a_stage_offset + warp_smem_a_m * (BK + A_PAD)
                                              + 0); // BK=WMMA_K=16
                wmma::load_matrix_sync(A_frag[i], load_smem_a_frag_ptr, BK + A_PAD);
            }

#pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j) {
                // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
                int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
                half *load_smem_b_frag_ptr = (s_b + stage_sel * s_b_stage_offset + 0 * (BN + B_PAD) + warp_smem_b_n); // BK=WMMA_K=16
                wmma::load_matrix_sync(B_frag[j], load_smem_b_frag_ptr, BN + B_PAD);
            }

#pragma unroll
            for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
                for (int j = 0; j < WARP_TILE_N; ++j) {
                    wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
                }
            }
        }
    }

// finally, store back to C matrix.
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            const int store_gmem_a_m = by * BM + warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
            const int store_gmem_a_n = bx * BN + warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
            wmma::store_matrix_sync(C + store_gmem_a_m * N + store_gmem_a_n, C_frag[i][j], N,
                                    wmma::mem_row_major);
        }
    }
}

// stage with 256x256 block, mma4x4, warp4x4(64,64,16), dynamic smem
// __launch_bounds__: avoid error 'too many resources required for launch'
// reference: https://blog.csdn.net/feng__shuai/article/details/124395023
template <const int WMMA_M = 16,
          const int WMMA_N = 16,
          const int WMMA_K = 16,
          const int WMMA_TILE_M = 4,
          const int WMMA_TILE_N = 4,
          const int WARP_TILE_M = 4,
          const int WARP_TILE_N = 4,
          const int A_PAD = 0,
          const int B_PAD = 0,
          const int K_STAGE = 2,
          const bool BLOCK_SWIZZLE = false>
__global__ void __launch_bounds__(512)
    hgemm_wmma_m16n16k16_mma4x4_warp4x4_stages_dsmem_kernel(
        half *A, half *B, half *C, int M, int N, int K) {
    // 512 threads(16 warps) per block / 256 threads, 8 warps
    // const int bx = blockIdx.x;
    // BLOCK_SWIZZLE 0/1 控制是否使用 block swizzle
    const int bx = ((int)BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
    const int by = blockIdx.y;
    const int NUM_K_TILES = div_ceil(K, WMMA_K);
    constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; // 16x4*4=256
    constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 16x4*4=256
    constexpr int BK = WMMA_K;                             // 16
    extern __shared__ half smem[];
    half *s_a = smem;
    half *s_b = smem + K_STAGE * BM * (BK + A_PAD);
    constexpr int s_a_stage_offset = BM * (BK + A_PAD);
    constexpr int s_b_stage_offset = BK * (BN + B_PAD);

    // 要保证相同的warp下thread执行相同的指令
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / WARP_SIZE; // 0~15 warp_id within block
    const int warp_m = warp_id / 4;      // 0,1,2,3
    const int warp_n = warp_id % 4;      // 0,1,2,3

    // 先计算shared memory中的索引
    // tid和需要加载的smem s_a[BM][BK] 之间的索引关系 BM=256 BK=16 按行读取 A行主序
    // 对于s_a每行16个数据，每个线程读取8个，需要2个线程；总共256行，需要刚好256x2=512线程
    int load_smem_a_m = tid / 2;                // row 0~255
    int load_smem_a_k = (tid % 2 == 0) ? 0 : 8; // col 0, 8
    // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=16 BN=256 按行读取 B行主序
    // 对于s_b每行256个数据，每个线程读8个数据，需要32个线程；总共16行，需要32x16=512个线程
    int load_smem_b_k = tid / 32;       // row 0~15
    int load_smem_b_n = (tid % 32) * 8; // col 0,8,...,256
    // 再计算全局内存中的索引
    // 要加载到s_a中的元素对应到A全局内存中的行数 每个block负责出C中大小为BM*BN的块
    int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
    int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
    if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>
        C_frag[WARP_TILE_M][WARP_TILE_N];

#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            wmma::fill_fragment(C_frag[i][j], 0.0);
        }
    }

    // only cvta smem base ptr once for cp.async.
    uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
    uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

#pragma unroll
    for (int k = 0; k < (K_STAGE - 1); ++k) { // 0, 1
        // k * WMMA_K, WMMA_K=16 -> (k << 4)
        int load_gmem_a_k = k * WMMA_K + load_smem_a_k; // global col of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = k * WMMA_K + load_smem_b_k; // global row of b
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
    for (int k = (K_STAGE - 1); k < NUM_K_TILES; k++) {
        // s2/4 can use bitwise ops but s3 can not, so, we use mod
        // ops for all stages kernel. s2: (k + 1)&1, s4: (k + 1)&3
        // s3: (k + 1) % 3
        int smem_sel = (k + 1) % K_STAGE; // s3 k 2->0, k 3->1, k 4->2...
        int smem_sel_next = k % K_STAGE;  // s3 k 2->2, k 3->0, k 4->1...

        // k * WMMA_K, WMMA_K=16 -> (k << 4)
        int load_gmem_a_k = k * WMMA_K + load_smem_a_k; // global col of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = k * WMMA_K + load_smem_b_k; // global row of b
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

        // load stage 2, k start from 2
        uint32_t load_smem_a_ptr = (smem_a_base_ptr + (smem_sel_next * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) * sizeof(half));
        CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

        uint32_t load_smem_b_ptr = (smem_b_base_ptr + (smem_sel_next * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + load_smem_b_n) * sizeof(half));
        CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
        CP_ASYNC_COMMIT_GROUP();

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                       wmma::row_major>
            A_frag[WARP_TILE_M];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                       wmma::row_major>
            B_frag[WARP_TILE_N];

// compute stage 0
#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
            // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
            int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
            half *load_smem_a_frag_ptr = (s_a + smem_sel * s_a_stage_offset + warp_smem_a_m * (BK + A_PAD)
                                          + 0); // BK=WMMA_K=16
            wmma::load_matrix_sync(A_frag[i], load_smem_a_frag_ptr, BK + A_PAD);
        }

#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
            int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
            half *load_smem_b_frag_ptr = (s_b + smem_sel * s_b_stage_offset + 0 * (BN + B_PAD) + warp_smem_b_n); // BK=WMMA_K=16
            wmma::load_matrix_sync(B_frag[j], load_smem_b_frag_ptr, BN + B_PAD);
        }

#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j) {
                wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
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
            const int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                           wmma::row_major>
                A_frag[WARP_TILE_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                           wmma::row_major>
                B_frag[WARP_TILE_N];

#pragma unroll
            for (int i = 0; i < WARP_TILE_M; ++i) {
                // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
                int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
                half *load_smem_a_frag_ptr = (s_a + stage_sel * s_a_stage_offset + warp_smem_a_m * (BK + A_PAD)
                                              + 0); // BK=WMMA_K=16
                wmma::load_matrix_sync(A_frag[i], load_smem_a_frag_ptr, BK + A_PAD);
            }

#pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j) {
                // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
                int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
                half *load_smem_b_frag_ptr = (s_b + stage_sel * s_b_stage_offset + 0 * (BN + B_PAD) + warp_smem_b_n); // BK=WMMA_K=16
                wmma::load_matrix_sync(B_frag[j], load_smem_b_frag_ptr, BN + B_PAD);
            }

#pragma unroll
            for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
                for (int j = 0; j < WARP_TILE_N; ++j) {
                    wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
                }
            }
        }
    }

// finally, store back to C matrix.
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            const int store_gmem_a_m = by * BM + warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
            const int store_gmem_a_n = bx * BN + warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
            wmma::store_matrix_sync(C + store_gmem_a_m * N + store_gmem_a_n, C_frag[i][j], N,
                                    wmma::mem_row_major);
        }
    }
}

// 256x128, stages, mma4x2, warp4x4(64,64,16)
template <const int WMMA_M = 16,
          const int WMMA_N = 16,
          const int WMMA_K = 16,
          const int WMMA_TILE_M = 4,
          const int WMMA_TILE_N = 2,
          const int WARP_TILE_M = 4,
          const int WARP_TILE_N = 4,
          const int WARP_TILE_K = 1,
          const int A_PAD = 0,
          const int B_PAD = 0,
          const int K_STAGE = 2,
          const bool BLOCK_SWIZZLE = false>
__global__ void __launch_bounds__(256)
    hgemm_wmma_m16n16k16_mma4x2_warp4x4_stages_dsmem_kernel(
        half *A, half *B, half *C, int M, int N, int K) {
    // 256 threads(8 warps) per block.
    // const int bx = blockIdx.x;
    // BLOCK_SWIZZLE 0/1 control use block swizzle or not.
    const int bx = ((int)BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
    const int by = blockIdx.y;
    const int NUM_K_TILES = div_ceil(K, WMMA_K * WARP_TILE_K);
    constexpr int BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M; // 16x4*4=256
    constexpr int BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N; // 16x2*4=128
    constexpr int BK = WMMA_K * WARP_TILE_K;               // 16*2=32
    // s2: 2*128*(32)*2=16KB, 2*32*(128+16)*2=18KB, ~42KB
    // s3: 3*128*(32)*2=24KB, 3*32*(128+16)*2=27KB, ~51KB
    // s4: 4*128*(32)*2=32KB, 4*32*(128+16)*2=36KB, ~68KB
    // s4: 5*128*(32)*2=40KB, 5*32*(128+16)*2=45KB, ~85KB
    extern __shared__ half smem[];
    half *s_a = smem;
    half *s_b = smem + K_STAGE * BM * (BK + A_PAD);
    constexpr int s_a_stage_offset = BM * (BK + A_PAD);
    constexpr int s_b_stage_offset = BK * (BN + B_PAD);

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
    const int warp_m = warp_id / 2;      // 0,1,2,3
    const int warp_n = warp_id % 2;      // 0,1

    // 先计算shared memory中的索引
    // tid和需要加载的smem s_a[BM][BK] 之间的索引关系 BM=256 BK=32 按行读取 A行主序
    // 对于s_a每行16个数据，每个线程读取16个，需要1个线程；总共256行，刚好256线程
    int load_smem_a_m = tid; // row 0~255
    int load_smem_a_k = 0;   // col 0,16
    // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=16 BN=128 按行读取 B行主序
    // 对于s_b每行128个数据，每个线程读8个数据，需要16个线程；总共16行，需要16x16=256个线程
    int load_smem_b_k = tid / 16;       // row 0~15
    int load_smem_b_n = (tid % 16) * 8; // col 0,8,...,120
    // 再计算全局内存中的索引
    // 要加载到s_a中的元素对应到A全局内存中的行数 每个block负责出C中大小为BM*BN的块
    int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
    int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
    if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>
        C_frag[WARP_TILE_M][WARP_TILE_N];

#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            wmma::fill_fragment(C_frag[i][j], 0.0);
        }
    }

    // only cvta smem base ptr once for cp.async.
    uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
    uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

#pragma unroll
    for (int k = 0; k < (K_STAGE - 1); ++k) { // 0, 1
        // k * WMMA_K, WMMA_K=16 -> (k << 4)
        int load_gmem_a_k = k * (WMMA_K * WARP_TILE_K) + load_smem_a_k; // global col of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = k * (WMMA_K * WARP_TILE_K) + load_smem_b_k; // global row of b
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

        uint32_t load_smem_a_ptr = (smem_a_base_ptr + (k * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) * sizeof(half));

        uint32_t load_smem_b_ptr = (smem_b_base_ptr + (k * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + load_smem_b_n) * sizeof(half));

        CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);
        CP_ASYNC_CG(load_smem_a_ptr + 16, &A[load_gmem_a_addr + 8], 16);
        CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);

        CP_ASYNC_COMMIT_GROUP();
    }

    CP_ASYNC_WAIT_GROUP(K_STAGE - 2); // s2->0, s3->1, s4->2
    __syncthreads();

#pragma unroll
    for (int k = (K_STAGE - 1); k < NUM_K_TILES; k++) {
        // s2/4 can use bitwise ops but s3 can not, so, we use mod
        // ops for all stages kernel. s2: (k + 1)&1, s4: (k + 1)&3
        // s3: (k + 1) % 3
        int smem_sel = (k + 1) % K_STAGE; // s3 k 2->0, k 3->1, k 4->2...
        int smem_sel_next = k % K_STAGE;  // s3 k 2->2, k 3->0, k 4->1...

        // k * WMMA_K, WMMA_K=16 -> (k << 4)
        int load_gmem_a_k = k * (WMMA_K * WARP_TILE_K) + load_smem_a_k; // global col of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = k * (WMMA_K * WARP_TILE_K) + load_smem_b_k; // global row of b
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

        // load stage 2, k start from 2
        uint32_t load_smem_a_ptr = (smem_a_base_ptr + (smem_sel_next * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) * sizeof(half));

        uint32_t load_smem_b_ptr = (smem_b_base_ptr + (smem_sel_next * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + load_smem_b_n) * sizeof(half));

        CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);
        CP_ASYNC_CG(load_smem_a_ptr + 16, &A[load_gmem_a_addr + 8], 16);
        CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);

        CP_ASYNC_COMMIT_GROUP();

#pragma unroll
        for (int warp_k = 0; warp_k < WARP_TILE_K; ++warp_k) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                           wmma::row_major>
                A_frag[WARP_TILE_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                           wmma::row_major>
                B_frag[WARP_TILE_N];
            const int warp_smem_k = warp_k * WMMA_K; // 0,16

// compute stage 0
#pragma unroll
            for (int i = 0; i < WARP_TILE_M; ++i) {
                // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
                int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
                half *load_smem_a_frag_ptr = (s_a + smem_sel * s_a_stage_offset + warp_smem_a_m * (BK + A_PAD) + warp_smem_k);
                wmma::load_matrix_sync(A_frag[i], load_smem_a_frag_ptr, BK + A_PAD);
            }

#pragma unroll
            for (int j = 0; j < WARP_TILE_N; ++j) {
                // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
                int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
                half *load_smem_b_frag_ptr = (s_b + smem_sel * s_b_stage_offset + warp_smem_k * (BN + B_PAD) + warp_smem_b_n);
                wmma::load_matrix_sync(B_frag[j], load_smem_b_frag_ptr, BN + B_PAD);
            }

#pragma unroll
            for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
                for (int j = 0; j < WARP_TILE_N; ++j) {
                    wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
                }
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
            const int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);

#pragma unroll
            for (int warp_k = 0; warp_k < WARP_TILE_K; ++warp_k) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                               wmma::row_major>
                    A_frag[WARP_TILE_M];
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                               wmma::row_major>
                    B_frag[WARP_TILE_N];
                const int warp_smem_k = warp_k * WMMA_K; // 0,16

// compute stage 0
#pragma unroll
                for (int i = 0; i < WARP_TILE_M; ++i) {
                    // load 2 tiles -> reg, smem a -> frags a, warp_m 0~3
                    int warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
                    half *load_smem_a_frag_ptr = (s_a + stage_sel * s_a_stage_offset + warp_smem_a_m * (BK + A_PAD) + warp_smem_k);
                    wmma::load_matrix_sync(A_frag[i], load_smem_a_frag_ptr, BK + A_PAD);
                }

#pragma unroll
                for (int j = 0; j < WARP_TILE_N; ++j) {
                    // load 4 tiles -> reg, smem b -> frags b, warp_n 0~2
                    int warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
                    half *load_smem_b_frag_ptr = (s_b + stage_sel * s_b_stage_offset + warp_smem_k * (BN + B_PAD) + warp_smem_b_n);
                    wmma::load_matrix_sync(B_frag[j], load_smem_b_frag_ptr, BN + B_PAD);
                }

#pragma unroll
                for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
                    for (int j = 0; j < WARP_TILE_N; ++j) {
                        wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
                    }
                }
            }
        }
    }

// finally, store back to C matrix.
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            const int store_gmem_a_m = by * BM + warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;
            const int store_gmem_a_n = bx * BN + warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;
            wmma::store_matrix_sync(C + store_gmem_a_m * N + store_gmem_a_n, C_frag[i][j], N,
                                    wmma::mem_row_major);
        }
    }
}

template <typename InputType>
void sgemm_tensorcore_warptiled_impl(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
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

    // Block tiling: 256x128 (4 warps x 4 tiles per warp)
    constexpr int BLOCK_ROW_WARPS = 4, BLOCK_COL_WARPS = 4;
    constexpr int WARP_ROW_TILES = 4, WARP_COL_TILES = 2;
    constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
    constexpr int BM = WARP_ROW_TILES * BLOCK_ROW_WARPS * WMMA_M;
    constexpr int BN = WARP_COL_TILES * BLOCK_COL_WARPS * WMMA_N;

    dim3 grid_dim(ceil_div(num_cols_b, BN), ceil_div(num_rows_a, BM));
    dim3 block_dim(BLOCK_ROW_WARPS * BLOCK_COL_WARPS * 32);

    sgemm_tensorcore_warptiled_kernel<InputType, BLOCK_ROW_WARPS, BLOCK_COL_WARPS, WARP_ROW_TILES, WARP_COL_TILES, WMMA_M, WMMA_N, WMMA_K>
        <<<grid_dim, block_dim>>>(
            num_cols_b, num_cols_a,
            alpha, d_matrix_a, d_matrix_b, beta, d_output_matrix);
}

void sgemm_tensorcore_fp16(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                           torch::Tensor &output_matrix, float alpha, float beta) {
    sgemm_tensorcore_warptiled_impl<half>(matrix_a, matrix_b, output_matrix, alpha, beta, torch::kFloat16);
}

void sgemm_tensorcore_bf16(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                           torch::Tensor &output_matrix, float alpha, float beta) {
    sgemm_tensorcore_warptiled_impl<nv_bfloat16>(matrix_a, matrix_b, output_matrix, alpha, beta, torch::kBFloat16);
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

// 1 warp per block(32 threads), m16n16k16. A, B, C: all row_major.
void hgemm_wmma_m16n16k16_naive(
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
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    dim3 block(WARP_SIZE);
    dim3 grid(div_ceil(N, WMMA_N), div_ceil(M, WMMA_M));

    hgemm_wmma_m16n16k16_naive_kernel<
        WMMA_M, WMMA_N, WMMA_K><<<grid, block>>>(
        reinterpret_cast<half *>(a.data_ptr()),
        reinterpret_cast<half *>(b.data_ptr()),
        reinterpret_cast<half *>(c.data_ptr()),
        M, N, K);
}

void hgemm_wmma_m16n16k16_mma4x2(
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
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int WMMA_TILE_M = 4;
    constexpr int WMMA_TILE_N = 2;
    constexpr int NUM_THREADS = (WMMA_TILE_M * WMMA_TILE_N * WARP_SIZE); // 4 * 2 * 32 = 256

    dim3 block(NUM_THREADS);
    dim3 grid(div_ceil(N, WMMA_N * WMMA_TILE_N),
              div_ceil(M, WMMA_M * WMMA_TILE_M));

    hgemm_wmma_m16n16k16_mma4x2_kernel<
        WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N><<<grid, block>>>(
        reinterpret_cast<half *>(a.data_ptr()),
        reinterpret_cast<half *>(b.data_ptr()),
        reinterpret_cast<half *>(c.data_ptr()),
        M, N, K);
}

void hgemm_wmma_m16n16k16_mma4x2_warp2x4(
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
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int WMMA_TILE_M = 4;
    constexpr int WMMA_TILE_N = 2;
    constexpr int WARP_TILE_M = 2;
    constexpr int WARP_TILE_N = 4;
    constexpr int NUM_THREADS = (WMMA_TILE_M * WMMA_TILE_N * WARP_SIZE); // 4 * 2 * 32 = 256

    dim3 block(NUM_THREADS);
    dim3 grid(div_ceil(N, WMMA_N * WMMA_TILE_N * WARP_TILE_N),
              div_ceil(M, WMMA_M * WMMA_TILE_M * WARP_TILE_M));

    hgemm_wmma_m16n16k16_mma4x2_warp2x4_kernel<
        WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N,
        WARP_TILE_M, WARP_TILE_N><<<grid, block>>>(
        reinterpret_cast<half *>(a.data_ptr()),
        reinterpret_cast<half *>(b.data_ptr()),
        reinterpret_cast<half *>(c.data_ptr()),
        M, N, K);
}

// double buffer, padding
void hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async(
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
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int WMMA_TILE_M = 4;
    constexpr int WMMA_TILE_N = 2;
    constexpr int WARP_TILE_M = 2;
    constexpr int WARP_TILE_N = 4;
    constexpr int NUM_THREADS = (WMMA_TILE_M * WMMA_TILE_N * WARP_SIZE); // 4 * 2 * 32 = 256

    dim3 block(NUM_THREADS);
    dim3 grid(div_ceil(N, WMMA_N * WMMA_TILE_N * WARP_TILE_N),
              div_ceil(M, WMMA_M * WMMA_TILE_M * WARP_TILE_M));

    hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async_kernel<
        WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N,
        WARP_TILE_M, WARP_TILE_N, 8><<<grid, block>>>(
        reinterpret_cast<half *>(a.data_ptr()),
        reinterpret_cast<half *>(b.data_ptr()),
        reinterpret_cast<half *>(c.data_ptr()),
        M, N, K);
}

// m32n8k16
void hgemm_wmma_m32n8k16_mma2x4_warp2x4_dbuf_async(
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
    constexpr int WMMA_M = 32;
    constexpr int WMMA_N = 8;
    constexpr int WMMA_K = 16;
    constexpr int WMMA_TILE_M = 2;
    constexpr int WMMA_TILE_N = 4;
    constexpr int WARP_TILE_M = 2;
    constexpr int WARP_TILE_N = 4;
    constexpr int NUM_THREADS = (WMMA_TILE_M * WMMA_TILE_N * WARP_SIZE); // 2 * 4 * 32 = 256

    dim3 block(NUM_THREADS);
    dim3 grid(div_ceil(N, WMMA_N * WMMA_TILE_N * WARP_TILE_N),
              div_ceil(M, WMMA_M * WMMA_TILE_M * WARP_TILE_M));

    hgemm_wmma_m32n8k16_mma2x4_warp2x4_dbuf_async_kernel<
        WMMA_M, WMMA_N, WMMA_K, WMMA_TILE_M, WMMA_TILE_N,
        WARP_TILE_M, WARP_TILE_N, 8><<<grid, block>>>(
        reinterpret_cast<half *>(a.data_ptr()),
        reinterpret_cast<half *>(b.data_ptr()),
        reinterpret_cast<half *>(c.data_ptr()),
        M, N, K);
}
