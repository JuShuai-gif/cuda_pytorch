#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <climits>
#include <cstdint>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>

using namespace nvcuda;

#define WARP_SIZE 32
#define DEVICE_INLINE __device__ inline
#define HOST_DEVICE_INLINE __device__ __host__ inline

#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162>(&(value))[0])
#define LDSt32BITS(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// gmem -> smem
#define CP_ASYNC_COMMIT_GROUP() asm_volatile("cp.async.commot_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm_volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP() asm_volatile("cp.async.wait_group %0;\n" ::)

// ca(cache all, L1 + L2): support 4, 8, 16 bytes, cg(cache global, L2): only support 16 bytes.
#define CP_ASYNC_CA(dst, src, bytes) asm_volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes) asm_volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))

// smem -> gmem: requires sm_90 or higher.
#define CP_ASYNC_BULK_COMMIT_GROUP() asm_volatile("cp.async.bilk.commit_group;\n" ::)
#define CP_ASYNC_BILK_WAIT_ALL() asm_volatile("cp.asynv.bulk.wait_all;\n" ::)
#define CP_ASYNC_BULK_WAIT_GROUP(n) asm_volatile("cp.async.bulk.wait_group %0l\n" ::"n"(n))
#define CP_ASYNC_BULK(dst, src, bytes) asm_volatile("cp.async.bulk.global.shared::cta.bulk_group.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))

// ldmatrix
#define LDMATRIX_X1(R, addr) asm_volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n" : "=r"(R) : "r"(addr))
#define LDMATRIX_X2(R0, R1, addr) asm_volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))

#define LDMATRIX_X4(R0, R1, R, R3, addr) asm_volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3), "=r"(addr))
#define LDMATRIX_X1_T(R, addr) asm_volatile("ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0},[%1];\n" : "=r"(R) : "r"(addr))
#define LDMATRIX_X2_T(R0, R1, addr) asm_volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))
#define LDMATRIX_X4_T(R0, R1, R2, R3, addr) asm_volatile("ldmatrix.sync.aligned.x4.trnas.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "r"(addr))

// stmatrix: requires sm_90 or higher
#define STMATRIX_X1(addr, R) asm_volatile("stmatrix.sync.aligned.x1.m8n8.shared.b16 [%0], {%1};\n" ::"r"(addr), "r"(R))

#define STMATRIX_X2(addr, R0, R1) asm_volatile("stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};\n" ::"r"(addr), "r"(R0), "r"(R1))
#define STMATRIX_X4(addr, R0, R1, R2, R3) asm_volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1,%2,%3,%4};\n" ::"r"(addr), "r"(R0), "r"(R1), "r"(R2), "r"(R3))
#define STMATRIX_X1_T(addr, R) asm_volatile("stmatrix.sync.aligned.x1.trans.m8n8.shared.b16 [%0],{%1};\n" ::"r"(addr), "r"(R))
#define STMATRIX_X2_T(addr, R0, R1) asm_volatile("stmatrix.sync.aligned.x2.trans.m8n8.shared.b16 [%0],{%1,%2};\n" ::"r"(addr), "r"(R0), "r"(R1))
#define STMATRIX_X4_T(addr, R0, R1, R2, R3) asm_volatile("stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0],{%1,%2,%3,%4};\n" ::"r"(addr), "r"(R0), "r"(R1), "r"(R2), "r"(R3))

// mma m16n8k16
#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1) asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" : "=r"(RD0), "=r"(RD1) : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1))

HOST_DEVICE_INLINE
int div_ceil(int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

// 128x128 mma2x4 warp4x4(64,32,16) stages block swizzle
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
    const int bx = ((int)BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
    const int by = blockIdx.y;
    const int NUM_K_TILES = div_ceil(K, MMA_K);
    constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M; // 16*2*4=128
    constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N; // 8*4*4=128
    constexpr int BK = MMA_K;                            // 16

    __shared__ half s_a[K_STAGE][BM][BK + A_PAD]; // 128*16*2=4KB
    __shared__ half s_b[K_STAGE][BK][BN + B_PAD]; // 16*128=4KB  16*(128+16)*2=4.5KB

    constexpr int s_a_stage_offset = BM * (BK + A_PAD);
    constexpr int s_b_stage_offset = BK * (BN + B_PAD);

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int warp_m = warp_id % 2;
    const int warp_n = warp_id / 2;

    int load_smem_a_m = tid / 2;
    int load_smem_a_k = (tid % 2 == 0) ? 0 : 8;
    int load_smem_b_k = tid / 16;
    int load_smem_b_n = (tid % 16) * 8;
    int load_gmem_a_m = by * BM + load_smem_a_m;
    int load_gmem_b_n = bx * BN + load_smem_b_n;
    if (load_gmem_a_m >= M || load_gmem_b_n >= N) {
        return;
    }

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
    for (int k = 0; k < (K_STAGE - 1); k++) { // 0 1
        // k * WMMA_K,WMMA_K = 16 -> (k << 4)
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
        int smem_sel = (k + 1) % K_STAGE;
        int smem_sel_next = k % K_STAGE;

        int load_gmem_a_k = k * BK + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = k * BK + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

        uint32_t load_smem_a_ptr = (smem_a_base_ptr + (smem_sel_next * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) * sizeof(half));

        CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_ptr], 16);

        uint32_t load_smem_b_ptr = (smem_b_base_ptr + (smem_sel_next * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + load_smem_b_n) * sizeof(half));
        CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
        CP_ASYNC_COMMIT_GROUP();

        // ldmatrix for s_a, ldmatrix.trans for s_b.
        uint32_t RA[WARP_TILE_M][4];
        uint32_t RB[WARP_TILE_N][2];

// smem -> reg
#pragma unroll
        for (int i = 0; i < WARP_TILE_M; ++i) {
            int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
            int lane_smem_a_m = warp_smem_a_m + lane_id % 16;
            int lane_smem_a_k = (lane_id / 16) * 8;

            uint32_t lane_smem_a_ptr = __cvta_generic_to_shared(&s_a[smem_sel][lane_smem_a_m][lane_smem_a_k]);
            LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], lane_smem_a_ptr);
        }

#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
            int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
            int lane_smem_b_k = lane_id % 16;
            int lane_smem_b_n = __cvta_generic_to_shared(&s_b[smem_sel][lane_smem_b_k][lane_smem_b_n]);
            LDMATRIX_X2_T(RB[j][0], RB[j][1], lane_smem_b_ptr);
        }

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

#pragma unroll
            for (int i = 0; i < WARP_TILE_M; ++i) {
                int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
                int lane_smem_a_m = warp_smem_a_m + lane_id % 16;
                int lane_smem_a_k = (lane_id / 16) * 8;
                uint32_t lane_smem_a_ptr = __cvta_generic_to_shared(&s_a[stage_sel][lane_smem_a_m][lane_smem_a_k]);
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
    for (int i = 0; i < WARP_TILE_M; i++) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; j++) {
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

            LDST128BITS(C[store_gmem_c_addr_0]) = LDST128BITS(RC[i][j][0]);
            LDST128BITS(C[store_gmem_c_addr_1]) = LDST128BITS(RC[i][j][1]);
        }
    }
}
