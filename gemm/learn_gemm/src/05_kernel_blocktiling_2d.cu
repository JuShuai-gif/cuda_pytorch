#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include "gemm_kernels.cuh"
#include "utils.cuh"

// 将每个线程所拥有的寄存器分配给二维数组
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_blocktiling_2d_kernel(int num_rows_a, int num_cols_b, int num_cols_a,
                                            float alpha, const float *matrix_a,
                                            const float *matrix_b, float beta,
                                            float *matrix_c) {
    const uint block_row = blockIdx.x;
    const uint block_col = blockIdx.y;

    __shared__ float tile_a[BM * BK];
    __shared__ float tile_b[BK * BN];

    const uint thread_row = threadIdx.x / (BN / TN);
    const uint thread_col = threadIdx.x % (BN / TN);
    const uint num_threads = (BM / TM) * (BN / TN);

    matrix_a += block_row * BM * num_cols_a;
    matrix_b += block_col * BN;
    matrix_c += block_row * BM * num_cols_b + block_col * BN;

    float thread_results[TM * TN] = {0.0f};
    float register_m[TM] = {0.0f};
    float register_n[TN] = {0.0f};

    for (uint block_k_idx = 0; block_k_idx < num_cols_a; block_k_idx += BK) {
#pragma unroll
        for (uint load_offset = 0; load_offset < BM * BK; load_offset += num_threads) {
            uint load_idx = threadIdx.x + load_offset;
            uint a_row = load_idx / BK;
            uint a_col = load_idx % BK;
            tile_a[load_idx] = matrix_a[a_row * num_cols_a + a_col];
        }

#pragma unroll
        for (uint load_offset = 0; load_offset < BK * BN; load_offset += num_threads) {
            uint load_idx = threadIdx.x + load_offset;
            uint b_row = load_idx / BN;
            uint b_col = load_idx % BN;
            tile_b[load_idx] = matrix_b[b_row * num_cols_b + b_col];
        }

        __syncthreads();

        matrix_a += BK;
        matrix_b += BK * num_cols_b;

        for (uint dot_idx = 0; dot_idx < BK; ++dot_idx) {
            for (uint i = 0; i < TM; ++i) {
                register_m[i] = tile_a[(thread_row * TM + i) * BK + dot_idx];
            }

            for (uint i = 0; i < TN; ++i) {
                register_n[i] = tile_b[dot_idx * BN + thread_col * TN + i];
            }

            for (uint res_idx_m = 0; res_idx_m < TM; ++res_idx_m) {
                for (uint res_idx_n = 0; res_idx_n < TN; ++res_idx_n) {
                    thread_results[res_idx_m * TN + res_idx_n] +=
                        register_m[res_idx_m] * register_n[res_idx_n];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (uint res_idx_m = 0; res_idx_m < TM; ++res_idx_m) {
#pragma unroll
        for (uint res_idx_n = 0; res_idx_n < TN; ++res_idx_n) {
            const uint c_idx = (thread_row * TM + res_idx_m) * num_cols_b + (thread_col * TN + res_idx_n);
            matrix_c[c_idx] = alpha * thread_results[res_idx_m * TN + res_idx_n] + beta * matrix_c[c_idx];
        }
    }
}

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_blocktiling_2d_edge_kernel(int num_rows_a, int num_cols_b, int num_cols_a,
                                                 float alpha, const float *matrix_a,
                                                 const float *matrix_b, float beta,
                                                 float *matrix_c,
                                                 int block_row_offset, int block_col_offset) {
    const uint block_row = blockIdx.x + block_row_offset;
    const uint block_col = blockIdx.y + block_col_offset;

    __shared__ float tile_a[BM * BK];
    __shared__ float tile_b[BK * BN];

    const uint thread_row = threadIdx.x / (BN / TN);
    const uint thread_col = threadIdx.x % (BN / TN);
    const uint num_threads = (BM / TM) * (BN / TN);

    matrix_a += block_row * BM * num_cols_a;
    matrix_b += block_col * BN;
    matrix_c += block_row * BM * num_cols_b + block_col * BN;

    float thread_results[TM * TN] = {0.0f};
    float register_m[TM] = {0.0f};
    float register_n[TN] = {0.0f};

    for (uint block_k_idx = 0; block_k_idx < num_cols_a; block_k_idx += BK) {
#pragma unroll
        for (uint load_offset = 0; load_offset < BM * BK; load_offset += num_threads) {
            uint load_idx = threadIdx.x + load_offset;
            uint a_row = load_idx / BK;
            uint a_col = load_idx % BK;
            uint global_row_a = block_row * BM + a_row;
            uint global_col_a = block_k_idx + a_col;
            tile_a[load_idx] = (global_row_a < num_rows_a && global_col_a < num_cols_a) ? matrix_a[a_row * num_cols_a + a_col] : 0.0f;
        }

#pragma unroll
        for (uint load_offset = 0; load_offset < BK * BN; load_offset += num_threads) {
            uint load_idx = threadIdx.x + load_offset;
            uint b_row = load_idx / BN;
            uint b_col = load_idx % BN;
            uint global_row_b = block_k_idx + b_row;
            uint global_col_b = block_col * BN + b_col;
            tile_b[load_idx] = (global_row_b < num_cols_a && global_col_b < num_cols_b) ? matrix_b[b_row * num_cols_b + b_col] : 0.0f;
        }

        __syncthreads();

        matrix_a += BK;
        matrix_b += BK * num_cols_b;

        for (uint dot_idx = 0; dot_idx < BK; ++dot_idx) {
            for (uint i = 0; i < TM; ++i) {
                register_m[i] = tile_a[(thread_row * TM + i) * BK + dot_idx];
            }

            for (uint i = 0; i < TN; ++i) {
                register_n[i] = tile_b[dot_idx * BN + thread_col * TN + i];
            }

            for (uint res_idx_m = 0; res_idx_m < TM; ++res_idx_m) {
                for (uint res_idx_n = 0; res_idx_n < TN; ++res_idx_n) {
                    thread_results[res_idx_m * TN + res_idx_n] +=
                        register_m[res_idx_m] * register_n[res_idx_n];
                }
            }
        }

        __syncthreads();
    }

    for (uint res_idx_m = 0; res_idx_m < TM; ++res_idx_m) {
        for (uint res_idx_n = 0; res_idx_n < TN; ++res_idx_n) {
            const uint global_row = block_row * BM + thread_row * TM + res_idx_m;
            const uint global_col = block_col * BN + thread_col * TN + res_idx_n;

            if (global_row < num_rows_a && global_col < num_cols_b) {
                const uint c_idx = (thread_row * TM + res_idx_m) * num_cols_b + (thread_col * TN + res_idx_n);
                matrix_c[c_idx] = alpha * thread_results[res_idx_m * TN + res_idx_n] + beta * matrix_c[c_idx];
            }
        }
    }
}

/*
HGEMM: Block Tile + Thread Tile + K Tile + half2x2, with smem
BK:TILE_K=8 BM=BN=128
TM=TN=8 增加计算密度 BM/TM=16 BN/TN=16
dim3 blockDim(BN/TN, BM/TM);
dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM)
*/
template <const int BM = 128, const int BN = 128, const int BK = 8, const int TM = 8, const int TN = 8>
__global__ void hgemm_t_8x8_sliced_k_f16x4_kernel(half *a, half *b, half *c, int M, int N, int K) {
    // [1]  Block Tile: 一个16x16的block处理C上大小为128X128的一个目标块
    // [2] Thread Tile: 每个thread负责计算TM*TN(8*8)个元素，增加计算密度
    // [3]      K Tile: 将K分块，每块BK大小，迭代(K+BK-1/BK)次，
    //                  每次计算TM*TN个元素各自的部分乘累加
    // [4]   Vectorize: 减少load和store指令，使用half2

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = threadIdx.y * blockDim.x + tx;

    __shared__ half s_a[BM][BK], s_b[BK][BN];

    // 对于s_a每行8个数据，每个线程读取4个，需要2个线程；总共128行，需要128x2刚好256线程
    int load_smem_a_m = tid / 2;                // 行索引
    int load_smem_a_k = (tid % 2 == 0) ? 0 : 4; // 每行的列索引

    // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=8 BN=128 按行读取 B行主序
    // 对于s_b每行128个数据，每个线程读4个数据，需要32个线程；总共8行，需要32x8=256个线程
    int load_smem_b_k = tid / 32;       // 行号
    int load_smem_b_n = (tid % 32) * 4; // 列号

    // 要加载到s_a中的元素对应到A全局内存中的行数 每个block负责出C中大小为BM*BN的块
    int load_gmem_a_m = by * BM + load_smem_a_m; // 竖着
    int load_gmem_b_n = bx * BN + load_smem_b_n; // 横着
    if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

    half r_c[TM][TN] = {__float2half(0.0f)}; // 8x8

    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
        // 加载数据到共享内存smem s_a BM*BK 128*8 vectorize float4
        int load_gmem_a_k = bk * BK + load_smem_a_k; // global col of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        HALF2(s_a[load_smem_a_m][load_smem_a_k + 0]) = HALF2(a[load_gmem_a_addr + 0]);
        HALF2(s_a[load_smem_a_m][load_smem_a_k + 2]) = HALF2(a[load_gmem_a_addr + 2]);
        // 加载数据到共享内存smem s_b BK*BN 8*128 vectorize float4
        int load_gmem_b_k = bk * BK + load_smem_b_k; // global row of b
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        HALF2(s_b[load_smem_b_k][load_smem_b_n + 0]) = HALF2(b[load_gmem_b_addr + 0]);
        HALF2(s_b[load_smem_b_k][load_smem_b_n + 2]) = HALF2(b[load_gmem_b_addr + 2]);
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; k++) {
// 3. 每个线程负责计算BM*BN(12x128)中的TM*TN(8x8)个元素
#pragma unroll
            for (int m = 0; m < TM; m++) {
#pragma unroll
                for (int n = 0; n < TN; n++) {
                    // k from 0~7，0 ~ BK, ty and tx range from 0 to 15, 16x8=128
                    int comp_smem_a_m = ty * TM + m; // 128*8 128/TM(8)=16 M方向 16线程
                    int comp_smem_b_n = tx * TN + n; // 8*128 128/TN(8)=16 N方向 16线程
                    r_c[m][n] += s_a[comp_smem_a_m][k] * s_b[k][comp_smem_b_n];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int m = 0; m < TM; ++m) {
        int store_gmem_c_m = by * BM + ty * TM + m;
#pragma unroll
        for (int n = 0; n < TN; n += 2) {
            int store_gmem_c_n = bx * BN + tx * TN + n;
            int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
            HALF2(c[store_gmem_c_addr]) = HALF2(r_c[m][n]);
        }
    }
}

template <const int BM = 128, const int BN = 128, const int BK = 8, const int TM = 8, const int TN = 8>
__global__ void hgemm_t_8x8_sliced_k_f16x4_pack_kernel(half *a, half *b, half *c, int M, int N, int K) {
    // [1]  Block Tile: 一个16x16的block处理C上大小为128X128的一个目标块
    // [2] Thread Tile: 每个thread负责计算TM*TN(8*8)个元素，增加计算密度
    // [3]      K Tile: 将K分块，每块BK大小，迭代(K+BK-1/BK)次，
    //                  每次计算TM*TN个元素各自的部分乘累加
    // [4]   Vectorize: 减少load和store指令，使用half2

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = threadIdx.y * blockDim.x + tx;

    __shared__ half s_a[BM][BK], s_b[BK][BN];

    // 对于s_a每行8个数据，每个线程读取4个，需要2个线程；总共128行，需要128x2刚好256线程
    int load_smem_a_m = tid / 2;                // 行索引
    int load_smem_a_k = (tid % 2 == 0) ? 0 : 4; // 每行的列索引

    // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=8 BN=128 按行读取 B行主序
    // 对于s_b每行128个数据，每个线程读4个数据，需要32个线程；总共8行，需要32x8=256个线程
    int load_smem_b_k = tid / 32;       // 行号
    int load_smem_b_n = (tid % 32) * 4; // 列号

    // 要加载到s_a中的元素对应到A全局内存中的行数 每个block负责出C中大小为BM*BN的块
    int load_gmem_a_m = by * BM + load_smem_a_m; // 竖着
    int load_gmem_b_n = bx * BN + load_smem_b_n; // 横着
    if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

    half r_c[TM][TN] = {__float2half(0.0f)}; // 8x8

    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
        // 加载数据到共享内存smem s_a BM*BK 128*8 vectorize float4
        int load_gmem_a_k = bk * BK + load_smem_a_k; // global col of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;

        // 数据打包
        LDST64BITS(s_a[load_smem_a_m][load_smem_a_k]) = LDST64BITS(a[load_gmem_a_addr]);

        // 加载数据到共享内存smem s_b BK*BN 8*128 vectorize float4
        int load_gmem_b_k = bk * BK + load_smem_b_k; // global row of b
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

        // 数据打包
        LDST64BITS(s_b[load_smem_b_k][load_smem_b_n]) = LDST64BITS(b[load_gmem_b_addr]);

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; k++) {
// 3. 每个线程负责计算BM*BN(12x128)中的TM*TN(8x8)个元素
#pragma unroll
            for (int m = 0; m < TM; m++) {
#pragma unroll
                for (int n = 0; n < TN; n++) {
                    // k from 0~7，0 ~ BK, ty and tx range from 0 to 15, 16x8=128
                    int comp_smem_a_m = ty * TM + m; // 128*8 128/TM(8)=16 M方向 16线程
                    int comp_smem_b_n = tx * TN + n; // 8*128 128/TN(8)=16 N方向 16线程
                    r_c[m][n] = __hfma(s_a[comp_smem_a_m][k], s_b[k][comp_smem_b_n],
                                       r_c[m][n]); // HFMA(x,y,z)=x*y+z
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int m = 0; m < TM; ++m) {
        int store_gmem_c_m = by * BM + ty * TM + m;
#pragma unroll
        for (int n = 0; n < TN; n += 4) {
            int store_gmem_c_n = bx * BN + tx * TN + n;
            int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
            LDST64BITS(c[store_gmem_c_addr]) = LDST64BITS(r_c[m][n]);
        }
    }
}

// 避免 bank conflict 版本
template <const int BM = 128, const int BN = 128, const int BK = 8, const int TM = 8, const int TN = 8>
__global__ void hgemm_t_8x8_sliced_k_f16x4_bcf_kernel(
    half *a, half *b, half *c, const int M, const int N, const int K) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int tid = ty * blockDim.x + tx;

    __shared__ half s_a[BK][BM];
    __shared__ half s_b[BK][BN];

    // 搬运缓冲区
    half r_load_a[TM / 2];
    half r_load_b[TN / 2];

    // 计算缓冲区
    half r_comp_a[TM];
    half r_comp_b[TN];

    // 累加器
    half r_c[TM][TN] = {__float2half(0.0f)};

    // 每行需要两个线程处理，一个线程加载4 个 数据
    int load_a_smem_m = tid / 2;

    // 第一个线程处理 0-3 个数据，第二个线程处理 4-7 个数据
    int load_a_smem_k = (tid & 1) << 2; // (0,4)

    // 每行需要 8 个线程进行处理 0-7
    int load_b_smem_k = tid / 32;

    // 每行的索引号 (0,4,8,12,...,124)
    int load_b_smem_n = (tid & 31) << 2;

    // 计算全局索引
    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    if (load_a_gmem_m >= M || load_b_gmem_n >= N)
        return;

    for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;

        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    }

    // 加载两个到 r_load_a 中，
    // 0 1
    HALF2(r_load_a[0]) = HALF2(a[load_a_gmem_addr + 0]);
    // 2 3
    HALF2(r_load_a[2]) = HALF2(a[load_a_gmem_addr + 2]);
    // 0 1
    HALF2(r_load_b[0]) = HALF2(b[load_b_gmem_addr + 0]);
    // 2 3
    HALF2(r_load_b[2]) = HALF2(b[load_b_gmem_addr + 2]);

    // r_load_a 中是行优先排列，这也是 A 的内存排列方式
    /*
    现在将 s_a 竖着排列,如果还按之前的 128*8 的格式排列，那样8行才能组成32个bank，这样必定造成4路冲突
    相反，本来B就是128*8,就不会有这个问题了
    */
    s_a[load_a_smem_k][load_a_smem_m] = r_load_a[0];
    s_a[load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
    s_a[load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
    s_a[load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];

    HALF2(s_b[load_b_smem_k][load_b_smem_n + 0]) = HALF2(r_load_b[0]);
    HALF2(s_b[load_b_smem_k][load_b_smem_n + 2]) = HALF2(r_load_b[2]);

    __syncthreads();
    for (int tk = 0; tk < BK; tk++) {
        // BCF读取s_a: ty*TM/2=ty*4, 取值0,4,8,...,60; offset=0,2,64,66错开bank访问
        // 相邻线程(ty差1)访问地址相差2个bank, BM/2=64将上下半区隔离避免冲突
        HALF2(r_comp_a[0]) = HALF2(s_a[tk][ty * TM / 2]);
        HALF2(r_comp_a[2]) = HALF2(s_a[tk][ty * TM / 2 + 2]);
        HALF2(r_comp_a[4]) = HALF2(s_a[tk][ty * TM / 2 + BM / 2]);
        HALF2(r_comp_a[6]) = HALF2(s_a[tk][ty * TM / 2 + BM / 2 + 2]);

        // BCF读取s_b: tx*TN/2=tx*4, 取值0,4,8,...,60; offset=0,2,64,66错开bank访问
        // 原理同上, BN/2=64将左右半区隔离
        HALF2(r_comp_b[0]) = HALF2(s_b[tk][tx * TN / 2]);
        HALF2(r_comp_b[2]) = HALF2(s_b[tk][tx * TN / 2 + 2]);
        HALF2(r_comp_b[4]) = HALF2(s_b[tk][tx * TN / 2 + BN / 2]);
        HALF2(r_comp_b[6]) = HALF2(s_b[tk][tx * TN / 2 + BN / 2 + 2]);

#pragma unroll
        for (int tm = 0; tm < TM; tm++) {
#pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
            }
        }
        __syncthreads();
    }

// 处理 C 矩阵块的四个象限
// 0 1
#pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
        HALF2(c[store_c_gmem_addr + 0]) = HALF2(r_c[i][0]);
        HALF2(c[store_c_gmem_addr + 2]) = HALF2(r_c[i][2]);
        HALF2(c[store_c_gmem_addr + BN / 2 + 0]) = HALF2(r_c[i][4]);
        HALF2(c[store_c_gmem_addr + BN / 2 + 2]) = HALF2(r_c[i][6]);
    }
// 2 3
#pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
        HALF2(c[store_c_gmem_addr + 0]) = HALF2(r_c[i + TM / 2][0]);
        HALF2(c[store_c_gmem_addr + 2]) = HALF2(r_c[i + TM / 2][2]);
        HALF2(c[store_c_gmem_addr + BN / 2 + 0]) = HALF2(r_c[i + TM / 2][4]);
        HALF2(c[store_c_gmem_addr + BN / 2 + 2]) = HALF2(r_c[i + TM / 2][6]);
    }
}

// 避免 bank conflict 版本 + 向量化访问
template <const int BM = 128,
          const int BN = 128,
          const int BK = 8,
          const int TM = 8,
          const int TN = 8,
          const int OFFSET = 0>
__global__ void hgemm_t_8x8_sliced_k_f16x4_pack_bcf_kernel(
    half *a, half *b, half *c, const int M, const int N, const int K) {
    // threads: 128/8 * 128/8 = 256
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ half s_a[BK][BM + OFFSET]; // 8*128*2=2KB
    __shared__ half s_b[BK][BN + OFFSET]; // 8*128*2=2KB

    half r_load_a[TM / 2];                   // 4
    half r_load_b[TN / 2];                   // 4
    half r_comp_a[TM];                       // 8
    half r_comp_b[TN];                       // 8
    half r_c[TM][TN] = {__float2half(0.0f)}; // 8x8

    int load_a_smem_m = tid / 2;        // tid / 2，(0,1,2,...,128)
    int load_a_smem_k = (tid & 1) << 2; // (0,4)

    int load_b_smem_k = tid / 32;        // 0~8
    int load_b_smem_n = (tid & 31) << 2; // (0,4,8,12,...,124)

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    if (load_a_gmem_m >= M || load_b_gmem_n >= N) return;

    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
        LDST64BITS(r_load_a[0]) = LDST64BITS(a[load_a_gmem_addr]);
        LDST64BITS(r_load_b[0]) = LDST64BITS(b[load_b_gmem_addr]);

        s_a[load_a_smem_k][load_a_smem_m] = r_load_a[0];     // e.g layer_0 b0
        s_a[load_a_smem_k + 1][load_a_smem_m] = r_load_a[1]; // e.g layer_2 b0
        s_a[load_a_smem_k + 2][load_a_smem_m] = r_load_a[2]; // e.g layer_4 b0
        s_a[load_a_smem_k + 3][load_a_smem_m] = r_load_a[3]; // e.g layer_6 b0

        LDST64BITS(s_b[load_b_smem_k][load_b_smem_n]) = LDST64BITS(r_load_b[0]);

        __syncthreads();

#pragma unroll
        for (int tk = 0; tk < BK; tk++) {
            LDST64BITS(r_comp_a[0]) = LDST64BITS(s_a[tk][ty * TM / 2]);
            LDST64BITS(r_comp_a[4]) = LDST64BITS(s_a[tk][ty * TM / 2 + BM / 2]);

            LDST64BITS(r_comp_b[0]) = LDST64BITS(s_b[tk][tx * TN / 2]);
            LDST64BITS(r_comp_b[4]) = LDST64BITS(s_b[tk][tx * TN / 2 + BN / 2]);

#pragma unroll
            for (int tm = 0; tm < TM; tm++) {
#pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    // r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
                    r_c[tm][tn] = __hfma(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
        LDST64BITS(c[store_c_gmem_addr]) = LDST64BITS(r_c[i][0]);
        LDST64BITS(c[store_c_gmem_addr + BN / 2]) = LDST64BITS(r_c[i][4]);
    }
#pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
        LDST64BITS(c[store_c_gmem_addr]) = LDST64BITS(r_c[i + TM / 2][0]);
        LDST64BITS(c[store_c_gmem_addr + BN / 2]) = LDST64BITS(r_c[i + TM / 2][4]);
    }
}

template <const int BM = 128,
          const int BN = 128,
          const int BK = 8,
          const int TM = 8,
          const int TN = 8,
          const int OFFSET = 0>
__global__ void hgemm_t_8x8_sliced_k_f16x8_pack_bcf_kernel(
    half *a, half *b, half *c, const int M, const int N, const int K) {
    // threads: 128/8 * 128/8 = 256
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ half s_a[BK][BM + OFFSET]; // 8*128*2=2KB
    __shared__ half s_b[BK][BN + OFFSET]; // 8*128*2=2KB

    half r_load_a[TM / 2];                   // 4
    half r_load_b[TN / 2];                   // 4
    half r_comp_a[TM];                       // 8
    half r_comp_b[TN];                       // 8
    half r_c[TM][TN] = {__float2half(0.0f)}; // 8x8

    // mapping tid to s_a[BK][BM], for each orginal m-th row, load 4 + 4 K-dim
    // row major values from A matrix, and store it in COL major s_a[BK][BM].
    int load_a_smem_m = tid / 2;        // tid / 2，(0,1,2,...,128)
    int load_a_smem_k = (tid & 1) << 2; // (0,4)
    // mapping tid to s_b[BK][BN], for each orginal k-th row, load 4 + 4 N-dim
    // row major values from B matrix, and store it in ROW major s_b[BK][BN].
    int load_b_smem_k = tid / 32;        // 0~8
    int load_b_smem_n = (tid & 31) << 2; // (0,4,8,12,...,124)

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    if (load_a_gmem_m >= M || load_b_gmem_n >= N) return;

    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
        LDST64BITS(r_load_a[0]) = LDST64BITS(a[load_a_gmem_addr]);
        LDST64BITS(r_load_b[0]) = LDST64BITS(b[load_b_gmem_addr]);

        // s_a[8][128] write: 4路 bank conflicts
        s_a[load_a_smem_k][load_a_smem_m] = r_load_a[0];
        s_a[load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        // s_b[8][128] write: 2路 bank conflicts
        LDST64BITS(s_b[load_b_smem_k][load_b_smem_n]) = LDST64BITS(r_load_b[0]);

        __syncthreads();

#pragma unroll
        for (int tk = 0; tk < BK; tk++) {
            // bank conflicts analysis, tx/ty 0~15, 0~7 bank 4*8=32 bytes
            // 进入具体线程后，可以认为该线程对应的值都已经固定了，比如tid, tx, ty.
            // 因此对于这个循环的理解，应该按照tk迭代，tid, tx, ty固定为某个值来理解.
            // 但是分析bank conflicts需要考虑warp内线程的并发行为，因此，应该分析
            // 不同线程在同一个时间点的bank访存情况.
            // s_a[8][128] load: 16路 bank conflicts
            // tid 0~15,  tk 0~7 -> ty 0 -> [0~7][0+0~7]  bank 0~3 layers_0~15
            // tid 16~31, tk 0~7 -> ty 1 -> [0~7][0+8~15] bank 4~7 layers_0~15
            LDST128BITS(r_comp_a[0]) = LDST128BITS(s_a[tk][ty * TM]);
            // s_b[8][128] load: 4路 bank conflicts
            // tid 0, tk 0~7 -> tx 0 -> [0~7][0+0~7]   bank 0~3
            // tid 1, tk 0~7 -> tx 1 -> [0~7][0+8~15]  bank 4~7
            // tid 7, tk 0~7 -> tx 7 -> [0~7][0+56~63] bank 28~31
            // tid 0/8/16/24, tk 0~7 -> tx 0/8/16/24 -> [0~7][0+...] bank 0~3
            LDST128BITS(r_comp_b[0]) = LDST128BITS(s_b[tk][tx * TN]);
            // TODO: 手工实现 swizzle之行列号异或
            // https://zhuanlan.zhihu.com/p/722286440

#pragma unroll
            for (int tm = 0; tm < TM; tm++) {
#pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    // r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
                    r_c[tm][tn] = __hfma(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; i++) {
        int store_c_gmem_m = by * BM + ty * TM + i;
        int store_c_gmem_n = bx * BN + tx * TN;
        int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
        LDST128BITS(c[store_c_gmem_addr]) = LDST128BITS(r_c[i][0]);
    }
}

// 双缓冲
template <const int BM = 128,
          const int BN = 128,
          const int BK = 8,
          const int TM = 8,
          const int TN = 8,
          const int OFFSET = 0>
__global__ void hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf_kernel(
    half *a, half *b, half *c, const int M, const int N, const int K) {
    // threads: 128/8 * 128/8 = 256
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ half s_a[2][BK][BM + OFFSET]; // 8*128*2=2KB
    __shared__ half s_b[2][BK][BN + OFFSET]; // 8*128*2=2KB

    half r_load_a[TM / 2];                   // 4
    half r_load_b[TN / 2];                   // 4
    half r_comp_a[TM];                       // 8
    half r_comp_b[TN];                       // 8
    half r_c[TM][TN] = {__float2half(0.0f)}; // 8x8

    // mapping tid to s_a[BK][BM], for each orginal m-th row, load 4 + 4 K-dim
    // row major values from A matrix, and store it in COL major s_a[BK][BM].
    int load_a_smem_m = tid / 2;        // tid / 2，(0,1,2,...,128)
    int load_a_smem_k = (tid & 1) << 2; // (0,4)
    // mapping tid to s_b[BK][BN], for each orginal k-th row, load 4 + 4 N-dim
    // row major values from B matrix, and store it in ROW major s_b[BK][BN].
    int load_b_smem_k = tid / 32;        // 0~8
    int load_b_smem_n = (tid & 31) << 2; // (0,4,8,12,...,124)

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;
    if (load_a_gmem_m >= M || load_b_gmem_n >= N) return;

    // 1）主循环从bk = 1 开始，第一次数据加载在主循环之前，最后一次计算在主循环之后，这是pipeline 的特点决定的；
    // 2）由于计算和下一次访存使用的Shared Memory不同，因此主循环中每次循环只需要一次__syncthreads()即可
    // 3）由于GPU不能向CPU那样支持乱序执行，主循环中需要先将下一次循环计算需要的Gloabal Memory中的数据load
    // 到寄存器，然后进行本次计算，之后再将load到寄存器中的数据写到Shared Memory，这样在LDG指令向Global
    // Memory做load时，不会影响后续FFMA及其它运算指令的 launch 执行，也就达到了Double Buffering的目的。

    // bk = 0 is loading here, buffer 0
    {
        int load_a_gmem_k = load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
        LDST64BITS(r_load_a[0]) = LDST64BITS(a[load_a_gmem_addr]);
        LDST64BITS(r_load_b[0]) = LDST64BITS(b[load_b_gmem_addr]);

        s_a[0][load_a_smem_k + 0][load_a_smem_m] = r_load_a[0];
        s_a[0][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[0][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[0][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        LDST64BITS(s_b[0][load_b_smem_k][load_b_smem_n]) = LDST64BITS(r_load_b[0]);
    }
    // Without this synchronization, accuracy may occasionally be abnormal.
    __syncthreads();

    // bk start from 1，需要注意的是，虽然 bk 从 1 开始，但实际上 bk=1时，使用的是
    // 第0块BK中的数据（已经加载到共享内存s_a[0]和s_b[0]）；bk=2时，实际计算的是第1块
    // BK中的数据。其余以此类推，这个循环结束后，剩下最后一块BK大小的数据需要计算。
    for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {
        int smem_sel = (bk - 1) & 1; // bk 1->0, bk 2->1, bk 3->0, ...
        int smem_sel_next = bk & 1;  // bk 1->1, bk 2->0, bk 3->1, ...

        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
        LDST64BITS(r_load_a[0]) = LDST64BITS(a[load_a_gmem_addr]);
        LDST64BITS(r_load_b[0]) = LDST64BITS(b[load_b_gmem_addr]);

#pragma unroll
        for (int tk = 0; tk < BK; tk++) {
            LDST128BITS(r_comp_a[0]) = LDST128BITS(s_a[smem_sel][tk][ty * TM]);
            LDST128BITS(r_comp_b[0]) = LDST128BITS(s_b[smem_sel][tk][tx * TN]);

#pragma unroll
            for (int tm = 0; tm < TM; tm++) {
#pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    r_c[tm][tn] = __hfma(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
                }
            }
        }

        // 对比非double buffers版本，此处不需要__syncthreads()，总共节省了
        // ((K + BK - 1) / BK) - 1 次block内的同步操作。比如，bk=1时，HFMA计算
        // 使用的是s_a[0]和s_b[0]，因此，和s_a[1]和s_b[1]的加载是没有依赖关系的。
        // 从global内存到s_a[1]和s_b[1]和HFMA计算可以并行。s_a[1]和s_b[1]用于
        // 加载下一块BK需要的数据到共享内存。
        s_a[smem_sel_next][load_a_smem_k + 0][load_a_smem_m] = r_load_a[0];
        s_a[smem_sel_next][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[smem_sel_next][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[smem_sel_next][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        LDST64BITS(s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]) = LDST64BITS(r_load_b[0]);

        __syncthreads();
    }

// 计算剩下最后一块BK
#pragma unroll
    for (int tk = 0; tk < BK; tk++) {
        LDST128BITS(r_comp_a[0]) = LDST128BITS(s_a[1][tk][ty * TM]);
        LDST128BITS(r_comp_b[0]) = LDST128BITS(s_b[1][tk][tx * TN]);

#pragma unroll
        for (int tm = 0; tm < TM; tm++) {
#pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                r_c[tm][tn] = __hfma(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
            }
        }
    }

#pragma unroll
    for (int i = 0; i < TM; i++) {
        int store_c_gmem_m = by * BM + ty * TM + i;
        int store_c_gmem_n = bx * BN + tx * TN;
        int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
        LDST128BITS(c[store_c_gmem_addr]) = LDST128BITS(r_c[i][0]);
    }
}

void sgemm_blocktiling_2d(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                          torch::Tensor &output_matrix, float alpha, float beta) {
    TORCH_CHECK(matrix_a.device().is_cuda(), "Matrix A must be on CUDA device");
    TORCH_CHECK(matrix_b.device().is_cuda(), "Matrix B must be on CUDA device");
    TORCH_CHECK(matrix_a.dtype() == torch::kFloat32, "Matrix A must be float32");
    TORCH_CHECK(matrix_b.dtype() == torch::kFloat32, "Matrix B must be float32");
    TORCH_CHECK(matrix_a.dim() == 2, "Matrix A must be 2D");
    TORCH_CHECK(matrix_b.dim() == 2, "Matrix B must be 2D");

    const int num_rows_a = static_cast<int>(matrix_a.size(0));
    const int num_cols_a = static_cast<int>(matrix_a.size(1));
    const int num_cols_b = static_cast<int>(matrix_b.size(1));

    TORCH_CHECK(matrix_b.size(0) == num_cols_a, "Matrix dimensions must match: A is MxK, B must be KxN");
    TORCH_CHECK(output_matrix.device().is_cuda(), "Matrix C must be on CUDA device");
    TORCH_CHECK(output_matrix.dtype() == torch::kFloat32, "Matrix C must be float32");
    TORCH_CHECK(output_matrix.size(0) == num_rows_a && output_matrix.size(1) == num_cols_b, "Matrix C must be MxN");

    const float *d_matrix_a = matrix_a.data_ptr<float>();
    const float *d_matrix_b = matrix_b.data_ptr<float>();
    float *d_output_matrix = output_matrix.data_ptr<float>();

    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;

    dim3 block_dim((BM / TM) * (BN / TN));

    const int num_blocks_m = ceil_div(num_rows_a, BM);
    const int num_blocks_n = ceil_div(num_cols_b, BN);
    const int main_blocks_m = num_rows_a / BM;
    const int main_blocks_n = num_cols_b / BN;

    if (main_blocks_m > 0 && main_blocks_n > 0) {
        dim3 main_grid(main_blocks_m, main_blocks_n);
        sgemm_blocktiling_2d_kernel<BM, BN, BK, TM, TN><<<main_grid, block_dim>>>(
            num_rows_a, num_cols_b, num_cols_a,
            alpha, d_matrix_a, d_matrix_b, beta, d_output_matrix);
    }

    if (main_blocks_m > 0 && num_blocks_n > main_blocks_n) {
        dim3 edge_right_grid(main_blocks_m, 1);
        sgemm_blocktiling_2d_edge_kernel<BM, BN, BK, TM, TN><<<edge_right_grid, block_dim>>>(
            num_rows_a, num_cols_b, num_cols_a,
            alpha, d_matrix_a, d_matrix_b, beta, d_output_matrix,
            0, main_blocks_n);
    }

    if (num_blocks_m > main_blocks_m && main_blocks_n > 0) {
        dim3 edge_bottom_grid(1, main_blocks_n);
        sgemm_blocktiling_2d_edge_kernel<BM, BN, BK, TM, TN><<<edge_bottom_grid, block_dim>>>(
            num_rows_a, num_cols_b, num_cols_a,
            alpha, d_matrix_a, d_matrix_b, beta, d_output_matrix,
            main_blocks_m, 0);
    }

    if (num_blocks_m > main_blocks_m && num_blocks_n > main_blocks_n) {
        dim3 edge_corner_grid(1, 1);
        sgemm_blocktiling_2d_edge_kernel<BM, BN, BK, TM, TN><<<edge_corner_grid, block_dim>>>(
            num_rows_a, num_cols_b, num_cols_a,
            alpha, d_matrix_a, d_matrix_b, beta, d_output_matrix,
            main_blocks_m, main_blocks_n);
    }
}

// 宏定义：用于简化PyTorch扩展的绑定
#define STRINGFY(str) #str // 将标识符转换为字符串
#define TORCH_BINDING_COMMON_EXTENSION(func) \
    m.def(STRINGFY(func), &func, STRINGFY(func)); // 绑定函数到PyTorch

// 检查张量数据类型的宏
#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                       \
    if (((T).options().dtype() != (th_type))) {                    \
        std::cout << "Tensor Info:" << (T).options() << std::endl; \
        throw std::runtime_error("values must be " #th_type);      \
    }

// 检查张量形状的宏
#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                \
    if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {  \
        throw std::runtime_error("Tensor size mismatch!"); \
    }

void hgemm_t_8x8_sliced_k_f16x4(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)
    constexpr int BM = 128; // 每个线程块处理C矩阵块的行数
    constexpr int BN = 128; // 每个线程块处理C矩阵块的列数
    constexpr int BK = 8;   // K维度分块大小
    constexpr int TM = 8;   // 每个线程在M方向计算的元素数
    constexpr int TN = 8;   // 每个线程在N方向计算的元素数

    dim3 block(BN / TN, BM / TM); // 线程块维度：BN/TN x BM/TM = 16x16 = 256个线程
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    hgemm_t_8x8_sliced_k_f16x4_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(
        reinterpret_cast<half *>(a.data_ptr()),
        reinterpret_cast<half *>(b.data_ptr()),
        reinterpret_cast<half *>(c.data_ptr()),
        M, N, K);
}

// t 8x8 fp16x4 pack
void hgemm_t_8x8_sliced_k_f16x4_pack(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;

    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    hgemm_t_8x8_sliced_k_f16x4_pack_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(
        reinterpret_cast<half *>(a.data_ptr()),
        reinterpret_cast<half *>(b.data_ptr()),
        reinterpret_cast<half *>(c.data_ptr()),
        M, N, K);
}

// reduce bank conflicts
void hgemm_t_8x8_sliced_k_f16x4_bcf(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;

    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    hgemm_t_8x8_sliced_k_f16x4_bcf_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(
        reinterpret_cast<half *>(a.data_ptr()),
        reinterpret_cast<half *>(b.data_ptr()),
        reinterpret_cast<half *>(c.data_ptr()),
        M, N, K);
}

// reduce bank conflicts, f16x4 pack, t 8x8
void hgemm_t_8x8_sliced_k_f16x4_pack_bcf(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;

    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    hgemm_t_8x8_sliced_k_f16x4_pack_bcf_kernel<BM, BN, BK, TM, TN, 4><<<grid, block>>>(
        reinterpret_cast<half *>(a.data_ptr()),
        reinterpret_cast<half *>(b.data_ptr()),
        reinterpret_cast<half *>(c.data_ptr()),
        M, N, K);
}

// reduce bank conflicts, t 8x8 fp16x8 pack, pad
void hgemm_t_8x8_sliced_k_f16x8_pack_bcf(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;

    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    hgemm_t_8x8_sliced_k_f16x8_pack_bcf_kernel<BM, BN, BK, TM, TN, 8><<<grid, block>>>(
        reinterpret_cast<half *>(a.data_ptr()),
        reinterpret_cast<half *>(b.data_ptr()),
        reinterpret_cast<half *>(c.data_ptr()),
        M, N, K);
}

void hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;

    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf_kernel<BM, BN, BK, TM, TN, 8><<<grid, block>>>(
        reinterpret_cast<half *>(a.data_ptr()),
        reinterpret_cast<half *>(b.data_ptr()),
        reinterpret_cast<half *>(c.data_ptr()),
        M, N, K);
}
