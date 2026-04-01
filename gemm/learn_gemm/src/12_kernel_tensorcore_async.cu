#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <float.h>
#include <vector>
#include <algorithm>

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <torch/types.h>
#include <torch/torch.h>
#include <torch/extension.h>

#include <cooperative_groups.h>
#include <cuda/pipeline>
#include "gemm_kernels.cuh"
#include "utils.cuh"

namespace cg = cooperative_groups;

#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
#define CP_ASYNC_CA(dst, src, bytes) asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes) asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))

template <const int BM = 128, const int BN = 128, const int BK = 16,
          const int TM = 8, const int TN = 8, const int OFFSET = 0>
__global__ void hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_kernel(
    half *a, half *b, half *c, int M, int N, int K) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int tid = threadIdx.y * blockDim.x + tx; // tid within the block

    __shared__ half s_a[2][BK][BM + OFFSET], s_b[2][BK][BN + OFFSET];
    half r_load_a[TM];                       // 8
    half r_load_b[TN];                       // 8
    half r_comp_a[TM];                       // 8
    half r_comp_b[TN];                       // 8
    half r_c[TM][TN] = {__float2half(0.0f)}; // 8x8

    int load_smem_a_m = tid / 2;                // row 0~127
    int load_smem_a_k = (tid % 2 == 0) ? 0 : 8; // col 0,8

    int load_smem_b_k = tid / 16;       // row 0~15
    int load_smem_b_n = (tid % 16) * 8; // col 0,8,...,120

    int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
    int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
    if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

    {
        // 从这个读取过程可以看出，A 是按行存储的
        int load_gmem_a_k = load_smem_a_k; // global col of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;

        // B 也是按行存储的
        int load_gmem_b_k = load_smem_b_k; // global row of b
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

        // 存储方式是同一行，连续列
        // a0 a1 a2 a3 a4 a5 a6 a7
        /*
        n
        k   ---------------------
        0 |  b00 b01 b02 b03 ...
        1 |  b10 b11 b12 b13 ...
        2 |  b20 b21 b22 b23 ...
        然后两个half组成一个 bank

        访问的时候是同一行连续访问，shared 仍然 row-major
        */
        LDST128BITS(s_b[0][load_smem_b_k][load_smem_b_n]) = (LDST128BITS(b[load_gmem_b_addr]));

        // 先一次读 8 个到寄存器缓冲区中，8个连存 [0...7]
        /*
        r_load_a[0] = A[row][col]
        r_load_a[1] = A[row][col+1]
        ...
        r_load_a[7] = A[row][col+7]
        */
        LDST128BITS(r_load_a[0]) = LDST128BITS(a[load_gmem_a_addr]);
// 从寄存器缓冲区读到共享内存中，转置写入
/*
写入是按列写入

*/
#pragma unroll
        for (int i = 0; i < 8; ++i) { // reg -> shared, fast
            s_a[0][load_smem_a_k + i][load_smem_a_m] = r_load_a[i];
        }
    }
    __syncthreads();

    // bk start from 1，需要注意的是，虽然 bk 从 1 开始，但实际上 bk=1时，使用的是
    // 第0块BK中的数据（已经加载到共享内存s_a[0]和s_b[0]）；bk=2时，实际计算的是第1块
    // BK中的数据。其余以此类推，这个循环结束后，剩下最后一块BK大小的数据需要计算。
    for (int bk = 1; bk < (K + BK - 1) / BK; ++bk) {
        int smem_sel = (bk - 1) & 1; // bk 1->0, bk 2->1, bk 3->0, ...
        int smem_sel_next = bk & 1;  // bk 1->1, bk 2->0, bk 3->1, ...

        int load_gmem_a_k = bk * BK + load_smem_a_k; // global col of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;

        int load_gmem_b_k = bk * BK + load_smem_b_k; // global row of b
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

        LDST128BITS(r_load_a[0]) = LDST128BITS(a[load_gmem_a_addr]);
        LDST128BITS(r_load_b[0]) = LDST128BITS(b[load_gmem_b_addr]);

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

#pragma unroll
        for (int i = 0; i < 8; ++i) { // reg -> shared, fast
            s_a[smem_sel_next][load_smem_a_k + i][load_smem_a_m] = r_load_a[i];
        }
        LDST128BITS(s_b[smem_sel_next][load_smem_b_k][load_smem_b_n]) = (LDST128BITS(r_load_b[0]));

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
    for (int i = 0; i < TM; ++i) {
        int store_gmem_c_m = by * BM + ty * TM + i;
        int store_gmem_c_n = bx * BN + tx * TN;
        int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
        LDST128BITS(c[store_gmem_c_addr]) = LDST128BITS(r_c[i][0]);
    }
}

template <const int BM = 128, const int BN = 128, const int BK = 16,
          const int TM = 8, const int TN = 8, const int OFFSET = 0>
__global__ void hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async_kernel(
    half *a, half *b, half *c, int M, int N, int K) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = threadIdx.y * blockDim.x + tx; // tid within the block
    // 2*128*16*2=8KB, 2*16*128*2=8KB
    __shared__ half s_a[2][BK][BM + OFFSET];
    __shared__ half s_b[2][BK][BN + OFFSET];
    half r_load_a[TM];                       // 8
    half r_comp_a[TM];                       // 8
    half r_comp_b[TN];                       // 8
    half r_c[TM][TN] = {__float2half(0.0f)}; // 8x8

    int load_smem_a_m = tid / 2;                 // row 0~127
    int load_smem_a_k = (tid % 2 == 0) ? 0 : 8;  // col 0,8
    int load_smem_b_k = tid / 16;                // row 0~15
    int load_smem_b_n = (tid % 16) * 8;          // col 0,8,...,120
    int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
    int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
    if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

    // bk = 0 is loading here, buffer 0
    {
        int load_gmem_a_k = load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

        // 将共享内存地址转换为异步拷贝操作所需的指针格式
        uint32_t load_smem_b_ptr = __cvta_generic_to_shared(&s_b[0][load_smem_b_k][load_smem_b_n]);

        // 异步从全局内存复制16字节(8个half)到共享内存，不阻塞线程执行
        CP_ASYNC_CG(load_smem_b_ptr, &b[load_gmem_b_addr], 16);

        // 提交异步拷贝操作组，开始执行
        CP_ASYNC_COMMIT_GROUP();

        // load 8 half in 1 memory issue.
        // 从全局内存加载 128 位(8 个half)到寄存器
        LDST128BITS(r_load_a[0]) = LDST128BITS(a[load_gmem_a_addr]);
        /*
        这里不需要 __syncthreads() 因为这两步都是当前线程自己完成的。
        只要当前线程后面马上继续使用自己的寄存器值，把它写到 shared memory，线程内部天然是顺序执行的，不需要 __syncthreads()。

        什么时候需要 __syncthreads()
        需要 barrier 的场景是：别的线程要来读你刚写进 shared memory 的数据。
        */
        // 循环将寄存器数据写入共享内存
#pragma unroll
        for (int i = 0; i < 8; ++i) { // reg -> shared, fast
            s_a[0][load_smem_a_k + i][load_smem_a_m] = r_load_a[i];
        }
        // 等待当前线程发起的所有 cp.async 异步拷贝全部完成
        CP_ASYNC_WAIT_GROUP(0);
    }
    __syncthreads();

    for (int bk = 1; bk < (K + BK - 1) / BK; ++bk) {
        int smem_sel = (bk - 1) & 1; // bk 1->0, bk 2->1, bk 3->0, ...
        int smem_sel_next = bk & 1;  // bk 1->1, bk 2->0, bk 3->1, ...

        int load_gmem_a_k = bk * BK + load_smem_a_k; // global col of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = bk * BK + load_smem_b_k; // global row of b
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

        uint32_t load_smem_b_ptr = __cvta_generic_to_shared(
            &s_b[smem_sel_next][load_smem_b_k][load_smem_b_n]);
        CP_ASYNC_CG(load_smem_b_ptr, &b[load_gmem_b_addr], 16);
        CP_ASYNC_COMMIT_GROUP();

#pragma unroll
        for (int tk = 0; tk < BK; tk++) {
            LDST128BITS(r_comp_a[0]) = LDST128BITS(s_a[smem_sel][tk][ty * TM]);
            LDST128BITS(r_comp_b[0]) = LDST128BITS(s_b[smem_sel][tk][tx * TN]);

#pragma unroll
            for (int tm = 0; tm < TM; tm++) {
#pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    // 计算还是使用的融合乘加操作
                    r_c[tm][tn] = __hfma(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
                }
            }
        }
        LDST128BITS(r_load_a[0]) = LDST128BITS(a[load_gmem_a_addr]);
#pragma unroll
        for (int i = 0; i < 8; ++i) { // reg -> shared, fast
            s_a[smem_sel_next][load_smem_a_k + i][load_smem_a_m] = r_load_a[i];
        }
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
    }

    // 计算最后一个块
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
    for (int i = 0; i < TM; ++i) {
        int store_gmem_c_m = by * BM + ty * TM + i;
        int store_gmem_c_n = bx * BN + tx * TN;
        int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
        LDST128BITS(c[store_gmem_c_addr]) = LDST128BITS(r_c[i][0]);
    }
}

// compare w/o cp.async
template <const int BM = 128, const int BN = 128, const int BK = 32,
          const int TM = 8, const int TN = 8, const int OFFSET = 0>
__global__ void hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_kernel(
    half *a, half *b, half *c, int M, int N, int K) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = threadIdx.y * blockDim.x + tx;

    // 2*128*32*2=16KB, 2*32*128*2=16KB
    __shared__ half s_a[2][BK][BM + OFFSET], s_b[2][BK][BN + OFFSET];

    half r_load_a[16];                       // 16
    half r_comp_a[TM];                       // 8
    half r_comp_b[TN];                       // 8
    half r_c[TM][TN] = {__float2half(0.0f)}; // 8x8

    // 0. 先计算shared memory中的索引
    // tid和需要加载的smem s_a[BM][BK] 之间的索引关系 BM=128 BK=8 按行读取 A行主序
    // 对于s_a每行32个数据，每个线程读取16个，需要2个线程；总共128行，需要128x2刚好256线程
    int load_smem_a_m = tid / 2;                 // row 0~127
    int load_smem_a_k = (tid % 2 == 0) ? 0 : 16; // col 0,16
    // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=32 BN=128 按行读取 B行主序
    // 对于s_b每行128个数据，每个线程读16个数据，需要8个线程；总共32行，需要32x8=256个线程
    int load_smem_b_k = tid / 8;        // row 0~32
    int load_smem_b_n = (tid % 8) * 16; // col 0,16,...,
    // 1. 再计算全局内存中的索引
    // 要加载到s_a中的元素对应到A全局内存中的行数 每个block负责出C中大小为BM*BN的块
    int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
    int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
    if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

    // bk = 0 is loading here, buffer 0
    {
        int load_gmem_a_k = load_smem_a_k; // global col of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = load_smem_b_k; // global row of b
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
// load 16 half per threads
// 每个线程加载 16 个 half
#pragma unroll
        // 分成两次加载,因为现在K方向 16 切分一次
        for (int i = 0; i < 16; i += 8) {
            LDST128BITS(s_b[0][load_smem_b_k][load_smem_b_n + i]) = (LDST128BITS(b[load_gmem_b_addr + i]));
            LDST128BITS(r_load_a[i]) = LDST128BITS(a[load_gmem_a_addr + i]);
        }
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            s_a[0][load_smem_a_k + i][load_smem_a_m] = r_load_a[i];
        }
    }
    __syncthreads();

    for (int bk = 1; bk < (K + BK - 1) / BK; ++bk) {
        int smem_sel = (bk - 1) & 1; // bk 1->0, bk 2->1, bk 3->0, ...
        int smem_sel_next = bk & 1;  // bk 1->1, bk 2->0, bk 3->1, ...

        int load_gmem_a_k = bk * BK + load_smem_a_k; // global col of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = bk * BK + load_smem_b_k; // global row of b
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

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
// load 16 half per threads
#pragma unroll
        for (int i = 0; i < 16; i += 8) {
            LDST128BITS(s_b[smem_sel_next][load_smem_b_k][load_smem_b_n + i]) = (LDST128BITS(b[load_gmem_b_addr + i]));
            LDST128BITS(r_load_a[i]) = LDST128BITS(a[load_gmem_a_addr + i]);
        }
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            s_a[smem_sel_next][load_smem_a_k + i][load_smem_a_m] = r_load_a[i];
        }
        __syncthreads();
    }

    // 处理最后一个块

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
    for (int i = 0; i < TM; ++i) {
        int store_gmem_c_m = by * BM + ty * TM + i;
        int store_gmem_c_n = bx * BN + tx * TN;
        int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
        LDST128BITS(c[store_gmem_c_addr]) = LDST128BITS(r_c[i][0]);
    }
}

template <const int BM = 128, const int BN = 128, const int BK = 32,
          const int TM = 8, const int TN = 8, const int OFFSET = 0>
__global__ void hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_async_kernel(
    half *a, half *b, half *c, int M, int N, int K) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = threadIdx.y * blockDim.x + tx;

    // 2*128*32*2=16KB, 2*32*128*2=16KB
    __shared__ half s_a[2][BK][BM + OFFSET], s_b[2][BK][BN + OFFSET];

    half r_load_a[16];                       // 16
    half r_comp_a[TM];                       // 8
    half r_comp_b[TN];                       // 8
    half r_c[TM][TN] = {__float2half(0.0f)}; // 8x8

    // 0. 先计算shared memory中的索引
    // tid和需要加载的smem s_a[BM][BK] 之间的索引关系 BM=128 BK=8 按行读取 A行主序
    // 对于s_a每行32个数据，每个线程读取16个，需要2个线程；总共128行，需要128x2刚好256线程
    int load_smem_a_m = tid / 2;                 // row 0~127
    int load_smem_a_k = (tid % 2 == 0) ? 0 : 16; // col 0,16
    // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=32 BN=128 按行读取 B行主序
    // 对于s_b每行128个数据，每个线程读16个数据，需要8个线程；总共32行，需要32x8=256个线程
    int load_smem_b_k = tid / 8;        // row 0~32
    int load_smem_b_n = (tid % 8) * 16; // col 0,16,...,
    // 1. 再计算全局内存中的索引
    // 要加载到s_a中的元素对应到A全局内存中的行数 每个block负责出C中大小为BM*BN的块
    int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
    int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
    if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

    // bk = 0 is loading here, buffer 0
    {
        int load_gmem_a_k = load_smem_a_k; // global col of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = load_smem_b_k; // global row of b
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

        uint32_t load_smem_b_ptr = __cvta_generic_to_shared(&s_b[0][load_smem_b_k][load_smem_b_n]);

#pragma unroll
        // 也是加载两次
        for (int i = 0; i < 16; i += 8) {
            CP_ASYNC_CA(load_smem_b_ptr + i * 2, &b[load_gmem_b_addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();

#pragma unroll
        for (int i = 0; i < 16; i += 8) {
            LDST128BITS(r_load_a[i]) = LDST128BITS(a[load_gmem_a_addr + i]);
        }

#pragma unroll
        for (int i = 0; i < 16; ++i) {
            s_a[0][load_smem_a_k + i][load_smem_a_m] = r_load_a[i];
        }
        CP_ASYNC_WAIT_GROUP(0);
    }
    __syncthreads();

    for (int bk = 1; bk < (K + BK - 1) / BK; ++bk) {
        int smem_sel = (bk - 1) & 1; // bk 1->0, bk 2->1, bk 3->0, ...
        int smem_sel_next = bk & 1;  // bk 1->1, bk 2->0, bk 3->1, ...

        int load_gmem_a_k = bk * BK + load_smem_a_k; // global col of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = bk * BK + load_smem_b_k; // global row of b
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

        uint32_t load_smem_b_ptr = __cvta_generic_to_shared(
            &s_b[smem_sel_next][load_smem_b_k][load_smem_b_n]);
#pragma unroll
        for (int i = 0; i < 16; i += 8) {
            CP_ASYNC_CA(load_smem_b_ptr + i * 2, &b[load_gmem_b_addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();

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
#pragma unroll
        for (int i = 0; i < 16; i += 8) {
            LDST128BITS(r_load_a[i]) = LDST128BITS(a[load_gmem_a_addr + i]);
        }
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            s_a[smem_sel_next][load_smem_a_k + i][load_smem_a_m] = r_load_a[i];
        }

        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
    }

    // 处理最后一个块

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
    // 写回结果
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        int store_gmem_c_m = by * BM + ty * TM + i;
        int store_gmem_c_n = bx * BN + tx * TN;
        int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
        LDST128BITS(c[store_gmem_c_addr]) = LDST128BITS(r_c[i][0]);
    }
}




// t 16x8, 128x128, k 32, 8x16=128 threads per block, w/o cp.async
template <const int BM = 128, const int BN = 128, const int BK = 32,
          const int TM = 16, const int TN = 8, const int OFFSET = 0>
__global__ void hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf_kernel(
    half *a, half *b, half *c, int M, int N, int K) {
    // block(BN/TN, BM/TM) -> (x=16, y=8)
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;                    // 0~15
    int ty = threadIdx.y;                    // 0~7
    int tid = threadIdx.y * blockDim.x + tx; // 0~127
    // 2*128*32*2=16KB, 2*32*128*2=16KB
    __shared__ half s_a[2][BK][BM + OFFSET], s_b[2][BK][BN + OFFSET];
    half r_load_a[32];                       // 32
    half r_comp_a[TM];                       // 16
    half r_comp_b[TN];                       // 8
    half r_c[TM][TN] = {__float2half(0.0f)}; // 16x8

    // 0. 先计算shared memory中的索引
    // tid和需要加载的smem s_a[BM][BK] 之间的索引关系 BM=128 BK=32 按行读取 A行主序
    // 对于s_a每行32个数据，每个线程读取32个，需要1个线程；总共128行，需要128x1刚好128线程
    int load_smem_a_m = tid; // row 0~127
    int load_smem_a_k = 0;   // col 0
    // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=32 BN=128 按行读取 B行主序
    // 对于s_b每行128个数据，每个线程读32个数据，需要4个线程；总共32行，需要32x4=128个线程
    int load_smem_b_k = tid / 4;        // row 0~32, 128/4
    int load_smem_b_n = (tid % 4) * 32; // col 0,32,64,...
    // 1. 再计算全局内存中的索引
    // 要加载到s_a中的元素对应到A全局内存中的行数 每个block负责出C中大小为BM*BN的块
    int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
    int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
    if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

    // bk = 0 is loading here, buffer 0
    {
        int load_gmem_a_k = load_smem_a_k; // global col of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = load_smem_b_k; // global row of b
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
// load 32(BK) half per threads, 4x128bits memory issues.
#pragma unroll
        // 一次加载 32 个
        for (int i = 0; i < 32; i += 8) {
            LDST128BITS(s_b[0][load_smem_b_k][load_smem_b_n + i]) = (LDST128BITS(b[load_gmem_b_addr + i]));
            LDST128BITS(r_load_a[i]) = LDST128BITS(a[load_gmem_a_addr + i]);
        }
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            s_a[0][load_smem_a_k + i][load_smem_a_m] = r_load_a[i];
        }
    }
    __syncthreads();

    for (int bk = 1; bk < (K + BK - 1) / BK; ++bk) {
        int smem_sel = (bk - 1) & 1; // bk 1->0, bk 2->1, bk 3->0, ...
        int smem_sel_next = bk & 1;  // bk 1->1, bk 2->0, bk 3->1, ...

        int load_gmem_a_k = bk * BK + load_smem_a_k; // global col of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = bk * BK + load_smem_b_k; // global row of b
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

#pragma unroll
        for (int tk = 0; tk < BK; tk++) {
            LDST128BITS(r_comp_a[0]) = LDST128BITS(s_a[smem_sel][tk][ty * TM]);
            LDST128BITS(r_comp_a[8]) = LDST128BITS(s_a[smem_sel][tk][ty * TM + 8]);
            LDST128BITS(r_comp_b[0]) = LDST128BITS(s_b[smem_sel][tk][tx * TN]);

#pragma unroll
            for (int tm = 0; tm < TM; tm++) {
#pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    r_c[tm][tn] = __hfma(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
                }
            }
        }
// load 32 half per threads
#pragma unroll
        for (int i = 0; i < 32; i += 8) {
            LDST128BITS(s_b[smem_sel_next][load_smem_b_k][load_smem_b_n + i]) = (LDST128BITS(b[load_gmem_b_addr + i]));
            LDST128BITS(r_load_a[i]) = LDST128BITS(a[load_gmem_a_addr + i]);
        }
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            s_a[smem_sel_next][load_smem_a_k + i][load_smem_a_m] = r_load_a[i];
        }
        __syncthreads();
    }

#pragma unroll
    for (int tk = 0; tk < BK; tk++) {
        LDST128BITS(r_comp_a[0]) = LDST128BITS(s_a[1][tk][ty * TM]);
        LDST128BITS(r_comp_a[8]) = LDST128BITS(s_a[1][tk][ty * TM + 8]);
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
    for (int i = 0; i < TM; ++i) {
        int store_gmem_c_m = by * BM + ty * TM + i;
        int store_gmem_c_n = bx * BN + tx * TN;
        int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
        LDST128BITS(c[store_gmem_c_addr]) = LDST128BITS(r_c[i][0]);
    }
}

// t 16x8, 128x128, k 32, 8x16=128 threads per block, with cp.async
template <const int BM = 128, const int BN = 128, const int BK = 32,
          const int TM = 16, const int TN = 8, const int OFFSET = 0>
__global__ void hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf_async_kernel(
    half *a, half *b, half *c, int M, int N, int K) {
    // block(BN/TN, BM/TM) -> (x=16, y=8)
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;                    // 0~15
    int ty = threadIdx.y;                    // 0~7
    int tid = threadIdx.y * blockDim.x + tx; // 0~127
    // 2*128*32*2=16KB, 2*32*128*2=16KB
    __shared__ half s_a[2][BK][BM + OFFSET], s_b[2][BK][BN + OFFSET];
    half r_load_a[32];                       // 32
    half r_comp_a[TM];                       // 16
    half r_comp_b[TN];                       // 8
    half r_c[TM][TN] = {__float2half(0.0f)}; // 16x8

    // 0. 先计算shared memory中的索引
    // tid和需要加载的smem s_a[BM][BK] 之间的索引关系 BM=128 BK=32 按行读取 A行主序
    // 对于s_a每行32个数据，每个线程读取32个，需要1个线程；总共128行，需要128x1刚好128线程
    int load_smem_a_m = tid; // row 0~127
    int load_smem_a_k = 0;   // col 0
    // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=32 BN=128 按行读取 B行主序
    // 对于s_b每行128个数据，每个线程读32个数据，需要4个线程；总共32行，需要32x4=128个线程
    int load_smem_b_k = tid / 4;        // row 0~32, 128/4
    int load_smem_b_n = (tid % 4) * 32; // col 0,32,64,...
    // 1. 再计算全局内存中的索引
    // 要加载到s_a中的元素对应到A全局内存中的行数 每个block负责出C中大小为BM*BN的块
    int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
    int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
    if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;

    // bk = 0 is loading here, buffer 0
    {
        int load_gmem_a_k = load_smem_a_k; // global col of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = load_smem_b_k; // global row of b
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        // load 32(BK) half per threads, 4x128bits memory issues.
        uint32_t load_smem_b_ptr = __cvta_generic_to_shared(
            &s_b[0][load_smem_b_k][load_smem_b_n]);
#pragma unroll
        // 遍历四次
        for (int i = 0; i < 32; i += 8) {
            CP_ASYNC_CA(load_smem_b_ptr + i * 2, &b[load_gmem_b_addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();

#pragma unroll
        for (int i = 0; i < 32; i += 8) {
            LDST128BITS(r_load_a[i]) = LDST128BITS(a[load_gmem_a_addr + i]);
        }
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            s_a[0][load_smem_a_k + i][load_smem_a_m] = r_load_a[i];
        }
        CP_ASYNC_WAIT_GROUP(0);
    }
    __syncthreads();

    for (int bk = 1; bk < (K + BK - 1) / BK; ++bk) {
        int smem_sel = (bk - 1) & 1; // bk 1->0, bk 2->1, bk 3->0, ...
        int smem_sel_next = bk & 1;  // bk 1->1, bk 2->0, bk 3->1, ...

        int load_gmem_a_k = bk * BK + load_smem_a_k; // global col of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = bk * BK + load_smem_b_k; // global row of b
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

        uint32_t load_smem_b_ptr = __cvta_generic_to_shared(
            &s_b[smem_sel_next][load_smem_b_k][load_smem_b_n]);
#pragma unroll
        for (int i = 0; i < 32; i += 8) {
            CP_ASYNC_CA(load_smem_b_ptr + i * 2, &b[load_gmem_b_addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();

#pragma unroll
        for (int tk = 0; tk < BK; tk++) {
            LDST128BITS(r_comp_a[0]) = LDST128BITS(s_a[smem_sel][tk][ty * TM]);
            LDST128BITS(r_comp_a[8]) = LDST128BITS(s_a[smem_sel][tk][ty * TM + 8]);
            LDST128BITS(r_comp_b[0]) = LDST128BITS(s_b[smem_sel][tk][tx * TN]);

#pragma unroll
            for (int tm = 0; tm < TM; tm++) {
#pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    r_c[tm][tn] = __hfma(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
                }
            }
        }
// load 32 half per threads
#pragma unroll
        for (int i = 0; i < 32; i += 8) {
            LDST128BITS(r_load_a[i]) = LDST128BITS(a[load_gmem_a_addr + i]);
        }
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            s_a[smem_sel_next][load_smem_a_k + i][load_smem_a_m] = r_load_a[i];
        }

        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
    }

#pragma unroll
    for (int tk = 0; tk < BK; tk++) {
        LDST128BITS(r_comp_a[0]) = LDST128BITS(s_a[1][tk][ty * TM]);
        LDST128BITS(r_comp_a[8]) = LDST128BITS(s_a[1][tk][ty * TM + 8]);
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
    for (int i = 0; i < TM; ++i) {
        int store_gmem_c_m = by * BM + ty * TM + i;
        int store_gmem_c_n = bx * BN + tx * TN;
        int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
        LDST128BITS(c[store_gmem_c_addr]) = LDST128BITS(r_c[i][0]);
    }
}


































// Async version of tensor core kernel using cp.async instructions
// This version uses hardware asynchronous memory copy (cp.async) with 2-stage double buffering
// Requires SM 8.0+ (Ampere and newer) for cp.async support
template <typename InputType,
          const int BLOCK_ROW_WARPS = 4,
          const int BLOCK_COL_WARPS = 4,
          const int WARP_ROW_TILES = 4,
          const int WARP_COL_TILES = 2,
          const int WMMA_M = 16,
          const int WMMA_N = 16,
          const int WMMA_K = 16>
__global__ void
sgemm_tensorcore_async_kernel(int num_rows_a, int num_cols_b, int num_cols_a,
                              float alpha, const InputType *matrix_a,
                              const InputType *matrix_b, float beta,
                              float *matrix_c) {
    // Thread and warp identification
    const int warp_id = threadIdx.x / 32;
    const int warp_row = warp_id / BLOCK_COL_WARPS;
    const int warp_col = warp_id % BLOCK_COL_WARPS;

    // Compute block tile dimensions
    constexpr int BLOCK_ROW_TILES = WARP_ROW_TILES * BLOCK_ROW_WARPS;
    constexpr int BLOCK_COL_TILES = WARP_COL_TILES * BLOCK_COL_WARPS;
    constexpr int BM = BLOCK_ROW_TILES * WMMA_M; // 256
    constexpr int BN = BLOCK_COL_TILES * WMMA_N; // 128
    constexpr int BK = WMMA_K;                   // 16

    // Double-buffered shared memory for async pipeline
    constexpr int NUM_STAGES = 2;
    __shared__ InputType tile_a[NUM_STAGES][BM * BK];
    __shared__ InputType tile_b[NUM_STAGES][BK * BN];

    const InputType *global_a = matrix_a;
    const InputType *global_b = matrix_b;
    float *global_c = matrix_c;

    // WMMA fragments
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, InputType, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, InputType, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag[WARP_ROW_TILES][WARP_COL_TILES];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize accumulators
#pragma unroll
    for (int i = 0; i < WARP_ROW_TILES; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_COL_TILES; ++j) {
            nvcuda::wmma::fill_fragment(acc_frag[i][j], 0.0f);
        }
    }

    constexpr int NUM_THREADS = BLOCK_ROW_WARPS * BLOCK_COL_WARPS * 32;

    // Create pipeline for async operations
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, NUM_STAGES> shared_state;
    auto pipeline = cuda::make_pipeline(cg::this_thread_block(), &shared_state);

    // Double buffering control
    int read_buffer = 0;

    // ===== Prologue: Async load first tile into buffer 0 =====
    {
        pipeline.producer_acquire();

        // Async load A tile using cp.async
        for (int idx = threadIdx.x; idx < BM * BK; idx += NUM_THREADS) {
            int row = idx / BK;
            int col = idx % BK;
            int global_row = blockIdx.y * BM + row;
            int global_col = col;

            cuda::memcpy_async(&tile_a[0][row * BK + col],
                               &global_a[global_row * num_cols_a + global_col],
                               cuda::aligned_size_t<sizeof(InputType)>(sizeof(InputType)),
                               pipeline);
        }

        // Async load B tile using cp.async
        for (int idx = threadIdx.x; idx < BK * BN; idx += NUM_THREADS) {
            int row = idx / BN;
            int col = idx % BN;
            int global_row = row;
            int global_col = blockIdx.x * BN + col;

            cuda::memcpy_async(&tile_b[0][col * BK + row],
                               &global_b[global_row * num_cols_b + global_col],
                               cuda::aligned_size_t<sizeof(InputType)>(sizeof(InputType)),
                               pipeline);
        }

        pipeline.producer_commit();
    }

    // ===== Main K-loop with async double buffering =====
    for (int block_k_idx = 0; block_k_idx < num_cols_a; block_k_idx += BK) {
        // Determine which buffer to write next tile into
        int write_buffer = read_buffer ^ 1;

        // ===== Async load next tile (if not last iteration) =====
        if (block_k_idx + BK < num_cols_a) {
            pipeline.producer_acquire();

            // Async load next A tile using cp.async
            for (int idx = threadIdx.x; idx < BM * BK; idx += NUM_THREADS) {
                int row = idx / BK;
                int col = idx % BK;
                int global_row = blockIdx.y * BM + row;
                int global_col = block_k_idx + BK + col;

                cuda::memcpy_async(&tile_a[write_buffer][row * BK + col],
                                   &global_a[global_row * num_cols_a + global_col],
                                   cuda::aligned_size_t<sizeof(InputType)>(sizeof(InputType)),
                                   pipeline);
            }

            // Async load next B tile using cp.async
            for (int idx = threadIdx.x; idx < BK * BN; idx += NUM_THREADS) {
                int row = idx / BN;
                int col = idx % BN;
                int global_row = block_k_idx + BK + row;
                int global_col = blockIdx.x * BN + col;

                cuda::memcpy_async(&tile_b[write_buffer][col * BK + row],
                                   &global_b[global_row * num_cols_b + global_col],
                                   cuda::aligned_size_t<sizeof(InputType)>(sizeof(InputType)),
                                   pipeline);
            }

            pipeline.producer_commit();
        }

        // ===== Wait for current buffer to be ready, then compute =====
        pipeline.consumer_wait();
        cg::this_thread_block().sync();

        // ===== Compute using current read_buffer =====
        // This happens while next buffer is being loaded asynchronously
#pragma unroll
        for (int i = 0; i < WARP_ROW_TILES; ++i) {
#pragma unroll
            for (int j = 0; j < WARP_COL_TILES; ++j) {
                int a_tile_row = warp_row * WARP_ROW_TILES + i;
                int b_tile_col = warp_col * WARP_COL_TILES + j;

                InputType const *a_tile_ptr = tile_a[read_buffer] + (a_tile_row * WMMA_M) * BK;
                InputType const *b_tile_ptr = tile_b[read_buffer] + (b_tile_col * WMMA_N) * BK;

                nvcuda::wmma::load_matrix_sync(a_frag, a_tile_ptr, BK);
                nvcuda::wmma::load_matrix_sync(b_frag, b_tile_ptr, BK);

                nvcuda::wmma::mma_sync(acc_frag[i][j], a_frag, b_frag, acc_frag[i][j]);
            }
        }

        // Release current buffer and switch to next
        pipeline.consumer_release();
        read_buffer = write_buffer;
    }

    // ===== Write results to global memory =====
    // No bounds checking - assumes aligned dimensions
#pragma unroll
    for (int i = 0; i < WARP_ROW_TILES; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_COL_TILES; ++j) {
            int c_tile_row = warp_row * WARP_ROW_TILES + i;
            int c_tile_col = warp_col * WARP_COL_TILES + j;

            int global_row = blockIdx.y * BM + c_tile_row * WMMA_M;
            int global_col = blockIdx.x * BN + c_tile_col * WMMA_N;

            float *c_ptr = global_c + global_row * num_cols_b + global_col;

            // Load existing C and apply alpha/beta scaling
            nvcuda::wmma::load_matrix_sync(c_frag, c_ptr, num_cols_b, nvcuda::wmma::mem_row_major);

#pragma unroll
            for (int t = 0; t < c_frag.num_elements; ++t) {
                c_frag.x[t] = alpha * acc_frag[i][j].x[t] + beta * c_frag.x[t];
            }

            // Write result back
            nvcuda::wmma::store_matrix_sync(c_ptr, c_frag, num_cols_b, nvcuda::wmma::mem_row_major);
        }
    }
}

// ============================================================================
// Launcher Functions: FP16 and BF16 variants
// ============================================================================

template <typename InputType, typename TorchType>
void sgemm_tensorcore_async_launcher(
    const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
    torch::Tensor &output_matrix, float alpha, float beta,
    torch::ScalarType expected_dtype, const char *dtype_name) {
    TORCH_CHECK(matrix_a.device().is_cuda(), "Matrix A must be on CUDA device");
    TORCH_CHECK(matrix_b.device().is_cuda(), "Matrix B must be on CUDA device");
    TORCH_CHECK(matrix_a.dtype() == expected_dtype,
                std::string("Matrix A must be ") + dtype_name + " for Tensor Core async kernel");
    TORCH_CHECK(matrix_b.dtype() == expected_dtype,
                std::string("Matrix B must be ") + dtype_name + " for Tensor Core async kernel");
    TORCH_CHECK(output_matrix.dtype() == torch::kFloat32, "Matrix C must be float32");
    TORCH_CHECK(matrix_a.dim() == 2, "Matrix A must be 2D");
    TORCH_CHECK(matrix_b.dim() == 2, "Matrix B must be 2D");

    const int num_rows_a = static_cast<int>(matrix_a.size(0));
    const int num_cols_a = static_cast<int>(matrix_a.size(1));
    const int num_cols_b = static_cast<int>(matrix_b.size(1));

    TORCH_CHECK(matrix_b.size(0) == num_cols_a,
                "Matrix dimensions must match: A is MxK, B must be KxN");
    TORCH_CHECK(output_matrix.size(0) == num_rows_a && output_matrix.size(1) == num_cols_b,
                "Matrix C must be MxN");

    const auto *d_matrix_a = reinterpret_cast<const InputType *>(matrix_a.data_ptr<TorchType>());
    const auto *d_matrix_b = reinterpret_cast<const InputType *>(matrix_b.data_ptr<TorchType>());
    float *d_output_matrix = output_matrix.data_ptr<float>();

    constexpr int BLOCK_ROW_WARPS = 4;
    constexpr int BLOCK_COL_WARPS = 4;
    constexpr int WARP_ROW_TILES = 4;
    constexpr int WARP_COL_TILES = 2;
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    constexpr int BM = WARP_ROW_TILES * BLOCK_ROW_WARPS * WMMA_M; // 256
    constexpr int BN = WARP_COL_TILES * BLOCK_COL_WARPS * WMMA_N; // 128

    dim3 grid_dim(ceil_div(num_cols_b, BN), ceil_div(num_rows_a, BM));
    dim3 block_dim(BLOCK_ROW_WARPS * BLOCK_COL_WARPS * 32);

    sgemm_tensorcore_async_kernel<InputType, BLOCK_ROW_WARPS, BLOCK_COL_WARPS,
                                  WARP_ROW_TILES, WARP_COL_TILES,
                                  WMMA_M, WMMA_N, WMMA_K>
        <<<grid_dim, block_dim>>>(
            num_rows_a, num_cols_b, num_cols_a,
            alpha, d_matrix_a, d_matrix_b, beta, d_output_matrix);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

void sgemm_tensorcore_async_fp16(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                                 torch::Tensor &output_matrix, float alpha, float beta) {
    sgemm_tensorcore_async_launcher<half, at::Half>(
        matrix_a, matrix_b, output_matrix, alpha, beta,
        torch::kFloat16, "float16");
}

void sgemm_tensorcore_async_bf16(const torch::Tensor &matrix_a, const torch::Tensor &matrix_b,
                                 torch::Tensor &output_matrix, float alpha, float beta) {
    sgemm_tensorcore_async_launcher<nv_bfloat16, at::BFloat16>(
        matrix_a, matrix_b, output_matrix, alpha, beta,
        torch::kBFloat16, "bfloat16");
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

void hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf(
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
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 16;
    constexpr int TM = 8;
    constexpr int TN = 8;

    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_kernel<
        BM, BN, BK, TM, TN, 8><<<grid, block>>>(
        reinterpret_cast<half *>(a.data_ptr()),
        reinterpret_cast<half *>(b.data_ptr()),
        reinterpret_cast<half *>(c.data_ptr()),
        M, N, K);
}

// t 8x8 fp16x8 pack, double buffers, k 16, copy async
void hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async(
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
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 16;
    constexpr int TM = 8;
    constexpr int TN = 8;

    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async_kernel<
        BM, BN, BK, TM, TN, 8><<<grid, block>>>(
        reinterpret_cast<half *>(a.data_ptr()),
        reinterpret_cast<half *>(b.data_ptr()),
        reinterpret_cast<half *>(c.data_ptr()),
        M, N, K);
}

void hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf(
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
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int TM = 8;
    constexpr int TN = 8;

    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_kernel<
        BM, BN, BK, TM, TN, 8><<<grid, block>>>(
        reinterpret_cast<half *>(a.data_ptr()),
        reinterpret_cast<half *>(b.data_ptr()),
        reinterpret_cast<half *>(c.data_ptr()),
        M, N, K);
}

void hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_async(
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
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int TM = 8;
    constexpr int TN = 8;

    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_async_kernel<
        BM, BN, BK, TM, TN, 8><<<grid, block>>>(
        reinterpret_cast<half *>(a.data_ptr()),
        reinterpret_cast<half *>(b.data_ptr()),
        reinterpret_cast<half *>(c.data_ptr()),
        M, N, K);
}

// t 16x8, 128x128, k 32, 8x16=128 threads per block, with cp.async
void hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf(
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
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int TM = 16;
    constexpr int TN = 8;

    dim3 block(BN / TN, BM / TM); // (16,8)
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf_kernel<
        BM, BN, BK, TM, TN, 8><<<grid, block>>>(
        reinterpret_cast<half *>(a.data_ptr()),
        reinterpret_cast<half *>(b.data_ptr()),
        reinterpret_cast<half *>(c.data_ptr()),
        M, N, K);
}

void hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf_async(
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
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int TM = 16;
    constexpr int TN = 8;

    dim3 block(BN / TN, BM / TM); // (16,8)
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf_async_kernel<
        BM, BN, BK, TM, TN, 8><<<grid, block>>>(
        reinterpret_cast<half *>(a.data_ptr()),
        reinterpret_cast<half *>(b.data_ptr()),
        reinterpret_cast<half *>(c.data_ptr()),
        M, N, K);
}
