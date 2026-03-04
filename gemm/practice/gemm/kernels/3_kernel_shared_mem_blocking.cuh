#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// 共享内存分块优化：使用共享内存缓存数据块，减少全局内存访问
// 优化目标：利用共享内存（SMEM）的高速缓存特性，重用从全局内存加载的数据
// 典型值：BLOCKSIZE = 32（一个warp的大小）

template <const uint BLOCKSIZE>
__global__ void sgemm_global_mem_coalesce(int M,int N,int K,float alpha,const float* A,
                            const float* B,float beta,float *C){
    
    // 当前线程块在输出矩阵C中的位置（块坐标）
    // 每个线程块计算一个BLOCKSIZE×BLOCKSIZE的输出子矩阵
    const uint cRow = blockIdx.x;   // 表示第几个块的行（块行索引）
    const uint cCol = blockIdx.y;   // 表示第几个块的列（块列索引）

    // 当前块分配共享内存（SMEM = Shared Memory）
    // 共享内存在块内所有线程间共享，访问速度比全局内存快100倍以上
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];  // 缓存A矩阵的BLOCKSIZE×BLOCKSIZE子块
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];  // 缓存B矩阵的BLOCKSIZE×BLOCKSIZE子块
    
    // 将一维线程索引映射到二维线程网格坐标
    // threadIdx.x是线程在线程块中的一维索引（0到BLOCKSIZE²-1）
    const uint threadCol = threadIdx.x % BLOCKSIZE;  // 线程在块内的列索引（0到BLOCKSIZE-1）
    const uint threadRow = threadIdx.x / BLOCKSIZE;  // 线程在块内的行索引（0到BLOCKSIZE-1）

    // 将指针移动到当前块对应的起始位置：
    // 示例：假设BLOCKSIZE=32，cRow=1，cCol=2，K=128，N=256
    // A += 1 * 32 * 128 = 4096 → 指向A矩阵第32行开始（第1个32行块）
    // B += 2 * 32 = 64 → 指向B矩阵第64列开始（第2个32列块）
    // C += 1 * 32 * 256 + 2 * 32 = 8192 + 64 = 8256 → 指向C矩阵第32行第64列开始
    A += cRow * BLOCKSIZE * K;                      // row = cRow, col = 0（A的第cRow个BLOCKSIZE行）
    B += cCol * BLOCKSIZE;                          // row = 0, col = cCol（B的第cCol个BLOCKSIZE列）
    C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;   // row = cRow, col = cCol（C的对应位置）

    float tmp = 0.0f;  // 临时变量，用于累加当前线程的计算结果

    // 外层循环：沿K维度分块（分块矩阵乘法）
    // 错误修复：应该是bkIdx < K，而不是bkIdx < BLOCKSIZE
    // 每次迭代处理K维度上的一个BLOCKSIZE大小的块
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE)
    {
        // 让每个线程加载A和B中的一个元素到共享内存
        // 使threadCol（=threadIdx.x）成为连续索引，以实现全局内存访问合并
        // 关键优化：同一warp中的线程访问连续的内存地址，实现内存合并访问
        
        // 从全局内存加载数据到共享内存：
        As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];  // 加载A的子块
        Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];  // 加载B的子块
    
        // 同步块内所有线程，确保共享内存数据加载完成
        __syncthreads();
        
        // 前进到下一个块（沿K维度移动）：
        A += BLOCKSIZE;        // A指针向右移动BLOCKSIZE列
        B += BLOCKSIZE * N;    // B指针向下移动BLOCKSIZE行（因为B是列主序存储）

        // 计算当前缓存块的点积（矩阵乘法核心计算）
        // 计算：C_sub += A_sub × B_sub 的累加
        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx)
        {
            // 累加：A的第threadRow行 × B的第threadCol列
            // As[threadRow * BLOCKSIZE + dotIdx]: A的第threadRow行，第dotIdx列
            // Bs[dotIdx * BLOCKSIZE + threadCol]: B的第dotIdx行，第threadCol列
            tmp += As[threadRow * BLOCKSIZE + dotIdx] *
                    Bs[dotIdx * BLOCKSIZE + threadCol];
        }

        // 再次同步，避免较快的线程在较慢的线程完成计算前加载下一个块
        __syncthreads();
    }
    
    // 将结果写回全局内存（输出矩阵C）
    // 应用缩放因子：C = α*(A@B) + β*C
    C[threadRow * N + threadCol] =
        alpha * tmp + beta * C[threadRow * N + threadCol];
}











































