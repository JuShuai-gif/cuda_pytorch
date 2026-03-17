#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// 1D分块平铺优化：每个线程计算多个结果（TM），提高计算强度
// 模板参数：
// - BM: 块的行大小（Block行维度）
// - BN: 块的列大小（Block列维度）  
// - BK: 块的内积大小（Block内积维度，沿K维度）
// - TM: 线程乘法因子（每个线程计算TM个结果）
// 优化目标：增加每个线程的计算量，隐藏内存访问延迟，提高计算强度

template <const int BM,const int BN,const int BK,const int TM>
__global__ void sgemm1DBlocktiling(int M, int N, int K, float alpha,
                                   const float *A, const float *B, float beta,
                                   float *C) {
    // 当前线程块在输出矩阵C中的位置（块坐标）
    // 每个线程块计算一个BM×BN的输出子矩阵
    const uint cRow = blockIdx.y;  // 块行索引（对应输出矩阵C的行块）
    const uint cCol = blockIdx.x;  // 块列索引（对应输出矩阵C的列块）

    // 将一维线程索引映射到二维线程网格坐标
    // 示例：假设BN=32，threadIdx.x从0到1023（假设块大小1024）
    // - threadIdx.x=0: threadCol=0, threadRow=0
    // - threadIdx.x=1: threadCol=1, threadRow=0
    // - threadIdx.x=31: threadCol=31, threadRow=0
    // - threadIdx.x=32: threadCol=0, threadRow=1
    // 这样每个线程负责计算输出子矩阵中的一个TM×1的垂直条带
    const int threadCol = threadIdx.x % BN;  // 线程在块内的列索引（0到BN-1）
    const int threadRow = threadIdx.x / BN;  // 线程在块内的行索引（0到BM/TM-1）

    // 在共享内存中为当前块分配空间（SMEM = Shared Memory）
    // 示例：假设BM=128，BK=8，BN=128
    // As大小：128×8 = 1024个浮点数（4KB）
    // Bs大小：8×128 = 1024个浮点数（4KB）
    __shared__ float As[BM * BK];  // 存储A矩阵的BM×BK子块（BM行，BK列）
    __shared__ float Bs[BK * BN];  // 存储B矩阵的BK×BN子块（BK行，BN列）

    // 将指针移动到当前块对应的起始位置：
    // 示例：假设BM=128，BN=128，BK=8，cRow=2，cCol=3，K=1024，N=2048
    // A += 2 * 128 * 1024 = 262144 → 指向A矩阵第256行开始（第2个128行块）
    // B += 3 * 128 = 384 → 指向B矩阵第384列开始（第3个128列块）
    // C += 2 * 128 * 2048 + 3 * 128 = 524288 + 384 = 524672 → 指向C矩阵第256行第384列开始
    A += cRow * BM * K;                    // 移动到A矩阵第cRow个BM行的起始位置
    B += cCol * BN;                        // 移动到B矩阵第cCol个BN列的起始位置
    C += cRow * BM * N + cCol * BN;        // 移动到输出矩阵C中当前块的位置

    assert(BM * BK == blockDim.x);  // 确保线程数等于加载A子块所需的线程数
    assert(BN * BK == blockDim.x);  // 确保线程数等于加载B子块所需的线程数
  
    const uint innerColA = threadIdx.x % BK; // A子块中的列索引（用于warp级别的全局内存合并访问）
    const uint innerRowA = threadIdx.x / BK; // A子块中的行索引
    const uint innerColB = threadIdx.x % BN; // B子块中的列索引（用于warp级别的全局内存合并访问）
    const uint innerRowB = threadIdx.x / BN; // B子块中的行索引

    // 在线程寄存器文件中分配线程本地缓存用于存储结果
    // 每个线程计算TM个结果（TM = Thread Multiplicity，线程乘法因子）
    float threadResults[TM] = {0.0};  // 初始化TM个结果为0.0

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK)
    {
    // 填充共享内存缓存（从全局内存加载数据到共享内存）
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];  // 加载A的BM×BK子块
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];  // 加载B的BK×BN子块
    __syncthreads();  // 同步所有线程，确保共享内存数据加载完成

    // 前进到下一个块（沿K维度移动）
    A += BK;        // A指针向右移动BK列
    B += BK * N;    // B指针向下移动BK行（因为B是列主序存储）

// 计算每个线程的结果（矩阵乘法核心计算）
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // 我们将点积循环放在外层，这有助于重用Bs条目，可以将其缓存在临时变量中。
      // 计算：C_sub = A_sub × B_sub 的累加
      float tmpB = Bs[dotIdx * BN + threadCol];  // 从共享内存读取B的一个元素（当前列threadCol，第dotIdx行）
      for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        // 累加矩阵乘法结果：A的行(threadRow*TM+resIdx) × B的列(threadCol)
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;  // A的第(threadRow*TM+resIdx)行，第dotIdx列
      }
    }
    __syncthreads();  // 同步，确保所有线程完成当前块的计算后再加载下一个块
    }
    
  // 将结果写回全局内存（输出矩阵C）
  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    // 计算输出位置：C的第(threadRow*TM+resIdx)行，第threadCol列
    // 应用缩放因子：C = alpha*A*B + beta*C
    C[(threadRow * TM + resIdx) * N + threadCol] =
        alpha * threadResults[resIdx] +  // alpha * 累加结果
        beta * C[(threadRow * TM + resIdx) * N + threadCol];  // beta * 原始C值
  }


}






































