#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm1DBlocktiling(int M, int N, int K, float alpha,
                                   const float *A, const float *B, float beta,
                                   float *C) {
  // 如果交换x和y坐标，对于大矩阵性能会降低约30%。
  // 当前这种30%更快的配置确保具有连续blockID的块按顺序访问B的列，同时共享A的同一行。
  // 较慢的配置会共享A的列，但访问B将是非顺序的。因此更快的配置具有更好的空间局部性，从而有更高的L2缓存命中率。
  const uint cRow = blockIdx.y;  // 当前线程块在网格中的行索引（对应输出矩阵C的行块）
  const uint cCol = blockIdx.x;  // 当前线程块在网格中的列索引（对应输出矩阵C的列块）

  // 每个warp将计算32*TM个元素，其中32是列维度（warp大小）。
  // 将一维线程索引映射到二维线程网格坐标：
  const int threadCol = threadIdx.x % BN;  // 线程在块内的列索引（0到BN-1）
  const int threadRow = threadIdx.x / BN;  // 线程在块内的行索引（0到BM/TM-1）

  // 在共享内存中为当前块分配空间（SMEM = Shared Memory）
  __shared__ float As[BM * BK];  // 存储A矩阵的BM×BK子块（BM行，BK列）
  __shared__ float Bs[BK * BN];  // 存储B矩阵的BK×BN子块（BK行，BN列）

  // 将指针移动到当前块对应的起始位置：
  A += cRow * BM * K;                    // 移动到A矩阵第cRow个BM行的起始位置（每个BM行有K列）
  B += cCol * BN;                        // 移动到B矩阵第cCol个BN列的起始位置
  C += cRow * BM * N + cCol * BN;        // 移动到输出矩阵C中当前块的位置（第cRow个BM行，第cCol个BN列）

  // TODO: 调整每个线程加载多个条目以更好地利用缓存大小
  assert(BM * BK == blockDim.x);  // 确保线程数等于加载A子块所需的线程数
  assert(BN * BK == blockDim.x);  // 确保线程数等于加载B子块所需的线程数
  
  // 计算线程在共享内存块中的位置（用于从全局内存加载数据到共享内存）：
  const uint innerColA = threadIdx.x % BK; // A子块中的列索引（用于warp级别的全局内存合并访问）
  const uint innerRowA = threadIdx.x / BK; // A子块中的行索引
  const uint innerColB = threadIdx.x % BN; // B子块中的列索引（用于warp级别的全局内存合并访问）
  const uint innerRowB = threadIdx.x / BN; // B子块中的行索引

  // 在线程寄存器文件中分配线程本地缓存用于存储结果
  // 每个线程计算TM个结果（TM = Thread Multiplicity，线程乘法因子）
  float threadResults[TM] = {0.0};  // 初始化TM个结果为0.0

  // 外层循环：遍历K维度的块（分块矩阵乘法）
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
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