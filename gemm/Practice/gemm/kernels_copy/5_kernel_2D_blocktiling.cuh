#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// 2D分块平铺优化：每个线程计算TM×TN个结果（二维平铺），进一步提高计算强度
// 模板参数：
// - BM: 块的行大小（Block行维度）
// - BN: 块的列大小（Block列维度）
// - BK: 块的内积大小（Block内积维度，沿K维度）
// - TM: 线程行乘法因子（每个线程计算TM行）
// - TN: 线程列乘法因子（每个线程计算TN列）
// 优化目标：2D平铺增加数据重用，减少共享内存访问，提高计算强度

// __launch_bounds__指定每个SM的最大线程数和最小块数，帮助编译器优化
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemm2DBlocktiling(int M, int N, int K, float alpha, const float *A,
                       const float *B, float beta, float *C) {
  // 当前线程块在输出矩阵C中的位置（块坐标）
  // 每个线程块计算一个BM×BN的输出子矩阵
  const uint cRow = blockIdx.y;  // 块行索引（对应输出矩阵C的行块）
  const uint cCol = blockIdx.x;  // 块列索引（对应输出矩阵C的列块）

  // 计算块平铺的统计信息
  const uint totalResultsBlocktile = BM * BN;  // 每个块需要计算的总元素数
  // 一个线程负责计算块平铺中的TM*TN个元素
  const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);  // 每个块需要的线程数

  // 验证：ResultsPerBlock / ResultsPerThread == ThreadsPerBlock
  assert(numThreadsBlocktile == blockDim.x);

  // BN/TN是跨越一列所需的线程数
  // 将一维线程索引映射到二维线程网格坐标
  // 示例：假设BN=128，TN=8，则BN/TN=16
  // - threadIdx.x=0: threadCol=0, threadRow=0
  // - threadIdx.x=1: threadCol=1, threadRow=0
  // - threadIdx.x=15: threadCol=15, threadRow=0
  // - threadIdx.x=16: threadCol=0, threadRow=1
  // 这样每个线程负责计算输出子矩阵中的一个TM×TN的小矩形区域
  const int threadCol = threadIdx.x % (BN / TN);  // 线程在块内的列索引（0到BN/TN-1）
  const int threadRow = threadIdx.x / (BN / TN);  // 线程在块内的行索引（0到BM/TM-1）

  // 在共享内存中为当前块分配空间（SMEM = Shared Memory）
  // 示例：假设BM=128，BN=128，BK=8
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

  // 计算此线程将加载到SMEM中的索引
  const uint innerRowA = threadIdx.x / BK;  // A子块中的行索引（用于从全局内存加载）
  const uint innerColA = threadIdx.x % BK;  // A子块中的列索引（用于从全局内存加载）
  
  // 计算单个步骤中单个块加载的As行数
  // strideA = 每个线程加载时在行方向上的步长
  const uint strideA = numThreadsBlocktile / BK;
  
  const uint innerRowB = threadIdx.x / BN;  // B子块中的行索引（用于从全局内存加载）
  const uint innerColB = threadIdx.x % BN;  // B子块中的列索引（用于从全局内存加载）
  
  // 对于As和Bs，我们希望每次加载跨越完整的列宽，以实现更好的全局内存合并
  // （而不是跨越完整的行宽并迭代列）
  const uint strideB = numThreadsBlocktile / BN;

  // 在线程寄存器文件中分配线程本地缓存用于存储结果
  // 每个线程计算TM×TN个结果，存储在寄存器中
  // 示例：假设TM=8，TN=8，则threadResults大小为64个浮点数
  float threadResults[TM * TN] = {0.0};  // 初始化TM×TN个结果为0.0
  
  // 用于As和Bs的寄存器缓存
  // 将共享内存中的数据加载到寄存器中，减少共享内存访问次数
  float regM[TM] = {0.0};  // 缓存A的TM个元素（一行中的TM个连续元素）
  float regN[TN] = {0.0};  // 缓存B的TN个元素（一列中的TN个连续元素）

  // 最外层循环：遍历K维度的块（分块矩阵乘法）
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // 填充共享内存缓存（从全局内存加载数据到共享内存）
    // 使用循环加载，每个线程加载多个元素，提高加载效率
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
      // 加载A的BM×BK子块：每个线程加载strideA行中的一列
      As[(innerRowA + loadOffset) * BK + innerColA] =
          A[(innerRowA + loadOffset) * K + innerColA];
    }
    for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      // 加载B的BK×BN子块：每个线程加载strideB行中的一列
      Bs[(innerRowB + loadOffset) * BN + innerColB] =
          B[(innerRowB + loadOffset) * N + innerColB];
    }
    __syncthreads();  // 同步所有线程，确保共享内存数据加载完成

    // 前进到下一个块（沿K维度移动）
    A += BK;     // A指针向右移动BK列
    B += BK * N; // B指针向下移动BK行（因为B是列主序存储）

    // 计算每个线程的结果（矩阵乘法核心计算）
    // 2D平铺优化：每个线程计算TM×TN个小矩阵乘法
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // 将数据从共享内存块加载到寄存器中
      // 关键优化：将共享内存访问转换为寄存器访问，减少共享内存带宽压力
      for (uint i = 0; i < TM; ++i) {
        // 加载A的TM个元素：A的第(threadRow*TM+i)行，第dotIdx列
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
      }
      for (uint i = 0; i < TN; ++i) {
        // 加载B的TN个元素：B的第dotIdx行，第(threadCol*TN+i)列
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      
      // 执行TM×TN个小矩阵乘法（外积累加）
      // 计算：C_sub += A_column × B_row 的累加
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          // 累加：A的第resIdxM行 × B的第resIdxN列
          threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();  // 同步，确保所有线程完成当前块的计算后再加载下一个块
  }

  // 将结果写回全局内存（输出矩阵C）
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      // 计算输出位置：C的第(threadRow*TM+resIdxM)行，第(threadCol*TN+resIdxN)列
      // 应用缩放因子：C = alpha*A*B + beta*C
      C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
          alpha * threadResults[resIdxM * TN + resIdxN] +  // alpha * 累加结果
          beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];  // beta * 原始C值
    }
  }
}
















































