#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// 解决共享内存bank冲突优化：重新组织数据布局以避免bank冲突
// 模板参数：
// - BM: 块的行大小（Block行维度）
// - BN: 块的列大小（Block列维度）
// - BK: 块的内积大小（Block内积维度，沿K维度）
// - TM: 线程行乘法因子（每个线程计算TM行）
// - TN: 线程列乘法因子（每个线程计算TN列）
// 优化目标：避免共享内存bank冲突，提高共享内存带宽利用率
// 背景：共享内存被组织成32个bank（对应32个线程的warp），如果同一warp中的多个线程访问同一bank的不同地址，会发生bank冲突

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemmResolveBankConflicts(int M, int N, int K, float alpha,
                                          float *A, float *B, float beta,
                                          float *C) {
  // 当前线程块在输出矩阵C中的位置（块坐标）
  // 每个线程块计算一个BM×BN的输出子矩阵
  const uint cRow = blockIdx.y;  // 块行索引（对应输出矩阵C的行块）
  const uint cCol = blockIdx.x;  // 块列索引（对应输出矩阵C的列块）

  // BN/TN是跨越一列所需的线程数
  // 将一维线程索引映射到二维线程网格坐标
  const int threadCol = threadIdx.x % (BN / TN);  // 线程在块内的列索引（0到BN/TN-1）
  const int threadRow = threadIdx.x / (BN / TN);  // 线程在块内的行索引（0到BM/TM-1）

  // 在共享内存中为当前块分配空间（SMEM = Shared Memory）
  __shared__ float As[BM * BK];  // 存储A矩阵的BM×BK子块（BM行，BK列）
  __shared__ float Bs[BK * BN];  // 存储B矩阵的BK×BN子块（BK行，BN列）

  // 将指针移动到当前块对应的起始位置：
  A += cRow * BM * K;                    // 移动到A矩阵第cRow个BM行的起始位置
  B += cCol * BN;                        // 移动到B矩阵第cCol个BN列的起始位置
  C += cRow * BM * N + cCol * BN;        // 移动到输出矩阵C中当前块的位置

  // 计算此线程将加载到SMEM中的索引
  // 关键优化：每个线程在每个步骤加载128位/32位 = 4个元素（float4向量）
  const uint innerRowA = threadIdx.x / (BK / 4);  // A子块中的行索引（考虑向量化）
  const uint innerColA = threadIdx.x % (BK / 4);  // A子块中的列索引（考虑向量化，实际列索引需要乘以4）
  const uint innerRowB = threadIdx.x / (BN / 4);  // B子块中的行索引（考虑向量化）
  const uint innerColB = threadIdx.x % (BN / 4);  // B子块中的列索引（考虑向量化，实际列索引需要乘以4）

  // 在线程寄存器文件中分配线程本地缓存用于存储结果
  float threadResults[TM * TN] = {0.0};  // 初始化TM×TN个结果为0.0
  float regM[TM] = {0.0};  // 缓存A的TM个元素（寄存器缓存）
  float regN[TN] = {0.0};  // 缓存B的TN个元素（寄存器缓存）

  // 最外层循环：遍历K维度的块（分块矩阵乘法）
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // 填充共享内存缓存（从全局内存加载数据到共享内存）
    
    // 加载A矩阵数据时进行转置
    // 关键优化：转置存储A，使后续访问模式更友好
    float4 tmp =
        reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;  // 转置存储：列主序
    As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

    // 关键优化：存储B时进行"线性化"以避免bank冲突
    // 复杂索引计算：Bs[((innerColB % 2) * 4 + innerRowB * 8 + i) * 16 + innerColB / 2]
    // 这种布局确保同一warp中的线程访问不同的bank，避免bank冲突
    // 假设：BN=128，BK=8，BN/4=32（一个warp的大小）
    // 索引计算分解：
    // 1. innerColB % 2: 将列索引分为奇偶两组（0或1）
    // 2. *4: 每组有4个连续元素
    // 3. + innerRowB * 8: 行偏移，每行8个元素
    // 4. + i: 向量中的第i个元素（0,1,2,3）
    // 5. *16: 每16个元素一组（bank数量相关）
    // 6. + innerColB / 2: 列索引除以2
    tmp = reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
    Bs[((innerColB % 2) * 4 + innerRowB * 8 + 0) * 16 + innerColB / 2] = tmp.x;
    Bs[((innerColB % 2) * 4 + innerRowB * 8 + 1) * 16 + innerColB / 2] = tmp.y;
    Bs[((innerColB % 2) * 4 + innerRowB * 8 + 2) * 16 + innerColB / 2] = tmp.z;
    Bs[((innerColB % 2) * 4 + innerRowB * 8 + 3) * 16 + innerColB / 2] = tmp.w;
    
    __syncthreads();  // 同步所有线程，确保共享内存数据加载完成

    // 前进到下一个块（沿K维度移动）
    A += BK;     // A指针向右移动BK列
    B += BK * N; // B指针向下移动BK行（因为B是列主序存储）

    // 计算每个线程的结果（矩阵乘法核心计算）
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // 将数据从共享内存块加载到寄存器中
      for (uint i = 0; i < TM; ++i) {
        // 加载A的TM个元素：注意As是转置存储（列主序）
        regM[i] = As[dotIdx * BM + threadRow * TM + i];
      }
      for (uint i = 0; i < TN; ++i) {
        // 关键优化：使用避免bank冲突的索引访问Bs
        // Bs[(dotIdx * 8 + i) * 16 + threadCol]
        // 这种访问模式确保同一warp中的线程访问不同的bank
        // 假设：dotIdx=0，i=0..TN-1，threadCol=0..BN/TN-1
        // 对于同一warp中的线程，threadCol不同，因此访问不同的地址
        regN[i] = Bs[(dotIdx * 8 + i) * 16 + threadCol];
      }
      
      // 执行TM×TN个小矩阵乘法（外积累加）
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
  // 使用向量化存储（float4），一次性写入4个连续结果
  for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
    for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
      // 将C向量加载到寄存器中（向量化加载）
      float4 tmp = reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
      
      // 在寄存器中执行GEMM更新：C = alpha*A*B + beta*C
      tmp.x = alpha * threadResults[resIdxM * TN + resIdxN] + beta * tmp.x;
      tmp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * tmp.y;
      tmp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * tmp.z;
      tmp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * tmp.w;
      
      // 写回更新后的向量（向量化存储）
      reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] =
          tmp;
    }
  }
}

















































