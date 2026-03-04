#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
const int K9_NUM_THREADS = 256;  // 自动调优的固定线程数

// 自动调优优化：结合多种优化技术，使用warp平铺进一步提高性能
// 模板参数：
// - BM: 块的行大小（Block行维度）
// - BN: 块的列大小（Block列维度）
// - BK: 块的内积大小（Block内积维度，沿K维度）
// - TM: 线程行乘法因子（每个线程计算TM行）
// - TN: 线程列乘法因子（每个线程计算TN列）
// 优化目标：结合warp平铺、向量化、寄存器缓存等多种优化，实现高性能矩阵乘法
// 这是前面所有优化的综合版本，通常通过自动调优找到最佳参数组合

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__(K9_NUM_THREADS)
    sgemmAutotuned(int M, int N, int K, float alpha, float *A, float *B,
                   float beta, float *C) {
  // 当前线程块在输出矩阵C中的位置（块坐标）
  // 每个线程块计算一个BM×BN的输出子矩阵
  const uint cRow = blockIdx.y;  // 块行索引（对应输出矩阵C的行块）
  const uint cCol = blockIdx.x;  // 块列索引（对应输出矩阵C的列块）

  // warp平铺大小：每个warp计算WM×WN的子区域
  // 关键优化：将块进一步划分为warp平铺，提高数据局部性
  constexpr int WM = TM * 16;  // warp平铺的行大小（TM × 16个线程的行维度）
  constexpr int WN = TN * 16;  // warp平铺的列大小（TN × 16个线程的列维度）
  
  // warp平铺的迭代次数：块需要被划分为多少个warp平铺
  constexpr int WMITER = CEIL_DIV(BM, WM);  // 行方向上的warp平铺迭代次数
  constexpr int WNITER = CEIL_DIV(BN, WN);  // 列方向上的warp平铺迭代次数

  // 线程在warp平铺中的位置
  // 关键优化：使用warp平铺内的二维坐标，提高数据局部性
  const int threadCol = threadIdx.x % (WN / TN);  // 线程在warp平铺内的列索引
  const int threadRow = threadIdx.x / (WN / TN);  // 线程在warp平铺内的行索引

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
  
  // 行步长：每个线程加载多行数据，提高加载效率
  constexpr uint rowStrideA = (K9_NUM_THREADS * 4) / BK;  // A的行加载步长
  
  const uint innerRowB = threadIdx.x / (BN / 4);  // B子块中的行索引（考虑向量化）
  const uint innerColB = threadIdx.x % (BN / 4);  // B子块中的列索引（考虑向量化，实际列索引需要乘以4）
  
  constexpr uint rowStrideB = K9_NUM_THREADS / (BN / 4);  // B的行加载步长

  // 在线程寄存器文件中分配线程本地缓存用于存储结果
  // 关键优化：为所有warp平铺的结果分配寄存器缓存
  // 大小：WMITER × WNITER × TM × TN（所有warp平铺的所有结果）
  float threadResults[WMITER * WNITER * TM * TN] = {0.0};  // 初始化所有结果为0.0
  float regM[TM] = {0.0};  // 缓存A的TM个元素（寄存器缓存）
  float regN[TN] = {0.0};  // 缓存B的TN个元素（寄存器缓存）

  // 最外层循环：遍历K维度的块（分块矩阵乘法）
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // 填充共享内存缓存（从全局内存加载数据到共享内存）
    
    // 加载A矩阵数据：每个线程加载多行数据（使用rowStrideA步长）
    for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
      float4 tmp = reinterpret_cast<float4 *>(
          &A[(innerRowA + offset) * K + innerColA * 4])[0];
      // 存储A时进行转置（列主序）
      As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
      As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
      As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
      As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
    }

    // 加载B矩阵数据：每个线程加载多行数据（使用rowStrideB步长）
    for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
      reinterpret_cast<float4 *>(
          &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
          reinterpret_cast<float4 *>(
              &B[(innerRowB + offset) * N + innerColB * 4])[0];
    }
    __syncthreads();  // 同步所有线程，确保共享内存数据加载完成

    // 关键优化：warp平铺迭代
    // 将BM×BN的块进一步划分为WM×WN的warp平铺
    for (uint wmIdx = 0; wmIdx < WMITER; ++wmIdx) {
      for (uint wnIdx = 0; wnIdx < WNITER; ++wnIdx) {
        // 计算每个线程在当前warp平铺中的结果
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
          // 将数据从共享内存块加载到寄存器中
          for (uint i = 0; i < TM; ++i) {
            // 加载A的TM个元素：考虑warp平铺偏移(wmIdx * WM)
            regM[i] = As[dotIdx * BM + (wmIdx * WM) + threadRow * TM + i];
          }
          for (uint i = 0; i < TN; ++i) {
            // 加载B的TN个元素：考虑warp平铺偏移(wnIdx * WN)
            regN[i] = Bs[dotIdx * BN + (wnIdx * WN) + threadCol * TN + i];
          }
          
          // 执行TM×TN个小矩阵乘法（外积累加）
          for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
            for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
              // 累加：A的第resIdxM行 × B的第resIdxN列
              // 复杂索引：将结果存储到threadResults的适当位置
              threadResults[(wmIdx * TM + resIdxM) * (WNITER * TN) +
                            wnIdx * TN + resIdxN] +=
                  regM[resIdxM] * regN[resIdxN];
            }
          }
        }
      }
    }
    __syncthreads();  // 同步，确保所有线程完成当前块的计算
    
    // 前进到下一个块（沿K维度移动）
    A += BK;     // A指针向右移动BK列
    B += BK * N; // B指针向下移动BK行（因为B是列主序存储）
  }

  // 将结果写回全局内存（输出矩阵C）
  // 关键优化：按照warp平铺的顺序写回结果
  for (uint wmIdx = 0; wmIdx < WMITER; ++wmIdx) {
    for (uint wnIdx = 0; wnIdx < WNITER; ++wnIdx) {
      // 计算当前warp平铺在输出矩阵C中的起始位置
      float *C_interim = C + (wmIdx * WM * N) + (wnIdx * WN);
      
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          // 将C向量加载到寄存器中（向量化加载）
          float4 tmp = reinterpret_cast<float4 *>(
              &C_interim[(threadRow * TM + resIdxM) * N + threadCol * TN +
                         resIdxN])[0];
          
          // 在寄存器中执行GEMM更新：C = alpha*A*B + beta*C
          // 计算threadResults中的索引
          const int i =
              (wmIdx * TM + resIdxM) * (WNITER * TN) + wnIdx * TN + resIdxN;
          tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
          tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
          tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
          tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
          
          // 写回更新后的向量（向量化存储）
          reinterpret_cast<float4 *>(&C_interim[(threadRow * TM + resIdxM) * N +
                                                threadCol * TN + resIdxN])[0] =
              tmp;
        }
      }
    }
  }
}