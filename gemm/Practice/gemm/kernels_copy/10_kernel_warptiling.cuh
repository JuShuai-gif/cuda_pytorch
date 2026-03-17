#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
const int WARPSIZE = 32; // warpSize在编译时不是constexpr，所以需要定义常量

// warp平铺优化命名空间：将优化函数组织在命名空间中
namespace wt {

// 从全局内存加载数据到共享内存的设备函数
// 模板参数：
// - BM, BN, BK: 块的行、列、内积大小
// - rowStrideA, rowStrideB: 行加载步长
template <const int BM, const int BN, const int BK, const int rowStrideA,
          const int rowStrideB>
__device__ void loadFromGmem(int N, int K, const float *A, const float *B,
                             float *As, float *Bs, int innerRowA, int innerColA,
                             int innerRowB, int innerColB) {
  // 加载A矩阵数据：每个线程加载多行数据（使用rowStrideA步长）
  for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
    // 关键优化：使用float4向量化加载，一次性加载4个连续元素
    const float4 tmp = reinterpret_cast<const float4 *>(
        &A[(innerRowA + offset) * K + innerColA * 4])[0];
    
    // 注释掉的汇编代码显示可以使用内联汇编进行优化加载
    // float4 tmp;
    // asm("ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];"
    //     : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
    //     : "l"(&A[(innerRowA + offset) * K + innerColA * 4]));
    
    // 存储A时进行转置（列主序），提高后续访问效率
    As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
  }

  // 加载B矩阵数据：每个线程加载多行数据（使用rowStrideB步长）
  for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
    // 直接使用向量化存储，不需要拆解
    reinterpret_cast<float4 *>(
        &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
        reinterpret_cast<const float4 *>(
            &B[(innerRowB + offset) * N + innerColB * 4])[0];
    
    // 注释掉的汇编代码显示可以使用内联汇编进行优化加载
    // asm("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
    //     : "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 0]),
    //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 1]),
    //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 2]),
    //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 3])
    //     : "l"(&B[(innerRowB + offset) * N + innerColB * 4]));
  }
}

// 从共享内存处理数据的设备函数（warp平铺矩阵乘法核心计算）
// 模板参数：
// - BM, BN, BK: 块的行、列、内积大小
// - WM, WN: warp平铺的行、列大小
// - WMITER, WNITER: warp平铺在行、列方向上的子平铺迭代次数
// - WSUBM, WSUBN: warp子平铺的行、列大小
// - TM, TN: 每个线程计算的行、列元素数
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void
processFromSmem(float *regM, float *regN, float *threadResults, const float *As,
                const float *Bs, const uint warpRow, const uint warpCol,
                const uint threadRowInWarp, const uint threadColInWarp) {
  // 遍历K维度（内积维度）
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    // 为整个warp平铺填充寄存器
    // 关键优化：一次性加载warp平铺的所有数据到寄存器，减少共享内存访问
    
    // 加载A矩阵数据到寄存器：WMITER个子平铺，每个子平铺TM个元素
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint i = 0; i < TM; ++i) {
        // 复杂索引计算：考虑warp行偏移、子平铺偏移和线程偏移
        regM[wSubRowIdx * TM + i] =
            As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
               threadRowInWarp * TM + i];
      }
    }
    
    // 加载B矩阵数据到寄存器：WNITER个子平铺，每个子平铺TN个元素
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      for (uint i = 0; i < TN; ++i) {
        regN[wSubColIdx * TN + i] =
            Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
               threadColInWarp * TN + i];
      }
    }

    // 执行warp平铺矩阵乘法
    // 关键优化：在寄存器中执行所有计算，最大化数据重用
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        // 计算每个线程在当前warp子平铺中的结果
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
          for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            // 累加矩阵乘法结果
            threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                          (wSubColIdx * TN) + resIdxN] +=
                regM[wSubRowIdx * TM + resIdxM] *
                regN[wSubColIdx * TN + resIdxN];
          }
        }
      }
    }
  }
}

} // namespace wt

/*
 * warp平铺优化内核：高级warp平铺技术，进一步提高性能
 * 模板参数说明：
 * @tparam BM 线程块在M维度上的SMEM缓存大小
 * @tparam BN 线程块在N维度上的SMEM缓存大小
 * @tparam BK 线程块在K维度上的SMEM缓存大小
 * @tparam WM 每个warp计算的连续平铺的M维度大小
 * @tparam WN 每个warp计算的连续平铺的N维度大小
 * @tparam WNITER N维度上的子warp平铺步骤数
 * @tparam TM 每个线程在M维度上的平铺大小
 * @tparam TN 每个线程在N维度上的平铺大小
 * @tparam NUM_THREADS 线程块中的线程数
 * 
 * 优化目标：精细的warp平铺，最大化寄存器重用，减少共享内存访问
 */
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    sgemmWarptiling(int M, int N, int K, float alpha, float *A, float *B,
                    float beta, float *C) {
  // 当前线程块在输出矩阵C中的位置（块坐标）
  const uint cRow = blockIdx.y;  // 块行索引（对应输出矩阵C的行块）
  const uint cCol = blockIdx.x;  // 块列索引（对应输出矩阵C的列块）

  // warp在线程块平铺中的位置
  const uint warpIdx = threadIdx.x / WARPSIZE; // 线程所在的warp索引
  const uint warpCol = warpIdx % (BN / WN);    // warp在块平铺中的列位置
  const uint warpRow = warpIdx / (BN / WN);    // warp在块平铺中的行位置

  // warp子平铺的大小计算
  // WMITER: M维度上的warp子平铺迭代次数
  constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr uint WSUBM = WM / WMITER; // warp子平铺的M维度大小（例如：64/2=32）
  constexpr uint WSUBN = WN / WNITER; // warp子平铺的N维度大小（例如：32/2=16）

  // 线程在warp子平铺中的位置
  const uint threadIdxInWarp = threadIdx.x % WARPSIZE;         // warp内的线程索引[0, 31]
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN); // 线程在warp子平铺中的列位置
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN); // 线程在warp子平铺中的行位置

  // 在共享内存中为当前块分配空间（SMEM = Shared Memory）
  __shared__ float As[BM * BK];  // 存储A矩阵的BM×BK子块
  __shared__ float Bs[BK * BN];  // 存储B矩阵的BK×BN子块

  // 将指针移动到当前块对应的起始位置：
  A += cRow * BM * K;  // 移动到A矩阵第cRow个BM行的起始位置
  B += cCol * BN;      // 移动到B矩阵第cCol个BN列的起始位置
  
  // 关键优化：将C指针直接移动到warp的输出平铺位置
  // 避免在计算过程中重复计算偏移
  C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

  // 计算此线程将加载到SMEM中的索引
  // 关键优化：每个线程在每个步骤加载128位/32位 = 4个元素（float4向量）
  const uint innerRowA = threadIdx.x / (BK / 4);  // A子块中的行索引（考虑向量化）
  const uint innerColA = threadIdx.x % (BK / 4);  // A子块中的列索引（考虑向量化）
  constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;  // A的行加载步长
  
  const uint innerRowB = threadIdx.x / (BN / 4);  // B子块中的行索引（考虑向量化）
  const uint innerColB = threadIdx.x % (BN / 4);  // B子块中的列索引（考虑向量化）
  constexpr uint rowStrideB = NUM_THREADS / (BN / 4);  // B的行加载步长

  // 在线程寄存器文件中分配线程本地缓存用于存储结果
  // 关键优化：为所有warp子平铺的结果分配寄存器缓存
  float threadResults[WMITER * TM * WNITER * TN] = {0.0};  // 初始化所有结果为0.0
  
  // 在warp平铺级别缓存到寄存器中
  float regM[WMITER * TM] = {0.0};  // 缓存A的所有warp子平铺数据
  float regN[WNITER * TN] = {0.0};  // 缓存B的所有warp子平铺数据

  // 最外层循环：遍历K维度的块（分块矩阵乘法）
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // 调用设备函数从全局内存加载数据到共享内存
    wt::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
        N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
    __syncthreads();  // 同步所有线程，确保共享内存数据加载完成
    
    // 调用设备函数从共享内存处理数据（warp平铺矩阵乘法）
    wt::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
                        TN>(regM, regN, threadResults, As, Bs, warpRow, warpCol,
                            threadRowInWarp, threadColInWarp);
    
    // 前进到下一个块（沿K维度移动）
    A += BK;     // A指针向右移动BK列
    B += BK * N; // B指针向下移动BK行（因为B是列主序存储）
    __syncthreads();  // 同步，确保所有线程完成当前块的计算
  }

  // 将结果写回全局内存（输出矩阵C）
  // 关键优化：按照warp子平铺的顺序写回结果
  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      // 将C指针移动到当前warp子平铺位置
      float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
      
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          // 将C向量加载到寄存器中（向量化加载）
          float4 tmp = reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0];
          
          // 在寄存器中执行GEMM更新：C = alpha*A*B + beta*C
          // 计算threadResults中的索引
          const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;
          tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
          tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
          tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
          tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
          
          // 写回更新后的向量（向量化存储）
          reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0] = tmp;
        }
      }
    }
  }
}