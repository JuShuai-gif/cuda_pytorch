#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// 双缓冲优化命名空间：将优化函数组织在命名空间中
namespace db {

// 从全局内存加载数据到共享内存的设备函数（与第10个文件类似）
template <const int BM, const int BN, const int BK, const int rowStrideA,
          const int rowStrideB>
__device__ void loadFromGmem(const int N, const int K, float *A, float *B,
                             float *As, float *Bs, const int innerRowA,
                             const int innerColA, const int innerRowB,
                             const int innerColB) {
  // 加载A矩阵数据：使用向量化加载和转置存储
  for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
    float4 tmp = reinterpret_cast<float4 *>(
        &A[(innerRowA + offset) * K + innerColA * 4])[0];
    // 存储A时进行转置（列主序）
    As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
  }

  // 加载B矩阵数据：直接向量化存储
  for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
    reinterpret_cast<float4 *>(
        &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
        reinterpret_cast<float4 *>(
            &B[(innerRowB + offset) * N + innerColB * 4])[0];
  }
}

// 从共享内存处理数据的设备函数（warp平铺矩阵乘法核心计算）
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void
processFromSmem(float *regM, float *regN, float *threadResults, const float *As,
                const float *Bs, const uint warpRow, const uint warpCol,
                const uint threadRowInWarp, const uint threadColInWarp) {
  // 遍历K维度（内积维度），执行warp平铺矩阵乘法
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    // 为整个warp平铺填充寄存器
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint i = 0; i < TM; ++i) {
        regM[wSubRowIdx * TM + i] =
            As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
               threadRowInWarp * TM + i];
      }
    }
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      for (uint i = 0; i < TN; ++i) {
        regN[wSubColIdx * TN + i] =
            Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
               threadColInWarp * TN + i];
      }
    }

    // 执行warp平铺矩阵乘法
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
          for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
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

} // namespace db

// 双缓冲优化内核：使用双缓冲技术隐藏内存加载延迟
// 模板参数与第10个文件相同，但添加了双缓冲优化
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    sgemmDoubleBuffering(const int M, const int N, const int K,
                         const float alpha, float *A, float *B, float beta,
                         float *C) {
  const uint cRow = blockIdx.y;  // 块行索引
  const uint cCol = blockIdx.x;  // 块列索引

  // warp在线程块平铺中的位置（与第10个文件相同）
  const uint warpIdx = threadIdx.x / WARPSIZE;
  const uint warpCol = warpIdx % (BN / WN);
  const uint warpRow = warpIdx / (BN / WN);

  // warp子平铺的大小计算（与第10个文件相同）
  constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr uint WSUBM = WM / WMITER; // warp子平铺的M维度大小
  constexpr uint WSUBN = WN / WNITER; // warp子平铺的N维度大小

  // 线程在warp子平铺中的位置（与第10个文件相同）
  const uint threadIdxInWarp = threadIdx.x % WARPSIZE;
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

  // 关键优化：为双缓冲分配两倍的共享内存空间
  // As和Bs都有两个缓冲区：缓冲区0和缓冲区1
  __shared__ float As[2 * BM * BK];  // 双缓冲A矩阵共享内存
  __shared__ float Bs[2 * BK * BN];  // 双缓冲B矩阵共享内存

  // 设置双缓冲分割：将线程分为两组
  // doubleBufferIdx = 0: 处理缓冲区0，加载缓冲区1
  // doubleBufferIdx = 1: 处理缓冲区1，加载缓冲区0
  bool doubleBufferIdx = threadIdx.x >= (NUM_THREADS / 2);

  // 将指针移动到当前块对应的起始位置：
  A += cRow * BM * K;  // 移动到A矩阵第cRow个BM行的起始位置
  B += cCol * BN;      // 移动到B矩阵第cCol个BN列的起始位置
  
  // 将C指针直接移动到warp的输出平铺位置
  C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

  // 计算此线程将加载到SMEM中的索引
  // 关键优化：对于加载操作，我们假装只有实际线程数的一半
  // 这是因为双缓冲将线程分为两组，每组负责不同的缓冲区
  const uint innerRowA = (threadIdx.x % (NUM_THREADS / 2)) / (BK / 4);
  const uint innerColA = (threadIdx.x % (NUM_THREADS / 2)) % (BK / 4);
  constexpr uint rowStrideA = ((NUM_THREADS / 2) * 4) / BK;  // 使用一半线程数的行步长
  
  const uint innerRowB = (threadIdx.x % (NUM_THREADS / 2)) / (BN / 4);
  const uint innerColB = (threadIdx.x % (NUM_THREADS / 2)) % (BN / 4);
  constexpr uint rowStrideB = (NUM_THREADS / 2) / (BN / 4);  // 使用一半线程数的行步长

  // 在线程寄存器文件中分配线程本地缓存用于存储结果
  float threadResults[WMITER * TM * WNITER * TN] = {0.0};  // 初始化所有结果为0.0
  
  // 在warp平铺级别缓存到寄存器中
  float regM[WMITER * TM] = {0.0};  // 缓存A的所有warp子平铺数据
  float regN[WNITER * TN] = {0.0};  // 缓存B的所有warp子平铺数据

  // 初始化：第一组线程（doubleBufferIdx == 0）加载第一个缓冲区（B0）
  if (doubleBufferIdx == 0) {
    // 加载第一个缓冲区（B0）
    db::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
        N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
  }
  __syncthreads();  // 同步所有线程，确保第一个缓冲区加载完成

  // 最外层循环：遍历K维度的块，每次迭代处理2*BK（双缓冲）
  // 关键优化：双缓冲流水线，隐藏内存加载延迟
  for (uint bkIdx = 0; bkIdx < K; bkIdx += 2 * BK) {
    if (doubleBufferIdx == 0) {
      // 第一组线程：处理当前缓冲区（B0），加载下一个缓冲区（B1）
      
      // 处理当前缓冲区（B0）
      db::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
                          TN>(regM, regN, threadResults, As, Bs, warpRow,
                              warpCol, threadRowInWarp, threadColInWarp);
      __syncthreads();

      // 处理下一个缓冲区（B1），如果存在
      if (bkIdx + BK < K) {
        db::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN,
                            TM, TN>(regM, regN, threadResults, As + (BM * BK),
                                    Bs + (BK * BN), warpRow, warpCol,
                                    threadRowInWarp, threadColInWarp);
      }
      __syncthreads();

      // 加载下下个缓冲区（B0），为下一次迭代做准备
      if (bkIdx + 2 * BK < K) {
        db::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
            N, K, A + 2 * BK, B + 2 * BK * N, As, Bs, innerRowA, innerColA,
            innerRowB, innerColB);
      }
    } else {
      // 第二组线程：加载当前缓冲区（B1），处理上一个缓冲区（B0）
      
      // 加载当前缓冲区（B1）
      if (bkIdx + BK < K) {
        db::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
            N, K, A + BK, B + BK * N, As + (BM * BK), Bs + (BK * BN), innerRowA,
            innerColA, innerRowB, innerColB);
      }
      __syncthreads();

      // 处理当前缓冲区（B0）
      db::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
                          TN>(regM, regN, threadResults, As, Bs, warpRow,
                              warpCol, threadRowInWarp, threadColInWarp);
      __syncthreads();

      // 处理下一个缓冲区（B1），如果存在
      if (bkIdx + BK < K) {
        db::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN,
                            TM, TN>(regM, regN, threadResults, As + (BM * BK),
                                    Bs + (BK * BN), warpRow, warpCol,
                                    threadRowInWarp, threadColInWarp);
      }
    }

    // 前进两个BK（因为双缓冲每次处理两个块）
    A += 2 * BK;     // A指针向右移动2*BK列
    B += 2 * BK * N; // B指针向下移动2*BK行
    __syncthreads();  // 同步所有线程
  }

  // write out the results
  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      // move C pointer to current warp subtile
      float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          // load C vector into registers
          float4 tmp = reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0];
          // perform GEMM update in reg
          const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;
          tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
          tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
          tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
          tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
          // write back
          reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0] = tmp;
        }
      }
    }
  }
}