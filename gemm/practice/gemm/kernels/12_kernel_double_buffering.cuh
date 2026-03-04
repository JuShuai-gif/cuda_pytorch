#pragma once

#include <algorithm>
#include <cassert>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda/barrier>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// 匿名命名空间：包含使用CUDA异步内存复制的设备函数
namespace {

// 从全局内存加载数据到共享内存的设备函数（使用CUDA异步内存复制）
// 关键优化：使用cuda::memcpy_async进行异步内存复制，隐藏内存延迟
template <const int BM, const int BN, const int BK, const int rowStrideA,
          const int rowStrideB, typename T>
__device__ void loadFromGmem(int N, int K, float *A, float *B, float *As,
                             float *Bs, int innerRowA, int innerColA,
                             int innerRowB, int innerColB, T &barrier) {

  // 加载A矩阵数据：使用异步内存复制
  // 关键优化：cuda::memcpy_async允许在计算进行时异步加载数据
  for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
    // 异步复制A的4个元素（分别复制，因为需要转置存储）
    cuda::memcpy_async(&As[(innerColA * 4 + 0) * BM + innerRowA + offset],
                       &A[(innerRowA + offset) * K + innerColA * 4],
                       cuda::aligned_size_t<sizeof(float)>(sizeof(float)),
                       barrier);
    cuda::memcpy_async(&As[(innerColA * 4 + 1) * BM + innerRowA + offset],
                       &A[(innerRowA + offset) * K + innerColA * 4 + 1],
                       cuda::aligned_size_t<sizeof(float)>(sizeof(float)),
                       barrier);
    cuda::memcpy_async(&As[(innerColA * 4 + 2) * BM + innerRowA + offset],
                       &A[(innerRowA + offset) * K + innerColA * 4 + 2],
                       cuda::aligned_size_t<sizeof(float)>(sizeof(float)),
                       barrier);
    cuda::memcpy_async(&As[(innerColA * 4 + 3) * BM + innerRowA + offset],
                       &A[(innerRowA + offset) * K + innerColA * 4 + 3],
                       cuda::aligned_size_t<sizeof(float)>(sizeof(float)),
                       barrier);
  }

  // 加载B矩阵数据：使用异步内存复制（向量化）
  for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
    // 异步复制B的4个元素（一次性复制float4）
    cuda::memcpy_async(&Bs[(innerRowB + offset) * BN + innerColB * 4],
                       &B[(innerRowB + offset) * N + innerColB * 4],
                       cuda::aligned_size_t<sizeof(float4)>(sizeof(float4)),
                       barrier);
  }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void
processFromSmem(float *regM, float *regN, float *threadResults, const float *As,
                const float *Bs, const uint warpRow, const uint warpCol,
                const uint threadRowInWarp, const uint threadColInWarp) {
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    // populate registers for whole warptile
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

    // execute warptile matmul
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        // calculate per-thread results
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

} // namespace

/*
 * 高级双缓冲优化内核：使用CUDA异步内存复制和屏障进行双缓冲
 * 模板参数与第11个文件相同
 * 关键优化：使用cuda::memcpy_async和cuda::barrier实现高效的双缓冲流水线
 */
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    runSgemmDoubleBuffering2(int M, int N, int K, float alpha, float *A,
                             float *B, float beta, float *C) {
  // 使用协作组和CUDA屏障进行同步
  auto block = cooperative_groups::this_thread_block();
  
  // 关键优化：使用CUDA屏障进行细粒度同步
  __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> frontBarrier;
  __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> backBarrier;
  auto frontBarrierPtr = &frontBarrier;
  auto backBarrierPtr = &backBarrier;
  
  // 初始化屏障（只需要一个线程执行）
  if (block.thread_rank() == 0) {
    init(&frontBarrier, block.size());
    init(&backBarrier, block.size());
  }
  __syncthreads();

  const uint cRow = blockIdx.y;  // 块行索引
  const uint cCol = blockIdx.x;  // 块列索引

  // warp在线程块平铺中的位置（与前面文件相同）
  const uint warpIdx = threadIdx.x / WARPSIZE;
  const uint warpCol = warpIdx % (BN / WN);
  const uint warpRow = warpIdx / (BN / WN);

  // warp子平铺的大小计算（与前面文件相同）
  constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr uint WSUBM = WM / WMITER; // warp子平铺的M维度大小
  constexpr uint WSUBN = WN / WNITER; // warp子平铺的N维度大小

  // 线程在warp子平铺中的位置（与前面文件相同）
  const uint threadIdxInWarp = threadIdx.x % WARPSIZE;
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

  // 为双缓冲分配两倍的共享内存空间
  __shared__ float As[2 * BM * BK];  // 双缓冲A矩阵共享内存
  __shared__ float Bs[2 * BK * BN];  // 双缓冲B矩阵共享内存

  // 将指针移动到当前块对应的起始位置：
  A += cRow * BM * K;  // 移动到A矩阵第cRow个BM行的起始位置
  B += cCol * BN;      // 移动到B矩阵第cCol个BN列的起始位置
  
  // 将C指针直接移动到warp的输出平铺位置
  C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

  // 计算此线程将加载到SMEM中的索引（与前面文件相同）
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);
  constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

  // 在线程寄存器文件中分配线程本地缓存用于存储结果
  float threadResults[WMITER * TM * WNITER * TN] = {0.0};
  float regM[WMITER * TM] = {0.0};  // 缓存A的所有warp子平铺数据
  float regN[WNITER * TN] = {0.0};  // 缓存B的所有warp子平铺数据

  // 双缓冲偏移：跟踪当前使用的缓冲区
  int As_offset = 0;  // A矩阵当前缓冲区偏移（0或1）
  int Bs_offset = 0;  // B矩阵当前缓冲区偏移（0或1）

  // 双缓冲：将第一个块平铺加载到SMEM中（使用异步内存复制）
  loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
      N, K, A, B, As + As_offset * BM * BK, Bs + Bs_offset * BK * BN, innerRowA,
      innerColA, innerRowB, innerColB, (*frontBarrierPtr));

  // 最外层循环：遍历K维度的块（使用异步内存复制的双缓冲）
  // 关键优化：使用两个屏障（frontBarrier和backBarrier）实现流水线
  for (uint bkIdx = 0; bkIdx < K - BK; bkIdx += BK) {
    // 双缓冲：异步加载下一个块平铺到SMEM中（使用backBarrier）
    loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
        N, K, A + BK, B + BK * N, As + (1 - As_offset) * BM * BK,
        Bs + (1 - Bs_offset) * BK * BN, innerRowA, innerColA, innerRowB,
        innerColB, (*backBarrierPtr));

    // 计算当前块平铺：等待frontBarrier的异步复制完成
    (*frontBarrierPtr).arrive_and_wait();
    processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
        regM, regN, threadResults, As + As_offset * BM * BK,
        Bs + Bs_offset * BK * BN, warpRow, warpCol, threadRowInWarp,
        threadColInWarp);
    
    // 前进到下一个块
    A += BK;     // A指针向右移动BK列
    B += BK * N; // B指针向下移动BK行

    // 切换缓冲区偏移：0->1 或 1->0
    As_offset = 1 - As_offset;
    Bs_offset = 1 - Bs_offset;
    
    // 交换前后屏障：实现双缓冲流水线
    auto tmp = frontBarrierPtr;
    frontBarrierPtr = backBarrierPtr;
    backBarrierPtr = tmp;

    __syncthreads();  // 同步所有线程
  }

  // 计算最后一个块平铺：等待最后一个异步复制完成
  (*frontBarrierPtr).arrive_and_wait();
  processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
      regM, regN, threadResults, As + As_offset * BM * BK,
      Bs + Bs_offset * BK * BN, warpRow, warpCol, threadRowInWarp,
      threadColInWarp);

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