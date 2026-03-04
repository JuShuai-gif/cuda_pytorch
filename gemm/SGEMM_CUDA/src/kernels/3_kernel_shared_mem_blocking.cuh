#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
  // the output block that we want to compute in this threadblock
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  // allocate buffer for current block in fast shared mem
  // shared mem is shared between all threads in a block
  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  // the inner row & col that we're accessing in this thread
  // 将一维线程索引映射到二维网格坐标：
  // threadCol = threadIdx.x % BLOCKSIZE 计算线程在块内的列索引（0到BLOCKSIZE-1）
  // threadRow = threadIdx.x / BLOCKSIZE 计算线程在块内的行索引（0到BLOCKSIZE-1）
  // 这样每个线程对应输出矩阵C中一个BLOCKSIZE×BLOCKSIZE块内的一个元素位置
  const uint threadCol = threadIdx.x % BLOCKSIZE;
  const uint threadRow = threadIdx.x / BLOCKSIZE;

  // advance pointers to the starting positions
  // 计算当前线程块对应的矩阵起始位置：
  // A矩阵：移动到第cRow个BLOCKSIZE行的起始位置（每个BLOCKSIZE行有K列）
  A += cRow * BLOCKSIZE * K;                    // row=cRow, col=0
  // B矩阵：移动到第cCol个BLOCKSIZE列的起始位置（每列有1个元素，但实际是列主序存储）
  B += cCol * BLOCKSIZE;                        // row=0, col=cCol
  // C矩阵：移动到输出矩阵中当前线程块对应的位置（第cRow行，第cCol列）
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row=cRow, col=cCol

  float tmp = 0.0;
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    // Have each thread load one of the elements in A & B
    // Make the threadCol (=threadIdx.x) the consecutive index
    // to allow global memory access coalescing
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

    // block threads in this block until cache is fully populated
    __syncthreads();
    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    // execute the dotproduct on the currently cached block
    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
      tmp += As[threadRow * BLOCKSIZE + dotIdx] *
             Bs[dotIdx * BLOCKSIZE + threadCol];
    }
    // need to sync again at the end, to avoid faster threads
    // fetching the next block into the cache before slower threads are done
    __syncthreads();
  }
  C[threadRow * N + threadCol] =
      alpha * tmp + beta * C[threadRow * N + threadCol];
}