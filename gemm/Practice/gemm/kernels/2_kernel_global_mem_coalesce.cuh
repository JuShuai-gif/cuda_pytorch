#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

template <const uint BLOCKSIZE>
__global__ void sgemm_global_mem_coalesce(int M,int N,int K,float alpha,const float* A,
                            const float* B,float beta,float *C){

    const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);  // 行索引
    const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);  // 列索引

    // 边界检查：确保线程在矩阵范围内
    if (cRow < M && cCol < N) {
        float tmp = 0.0;  // 临时变量，用于累加点积结果
        
        // 计算C[cRow][cCol] = Σ(A[cRow][i] * B[i][cCol])，i从0到K-1
        for (int i = 0; i < K; ++i) {
            tmp += A[cRow * K + i] * B[i * N + cCol];
        }
        
        // 应用缩放因子：C = α*(A@B) + β*C
        C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
    }
}















