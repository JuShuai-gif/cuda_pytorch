#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// 全局内存合并访问优化：将矩阵分成BLOCKSIZE×BLOCKSIZE的小块
// 优化目标：改善全局内存访问模式，使相邻线程访问连续的内存地址（内存合并）
// 典型值：BLOCKSIZE = 32（一个warp的大小）

template <const uint BLOCKSIZE>
__global__ void sgemm_global_mem_coalesce(int M,int N,int K,float alpha,const float* A,
                            const float* B,float beta,float *C){
    // 计算当前线程对应的输出矩阵C中的位置（cRow行，cCol列）
    // 使用一维线程块（threadIdx.x）但映射到二维输出网格
    // 关键优化：通过threadIdx.x / BLOCKSIZE和threadIdx.x % BLOCKSIZE计算行列索引
    // 这样同一warp中的线程（连续的threadIdx.x）将访问连续的列，实现内存合并访问
    
    // 示例：假设BLOCKSIZE=32，threadIdx.x从0到1023（假设块大小1024）
    // - threadIdx.x=0: cRow=blockIdx.x*32+0, cCol=blockIdx.y*32+0
    // - threadIdx.x=1: cRow=blockIdx.x*32+0, cCol=blockIdx.y*32+1
    // - threadIdx.x=31: cRow=blockIdx.x*32+0, cCol=blockIdx.y*32+31
    // - threadIdx.x=32: cRow=blockIdx.x*32+1, cCol=blockIdx.y*32+0
    // 这样同一warp（0-31）的线程访问同一行的连续32列，实现内存合并
    
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





























