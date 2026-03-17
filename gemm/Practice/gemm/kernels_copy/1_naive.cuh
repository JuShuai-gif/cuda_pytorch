#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// 朴素矩阵乘法内核：MN = MK * KN（M×N矩阵 = M×K矩阵 × K×N矩阵）

__global__ void sgemm_naive(int M,int N,int K,float alpha,const float* A,
                            const float* B,float beta,float *C){
    // 计算当前线程对应的输出矩阵C中的位置（x行，y列）
    // 使用二维网格和二维线程块：blockIdx.x/y是块索引，threadIdx.x/y是线程在块内的索引
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;  // 行索引（0到M-1）
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;  // 列索引（0到N-1）

    // 边界检查：确保线程在矩阵范围内
    if (x < M && y < N)
    {
        float tmp = 0.0;  // 临时变量，用于累加点积结果
        
        // 计算C[x][y] = Σ(A[x][i] * B[i][y])，i从0到K-1
        // 这是标准的矩阵乘法公式：C的第x行第y列 = A的第x行与B的第y列的点积
        for (int i = 0; i < K; ++i)
        {
            // x表示行（竖着的维度），y表示列（横着的维度）
            // A[x * K + i]: 访问A矩阵第x行第i列（行主序存储）
            // B[i * N + y]: 访问B矩阵第i行第y列（行主序存储）
            tmp += A[x * K + i] * B[i * N + y];
        }

        // 应用缩放因子：C = α*(A@B) + β*C
        // 这是BLAS标准接口：C ← α·A·B + β·C
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}
























