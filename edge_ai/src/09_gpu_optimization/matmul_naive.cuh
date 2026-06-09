#pragma once

#include <cuda_runtime.h>

// 朴素矩阵乘法: C = A * B
// 每个线程计算 C 的一个元素
__global__ void matmul_naive(const float *__restrict__ A,
                              const float *__restrict__ B,
                              float *__restrict__ C,
                              int N);
