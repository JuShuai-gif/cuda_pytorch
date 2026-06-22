#pragma once

#include <cuda_runtime.h>

#define TILE_SIZE 32
#define TILE_SIZE_OPT 32
#define SMEM_PAD 1 // 填充以避免 bank 冲突

// 使用共享内存的分块矩阵乘法
// 通过将分块加载到共享内存中来减少全局内存访问
__global__ void matmul_tiled(const float *__restrict__ A,
                              const float *__restrict__ B,
                              float *__restrict__ C,
                              int N);

// 优化的分块矩阵乘法，包含以下额外优化:
// - float4 向量化加载以提高内存带宽
// - 循环展开提示
// - 避免 bank 冲突的共享内存填充
__global__ void matmul_optimized(const float *__restrict__ A,
                                  const float *__restrict__ B,
                                  float *__restrict__ C,
                                  int N);
