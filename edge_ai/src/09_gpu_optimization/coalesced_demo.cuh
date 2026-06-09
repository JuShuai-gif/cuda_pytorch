#pragma once

#include <cuda_runtime.h>
#include <string>
#include <vector>

// 合并访问：相邻线程读取相邻内存位置
__global__ void coalesced_read(const float *__restrict__ input,
                                float *__restrict__ output, int N);

// 跨步访问：线程以跨步方式读取，导致多次内存事务
__global__ void strided_read(const float *__restrict__ input,
                              float *__restrict__ output,
                              int N, int stride);

void demo_memory_coalescing();
