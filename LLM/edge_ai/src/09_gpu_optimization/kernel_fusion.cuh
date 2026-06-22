#pragma once

#include <cuda_runtime.h>

// 分离内核: 卷积输出 -> 偏置 -> ReLU（需要 3 次内核启动）
__global__ void bias_kernel(float *data, const float *bias, int N, int C);

__global__ void relu_kernel(float *data, int N);

// 融合内核: 卷积输出 -> 偏置 -> ReLU 在单次遍历中完成
// 这模拟了 TensorRT 的层融合
__global__ void fused_bias_relu_kernel(float *data, const float *bias, int N, int C);

void demo_kernel_fusion();
