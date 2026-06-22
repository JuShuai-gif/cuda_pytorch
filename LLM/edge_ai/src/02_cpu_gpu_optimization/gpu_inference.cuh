#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// ============================================================================
// 检查 CUDA 错误
// ============================================================================
#define CUDA_CHECK(err) \
    do { \
        cudaError_t err_ = (err); \
        if (err_ != cudaSuccess) { \
            std::cerr << "CUDA 错误位于 " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err_) << "\n"; \
            exit(1); \
        } \
    } while(0)

// ============================================================================
// GPU 卷积: 在 CxHxW float32 图像张量上应用 N 个 3x3 卷积核。
// 输入布局: [C][H][W] (通道优先，用于合并访问)。
// 权重布局: [C_out][C_in][3][3]。
// 输出布局: [C_out][H-2][W-2] (valid 卷积)。
// 调用者需管理输入、权重、输出的设备内存。
// ============================================================================
void gpu_conv2d(const float* d_input, const float* d_weights,
                float* d_output,
                int H, int W, int C_in, int C_out,
                cudaStream_t stream = 0);

// ============================================================================
// GPU ReLU 激活: 对一维数组逐元素计算 max(0, x)。
// ============================================================================
void gpu_relu(float* d_data, int total_elements, cudaStream_t stream = 0);

// ============================================================================
// GPU 2x2 最大池化，步长 2，在 CxHxW 特征图上操作。
// 输出维度为 C x (H/2) x (W/2)。
// ============================================================================
void gpu_maxpool(const float* d_input, float* d_output,
                 int H, int W, int C,
                 cudaStream_t stream = 0);

// ============================================================================
// GPU 检测头: 将 CxHxW 特征图解码为每个空间位置的 N_det x H x W
// 检测参数 (N_det = 5: 置信度, x, y, w, h)。
// 使用线性层: output[d][h][w] = sum_c(feature[c][h][w] * W[d][c]) + b[d]
// ============================================================================
void gpu_detection_head(const float* d_features, const float* d_head_weights,
                        const float* d_head_bias,
                        float* d_output,
                        int H, int W, int C_in, int N_det,
                        cudaStream_t stream = 0);

// ============================================================================
// 在设备上分配并初始化卷积权重。
// 返回设备指针。调用者需 cudaFree。
// ============================================================================
float* gpu_alloc_init_conv_weights(int C_in, int C_out, int seed = 42);

// ============================================================================
// 在设备上分配并初始化检测头权重。
// ============================================================================
void gpu_alloc_init_head_weights(float** d_weights, float** d_bias,
                                 int C_in, int N_det, int seed = 123);
