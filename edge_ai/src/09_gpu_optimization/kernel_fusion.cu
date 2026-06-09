#include "kernel_fusion.cuh"
#include "timer.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

__global__ void bias_kernel(float *data, const float *bias,
                             int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int c = idx % C;
    data[idx] += bias[c];
}

__global__ void relu_kernel(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    data[idx] = fmaxf(0.0f, data[idx]);
}

__global__ void fused_bias_relu_kernel(float *data, const float *bias,
                                        int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int c = idx % C;
    float val = data[idx] + bias[c];
    data[idx] = fmaxf(0.0f, val);
}

void demo_kernel_fusion() {
    constexpr int BATCH = 4;
    constexpr int C = 64;
    constexpr int H = 256;
    constexpr int W = 256;
    constexpr int N = BATCH * C * H * W;
    constexpr size_t DATA_BYTES = N * sizeof(float);
    constexpr size_t BIAS_BYTES = C * sizeof(float);
    constexpr int BLOCK_SIZE = 256;
    constexpr int BLOCKS = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    constexpr int NUM_ITERS = 100; // 重复以获得稳定计时

    float *d_data;
    float *d_bias;
    CUDA_CHECK(cudaMalloc(&d_data, DATA_BYTES));
    CUDA_CHECK(cudaMalloc(&d_bias, BIAS_BYTES));

    // 初始化偏置
    std::vector<float> h_bias(C);
    for (int i = 0; i < C; ++i) {
        h_bias[i] = static_cast<float>(i) * 0.1f - 3.0f;
    }
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), BIAS_BYTES,
                          cudaMemcpyHostToDevice));

    float separate_ms = 0.0f, fused_ms = 0.0f;

    // 分离内核
    {
        NVTX_RANGE_START("kernel_fusion_separate");
        GPUTimer timer;
        timer.start();
        for (int iter = 0; iter < NUM_ITERS; ++iter) {
            bias_kernel<<<BLOCKS, BLOCK_SIZE>>>(d_data, d_bias, N, C);
            relu_kernel<<<BLOCKS, BLOCK_SIZE>>>(d_data, N);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.stop();
        separate_ms = timer.elapsed_ms() / NUM_ITERS;
        NVTX_RANGE_END();
    }

    // 融合内核
    {
        NVTX_RANGE_START("kernel_fusion_fused");
        GPUTimer timer;
        timer.start();
        for (int iter = 0; iter < NUM_ITERS; ++iter) {
            fused_bias_relu_kernel<<<BLOCKS, BLOCK_SIZE>>>(d_data, d_bias, N, C);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.stop();
        fused_ms = timer.elapsed_ms() / NUM_ITERS;
        NVTX_RANGE_END();
    }

    std::cout << "  数据大小: " << BATCH << "x" << C << "x" << H << "x" << W
              << " = " << (N / (1024.0f * 1024.0f)) << "M 元素\n";
    std::cout << "  平均迭代次数: " << NUM_ITERS << "\n\n";
    std::cout << "  " << std::left << std::setw(35) << "方法"
              << std::right << std::setw(12) << "平均(ms)"
              << std::setw(12) << "加速比" << "\n";
    std::cout << "  " << std::string(59, '-') << "\n";
    std::cout << "  " << std::left << std::setw(35)
              << "分离 (Bias -> ReLU)"
              << std::right << std::fixed << std::setprecision(4)
              << std::setw(12) << separate_ms
              << std::setw(12) << "1.00x\n";
    std::cout << "  " << std::left << std::setw(35)
              << "融合 (Bias+ReLU)"
              << std::right << std::fixed << std::setprecision(4)
              << std::setw(12) << fused_ms
              << std::fixed << std::setprecision(2)
              << std::setw(11) << (separate_ms / fused_ms) << "x\n";

    std::cout << "\n  => 融合消除了一次 GPU 内核启动开销\n"
              << "  以及一次完整的全局内存遍历。\n"
              << "  这是 TensorRT 层融合背后的核心理念。\n";

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_bias));
}
