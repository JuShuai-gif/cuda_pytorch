#include "coalesced_demo.cuh"
#include "timer.h"

#include <cstdint>
#include <iomanip>
#include <iostream>

__global__ void coalesced_read(const float *__restrict__ input,
                                float *__restrict__ output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    output[tid] = input[tid] * 2.0f;
}

__global__ void strided_read(const float *__restrict__ input,
                              float *__restrict__ output,
                              int N, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    int idx = static_cast<int>((static_cast<int64_t>(tid) * stride) % N);
    output[idx] = input[idx] * 2.0f;
}

void demo_memory_coalescing() {
    constexpr int N = 16 * 1024 * 1024; // 16M 个浮点数 = 64MB
    constexpr size_t BYTES = N * sizeof(float);
    constexpr int BLOCK_SIZE = 256;

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, BYTES));
    CUDA_CHECK(cudaMalloc(&d_output, BYTES));

    // 使用一个简单内核初始化
    {
        int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        coalesced_read<<<blocks, BLOCK_SIZE>>>(d_input, d_output, N);
    }

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    std::vector<std::pair<std::string, float>> results;

    // 合并访问
    {
        NVTX_RANGE_START("coalesced_access");
        GPUTimer timer;
        timer.start();
        coalesced_read<<<blocks, BLOCK_SIZE>>>(d_input, d_output, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.stop();
        float ms = timer.elapsed_ms();
        results.push_back({"合并访问 (步长=1)", ms});
        NVTX_RANGE_END();
    }

    // 跨步访问 - 步长 = 32（相差一个 warp）
    {
        NVTX_RANGE_START("strided_access_32");
        GPUTimer timer;
        timer.start();
        strided_read<<<blocks, BLOCK_SIZE>>>(d_input, d_output, N, 32);
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.stop();
        float ms = timer.elapsed_ms();
        results.push_back({"跨步访问 (步长=32)", ms});
        NVTX_RANGE_END();
    }

    // 跨步访问 - 步长 = 256
    {
        NVTX_RANGE_START("strided_access_256");
        GPUTimer timer;
        timer.start();
        strided_read<<<blocks, BLOCK_SIZE>>>(d_input, d_output, N, 256);
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.stop();
        float ms = timer.elapsed_ms();
        results.push_back({"跨步访问 (步长=256)", ms});
        NVTX_RANGE_END();
    }

    float baseline = results[0].second;

    std::cout << "  数组大小: " << (N / (1024 * 1024)) << "M 浮点数 ("
              << (BYTES / (1024 * 1024)) << " MB)\n";
    std::cout << "  线程块大小: " << BLOCK_SIZE << " 线程\n\n";
    std::cout << "  " << std::left << std::setw(30) << "访问模式"
              << std::right << std::setw(12) << "时间(ms)"
              << std::setw(12) << "带宽"
              << std::setw(10) << "比率" << "\n";
    std::cout << "  " << std::string(64, '-') << "\n";

    for (const auto &r : results) {
        float bw = (BYTES * 2.0f) / (r.second / 1000.0f) / (1024 * 1024 * 1024);
        // 乘以 2 因为同时有读和写
        std::cout << "  " << std::left << std::setw(30) << r.first
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(12) << r.second
                  << std::fixed << std::setprecision(1)
                  << std::setw(11) << bw << " GB/s"
                  << std::fixed << std::setprecision(2)
                  << std::setw(9) << (r.second / baseline) << "x\n";
    }

    std::cout << "\n  => 合并访问可实现更高的带宽。\n"
              << "  跨步访问浪费了内存事务。\n";

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}
