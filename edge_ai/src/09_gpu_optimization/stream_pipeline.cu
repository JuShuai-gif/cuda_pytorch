#include "stream_pipeline.cuh"
#include "timer.h"

#include <cmath>
#include <iomanip>
#include <iostream>

__global__ void compute_kernel(float *data, int offset, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    int actual_idx = offset + idx;
    float val = data[actual_idx];
    // 模拟计算负载
    for (int k = 0; k < 256; ++k) {
        val = sinf(val) * scale + cosf(val);
    }
    data[actual_idx] = val;
}

void demo_cuda_streams() {
    constexpr int TOTAL_SIZE = 32 * 1024 * 1024; // 32M 浮点数
    constexpr int CHUNK_SIZE = TOTAL_SIZE / 4;
    constexpr size_t CHUNK_BYTES = CHUNK_SIZE * sizeof(float);
    constexpr int BLOCK_SIZE = 256;
    constexpr int NUM_STREAMS = 4;

    float *h_data;
    CUDA_CHECK(cudaMallocHost(&h_data, TOTAL_SIZE * sizeof(float)));
    for (int i = 0; i < TOTAL_SIZE; ++i) {
        h_data[i] = static_cast<float>(i) * 0.001f;
    }

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, TOTAL_SIZE * sizeof(float)));

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    int blocks = (CHUNK_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 顺序执行（单流）作为对比
    {
        NVTX_RANGE_START("streams_sequential");
        GPUTimer timer;
        timer.start();
        for (int i = 0; i < NUM_STREAMS; ++i) {
            int offset = i * CHUNK_SIZE;
            CUDA_CHECK(cudaMemcpyAsync(d_data + offset, h_data + offset,
                                        CHUNK_BYTES,
                                        cudaMemcpyHostToDevice));
            compute_kernel<<<blocks, BLOCK_SIZE>>>(d_data, offset, CHUNK_SIZE,
                                                    2.0f);
            CUDA_CHECK(cudaMemcpyAsync(h_data + offset, d_data + offset,
                                        CHUNK_BYTES,
                                        cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.stop();
        float ms = timer.elapsed_ms();
        NVTX_RANGE_END();

        std::cout << "  顺序执行（单流）: "
                  << std::fixed << std::setprecision(3) << ms << " ms\n";
    }

    // 重置数据
    for (int i = 0; i < TOTAL_SIZE; ++i) {
        h_data[i] = static_cast<float>(i) * 0.001f;
    }

    // 流水线执行（多流）
    {
        NVTX_RANGE_START("streams_pipelined");
        GPUTimer timer;
        timer.start();
        for (int i = 0; i < NUM_STREAMS; ++i) {
            int offset = i * CHUNK_SIZE;
            cudaStream_t stream = streams[i];
            CUDA_CHECK(cudaMemcpyAsync(d_data + offset, h_data + offset,
                                        CHUNK_BYTES, cudaMemcpyHostToDevice,
                                        stream));
            compute_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
                d_data, offset, CHUNK_SIZE, 2.0f);
            CUDA_CHECK(cudaMemcpyAsync(h_data + offset, d_data + offset,
                                        CHUNK_BYTES, cudaMemcpyDeviceToHost,
                                        stream));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.stop();
        float ms = timer.elapsed_ms();
        NVTX_RANGE_END();

        std::cout << "  流水线执行（" << NUM_STREAMS << " 个流）: "
                  << std::fixed << std::setprecision(3) << ms << " ms\n";
    }

    std::cout << "\n  => 使用多个流时，第 N+1 块的主机到设备传输\n"
              << "  可以与第 N 块的内核执行重叠。\n"
              << "  这在 Nsight Systems 时间线中可见。\n";

    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFreeHost(h_data));
}
