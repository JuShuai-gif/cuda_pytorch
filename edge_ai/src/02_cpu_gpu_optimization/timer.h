#pragma once

#include <chrono>
#include <cuda_runtime.h>

// ============================================================================
// 用于基准测试的高精度 CPU 计时器
// ============================================================================
class CpuTimer {
public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

// ============================================================================
// 基于 CUDA 事件的 GPU 计时器，用于精确的内核计时
// ============================================================================
class GpuTimer {
public:
    GpuTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start(cudaStream_t stream = 0) {
        cudaEventRecord(start_, stream);
    }
    void stop(cudaStream_t stream = 0) {
        cudaEventRecord(stop_, stream);
    }
    float elapsed_ms() const {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};
