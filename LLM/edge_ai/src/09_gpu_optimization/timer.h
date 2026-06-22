#pragma once

#include <cuda_runtime.h>
#include <iostream>

// 用于 Nsight 分析的 NVTX 注解
#ifdef USE_NVTX
#include <nvToolsExt.h>
#define NVTX_RANGE_START(name) nvtxRangePushA(name)
#define NVTX_RANGE_END() nvtxRangePop()
#define NVTX_MARK(name) nvtxMarkA(name)
#else
#define NVTX_RANGE_START(name)
#define NVTX_RANGE_END()
#define NVTX_MARK(name)
#endif

// 错误检查宏
#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            std::cerr << "CUDA 错误位于 " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl;  \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

// 使用 CUDA 事件的 GPU 计时器
class GPUTimer {
public:
    GPUTimer(cudaStream_t stream = 0) : stream_(stream) {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GPUTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() {
        CUDA_CHECK(cudaEventRecord(start_, stream_));
    }
    void stop() {
        CUDA_CHECK(cudaEventRecord(stop_, stream_));
    }

    float elapsed_ms() {
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_, stop_;
    cudaStream_t stream_;
};
