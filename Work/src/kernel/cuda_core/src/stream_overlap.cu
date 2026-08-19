// Stream overlap probe: single stream vs multiple streams.
//
// Work in the same stream is serialized in FIFO order; work in different
// streams can overlap on idle SMs.  The overlap is only visible when a single
// kernel does not saturate the GPU.  Here each kernel uses only a few blocks
// (a fraction of the 20 SMs): on one stream the four kernels run back-to-back
// leaving most SMs idle, while on four streams they fill the SMs together.
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

#include "cuda_common.h"

using namespace cuda_lab;

constexpr int CHUNK = 4 * 1024 * 1024;  // elements per chunk

// Grid-stride loop so the kernel is correct for any block count, letting us
// deliberately under-utilize the GPU with few blocks per kernel.
__global__ void saxpy_chunk(float* y, const float* x, float a, int n) {
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        y[i] = a * x[i] + y[i];
    }
}

int main() {
    print_device_info();

    const int n_streams = 4;
    const int threads = 256;
    const int blocks_per_chunk = 5;  // 5 of 20 SMs per kernel -> room to overlap

    float *x, *y;
    CUDA_CHECK(cudaMalloc(&x, static_cast<size_t>(CHUNK) * n_streams * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&y, static_cast<size_t>(CHUNK) * n_streams * sizeof(float)));
    CUDA_CHECK(cudaMemset(x, 1, static_cast<size_t>(CHUNK) * n_streams * sizeof(float)));
    CUDA_CHECK(cudaMemset(y, 0, static_cast<size_t>(CHUNK) * n_streams * sizeof(float)));

    cudaStream_t streams[4];
    for (int i = 0; i < n_streams; i++) CUDA_CHECK(cudaStreamCreate(&streams[i]));

    auto run_single = [&]() {
        for (int i = 0; i < n_streams; i++) {
            saxpy_chunk<<<blocks_per_chunk, threads>>>(y + i * CHUNK, x + i * CHUNK, 2.0f, CHUNK);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    };

    auto run_multi = [&]() {
        for (int i = 0; i < n_streams; i++) {
            saxpy_chunk<<<blocks_per_chunk, threads, 0, streams[i]>>>(y + i * CHUNK, x + i * CHUNK, 2.0f, CHUNK);
        }
        for (int i = 0; i < n_streams; i++) CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    };

    // Warm up.
    run_single();
    run_multi();

    const int iters = 50;
    std::vector<double> single_ms, multi_ms;
    for (int k = 0; k < iters; k++) {
        WallTimer w;
        w.start();
        run_single();
        single_ms.push_back(w.ms());
    }
    for (int k = 0; k < iters; k++) {
        WallTimer w;
        w.start();
        run_multi();
        multi_ms.push_back(w.ms());
    }

    JsonReport r;
    r.begin();
    r.put("experiment", "stream_overlap");
    r.put("n_streams", static_cast<long long>(n_streams));
    r.put("chunk_elements", static_cast<long long>(CHUNK));
    r.put("blocks_per_chunk", static_cast<long long>(blocks_per_chunk));
    r.put("single_stream_ms_mean", mean(single_ms));
    r.put("multi_stream_ms_mean", mean(multi_ms));
    r.put("speedup_x", mean(single_ms) / mean(multi_ms));
    std::printf("%s", r.end().c_str());

    for (int i = 0; i < n_streams; i++) CUDA_CHECK(cudaStreamDestroy(streams[i]));
    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(y));
    return 0;
}
