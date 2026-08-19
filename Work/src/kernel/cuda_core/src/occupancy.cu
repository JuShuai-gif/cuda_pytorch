// Register pressure vs occupancy probe.
//
// The SM has a fixed register file (regsPerBlock = 65536).  More registers
// per thread means fewer resident threads, lowering occupancy.  The compiler
// decides register usage, and __launch_bounds__ steers it:
//
//   __launch_bounds__(256, minBlocks) tells the compiler "at least minBlocks
//   blocks of 256 threads must fit per SM", which caps registers/thread at
//   regsPerBlock / (256 * minBlocks) and forces spills if the kernel needs
//   more.  Fewer minBlocks lets the compiler use many registers per thread.
//
// This experiment queries the occupancy API for the first-hand numbers:
// registers/thread, local (spill) bytes, blocks/SM, and achieved occupancy.
#include <cuda_runtime.h>

#include <cstdio>

#include "cuda_common.h"

using namespace cuda_lab;

constexpr int N = 1 << 20;

// Keep a 64-float array live so the compiler must either keep it in registers
// or spill it to local (global) memory.
__global__ void kernel_default(const float* in, float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float acc[64];
    #pragma unroll
    for (int j = 0; j < 64; j++) acc[j] = in[j] * (i + 1);
    float s = 0.0f;
    #pragma unroll
    for (int j = 0; j < 64; j++) s += acc[j];
    if (i < N) out[i] = s;
}

// Force at least 6 blocks/SM of 256 threads => at most ~42 registers/thread.
__global__ void __launch_bounds__(256, 6) kernel_high_occupancy(const float* in, float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float acc[64];
    #pragma unroll
    for (int j = 0; j < 64; j++) acc[j] = in[j] * (i + 1);
    float s = 0.0f;
    #pragma unroll
    for (int j = 0; j < 64; j++) s += acc[j];
    if (i < N) out[i] = s;
}

// Allow as few as 1 block/SM of 256 threads => up to 256 registers/thread.
__global__ void __launch_bounds__(256, 1) kernel_low_occupancy(const float* in, float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float acc[64];
    #pragma unroll
    for (int j = 0; j < 64; j++) acc[j] = in[j] * (i + 1);
    float s = 0.0f;
    #pragma unroll
    for (int j = 0; j < 64; j++) s += acc[j];
    if (i < N) out[i] = s;
}

template <typename F>
void report_kernel(const char* name, F kernel) {
    cudaFuncAttributes attr{};
    CUDA_CHECK(cudaFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel)));

    int max_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, kernel, 256, 0));

    int threads_per_sm = max_blocks * 256;
    double occupancy = static_cast<double>(threads_per_sm) / 1536.0;

    JsonReport r;
    r.begin();
    r.put("kernel", name);
    r.put("registers_per_thread", static_cast<long long>(attr.numRegs));
    r.put("local_bytes_per_thread", static_cast<long long>(attr.localSizeBytes));
    r.put("max_blocks_per_sm", static_cast<long long>(max_blocks));
    r.put("threads_per_sm", static_cast<long long>(threads_per_sm));
    r.put("occupancy_frac", occupancy);
    std::printf("%s", r.end().c_str());
}

int main() {
    print_device_info();

    float *in, *out;
    CUDA_CHECK(cudaMalloc(&in, 64 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&out, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(in, 1, 64 * sizeof(float)));

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    // Warm up so the first-launch cost is not measured.
    kernel_default<<<blocks, threads>>>(in, out);
    CUDA_CHECK(cudaDeviceSynchronize());

    report_kernel("high_occupancy", kernel_high_occupancy);
    report_kernel("default", kernel_default);
    report_kernel("low_occupancy", kernel_low_occupancy);

    CUDA_CHECK(cudaFree(in));
    CUDA_CHECK(cudaFree(out));
    return 0;
}
