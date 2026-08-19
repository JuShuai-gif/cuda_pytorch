// Memory coalescing probe.
//
// A warp loads 32 floats either contiguously (thread i -> a[i]) or with a
// large stride (thread i -> a[i * STRIDE]).  The contiguous access is merged
// into a single wide transaction; the strided access forces one transaction
// per thread, collapsing effective bandwidth.  This is the "memory-bound"
// difference that nsys/ncu attribute to poor gld_efficiency.
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

#include "cuda_common.h"

using namespace cuda_lab;

constexpr int STRIDE = 32;

// Coalesced: thread i reads in[i].  The warp's 32 accesses are contiguous.
__global__ void read_coalesced(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * 2.0f;
}

// Strided: thread i reads in[i * STRIDE].  Adjacent threads are STRIDE apart.
__global__ void read_strided(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = i * STRIDE;
    if (idx < n) out[idx] = in[idx] * 2.0f;
}

int main() {
    print_device_info();

    const int n = 1 << 22;  // 4M floats = 16 MB per buffer
    float *in, *out;
    CUDA_CHECK(cudaMalloc(&in, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&out, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMemset(in, 1, static_cast<size_t>(n) * sizeof(float)));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    const int iters = 100;

    EventTimer timer;
    std::vector<double> coalesced_ms, strided_ms;

    // Warm up.
    read_coalesced<<<blocks, threads>>>(in, out, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int k = 0; k < iters; k++) {
        timer.start();
        read_coalesced<<<blocks, threads>>>(in, out, n);
        timer.stop();
        CUDA_CHECK(cudaDeviceSynchronize());
        coalesced_ms.push_back(timer.ms());
    }
    for (int k = 0; k < iters; k++) {
        timer.start();
        read_strided<<<blocks, threads>>>(in, out, n);
        timer.stop();
        CUDA_CHECK(cudaDeviceSynchronize());
        strided_ms.push_back(timer.ms());
    }

    // Bytes moved per kernel = n reads + n writes = 2 * n * 4 bytes.
    const double bytes = 2.0 * n * sizeof(float);
    const double co_mean = mean(coalesced_ms);
    const double st_mean = mean(strided_ms);
    const double co_gbps = (bytes / (co_mean * 1e-3)) / 1e9;
    const double st_gbps = (bytes / (st_mean * 1e-3)) / 1e9;

    JsonReport r;
    r.begin();
    r.put("experiment", "coalescing");
    r.put("n", static_cast<long long>(n));
    r.put("stride", static_cast<long long>(STRIDE));
    r.put("coalesced_ms_mean", co_mean);
    r.put("strided_ms_mean", st_mean);
    r.put("coalesced_gbps", co_gbps);
    r.put("strided_gbps", st_gbps);
    r.put("strided_slowdown_x", st_mean / co_mean);
    std::printf("%s", r.end().c_str());

    CUDA_CHECK(cudaFree(in));
    CUDA_CHECK(cudaFree(out));
    return 0;
}
