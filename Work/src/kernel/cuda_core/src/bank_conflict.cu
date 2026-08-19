// Shared-memory bank conflict probe.
//
// Shared memory is divided into 32 banks (4 bytes each).  When a warp's
// threads hit distinct addresses in the same bank, the accesses serialize.
// Three access patterns:
//   no conflict : thread i -> s[i]
//   2-way       : thread i -> s[i % 16]  (16 addresses reuse banks)
//   32-way      : thread i -> s[i * 32]  (all threads hit bank 0)
//
// The kernel accumulates the loaded value so the compiler cannot eliminate
// the loads.  This demonstrates why bank-conflict-free layouts matter for
// transpose / reduction / matmul shared-memory tiles.
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

#include "cuda_common.h"

using namespace cuda_lab;

constexpr int N = 1024;  // shared array size (floats)

__global__ void shared_none(const float* in, float* out) {
    __shared__ float s[N];
    int tid = threadIdx.x;
    s[tid] = in[tid];
    __syncthreads();
    float acc = 0.0f;
    for (int k = 0; k < 256; k++) {
        acc += s[(tid + k) % N];  // all 32 threads hit 32 different banks
    }
    out[tid] = acc;
}

__global__ void shared_2way(const float* in, float* out) {
    __shared__ float s[N];
    int tid = threadIdx.x;
    s[tid] = in[tid];
    __syncthreads();
    float acc = 0.0f;
    for (int k = 0; k < 256; k++) {
        // thread i -> address 2i, so thread i and i+16 map to the same bank
        // (bank (2i)%32 == bank (2i+32)%32) but different addresses.
        acc += s[((tid * 2) + k) % N];
    }
    out[tid] = acc;
}

__global__ void shared_32way(const float* in, float* out) {
    __shared__ float s[N];
    int tid = threadIdx.x;
    s[tid] = in[tid];
    __syncthreads();
    float acc = 0.0f;
    for (int k = 0; k < 256; k++) {
        acc += s[((tid + k) * 32) % N];  // every thread hits bank (tid*32)%32 = 0
    }
    out[tid] = acc;
}

int main() {
    print_device_info();

    float *in, *out;
    CUDA_CHECK(cudaMalloc(&in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&out, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(in, 1, N * sizeof(float)));

    const int threads = 256;  // one block of 256 threads
    const int iters = 200;

    EventTimer timer;
    std::vector<double> none_ms, two_ms, full_ms;

    shared_none<<<1, threads>>>(in, out);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int k = 0; k < iters; k++) {
        timer.start();
        shared_none<<<1, threads>>>(in, out);
        timer.stop();
        CUDA_CHECK(cudaDeviceSynchronize());
        none_ms.push_back(timer.ms());
    }
    for (int k = 0; k < iters; k++) {
        timer.start();
        shared_2way<<<1, threads>>>(in, out);
        timer.stop();
        CUDA_CHECK(cudaDeviceSynchronize());
        two_ms.push_back(timer.ms());
    }
    for (int k = 0; k < iters; k++) {
        timer.start();
        shared_32way<<<1, threads>>>(in, out);
        timer.stop();
        CUDA_CHECK(cudaDeviceSynchronize());
        full_ms.push_back(timer.ms());
    }

    JsonReport r;
    r.begin();
    r.put("experiment", "bank_conflict");
    r.put("threads", static_cast<long long>(threads));
    r.put("none_ms_mean", mean(none_ms));
    r.put("2way_ms_mean", mean(two_ms));
    r.put("32way_ms_mean", mean(full_ms));
    r.put("2way_slowdown_x", mean(two_ms) / mean(none_ms));
    r.put("32way_slowdown_x", mean(full_ms) / mean(none_ms));
    std::printf("%s", r.end().c_str());

    CUDA_CHECK(cudaFree(in));
    CUDA_CHECK(cudaFree(out));
    return 0;
}
