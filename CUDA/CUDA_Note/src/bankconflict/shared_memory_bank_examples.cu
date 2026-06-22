#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                       \
  do {                                                                        \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(err__));                                \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                         \
  } while (0)

__device__ __forceinline__ void fill_smem(uint32_t *smem, uint32_t tid) {
  for (int i = 0; i < 4; ++i) {
    smem[i * 32 + tid] = tid + i * 100;
  }
}

// LDS.64-like pattern: each thread reads one uint2.
// Expected shape: two half-warp transactions, no bank conflict.
__global__ void uint2_contiguous(uint2 *out) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  fill_smem(smem, tid);
  __syncthreads();

  out[tid] = reinterpret_cast<const uint2 *>(smem)[tid];
}

// LDS.64-like pattern where t0/t1, t2/t3, ... read the same uint2.
// This satisfies the i xor 1 broadcast/merge condition discussed in the PDF.
__global__ void uint2_pair_broadcast(uint2 *out) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  fill_smem(smem, tid);
  __syncthreads();

  out[tid] = reinterpret_cast<const uint2 *>(smem)[tid / 2];
}

// LDS.128-like pattern: each thread reads one uint4.
// Expected shape: four quarter-warp transactions, no bank conflict.
__global__ void uint4_contiguous(uint4 *out) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  fill_smem(smem, tid);
  __syncthreads();

  out[tid] = reinterpret_cast<const uint4 *>(smem)[tid];
}

// LDS.128-like pattern where each half warp can merge its two quarter warps.
// This mirrors the "two transactions for the whole warp" idea.
__global__ void uint4_pair_merge(uint4 *out) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  fill_smem(smem, tid);
  __syncthreads();

  uint32_t addr = (tid / 8) * 2 + ((tid % 8) / 2) % 2;
  out[tid] = reinterpret_cast<const uint4 *>(smem)[addr];
}

// PDF-style uint4 pattern that can produce 2-way bank conflicts inside each
// half warp after the quarter-warp requests are merged.
__global__ void uint4_conflict_like_pdf(uint4 *out) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  fill_smem(smem, tid);
  __syncthreads();

  uint32_t addr = (tid / 16) * 4 + (tid % 16) / 8 + (tid % 8) / 4 * 8;
  out[tid] = reinterpret_cast<const uint4 *>(smem)[addr];
}

#define RUN_CASE(type, kernel_name)                                            \
  do {                                                                        \
    type *out__ = nullptr;                                                     \
    CUDA_CHECK(cudaMalloc(&out__, sizeof(type) * 32));                         \
    CUDA_CHECK(cudaMemset(out__, 0, sizeof(type) * 32));                       \
    kernel_name<<<1, 32>>>(out__);                                             \
    CUDA_CHECK(cudaGetLastError());                                            \
    CUDA_CHECK(cudaDeviceSynchronize());                                       \
    std::printf("ran %s\n", #kernel_name);                                    \
    CUDA_CHECK(cudaFree(out__));                                               \
  } while (0)

int main() {
  RUN_CASE(uint2, uint2_contiguous);
  RUN_CASE(uint2, uint2_pair_broadcast);
  RUN_CASE(uint4, uint4_contiguous);
  RUN_CASE(uint4, uint4_pair_merge);
  RUN_CASE(uint4, uint4_conflict_like_pdf);
  CUDA_CHECK(cudaDeviceReset());
  return 0;
}
