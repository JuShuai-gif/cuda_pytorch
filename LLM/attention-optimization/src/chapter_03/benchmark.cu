/**
 * Chapter 03 profiling benchmark.
 *
 * A tiny CUDA microbenchmark with named ranges in the source comments. Run it
 * with cudaEvent for latency, Nsight Systems for timeline, and Nsight Compute
 * for kernel counters. The kernel is intentionally simple so profiler output is
 * easy to interpret before moving to attention kernels.
 */
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CUDA_CHECK(err)                                                     \
  do {                                                                      \
    cudaError_t e = (err);                                                  \
    if (e != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,   \
                   cudaGetErrorString(e));                                  \
      std::exit(1);                                                         \
    }                                                                       \
  } while (0)

__global__ void vector_scale_add(const float* x, const float* y, float* z,
                                 float alpha, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) z[idx] = alpha * x[idx] + y[idx];
}

static float time_kernel(const float* x, const float* y, float* z, int n,
                         int iters) {
  dim3 block(256);
  dim3 grid((n + block.x - 1) / block.x);
  for (int i = 0; i < 10; ++i) vector_scale_add<<<grid, block>>>(x, y, z, 0.5f, n);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) vector_scale_add<<<grid, block>>>(x, y, z, 0.5f, n);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / iters;
}

int main() {
  const int n = 1 << 24;
  const int iters = 100;
  std::vector<float> h_x(n), h_y(n), h_z(n);
  for (int i = 0; i < n; ++i) {
    h_x[i] = std::sin(0.01f * i);
    h_y[i] = std::cos(0.01f * i);
  }

  float *d_x = nullptr, *d_y = nullptr, *d_z = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_z, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  float ms = time_kernel(d_x, d_y, d_z, n, iters);
  double bytes = 3.0 * n * sizeof(float);
  double bandwidth = (bytes / 1e9) / (ms / 1000.0);
  std::printf("Chapter 03 CUDA profiling benchmark\n");
  std::printf("N=%d latency=%.4f ms effective_bw=%.2f GB/s\n", n, ms, bandwidth);
  std::printf("Profile with: ncu --set full ./chapters/profiling_benchmark\n");

  CUDA_CHECK(cudaMemcpy(h_z.data(), d_z, n * sizeof(float), cudaMemcpyDeviceToHost));
  std::printf("check z[123]=%.6f\n", h_z[123]);
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  CUDA_CHECK(cudaFree(d_z));
  return 0;
}
