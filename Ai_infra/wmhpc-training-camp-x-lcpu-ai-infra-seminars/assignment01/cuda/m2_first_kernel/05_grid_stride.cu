// 问题 2.7：grid-stride loop（改造题）。
// 现状：launch 只给了 64 个 block，线程总数远小于 n，所以输出 FAIL。
// 任务：不许改 launch 配置，把 kernel 改成 grid-stride loop——每个线程
//      跨过整个 grid 的步长处理多个元素——让任意 n 都能 PASS。
// 参考：NVIDIA 博客 "CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops"
//      https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
//
// ===================== grid-stride loop 原理 =====================
//
// 核心思想：每个线程不再只处理一个元素，而是以"grid 总线程数"为步长，
// 跳跃处理多个元素。这样不论 launch 给多少线程，都能覆盖任意大小的 n。
//
// stride = blockDim.x * gridDim.x = 256 × 64 = 16384
// 每个线程从自己的 id 出发，每次 +16384，直到超出 n。
// 16M / 16384 = 1024，每个线程处理 1024 个元素，全覆盖无遗漏。
//
// 好处：launch 配置和 n 完全解耦——随便给多少个 block 都能跑对。
// ==================================================================
#include "common.h"

__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;       // 线程在 grid 中的起点
    int stride = blockDim.x * gridDim.x;                   // grid 总线程数 = 步长
    for (int i = id; i < n; i += stride) c[i] = a[i] + b[i];
}

int main() {
    const int n = 1 << 24;  // 16M 元素，远多于 64 * 256 = 16384 个线程
    size_t bytes = (size_t)n * sizeof(float);

    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);
    fill_random(h_a, n, 1);
    fill_random(h_b, n, 2);
    for (int i = 0; i < n; i++) h_ref[i] = h_a[i] + h_b[i];

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_c, 0, bytes));

    vectorAdd<<<64, 256>>>(d_a, d_b, d_c, n);  // launch 配置不许动
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    REPORT(check_close(h_c, h_ref, n));
    return 0;
}
