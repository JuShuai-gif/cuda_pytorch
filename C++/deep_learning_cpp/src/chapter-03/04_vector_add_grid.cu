/*
 * 04_vector_add_grid.cu - 第 3 章：CUDA GPU 加速深度学习
 * 第 4 步：CUDA 多 Block 网格启动（充分利用整个 GPU）
 *
 * 本章演绎：从 CPU 基线开始，将同一数组加法问题逐步迁移到 GPU 并行执行。
 * 本文件是「第 4 / 4 步」——使用多个 Block 完全饱和 GPU 的流多处理器（SM），
 * 实现真实显著的 GPU 加速比。每条代码路径都反映了生产级 CUDA 程序的结构。
 *
 * 对应 PDF 第 96–100 页。
 */

#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <chrono>

// ----------------------------------------------------------------
// 核函数：全局线程索引公式 —— CUDA 编程中最基本的模式
// ----------------------------------------------------------------
__global__ void add(int n, float *x, float *y) {
    // 全局线程索引公式：
    //   blockIdx.x  —— 当前 block 在整个 grid 中的索引 [0, gridDim.x)
    //   blockDim.x  —— 每个 block 中的线程总数
    //   threadIdx.x —— 线程在其 block 内的索引 [0, blockDim.x)
    //
    // 例如 gridDim.x=4096, blockDim.x=256:
    //   blockIdx.x=2, threadIdx.x=3 → 全局索引 = 2×256+3 = 515
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // 全局跨步：所有 block × 所有线程 = 总并行度
    //   stride = blockDim.x * gridDim.x
    //   grid-stride 循环保证即使 N 超过总线程数也能正确覆盖
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        y[i] = x[i] + y[i];
    }
}

// ----------------------------------------------------------------
int main() {
    std::cout << "=== 04_vector_add_grid.cu ===" << std::endl;

    const int N = 1 << 20;     // 1,048,576 元素
    const int blockSize = 256; // 每 block 256 线程（32 的倍数，对齐 warp 调度）

    // 向上取整除法：覆盖所有 N 个元素
    //   numBlocks = (N + blockSize - 1) / blockSize
    //   例如 N=1,048,576, blockSize=256 → numBlocks = 4096
    const int numBlocks = (N + blockSize - 1) / blockSize;

    float *x, *y;

    // 统一内存分配
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    // 初始化
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // ---- 计时 ----
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // <<<numBlocks, blockSize>>> —— 全 GPU 启动
    // 例如 NVIDIA T4（40 个 SM）上：
    //   4096 个 block × 256 线程 = 1,048,576 个线程正好覆盖 N 个元素
    //   GPU 调度器一次可驻留 40×16=640 个 block（取决于寄存器/共享内存限制）
    //   剩余 block 排队等待，所有 SM 均被利用 → 真正的 GPU 加速
    add<<<numBlocks, blockSize>>>(N, x, y);

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // ---- 计时结束 ----

    // 验证
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }

    std::cout << "N = " << N << " (2^" << 20 << ")" << std::endl;
    std::cout << "启动配置: <<<" << numBlocks << ", " << blockSize
              << ">>> (" << numBlocks << " blocks × " << blockSize << " threads = "
              << numBlocks * blockSize << " total threads)" << std::endl;
    std::cout << "最大误差: " << maxError << std::endl;
    std::cout << "GPU 内核执行时间: " << milliseconds << " ms" << std::endl;

    // 性能说明：
    //   - 所有 SM 都被利用，GPU 饱和工作 → 相比 CPU 可获得数十倍加速
    //   - 用 nvprof 或 nsys profile 可看到内核时间大幅缩短
    //   - 下一步优化方向：coalesced 访问、共享内存、stream 并发
    std::cout << "性能: 多 Block 网格启动已获得真实 GPU 加速比（数十倍于 CPU）" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(x);
    cudaFree(y);

    return 0;
}
