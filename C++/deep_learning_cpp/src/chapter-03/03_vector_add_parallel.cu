/*
 * 03_vector_add_parallel.cu - 第 3 章：CUDA GPU 加速深度学习
 * 第 3 步：CUDA 多线程并行（1 个 Block × 256 个线程）
 *
 * 本章演绎：从 CPU 基线开始，将同一数组加法问题逐步迁移到 GPU 并行执行。
 * 本文件是「第 3 / 4 步」——启动 1 个 block 含 256 个线程，每个线程用 grid-stride
 * 循环覆盖跨越的元素。相比单线程版本，已在 GPU 上获得显著并行加速。
 *
 * 对应 PDF 第 93–96 页。
 */

#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <chrono>

// ----------------------------------------------------------------
// 核函数：同 02_vector_add_cuda.cu 中的 add() 完全一致
// grid-stride 循环使得同一个 kernel 可以适配任意线程数 / 块数配置
// ----------------------------------------------------------------
__global__ void add(int n, float *x, float *y) {
    // threadIdx.x  —— 当前线程在其 block 内的索引 [0, blockDim.x)
    // blockDim.x   —— 每个 block 中的线程总数
    int index = threadIdx.x; // 每个线程唯一的起始偏移
    int stride = blockDim.x; // 所有线程的总数（跨步大小）

    // grid-stride 循环：
    // 线程 0 处理元素 [0, 256, 512, ...]
    // 线程 1 处理元素 [1, 257, 513, ...]
    // ...
    // 线程 255 处理元素 [255, 511, 767, ...]
    //
    // ⚠ 为什么没有数据竞争（data race）？
    //    因为每个线程通过 index 拿到互不重叠的起始偏移，
    //    每次跨步 stride 后仍然不重叠——每个 y[i] 只被一个线程写入。
    for (int i = index; i < n; i += stride) {
        y[i] = x[i] + y[i];
    }
}

// ----------------------------------------------------------------
int main() {
    std::cout << "=== 03_vector_add_parallel.cu ===" << std::endl;

    const int N = 1 << 20;     // 1,048,576 元素
    const int blockSize = 256; // 每个 block 256 个线程（NVIDIA warp=32 的倍数）

    float *x, *y;

    // 统一内存：CPU 和 GPU 用同一指针，运行时自动管理数据迁移
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

    // <<<1, 256>>> —— 1 个 block，256 个线程
    // 每个线程用 grid-stride 循环处理约 N/256 ≈ 4096 个元素
    add<<<1, 256>>>(N, x, y);

    cudaDeviceSynchronize(); // 等待 GPU 完成

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
    std::cout << "启动配置: <<<1, " << blockSize << ">>> (1 block, " << blockSize << " threads)" << std::endl;
    std::cout << "每个线程处理约 " << N / blockSize << " 个元素" << std::endl;
    std::cout << "最大误差: " << maxError << std::endl;
    std::cout << "GPU 内核执行时间: " << milliseconds << " ms" << std::endl;

    // 概念性对比说明：
    //   - 单线程 GPU（<<<1,1>>>）：每个元素一个 for 迭代，无并行 → 比 CPU 更慢（启动开销 + 页迁移）
    //   - 256 线程 GPU（<<<1,256>>>）：同一 block 内 256 个线程并发，时间应减少约 1~2 个数量级
    //   - 多 Block GPU（<<<numBlocks,256>>>）：多个 SM 同时执行，充分饱和 GPU → 性能进一步跃升
    std::cout << "对比概念: 相比单线程 CPU/GPU 应快约 1~2 个数量级（受限于只有 1 个 SM 在工作）" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(x);
    cudaFree(y);

    return 0;
}
