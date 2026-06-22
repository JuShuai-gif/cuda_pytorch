/*
 * 02_vector_add_cuda.cu - 第 3 章：CUDA GPU 加速深度学习
 * 第 2 步：CUDA 单线程核函数（第一个 CUDA 程序）
 *
 * 本章演绎：从 CPU 基线开始，将同一数组加法问题逐步迁移到 GPU 并行执行。
 * 本文件是「第 2 / 4 步」——最原始的 CUDA kernel，仅用 1 个 block、1 个线程。
 *
 * 对应 PDF 第 89–91 页。
 */

#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <chrono>

// ----------------------------------------------------------------
// __global__ 标记此函数为 GPU 核函数（kernel），由主机（CPU）调用，在设备（GPU）上执行
// 返回值只能为 void；三重尖括号 <<<gridDim, blockDim>>> 是 CUDA 独有语法
// ----------------------------------------------------------------
__global__ void add(int n, float *x, float *y) {
    // threadIdx.x  —— 当前线程在其 block 内的索引（这里永远 =0，因为只有 1 个线程）
    // blockDim.x   —— 每个 block 中的线程总数（这里 =1）
    int index = threadIdx.x; // 线程在 block 内的起始偏移
    int stride = blockDim.x; // 每次跳过的元素数（grid-stride 模式）

    // grid-stride 循环：每个线程从自己的 index 开始，每次跨越 stride 个元素
    // 在单线程场景下 index=0, stride=1，此循环 完全等价于 CPU 的单线程 for 循环
    // ⚠ 无数据竞争——每个元素只被唯一一个线程访问
    for (int i = index; i < n; i += stride) {
        y[i] = x[i] + y[i];
    }
}

// ----------------------------------------------------------------
int main() {
    std::cout << "=== 02_vector_add_cuda.cu ===" << std::endl;

    // N = 2^20 = 1,048,576 个元素（float）≈ 4 MB 每个数组
    const int N = 1 << 20;

    float *x, *y;

    // cudaMallocManaged 分配统一内存（Unified Memory）
    // CPU 和 GPU 使用 同一个指针 访问，无需手动 cudaMemcpy
    // 运行时自动处理底层页迁移（Pascal+ 硬件支持按需迁移）
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    // 在 CPU 端初始化数据
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f; // 每个 x[i] 填充 1.0
        y[i] = 2.0f; // 每个 y[i] 填充 2.0 —— 期望结果 y[i] = 3.0
    }

    // ---- CUDA 事件计时开始 ----
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // <<<1, 1>>> —— 启动配置：1 个 block，每个 block 1 个线程
    // 内核启动是 异步 的（host 不会等待），因此需要后续同步
    add<<<1, 1>>>(N, x, y);

    // cudaDeviceSynchronize —— 阻塞主机，直到 GPU 上所有排队的工作完成
    // 这是必须的！否则主机会在内核完成前就读取结果并计时
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // ---- 计时结束 ----

    // 验证结果：所有 y[i] 应当等于 1.0 + 2.0 = 3.0
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }
    std::cout << "N = " << N << " (2^" << 20 << ")" << std::endl;
    std::cout << "启动配置: <<<1, 1>>> (1 block, 1 thread)" << std::endl;
    std::cout << "最大误差: " << maxError << std::endl;
    std::cout << "GPU 内核执行时间: " << milliseconds << " ms" << std::endl;
    std::cout << "注意: 单线程 GPU —— 与 CPU 性能相当，尚未获得加速" << std::endl;

    // 清理资源
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(x); // 释放统一内存
    cudaFree(y);

    return 0;
}
