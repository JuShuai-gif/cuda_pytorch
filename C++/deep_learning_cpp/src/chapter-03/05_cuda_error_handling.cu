/*
 * ================================================================
 * 文件名: 05_cuda_error_handling.cu
 * 第 3 章：CUDA GPU 加速深度学习
 * 主题: CUDA 三步错误检查方法论
 *
 * 演示内容:
 *   第一步 - 检查 API 调用返回值
 *   第二步 - 检查核函数启动错误 (cudaGetLastError)
 *   第三步 - 检查异步执行错误 (cudaDeviceSynchronize)
 *   封装 - 可复用的 checkCuda() 内联函数
 * ================================================================
 */

#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>
#include <cassert>
#include <cmath>

// ================================================================
// 可复用的 CUDA 错误检查函数
// 用法: checkCuda(cudaSomeFunction(...))
// ================================================================

inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::fprintf(stderr,
                     "CUDA 运行时错误: %s (错误码: %d)\n"
                     "  文件: %s\n"
                     "  行号: %d\n",
                     cudaGetErrorString(result),
                     static_cast<int>(result),
                     __FILE__,
                     __LINE__);
        // 在 GPU 上发生错误后必须重置设备状态
        cudaDeviceReset();
        std::exit(result);
    }
    return result;
}

// ================================================================
// 成功运行示例用的简单核函数
// ================================================================

__global__ void vectorAddKernel(const float *a, const float *b, float *c, int n) {
    // 计算当前线程负责的全局索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 执行向量加法
        c[idx] = a[idx] + b[idx];
    }
}

// ================================================================
// main 函数 - 演示完整的三步错误检查流程
// ================================================================

int main() {
    // -------------------- 步骤 0: 打印设备信息 --------------------
    std::printf("===== CUDA 三步错误检查方法演示 =====\n\n");

    int dev_count = 0;
    checkCuda(cudaGetDeviceCount(&dev_count));
    std::printf("检测到 %d 个 CUDA 设备\n", dev_count);

    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, 0));
    std::printf("当前设备: %s\n\n", prop.name);

    // ================================================================
    // 第一步: 检查 API 调用的返回值
    //   同步 API 会立即返回错误码，直接检查即可
    // ================================================================
    std::printf("// ================================================================\n");
    std::printf("// 第一步: 检查 API 调用返回值\n");
    std::printf("//   同步 API 立即返回 cudaError_t，直接判断是否等于 cudaSuccess\n");
    std::printf("// ================================================================\n\n");

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;
    const int N = 1024;
    const size_t bytes = N * sizeof(float);

    std::printf(">>> 正确用法: 使用 checkCuda 包装 cudaMalloc\n");
    checkCuda(cudaMalloc(&d_a, bytes));
    checkCuda(cudaMalloc(&d_b, bytes));
    checkCuda(cudaMalloc(&d_c, bytes));
    std::printf("    所有 cudaMalloc 调用成功 ✓\n\n");

    // 演示错误检测: 传递空指针给 cudaFree
    std::printf(">>> 错误捕获演示: 传递 nullptr 给 cudaMalloc 的第一个参数\n");
    {
        // cudaMalloc(nullptr, bytes) 是无效调用，会返回错误
        cudaError_t err = cudaMalloc(nullptr, bytes);
        if (err != cudaSuccess) {
            std::printf("    ✓ 捕获到第一步错误: %s (错误码: %d)\n",
                        cudaGetErrorString(err), static_cast<int>(err));
        }
    }
    std::printf("\n");

    // 清理测试分配的内存
    checkCuda(cudaFree(d_a));
    checkCuda(cudaFree(d_b));
    checkCuda(cudaFree(d_c));

    // ================================================================
    // 第二步: 检查核函数启动错误
    //   核函数启动是异步的，用 cudaGetLastError() 捕获配置错误
    // ================================================================
    std::printf("// ================================================================\n");
    std::printf("// 第二步: 检查核函数启动错误 (cudaGetLastError)\n");
    std::printf("//   核函数启动失败不会立即抛出异常，需要 cudaGetLastError 捕获\n");
    std::printf("// ================================================================\n\n");

    std::printf(">>> 错误捕获演示: <<<1, -1>>> - 线程块维度无效\n");
    {
        /* 故意使用 -1 作为线程块大小，这是一个无效的启动配置 */
        vectorAddKernel<<<1, -1>>>(d_a, d_b, d_c, N);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::printf("    ✓ 捕获到第二步错误: %s (错误码: %d)\n",
                        cudaGetErrorString(err), static_cast<int>(err));
            std::printf("    (预期行为: 线程块大小 -1 是无效配置)\n");
        }
    }
    std::printf("\n");

    // ================================================================
    // 第三步: 检查异步执行错误
    //   核函数执行过程中或内存拷贝中的错误用 cudaDeviceSynchronize 捕获
    // ================================================================
    std::printf("// ================================================================\n");
    std::printf("// 第三步: 检查异步执行错误 (cudaDeviceSynchronize)\n");
    std::printf("//   核函数执行时发生的运行时错误用同步捕获\n");
    std::printf("// ================================================================\n\n");

    // 分配两块 GPU 内存
    float *d_array = nullptr;
    checkCuda(cudaMalloc(&d_array, 256 * sizeof(float)));

    std::printf(">>> 错误捕获演示: 向无效地址进行异步内存拷贝\n");
    {
        /* 故意写入越界（地址偏移超出分配范围），
           这会在异步执行中产生非法的内存访问错误 */
        cudaError_t err = cudaMemsetAsync(
            d_array + 512, // 越界偏移: 512 > 256
            0,
            sizeof(float),
            (cudaStream_t)0 // 默认流
        );

        // 即使前面的 memset 不报错，同步时也会暴露异步错误
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::printf("    ✓ 捕获到第三步错误: %s (错误码: %d)\n",
                        cudaGetErrorString(err), static_cast<int>(err));
            std::printf("    (预期行为: 非法的内存访问操作)\n");
        }
    }
    std::printf("\n");
    checkCuda(cudaFree(d_array));

    // ================================================================
    // 完整示例: 正确的核函数启动 + 三步检查
    // ================================================================
    std::printf("// ================================================================\n");
    std::printf("// 完整示例: 正确启动流程 + 三步检查\n");
    std::printf("// ================================================================\n\n");

    // 主机端数据分配
    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c = new float[N];
    float *h_c_ref = new float[N]; // CPU 参考结果

    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
        h_c_ref[i] = h_a[i] + h_b[i];
    }

    // ---------- GPU 分配 + 数据拷贝 ----------
    std::printf("步骤1: GPU 内存分配...\n");
    checkCuda(cudaMallocManaged(&d_a, bytes));
    checkCuda(cudaMallocManaged(&d_b, bytes));
    checkCuda(cudaMallocManaged(&d_c, bytes));
    std::printf("  ✓ 统一内存分配成功\n");

    std::printf("步骤2: 数据拷贝到 GPU...\n");
    /* 显式同步拷贝以确保数据就绪 */
    checkCuda(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    std::printf("  ✓ 主机 -> GPU 数据拷贝成功\n");

    // ---------- 核函数启动 ----------
    std::printf("步骤3: 启动核函数...\n");
    const int threads_per_block = 256;
    const int blocks = (N + threads_per_block - 1) / threads_per_block;

    vectorAddKernel<<<blocks, threads_per_block>>>(d_a, d_b, d_c, N);
    /* 第二步检查: 核函数启动配置是否有效 */
    checkCuda(cudaGetLastError());
    std::printf("  ✓ 核函数启动成功 (第二步检查通过)\n");

    /* 第三步检查: 核函数执行是否有运行时错误 */
    checkCuda(cudaDeviceSynchronize());
    std::printf("  ✓ 核函数执行完毕，无运行时错误 (第三步检查通过)\n");

    // ---------- 结果验证 ----------
    std::printf("步骤4: 取回并验证结果...\n");
    checkCuda(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    bool passed = true;
    for (int i = 0; i < N; ++i) {
        if (std::fabs(h_c[i] - h_c_ref[i]) > 1e-5f) {
            std::printf("  ✗ 位置 %d: 期望 %f, 实际 %f\n",
                        i, h_c_ref[i], h_c[i]);
            passed = false;
            break;
        }
    }
    if (passed) {
        std::printf("  ✓ 所有 %d 个元素验证通过\n", N);
    }

    // ---------- 资源清理 ----------
    checkCuda(cudaFree(d_a));
    checkCuda(cudaFree(d_b));
    checkCuda(cudaFree(d_c));
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    delete[] h_c_ref;

    std::printf("\n===== 演示完成: 三步错误检查方法全部通过 =====\n");
    std::printf("  checkCuda(API调用) -> kernel<<<>>> -> checkCuda(cudaGetLastError()) -> checkCuda(cudaDeviceSynchronize())\n");

    // 重置设备以确保干净的退出状态
    checkCuda(cudaDeviceReset());
    return 0;
}
