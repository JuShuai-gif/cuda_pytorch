/*
 * 01_vector_add_cpu.cpp
 * 第 3 章：CUDA GPU 加速深度学习
 *
 * 纯 CPU 向量加法基线实现。
 * 在将计算移植到 GPU 之前，必须先在 CPU 上建立正确的参考实现
 * 和性能基线。本文件实现 1M 元素的逐元素向量加法，并测量执行时间。
 *
 * 涵盖的技术：
 *   - 朴素的逐元素向量加法（O(N) 时间复杂度）
 *   - std::chrono 高精度计时
 *   - 结果验证（最大误差检查）
 *   - 动态内存分配与释放
 *
 * 此基线将在后续 CUDA 实现中作为正确性和性能的参照。
 */

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <algorithm>

// ----------------------------------------------------------------
// 向量加法：out[i] = inputA[i] + inputB[i]
// 纯 CPU 实现，作为 GPU 版本的参照基线。
// 输入：length 为元素数量，inputA 和 inputB 为加数，
//       out 为结果输出数组（需预分配 length 个元素）。
// ----------------------------------------------------------------
void vectorAddCPU(int length, const float *inputA, const float *inputB,
                  float *out) {
    for (int i = 0; i < length; ++i) {
        out[i] = inputA[i] + inputB[i];
    }
}

// ----------------------------------------------------------------
// 主函数：分配内存、填充数据、执行向量加法、验证结果、
// 测量耗时并清理资源。
// ----------------------------------------------------------------
int main() {
    printf("=== CPU 向量加法基线测试 ===\n\n");

    // --- 分配与初始化 ---
    const int N = 1 << 20; // 1,048,576 个元素
    const size_t bytes = N * sizeof(float);

    printf("[初始化] 分配 %d 个 float 元素（%.2f MB）\n",
           N, bytes / (1024.0 * 1024.0));

    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    if (!h_A || !h_B || !h_C) {
        printf("[错误] 内存分配失败。\n");
        free(h_A);
        free(h_B);
        free(h_C);
        return 1;
    }

    // 填充输入数组：A 全为 0.5f，B 全为 2.5f
    // 期望结果：C 全为 3.0f
    for (int i = 0; i < N; ++i) {
        h_A[i] = 0.5f;
        h_B[i] = 2.5f;
    }
    printf("[数据] h_A 全部填充为 0.5，h_B 全部填充为 2.5\n");

    // --- 执行向量加法并计时 ---
    printf("[执行] 开始 CPU 向量加法...\n");

    auto start = std::chrono::high_resolution_clock::now();
    vectorAddCPU(N, h_A, h_B, h_C);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    printf("[计时] CPU 向量加法耗时: %.3f 毫秒\n", elapsed.count());

    // --- 验证结果 ---
    // 期望每个元素为 3.0f，计算最大绝对误差
    float maxError = 0.0f;
    for (int i = 0; i < N; ++i) {
        float error = std::fabs(h_C[i] - 3.0f);
        maxError = std::max(maxError, error);
    }
    printf("[验证] 最大绝对误差: %e\n", maxError);

    if (maxError < 1e-5f) {
        printf("[通过] 向量加法结果正确（误差 < 1e-5）。\n");
    } else {
        printf("[失败] 向量加法结果不正确！\n");
    }

    // --- 打印前几个元素供手动检查 ---
    printf("[检查] 前 5 个结果元素:\n");
    for (int i = 0; i < 5 && i < N; ++i) {
        printf("  C[%d] = %f\n", i, h_C[i]);
    }

    // --- 计算带宽 ---
    // 读取 A (N) + 读取 B (N) + 写入 C (N) = 3N 次 float 操作
    double totalBytes = 3.0 * N * sizeof(float);
    double bandwidth = totalBytes / (elapsed.count() / 1000.0) / (1024.0 * 1024.0 * 1024.0);
    printf("[带宽] 有效内存带宽: %.2f GB/s\n", bandwidth);

    // --- 清理 ---
    free(h_A);
    free(h_B);
    free(h_C);
    printf("\n[清理] 内存已释放。\n");

    return (maxError < 1e-5f) ? 0 : 1;
}
