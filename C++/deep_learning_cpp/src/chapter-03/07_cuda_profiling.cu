/*
 * 07_cuda_profiling.cu - 第 3 章：CUDA GPU 加速深度学习
 * CUDA 程序性能分析：计时、带宽计算与 Profiling 工具使用指南
 *
 * 本章演示：
 *   - 使用 cudaEvent 精确测量内核执行时间
 *   - 计算有效内存带宽（GB/s）和计算吞吐量（GFLOP/s）
 *   - 多次运行核函数观察 GPU 预热效应（首次运行 vs 后续运行）
 *   - 统一内存（Unified Memory）首次访问时的页迁移开销分析
 *   - nvprof / ncu / nsys 三种性能分析工具的使用方法
 */

#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <chrono>

// ---- CUDA 错误检查工具函数 ----
// 封装 cudaError_t 检查，失败时打印错误信息并退出
void checkCudaErrors(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA 错误 [" << msg << "]: "
                  << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// ---- 向量加法核函数：y[i] = x[i] + y[i] ----
// 使用 grid-stride 循环模式，每个线程处理多个元素
__global__ void vectorAdd(int n, float *x, float *y) {
    // 计算全局线程索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // grid-stride：线程总数 = blockDim.x * gridDim.x
    int stride = blockDim.x * gridDim.x;

    // 每个线程从自己的 idx 出发，每次跨越 stride 个元素
    for (int i = idx; i < n; i += stride) {
        y[i] = x[i] + y[i]; // 一次浮点加法 + 两次读取 + 一次写入
    }
}

int main() {
    // ============================================================
    // 第 0 部分：参数配置
    // ============================================================
    std::cout << "================================================" << std::endl;
    std::cout << "=== 07_cuda_profiling.cu                        ===" << std::endl;
    std::cout << "=== CUDA 性能分析：计时、带宽与 Profiling       ===" << std::endl;
    std::cout << "================================================" << std::endl;

    const int N = 1 << 22;                  // 4,194,304 个元素（4M）
    const size_t bytes = N * sizeof(float); // 每个数组 ≈ 16 MB
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    // 总数据移动量：读取 X(读) + 读取 Y(读) + 写入 Y(写) = 3N * sizeof(float)
    const size_t totalBytes = 3 * bytes;
    // 总浮点运算次数：每个元素 1 次加法
    const long long totalFLOP = N;

    // ---- 分配统一内存（Unified Memory） ----
    float *x, *y;
    checkCudaErrors(cudaMallocManaged(&x, bytes), "cudaMallocManaged x");
    checkCudaErrors(cudaMallocManaged(&y, bytes), "cudaMallocManaged y");

    // ---- 在 CPU 端初始化数据 ----
    // 首次写入时，数据页驻留在 CPU 端；GPU 首次访问时才发生页迁移
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // ---- 创建 CUDA 事件用于 GPU 计时 ----
    // cudaEvent 使用 GPU 硬件计时器，精度远高于主机端时钟
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start), "cudaEventCreate start");
    checkCudaErrors(cudaEventCreate(&stop), "cudaEventCreate stop");

    // ============================================================
    // 第 1 部分：多次运行核函数，观察预热效应
    // ============================================================
    std::cout << "\n[步骤 1] 多次运行核函数——观察 GPU 预热效应" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "N          = " << N << " (2^22) 个 float 元素" << std::endl;
    std::cout << "每个数组   = " << (bytes / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "总数据移动 = " << (totalBytes / 1024.0 / 1024.0) << " MB (读X + 读Y + 写Y)" << std::endl;
    std::cout << "总浮点运算 = " << totalFLOP << " FLOP" << std::endl;
    std::cout << "启动配置   = <<<" << gridSize << ", " << blockSize << ">>>" << std::endl;

    // ---- 预热说明 ----
    // GPU 首次启动内核需要：
    // 1. CUDA 上下文初始化（lazy context creation）
    // 2. 统一内存页迁移（page fault → page migration from CPU to GPU）
    // 3. GPU SM 唤醒与指令缓存冷启动
    // 后续运行仅涉及内核执行和已迁移数据的访问，因此速度更快

    const int numRuns = 7; // 运行 7 次以充分展示预热效应
    float times[numRuns];

    for (int run = 0; run < numRuns; run++) {
        // 重置 y 数组（后续运行仍需确保数据正确性）
        for (int i = 0; i < N; i++) {
            y[i] = 2.0f;
        }

        // 确保之前所有 GPU 工作完成后再开始计时
        cudaDeviceSynchronize();

        // 记录开始事件
        cudaEventRecord(start);
        // 启动核函数（异步）
        vectorAdd<<<gridSize, blockSize>>>(N, x, y);
        // 记录结束事件
        cudaEventRecord(stop);

        // 等待 GPU 完成，然后计算耗时
        checkCudaErrors(cudaEventSynchronize(stop), "cudaEventSynchronize");
        checkCudaErrors(
            cudaEventElapsedTime(&times[run], start, stop),
            "cudaEventElapsedTime");
        // 检查核函数启动是否有错误
        checkCudaErrors(cudaGetLastError(), "内核启动");
    }

    // ---- 打印每次运行的计时结果 ----
    std::cout << "\n--- 运行时间记录 ---" << std::endl;
    for (int run = 0; run < numRuns; run++) {
        std::cout << "  运行 " << (run + 1) << ": " << times[run] << " ms";
        if (run == 0) {
            // 首次运行包含统一内存页迁移 + CUDA 上下文初始化延迟
            std::cout << "  ← 首次运行（含页迁移开销）";
        } else if (run == 1) {
            // 第 2 次运行通常已接近稳态性能
            std::cout << "  ← 接近稳态性能";
        }
        std::cout << std::endl;
    }

    // ---- 计算首次 vs 后续的加速比 ----
    float firstRunTime = times[0];
    // 取第 3~7 次运行的平均值作为稳态时间（排除前两次的过渡阶段）
    float steadySum = 0.0f;
    for (int i = 2; i < numRuns; i++) {
        steadySum += times[i];
    }
    float steadyAvg = steadySum / (numRuns - 2);
    float speedup = firstRunTime / steadyAvg;

    std::cout << "\n--- 预热效应总结 ---" << std::endl;
    std::cout << "  首次运行时间:       " << firstRunTime << " ms" << std::endl;
    std::cout << "  稳态平均时间 (3~7): " << steadyAvg << " ms" << std::endl;
    std::cout << "  有效加速比:         " << speedup << "x" << std::endl;
    std::cout << "  说明: 首次运行慢 " << speedup << " 倍，主要因统一内存页迁移" << std::endl;

    // ============================================================
    // 第 2 部分：计算有效内存带宽
    // ============================================================
    std::cout << "\n[步骤 2] 计算有效内存带宽 (Effective Bandwidth)" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    // 公式: 带宽 = 总数据量 (字节) / 执行时间 (秒)
    // totalBytes = 3 * N * sizeof(float)：读 X、读 Y、写 Y

    std::cout << "  数据移动量 = 3 × " << N << " × " << sizeof(float)
              << " = " << totalBytes << " 字节" << std::endl;
    std::cout << "  数据移动量 = " << (totalBytes / (1024.0 * 1024.0)) << " MB" << std::endl;

    for (int run = 0; run < numRuns; run++) {
        // 时间单位: ms → s
        float timeSec = times[run] / 1000.0f;
        // 带宽 (GB/s) = 总字节数 / 时间 (秒) / 1e9
        float bandwidthGBps = (totalBytes / timeSec) / 1e9f;

        std::cout << "  运行 " << (run + 1) << ": "
                  << times[run] << " ms → 有效带宽 = "
                  << bandwidthGBps << " GB/s";
        if (run == 0) {
            std::cout << " (含页迁移，低于实际峰值)";
        }
        std::cout << std::endl;
    }

    // ============================================================
    // 第 3 部分：计算浮点运算吞吐量（GFLOP/s）
    // ============================================================
    std::cout << "\n[步骤 3] 计算浮点运算吞吐量 (GFLOP/s)" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    // 公式: GFLOP/s = 总浮点运算次数 / 执行时间 (秒) / 1e9
    // 向量加法：每个元素 1 次加法 → 共 N 次浮点运算

    std::cout << "  总浮点运算 = " << totalFLOP << " FLOP" << std::endl;

    for (int run = 0; run < numRuns; run++) {
        float timeSec = times[run] / 1000.0f;
        float gflops = (totalFLOP / timeSec) / 1e9f;

        std::cout << "  运行 " << (run + 1) << ": "
                  << times[run] << " ms → 吞吐量 = "
                  << gflops << " GFLOP/s";
        if (run == 0) {
            std::cout << " (上限受限于带宽)";
        }
        std::cout << std::endl;
    }

    // ---- 稳态性能总结 ----
    float steadyTimeSec = steadyAvg / 1000.0f;
    float steadyBW = (totalBytes / steadyTimeSec) / 1e9f;
    float steadyGFLOPS = (totalFLOP / steadyTimeSec) / 1e9f;

    std::cout << "\n--- 稳态性能总结 ---" << std::endl;
    std::cout << "  稳态执行时间:   " << steadyAvg << " ms" << std::endl;
    std::cout << "  有效内存带宽:   " << steadyBW << " GB/s" << std::endl;
    std::cout << "  浮点吞吐量:     " << steadyGFLOPS << " GFLOP/s" << std::endl;

    // ============================================================
    // 第 4 部分：性能分析工具使用指南（注释说明）
    // ============================================================
    std::cout << "\n[步骤 4] 性能分析工具使用指南" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "  以下是三种常用 NVIDIA 性能分析工具的使用方法：" << std::endl;

    // ---------------- nvprof 使用说明 ----------------
    // nvprof 是 CUDA 传统的命令行性能分析器（CUDA 10 之前默认推荐）
    // 用法：
    //   nvprof ./07_cuda_profiling
    //   nvprof --print-gpu-trace ./07_cuda_profiling     (打印 GPU 操作时间线)
    //   nvprof --metrics achieved_occupancy ./07_cuda_profiling  (查看占用率指标)
    //
    // nvprof 输出包含：
    //   - 每个内核的执行时间、调用次数、平均/最大/最小时间
    //   - CUDA API 调用耗时（如 cudaMalloc、cudaMemcpy）
    //   - GPU 硬件计数器（使用 --metrics 选项）

    std::cout << "\n  (1) nvprof —— CUDA 传统性能分析器" << std::endl;
    std::cout << "      nvprof ./07_cuda_profiling" << std::endl;
    std::cout << "      nvprof --print-gpu-trace ./07_cuda_profiling" << std::endl;
    std::cout << "      nvprof --metrics achieved_occupancy ./07_cuda_profiling" << std::endl;
    std::cout << "      说明：nvprof 已逐步被 ncu 取代，" << std::endl;
    std::cout << "            CUDA 11+ 推荐使用 Nsight Compute (ncu)" << std::endl;

    // ---------------- ncu 使用说明 ----------------
    // ncu (NVIDIA Nsight Compute) 是新一代 GPU 内核性能分析器
    // 用法：
    //   ncu ./07_cuda_profiling
    //   ncu --set full ./07_cuda_profiling              (完整分析，信息最详细)
    //   ncu -o report ./07_cuda_profiling               (输出到 report.ncu-rep 文件)
    //   ncu --section MemoryWorkloadAnalysis ./07_cuda_profiling   (仅分析内存)
    //
    // ncu 输出包含：
    //   - 内核的详细性能指标（占用率、内存带宽利用率、计算吞吐量）
    //   - 瓶颈分析（内存受限 vs 计算受限）
    //   - Roofline 性能分析图表（需要 --set full）
    //   - 源代码级别的性能建议

    std::cout << "\n  (2) ncu (NVIDIA Nsight Compute) —— 内核级分析器" << std::endl;
    std::cout << "      ncu ./07_cuda_profiling" << std::endl;
    std::cout << "      ncu --set full ./07_cuda_profiling" << std::endl;
    std::cout << "      ncu -o report ./07_cuda_profiling" << std::endl;
    std::cout << "      ncu --section MemoryWorkloadAnalysis ./07_cuda_profiling" << std::endl;
    std::cout << "      说明：ncu 提供最详细的内核性能指标，" << std::endl;
    std::cout << "            包括占用率、内存带宽利用率和瓶颈分析" << std::endl;

    // ---------------- nsys 使用说明 ----------------
    // nsys (NVIDIA Nsight Systems) 是系统级时间线分析器
    // 用法：
    //   nsys profile ./07_cuda_profiling
    //   nsys profile --stats=true ./07_cuda_profiling     (打印统计摘要)
    //   nsys profile -o timeline ./07_cuda_profiling      (输出到 timeline.qdrep)
    //
    // nsys 输出包含：
    //   - CPU 和 GPU 活动的时间线视图
    //   - CUDA API 调用、内核执行、内存拷贝的时序关系
    //   - 多线程/多流并发情况的可视化
    //   - 系统级开销分析（线程同步、I/O 阻塞等）

    std::cout << "\n  (3) nsys (NVIDIA Nsight Systems) —— 系统级时间线分析器" << std::endl;
    std::cout << "      nsys profile ./07_cuda_profiling" << std::endl;
    std::cout << "      nsys profile --stats=true ./07_cuda_profiling" << std::endl;
    std::cout << "      nsys profile -o timeline ./07_cuda_profiling" << std::endl;
    std::cout << "      说明：nsys 提供 CPU-GPU 交互的时间线视图，" << std::endl;
    std::cout << "            适用于发现系统级瓶颈和异步问题" << std::endl;

    // ---- 工具选择建议 ----
    // 1. 初步定位瓶颈范围     → nsys profile（系统级全景图）
    // 2. 深入分析内核性能     → ncu（单内核详细指标）
    // 3. 快速检查执行时间     → nvprof（轻量级，输出简洁）
    // 4. 可视化时间线分析     → nsys -o xxx.qdrep + Nsight Systems GUI

    std::cout << "\n  --- 工具选择建议 ---" << std::endl;
    std::cout << "  初步定位 → nsys profile（系统级全景图）" << std::endl;
    std::cout << "  内核分析 → ncu（单内核详细指标）" << std::endl;
    std::cout << "  快速检查 → nvprof（轻量级）" << std::endl;

    // ============================================================
    // 第 5 部分：验证计算结果 + 清理资源
    // ============================================================
    std::cout << "\n[步骤 5] 验证计算结果" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;

    // 验证：所有 y[i] 应等于 1.0 + 2.0 = 3.0
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }
    std::cout << "  最大误差: " << maxError << std::endl;
    if (maxError < 1e-5f) {
        std::cout << "  结果正确 ✓" << std::endl;
    } else {
        std::cout << "  结果错误 ✗ (误差过大)" << std::endl;
    }

    // ---- 清理 CUDA 资源 ----
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(x);
    cudaFree(y);

    // ---- 最终总结 ----
    std::cout << "\n================================================" << std::endl;
    std::cout << "=== 性能分析总结                                  ===" << std::endl;
    std::cout << "================================================" << std::endl;
    std::cout << "  数据规模:     " << N << " 个 float 元素" << std::endl;
    std::cout << "  稳态时间:     " << steadyAvg << " ms" << std::endl;
    std::cout << "  有效带宽:     " << steadyBW << " GB/s" << std::endl;
    std::cout << "  浮点吞吐量:   " << steadyGFLOPS << " GFLOP/s" << std::endl;
    std::cout << "  预热加速比:   " << speedup << "x (首次 vs 稳态)" << std::endl;
    std::cout << "  性能特征:     向量加法——典型的带宽受限操作" << std::endl;
    std::cout << "                (内存带宽而非计算能力是性能瓶颈)" << std::endl;
    std::cout << std::endl;
    std::cout << "  建议下一步: 使用 ncu --set full ./07_cuda_profiling" << std::endl;
    std::cout << "              查看详细的 Roofline 分析和优化建议" << std::endl;
    std::cout << "================================================" << std::endl;

    return 0;
}
