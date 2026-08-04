// 04_gpu_concepts.cpp — GPU 并行基础概念 (纯理论 + 模拟)
// 演示: CPU 模拟 GPU 的 SIMT 执行模型、内存层次对比

#include <algorithm>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

// ===== 1. GPU 执行模型模拟 (SIMT) =====
// CPU 多线程模拟 GPU 的 thousands-of-threads 模型
void demo_simt_model() {
    std::cout << "=== SIMT 模型模拟 (GPU 执行模型) ===\n";

    // GPU 典型的启动参数
    const int kBlocksPerGrid = 8;   // 类似 GPU block
    const int kThreadsPerBlock = 32; // 类似 GPU warp size
    const int kTotalThreads = kBlocksPerGrid * kThreadsPerBlock;

    std::cout << "  Grid: " << kBlocksPerGrid << " blocks\n";
    std::cout << "  Block: " << kThreadsPerBlock << " threads\n";
    std::cout << "  Total: " << kTotalThreads << " threads\n\n";

    // 模拟: 向量加法 (每个线程处理一个元素)
    std::vector<float> a(kTotalThreads, 1.0f);
    std::vector<float> b(kTotalThreads, 2.0f);
    std::vector<float> c(kTotalThreads, 0.0f);

    std::vector<std::jthread> threads;
    threads.reserve(kTotalThreads);

    auto start = std::chrono::high_resolution_clock::now();

    // CPU 上模拟 GPU 的数千线程(实际上 CPU 线程数远大于硬件核心)
    for (int block = 0; block < kBlocksPerGrid; ++block) {
        for (int tid = 0; tid < kThreadsPerBlock; ++tid) {
            int global_id = block * kThreadsPerBlock + tid;
            threads.emplace_back([&, global_id]() {
                c[global_id] = a[global_id] + b[global_id];
            });
        }
    }
    for (auto& t : threads) t.join();

    auto elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start);

    bool correct = true;
    for (int i = 0; i < kTotalThreads; ++i) {
        if (c[i] != 3.0f) { correct = false; break; }
    }
    std::cout << "  向量加法结果: " << (correct ? "OK" : "FAIL")
              << " | 耗时: " << elapsed.count() << " us\n";
    std::cout << "  注意: CPU 上创建 " << kTotalThreads
              << " 个线程是反模式!\n";
    std::cout << "  GPU 的优势: 硬件原生支持数千轻量级线程\n";
}

// ===== 2. 内存层次对比 =====
void demo_memory_hierarchy() {
    std::cout << "\n=== CPU vs GPU 内存层次 ===\n";

    std::cout << "  CPU 内存层次:\n";
    std::cout << "    Register    < 1ns  |  ~1KB per core\n";
    std::cout << "    L1 Cache    ~1ns   |  32KB per core\n";
    std::cout << "    L2 Cache    ~5ns   |  256KB-1MB per core\n";
    std::cout << "    L3 Cache    ~15ns  |  8-32MB shared\n";
    std::cout << "    RAM (DDR5)  ~100ns |  GBs\n\n";

    std::cout << "  GPU 内存层次 (NVIDIA):\n";
    std::cout << "    Register    即时    |  255 per thread\n";
    std::cout << "    Shared Mem  ~5ns   |  48-164KB per block\n";
    std::cout << "    L1 Cache    ~30ns  |  128KB per SM\n";
    std::cout << "    L2 Cache    ~200ns |  ~6MB shared\n";
    std::cout << "    HBM (HBM3)  ~400ns |  80GB+ bandwidth ~3TB/s\n\n";

    std::cout << "  关键差异:\n";
    std::cout << "    CPU: 大缓存、低延迟、分支预测、少量强核\n";
    std::cout << "    GPU: 小缓存、高延迟隐藏、无分支、数千弱核\n";
    std::cout << "    CPU 优化延迟，GPU 优化吞吐量\n";
}

// ===== 3. Stream 并发模拟 =====
void demo_stream_concept() {
    std::cout << "\n=== CUDA Stream 概念 ===\n";

    std::cout << "  Stream 1: [Copy In] → [Kernel A] → [Copy Out]\n";
    std::cout << "  Stream 2:          [Copy In] → [Kernel B] → [Copy Out]\n";
    std::cout << "  Stream 3:                   [Copy In] → [Kernel C] → ...\n\n";

    std::cout << "  关键:\n";
    std::cout << "  - 同 stream 内的操作顺序执行\n";
    std::cout << "  - 不同 stream 之间的操作可并发\n";
    std::cout << "  - 实现计算与数据传输重叠\n";
    std::cout << "  - 类似 CPU 上的 Pipeline 模式\n";
}

// ===== 4. CPU/GPU 异构模拟 =====
void demo_heterogeneous_computing() {
    std::cout << "\n=== CPU+GPU 异构计算模拟 ===\n";

    const int kNumTasks = 10;

    // 模拟: CPU 做预处理，GPU 做计算，CPU 做后处理
    auto cpu_preprocess = [](int i) -> int {
        std::this_thread::sleep_for(1ms); // I/O, 分支密集
        return i * 2;
    };

    auto gpu_compute = [](int input) -> int {
        std::this_thread::sleep_for(3ms); // 大规模并行计算
        return input * input;
    };

    auto cpu_postprocess = [](int result) {
        std::this_thread::sleep_for(1ms); // 聚合, 判断
        return result / 2;
    };

    auto start = std::chrono::high_resolution_clock::now();

    // 流水线: 预处理和计算重叠
    std::vector<int> results(kNumTasks);
    for (int i = 0; i < kNumTasks; ++i) {
        int pre = cpu_preprocess(i);
        int gpu = gpu_compute(pre);
        results[i] = cpu_postprocess(gpu);
    }

    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start);

    std::cout << "  处理 " << kNumTasks << " 个任务: "
              << elapsed.count() << " ms\n";
    std::cout << "  验证: result[0]=" << results[0]
              << " (期望 " << (0 * 2) * (0 * 2) / 2 << ")\n";
    std::cout << "  实际 GPU 编程可用 CUDA stream + "
              << "cudaMemcpyAsync 实现流水线重叠\n";
}

int main() {
    demo_simt_model();
    demo_memory_hierarchy();
    demo_stream_concept();
    demo_heterogeneous_computing();

    std::cout << "\n注意: GPU 编程需要 NVIDIA CUDA Toolkit 或 AMD ROCm。\n";
    std::cout << "本演示用 CPU 代码模拟 GPU 的核心概念。\n";
    return 0;
}
