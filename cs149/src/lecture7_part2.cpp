// lecture7_part2.cpp
// Stanford CS149 第7讲：GPU 架构与 CUDA 编程
// 第二部分：一维卷积 — 朴素版本 vs. 共享内存版本
//
// 演示第7讲中一维卷积的两种实现方式：
//   版本一（朴素）：每个线程直接从 global memory 读取数据（无数据复用）
//   版本二（共享内存）：线程协作将数据加载到 shared memory 中（数据复用）
//
// 核心概念对比：
//   - Global Memory（全局内存 / DRAM）：
//     位于 GPU 芯片外部的 HBM/DRAM，容量大（几 GB~几十 GB），
//     但延迟极高（约 300~500 个时钟周期），bandwidth 相对较低。
//     所有 SM 都可以访问 global memory。
//   - Shared Memory（共享内存 / SRAM）：
//     位于 SM 内部的片上 SRAM，容量小（通常 48~128 KB/block），
//     延迟极低（约 20 个时钟周期），bandwidth 极高。
//     仅同一个 block 内的线程可以访问其各自的 shared memory。
//   - 在真实 GPU 中，shared memory 的速度大约是 global memory 的 100 倍。
//   - 本模拟通过统计 "内存访问次数" 来量化两种方案的差异。
//
// 版本一问题（朴素卷积）：
//   每个线程独立从 global memory 读取 filter_size 个元素。
//   相邻线程会重复读取大量相同数据，造成 global memory 访问冗余。
//   例如窗口为 3 的卷积，每个元素被 3 个 thread 重复读取。
//
// 版本二解决方案（共享内存协作加载）：
//   block 内所有线程协作，每个线程将 1 个元素加载到 shared memory 中。
//   额外线程加载 "halo 元素"（block 边界以外的 2 个额外元素）。
//   __syncthreads() 同步后，所有线程从 shared memory 读取数据。
//   这大幅减少了 global memory 访问次数。
//
// 编译：g++ -std=c++17 -pthread lecture7_part2.cpp -o lecture7_part2
// 运行：./lecture7_part2

#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <iomanip>
#include <cstring>
#include <chrono>

// ============================================================================
// 模拟配置参数
// ============================================================================

constexpr int THREADS_PER_BLK   = 128;
constexpr int CONV_FILTER_SIZE  = 3;

// 模拟内存访问的全局计数器
// 这些计数器分别代表从 GPU 全局内存（DRAM）和共享内存（片上 SRAM）的读取次数
// 使用 std::atomic 保证多线程并发递增时的正确性
std::atomic<long long> global_reads{0};
std::atomic<long long> shared_reads{0};

// ============================================================================
// 版本一：朴素实现 — 每个线程直接从 global memory 读取
//
// 等效 CUDA 代码：
//   __global__ void convolve(int N, float* input, float* output) {
//       int index = blockIdx.x * blockDim.x + threadIdx.x;
//       float result = 0.0f;
//       for (int i=0; i<3; i++)
//           result += input[index + i];   // 直接从 global memory 读取
//       output[index] = result / 3.f;
//   }
//
// 问题：相邻线程（如 index 和 index+1）之间存在大量数据重叠访问。
//       例如 filter_size=3 时，每个元素被 3 个 thread 各读取一次。
//       总 global memory 读取次数 = N * filter_size = 3N
// ============================================================================

void convolveV1_naive(const float* input, float* output,
                      int startIndex, int blockThreads, int totalN)
{
    for (int t = 0; t < blockThreads; t++) {
        int index = startIndex + t;
        if (index >= totalN) continue;  // 边界保护

        float result = 0.0f;
        // 每个线程从 "global memory" 读取 3 个元素
        // 这些读操作都直接访问 DRAM，延迟很高
        for (int i = 0; i < CONV_FILTER_SIZE; i++) {
            result += input[index + i];
            global_reads++;  // 统计 global memory 读取次数
        }
        output[index] = result / static_cast<float>(CONV_FILTER_SIZE);
    }
}

// ============================================================================
// 版本二：共享内存 — 线程协作加载数据
//
// 等效 CUDA 代码：
//   __global__ void convolve(int N, float* input, float* output) {
//       __shared__ float support[THREADS_PER_BLK+2];
//       int index = blockIdx.x * blockDim.x + threadIdx.x;
//       // 协作加载：每个线程将一个元素加载到 shared memory
//       support[threadIdx.x] = input[index];
//       // 额外的 2 个线程加载 block 边界外的 "halo" 元素
//       if (threadIdx.x < 2)
//           support[THREADS_PER_BLK+threadIdx.x] = input[index+THREADS_PER_BLK];
//       __syncthreads();  // 同步 barrier：确保所有数据已加载完毕
//       float result = 0.0f;
//       // 从 shared memory 读取（快速片上访问）
//       for (int i=0; i<3; i++)
//           result += support[threadIdx.x + i];
//       output[index] = result / 3.f;
//   }
//
// 优势：每个 block 只需加载 (THREADS_PER_BLK + 2) 个元素到 shared memory，
//       之后的卷积计算都从快速片上 SRAM 读取。
//       Global memory 总读取量 ≈ N + 2*numBlocks，远小于朴素版本的 3N。
// ============================================================================

void convolveV2_shared(const float* input, float* output,
                       int startIndex, int blockThreads, int totalN)
{
    // 每个 block 的 shared memory（模拟为栈上数组）
    // 在真实 CUDA 中，这是 __shared__ float support[THREADS_PER_BLK+2];
    // 额外 +2 是为了存储 block 右边界外的 "halo" 元素
    float support[THREADS_PER_BLK + 2] = {};

    // 协作加载阶段：每个线程将一个元素加载到 shared memory
    // 这是数据复用的关键：每个元素从 global memory 只被加载一次
    for (int t = 0; t < blockThreads; t++) {
        int index = startIndex + t;
        if (index < totalN) {
            support[t] = input[index];
            global_reads++;  // 从 global 加载到 shared（仅此一次）
        }
    }

    // 额外线程加载 "halo" 元素（block 边界之外的 +2 元素）
    // 等价于 CUDA 代码：if (threadIdx.x < 2)
    // 这些 halo 元素保证了 block 边界处卷积的正确性
    int nextIndex = startIndex + blockThreads;
    if (nextIndex < totalN) {
        support[blockThreads] = input[nextIndex];
        global_reads++;
    }
    if (nextIndex + 1 < totalN) {
        support[blockThreads + 1] = input[nextIndex + 1];
        global_reads++;
    }

    // __syncthreads() barrier — 所有线程等待协作加载完成后才继续
    // （在我们的顺序模拟中，这一步是隐式的）

    // 计算阶段：每个线程从 shared memory 读取数据完成卷积
    for (int t = 0; t < blockThreads; t++) {
        int index = startIndex + t;
        if (index >= totalN) continue;

        float result = 0.0f;
        for (int i = 0; i < CONV_FILTER_SIZE; i++) {
            result += support[t + i];
            shared_reads++;  // 从 shared memory 读取（快速片上访问）
        }
        output[index] = result / static_cast<float>(CONV_FILTER_SIZE);
    }
}

// ============================================================================
// Host 端：启动所有 block
// 使用 template 参数 KernelFunc 支持两种 kernel 版本
// ============================================================================

template<typename KernelFunc>
void launchKernel(KernelFunc kernel,
                  const float* input, float* output,
                  int totalN, int threadsPerBlk)
{
    // 上取整除法计算所需 block 数量
    int numBlocks = (totalN + threadsPerBlk - 1) / threadsPerBlk;

    std::vector<std::thread> blockThreads;
    for (int blk = 0; blk < numBlocks; blk++) {
        int startIndex = blk * threadsPerBlk;
        int blkThreads = std::min(threadsPerBlk, totalN - startIndex);

        blockThreads.emplace_back(
            kernel, input, output, startIndex, blkThreads, totalN
        );
    }

    for (auto& t : blockThreads) {
        t.join();
    }
}

// ============================================================================
// 性能计时辅助函数
// 使用 high_resolution_clock 精确测量执行时间（毫秒）
// ============================================================================

template<typename Func>
double measureTime(Func f)
{
    auto start = std::chrono::high_resolution_clock::now();
    f();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// ============================================================================
// 主函数
// ============================================================================

int main()
{
    std::cout << "==================================================\n";
    std::cout << "第7讲 第二部分：一维卷积内存访问分析\n";
    std::cout << "==================================================\n\n";

    constexpr int N = 1 << 20;  // 1,048,576 个元素（约 1M）
    constexpr int outputN = N - (CONV_FILTER_SIZE - 1);

    // 分配并初始化输入数据
    std::vector<float> input(N);
    for (int i = 0; i < N; i++) {
        input[i] = static_cast<float>(i + 1);
    }

    // 第一步：版本一 — 朴素实现（全部使用 global memory 访问）
    {
        std::cout << "--- 版本一：朴素实现（仅使用 global memory）---\n";
        std::vector<float> outputV1(outputN, 0.0f);
        global_reads = 0;
        shared_reads = 0;

        double timeV1 = measureTime([&]() {
            launchKernel(convolveV1_naive, input.data(), outputV1.data(),
                        outputN, THREADS_PER_BLK);
        });

        std::cout << "  数据大小：" << outputN << " 个输出元素\n";
        std::cout << "  线程块数：" << (outputN + THREADS_PER_BLK - 1) / THREADS_PER_BLK << "\n";
        std::cout << "  Global memory 读取次数：" << global_reads.load() << "\n";
        std::cout << "    每个输出元素平均："
                  << static_cast<double>(global_reads.load()) / outputN << "\n";
        std::cout << "  执行时间：" << std::fixed << std::setprecision(2) << timeV1 << " 毫秒\n";
        std::cout << "  输出样本 output[0..4]：";
        for (int i = 0; i < 5; i++) std::cout << outputV1[i] << " ";
        std::cout << "\n\n";
    }

    // 第二步：版本二 — 共享内存实现（协作加载）
    {
        std::cout << "--- 版本二：共享内存（协作加载）---\n";
        std::vector<float> outputV2(outputN, 0.0f);
        global_reads = 0;
        shared_reads = 0;

        double timeV2 = measureTime([&]() {
            launchKernel(convolveV2_shared, input.data(), outputV2.data(),
                        outputN, THREADS_PER_BLK);
        });

        int numBlocks = (outputN + THREADS_PER_BLK - 1) / THREADS_PER_BLK;
        std::cout << "  数据大小：" << outputN << " 个输出元素\n";
        std::cout << "  线程块数：" << numBlocks << "\n";
        std::cout << "  Global memory 读取次数：" << global_reads.load() << "\n";
        std::cout << "    （每个 block 将 " << (THREADS_PER_BLK + 2)
                  << " 个元素加载到 shared memory）\n";
        std::cout << "    每个输出元素平均："
                  << static_cast<double>(global_reads.load()) / outputN << "\n";
        std::cout << "  Shared memory 读取次数：" << shared_reads.load() << "\n";
        std::cout << "  执行时间：" << std::fixed << std::setprecision(2) << timeV2 << " 毫秒\n";
        std::cout << "  输出样本 output[0..4]：";
        for (int i = 0; i < 5; i++) std::cout << outputV2[i] << " ";
        std::cout << "\n\n";
    }

    // 对比总结
    {
        std::cout << "--- 版本对比分析 ---\n";
        std::cout << "版本一（朴素实现）：\n";
        std::cout << "  共 " << outputN << " 个线程，每个线程从 global memory 读取 " << CONV_FILTER_SIZE
                  << " 个元素。\n";
        std::cout << "  Global memory 总读取次数：" << outputN << " × " << CONV_FILTER_SIZE
                  << " = " << outputN * CONV_FILTER_SIZE << "\n\n";

        int numBlocks = (outputN + THREADS_PER_BLK - 1) / THREADS_PER_BLK;
        std::cout << "版本二（共享内存）：\n";
        std::cout << "  共 " << numBlocks << " 个 block，每个 block 协作加载 "
                  << (THREADS_PER_BLK + 2) << " 个元素。\n";
        std::cout << "  Global memory 总读取次数：" << numBlocks << " × "
                  << (THREADS_PER_BLK + 2) << " = "
                  << numBlocks * (THREADS_PER_BLK + 2) << "\n\n";

        double reduction = 1.0 - static_cast<double>(numBlocks * (THREADS_PER_BLK + 2))
                                   / (outputN * CONV_FILTER_SIZE);
        std::cout << "Global memory 读取次数减少比例："
                  << std::fixed << std::setprecision(1) << reduction * 100 << "%\n";
    }

    std::cout << "\n==================================================\n";
    std::cout << "演示的核心概念：\n";
    std::cout << "  - 朴素方法：O(N*filter_size) 次 global 读取\n";
    std::cout << "  - 共享内存：O(N + 2*blocks) 次 global 读取\n";
    std::cout << "  - 线程间的协作数据加载（cooperative loading）\n";
    std::cout << "  - __syncthreads() barrier 同步概念\n";
    std::cout << "  - Block 边界处的 halo 元素处理\n";
    std::cout << "==================================================\n";

    return 0;
}
