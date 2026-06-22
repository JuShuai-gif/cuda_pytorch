// lecture2_part1.cpp - 多核执行与并行模式
// =============================================================================
// CS149 第2讲核心概念：
//   - 三种形式的并行执行：
//     1. 超标量（Superscalar）：在一条指令流内挖掘 ILP（硬件自动完成）
//        不改变编程模型，程序员无需干预。限于单个指令流中的依赖关系。
//
//     2. SIMD（单指令多数据）：由一个指令控制多个 ALU（在一个核内）
//        编译器生成向量指令，同时对多个数据元素执行同一操作。
//        例如：AVX2 可同时处理 8 个 32 位浮点数。
//
//     3. 多核（Multi-core）：多个独立核，每个运行各自的指令流
//        需要程序员显式创建线程或使用并行编程框架。
//        每个核有自己的一级缓存，共享最后一级缓存和内存。
//
//   - 多核时代：将晶体管预算用于制造更多核心，而非更复杂的单核
//     在单核设计中，晶体管被用于乱序执行引擎、分支预测器等。
//     但功耗墙和 ILP 墙限制了单核性能提升 → 转向多核。
//
//   - 数据并行表达式：forall 构造（声明循环迭代是独立的）
//     "forall" 是 Kayvon 提出的理想化并行抽象：
//     程序员声明"这些迭代之间没有依赖关系"，
//     运行时/编译器自动决定如何将迭代映射到执行资源上。
//     这比手动创建线程更高层、更安全。
//
//   - 表达并行：C++ threads vs. 数据并行抽象
//     C++ threads：底层、手动管理、容易出错
//     forall/ISPC/OpenMP：高层、编译器辅助、声明式
//
//   - 一致的执行流程是 SIMD 效率的前提
//     如果所有数据都要执行相同的操作（如 sin(x) 展开），
//     SIMD 可以达到 100% 利用率。
//     如果有条件分支（如 if x > 0 then...），
//     则需要掩码处理，效率降低。
//
//   - CPU 上的 SIMD：AVX2（256 位 → 8×32bit）、AVX512（512 位）、ARM Neon（128 位）
//     这些是实际的 SIMD 指令集扩展，编译器可以自动向量化或手动使用 intrinsic。
//
// 编译: g++ -std=c++17 -O2 lecture2_part1.cpp -o lecture2_part1
// =============================================================================

#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <functional>
#include <future>
#include <algorithm>

// =============================================================================
// sin(x) 泰勒展开：sin(x) = x - x^3/3! + x^5/5! - x^7/7! + ...
// （这是课程讲义中使用的示例程序）
//
// 泰勒展开的特点：
//   1. 所有迭代使用相同的计算模式（乘、除、符号翻转）→ 一致的执行流程
//   2. 每个元素的计算完全独立 → 适合数据并行化
//   3. 计算密集（多次乘除运算）→ 算术强度高
// =============================================================================
float sin_taylor(float x, int terms) {
    float value = x;
    float numer = x * x * x;       // x^3
    float denom = 6.0f;            // 3!
    int sign = -1;

    for (int j = 1; j <= terms; j++) {
        value += sign * numer / denom;
        numer *= x * x;
        denom *= (2 * j + 2) * (2 * j + 3);
        sign *= -1;
    }
    return value;
}

// ---------------------------------------------------------------------------
// 串行 sin(x)：在一个核上逐个元素处理
// 这是性能的基准（baseline），所有加速比都相对于此计算。
// ---------------------------------------------------------------------------
void sinx_sequential(int N, int terms, const float* x, float* result) {
    for (int i = 0; i < N; i++) {
        result[i] = sin_taylor(x[i], terms);
    }
}

// ---------------------------------------------------------------------------
// 使用 C++ 线程的并行 sin(x)（多核方式）
// 手动将工作分割到多个线程中
//
// 这种方式是"显式多线程编程"：
//   - 程序员需要手动决定分块策略
//   - 需要管理线程生命周期（创建、join）
//   - 存在竞态条件的风险（虽然本例中没有共享可变状态）
// ---------------------------------------------------------------------------
void sinx_parallel_threads(int N, int terms, const float* x, float* result,
                            int num_threads) {
    int chunk = (N + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back([=, &result]() {
            int start = t * chunk;
            int end = std::min(start + chunk, N);
            for (int i = start; i < end; i++) {
                result[i] = sin_taylor(x[i], terms);
            }
        });
    }

    for (auto& th : threads) th.join();
}

// ---------------------------------------------------------------------------
// 数据并行表达式：模拟 "forall" 构造
// 这是 Kayvon 设想的 forall：程序员声明迭代是独立的
//
// forall 与手动线程的区别：
//   - forall 是声明的（declarative）：你说"这可以并行"，而不是"如何并行"
//   - 线程是命令的（imperative）：你精确指定如何分割和执行
//
// 在这个模拟中，我们使用 std::async（C++ 的异步任务启动），
// 但仍然手动分块——真正的 forall 实现会由运行时自动管理分块。
// ---------------------------------------------------------------------------
void sinx_parallel_forall(int N, int terms, const float* x, float* result,
                           int num_threads) {
    // "forall" 抽象：程序员说"这些迭代是独立的"
    // 运行时/编译器决定如何将迭代映射到执行资源
    // 这里我们模拟自动分解为多个块

    int chunk = (N + num_threads - 1) / num_threads;
    std::vector<std::future<void>> futures;

    for (int t = 0; t < num_threads; t++) {
        int start = t * chunk;
        int end = std::min(start + chunk, N);
        futures.push_back(std::async(std::launch::async, [=, &result]() {
            for (int i = start; i < end; i++) {
                result[i] = sin_taylor(x[i], terms);
            }
        }));
    }

    for (auto& f : futures) f.wait();
}

// ---------------------------------------------------------------------------
// SIMD 模拟（8-wide，类似 AVX2）
// 手动一次处理 8 个元素，模拟 SIMD 向量操作
//
// 真实 SIMD 的工作方式：
//   1. 向量加载（如 _mm256_load_ps）：一次从内存加载 8 个 float
//   2. 向量计算（如 _mm256_mul_ps）：同时对 8 个 float 执行乘法
//   3. 向量存储（如 _mm256_store_ps）：一次将 8 个 float 写回内存
//
// 在这个模拟中，我们通过循环显式处理 8 个元素来近似 SIMD 行为。
// 真实 SIMD 使用特定的 CPU 指令，速度更快。
// ---------------------------------------------------------------------------
void sinx_simd_8wide(int N, int terms, const float* x, float* result) {
    // 一次处理 8 个元素（模拟 8-wide SIMD）
    for (int i = 0; i < N; i += 8) {
        float values[8];
        int end = std::min(i + 8, N);

        // "向量加载"：加载 8 个元素
        for (int k = 0; k < 8 && (i + k) < N; k++) {
            values[k] = x[i + k];
        }

        // "向量计算"：计算每个元素的 sin 值
        for (int k = 0; k < 8 && (i + k) < N; k++) {
            values[k] = sin_taylor(values[k], terms);
        }

        // "向量存储"：存储 8 个结果
        for (int k = 0; k < 8 && (i + k) < N; k++) {
            result[i + k] = values[k];
        }
    }
}

// ---------------------------------------------------------------------------
// 组合多核 + SIMD：4 核 × 8-wide SIMD = 并行处理 32 个元素
// （与课程示例匹配：4 核 Intel CPU 配合 AVX2）
//
// 这是现代 CPU 性能的完整图景：
//   峰值 FLOPs = 核数 × SIMD 宽度 × ALU 数 × 频率
//   例如 Intel i7-7700K：4 × 8 × 3 × 4.2 GHz ≈ 400 GFLOPs
//
// 每一层并行提供了独立的加速因子，它们是相乘的关系。
// ---------------------------------------------------------------------------
void sinx_multicore_simd(int N, int terms, const float* x, float* result) {
    const int SIMD_WIDTH = 8;
    const int NUM_CORES = 4;
    int total_parallelism = SIMD_WIDTH * NUM_CORES;

    // 每个核处理多个 SIMD_WIDTH 大小的块
    int elements_per_core = ((N + SIMD_WIDTH - 1) / SIMD_WIDTH + NUM_CORES - 1) / NUM_CORES
                            * SIMD_WIDTH;

    std::vector<std::thread> threads;
    for (int t = 0; t < NUM_CORES; t++) {
        threads.emplace_back([=, &result]() {
            int start = t * elements_per_core;
            int end = std::min(start + elements_per_core, N);
            // 每个核内部运行一个 8-wide SIMD 内循环
            for (int i = start; i < end; i += SIMD_WIDTH) {
                for (int k = 0; k < SIMD_WIDTH && (i + k) < N; k++) {
                    result[i + k] = sin_taylor(x[i + k], terms);
                }
            }
        });
    }

    for (auto& th : threads) th.join();
}

// ---------------------------------------------------------------------------
// 基准测试辅助函数
// ---------------------------------------------------------------------------
template<typename Func, typename... Args>
double benchmark_ms(Func func, Args&&... args) {
    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    func(std::forward<Args>(args)...);
    auto end = high_resolution_clock::now();
    return duration_cast<microseconds>(end - start).count() / 1000.0;
}

// =============================================================================
int main() {
    std::cout << "=== CS149 第2讲：多核与 SIMD 并行 ===\n" << std::endl;

    // ---- 第一部分：sin(x) 泰勒展开程序 ----
    std::cout << "[1] 示例程序：sin(x) 泰勒展开\n" << std::endl;
    std::cout << "    sin(x) = x - x^3/3! + x^5/5! - x^7/7! + ...\n" << std::endl;

    // 测试正确性
    float test_x = 0.5f;
    float result = sin_taylor(test_x, 5);
    std::cout << "    sin(0.5) ≈ " << std::fixed << std::setprecision(6) 
              << result << "（std::sin = " << std::sin(test_x) << "）\n" << std::endl;

    // ---- 第二部分：基准测试不同执行策略 ----
    const int N = 1'000'000;
    const int TERMS = 5;

    std::vector<float> x(N);
    std::vector<float> y(N, 0.0f);

    // 填充输入值，范围在 [-π, π]
    for (int i = 0; i < N; i++) {
        x[i] = (static_cast<float>(i) / N - 0.5f) * 2.0f * static_cast<float>(M_PI);
    }

    std::cout << "[2] 性能对比（N=" << N << " 个元素，" << TERMS 
              << " 项泰勒展开）\n" << std::endl;

    std::cout << "    " << std::setw(30) << "策略" 
              << std::setw(12) << "耗时(ms)" 
              << std::setw(10) << "加速比" << std::endl;
    std::cout << "    " << std::string(52, '-') << std::endl;

    // 串行基准
    std::fill(y.begin(), y.end(), 0.0f);
    double seq_time = benchmark_ms(sinx_sequential, N, TERMS, x.data(), y.data());
    std::cout << "    " << std::setw(30) << "串行（1 核）" 
              << std::setw(12) << std::fixed << std::setprecision(2) << seq_time
              << std::setw(10) << "1.00x" << std::endl;

    // 多核：2 线程
    std::fill(y.begin(), y.end(), 0.0f);
    double par2_time = benchmark_ms(sinx_parallel_threads, N, TERMS, 
                                     x.data(), y.data(), 2);
    std::cout << "    " << std::setw(30) << "多核（2 线程）" 
              << std::setw(12) << std::fixed << std::setprecision(2) << par2_time
              << std::setw(10) << std::setprecision(2) << (seq_time / par2_time) << "x" 
              << std::endl;

    // 多核：4 线程
    std::fill(y.begin(), y.end(), 0.0f);
    double par4_time = benchmark_ms(sinx_parallel_threads, N, TERMS, 
                                     x.data(), y.data(), 4);
    std::cout << "    " << std::setw(30) << "多核（4 线程）" 
              << std::setw(12) << std::setprecision(2) << par4_time
              << std::setw(10) << std::setprecision(2) << (seq_time / par4_time) << "x" 
              << std::endl;

    // SIMD 8-wide（单核）
    std::fill(y.begin(), y.end(), 0.0f);
    double simd_time = benchmark_ms(sinx_simd_8wide, N, TERMS, x.data(), y.data());
    std::cout << "    " << std::setw(30) << "SIMD 8-wide（1 核）" 
              << std::setw(12) << std::setprecision(2) << simd_time
              << std::setw(10) << std::setprecision(2) << (seq_time / simd_time) << "x" 
              << std::endl;

    // 多核 + SIMD（4 核 × 8-wide）
    std::fill(y.begin(), y.end(), 0.0f);
    double combined_time = benchmark_ms(sinx_multicore_simd, N, TERMS, x.data(), y.data());
    std::cout << "    " << std::setw(30) << "4 核 × 8-wide SIMD" 
              << std::setw(12) << std::setprecision(2) << combined_time
              << std::setw(10) << std::setprecision(2) << (seq_time / combined_time) << "x" 
              << std::endl;

    // ---- 第三部分：三种并行执行形式 ----
    std::cout << "\n[3] 三种并行执行形式\n" << std::endl;
    std::cout << "    ┌─────────────────┬──────────────────────────────────┐\n";
    std::cout << "    │ 超标量            │ 在一条指令流内挖掘 ILP           │\n";
    std::cout << "    │ （Superscalar）   │ （硬件在运行时自动发现）          │\n";
    std::cout << "    ├─────────────────┼──────────────────────────────────┤\n";
    std::cout << "    │ SIMD             │ 多个 ALU，一条指令               │\n";
    std::cout << "    │                  │ （编译器生成向量操作指令）        │\n";
    std::cout << "    ├─────────────────┼──────────────────────────────────┤\n";
    std::cout << "    │ 多核              │ 多条独立的指令流                │\n";
    std::cout << "    │ （Multi-core）    │ （程序员创建线程）               │\n";
    std::cout << "    └─────────────────┴──────────────────────────────────┘\n" << std::endl;

    // ---- 第四部分：计算吞吐量示例 ----
    std::cout << "[4] 真实处理器吞吐量示例\n" << std::endl;
    std::cout << "    Intel i7-7700K（4 核 × 8-wide AVX2 × 3 ALU × 4.2 GHz）：\n";
    double i7_flops = 4.0 * 8.0 * 3.0 * 4.2e9;
    std::cout << "    → " << std::fixed << std::setprecision(0) << i7_flops / 1e9 
              << " GFLOPs（课程中约为 400 GFLOPs）\n" << std::endl;

    std::cout << "    NVIDIA V100（80 SM × 64 fp32 ALU × 1.6 GHz）：\n";
    double v100_flops = 80.0 * 64.0 * 1.6e9;
    std::cout << "    → " << std::fixed << std::setprecision(0) << v100_flops / 1e12 
              << " TFLOPs（课程中约为 16 TFLOPs）\n" << std::endl;

    // ---- 第五部分：一致执行 vs. 分支发散执行 ----
    std::cout << "[5] 指令流一致性\n" << std::endl;
    std::cout << "    一致执行（Coherent）：相同的指令序列适用于\n"
              << "    所有数据元素（这是 SIMD 效率所必需的）\n" << std::endl;
    std::cout << "    分支发散执行（Divergent）：每个元素有不同的控制流\n"
              << "    → SIMD 通道被屏蔽 → 吞吐量降低\n" << std::endl;

    // ---- 第六部分：核心要点 ----
    std::cout << "\n[6] 第2讲核心要点\n" << std::endl;
    std::cout << "    - 多核时代之前：晶体管用于更花哨的单核（乱序执行、分支预测）\n";
    std::cout << "    - 多核时代：晶体管用于更多更简单的核心\n";
    std::cout << "    - 超标量：核内自动 ILP（硬件驱动）\n";
    std::cout << "    - SIMD：将控制成本分摊到多个 ALU 上（编译器驱动）\n";
    std::cout << "    - 多核：线程级并行（程序员驱动）\n";
    std::cout << "    - CPU 上的 SIMD：AVX2（256 位）、AVX512（512 位）、Neon（128 位）\n";
    std::cout << "    - GPU：极致 SIMD 宽度（8-32）+ 大量核心\n";
    std::cout << "    - 组合：多核 × SIMD × 频率 = 峰值吞吐量\n";

    return 0;
}
