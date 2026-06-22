// lecture1_part1.cpp - 加速比演示：带计时的并行求和
// =============================================================================
// CS149 第1讲核心概念：
//   - 加速比公式：speedup(P) = T(1) / T(P)
//     T(1) 是单处理器的执行时间，T(P) 是 P 个处理器的执行时间。
//     理想加速比为 P（线性加速），但现实中几乎不可能达到。
//
//   - 阿姆达尔定律（Amdahl's Law）：加速比受串行部分的限制
//     公式：S_perf(p) = 1 / ((1 - f_par) + f_par / p)
//     其中 f_par 是可并行化的执行时间占比。
//     即使有无限多个处理器，加速比也不会超过 1/(1-f_par)。
//     例如：90% 可并行化 → 最多 10 倍加速，与处理器数量无关。
//
//   - 通信开销限制加速比
//     线程间的同步（如 join）、数据合并（如 reduce）都需要串行执行。
//     这些通信成本无法被并行化，成为加速瓶颈。
//
//   - 工作负载不均衡限制加速比
//     如果某些线程获得的工作多于其他线程，
//     则快线程完成后需要等待慢线程（木桶效应），
//     导致部分处理器处于空闲状态，降低效率。
//
//   - 效率公式：efficiency = speedup / P
//     效率衡量实际加速比相对于理想加速比的比例。
//     FAST ≠ EFFICIENT：在 10 核上获得 2 倍加速 = 仅 20% 效率。
//
// 编译: g++ -std=c++17 -O2 -pthread lecture1_part1.cpp -o lecture1_part1
// =============================================================================

#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <random>
#include <cmath>

using namespace std::chrono;

// ---------------------------------------------------------------------------
// 串行求和（基准：1 个处理器）
// 每个元素依次累加到总数中，没有任何并行化。
// 这是性能比较的基线——所有加速比都相对于此时间计算。
// ---------------------------------------------------------------------------
double sequential_sum(const std::vector<double>& data) {
    double total = 0.0;
    for (double val : data) {
        total += val;
    }
    return total;
}

// ---------------------------------------------------------------------------
// 使用显式线程管理的并行求和
// 每个线程对数组中连续的一块（chunk）进行求和
//
// 工作方式：
//   1. 将数组平均分割为 num_threads 个块
//   2. 每个线程独立计算自己块的局部和（这部分完全并行，无同步开销）
//   3. 主线程收集所有局部和，执行最终归约（串行步骤——这就是通信成本）
//
// 注意：最终归约是阿姆达尔定律中"串行部分"的一个例子。
//       随着数据量增大，这个串行部分所占比例变小，但不为零。
// ---------------------------------------------------------------------------
double parallel_sum_chunks(const std::vector<double>& data, int num_threads) {
    size_t n = data.size();
    size_t chunk_size = (n + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;
    std::vector<double> partial_sums(num_threads, 0.0);

    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back([&, t]() {
            size_t start = t * chunk_size;
            size_t end = std::min(start + chunk_size, n);
            double local_sum = 0.0;
            for (size_t i = start; i < end; i++) {
                local_sum += data[i];
            }
            partial_sums[t] = local_sum;
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    // 最终归约（串行步骤——这就是通信成本）
    double total = 0.0;
    for (double s : partial_sums) {
        total += s;
    }
    return total;
}

// ---------------------------------------------------------------------------
// 模拟不均衡的工作负载分配（某些线程获得更多工作）
//
// 分配策略：线程 t 获得 (t+1) 倍的基础块大小。
//   线程 0 获得 1 份
//   线程 1 获得 2 份
//   线程 2 获得 3 份
//   以此类推...
//
// 这模拟了现实场景中数据依赖不同导致的工作量差异。
// 在并行编程中，动态负载均衡（如 work stealing）可用于缓解此问题。
// ---------------------------------------------------------------------------
double parallel_sum_unbalanced(const std::vector<double>& data, int num_threads) {
    size_t n = data.size();
    // 给后续线程越来越多的数据量，模拟负载不均衡
    std::vector<size_t> chunks(num_threads, 0);
    size_t total_assigned = 0;
    for (int t = 0; t < num_threads; t++) {
        // 线程 t 获得 (t+1) 倍的基础块大小
        chunks[t] = (t + 1) * (n / (num_threads * (num_threads + 1) / 2));
        if (t == num_threads - 1) {
            chunks[t] = n - total_assigned; // 最后一个线程获得剩余数据
        }
        total_assigned += chunks[t];
    }
    // 确保覆盖所有数据
    if (total_assigned < n) chunks.back() += n - total_assigned;

    std::vector<std::thread> threads;
    std::vector<double> partial_sums(num_threads, 0.0);

    size_t offset = 0;
    for (int t = 0; t < num_threads; t++) {
        size_t chunk = chunks[t];
        size_t start = offset;
        threads.emplace_back([&, t, start, chunk]() {
            double local_sum = 0.0;
            for (size_t i = start; i < start + chunk && i < data.size(); i++) {
                local_sum += data[i];
            }
            partial_sums[t] = local_sum;
        });
        offset += chunk;
    }

    for (auto& th : threads) {
        th.join();
    }

    double total = 0.0;
    for (double s : partial_sums) {
        total += s;
    }
    return total;
}

// ---------------------------------------------------------------------------
// 阿姆达尔定律计算器
// S_perf(p) = 1 / (1 - f_perf + f_perf / p)
// 其中 f_perf = 可并行化的工作占比
//
// 直观理解：
//   - 串行部分 1-f_perf 不随处理器增加而加速
//   - 并行部分 f_perf 可以除以 p 个处理器
//   - 无限处理器时：加速比趋近于 1/(1-f_perf)
// ---------------------------------------------------------------------------
double amdahl_speedup(int processors, double parallel_fraction) {
    return 1.0 / ((1.0 - parallel_fraction) + parallel_fraction / processors);
}

// ---------------------------------------------------------------------------
// 基准测试辅助函数：测量函数的执行时间
// 使用高精度时钟测量挂钟时间（wall-clock time）。
// 返回毫秒为单位的耗时。
// ---------------------------------------------------------------------------
template<typename Func, typename... Args>
double benchmark(Func func, Args&&... args) {
    auto start = high_resolution_clock::now();
    func(std::forward<Args>(args)...);
    auto end = high_resolution_clock::now();
    return duration_cast<microseconds>(end - start).count() / 1000.0; // 毫秒
}

// =============================================================================
int main() {
    std::cout << "=== CS149 第1讲：加速比与并行效率 ===\n" << std::endl;

    // ---- 第一部分：并行加速比测量 ----
    std::cout << "[1] 测量数组求和的并行加速比\n" << std::endl;

    const size_t N = 100'000'000; // 1亿个元素
    std::cout << "    数组大小：" << N << " 个 double（"
              << (N * sizeof(double) / (1024.0 * 1024.0)) << " MB）\n" << std::endl;

    // 生成随机数据
    std::vector<double> data(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < N; i++) {
        data[i] = dist(rng);
    }

    // 基准测试：串行求和
    double seq_time = benchmark([&]() {
        volatile double r = sequential_sum(data);
    });

    // 多次运行取稳定结果
    double seq_sum = sequential_sum(data);
    std::cout << "    串行求和结果 = " << std::fixed << std::setprecision(1)
              << seq_sum << "\n";
    std::cout << "    串行耗时 = " << seq_time << " ms\n" << std::endl;

    // 测量不同线程数下的并行加速比
    std::cout << "    " << std::left << std::setw(10) << "线程数"
              << std::setw(14) << "耗时(ms)"
              << std::setw(10) << "加速比"
              << std::setw(12) << "效率" << std::endl;
    std::cout << "    " << std::string(46, '-') << std::endl;

    int max_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (max_threads == 0) max_threads = 8;

    for (int p = 1; p <= max_threads; p++) {
        double par_time = benchmark([&]() {
            volatile double r = parallel_sum_chunks(data, p);
        });
        double speedup = seq_time / par_time;
        double efficiency = speedup / p * 100.0;

        std::cout << "    " << std::left << std::setw(10) << p
                  << std::setw(14) << std::fixed << std::setprecision(2) << par_time
                  << std::setw(10) << std::setprecision(2) << speedup << "x"
                  << std::setw(12) << std::setprecision(1) << efficiency << "%"
                  << std::endl;
    }

    // ---- 第二部分：阿姆达尔定律可视化 ----
    std::cout << "\n[2] 阿姆达尔定律：理论加速比上限\n" << std::endl;
    std::cout << "    加速比(P) = 1 / ((1 - f_par) + f_par / P)\n" << std::endl;

    std::vector<double> parallel_fractions = {0.50, 0.75, 0.90, 0.95, 0.99};
    std::vector<int> processor_counts = {1, 2, 4, 8, 16, 32, 64, 128, 1024};

    std::cout << "    " << std::setw(8) << "P";
    for (double f : parallel_fractions) {
        std::cout << std::setw(10) << ("f=" + std::to_string(static_cast<int>(f*100)) + "%");
    }
    std::cout << std::endl;
    std::cout << "    " << std::string(58, '-') << std::endl;

    for (int p : processor_counts) {
        std::cout << "    " << std::setw(8) << p;
        for (double f : parallel_fractions) {
            double sp = amdahl_speedup(p, f);
            std::cout << std::setw(10) << std::fixed << std::setprecision(2) << sp;
        }
        std::cout << std::endl;
    }

    // ---- 第三部分：工作负载不均衡演示 ----
    std::cout << "\n[3] 工作负载不均衡对加速比的影响\n" << std::endl;
    std::cout << "    均衡 vs 不均衡工作分配（4 线程）：\n" << std::endl;

    double bal_time = benchmark([&]() {
        volatile double r = parallel_sum_chunks(data, 4);
    });
    double unbal_time = benchmark([&]() {
        volatile double r = parallel_sum_unbalanced(data, 4);
    });

    std::cout << "    均衡分块：   " << std::fixed << std::setprecision(2)
              << bal_time << " ms\n";
    std::cout << "    不均衡分块： " << std::setprecision(2)
              << unbal_time << " ms\n";
    std::cout << "    不均衡导致的减速： " << std::setprecision(1)
              << (unbal_time / bal_time - 1.0) * 100 << "%\n";

    // ---- 第四部分：核心要点 ----
    std::cout << "\n[4] 第1讲核心要点\n";
    std::cout << "    - 加速比 = T(1) / T(P)，理论上限为 P\n";
    std::cout << "    - 通信开销限制了实际加速比\n";
    std::cout << "    - 阿姆达尔定律：大规模时串行部分主导加速瓶颈\n";
    std::cout << "    - 工作负载不均衡降低效率（导致处理器空闲）\n";
    std::cout << "    - FAST ≠ EFFICIENT（10 核上 2 倍加速 = 仅 20% 效率）\n";

    return 0;
}
