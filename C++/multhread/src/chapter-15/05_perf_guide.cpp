// 05_perf_guide.cpp — Linux perf 工具使用指南
// 说明: 这个文件主要是文档性质的注释，演示如何在并发程序中使用 perf
// 实际是完整可编译的示例程序，可用于 perf 分析

#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

/*
 * ============================================================
 * Linux perf 工具使用指南
 * ============================================================
 *
 * 1. 安装 perf:
 *    sudo apt install linux-tools-common linux-tools-generic
 *
 * 2. 编译本程序:
 *    g++ -std=c++20 -O2 -g -pthread 05_perf_guide.cpp -o perf_demo
 *
 * 3. 采样 CPU 热点:
 *    perf record -g ./perf_demo
 *    perf report
 *
 * 4. 查看硬件事件统计:
 *    perf stat -e cycles,instructions,cache-misses,cache-references \
 *             -e branch-misses,branch-instructions \
 *             -e cpu-clock,task-clock \
 *             ./perf_demo
 *
 * 5. 分析特定函数的性能:
 *    perf record -e cycles:pp ./perf_demo
 *    perf annotate
 *
 * 6. 生成火焰图 (需要 FlameGraph 脚本):
 *    perf record -F 99 -g ./perf_demo
 *    perf script > out.perf
 *    stackcollapse-perf.pl out.perf > out.folded
 *    flamegraph.pl out.folded > flamegraph.svg
 *
 * 7. 分析锁竞争:
 *    perf lock record ./perf_demo
 *    perf lock report
 *
 * 8. 分析 cache 行为:
 *    perf stat -e L1-dcache-loads,L1-dcache-load-misses \
 *             -e LLC-loads,LLC-load-misses \
 *             -e dTLB-loads,dTLB-load-misses \
 *             ./perf_demo
 *
 * 关键指标解读:
 *   - IPC (instructions per cycle): > 1 良好, < 0.5 CPU 在等待
 *   - cache-miss ratio: < 1% 优秀, > 10% 需要优化
 *   - branch-miss ratio: < 1% 优秀, > 5% 考虑优化分支预测
 *   - CPU migrations: 频繁迁移说明需要 CPU 绑核
 */

// ===== 可被 perf 分析的工作负载 =====

// 场景 A: CPU 密集型 (适合 perf stat)
void cpu_intensive_work() {
    volatile long long sum = 0;
    for (long long i = 0; i < 100'000'000; ++i) {
        sum += i;
    }
}

// 场景 B: Cache 敏感型 (适合 perf stat cache)
void cache_sensitive_work(std::vector<int>& data) {
    // 顺序访问 — cache 友好
    long long sum = 0;
    for (size_t i = 0; i < data.size(); ++i) {
        sum += data[i];
    }
    volatile long long avoid_opt = sum;
    (void)avoid_opt;
}

// 场景 C: 随机访问 — cache 不友好
void cache_unfriendly_work(std::vector<int>& data,
                             std::vector<size_t>& indices) {
    long long sum = 0;
    for (size_t i : indices) {
        sum += data[i];
    }
    volatile long long avoid_opt = sum;
    (void)avoid_opt;
}

// 场景 D: 高锁竞争 (适合 perf lock)
void high_contention_work() {
    std::mutex mtx;
    long long counter = 0;
    const int kThreads = 8;

    std::vector<std::jthread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < 1'000'000; ++i) {
                std::lock_guard lock(mtx);
                ++counter;
            }
        });
    }
    threads.clear();

    volatile long long avoid_opt = counter;
    (void)avoid_opt;
}

// ===== main: 运行不同场景供 perf 分析 =====
int main() {
    std::cout << "perf 分析演示程序\n";
    std::cout << "==================\n\n";

    std::cout << "[1] CPU 密集型工作...\n";
    auto start = std::chrono::high_resolution_clock::now();
    cpu_intensive_work();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start);
    std::cout << "    耗时: " << elapsed.count() << " ms\n\n";

    std::cout << "[2] Cache 对比测试...\n";
    const size_t kSize = 10'000'000;
    std::vector<int> data(kSize);
    for (size_t i = 0; i < kSize; ++i) data[i] = i % 100;

    start = std::chrono::high_resolution_clock::now();
    cache_sensitive_work(data);
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start);
    std::cout << "    顺序访问: " << elapsed.count() << " ms\n";

    // 准备随机索引
    std::vector<size_t> indices(kSize);
    for (size_t i = 0; i < kSize; ++i) indices[i] = i;
    // 随机打乱
    for (size_t i = kSize - 1; i > 0; --i) {
        size_t j = rand() % (i + 1);
        std::swap(indices[i], indices[j]);
    }

    start = std::chrono::high_resolution_clock::now();
    cache_unfriendly_work(data, indices);
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start);
    std::cout << "    随机访问: " << elapsed.count() << " ms\n\n";

    std::cout << "[3] 高锁竞争场景...\n";
    start = std::chrono::high_resolution_clock::now();
    high_contention_work();
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start);
    std::cout << "    耗时: " << elapsed.count() << " ms\n\n";

    std::cout << "使用 perf 分析建议:\n";
    std::cout << "  perf stat -d ./perf_demo         # 基本统计\n";
    std::cout << "  perf record -g ./perf_demo       # 采样调用栈\n";
    std::cout << "  perf stat -e cache-misses ./perf_demo  # Cache 分析\n";
    std::cout << "  perf lock record ./perf_demo     # 锁分析\n";

    return 0;
}
