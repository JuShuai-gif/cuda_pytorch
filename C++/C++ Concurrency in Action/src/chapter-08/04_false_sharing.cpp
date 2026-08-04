/**
 * 04_false_sharing.cpp — 伪共享 (False Sharing) 的演示与解决方案
 *
 * 伪共享: 多个线程修改相邻内存位置的变量时, 这些变量共享同一缓存行 (64 字节),
 * 导致缓存一致性协议频繁失效, 性能急剧下降。
 *
 * 解决方案:
 *  1. alignas(64): 将变量对齐到独立的缓存行
 *  2. Padding: 在变量之间插入填充字节
 *  3. 每个线程使用线程局部变量, 最后归约
 *
 * 编译: g++ -std=c++20 -O2 -pthread 04_false_sharing.cpp -o false_sharing
 */

#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <array>
#include <cstddef>

// ============================================================================
// 场景1: 有伪共享的计数器 (糟糕的设计)
// ============================================================================
struct BadCounters {
    std::atomic<long long> counter1{0}; // 与 counter2 在同一缓存行
    std::atomic<long long> counter2{0}; // ⚠️ 伪共享!
    std::atomic<long long> counter3{0};
    std::atomic<long long> counter4{0};
};

// ============================================================================
// 场景2: 缓存行对齐的计数器 (良好的设计)
// ============================================================================
struct GoodCounters {
    alignas(64) std::atomic<long long> counter1{0}; // 独占缓存行
    alignas(64) std::atomic<long long> counter2{0}; // 独占缓存行
    alignas(64) std::atomic<long long> counter3{0};
    alignas(64) std::atomic<long long> counter4{0};
};

// ============================================================================
// 场景3: 使用填充 (padding) 手动隔离缓存行
// ============================================================================
struct PaddedCounter {
    alignas(64) std::atomic<long long> value{0};
    // 自动填充至缓存行边界
    char __padding[64 - sizeof(std::atomic<long long>)];
};
static_assert(sizeof(PaddedCounter) == 64, "PaddedCounter must be 64 bytes");

// ============================================================================
// 性能测试辅助
// ============================================================================
constexpr long long kIterations = 100'000'000;

template <typename CounterArray>
double run_benchmark(const std::string& name, CounterArray& counters, int num_counters) {
    std::vector<std::jthread> threads;
    threads.reserve(static_cast<size_t>(num_counters));

    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < num_counters; ++t) {
        threads.emplace_back([&counters, t]() {
            for (long long i = 0; i < kIterations; ++i) {
                counters[t].value.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    for (auto& th : threads) th.join();

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // 验证结果
    long long total = 0;
    for (int t = 0; t < num_counters; ++t) {
        total += counters[t].value.load();
    }

    std::cout << "  " << std::left << std::setw(35) << name
              << std::right << std::setw(8) << elapsed << " ms"
              << "  (合计: " << total << ")\n";

    return static_cast<double>(elapsed);
}

// ============================================================================
// 演示伪共享的可视化效果
// ============================================================================
void demonstrate_cache_line_layout() {
    std::cout << "=== 缓存行布局示意 ===\n\n";

    BadCounters bad;
    GoodCounters good;

    std::cout << "  sizeof(BadCounters)  = " << sizeof(BadCounters) << " bytes\n";
    std::cout << "  sizeof(GoodCounters) = " << sizeof(GoodCounters) << " bytes\n";
    std::cout << "  缓存行大小 (通常)   = 64 bytes\n\n";

    std::cout << "  BadCounters 布局:\n";
    std::cout << "  [counter1|counter2|counter3|counter4]  <- 同一缓存行, 伪共享!\n\n";

    std::cout << "  GoodCounters 布局:\n";
    std::cout << "  [counter1......................padding]\n";
    std::cout << "  [counter2......................padding]\n";
    std::cout << "  [counter3......................padding]\n";
    std::cout << "  [counter4......................padding]\n\n";
}

// ============================================================================
// 基准测试
// ============================================================================
void run_benchmarks() {
    std::cout << "=== 伪共享性能影响 ===\n\n";
    std::cout << "  每线程迭代 " << kIterations << " 次, " << std::jthread::hardware_concurrency()
              << " 个硬件线程\n\n";

    // 测试: 2 个计数器
    {
        std::cout << "--- 2个计数器 (2线程) ---\n";
        PaddedCounter padded[2];
        std::array<PaddedCounter, 2> unpadded;
        double t_padded = run_benchmark("Padded (无伪共享)", padded, 2);
        double t_unpadded = run_benchmark("Unpadded (有伪共享)", unpadded, 2);
        if (t_unpadded > 0) {
            std::cout << "  慢化倍数: " << std::fixed << std::setprecision(1)
                      << (t_unpadded / t_padded) << "x\n\n";
        }
    }

    // 测试: 4 个计数器
    {
        std::cout << "--- 4个计数器 (4线程) ---\n";
        PaddedCounter padded[4];
        std::array<PaddedCounter, 4> unpadded;
        double t_padded = run_benchmark("Padded (无伪共享)", padded, 4);
        double t_unpadded = run_benchmark("Unpadded (有伪共享)", unpadded, 4);
        if (t_unpadded > 0) {
            std::cout << "  慢化倍数: " << std::fixed << std::setprecision(1)
                      << (t_unpadded / t_padded) << "x\n\n";
        }
    }
}

// ============================================================================
// 替代方案: 线程局部累加 + 最终归约
// ============================================================================
void demonstrate_local_accumulation() {
    std::cout << "=== 线程局部累加方案 (避免伪共享的替代方法) ===\n\n";

    constexpr int kThreads = 4;
    constexpr long long kN = 100'000'000;

    // 方案A: 伪共享版本 (原子计数器数组)
    {
        alignas(64) std::array<std::atomic<long long>, kThreads> counters{};

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::jthread> threads;
        for (int t = 0; t < kThreads; ++t) {
            threads.emplace_back([&, t]() {
                for (long long i = 0; i < kN; ++i) {
                    counters[t].fetch_add(1, std::memory_order_relaxed);
                }
            });
        }
        for (auto& th : threads) th.join();
        auto end = std::chrono::high_resolution_clock::now();

        long long total = 0;
        for (auto& c : counters) total += c.load();

        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "  原子计数器 (可能伪共享): " << ms << " ms, total=" << total << "\n";
    }

    // 方案B: 线程局部变量, 最后归约 (完全消除伪共享)
    {
        std::vector<long long> local_sums(kThreads, 0);

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::jthread> threads;
        for (int t = 0; t < kThreads; ++t) {
            threads.emplace_back([&, t]() {
                long long local = 0;
                for (long long i = 0; i < kN; ++i) {
                    ++local; // 纯局部变量, 无竞争无伪共享
                }
                local_sums[t] = local;
            });
        }
        for (auto& th : threads) th.join();
        auto end = std::chrono::high_resolution_clock::now();

        long long total = 0;
        for (auto s : local_sums) total += s;

        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "  线程局部 + 归约 (无伪共享):  " << ms << " ms, total=" << total << "\n";
    }
}

// ============================================================================
// main
// ============================================================================
int main() {
    demonstrate_cache_line_layout();
    run_benchmarks();
    demonstrate_local_accumulation();
    return 0;
}
