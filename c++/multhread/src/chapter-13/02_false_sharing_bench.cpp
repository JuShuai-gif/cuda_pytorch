// 02_false_sharing_bench.cpp — 伪共享详细分析与基准测试
// 对比: 相同 cache line vs 分隔到不同 cache line 的性能差异

#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

// ================================================================
// 场景 1: 伪共享 —— 两个原子变量在同一 cache line
// ================================================================
struct SharedLine {
    std::atomic<long long> counter1{0};
    std::atomic<long long> counter2{0};
    // counter1 和 counter2 极大概率在同一 64 字节 cache line 中
};

// ================================================================
// 场景 2: 无伪共享 —— 用 padding 分隔到不同 cache line
// ================================================================
struct alignas(128) PaddedLine { // 对齐到 128，确保两个计数在不同 cache line
    std::atomic<long long> counter1{0};
    char padding[64]; // 确保 counter2 在下一个 cache line
    std::atomic<long long> counter2{0};
};

// ================================================================
// 基准测试: 测量给定配置下的吞吐量
// ================================================================
template <typename Config>
double measure_throughput(int num_threads, long long iterations) {
    Config config;
    std::vector<std::jthread> threads;

    auto start = std::chrono::high_resolution_clock::now();

    // 线程 1: 频繁写 counter1
    threads.emplace_back([&]() {
        for (long long i = 0; i < iterations; ++i) {
            config.counter1.store(i, std::memory_order_relaxed);
        }
    });

    // 线程 2: 频繁写 counter2
    threads.emplace_back([&]() {
        for (long long i = 0; i < iterations; ++i) {
            config.counter2.store(i, std::memory_order_relaxed);
        }
    });

    threads.clear();

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start);
    return elapsed.count();
}

// ================================================================
// 多轮测试取平均
// ================================================================
template <typename Config>
double benchmark(int rounds, int threads, long long iterations) {
    double total = 0;
    for (int r = 0; r < rounds; ++r) {
        total += measure_throughput<Config>(threads, iterations);
    }
    return total / rounds;
}

// ================================================================
// main
// ================================================================
int main() {
    std::cout << "=== 伪共享 (False Sharing) 性能对比 ===\n\n";

    std::cout << "sizeof(SharedLine) = " << sizeof(SharedLine) << "\n";
    std::cout << "sizeof(PaddedLine) = " << sizeof(PaddedLine) << "\n\n";

    const long long kIterations = 10'000'000;
    const int kRounds = 3;

    std::cout << "运行 " << kRounds << " 轮，每轮 " << kIterations
              << " 次写入...\n\n";

    double shared_time = benchmark<SharedLine>(kRounds, 2, kIterations);
    double padded_time = benchmark<PaddedLine>(kRounds, 2, kIterations);

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "  伪共享 (SharedLine):   " << shared_time << " ms\n";
    std::cout << "  无伪共享 (PaddedLine):  " << padded_time << " ms\n";
    std::cout << "  加速比: " << shared_time / padded_time << "x\n\n";

    std::cout << "结论:\n";
    std::cout << "  伪共享导致同一个 cache line 在两个核心间反复无效化，\n";
    std::cout << "  即使两个线程修改的是不同变量。通过 cache line padding\n";
    std::cout << "  将它们分离到不同缓存行，可大幅提升性能。\n";

    return 0;
}
