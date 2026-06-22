// 01_benchmark_basic.cpp — 自制微基准测试框架
// 演示: 测量单/多线程吞吐量、扩展性、P99 延迟

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <functional>
#include <iostream>
#include <mutex>
#include <numeric>
#include <thread>
#include <vector>

using namespace std::chrono_literals;
using Clock = std::chrono::high_resolution_clock;

// ===== 1. 基础: 测量函数执行时间 =====
template <typename F>
double measure_ms(F&& func, int iterations = 1) {
    auto start = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        Clock::now() - start);
    return elapsed.count() / 1000.0 / iterations;
}

// ===== 2. 测量多线程吞吐量 =====
struct ThroughputResult {
    double ops_per_sec;
    double total_time_ms;
    long long total_ops;
};

template <typename WorkerFunc>
ThroughputResult measure_throughput(int num_threads,
                                     std::chrono::milliseconds duration,
                                     WorkerFunc worker) {
    std::atomic<long long> ops{0};
    std::atomic<bool> running{true};

    std::vector<std::jthread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&]() {
            while (running.load(std::memory_order_relaxed)) {
                worker();
                ops.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    auto start = Clock::now();
    std::this_thread::sleep_for(duration);
    running.store(false, std::memory_order_relaxed);

    for (auto& t : threads) t.join();

    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        Clock::now() - start);
    double secs = elapsed.count() / 1'000'000.0;

    return {ops.load() / secs, elapsed.count() / 1000.0, ops.load()};
}

// ===== 3. 测量 P50/P99 延迟 =====
struct LatencyStats {
    double p50_us;
    double p99_us;
    double avg_us;
    double max_us;
};

LatencyStats measure_latency(int num_threads, int ops_per_thread,
                              std::function<void()> operation) {
    std::vector<double> latencies;
    latencies.reserve(num_threads * ops_per_thread);
    std::mutex mtx;

    std::vector<std::jthread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < ops_per_thread; ++i) {
                auto start = Clock::now();
                operation();
                auto elapsed =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        Clock::now() - start);
                std::lock_guard lock(mtx);
                latencies.push_back(elapsed.count() / 1000.0);
            }
        });
    }
    threads.clear();

    std::sort(latencies.begin(), latencies.end());
    size_t n = latencies.size();
    double avg = std::accumulate(latencies.begin(), latencies.end(), 0.0) / n;
    double p50 = latencies[n * 50 / 100];
    double p99 = latencies[n * 99 / 100];
    double max = latencies.back();

    return {p50, p99, avg, max};
}

// ===== 4. 扩展性测试 =====
void demo_scalability() {
    std::cout << "=== 扩展性测试 (Amdahl 定律演示) ===\n\n";

    // 模拟: 80% 并行 + 20% 串行的工作负载
    auto mixed_workload = [](int thread_id) {
        long long sum = 0;
        // 并行部分 (80%)
        for (int i = 0; i < 100000; ++i) {
            sum += i * thread_id;
        }
        // 模拟串行部分 (20%) — 用 atomic 模拟
        volatile int avoid_opt = static_cast<int>(sum);
        (void)avoid_opt;
    };

    auto pure_parallel = [](int) {
        long long sum = 0;
        for (int i = 0; i < 125000; ++i) {
            sum += i;
        }
        volatile int avoid_opt = static_cast<int>(sum);
        (void)avoid_opt;
    };

    std::vector<int> thread_counts = {1, 2, 4, 8, 16};

    std::cout << std::setw(8) << "Threads"
              << std::setw(15) << "纯并行(ms)"
              << std::setw(15) << "80%并行(ms)"
              << std::setw(15) << "理想加速\n";

    double base_parallel = 0;
    double base_mixed = 0;

    for (int n : thread_counts) {
        // 纯并行
        auto start = Clock::now();
        std::vector<std::jthread> threads;
        for (int t = 0; t < n; ++t) {
            threads.emplace_back(pure_parallel, t);
        }
        threads.clear();
        double parallel_ms =
            std::chrono::duration_cast<std::chrono::microseconds>(
                Clock::now() - start).count() / 1000.0;
        if (n == 1) base_parallel = parallel_ms;

        // 混合负载
        start = Clock::now();
        threads.clear();
        for (int t = 0; t < n; ++t) {
            threads.emplace_back(mixed_workload, t);
        }
        threads.clear();
        double mixed_ms =
            std::chrono::duration_cast<std::chrono::microseconds>(
                Clock::now() - start).count() / 1000.0;
        if (n == 1) base_mixed = mixed_ms;

        // Amdahl: 80%并行, S=0.2 → max_speedup = 1/(0.2+0.8/n)
        double amdahl = 1.0 / (0.2 + 0.8 / n);

        std::cout << std::setw(8) << n
                  << std::setw(13) << std::fixed << std::setprecision(1)
                  << parallel_ms
                  << std::setw(15) << mixed_ms
                  << std::setw(13) << amdahl << "x\n";
    }

    std::cout << "\n  纯并行: 4核 → " << base_parallel
              << "ms (几乎线性扩展)\n";
    std::cout << "  80%并行: 受 Amdahl 定律限制，"
              << "最大加速 = 1/0.2 = 5x\n";
}

// ===== 5. 基础吞吐量测试 =====
void demo_throughput() {
    std::cout << "\n=== 吞吐量测试 ===\n";

    int counter = 0;
    std::mutex mtx;

    auto locked_op = [&]() {
        std::lock_guard lock(mtx);
        ++counter;
    };

    for (int threads : {1, 2, 4}) {
        auto result = measure_throughput(threads, 200ms, locked_op);
        std::cout << "  " << threads << " 线程: "
                  << std::fixed << std::setprecision(0)
                  << result.ops_per_sec << " ops/s"
                  << " (总 " << result.total_ops << " 次)\n";
    }
    std::cout << "  注意: 高竞争锁的吞吐量可能随线程数增加反而下降\n";
}

int main() {
    demo_scalability();
    demo_throughput();

    std::cout << "\n基准测试核心: "
              << "Latency(延迟) vs Throughput(吞吐量) vs Scalability(扩展性)\n";
    return 0;
}
