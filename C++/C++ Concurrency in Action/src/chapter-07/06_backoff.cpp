// 06_backoff.cpp — 无锁编程的退避策略 (Backoff Strategies)
//
// 当 CAS 失败时，简单的 while 循环会导致:
//  - 高总线争用 (bus contention)
//  - CPU 功耗浪费
//  - 缓存一致性风暴
//
// 本演示对比 5 种退避策略的性能表现:
//   1. No Backoff (plain spin)
//   2. PAUSE only (x86 pause 指令)
//   3. Yield (让出 CPU 时间片)
//   4. Exponential Backoff (指数增长等待)
//   5. Randomized Backoff (随机等待)

#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

// ================================================================
// 5 种退避策略的比较测试
// ================================================================

// 策略 1: 无退避 —— 猛转
struct NoBackoff {
    void operator()() { /* do nothing */ }
};

// 策略 2: PAUSE 指令 —— 降低功耗和总线争用
struct PauseBackoff {
    void operator()() {
#if defined(__x86_64__) || defined(_M_X64)
        __asm__ volatile("pause" ::: "memory");
#endif
    }
};

// 策略 3: Yield —— 让出 CPU 时间片
struct YieldBackoff {
    void operator()() {
        std::this_thread::yield();
    }
};

// 策略 4: 指数退避 —— spin_count 越大等越久
class ExponentialBackoff {
public:
    void operator()() {
        // delay = 2^spin_count microseconds, capped
        int delay_us = std::min(1 << spin_count_, 1024);
        // 忙等 (模拟短延迟, 避免 sleep 的系统调用开销)
        auto start = std::chrono::high_resolution_clock::now();
        while (std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::high_resolution_clock::now() - start)
                   .count() < delay_us) {
            __asm__ volatile("pause" ::: "memory");
        }
        ++spin_count_;
    }

    void reset() { spin_count_ = 0; }

private:
    int spin_count_ = 0;
};

// 策略 5: 随机退避 —— 减少所有线程同时重试的概率
class RandomizedBackoff {
public:
    void operator()() {
        std::uniform_int_distribution<int> dist(0, max_delay_);
        int delay_us = dist(rng_);
        auto start = std::chrono::high_resolution_clock::now();
        while (std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::high_resolution_clock::now() - start)
                   .count() < delay_us) {
            __asm__ volatile("pause" ::: "memory");
        }
        // 逐渐增加最大延迟
        max_delay_ = std::min(max_delay_ * 2, 1024);
    }

    void reset() { max_delay_ = 1; }

private:
    std::mt19937 rng_{std::random_device{}()};
    int max_delay_ = 1;
};

// ================================================================
// 使用退避策略的 CAS 循环
// ================================================================
template <typename Backoff>
long long cas_with_backoff(int num_threads, long long ops_per_thread) {
    std::atomic<long long> counter{0};
    std::atomic<bool> start{false};

    std::vector<std::jthread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&]() {
            // 局部退避状态 (每线程独立)
            Backoff backoff;

            // 等待开始信号
            while (!start.load(std::memory_order_relaxed)) {
                std::this_thread::yield();
            }

            for (long long i = 0; i < ops_per_thread; ++i) {
                long long expected = counter.load(std::memory_order_relaxed);
                while (!counter.compare_exchange_weak(
                    expected, expected + 1,
                    std::memory_order_release,
                    std::memory_order_relaxed)) {
                    backoff(); // 退避!
                }
                if constexpr (std::is_same_v<Backoff, ExponentialBackoff> ||
                              std::is_same_v<Backoff, RandomizedBackoff>) {
                    backoff.reset(); // 成功后重置退避状态
                }
            }
        });
    }

    // 同步启动
    start.store(true, std::memory_order_release);
    auto t0 = std::chrono::high_resolution_clock::now();

    for (auto& t : threads) t.join();

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - t0);

    return elapsed.count();
}

// ================================================================
// 基准测试入口
// ================================================================
int main() {
    std::cout << "=== 退避策略 (Backoff) 性能对比 ===\n";
    std::cout << "高竞争场景: 8 线程 × 500,000 次 CAS 递增\n";
    std::cout << "核心数: " << std::thread::hardware_concurrency() << "\n\n";

    const int kThreads = 8;
    const long long kOps = 500'000;
    const int kRounds = 3;

    struct Strategy {
        std::string name;
        long long (*func)(int, long long);
    };

    std::cout << std::setw(25) << "策略"
              << std::setw(12) << "平均耗时"
              << std::setw(12) << "相对性能" << "\n";
    std::cout << std::string(49, '-') << "\n";

    long long baseline = 0;
    for (auto& [name, func] : {
             Strategy{"1. No Backoff",
                      &cas_with_backoff<NoBackoff>},
             Strategy{"2. PAUSE only",
                      &cas_with_backoff<PauseBackoff>},
             Strategy{"3. std::yield",
                      &cas_with_backoff<YieldBackoff>},
             Strategy{"4. Exponential",
                      &cas_with_backoff<ExponentialBackoff>},
             Strategy{"5. Randomized",
                      &cas_with_backoff<RandomizedBackoff>}}) {
        long long total = 0;
        for (int r = 0; r < kRounds; ++r) {
            total += func(kThreads, kOps);
        }
        long long avg = total / kRounds;
        if (name.find("1.") != std::string::npos) baseline = avg;

        std::cout << std::setw(25) << name
                  << std::setw(8) << avg << " ms";
        if (baseline > 0 && name.find("1.") == std::string::npos) {
            double ratio = static_cast<double>(baseline) / avg;
            std::cout << std::setw(10) << std::fixed
                      << std::setprecision(1) << ratio << "x";
        } else if (name.find("1.") != std::string::npos) {
            std::cout << std::setw(10) << "1.0x (base)";
        }
        std::cout << "\n";
    }

    std::cout << "\n结论:\n";
    std::cout << "  - PAUSE: 减少总线争用，功耗更低，性能通常优于无退避\n";
    std::cout << "  - Yield: 让出 CPU，低竞争时浪费，高竞争时有益\n";
    std::cout << "  - Exponential: 动态适应竞争水平，通用最佳选择\n";
    std::cout << "  - Randomized: 避免同步重试风暴，多线程下最稳定\n";
    std::cout << "  - 生产环境推荐: PAUSE + Exponential 组合\n";

    return 0;
}
