// 07_lockfree_counter.cpp - 工业级无锁原子计数器
// 支持多线程高并发递增/递减
// 实现方式：
//   1. 简单 CAS 版本（适合低竞争）
//   2. 拆分计数器（per-thread slots，减少缓存行竞争）

#include <array>
#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

// ===== 1. 简单 CAS 无锁计数器 =====
// 优点：实现简单
// 缺点：高竞争下 CAS 重试多，性能下降
class SimpleLockFreeCounter {
public:
    void add(long long delta) {
        long long old = value_.load(std::memory_order_relaxed);
        long long desired;
        do {
            desired = old + delta;
        } while (!value_.compare_exchange_weak(
            old, desired,
            std::memory_order_release,
            std::memory_order_relaxed));
    }

    long long value() const {
        return value_.load(std::memory_order_acquire);
    }

    void reset() {
        value_.store(0, std::memory_order_relaxed);
    }

private:
    std::atomic<long long> value_{0};
};

// ===== 2. 拆分计数器（减少竞争） =====
// 每个线程拥有独立计数器槽，最终读取时求和
// 使用 cache-line padding 防止 false sharing

static constexpr size_t kCacheLineSize = 64; // 典型缓存行大小

// 对齐到缓存行大小，防止相邻槽位共享缓存行
struct alignas(kCacheLineSize) CounterSlot {
    std::atomic<long long> value{0};
    // padding 确保结构体大小正好为 kCacheLineSize
    char padding[kCacheLineSize - sizeof(std::atomic<long long>)];
};

template <size_t NumSlots = 16>
class StripedCounter {
public:
    void add(size_t slot, long long delta) {
        slots_[slot % NumSlots].value.fetch_add(delta, std::memory_order_relaxed);
    }

    void inc(size_t slot) { add(slot, 1); }

    long long value() const {
        long long sum = 0;
        for (size_t i = 0; i < NumSlots; ++i) {
            sum += slots_[i].value.load(std::memory_order_acquire);
        }
        return sum;
    }

private:
    std::array<CounterSlot, NumSlots> slots_;
};

// ===== 3. 近似计数器（统计计数，牺牲精确性换取性能） =====
class ApproximateCounter {
public:
    void inc() { add(1); }

    void add(long long delta) {
        // relaxed: 仅计数，不需要与其他操作同步
        value_.fetch_add(delta, std::memory_order_relaxed);
    }

    long long value() const {
        // 近似值（读取瞬间可能正在被修改）
        return value_.load(std::memory_order_relaxed);
    }

private:
    std::atomic<long long> value_{0};
};

// ===== 性能测试 =====
template <typename Counter>
void benchmark_counter(const std::string& name,
                       Counter& counter,
                       int num_threads,
                       int iters_per_thread,
                       bool use_slot = false) {
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::jthread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, slot = i]() {
            for (int j = 0; j < iters_per_thread; ++j) {
                if constexpr (std::is_same_v<Counter, StripedCounter<>>) {
                    counter.inc(static_cast<size_t>(slot));
                } else {
                    counter.add(1);
                }
            }
        });
    }
    threads.clear();

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start);

    long long actual = counter.value();

    std::cout << "  " << name << ": " << elapsed.count() << "ms"
              << " (值=" << actual
              << ", 期望=" << num_threads * iters_per_thread << ")\n";
}

int main() {
    const int kThreads = 8;
    const int kIters   = 200000;

    std::cout << "=== 无锁计数器性能对比（" << kThreads
              << " 线程 × " << kIters << " 次递增） ===\n\n";

    {
        SimpleLockFreeCounter counter;
        benchmark_counter("SimpleLockFreeCounter  ", counter, kThreads, kIters);
    }

    {
        // Split counter with 16 slots
        StripedCounter<16> counter;
        benchmark_counter("StripedCounter (16 slots)", counter, kThreads, kIters);
    }

    {
        ApproximateCounter counter;
        benchmark_counter("ApproximateCounter     ", counter, kThreads, kIters);
    }

    // ===== 演示缓存行对齐的重要性 =====
    {
        std::cout << "\n=== 缓存行对齐验证 ===\n";
        std::cout << "  sizeof(CounterSlot) = " << sizeof(CounterSlot)
                  << " (期望 == " << kCacheLineSize << ")\n";

        StripedCounter<4> sc;
        std::cout << "  sizeof(StripedCounter<4>) = "
                  << sizeof(sc) << "\n";
    }

    // ===== 演示多计数器类型同时运行 =====
    {
        std::cout << "\n=== 多类型同时递增 ===\n";
        SimpleLockFreeCounter simple_counter;
        StripedCounter<8>     striped_counter;

        std::vector<std::jthread> threads;
        for (int i = 0; i < 4; ++i) {
            threads.emplace_back([&, slot = i]() {
                for (int j = 0; j < 100000; ++j) {
                    simple_counter.add(1);
                    striped_counter.inc(static_cast<size_t>(slot));
                }
            });
        }
        threads.clear();

        std::cout << "  简单计数器: " << simple_counter.value()
                  << " (期望 400000)\n";
        std::cout << "  拆分计数器: " << striped_counter.value()
                  << " (期望 400000)\n";
    }

    return 0;
}
