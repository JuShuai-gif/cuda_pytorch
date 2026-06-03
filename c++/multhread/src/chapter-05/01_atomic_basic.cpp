// 01_atomic_basic.cpp - std::atomic 基本用法
// 演示: atomic_flag 自旋锁, atomic<bool> 标志, atomic<int> 计数器

#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

// ===== 1. atomic_flag 自旋锁（最简单的无锁同步）=====
class SpinLock {
public:
    void lock() {
        // test_and_set: 原子地将标志设为 true 并返回旧值
        // 循环直到成功获取（旧值为 false）
        while (flag_.test_and_set(std::memory_order_acquire)) {
            // 主动让出 CPU，避免忙等浪费（但仍是自旋）
            // 在 x86 上可配合 PAUSE 指令降低功耗
        }
    }

    void unlock() {
        flag_.clear(std::memory_order_release);
    }

private:
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
};

// ===== 2. atomic<bool> 用作开关标志 =====
class StopFlag {
public:
    void request_stop() { stop_.store(true, std::memory_order_relaxed); }
    bool is_stopped() const { return stop_.load(std::memory_order_relaxed); }

private:
    std::atomic<bool> stop_{false};
};

// ===== 3. atomic<int> 原子计数器 =====
class AtomicCounter {
public:
    void increment() { counter_.fetch_add(1, std::memory_order_relaxed); }
    int  value() const { return counter_.load(std::memory_order_relaxed); }
    void reset() { counter_.store(0, std::memory_order_relaxed); }

private:
    std::atomic<int> counter_{0};
};

// ===== 测试代码 =====

void test_spinlock() {
    std::cout << "=== atomic_flag 自旋锁 ===\n";
    SpinLock           spinlock;
    int                shared_counter = 0;
    const int          kNumThreads = 4;
    const int          kIters      = 100000;

    std::vector<std::jthread> threads;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < kNumThreads; ++i) {
        threads.emplace_back([&]() {
            for (int j = 0; j < kIters; ++j) {
                std::lock_guard<SpinLock> lock(spinlock);
                ++shared_counter;
            }
        });
    }
    threads.clear(); // jthread 自动 join

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start);

    std::cout << "  计数器: " << shared_counter
              << " (期望 " << kNumThreads * kIters << ")"
              << " | 耗时: " << elapsed.count() << "ms\n\n";
}

void test_stop_flag() {
    std::cout << "=== atomic<bool> 停止标志 ===\n";
    StopFlag stop_flag;

    // 工作线程：持续运行直到被要求停止
    std::jthread worker([&]() {
        int count = 0;
        while (!stop_flag.is_stopped()) {
            std::this_thread::sleep_for(10ms);
            ++count;
        }
        std::cout << "  [Worker] 被停止，执行了 " << count << " 次迭代\n";
    });

    // 主线程：500ms 后请求停止
    std::this_thread::sleep_for(200ms);
    stop_flag.request_stop();
    std::cout << "  [Main] 已请求停止\n\n";
}

void test_atomic_counter() {
    std::cout << "=== atomic<int> 原子计数器 ===\n";
    AtomicCounter       counter;
    const int           kNumThreads = 8;
    const int           kIters      = 500000;

    std::vector<std::jthread> threads;
    for (int i = 0; i < kNumThreads; ++i) {
        threads.emplace_back([&]() {
            for (int j = 0; j < kIters; ++j) {
                counter.increment();
            }
        });
    }
    threads.clear();

    std::cout << "  计数器: " << counter.value()
              << " (期望 " << kNumThreads * kIters << ")\n\n";
}

int main() {
    test_spinlock();
    test_stop_flag();
    test_atomic_counter();
    return 0;
}
