// 02_compare_exchange.cpp - CAS (Compare-And-Swap) 详解
// compare_exchange_strong vs compare_exchange_weak
// strong: 只在值不匹配时失败，更易用
// weak:  可能虚假失败（spurious failure），但某些平台更快，必须在循环中使用

#include <atomic>
#include <iostream>
#include <thread>
#include <vector>

// ===== 无锁计数器（使用 CAS 的 weak 版本） =====
class LockFreeCounter {
public:
    void increment() {
        int old_value = value_.load(std::memory_order_relaxed);
        int new_value;
        // CAS 循环：反复尝试直到成功
        do {
            new_value = old_value + 1;
            // weak 版本可能虚假失败，所以必须在循环中
        } while (!value_.compare_exchange_weak(
            old_value, new_value,
            std::memory_order_release,
            std::memory_order_relaxed));
    }

    int value() const {
        return value_.load(std::memory_order_acquire);
    }

private:
    std::atomic<int> value_{0};
};

// ===== 无锁最大值更新 =====
class AtomicMaximum {
public:
    // 原子地将 value 更新为 max(current, value)
    void update_max(int value) {
        int old = max_.load(std::memory_order_relaxed);
        // 如果 value <= 当前最大值，提前退出
        while (value > old) {
            // 尝试 CAS 更新，失败则重读旧值重试
            if (max_.compare_exchange_weak(old, value,
                    std::memory_order_relaxed,
                    std::memory_order_relaxed)) {
                break;
            }
        }
    }

    int value() const { return max_.load(std::memory_order_relaxed); }

private:
    std::atomic<int> max_{0};
};

// ===== 使用 strong 版本的单次赋值 =====
class OnceFlag {
public:
    // 仅当 flag 为 false 时执行 callable，并设置 flag = true
    // 确保 callable 只被调用一次
    template <typename F>
    bool call_once(F&& f) {
        bool expected = false;
        // strong 保证不会虚假失败，使用简单
        if (flag_.compare_exchange_strong(expected, true,
                std::memory_order_acq_rel,
                std::memory_order_relaxed)) {
            f();
            return true;
        }
        return false;
    }

    bool is_set() const { return flag_.load(std::memory_order_acquire); }

private:
    std::atomic<bool> flag_{false};
};

int main() {
    // ===== 测试 LockFreeCounter =====
    {
        std::cout << "=== LockFreeCounter (CAS weak) ===\n";
        LockFreeCounter counter;
        const int       kThreads = 8;
        const int       kIters   = 100000;

        std::vector<std::jthread> threads;
        for (int i = 0; i < kThreads; ++i) {
            threads.emplace_back([&]() {
                for (int j = 0; j < kIters; ++j) {
                    counter.increment();
                }
            });
        }
        threads.clear();

        std::cout << "  计数器: " << counter.value()
                  << " (期望 " << kThreads * kIters << ")\n\n";
    }

    // ===== 测试 AtomicMaximum =====
    {
        std::cout << "=== AtomicMaximum ===\n";
        AtomicMaximum max_monitor;
        const int     kThreads = 4;

        std::vector<std::jthread> threads;
        for (int i = 0; i < kThreads; ++i) {
            threads.emplace_back([&, i]() {
                for (int v = 1; v <= 100; ++v) {
                    max_monitor.update_max(i * 100 + v);
                }
            });
        }
        threads.clear();

        std::cout << "  全局最大值: " << max_monitor.value()
                  << " (期望 400)\n\n";
    }

    // ===== 测试 OnceFlag =====
    {
        std::cout << "=== OnceFlag (CAS strong) ===\n";
        OnceFlag once;
        int      call_count = 0;

        std::vector<std::jthread> threads;
        for (int i = 0; i < 10; ++i) {
            threads.emplace_back([&]() {
                once.call_once([&]() { ++call_count; });
            });
        }
        threads.clear();

        std::cout << "  callable 被调用次数: " << call_count
                  << " (期望 1)\n";
    }

    return 0;
}
