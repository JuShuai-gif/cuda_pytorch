// 02_compare_exchange.cpp - CAS (Compare-And-Swap) 详解
// compare_exchange_strong vs compare_exchange_weak
// strong: 只在值不匹配时失败，更易用
// weak:  可能虚假失败（spurious failure），但某些平台更快，必须在循环中使用
//
// CAS 使用场景：
// 1. 无锁数据结构（LockFreeCounter）— 取代 mutex 实现并发安全的计数/入队/出队
// 2. 原子更新极值（AtomicMaximum）  — 多线程竞争更新最大/最小值
// 3. 一次性初始化（OnceFlag）       — 类似 std::call_once，确保只执行一次
// 4. 无锁栈/队列、RCU、共享指针引用计数等

#include <atomic>
#include <iostream>
#include <thread>
#include <vector>

// ===== 无锁计数器（使用 CAS 的 weak 版本） =====
// 场景：多个线程并发递增计数器，用 CAS 循环替代 mutex
class LockFreeCounter {
public:
    void increment() {
        // 用 relaxed 读取当前值：只需要原子性，不需要同步顺序
        int old_value = value_.load(std::memory_order_relaxed);
        int new_value;
        // CAS 循环：反复尝试直到成功
        do {
            new_value = old_value + 1;
            // compare_exchange_weak 语义：
            //   - 如果 value_ == old_value，则将 value_ 设为 new_value，返回 true
            //   - 如果 value_ != old_value，则将 old_value 更新为当前 value_，返回 false
            // weak 版本可能虚假失败（即使 value_ == old_value 也返回 false），
            // 必须放在 while 循环中。但它在某些平台（如 ARM）性能更优。
            //
            // memory_order_release：写屏障，保证本线程之前的写入对后续 acquire 可见
            // memory_order_relaxed：失败时只需原子读取，不需要同步开销
        } while (!value_.compare_exchange_weak(
            old_value, new_value,
            std::memory_order_release,
            std::memory_order_relaxed));
    }

    int value() const {
        // acquire 与 increment 中的 release 配对，确保读到最新值
        return value_.load(std::memory_order_acquire);
    }

private:
    std::atomic<int> value_{0};
};

// ===== 无锁最大值更新 =====
// 场景：多个线程各自产生候选值，需要原子地保留全局最大值
class AtomicMaximum {
public:
    // 原子地将 value 更新为 max(current, value)
    void update_max(int value) {
        int old = max_.load(std::memory_order_relaxed);
        // 如果 value <= 当前最大值，说明不需要更新，循环自动退出
        while (value > old) {
            // 尝试 CAS 更新，如果失败（说明其他线程已经写入更大的值），
            // CAS 会把 old 自动更新为最新的 max_，然后循环重新比较
            if (max_.compare_exchange_weak(old, value,
                    std::memory_order_relaxed,
                    std::memory_order_relaxed)) {
                break;  // CAS 成功，本线程的 value 写入为新的最大值
            }
            // CAS 失败：old 已被自动更新为 max_ 的当前值，继续 while 判断
        }
    }

    int value() const { return max_.load(std::memory_order_relaxed); }

private:
    std::atomic<int> max_{0};
};

// ===== 使用 strong 版本的单次赋值 =====
// 场景：确保某个函数在多线程环境下只被执行一次（类似 std::call_once / pthread_once）
class OnceFlag {
public:
    // 仅当 flag 为 false 时执行 callable，并设置 flag = true
    // 多个线程并发调用 call_once，只有第一个能成功执行 callable
    template <typename F>
    bool call_once(F&& f) {
        bool expected = false;
        // strong 保证不会虚假失败：只要 expected == flag_ 的当前值，必定成功
        // 不存在"值匹配但返回 false"的情况，因此无需循环
        //
        // memory_order_acq_rel：成功时同时具备 acquire+release 语义
        //   - acquire：确保 callable 在 flag_ 设置完之后对其他线程可见
        //   - release：确保 callable 执行发生在 flag_ 设置之前（不会被重排到之后）
        if (flag_.compare_exchange_strong(expected, true,
                std::memory_order_acq_rel,
                std::memory_order_relaxed)) {
            f();          // 只有成功设置 flag 的线程会执行这里
            return true;
        }
        return false;     // 其他线程 CAS 失败（expected != true），直接返回
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
        // jthread 析构时自动 join，确保所有线程完成
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
                // 线程 i 产生值范围 (i*100+1) ~ (i*100+100)
                // 线程 3 产生 301~400，期望最大值为 400
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

        // 10 个线程并发调用 call_once，但 callable 只执行 1 次
        std::cout << "  callable 被调用次数: " << call_count
                  << " (期望 1)\n";
    }

    return 0;
}
