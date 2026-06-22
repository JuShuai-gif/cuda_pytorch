// 03_memory_fence.cpp — atomic_thread_fence 详解
// 演示: acquire fence / release fence / 与原子操作对比

#include <atomic>
#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

// ===== 1. 释放-获取栅栏: 保护非原子数据 =====
void demo_release_acquire_fence() {
    std::cout << "=== 1. release/acquire fence ===\n";

    std::atomic<bool> ready{false};
    int data = 0; // 非原子数据
    int result = -1;

    std::jthread producer([&]() {
        data = 42;
        // release fence: 保证 data=42 在 ready=true 之前对所有线程可见
        std::atomic_thread_fence(std::memory_order_release);
        ready.store(true, std::memory_order_relaxed);
    });

    std::jthread consumer([&]() {
        while (!ready.load(std::memory_order_relaxed))
            ;
        // acquire fence: 保证之后读取 data 能看到 release 前的写入
        std::atomic_thread_fence(std::memory_order_acquire);
        result = data;
    });

    producer.join();
    consumer.join();

    std::cout << "  data = " << data << ", result = " << result;
    if (result == 42) {
        std::cout << " (正确: acquire fence 看到了 release fence 前的写入)\n";
    } else {
        std::cout << " (意外: 可能编译器/CPU 重排)\n";
    }
}

// ===== 2. fence vs 原子操作内存序 =====
void demo_fence_vs_atomic_order() {
    std::cout << "\n=== 2. fence vs 原子操作内存序 ===\n";

    const int kIters = 100;
    std::atomic<int> sync{0};
    int data = 0;

    // 方案 A: 使用 fence
    {
        int errors = 0;
        for (int i = 0; i < kIters; ++i) {
            sync.store(0, std::memory_order_relaxed);
            data = 0;

            std::jthread t1([&]() {
                data = 1;
                std::atomic_thread_fence(std::memory_order_release);
                sync.store(1, std::memory_order_relaxed);
            });

            std::jthread t2([&]() {
                while (sync.load(std::memory_order_relaxed) != 1)
                    ;
                std::atomic_thread_fence(std::memory_order_acquire);
                if (data != 1) ++errors;
            });
            t1.join();
            t2.join();
        }
        std::cout << "  fence 方案: " << errors << "/" << kIters
                  << " 错误\n";
    }

    // 方案 B: 使用 release/acquire 原子操作
    {
        int errors = 0;
        std::atomic<int> flag{0};
        for (int i = 0; i < kIters; ++i) {
            flag.store(0, std::memory_order_relaxed);
            data = 0;

            std::jthread t1([&]() {
                data = 1;
                flag.store(1, std::memory_order_release);
            });

            std::jthread t2([&]() {
                while (flag.load(std::memory_order_acquire) != 1)
                    ;
                if (data != 1) ++errors;
            });
            t1.join();
            t2.join();
        }
        std::cout << "  原子序方案: " << errors << "/" << kIters
                  << " 错误\n";
    }
}

// ===== 3. seq_cst fence 的全局排序 =====
void demo_seq_cst_fence() {
    std::cout << "\n=== 3. seq_cst fence ===\n";

    std::atomic<int> x{0};
    std::atomic<int> y{0};
    std::atomic<int> reorder_detected{0};

    const int kIters = 10000;

    for (int i = 0; i < kIters; ++i) {
        x.store(0, std::memory_order_relaxed);
        y.store(0, std::memory_order_relaxed);

        std::jthread t1([&]() {
            x.store(1, std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            int r1 = y.load(std::memory_order_relaxed);
            if (r1 == 0) {
                // x=1 先于 y=1, 正常
            }
        });

        std::jthread t2([&]() {
            y.store(1, std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            int r2 = x.load(std::memory_order_relaxed);
            if (r2 == 0) {
                // y=1 先于 x=1, 正常
            }
        });

        t1.join();
        t2.join();
    }

    std::cout << "  seq_cst fence 保证了全局顺序一致性\n";
    std::cout << "  在 seq_cst fence 下，所有线程观测到的操作顺序一致\n";
}

// ===== 4. fence 的实际使用场景: 双检锁的 corrected =====
void demo_double_checked_locking() {
    std::cout << "\n=== 4. fence 在双检锁中的应用 ===\n";

    // 双检锁的正确实现需要 fence 或 atomic
    std::atomic<int*> instance{nullptr};
    std::mutex mtx;

    auto get_instance = [&]() -> int* {
        int* tmp = instance.load(std::memory_order_acquire);
        if (tmp == nullptr) {
            std::lock_guard lock(mtx);
            tmp = instance.load(std::memory_order_relaxed);
            if (tmp == nullptr) {
                tmp = new int(42);
                // release fence 确保 new int(42) 在 store 之前完成
                std::atomic_thread_fence(std::memory_order_release);
                instance.store(tmp, std::memory_order_relaxed);
            }
        }
        return tmp;
    };

    std::vector<std::jthread> threads;
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([&]() {
            int* p = get_instance();
            std::cout << "  线程获取: " << *p << "\n";
        });
    }
    threads.clear();

    delete instance.load();
    std::cout << "  fence 确保对象构造在指针发布之前完成\n";
}

int main() {
    demo_release_acquire_fence();
    demo_fence_vs_atomic_order();
    demo_seq_cst_fence();
    demo_double_checked_locking();

    std::cout << "\nfence 提供粗粒度内存顺序控制，适合保护非原子访问。\n";
    return 0;
}
