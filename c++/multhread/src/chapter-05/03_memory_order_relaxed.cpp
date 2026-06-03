// 03_memory_order_relaxed.cpp - memory_order_relaxed 松散序
// 仅保证原子性，不保证顺序（无同步关系）
// 适用场景：简单计数器，无数据依赖的共享变量

#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

// 全局共享原子变量（松散序）
std::atomic<int> g_counter{0};
std::atomic<bool> g_done{false};

// ===== 计数器线程：仅递增 =====
void increment_worker(int iters) {
    for (int i = 0; i < iters; ++i) {
        // relaxed: 只关心计数原子性，不关心与其他变量的顺序
        g_counter.fetch_add(1, std::memory_order_relaxed);
    }
}

// ===== 演示：relaxed 不保证可见性顺序 =====
void demo_visibility() {
    std::cout << "=== relaxed 不保证跨线程顺序 ===\n";

    std::atomic<int> x{0};
    std::atomic<int> y{0};

    // 线程 A: 写入 x, 然后写入 y (都是 relaxed)
    std::jthread t1([&]() {
        x.store(42, std::memory_order_relaxed);
        y.store(99, std::memory_order_relaxed);
    });

    // 线程 B: 读取 y, 然后读取 x (都是 relaxed)
    std::jthread t2([&]() {
        int y_val = 0;
        int x_val = 0;

        // 等待 y 被写入
        while ((y_val = y.load(std::memory_order_relaxed)) == 0) {
            std::this_thread::yield();
        }

        // ⚠️ 即使 y 已经是 99，x 可能仍然是 0！
        // 因为 relaxed 不建立 happens-before 关系
        x_val = x.load(std::memory_order_relaxed);

        std::cout << "  y=" << y_val << ", x=" << x_val;
        if (x_val == 0) {
            std::cout << " (顺序重排！x 仍为初始值)";
        } else {
            std::cout << " (顺序正确)";
        }
        std::cout << "\n";
    });
}

// ===== 演示 stopping flag 的 relaxed 用法（正确场景）=====
void demo_stopping_flag() {
    std::cout << "\n=== stopping flag (relaxed 适用场景) ===\n";

    std::atomic<bool> stop{false};
    int                progress = 0; // 非原子变量

    std::jthread worker([&]() {
        while (!stop.load(std::memory_order_relaxed)) {
            // 不做任何依赖于其他变量的操作
            // 仅仅需要知道何时停止
            std::this_thread::sleep_for(10ms);
            ++progress;
        }
        std::cout << "  [Worker] 处理了 " << progress << " 次\n";
    });

    std::this_thread::sleep_for(200ms);
    stop.store(true, std::memory_order_relaxed);
    std::cout << "  [Main] 已发送停止信号\n";
}

int main() {
    // --- 1. 高并发计数器 (relaxed 经典场景) ---
    {
        std::cout << "=== 高并发 relaxed 计数器 ===\n";
        const int kThreads = 8;
        const int kIters   = 500000;

        std::vector<std::jthread> threads;
        for (int i = 0; i < kThreads; ++i) {
            threads.emplace_back(increment_worker, kIters);
        }
        threads.clear();

        std::cout << "  g_counter = " << g_counter.load(std::memory_order_relaxed)
                  << " (期望 " << kThreads * kIters << ")\n\n";
    }

    // --- 2. 可见性问题演示 ---
    demo_visibility();

    // --- 3. 正确的 relaxed 场景 ---
    demo_stopping_flag();

    return 0;
}
