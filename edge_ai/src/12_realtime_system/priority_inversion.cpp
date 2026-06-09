#include "priority_inversion.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>

void demo_priority_inversion() {
    std::cout << "\n"
              << std::string(70, '=') << "\n";
    std::cout << "  优先级反转演示\n";
    std::cout << std::string(70, '=') << "\n";

    std::cout << "\n经典优先级反转场景：\n";
    std::cout << "  - 任务 H（高优先级，prio=1）：需要共享锁\n";
    std::cout << "  - 任务 M（中优先级，prio=2）：不需要锁，CPU 密集型\n";
    std::cout << "  - 任务 L（低优先级，prio=3）：持有共享锁\n\n";

    SharedResource res;
    std::atomic<bool> task_m_running{true};
    std::atomic<int64_t> h_start_time{0};
    std::atomic<int64_t> h_end_time{0};
    std::atomic<int64_t> h_block_duration{0};
    std::atomic<int64_t> m_total_work{0};

    // 准备测试
    {
        res.mtx.lock();
        res.locked_by_prio = 3; // 低优先级任务 L "持有"锁
    }

    // 启动任务 H（最高优先级，需要锁）
    std::thread task_h([&]() {
        // 模拟：在 L 获取锁之后稍晚到达
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        int64_t t0 = std::chrono::duration_cast<std::chrono::microseconds>(
                         std::chrono::high_resolution_clock::now().time_since_epoch())
                         .count();
        h_start_time = t0;

        std::cout << "[任务 H] 正在尝试获取锁...\n";
        res.mtx.lock();
        int64_t t1 = std::chrono::duration_cast<std::chrono::microseconds>(
                         std::chrono::high_resolution_clock::now().time_since_epoch())
                         .count();
        h_block_duration = t1 - t0;
        std::cout << "[任务 H] 锁已获取！阻塞了 " << h_block_duration / 1000.0
                  << " ms\n";
        res.mtx.unlock();
        h_end_time = std::chrono::duration_cast<std::chrono::microseconds>(
                         std::chrono::high_resolution_clock::now().time_since_epoch())
                         .count();
    });

    // 启动任务 M（中优先级，不需要锁，密集运行）
    std::thread task_m([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
        std::cout << "[任务 M] 正在运行中优先级工作（不需要锁）...\n";
        auto start = std::chrono::high_resolution_clock::now();
        volatile double dummy = 0.0;
        while (task_m_running) {
            for (int i = 0; i < 10000; i++) {
                dummy += std::sin(static_cast<double>(i));
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        m_total_work = std::chrono::duration_cast<std::chrono::microseconds>(
                           end - start)
                           .count();
        (void)dummy;
    });

    // 任务 L（低优先级，持有锁）运行一段时间
    std::cout << "[任务 L] 持有锁，正在执行工作...\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    std::cout << "[任务 L] 释放锁。\n";
    res.mtx.unlock();

    task_m_running = false;
    task_h.join();
    task_m.join();

    int64_t h_total = h_end_time - h_start_time;
    std::cout << "\n--- 优先级反转结果 ---\n";
    std::cout << "  任务 H 阻塞时间：" << h_block_duration / 1000.0 << " ms\n";
    std::cout << "  任务 H 总耗时：  " << h_total / 1000.0 << " ms\n";
    std::cout << "  任务 M 运行时长：" << m_total_work / 1000.0 << " ms\n";
    std::cout << "\n  问题：任务 H（最高优先级）被任务 M（中优先级）延迟了，\n"
              << "  这绝对不应该发生！\n"
              << "  任务 H 等待 L 的锁，但 L 无法运行，因为 M 抢占了它。\n";
}

void demo_priority_inheritance() {
    std::cout << "\n"
              << std::string(70, '=') << "\n";
    std::cout << "  优先级继承解决方案演示\n";
    std::cout << std::string(70, '=') << "\n";

    std::cout << "\n使用优先级继承：\n";
    std::cout << "  - 任务 H 在任务 L 持有的锁上阻塞\n";
    std::cout << "  - 任务 L 继承任务 H 的优先级\n";
    std::cout << "  - 任务 M 无法抢占任务 L（L 现在拥有高优先级）\n";
    std::cout << "  - 任务 L 快速完成，释放锁，H 继续执行\n\n";

    PriorityInheritanceMutex pi_mutex;
    std::atomic<bool> task_m_running{true};
    std::atomic<int64_t> h_block_time{0};
    std::atomic<int64_t> m_starved{1}; // 应该很小

    // 任务 L 先获取锁
    pi_mutex.lock(3); // 优先级 3（低）

    // 任务 M（中优先级，CPU 消耗者）
    std::thread task_m([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
        auto start = std::chrono::high_resolution_clock::now();
        volatile double dummy = 0.0;
        while (task_m_running) {
            for (int i = 0; i < 10000; i++) {
                dummy += std::sin(static_cast<double>(i));
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        m_starved = std::chrono::duration_cast<std::chrono::microseconds>(
                        end - start)
                        .count();
        (void)dummy;
    });

    // 任务 H（高优先级）- 尝试获取锁
    std::thread task_h([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        int64_t t0 = std::chrono::duration_cast<std::chrono::microseconds>(
                         std::chrono::high_resolution_clock::now().time_since_epoch())
                         .count();
        std::cout << "[任务 H] 正在尝试获取锁（带继承）...\n";
        pi_mutex.lock(1); // 优先级 1 = 最高
        int64_t t1 = std::chrono::duration_cast<std::chrono::microseconds>(
                         std::chrono::high_resolution_clock::now().time_since_epoch())
                         .count();
        h_block_time = t1 - t0;
        std::cout << "[任务 H] 锁已获取！阻塞了 " << h_block_time / 1000.0
                  << " ms\n";
        pi_mutex.unlock();
    });

    // 任务 L 持有锁但以"继承"的优先级运行
    std::cout << "[任务 L] 持有锁（通过继承提升了优先级）...\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(20)); // 短临界区
    std::cout << "[任务 L] 释放锁。\n";
    pi_mutex.unlock();

    task_m_running = false;
    task_h.join();
    task_m.join();

    std::cout << "\n--- 优先级继承结果 ---\n";
    std::cout << "  任务 H 阻塞时间：" << h_block_time / 1000.0 << " ms\n";
    std::cout << "  任务 M 干扰时长：" << m_starved / 1000.0 << " ms\n";
    std::cout << "\n  解决方案：任务 H 的阻塞时间现在受限于\n"
              << "  临界区长度，而不是任务 M 的无限工作量。\n";
}
