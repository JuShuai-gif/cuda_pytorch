// 02_thread_guard.cpp
// 知识点: RAII 线程管理 - thread_guard 类
// 演示: 使用 RAII 保护线程，确保在异常发生时也能正确 join

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// =============================================================================
// thread_guard: RAII 包装 std::thread
// 确保线程在作用域退出或异常发生时被 join
// 这是书中 2.1.2 节的经典实现
// =============================================================================
class ThreadGuard {
public:
    // 接受 std::thread 的右值引用，接管所有权
    explicit ThreadGuard(std::thread& t) : m_thread(t) {}

    // 禁止拷贝
    ThreadGuard(const ThreadGuard&)            = delete;
    ThreadGuard& operator=(const ThreadGuard&) = delete;

    // 析构时自动 join
    ~ThreadGuard() {
        if (m_thread.joinable()) {
            std::cout << "[ThreadGuard] 析构: 正在 join 线程...\n";
            m_thread.join();
            std::cout << "[ThreadGuard] 析构: join 完成\n";
        }
    }

private:
    std::thread& m_thread;
};

// =============================================================================
// 演示函数
// =============================================================================

// 一个可能抛出异常的函数
void fct_risky_operation(int id) {
    std::cout << "[线程 " << id << "] 开始执行\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    if (id == 2) {
        // 模拟异常情况
        throw std::runtime_error("[线程 " + std::to_string(id) +
                                 "] 发生异常!");
    }

    std::cout << "[线程 " << id << "] 正常完成\n";
}

// 不使用 ThreadGuard 的危险代码 (演示问题)
void fct_without_guard() {
    std::cout << "\n--- 不使用 ThreadGuard 的情况 ---\n";
    std::cout << "此代码演示: 异常导致线程未被 join\n";

    // 注释掉的危险代码:
    // std::thread t(fct_risky_operation, 1);
    // fct_risky_operation(2);  // 可能抛出异常
    // t.join();  // 如果上面抛异常，这行永远不会执行
    // 结果: std::terminate() 被调用!

    std::cout << "(危险代码已被注释，请参阅 02_thread_guard.cpp 源码)\n";
}

// 使用 ThreadGuard 的安全版本
void fct_with_guard() {
    std::cout << "\n--- 使用 ThreadGuard 的情况 ---\n";

    std::thread t([]() {
        std::cout << "[后台线程] 正在执行...\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        std::cout << "[后台线程] 完成\n";
    });

    ThreadGuard guard(t);  // RAII: 接管线程

    // 即使这里抛出异常，guard 析构时也会 join
    std::cout << "[主线程] 可能在这里抛异常...\n";

    try {
        throw std::runtime_error("主线程模拟异常!");
    } catch (const std::exception& e) {
        std::cout << "[主线程] 捕获异常: " << e.what() << "\n";
        // 抛出异常 → 栈展开 → guard 析构 → t.join()
        // 线程安全地被回收!
    }

    // guard 在作用域结束时析构
}

// 多个线程的 ThreadGuard 变体
class MultiGuard {
public:
    explicit MultiGuard(std::vector<std::thread>& threads)
        : m_threads(threads) {}

    ~MultiGuard() {
        for (auto& t : m_threads) {
            if (t.joinable()) {
                t.join();
            }
        }
        std::cout << "[MultiGuard] 所有线程已 join\n";
    }

    MultiGuard(const MultiGuard&)            = delete;
    MultiGuard& operator=(const MultiGuard&) = delete;

private:
    std::vector<std::thread>& m_threads;
};

int main() {
    std::cout << "=== ThreadGuard: RAII 线程管理 ===\n";

    // 演示问题场景
    fct_without_guard();

    // 演示 ThreadGuard 解决方案
    fct_with_guard();

    // 演示 MultiGuard: 管理线程池
    std::cout << "\n--- 使用 MultiGuard 管理多个线程 ---\n";
    {
        std::vector<std::thread> threads;

        for (int i = 0; i < 4; ++i) {
            threads.emplace_back([i]() {
                std::cout << "[线程 " << i
                          << "] ID=" << std::this_thread::get_id() << "\n";
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            });
        }

        MultiGuard guard(threads);
        // 正常退出或异常退出时，guard 都会 join 所有线程
    }

    std::cout << "\n=== ThreadGuard 设计要点 ===\n";
    std::cout << "1. 持有 std::thread& 引用，不接管所有权\n";
    std::cout << "2. 析构函数检查 joinable() 并调用 join()\n";
    std::cout << "3. 禁止拷贝，防止重复管理\n";
    std::cout << "4. C++20 的 std::jthread 内置了此功能\n";

    return 0;
}
