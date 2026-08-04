// 01_jthread_basic.cpp — std::jthread 基础用法
// 对比 std::thread vs std::jthread: 自动 join、内置 stop_token

#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

// ===== 1. std::thread 的典型陷阱 =====
void demo_thread_danger() {
    std::cout << "=== 1. std::thread 陷阱 ===\n";
    // 以下代码如果取消注释会导致 std::terminate:
    // { std::thread t([](){ std::this_thread::sleep_for(100ms); }); }
    // 错误: 线程在析构时仍 joinable

    // 正确做法: 显式 join 或 detach
    std::thread t([]() {
        std::this_thread::sleep_for(50ms);
    });
    t.join(); // 必须显式 join
    std::cout << "  std::thread: 手动 join 完成\n";
}

// ===== 2. std::jthread 自动 join =====
void demo_jthread_auto_join() {
    std::cout << "\n=== 2. jthread 自动 join ===\n";
    int counter = 0;
    {
        std::jthread t([&counter]() {
            for (int i = 0; i < 5; ++i) {
                std::this_thread::sleep_for(10ms);
                ++counter;
            }
        });
        std::cout << "  jthread 创建，即将离开作用域...\n";
    } // 自动 join，阻塞直到线程完成
    std::cout << "  jthread 析构完成，counter = " << counter
              << " (期望 5)\n";
}

// ===== 3. jthread 内置 stop_token =====
void demo_jthread_stop_token() {
    std::cout << "\n=== 3. jthread 内置 stop_token ===\n";

    std::jthread worker([](std::stop_token stoken) {
        int count = 0;
        while (!stoken.stop_requested()) {
            std::this_thread::sleep_for(20ms);
            ++count;
        }
        std::cout << "  Worker 收到停止信号，共迭代 " << count << " 次\n";
    });

    std::this_thread::sleep_for(100ms);
    // jthread 的析构或 request_stop() 会触发停止
    worker.request_stop();
    worker.join();
    std::cout << "  Worker 已停止\n";
}

// ===== 4. 多 jthread 管理 =====
void demo_multiple_jthreads() {
    std::cout << "\n=== 4. 多 jthread ===\n";
    const int kNumThreads = 5;

    std::vector<std::jthread> threads;
    for (int i = 0; i < kNumThreads; ++i) {
        threads.emplace_back([i](std::stop_token stoken) {
            std::cout << "    线程 " << i << " 启动\n";
            std::this_thread::sleep_for(30ms * (i + 1));
            std::cout << "    线程 " << i << " 完成\n";
        });
    }
    // vector 析构时自动 join 所有 jthread
    std::cout << "  等待所有线程完成...\n";
}

int main() {
    demo_thread_danger();
    demo_jthread_auto_join();
    demo_jthread_stop_token();
    demo_multiple_jthreads();

    std::cout << "\n所有演示完成。jthread 比 thread 更安全、更简洁！\n";
    return 0;
}
