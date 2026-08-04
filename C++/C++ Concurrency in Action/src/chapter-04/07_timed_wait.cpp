// 07_timed_wait.cpp - 限时等待技术
// 演示：wait_for, wait_until, 带超时的条件变量等待, std::chrono 时间工具

#include <chrono>
#include <condition_variable>
#include <future>
#include <iostream>
#include <mutex>
#include <thread>

using namespace std::chrono_literals;

// 模拟一个可能缓慢的任务
int slow_computation(int ms_delay) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms_delay));
    return ms_delay * 2;
}

int main() {
    // ===== 1. future::wait_for 限时等待 =====
    {
        std::cout << "=== future::wait_for ===\n";
        auto future = std::async(std::launch::async, slow_computation, 500);

        std::cout << "[Main] 等待 200ms...\n";
        auto status = future.wait_for(200ms);
        if (status == std::future_status::timeout) {
            std::cout << "[Main] 超时！任务尚未完成\n";
        }

        std::cout << "[Main] 等待 500ms...\n";
        status = future.wait_for(500ms);
        if (status == std::future_status::ready) {
            std::cout << "[Main] 任务完成，结果: " << future.get() << "\n";
        }
        std::cout << "\n";
    }

    // ===== 2. future::wait_until 等待到指定时间点 =====
    {
        std::cout << "=== future::wait_until ===\n";
        auto future = std::async(std::launch::async, slow_computation, 300);

        auto deadline = std::chrono::steady_clock::now() + 150ms;
        auto status   = future.wait_until(deadline);

        if (status == std::future_status::timeout) {
            std::cout << "[Main] 在截止时间前未完成，继续等待...\n";
        }

        // 阻塞直到完成
        std::cout << "[Main] 最终结果: " << future.get() << "\n\n";
    }

    // ===== 3. 带超时的条件变量等待 =====
    {
        std::cout << "=== 条件变量 wait_for 超时 ===\n";
        std::mutex              mutex;
        std::condition_variable cv;
        bool                    ready = false;

        // 一个线程在 500ms 后设置 ready
        std::jthread setter([&]() {
            std::this_thread::sleep_for(500ms);
            {
                std::lock_guard<std::mutex> lock(mutex);
                ready = true;
            }
            cv.notify_one();
            std::cout << "[Setter] 就绪标志已设置\n";
        });

        // 主线程带超时等待（最多 200ms）
        {
            std::unique_lock<std::mutex> lock(mutex);
            bool success = cv.wait_for(lock, 200ms, [&] { return ready; });

            if (success) {
                std::cout << "[Main] 收到就绪通知\n";
            } else {
                std::cout << "[Main] 200ms 超时，就绪标志尚未设置\n";
            }
        }

        std::this_thread::sleep_for(400ms); // 等待 setter 完成
        std::cout << "\n";
    }

    // ===== 4. 条件变量 wait_until =====
    {
        std::cout << "=== 条件变量 wait_until ===\n";
        std::mutex              mutex;
        std::condition_variable cv;
        bool                    data_ready = false;

        std::jthread producer([&]() {
            std::this_thread::sleep_for(300ms);
            {
                std::lock_guard<std::mutex> lock(mutex);
                data_ready = true;
            }
            cv.notify_one();
            std::cout << "[Producer] 数据就绪\n";
        });

        {
            std::unique_lock<std::mutex> lock(mutex);
            auto deadline = std::chrono::steady_clock::now() + 500ms;

            if (cv.wait_until(lock, deadline, [&] { return data_ready; })) {
                std::cout << "[Main] 在截止时间前收到数据\n";
            } else {
                std::cout << "[Main] 等待超时\n";
            }
        }
    }

    // ===== 5. chrono 常用时间工具演示 =====
    {
        std::cout << "\n=== std::chrono 工具 ===\n";

        auto start = std::chrono::high_resolution_clock::now();

        std::this_thread::sleep_for(100ms);

        auto end   = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::cout << "耗时: " << elapsed.count() << " 微秒\n";

        // C++20 字面量（如果编译器支持）
        auto duration_s = std::chrono::seconds(3);
        auto duration_ms = std::chrono::milliseconds(1500);
        std::cout << "3秒 = " << std::chrono::duration_cast<std::chrono::milliseconds>(duration_s).count() << "ms\n";
        std::cout << "1500ms = " << std::chrono::duration_cast<std::chrono::seconds>(duration_ms).count() << "s\n";
    }

    return 0;
}
