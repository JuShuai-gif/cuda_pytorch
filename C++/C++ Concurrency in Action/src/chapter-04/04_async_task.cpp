// 04_async_task.cpp - std::async 异步任务
// 演示 launch::async（强制新线程） vs launch::deferred（延迟调用）
// 以及异常在 future 中的传播

#include <chrono>
#include <exception>
#include <future>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

// 计算 1 到 n 的和（模拟耗时计算）
long long sum_range(long long n) {
    std::cout << "[Task] 线程ID=" << std::this_thread::get_id()
              << " 计算 1.." << n << " 的和\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    return n * (n + 1) / 2; // 高斯公式
}

// 可能抛出异常的任务
int risky_task(int input) {
    if (input < 0) throw std::invalid_argument("输入不能为负数");
    return input * 2;
}

int main() {
    // --- 1. std::async 默认策略 (launch::async | launch::deferred) ---
    {
        std::cout << "=== 默认策略 ===\n";
        auto future = std::async(std::launch::async, sum_range, 1000000);

        std::cout << "[Main] 异步任务已提交\n";
        // 做一些其他工作...
        std::cout << "[Main] 等待结果...\n";
        long long result = future.get();
        std::cout << "[Main] 结果: " << result << "\n\n";
    }

    // --- 2. launch::deferred: 延迟执行，直到调用 get() ---
    {
        std::cout << "=== deferred 延迟策略 ===\n";
        auto future = std::async(std::launch::deferred, sum_range, 100);

        std::cout << "[Main] 任务已创建但尚未执行\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        std::cout << "[Main] 现在调用 get()...\n";
        long long result = future.get(); // 此处才真正执行任务
        std::cout << "[Main] 结果: " << result << "\n\n";
    }

    // --- 3. 异常传播 ---
    {
        std::cout << "=== 异常传播 ===\n";
        auto future_ok  = std::async(std::launch::async, risky_task, 5);
        auto future_bad = std::async(std::launch::async, risky_task, -1);

        try {
            std::cout << "[Main] 正常结果: " << future_ok.get() << "\n";
        } catch (const std::exception& e) {
            std::cout << "不该出现: " << e.what() << "\n";
        }

        try {
            future_bad.get();
        } catch (const std::exception& e) {
            std::cout << "[Main] 捕获异步异常: " << e.what() << "\n";
        }
    }

    // --- 4. 多个异步任务并发执行 ---
    {
        std::cout << "\n=== 多任务并发 ===\n";
        std::vector<std::future<long long>> futures;
        for (int i = 1; i <= 5; ++i) {
            futures.push_back(std::async(std::launch::async, sum_range, i * 1000000));
        }

        for (auto& f : futures) {
            std::cout << "结果: " << f.get() << "\n";
        }
    }

    return 0;
}
