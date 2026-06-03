// 03_future_promise.cpp - std::future + std::promise 基本用法
// promise 在线程 A 中 set_value, future 在线程 B 中 get

#include <chrono>
#include <future>
#include <iostream>
#include <thread>

// 模拟耗时计算
int compute_heavy_value(int input) {
    std::cout << "[Worker] 开始计算... (输入=" << input << ")\n";
    std::this_thread::sleep_for(std::chrono::seconds(1));
    int result = input * input;
    std::cout << "[Worker] 计算完成: " << result << "\n";
    return result;
}

// 演示 promise/future 的异常传播
void demo_exception() {
    std::promise<int> promise;
    std::future<int>  future = promise.get_future();

    std::jthread worker([&promise]() {
        try {
            // 模拟工作线程抛出异常
            throw std::runtime_error("工作线程内部错误");
            promise.set_value(42); // 不会执行
        } catch (...) {
            // 通过 promise 传播异常到 future
            promise.set_exception(std::current_exception());
        }
    });

    try {
        int result = future.get(); // 此处抛出异常
        std::cout << "结果: " << result << "\n";
    } catch (const std::exception& e) {
        std::cout << "[Main] 捕获异常: " << e.what() << "\n";
    }
}

int main() {
    // --- 基本用法：promise 设置值，future 获取值 ---
    {
        std::promise<int> promise;
        std::future<int>  future = promise.get_future();

        std::jthread worker([&promise]() {
            int result = compute_heavy_value(10);
            promise.set_value(result); // 设置结果
        });

        // future::get() 阻塞等待，只能调用一次
        std::cout << "[Main] 等待结果...\n";
        int value = future.get();
        std::cout << "[Main] 获得结果: " << value << "\n\n";
    }

    // --- 异常传播 ---
    demo_exception();

    return 0;
}
