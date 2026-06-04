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
    // std::promise<T> —— 生产者（可以"承诺"将来会给出一个 T 类型的值或异常）
    // 在它被 set_value/set_exception 之前，消费者调用 get() 会一直阻塞等待
    std::promise<int> promise;

    // promise.get_future() —— 获取与这个 promise 绑定的 future 句柄
    // 一个 promise 只能生成一个 future，调用多次会抛 std::future_error
    // future 是唯一消费者：get() 只能调一次，第二次调用同样抛异常
    std::future<int> future = promise.get_future();

    std::jthread worker([&promise]() {
        try {
            // 模拟工作线程抛出异常
            throw std::runtime_error("工作线程内部错误");
            promise.set_value(42); // 不会执行 — 上面已经抛异常了
        } catch (...) {
            // 通过 promise 把异常传递给 future（不是抛到线程外部）
            // std::current_exception() 捕获当前异常对象
            promise.set_exception(std::current_exception());
        }
    });

    try {
        // future.get() 发现 promise 传过来的是异常，会在此处重新抛出
        int result = future.get();
        std::cout << "结果: " << result << "\n";
    } catch (const std::exception& e) {
        // 消费者捕获到从工作线程传播过来的异常
        std::cout << "[Main] 捕获异常: " << e.what() << "\n";
    }
}

int main() {
    // --- 基本用法：promise 设置值，future 获取值 ---
    {
        // 创建 promise/future 对 —— 一个生产者一个消费者
        std::promise<int> promise;
        std::future<int>  future = promise.get_future();

        // 工作线程（生产者）：做完计算后通过 promise.set_value() 告知结果
        std::jthread worker([&promise]() {
            int result = compute_heavy_value(10);
            promise.set_value(result); // 向 future 管道写入结果
        });

        // 主线程（消费者）：调用 future.get() 阻塞等待直到 promise 端写入
        // 如果 promise 永远不 set，get() 会永远阻塞（没有超时机制）
        // get() 只能调用一次，这是 C++ future 的设计约束
        std::cout << "[Main] 等待结果...\n";
        int value = future.get();
        std::cout << "[Main] 获得结果: " << value << "\n\n";
    }

    // --- 异常传播 ---
    demo_exception();

    return 0;
}
