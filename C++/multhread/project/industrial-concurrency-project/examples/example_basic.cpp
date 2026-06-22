// Ch2 & Ch4：基本任务提交示例
// 展示 ThreadPool 和 TaskQueue 最简单用法的最小示例。
// 演示：Ch2（线程管理）、Ch4.2（future/promise）、Ch6.2（队列）。

#include "task_scheduler/thread_pool.hpp"
#include "task_scheduler/task_queue.hpp"
#include "task_scheduler/logger.hpp"
#include <iostream>
#include <vector>
#include <cmath>

using namespace task_scheduler;

int main() {
    Logger::instance().info("=== 示例：基本任务提交 ===");

    // Ch9.1：创建 4 个工作线程的线程池。
    ThreadPool pool(4);

    // Ch9.1.1：提交简单任务并获取 future。
    auto f1 = pool.submit([] { return 42; });
    auto f2 = pool.submit([](int a, int b) { return a + b; }, 10, 20);
    auto f3 = pool.submit([](double x) { return std::sqrt(x); }, 2.0);

    // Ch4.2.2：通过 future::get() 获取结果。
    std::cout << "f1 = " << f1.get() << "\n";
    std::cout << "f2 = " << f2.get() << "\n";
    std::cout << "f3 = " << f3.get() << "\n";

    // Ch6.2：用于生产者-消费者的线程安全队列。
    TaskQueue<std::string> messages;

    // 生产者线程（Ch2.1：基本线程创建）
    std::jthread producer([&messages] {
        for (int i = 0; i < 5; ++i) {
            messages.push(TS_FORMAT("Message #{}", i));
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });

    // 消费者：通过线程池处理消息
    for (int i = 0; i < 5; ++i) {
        std::string msg = messages.wait_and_pop();
        pool.submit([msg = std::move(msg)] {
            Logger::instance().info(TS_FORMAT("Processing: {}", msg));
        });
    }

    pool.wait_for_tasks();
    Logger::instance().info("=== 基本示例完成 ===");
    return 0;
}
