// Chapter 2 & 4: Basic Task Submission Example
// Minimal example showing the simplest usage of ThreadPool and TaskQueue.
// Demonstrates: Ch2 (thread management), Ch4.2 (future/promise), Ch6.2 (queue).

#include "task_scheduler/thread_pool.hpp"
#include "task_scheduler/task_queue.hpp"
#include "task_scheduler/logger.hpp"
#include <iostream>
#include <vector>
#include <cmath>

using namespace task_scheduler;

int main() {
    Logger::instance().info("=== Example: Basic Task Submission ===");

    // Ch9.1: Create a thread pool with 4 workers.
    ThreadPool pool(4);

    // Ch9.1.1: Submit simple tasks and get futures back.
    auto f1 = pool.submit([] { return 42; });
    auto f2 = pool.submit([](int a, int b) { return a + b; }, 10, 20);
    auto f3 = pool.submit([](double x) { return std::sqrt(x); }, 2.0);

    // Ch4.2.2: Retrieve results via future::get().
    std::cout << "f1 = " << f1.get() << "\n";
    std::cout << "f2 = " << f2.get() << "\n";
    std::cout << "f3 = " << f3.get() << "\n";

    // Ch6.2: Thread-safe queue for producer-consumer.
    TaskQueue<std::string> messages;

    // Producer thread (Ch2.1: basic thread creation)
    std::jthread producer([&messages] {
        for (int i = 0; i < 5; ++i) {
            messages.push(TS_FORMAT("Message #{}", i));
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });

    // Consumer: process messages via thread pool
    for (int i = 0; i < 5; ++i) {
        std::string msg = messages.wait_and_pop();
        pool.submit([msg = std::move(msg)] {
            Logger::instance().info(TS_FORMAT("Processing: {}", msg));
        });
    }

    pool.wait_for_tasks();
    Logger::instance().info("=== Basic Example Complete ===");
    return 0;
}
