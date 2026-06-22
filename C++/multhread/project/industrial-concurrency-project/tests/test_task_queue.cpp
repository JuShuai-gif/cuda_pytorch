// Ch10：任务队列单元测试
// 使用 MPMC 模式测试线程安全队列（Ch6.2）。

#include "task_scheduler/task_queue.hpp"
#include <cassert>
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <set>

using namespace task_scheduler;

// Ch10.4.1：基本 push/pop。
// 测试基础入队出队
void test_basic_push_pop() {
    std::cout << "  [test_basic_push_pop] ";
    TaskQueue<int> q;
    q.push(1);
    q.push(2);
    q.push(3);

    assert(q.size() == 3);
    assert(q.try_pop().value() == 1);
    assert(q.wait_and_pop() == 2);
    assert(q.wait_and_pop() == 3);
    assert(q.empty());
    std::cout << "PASSED\n";
}

// Ch10.4.2：多生产者，单消费者。
// 测试多生产者单消费者模式
void test_mpsc() {
    std::cout << "  [test_mpsc] ";
    TaskQueue<int> q;
    constexpr int N = 100;
    constexpr int producers = 4;

    std::atomic<int> total_received{0};
    std::set<int> received;
    std::mutex received_mutex;

    // 多个生产者（Ch6.2.1）
    std::vector<std::jthread> producer_threads;
    for (int p = 0; p < producers; ++p) {
        producer_threads.emplace_back([&q, p] {
            for (int i = 0; i < N; ++i) {
                q.push(p * N + i);
            }
        });
    }

    // 单个消费者（Ch6.2.2）
    auto consumer = std::jthread([&q, &total_received, &received, &received_mutex] {
        for (int i = 0; i < N * producers; ++i) {
            int item = q.wait_and_pop();
            total_received.fetch_add(1);
            std::lock_guard lock(received_mutex);
            received.insert(item);
        }
    });

    for (auto& t : producer_threads) t.join();
    consumer.join();

    assert(total_received.load() == N * producers);
    assert(received.size() == static_cast<size_t>(N * producers));
    std::cout << "PASSED\n";
}

// Ch10.4.3：使用 try_pop 的多消费者。
// 测试多消费者 try_pop 模式
void test_mpmc_try_pop() {
    std::cout << "  [test_mpmc_try_pop] ";
    TaskQueue<int> q;
    constexpr int N = 200;
    std::atomic<int> consumed{0};
    std::mutex result_mutex;
    std::set<int> results;

    // 生产者
    auto producer = std::jthread([&q] {
        for (int i = 0; i < N; ++i) {
            q.push(i);
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    });

    // 使用 try_pop 的多消费者（Ch6.2.4）
    std::vector<std::jthread> consumers;
    for (int c = 0; c < 4; ++c) {
        consumers.emplace_back([&q, &consumed, &results, &result_mutex] {
            while (consumed.load() < N) {
                if (auto item = q.try_pop()) {
                    consumed.fetch_add(1);
                    std::lock_guard lock(result_mutex);
                    results.insert(*item);
                } else {
                    std::this_thread::yield();
                }
            }
        });
    }

    producer.join();
    for (auto& t : consumers) t.join();

    assert(consumed.load() == N);
    assert(results.size() == static_cast<size_t>(N));
    std::cout << "PASSED\n";
}

// Ch10.4.4：带超时的等待（Ch4.1.2）。
// 测试超时等待
void test_timeout_wait() {
    std::cout << "  [test_timeout_wait] ";
    TaskQueue<int> q;

    // 超时时应返回 nullopt
    auto result = q.wait_and_pop_for(std::chrono::milliseconds(10));
    assert(!result.has_value()); // 超时，无项

    // 有数据时正常返回
    q.push(42);
    result = q.wait_and_pop_for(std::chrono::milliseconds(100));
    assert(result.has_value());
    assert(result.value() == 42);
    std::cout << "PASSED\n";
}

// Ch10.4.5：空队列行为。
// 测试空队列的各种行为
void test_empty_queue() {
    std::cout << "  [test_empty_queue] ";
    TaskQueue<std::string> q;

    assert(q.empty());
    assert(q.size() == 0);
    assert(!q.try_pop().has_value());

    q.push("hello");
    assert(!q.empty());
    q.try_pop();
    assert(q.empty());
    std::cout << "PASSED\n";
}

// Ch10.4.6：批量 pop（Ch6.2.5）。
// 测试批量出队
void test_bulk_pop() {
    std::cout << "  [test_bulk_pop] ";
    TaskQueue<int> q;
    for (int i = 0; i < 10; ++i) q.push(i);

    std::vector<int> items(5);
    size_t count = q.try_pop_bulk(items.begin(), 5);
    assert(count == 5);
    assert(q.size() == 5);
    std::cout << "PASSED\n";
}

int main() {
    std::cout << "=== 任务队列测试 ===\n";
    test_basic_push_pop();
    test_mpsc();
    test_mpmc_try_pop();
    test_timeout_wait();
    test_empty_queue();
    test_bulk_pop();
    std::cout << "=== 所有任务队列测试通过 ===\n";
    return 0;
}
