// Ch9 & Ch10：线程池单元测试
// 测试 ThreadPool 的基本功能、边缘情况和线程安全性。

#include "task_scheduler/thread_pool.hpp"
#include <cassert>
#include <iostream>
#include <vector>
#include <atomic>
#include <set>
#include <algorithm>
#include <chrono>

using namespace task_scheduler;

// Ch10.2：测试辅助工具——跨线程计数调用（原子变量保证安全）。
// 调用计数器辅助结构
struct CallCounter {
    std::atomic<int> count{0};
    void increment() { count.fetch_add(1, std::memory_order_relaxed); }
};

// Ch10.3.1：测试基本提交和获取结果。
// 测试基本任务提交
void test_basic_submit() {
    std::cout << "  [test_basic_submit] ";
    ThreadPool pool(4);

    auto future = pool.submit([](int a, int b) { return a + b; }, 2, 3);
    int result = future.get();
    assert(result == 5);
    std::cout << "PASSED\n";
}

// Ch10.3.2：测试多个并发提交。
// 测试并发提交
void test_concurrent_submits() {
    std::cout << "  [test_concurrent_submits] ";
    ThreadPool pool(4);
    constexpr int N = 100;

    std::vector<std::future<int>> futures;
    futures.reserve(N);
    for (int i = 0; i < N; ++i) {
        futures.push_back(pool.submit([i] { return i * i; }));
    }

    std::set<int> results;
    for (auto& f : futures) {
        results.insert(f.get());
    }
    assert(results.size() == N);
    assert(*results.begin() == 0);
    assert(*results.rbegin() == (N-1) * (N-1));
    std::cout << "PASSED\n";
}

// Ch10.3.3：测试工作窃取（Ch8.4）。
// 测试工作窃取：将任务不均匀分布以触发窃取
void test_work_stealing() {
    std::cout << "  [test_work_stealing] ";
    ThreadPool pool(4);
    CallCounter counter;

    // 将任务提交到不均匀的本地队列以触发窃取。
    for (size_t i = 0; i < 50; ++i) {
        pool.submit_to_local(i % 2, [&counter] {
            counter.increment();
            // 变化执行时间以产生不平衡。
            std::this_thread::sleep_for(std::chrono::microseconds(100 + rand() % 200));
        });
    }

    pool.wait_for_tasks();
    assert(counter.count.load() == 50);
    std::cout << "PASSED\n";
}

// Ch10.3.4：测试线程池可以优雅关闭。
// 测试优雅关闭
void test_shutdown() {
    std::cout << "  [test_shutdown] ";
    {
        ThreadPool pool(4);
        pool.submit([] { std::this_thread::sleep_for(std::chrono::milliseconds(10)); });
        pool.submit([] { std::this_thread::sleep_for(std::chrono::milliseconds(10)); });
        // 析构函数调用 shutdown()——应 join 所有线程。
    }
    std::cout << "PASSED\n";
}

// Ch10.3.5：测试任务的异常传播。
// 测试异常传播
void test_exception_propagation() {
    std::cout << "  [test_exception_propagation] ";
    ThreadPool pool(2);

    auto future = pool.submit([]() -> int {
        throw std::runtime_error("test exception");
    });

    try {
        future.get();
        assert(false && "Expected exception not thrown");
    } catch (const std::runtime_error& e) {
        assert(std::string(e.what()) == "test exception");
    }
    std::cout << "PASSED\n";
}

// Ch10.3.6：测试 wait_for_tasks。
// 测试等待所有任务完成
void test_wait_for_tasks() {
    std::cout << "  [test_wait_for_tasks] ";
    ThreadPool pool(4);
    std::atomic<int> counter{0};

    for (int i = 0; i < 20; ++i) {
        pool.submit([&counter] {
            counter.fetch_add(1, std::memory_order_relaxed);
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        });
    }

    pool.wait_for_tasks();
    assert(counter.load() == 20);
    std::cout << "PASSED\n";
}

// Ch10.3.7：测试在关闭后提交（应抛出异常）。
// 测试关闭后拒绝新任务
void test_submit_after_shutdown() {
    std::cout << "  [test_submit_after_shutdown] ";
    ThreadPool pool(2);
    pool.shutdown();

    try {
        pool.submit([] { return 42; });
        assert(false && "Expected exception for submit after shutdown");
    } catch (const std::runtime_error&) {
        // 预期行为
    }
    std::cout << "PASSED\n";
}

// Ch10.3.8：压力测试——多线程，多任务。
// 压力测试：10000 任务，8 线程
void test_stress() {
    std::cout << "  [test_stress] ";
    ThreadPool pool(8);
    std::atomic<size_t> counter{0};
    constexpr size_t N = 10000;

    std::vector<std::future<void>> futures;
    futures.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        futures.push_back(pool.submit([&counter] {
            counter.fetch_add(1, std::memory_order_relaxed);
        }));
    }

    for (auto& f : futures) {
        f.get();
    }
    assert(counter.load() == N);
    std::cout << "PASSED\n";
}

int main() {
    std::cout << "=== 线程池测试 ===\n";
    test_basic_submit();
    test_concurrent_submits();
    test_work_stealing();
    test_shutdown();
    test_exception_propagation();
    test_wait_for_tasks();
    test_submit_after_shutdown();
    test_stress();
    std::cout << "=== 所有线程池测试通过 ===\n";
    return 0;
}
