// Chapter 9 & 10: Thread Pool Unit Tests
// Tests basic functionality, edge cases, and thread safety of ThreadPool.

#include "task_scheduler/thread_pool.hpp"
#include <cassert>
#include <iostream>
#include <vector>
#include <atomic>
#include <set>
#include <algorithm>
#include <chrono>

using namespace task_scheduler;

// Ch10.2: Test helper - counts calls across threads (atomic for safety).
struct CallCounter {
    std::atomic<int> count{0};
    void increment() { count.fetch_add(1, std::memory_order_relaxed); }
};

// Ch10.3.1: Test basic submit and get result.
void test_basic_submit() {
    std::cout << "  [test_basic_submit] ";
    ThreadPool pool(4);

    auto future = pool.submit([](int a, int b) { return a + b; }, 2, 3);
    int result = future.get();
    assert(result == 5);
    std::cout << "PASSED\n";
}

// Ch10.3.2: Test multiple concurrent submits.
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

// Ch10.3.3: Test work stealing (Ch8.4).
void test_work_stealing() {
    std::cout << "  [test_work_stealing] ";
    ThreadPool pool(4);
    CallCounter counter;

    // Submit tasks to uneven local queues to trigger stealing.
    for (size_t i = 0; i < 50; ++i) {
        pool.submit_to_local(i % 2, [&counter] {
            counter.increment();
            // Vary execution time to create imbalance.
            std::this_thread::sleep_for(std::chrono::microseconds(100 + rand() % 200));
        });
    }

    pool.wait_for_tasks();
    assert(counter.count.load() == 50);
    std::cout << "PASSED\n";
}

// Ch10.3.4: Test that pool can be shut down gracefully.
void test_shutdown() {
    std::cout << "  [test_shutdown] ";
    {
        ThreadPool pool(4);
        pool.submit([] { std::this_thread::sleep_for(std::chrono::milliseconds(10)); });
        pool.submit([] { std::this_thread::sleep_for(std::chrono::milliseconds(10)); });
        // Destructor calls shutdown() - should join all threads.
    }
    std::cout << "PASSED\n";
}

// Ch10.3.5: Test exception propagation from tasks.
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

// Ch10.3.6: Test wait_for_tasks.
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

// Ch10.3.7: Test submit after shutdown (should throw).
void test_submit_after_shutdown() {
    std::cout << "  [test_submit_after_shutdown] ";
    ThreadPool pool(2);
    pool.shutdown();

    try {
        pool.submit([] { return 42; });
        assert(false && "Expected exception for submit after shutdown");
    } catch (const std::runtime_error&) {
        // Expected
    }
    std::cout << "PASSED\n";
}

// Ch10.3.8: Stress test - many threads, many tasks.
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
    std::cout << "=== ThreadPool Tests ===\n";
    test_basic_submit();
    test_concurrent_submits();
    test_work_stealing();
    test_shutdown();
    test_exception_propagation();
    test_wait_for_tasks();
    test_submit_after_shutdown();
    test_stress();
    std::cout << "=== All ThreadPool tests passed ===\n";
    return 0;
}
