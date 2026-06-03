// Chapter 8 & 10: Task Scheduler Unit Tests
// Tests priority scheduling, batch submission, pipeline, and periodic tasks.

#include "task_scheduler/task_scheduler.hpp"
#include <cassert>
#include <iostream>
#include <vector>
#include <atomic>
#include <set>
#include <chrono>

using namespace task_scheduler;

// Ch10.5.1: Basic priority task execution.
void test_priority_execution() {
    std::cerr << "  [test_priority_execution] START" << std::endl;
    TaskScheduler scheduler(2);
    std::cerr << "  [test_priority_execution] scheduler created" << std::endl;
    auto f = scheduler.submit(TaskPriority::HIGH, "test_task",
        [](int x) { return x * 2; }, 21);
    std::cerr << "  [test_priority_execution] task submitted" << std::endl;
    int result = f.get();
    std::cerr << "  [test_priority_execution] result: " << result << std::endl;
    assert(result == 42);
    std::cerr << "  [test_priority_execution] PASSED" << std::endl;
}

// Ch10.5.2: Batch task submission.
void test_batch_submission() {
    std::cerr << "  [test_batch_submission] START" << std::endl;
    TaskScheduler scheduler(4);
    std::cerr << "  [test_batch_submission] scheduler created" << std::endl;
    auto futures = scheduler.submit_batch(
        TaskPriority::NORMAL, "batch", 50,
        [](int x) { return x * x; }, 3);
    std::cerr << "  [test_batch_submission] batch submitted, size=" << futures.size() << std::endl;
    for (size_t i = 0; i < futures.size(); ++i) {
        int r = futures[i].get();
        assert(r == 9);
        if (i % 10 == 0) std::cerr << "  [test_batch_submission] got result " << i << ": " << r << std::endl;
    }
    std::cerr << "  [test_batch_submission] PASSED" << std::endl;
}

// Ch10.5.3: Multiple priority levels.
void test_mixed_priorities() {
    std::cout << "  [test_mixed_priorities] ";
    TaskScheduler scheduler(4);
    std::atomic<int> execution_order{0};
    std::atomic<int> high_first{0};

    // Submit low priority first
    auto low_f = scheduler.submit(TaskPriority::LOW, "low",
        [&] { high_first.store(execution_order.fetch_add(1)); return 1; });

    // Submit high priority second
    auto high_f = scheduler.submit(TaskPriority::HIGH, "high",
        [&] { high_first.store(execution_order.fetch_add(1)); return 2; });

    // Both should complete
    int l = low_f.get();
    int h = high_f.get();

    assert(l == 1);
    assert(h == 2);

    // Ch6.3: Note: std::priority_queue alone doesn't guarantee execution order
    // in a multi-threaded pool - it guarantees dispatch order. The test just
    // verifies both tasks complete correctly.
    std::cout << "PASSED\n";
}

// Ch10.5.4: Pipeline execution (Ch8.3).
void test_pipeline() {
    std::cout << "  [test_pipeline] ";
    TaskScheduler scheduler(2);

    auto result = scheduler.submit_pipeline<int, std::string>(
        "test_pipeline",
        // Stage 1: generate int
        []() -> int { return 100; },
        // Stage 2: convert to string
        [](int x) -> std::string { return std::to_string(x * 2); }
    );

    assert(result.get() == "200");
    std::cout << "PASSED\n";
}

// Ch10.5.5: Periodic task scheduling (Ch8.5.1).
void test_periodic_task() {
    std::cout << "  [test_periodic_task] ";
    TaskScheduler scheduler(2);
    std::atomic<int> counter{0};

    auto stop = scheduler.schedule_periodic(
        [&counter] { counter.fetch_add(1); },
        std::chrono::milliseconds(20),
        TaskPriority::NORMAL,
        "test_periodic"
    );

    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    stop.request_stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(30));

    assert(counter.load() >= 3); // At least 3 invocations in ~80ms at 20ms interval
    std::cout << "PASSED\n";
}

// Ch10.5.6: Cache integration (Ch3.3.2).
void test_cache_integration() {
    std::cout << "  [test_cache_integration] ";
    TaskScheduler scheduler(2, 64);

    scheduler.cache_put<std::string, std::string>("key1", "value1");
    scheduler.cache_put<std::string, std::string>("key2", "value2");

    auto v1 = scheduler.cache_get<std::string, std::string>("key1");
    auto v2 = scheduler.cache_get<std::string, std::string>("key2");
    auto v3 = scheduler.cache_get<std::string, std::string>("key3");

    assert(v1.has_value() && v1.value() == "value1");
    assert(v2.has_value() && v2.value() == "value2");
    assert(!v3.has_value());
    std::cout << "PASSED\n";
}

// Ch10.5.7: Exception propagation through pipeline.
void test_pipeline_exception() {
    std::cout << "  [test_pipeline_exception] ";
    TaskScheduler scheduler(2);

    auto result = scheduler.submit_pipeline<int, int>(
        "failing_pipeline",
        []() -> int { throw std::runtime_error("stage1 failure"); return 0; },
        [](int x) -> int { return x * 2; }
    );

    try {
        result.get();
        assert(false && "Expected exception");
    } catch (const std::runtime_error& e) {
        assert(std::string(e.what()) == "stage1 failure");
    }
    std::cout << "PASSED\n";
}

// Ch10.5.8: Shutdown test.
void test_scheduler_shutdown() {
    std::cout << "  [test_scheduler_shutdown] ";
    {
        TaskScheduler scheduler(2);
        auto f = scheduler.submit(TaskPriority::NORMAL, "task1", [] { return 1; });
        f.get(); // Ensure task completes before destructor (Ch9.2.1: graceful shutdown).
        // Destructor calls shutdown
    }
    std::cout << "PASSED\n";
}

int main() {
    std::cerr << "=== TaskScheduler Tests ===" << std::endl;
    std::cerr << "  1: priority_execution" << std::endl; test_priority_execution();
    std::cerr << "  2: batch_submission" << std::endl; test_batch_submission();
    std::cerr << "  3: mixed_priorities" << std::endl; test_mixed_priorities();
    std::cerr << "  4: pipeline" << std::endl; test_pipeline();
    std::cerr << "  5: periodic_task" << std::endl; test_periodic_task();
    std::cerr << "  6: cache_integration" << std::endl; test_cache_integration();
    std::cerr << "  7: pipeline_exception" << std::endl; test_pipeline_exception();
    std::cerr << "  8: scheduler_shutdown" << std::endl; test_scheduler_shutdown();
    std::cerr << "=== All TaskScheduler tests passed ===" << std::endl;
    return 0;
}
