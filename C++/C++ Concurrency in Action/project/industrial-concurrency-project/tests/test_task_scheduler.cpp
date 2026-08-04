// Ch8 & Ch10：任务调度器单元测试
// 测试优先级调度、批量提交、流水线和定时任务。

#include "task_scheduler/task_scheduler.hpp"
#include <cassert>
#include <iostream>
#include <vector>
#include <atomic>
#include <set>
#include <chrono>

using namespace task_scheduler;

// Ch10.5.1：基本优先级任务执行。
// 测试基本优先级任务提交和执行
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

// Ch10.5.2：批量任务提交。
// 测试批量任务提交
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

// Ch10.5.3：多优先级级别。
// 测试混合优先级
void test_mixed_priorities() {
    std::cout << "  [test_mixed_priorities] ";
    TaskScheduler scheduler(4);
    std::atomic<int> execution_order{0};
    std::atomic<int> high_first{0};

    // 先提交低优先级
    auto low_f = scheduler.submit(TaskPriority::LOW, "low",
        [&] { high_first.store(execution_order.fetch_add(1)); return 1; });

    // 后提交高优先级
    auto high_f = scheduler.submit(TaskPriority::HIGH, "high",
        [&] { high_first.store(execution_order.fetch_add(1)); return 2; });

    // 两者都应该完成
    int l = low_f.get();
    int h = high_f.get();

    assert(l == 1);
    assert(h == 2);

    // Ch6.3：注意：单独的 std::priority_queue 不能保证多线程池中的执行顺序——
    // 它只保证分发顺序。此测试仅验证两个任务都能正确完成。
    std::cout << "PASSED\n";
}

// Ch10.5.4：流水线执行（Ch8.3）。
// 测试流水线执行
void test_pipeline() {
    std::cout << "  [test_pipeline] ";
    TaskScheduler scheduler(2);

    auto result = scheduler.submit_pipeline<int, std::string>(
        "test_pipeline",
        // 阶段 1：生成 int
        []() -> int { return 100; },
        // 阶段 2：转换为 string
        [](int x) -> std::string { return std::to_string(x * 2); }
    );

    assert(result.get() == "200");
    std::cout << "PASSED\n";
}

// Ch10.5.5：定时任务调度（Ch8.5.1）。
// 测试定时任务
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

    // 在 ~80ms 内以 20ms 间隔至少执行 3 次
    assert(counter.load() >= 3);
    std::cout << "PASSED\n";
}

// Ch10.5.6：缓存集成（Ch3.3.2）。
// 测试缓存集成
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
    assert(!v3.has_value()); // 不存在的键应返回 nullopt
    std::cout << "PASSED\n";
}

// Ch10.5.7：流水线中的异常传播。
// 测试流水线异常传播
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

// Ch10.5.8：关闭测试。
// 测试调度器关闭
void test_scheduler_shutdown() {
    std::cout << "  [test_scheduler_shutdown] ";
    {
        TaskScheduler scheduler(2);
        auto f = scheduler.submit(TaskPriority::NORMAL, "task1", [] { return 1; });
        f.get(); // 确保任务在析构函数之前完成（Ch9.2.1：优雅关闭）。
        // 析构函数调用 shutdown
    }
    std::cout << "PASSED\n";
}

int main() {
    std::cerr << "=== 任务调度器测试 ===" << std::endl;
    std::cerr << "  1: priority_execution" << std::endl; test_priority_execution();
    std::cerr << "  2: batch_submission" << std::endl; test_batch_submission();
    std::cerr << "  3: mixed_priorities" << std::endl; test_mixed_priorities();
    std::cerr << "  4: pipeline" << std::endl; test_pipeline();
    std::cerr << "  5: periodic_task" << std::endl; test_periodic_task();
    std::cerr << "  6: cache_integration" << std::endl; test_cache_integration();
    std::cerr << "  7: pipeline_exception" << std::endl; test_pipeline_exception();
    std::cerr << "  8: scheduler_shutdown" << std::endl; test_scheduler_shutdown();
    std::cerr << "=== 所有任务调度器测试通过 ===" << std::endl;
    return 0;
}
