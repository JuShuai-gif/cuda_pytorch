// Ch2：基本线程管理——入口点
// 演示面向 AI/ML 推理工作负载的完整 TaskScheduler 系统。
// 涵盖《C++ Concurrency in Action》所有章节的概念。

#include "task_scheduler/task_scheduler.hpp"
#include "task_scheduler/task_queue.hpp"
#include "task_scheduler/priority_task_queue.hpp"
#include "task_scheduler/concurrent_cache.hpp"
#include "task_scheduler/spinlock.hpp"
#include "task_scheduler/logger.hpp"
#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include <random>

using namespace task_scheduler;

// Ch8.4：模拟 AI/ML 推理工作负载。
// 每个"推理"任务处理一批数据并返回结果。
// 模拟推理工作量
double simulate_inference(int batch_id, int batch_size) {
    // Ch11.5：用确定性工作模拟计算。
    double sum = 0.0;
    for (int i = 0; i < batch_size * 1000; ++i) {
        sum += std::sin(i * 0.001) * std::cos(i * 0.002);
    }
    Logger::instance().debug(TS_FORMAT("Batch {} completed", batch_id));
    return sum;
}

// Ch8.2：推理流水线的预处理阶段。
// 预处理：标准化输入数据
std::vector<double> preprocess(const std::vector<double>& raw_data) {
    std::vector<double> normalized;
    normalized.reserve(raw_data.size());
    double mean = std::accumulate(raw_data.begin(), raw_data.end(), 0.0) / raw_data.size();
    for (auto v : raw_data) {
        normalized.push_back(v - mean);
    }
    Logger::instance().debug("Preprocessing completed");
    return normalized;
}

int main() {
    Logger::instance().info("=== 工业级并发项目：AI/ML 任务调度器 ===");

    // ================================================================
    // 演示 1：并发缓存（Ch3.3.2）
    // Ch3.3.2：并发缓存演示（读写锁）
    // ================================================================
    {
        Logger::instance().info("--- 演示 1：并发缓存（Ch3.3.2）---");
        ConcurrentCache<int, std::string> cache(16);

        // 多线程从缓存读取（shared_lock，Ch3.3.2）
        std::vector<std::jthread> readers;
        for (int i = 0; i < 4; ++i) {
            readers.emplace_back([&cache, i] {
                for (int j = 0; j < 100; ++j) {
                    cache.get(j % 10);
                }
            });
        }
        for (auto& t : readers) t.join();

        // 单个写入者（unique_lock，Ch3.3.1）
        for (int i = 0; i < 10; ++i) {
            cache.put(i, TS_FORMAT("value_{}", i));
        }
        Logger::instance().info(TS_FORMAT("Cache size: {}", cache.size()));
    }

    // ================================================================
    // 演示 2：TTAS 自旋锁（Ch5）
    // Ch5：自旋锁演示
    // ================================================================
    {
        Logger::instance().info("--- 演示 2：TTAS 自旋锁（Ch5）---");
        spinlock sl;
        int shared_counter = 0;

        std::vector<std::jthread> threads;
        for (int i = 0; i < 4; ++i) {
            threads.emplace_back([&sl, &shared_counter] {
                for (int j = 0; j < 1000; ++j) {
                    spinlock_guard guard(sl);
                    ++shared_counter;
                }
            });
        }
        for (auto& t : threads) t.join();
        Logger::instance().info(TS_FORMAT("Spinlock counter: {}", shared_counter));
    }

    // ================================================================
    // 演示 3：线程池 + 任务队列（Ch9.1）
    // Ch9.1：线程池演示
    // ================================================================
    {
        Logger::instance().info("--- 演示 3：线程池（Ch9.1）---");
        ThreadPool pool(4);

        std::vector<std::future<double>> futures;
        for (int i = 0; i < 20; ++i) {
            // Ch9.1.1：submit 返回 future
            futures.push_back(pool.submit(simulate_inference, i, 50));
        }

        // Ch4.2.2：等待所有 future
        double total = 0.0;
        for (auto& f : futures) {
            total += f.get();
        }
        Logger::instance().info(TS_FORMAT("Thread pool total: {:.2f}", total));
    }

    // ================================================================
    // 演示 4：带优先级调度的任务调度器（Ch8.5）
    // Ch8.5：任务调度器演示
    // ================================================================
    {
        Logger::instance().info("--- 演示 4：任务调度器（Ch8.5）---");
        TaskScheduler scheduler(4);

        // 提交不同优先级的任务（Ch6.3：优先级调度）
        auto f1 = scheduler.submit(TaskPriority::LOW, "low_priority_task",
            [] { return 1; });
        auto f2 = scheduler.submit(TaskPriority::HIGH, "high_priority_task",
            [] { return 2; });
        auto f3 = scheduler.submit(TaskPriority::CRITICAL, "critical_task",
            [] { return 3; });

        // Ch4.2.4：批量提交
        auto batch_futures = scheduler.submit_batch(
            TaskPriority::NORMAL, "batch_inference", 5,
            [] { return 42; });

        Logger::instance().info(TS_FORMAT("Low: {}, High: {}, Critical: {}",
            f1.get(), f2.get(), f3.get()));

        int batch_total = 0;
        for (auto& f : batch_futures) batch_total += f.get();
        Logger::instance().info(TS_FORMAT("Batch total: {}", batch_total));
    }

    // ================================================================
    // 演示 5：流水线（Ch8.3）
    // Ch8.3：流水线演示
    // ================================================================
    {
        Logger::instance().info("--- 演示 5：流水线（Ch8.3）---");
        TaskScheduler scheduler(2);

        // Ch8.3.1：两阶段流水线：预处理 -> 推理
        auto pipeline_result = scheduler.submit_pipeline<double, double>(
            "inference_pipeline",
            // 阶段 1：准备数据
            []() -> double {
                double sum = 0.0;
                for (int i = 0; i < 100; ++i) sum += i;
                return sum;
            },
            // 阶段 2：推理
            [](double data) -> double {
                return data * 1.5;
            }
        );

        Logger::instance().info(TS_FORMAT("Pipeline result: {:.2f}",
            pipeline_result.get()));
    }

    // ================================================================
    // 演示 6：定时任务（Ch8.5.1）
    // Ch8.5.1：定时任务演示
    // ================================================================
    {
        Logger::instance().info("--- 演示 6：定时任务（Ch8.5.1）---");
        TaskScheduler scheduler(2);

        int periodic_count = 0;
        auto stop = scheduler.schedule_periodic(
            [&periodic_count] { ++periodic_count; },
            std::chrono::milliseconds(50),
            TaskPriority::NORMAL,
            "health_check"
        );

        // 让它运行一会儿
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        // Ch9.2：停止定时任务
        stop.request_stop();

        Logger::instance().info(TS_FORMAT("Periodic task ran {} times",
            periodic_count));
    }

    Logger::instance().info("=== 所有演示成功完成 ===");
    return 0;
}
