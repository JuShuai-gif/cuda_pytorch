// Chapter 2: Basic Thread Management - Entry Point
// Demonstrates the complete TaskScheduler system for AI/ML inference workloads.
// Covers concepts from all chapters of "C++ Concurrency in Action".

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

// Ch8.4: Simulate an AI/ML inference workload.
// Each "inference" task processes a batch of data and returns a result.
double simulate_inference(int batch_id, int batch_size) {
    // Ch11.5: Simulate computation with deterministic work.
    double sum = 0.0;
    for (int i = 0; i < batch_size * 1000; ++i) {
        sum += std::sin(i * 0.001) * std::cos(i * 0.002);
    }
    Logger::instance().debug(TS_FORMAT("Batch {} completed", batch_id));
    return sum;
}

// Ch8.2: Preprocessing stage of inference pipeline.
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
    Logger::instance().info("=== Industrial Concurrency Project: AI/ML TaskScheduler ===");

    // ================================================================
    // Ch3.3.2: Concurrent Cache Demo
    // ================================================================
    {
        Logger::instance().info("--- Demo 1: Concurrent Cache (Ch3.3.2) ---");
        ConcurrentCache<int, std::string> cache(16);

        // Multiple threads reading from cache (shared_lock, Ch3.3.2)
        std::vector<std::jthread> readers;
        for (int i = 0; i < 4; ++i) {
            readers.emplace_back([&cache, i] {
                for (int j = 0; j < 100; ++j) {
                    cache.get(j % 10);
                }
            });
        }
        for (auto& t : readers) t.join();

        // Single writer (unique_lock, Ch3.3.1)
        for (int i = 0; i < 10; ++i) {
            cache.put(i, TS_FORMAT("value_{}", i));
        }
        Logger::instance().info(TS_FORMAT("Cache size: {}", cache.size()));
    }

    // ================================================================
    // Ch5: Spinlock Demo
    // ================================================================
    {
        Logger::instance().info("--- Demo 2: TTAS Spinlock (Ch5) ---");
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
    // Ch9.1: Thread Pool + Task Queue Demo
    // ================================================================
    {
        Logger::instance().info("--- Demo 3: Thread Pool (Ch9.1) ---");
        ThreadPool pool(4);

        std::vector<std::future<double>> futures;
        for (int i = 0; i < 20; ++i) {
            // Ch9.1.1: submit returns future
            futures.push_back(pool.submit(simulate_inference, i, 50));
        }

        // Ch4.2.2: Wait for all futures
        double total = 0.0;
        for (auto& f : futures) {
            total += f.get();
        }
        Logger::instance().info(TS_FORMAT("Thread pool total: {:.2f}", total));
    }

    // ================================================================
    // Ch8.5: Task Scheduler with Priority Demo
    // ================================================================
    {
        Logger::instance().info("--- Demo 4: Task Scheduler (Ch8.5) ---");
        TaskScheduler scheduler(4);

        // Submit tasks with different priorities (Ch6.3: priority scheduling)
        auto f1 = scheduler.submit(TaskPriority::LOW, "low_priority_task",
            [] { return 1; });
        auto f2 = scheduler.submit(TaskPriority::HIGH, "high_priority_task",
            [] { return 2; });
        auto f3 = scheduler.submit(TaskPriority::CRITICAL, "critical_task",
            [] { return 3; });

        // Ch4.2.4: Batch submission
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
    // Ch8.3: Pipeline Demo
    // ================================================================
    {
        Logger::instance().info("--- Demo 5: Pipeline (Ch8.3) ---");
        TaskScheduler scheduler(2);

        // Ch8.3.1: Two-stage pipeline: preprocess -> inference
        auto pipeline_result = scheduler.submit_pipeline<double, double>(
            "inference_pipeline",
            // Stage 1: Prepares data
            []() -> double {
                double sum = 0.0;
                for (int i = 0; i < 100; ++i) sum += i;
                return sum;
            },
            // Stage 2: Inference
            [](double data) -> double {
                return data * 1.5;
            }
        );

        Logger::instance().info(TS_FORMAT("Pipeline result: {:.2f}",
            pipeline_result.get()));
    }

    // ================================================================
    // Ch8.5.1: Periodic Task Demo
    // ================================================================
    {
        Logger::instance().info("--- Demo 6: Periodic Tasks (Ch8.5.1) ---");
        TaskScheduler scheduler(2);

        int periodic_count = 0;
        auto stop = scheduler.schedule_periodic(
            [&periodic_count] { ++periodic_count; },
            std::chrono::milliseconds(50),
            TaskPriority::NORMAL,
            "health_check"
        );

        // Let it run for a bit
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        // Ch9.2: Stop the periodic task
        stop.request_stop();

        Logger::instance().info(TS_FORMAT("Periodic task ran {} times",
            periodic_count));
    }

    Logger::instance().info("=== All demos completed successfully ===");
    return 0;
}
