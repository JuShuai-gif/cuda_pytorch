#pragma once
// Chapter 8 & 9: Task Scheduler - Core Orchestrator for AI/ML Inference
// Integrates:
//   - ThreadPool (Ch9.1) for parallel execution
//   - PriorityTaskQueue (Ch6.2+) for priority-based scheduling
//   - ConcurrentCache (Ch3.3.2) for caching inference results
//   - StopToken (Ch9.2) for graceful shutdown
//   - Logger (Ch11) for debugging and monitoring
//
// Features:
//   - Ch8.2: Task decomposition (splitting batches across workers)
//   - Ch8.3: Continuation-style task chaining via std::future (Ch4.2.3)
//   - Ch4.2: std::async-style fire-and-forget with future tracking
//   - Ch9.2: Cooperative interruption at task boundaries
//   - Ch3.2: Deadlock-avoidance via single-lock-per-component design

#include "task_scheduler/thread_pool.hpp"
#include "task_scheduler/priority_task_queue.hpp"
#include "task_scheduler/concurrent_cache.hpp"
#include "task_scheduler/stop_token.hpp"
#include "task_scheduler/logger.hpp"
#include <functional>
#include <future>
#include <vector>
#include <map>
#include <string>
#include <chrono>
#include <atomic>
#include <memory>

namespace task_scheduler {

// Ch8.5: Task metadata for scheduling decisions.
struct TaskMetadata {
    std::string name;
    TaskPriority priority{TaskPriority::NORMAL};
    std::chrono::milliseconds timeout{0}; // 0 = no timeout
    std::chrono::steady_clock::time_point submit_time;
};

// Ch8.1: TaskScheduler orchestrates task execution across the thread pool.
// It supports priority queues, batch processing, and periodic tasks.
class TaskScheduler {
public:
    TaskScheduler(size_t num_threads = 0,
                  size_t cache_size = 1024);
    ~TaskScheduler();

    TaskScheduler(const TaskScheduler&) = delete;
    TaskScheduler& operator=(const TaskScheduler&) = delete;

    // Ch8.2.1: Submit a single task with priority.
    // Returns std::future for result retrieval (Ch4.2.1).
    template <typename F, typename... Args>
        requires std::invocable<F, Args...>
    auto submit(TaskPriority priority, std::string_view name,
                F&& f, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>> {
        using ResultType = std::invoke_result_t<F, Args...>;

        auto task = std::make_shared<std::packaged_task<ResultType()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        std::future<ResultType> result = task->get_future();

        TaskMetadata meta;
        meta.name = name;
        meta.priority = priority;
        meta.submit_time = std::chrono::steady_clock::now();
        meta.timeout = default_timeout_;

        // Ch8.3: Enqueue in priority queue.
        priority_queue_.push(
            [this, task = std::move(task), meta]() mutable {
                Logger::instance().debug(
                    TS_FORMAT("TaskScheduler: executing '{}'", meta.name));
                try {
                    (*task)();
                } catch (const std::exception& e) {
                    Logger::instance().error(
                        TS_FORMAT("Task '{}' failed: {}", meta.name, e.what()));
                    throw;
                }
            },
            priority);

        // Ch8.4.1: Dispatch to thread pool for execution.
        dispatch_pending();
        return result;
    }

    // Ch8.2.2: Submit a batch of tasks with the same priority.
    // Returns a vector of futures (Ch4.2.4: shared_future alternative).
    template <typename F, typename... Args>
        requires std::invocable<F, Args...>
    auto submit_batch(TaskPriority priority, std::string_view name_prefix,
                      size_t count, F&& f, Args&&... args)
        -> std::vector<std::future<std::invoke_result_t<F, Args...>>> {
        using ResultType = std::invoke_result_t<F, Args...>;
        std::vector<std::future<ResultType>> futures;
        futures.reserve(count);

        for (size_t i = 0; i < count; ++i) {
            auto task = std::make_shared<std::packaged_task<ResultType()>>(
                std::bind(std::forward<F>(f), std::forward<Args>(args)...));

            futures.push_back(task->get_future());

            std::string task_name = TS_FORMAT("{}_{}", name_prefix, i);
            // Capture by value: task (shared_ptr) and task_name
            priority_queue_.push(
                [this, task = std::move(task), task_name]() mutable {
                    (*task)();
                },
                priority);
        }

        dispatch_pending();
        return futures;
    }

    // Ch8.3.1: Pipeline execution - chain multiple stages with dependency.
    // Stage 1 -> Stage 2 -> Stage 3. Each stage output feeds next stage input.
    // Returns future for final stage result (Ch4.2.2: future chaining).
    template <typename Input, typename Output>
    auto submit_pipeline(std::string_view pipeline_name,
                         std::function<Input()> stage1,
                         std::function<Output(Input)> stage2)
        -> std::future<Output> {
        // Ch4.2.3: Use std::async for simplicity, or submit to pool for control.
        // Here we demonstrate continuation via .then() pattern (Ch4.4.3).
        auto promise = std::make_shared<std::promise<Output>>();
        auto result = promise->get_future();

        pool_->submit([this, promise, stage1 = std::move(stage1),
                       stage2 = std::move(stage2), pipeline_name]() mutable {
            try {
                Logger::instance().debug(
                    TS_FORMAT("Pipeline '{}': stage1 executing", pipeline_name));
                Input intermediate = stage1();

                Logger::instance().debug(
                    TS_FORMAT("Pipeline '{}': stage2 executing", pipeline_name));
                Output final_result = stage2(std::move(intermediate));

                promise->set_value(std::move(final_result));
            } catch (...) {
                promise->set_exception(std::current_exception());
            }
        });

        return result;
    }

    // Ch8.5.1: Periodic task submission (Ch4.1.3: timed wait pattern).
    // Returns a stop_source that can stop the periodic task.
    using PeriodicCallback = std::function<void()>;
    stop_source schedule_periodic(PeriodicCallback callback,
                                  std::chrono::milliseconds interval,
                                  TaskPriority priority = TaskPriority::NORMAL,
                                  std::string_view name = "periodic");

    // Ch9.2.1: Shutdown the scheduler.
    void shutdown();

    // Ch9.2.2: Check if scheduler is shutting down.
    [[nodiscard]] bool is_stopping() const;

    // Ch8.5.2: Cache access for inference result reuse (Ch3.3.2).
    template <typename K, typename V>
    void cache_put(const K& key, const V& value) {
        cache_.put(key, value);
    }

    template <typename K, typename V>
    std::optional<V> cache_get(const K& key) {
        return cache_.get(key);
    }

    // Ch9.1.6: Wait for all current tasks to complete.
    void wait_for_all() { pool_->wait_for_tasks(); }

    // Ch8.5.3: Statistics.
    size_t pending_count() const { return priority_queue_.size(); }
    size_t active_threads() const { return pool_->worker_count(); }

private:
    // Ch8.4.1: Dispatch tasks from priority queue to thread pool.
    void dispatch_pending();

    // Ch8.4.2: Continuous dispatch loop for periodic tasks.
    void dispatch_loop();

    std::unique_ptr<ThreadPool> pool_;
    PriorityTaskQueue<std::function<void()>> priority_queue_;
    ConcurrentCache<std::string, std::string> cache_;

    std::chrono::milliseconds default_timeout_{0};
    std::atomic<bool> running_{true};

    // Ch9.2: Stop mechanism for periodic tasks.
    stop_source scheduler_stop_source_;
};

} // namespace task_scheduler
