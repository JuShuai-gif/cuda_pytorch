#pragma once
// Chapter 9.1: Thread Pools - Advanced Thread Management
// Implements a fixed-size thread pool with:
//   - Ch8.4: Work stealing for load balancing
//   - Ch9.1.1: submit() returning std::future (via std::packaged_task)
//   - Ch9.2: stop_token for cooperative interruption
//   - Ch4.2: std::future/std::promise for result propagation
//   - Ch4.4: std::packaged_task wrapping callables
//   - Ch3.2: std::mutex for queue protection
//   - Ch4.1: std::condition_variable for worker wake-up

#include "task_scheduler/task_queue.hpp"
#include "task_scheduler/stop_token.hpp"
#include <vector>
#include <thread>
#include <future>
#include <functional>
#include <type_traits>
#include <memory>
#include <atomic>
#include <random>

namespace task_scheduler {

class ThreadPool {
public:
    // Ch9.1: Constructor launches num_threads worker threads.
    // If num_threads == 0, uses hardware_concurrency (Ch8.4.1).
    explicit ThreadPool(size_t num_threads = 0);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    // Ch9.1.1: Submit a callable and get a std::future for the result.
    // Uses std::packaged_task (Ch4.4.1) internally.
    // Supports any callable with any argument types (Ch9.1.3: variadic templates).
    template <typename F, typename... Args>
        requires std::invocable<F, Args...>
    auto submit(F&& f, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>> {
        using ResultType = std::invoke_result_t<F, Args...>;

        // Ch4.4.1: packaged_task wraps the callable and connects to a future.
        // Use shared_ptr because packaged_task is move-only and we need to
        // store it in std::function (which requires copyable).
        auto task = std::make_shared<std::packaged_task<ResultType()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        std::future<ResultType> result = task->get_future();

        // Ch9.1.2: Submit to global queue or a worker's local queue.
        // For simplicity, push to global queue; production would use
        // round-robin distribution to local queues (see submit_to_local()).
        {
            std::lock_guard lock(queue_mutex_);
            if (stop_source_.stop_requested()) {
                throw std::runtime_error("ThreadPool: cannot submit to stopped pool");
            }
            active_tasks_.fetch_add(1, std::memory_order_release);
            // Ch4.4.2: Type-erase via std::function<void()> for uniform storage.
            global_queue_.push([task]() { (*task)(); });
        }
        cond_.notify_one();
        return result;
    }

    // Ch9.1.4: Submit to a specific worker's local queue (for work distribution).
    template <typename F, typename... Args>
        requires std::invocable<F, Args...>
    auto submit_to_local(size_t worker_idx, F&& f, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>> {
        using ResultType = std::invoke_result_t<F, Args...>;

        auto task = std::make_shared<std::packaged_task<ResultType()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        std::future<ResultType> result = task->get_future();

        if (worker_idx >= workers_.size()) {
            throw std::out_of_range("ThreadPool: worker index out of range");
        }
        active_tasks_.fetch_add(1, std::memory_order_release);
        workers_[worker_idx]->local_queue.push([task]() { (*task)(); });

        return result;
    }

    // Ch9.1.5: Wait until all submitted tasks have completed (does NOT stop pool).
    void wait_for_tasks();

    // Ch9.2.1: Request stop and wait for all threads to finish (graceful shutdown).
    void shutdown();

    // Ch9.2.2: Check if stop has been requested.
    [[nodiscard]] bool is_stopping() const {
        return stop_source_.stop_requested();
    }

    // Ch9.1.6: Number of worker threads.
    [[nodiscard]] size_t worker_count() const { return workers_.size(); }

    // Ch9.1.7: Approximate count of pending tasks.
    [[nodiscard]] size_t pending_tasks() const;

private:
    // Ch9.1.8: Per-worker thread state.
    // Each worker has its own local queue for work stealing (Ch8.4.2).
    struct Worker {
        size_t index;
        // Ch9.2.3: jthread (C++20) supports cooperative interruption natively.
        // We use our own stop_token wrapper for educational purposes.
        std::jthread thread;
        TaskQueue<std::function<void()>> local_queue;
        std::atomic<bool> running{true};
    };

    // Ch9.1.9: Main worker loop (runs in each thread).
    void worker_loop(size_t worker_idx);

    // Ch8.4.3: Work stealing - try to take a task from another worker's local queue.
    bool steal_task(size_t worker_idx, std::function<void()>& task);

    // Ch9.1.10: Try to get a task from own local queue first, then global, then steal.
    bool get_task(size_t worker_idx, std::function<void()>& task, stop_token st);

    // Ch8.4.1: Total number of active tasks counter for wait_for_tasks.
    std::atomic<size_t> active_tasks_{0};

    std::vector<std::unique_ptr<Worker>> workers_;

    // Ch9.1.11: Global task queue for initial submissions.
    TaskQueue<std::function<void()>> global_queue_;
    mutable std::mutex queue_mutex_; // Protects global_queue_ synchronization
    std::condition_variable cond_;

    stop_source stop_source_;

    // Ch9.2.1: Signal that all workers are initialized (prevents stealing during init).
    std::atomic<bool> pool_ready_{false};
};

} // namespace task_scheduler
