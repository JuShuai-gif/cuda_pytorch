// Chapter 9: Thread Pool Implementation
// Implements the worker loop and work stealing logic.
// Demonstrates Ch9.1 (thread pool design), Ch8.4 (work stealing),
// Ch4.1 (condition_variable), and Ch3.2 (mutex patterns).

#include "task_scheduler/thread_pool.hpp"
#include "task_scheduler/logger.hpp"
#include <random>
#include <algorithm>
#include <cassert>

namespace task_scheduler {

// Ch9.1: Constructor launches worker threads.
// Ch8.4.1: If num_threads is 0, use hardware_concurrency.
ThreadPool::ThreadPool(size_t num_threads) {
    size_t count = num_threads > 0 ? num_threads : std::thread::hardware_concurrency();
    count = std::max(count, size_t(1));

    auto st = stop_source_.get_token();

    // Pre-allocate to prevent vector reallocation during thread startup.
    workers_.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        // Push worker FIRST so workers_[i] exists before thread accesses it.
        workers_.push_back(std::make_unique<Worker>());
        workers_.back()->index = i;

        // Ch9.1.2: Launch each worker thread with the worker loop.
        workers_.back()->thread = std::jthread([this, i, st] {
            worker_loop(i);
        });
    }

    // Signal that all workers are initialized and stealing is safe (Ch9.2.1).
    pool_ready_.store(true, std::memory_order_release);
}

// Ch9.1.7: Destructor calls shutdown for graceful cleanup (RAII).
ThreadPool::~ThreadPool() {
    shutdown();
}

// Ch9.2: Graceful shutdown process.
void ThreadPool::shutdown() {
    // Ch9.2.1: Signal all workers to stop.
    stop_source_.request_stop();

    // Ch9.2.3: Wake all waiting workers so they can check stop flag.
    {
        std::lock_guard lock(queue_mutex_);
    }
    cond_.notify_all();
    for (auto& w : workers_) {
        w->local_queue.notify_all();
    }
    global_queue_.notify_all();

    // Ch9.1.6: jthread automatically joins in destructor (RAII).
    workers_.clear();
}

// Ch9.1.3: Worker main loop.
// Each worker runs this loop until stop is requested AND all tasks are drained.
void ThreadPool::worker_loop(size_t worker_idx) {
    auto st = stop_source_.get_token();

    while (true) {
        std::function<void()> task;

        // Ch9.1.4: Try to get a task.
        if (get_task(worker_idx, task, st)) {
            // Ch9.1.5: Execute the task.
            try {
                task();
            } catch (...) {
                Logger::instance().error("Worker thread caught unhandled exception");
            }
            active_tasks_.fetch_sub(1, std::memory_order_release);
            continue;
        }

        // Ch9.2.2: No task available and stop requested - exit.
        if (st.stop_requested()) {
            break;
        }

        // Ch4.1.2: Wait on condition variable for new tasks.
        {
            std::unique_lock lock(queue_mutex_);
            cond_.wait_for(lock, std::chrono::milliseconds(10));
        }
    }
}

// Ch8.4.3: Work stealing implementation.
// Each worker first checks its own local queue, then the global queue,
// then attempts to steal from another random worker's local queue.
bool ThreadPool::get_task(size_t worker_idx, std::function<void()>& task,
                          stop_token st) {
    auto& local_q = workers_[worker_idx]->local_queue;

    // Ch8.4.2: Step 1 - Check own local queue (fastest path, no contention).
    if (auto t = local_q.try_pop()) {
        task = std::move(*t);
        return true;
    }

    // Ch9.1.11: Step 2 - Check global queue (moderate contention, single mutex).
    {
        std::lock_guard lock(queue_mutex_);
        if (auto t = global_queue_.try_pop()) {
            task = std::move(*t);
            return true;
        }
    }

    // Ch8.4.3: Step 3 - Work stealing from neighbor workers.
    // Random selection reduces contention vs sequential stealing (Ch8.4.4).
    // Only steal after pool is fully initialized (pool_ready_ flag, Ch9.2.1).
    if (workers_.size() > 1 && pool_ready_.load(std::memory_order_acquire)) {
        size_t start = rand() % workers_.size();
        for (size_t offset = 0; offset < workers_.size(); ++offset) {
            if (st.stop_requested()) return false;
            size_t victim = (start + offset) % workers_.size();
            if (victim == worker_idx) continue;

            // Steal one task from the victim's local queue.
            if (auto t = workers_[victim]->local_queue.try_pop()) {
                task = std::move(*t);
                return true;
            }
        }
    }

    return false;
}

// Ch8.4.2: Steal task - alias for get_task with steal-only semantics.
bool ThreadPool::steal_task(size_t worker_idx, std::function<void()>& task) {
    if (workers_.size() <= 1) return false;

    size_t start = rand() % workers_.size();
    for (size_t offset = 1; offset < workers_.size(); ++offset) {
        size_t victim = (start + offset) % workers_.size();
        if (victim == worker_idx) continue;
        if (auto t = workers_[victim]->local_queue.try_pop()) {
            task = std::move(*t);
            return true;
        }
    }
    return false;
}

// Ch9.1.8: Blocks until all submitted tasks complete (active_tasks_ == 0).
void ThreadPool::wait_for_tasks() {
    // Ch5.3.1: Spin-wait with minimal overhead and occasional yield.
    while (active_tasks_.load(std::memory_order_acquire) > 0) {
        std::this_thread::yield();
        // Ch9.2.4: Check stop to avoid infinite wait.
        if (stop_source_.stop_requested()) break;
    }
}

// Ch9.1.9: Approximate count of pending tasks (not perfectly accurate, Ch6.2.7).
size_t ThreadPool::pending_tasks() const {
    size_t count = 0;
    {
        std::lock_guard lock(queue_mutex_);
        count += global_queue_.size();
    }
    for (auto& w : workers_) {
        count += w->local_queue.size();
    }
    return count + active_tasks_.load(std::memory_order_acquire);
}

} // namespace task_scheduler
