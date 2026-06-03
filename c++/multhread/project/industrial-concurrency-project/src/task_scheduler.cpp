// Chapter 8 & 9: Task Scheduler Implementation
// Implements dispatching logic and periodic task scheduling.

#include "task_scheduler/task_scheduler.hpp"
#include <chrono>
#include "task_scheduler/format_compat.hpp"

namespace task_scheduler {

TaskScheduler::TaskScheduler(size_t num_threads, size_t cache_size)
    : pool_(std::make_unique<ThreadPool>(num_threads))
    , cache_(cache_size) {
    Logger::instance().info(TS_FORMAT(
        "TaskScheduler initialized with {} threads, cache size {}",
        pool_->worker_count(), cache_size));
}

TaskScheduler::~TaskScheduler() {
    shutdown();
}

// Ch8.4.1: Dispatch pending tasks from priority queue to thread pool.
// Called after each submit to kick off execution.
void TaskScheduler::dispatch_pending() {
    // Ch8.4.2: Drain the priority queue into the thread pool.
    // Dispatch tasks as long as there are tasks in the queue and
    // distribute them round-robin across workers for load balancing.
    size_t dispatched = 0;
    while (true) {
        auto item = priority_queue_.try_pop();
        if (!item) break;

        // Ch9.1.2: Distribute to a specific worker for load balancing.
        // Use round-robin across workers (Ch8.4.5: even distribution).
        pool_->submit_to_local(
            dispatched % pool_->worker_count(),
            std::move(item->data));
        ++dispatched;
    }
}

// Ch8.5.1: Periodic task scheduling.
// Uses a background thread with timed waits (Ch4.1.3).
stop_source TaskScheduler::schedule_periodic(
    PeriodicCallback callback,
    std::chrono::milliseconds interval,
    TaskPriority priority,
    std::string_view name) {

    stop_source periodic_stop;
    auto st = periodic_stop.get_token();
    std::string task_name(name);

    // Ch9.1.3: Launch a dedicated thread for the periodic task.
    std::jthread([this, cb = std::move(callback), interval, priority,
                  task_name, st = std::move(st)]() mutable {
        Logger::instance().debug(TS_FORMAT(
            "Periodic task '{}' started (interval: {}ms)",
            task_name, interval.count()));

        while (!st.stop_requested()) {
            // Ch8.5.2: Submit the periodic callback as a high-priority task.
            // We don't wait for it - fire and forget.
            if (!scheduler_stop_source_.stop_requested()) {
                submit(priority, task_name, cb);
            }

            // Ch4.1.3: Timed wait with stop check.
            // Avoids busy-waiting while allowing responsive stop.
            if (!st.wait_for(interval)) {
                // Timeout expired normally, continue loop
                continue;
            }
            break; // Stop was requested
        }

        Logger::instance().debug(
            TS_FORMAT("Periodic task '{}' stopped", task_name));
    }).detach(); // Ch2.3: Detach the thread - it cleans itself up via stop_token.

    return periodic_stop;
}

void TaskScheduler::shutdown() {
    if (!running_.exchange(false)) return; // Already stopped (Ch5.3.2: atomic flag)

    Logger::instance().info("TaskScheduler shutting down...");

    // Ch9.2.1: Signal all components to stop.
    scheduler_stop_source_.request_stop();

    // Ch9.2.3: Notify priority queue waiters.
    priority_queue_.notify_all();

    // Ch9.1.7: Pool destructor handles thread join.
    pool_->shutdown();

    Logger::instance().info("TaskScheduler shutdown complete");
}

bool TaskScheduler::is_stopping() const {
    return scheduler_stop_source_.stop_requested();
}

} // namespace task_scheduler
