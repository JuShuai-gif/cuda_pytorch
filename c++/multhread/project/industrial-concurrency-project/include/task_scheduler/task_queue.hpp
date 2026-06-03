#pragma once
// Chapter 6.2: Lock-based Thread-Safe Queue
// A thread-safe queue supporting multiple producers and multiple consumers (MPMC).
// Uses std::mutex + std::condition_variable (Ch4.1).
// Supports both blocking (wait_and_pop) and non-blocking (try_pop) operations.

#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <utility>
#include <chrono>

namespace task_scheduler {

template <typename T>
class TaskQueue {
public:
    TaskQueue() = default;

    // Non-copyable (Ch3.2.1: prevent data race via deletion).
    TaskQueue(const TaskQueue&) = delete;
    TaskQueue& operator=(const TaskQueue&) = delete;
    TaskQueue(TaskQueue&&) = delete;
    TaskQueue& operator=(TaskQueue&&) = delete;

    // Ch6.2.1: Push - acquire mutex, push to underlying queue, notify one waiter.
    // Memory ordering: mutex lock provides acquire, unlock provides release.
    void push(T item) {
        {
            // Ch3.2.3: std::lock_guard for exception-safe locking (RAII).
            std::lock_guard lock(mutex_);
            queue_.push(std::move(item));
        }
        // Ch4.1.1: Notify outside the lock to avoid "hurry up and wait" problem.
        cond_.notify_one();
    }

    // Ch6.2.2: Blocking pop - waits until an item is available.
    // Uses std::unique_lock (Ch3.2.6) because condition_variable needs unlock capability.
    T wait_and_pop() {
        std::unique_lock lock(mutex_);
        // Ch4.1.1: condition_variable::wait atomically unlocks mutex and sleeps,
        // then re-acquires mutex before returning.
        cond_.wait(lock, [this] { return !queue_.empty(); });
        T item = std::move(queue_.front());
        queue_.pop();
        return item;
    }

    // Ch6.2.3: Wait with timeout (Ch4.1.2: wait_for with predicate).
    // Returns std::nullopt if timeout expires before an item arrives.
    template <typename Rep, typename Period>
    std::optional<T> wait_and_pop_for(const std::chrono::duration<Rep, Period>& timeout) {
        std::unique_lock lock(mutex_);
        if (cond_.wait_for(lock, timeout, [this] { return !queue_.empty(); })) {
            T item = std::move(queue_.front());
            queue_.pop();
            return std::make_optional(std::move(item));
        }
        return std::nullopt;
    }

    // Ch6.2.4: Non-blocking try_pop - returns immediately.
    // Returns std::nullopt if queue is empty.
    std::optional<T> try_pop() {
        std::lock_guard lock(mutex_);
        if (queue_.empty()) {
            return std::nullopt;
        }
        T item = std::move(queue_.front());
        queue_.pop();
        return std::make_optional(std::move(item));
    }

    // Ch6.2.5: Bulk operations for efficiency.
    template <typename OutputIt>
    size_t try_pop_bulk(OutputIt dest, size_t max_items) {
        std::lock_guard lock(mutex_);
        size_t count = 0;
        while (count < max_items && !queue_.empty()) {
            *dest++ = std::move(queue_.front());
            queue_.pop();
            ++count;
        }
        return count;
    }

    // Ch6.2.6: Utility queries.
    [[nodiscard]] bool empty() const {
        std::lock_guard lock(mutex_);
        return queue_.empty();
    }

    [[nodiscard]] size_t size() const {
        std::lock_guard lock(mutex_);
        return queue_.size();
    }

    // Wake all waiting consumers (useful during shutdown, Ch9.2.1).
    void notify_all() {
        cond_.notify_all();
    }

private:
    // Ch3.2.8: mutable mutex for const-qualified methods that need locking.
    mutable std::mutex mutex_;
    std::queue<T> queue_;
    std::condition_variable cond_; // Ch4.1.2: waits on mutex_ for non-empty predicate
};

} // namespace task_scheduler
