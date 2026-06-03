#pragma once
// Chapter 6.2+: Priority-based Thread-Safe Queue
// Extends the lock-based queue pattern (Ch6.2) with priority ordering.
// Uses std::priority_queue for ordering and addresses priority inversion (Ch3.2.8).
//
// Priority Inversion Avoidance Strategy (Ch3.2.7: deadlock avoidance concepts):
// 1. Uses a single mutex (simple, avoids lock ordering issues).
// 2. All operations are O(log n) due to priority_queue heap structure.
// 3. Batch operations group same-priority tasks to reduce context switches.

#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <functional>
#include <vector>
#include <chrono>

namespace task_scheduler {

// Priority levels for AI/ML inference tasks (Ch6.3: designing concurrent data structures
// for specific domain requirements).
enum class TaskPriority : int {
    CRITICAL = 0, // Real-time inference, latency-sensitive
    HIGH     = 1, // Batch inference, pipeline stage
    NORMAL   = 2, // Periodic tasks, cache updates
    LOW      = 3, // Background cleanup, logging flush
};

template <typename T>
class PriorityTaskQueue {
public:
    // Ch6.2.1: Priority queue orders by (priority, sequence_number) to maintain FIFO
    // within same priority level - prevents starvation within a priority band.
    struct PrioritizedItem {
        T data;
        TaskPriority priority;
        uint64_t sequence; // Monotonic counter for FIFO within same priority

        // std::priority_queue is a max-heap. We want min-heap behavior
        // (smaller priority value = higher urgency), so invert comparison.
        bool operator<(const PrioritizedItem& other) const {
            if (priority != other.priority) {
                return priority > other.priority; // Higher numeric = lower priority
            }
            return sequence > other.sequence; // Earlier sequence = higher priority
        }
    };

    PriorityTaskQueue() = default;

    PriorityTaskQueue(const PriorityTaskQueue&) = delete;
    PriorityTaskQueue& operator=(const PriorityTaskQueue&) = delete;
    PriorityTaskQueue(PriorityTaskQueue&&) = delete;
    PriorityTaskQueue& operator=(PriorityTaskQueue&&) = delete;

    // Ch6.2.1: Push with priority.
    void push(T item, TaskPriority priority) {
        {
            std::lock_guard lock(mutex_);
            queue_.push({std::move(item), priority, seq_++});
        }
        cond_.notify_one();
    }

    // Ch6.2.2: Blocking pop of the highest priority item.
    PrioritizedItem wait_and_pop() {
        std::unique_lock lock(mutex_);
        cond_.wait(lock, [this] { return !queue_.empty(); });
        auto item = std::move(const_cast<PrioritizedItem&>(queue_.top()));
        queue_.pop();
        return item;
    }

    // Ch6.2.3: Wait with timeout.
    template <typename Rep, typename Period>
    std::optional<PrioritizedItem> wait_and_pop_for(
        const std::chrono::duration<Rep, Period>& timeout) {
        std::unique_lock lock(mutex_);
        if (cond_.wait_for(lock, timeout, [this] { return !queue_.empty(); })) {
            auto item = std::move(const_cast<PrioritizedItem&>(queue_.top()));
            queue_.pop();
            return std::make_optional(std::move(item));
        }
        return std::nullopt;
    }

    // Ch6.2.4: Non-blocking try_pop - highest priority first.
    std::optional<PrioritizedItem> try_pop() {
        std::lock_guard lock(mutex_);
        if (queue_.empty()) {
            return std::nullopt;
        }
        auto item = std::move(const_cast<PrioritizedItem&>(queue_.top()));
        queue_.pop();
        return std::make_optional(std::move(item));
    }

    // Ch6.2.5: Pop all items of a specific priority band for batch processing.
    // Priority inversion avoidance: groups same-priority tasks together.
    template <typename OutputIt>
    size_t pop_by_priority(TaskPriority priority, OutputIt dest, size_t max_items) {
        std::lock_guard lock(mutex_);
        std::vector<PrioritizedItem> temp;
        size_t count = 0;

        while (count < max_items && !queue_.empty()) {
            auto& top = const_cast<PrioritizedItem&>(queue_.top());
            if (top.priority != priority) {
                temp.push_back(std::move(top));
                queue_.pop();
                continue;
            }
            *dest++ = std::move(top);
            queue_.pop();
            ++count;
        }
        // Re-insert items of different priorities
        for (auto& item : temp) {
            queue_.push(std::move(item));
        }
        return count;
    }

    [[nodiscard]] bool empty() const {
        std::lock_guard lock(mutex_);
        return queue_.empty();
    }

    [[nodiscard]] size_t size() const {
        std::lock_guard lock(mutex_);
        return queue_.size();
    }

    void notify_all() { cond_.notify_all(); }

private:
    mutable std::mutex mutex_;
    std::priority_queue<PrioritizedItem> queue_;
    std::condition_variable cond_;
    uint64_t seq_{0}; // Ch6.2.7: sequence counter for FIFO ordering within priority
};

} // namespace task_scheduler
