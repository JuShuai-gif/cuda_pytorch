#pragma once
// Ch6.2+：基于优先级的线程安全队列
// 扩展了 Ch6.2 的基于锁的队列模式，增加了优先级排序。
// 使用 std::priority_queue 进行排序，并解决了优先级反转问题（Ch3.2.8）。
//
// 优先级反转避免策略（Ch3.2.7：死锁避免概念）：
// 1. 使用单个 mutex（简单，避免锁顺序问题）。
// 2. 所有操作由于 priority_queue 堆结构，复杂度为 O(log n)。
// 3. 批量操作将同优先级任务分组以减少上下文切换。
// 优先级线程安全队列
// 扩展基于锁的队列模式，加入优先级排序
// 单 mutex 设计避免优先级反转

#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <functional>
#include <vector>
#include <chrono>

namespace task_scheduler {

// AI/ML 推理任务的优先级级别（Ch6.3：为特定领域需求设计并发数据结构）。
// 任务优先级枚举：数值越小优先级越高
enum class TaskPriority : int {
    CRITICAL = 0, // 实时推理，延迟敏感
    HIGH     = 1, // 批量推理，流水线阶段
    NORMAL   = 2, // 定时任务，缓存更新
    LOW      = 3, // 后台清理，日志刷新
};

template <typename T>
class PriorityTaskQueue {
public:
    // Ch6.2.1：优先级队列按 (priority, sequence_number) 排序，以在同优先级内保持 FIFO——
    // 防止同一优先级带内的饥饿现象。
    // 优先级项：包含数据、优先级和序号
    struct PrioritizedItem {
        T data;
        TaskPriority priority;
        uint64_t sequence; // 同一优先级内 FIFO 的单调计数器

        // std::priority_queue 是最大堆。我们需要最小堆行为
        //（较小的优先级值 = 更高的紧急程度），因此反转比较。
        // 比较运算符：priority_queue 默认是最大堆，我们反转比较实现最小堆
        bool operator<(const PrioritizedItem& other) const {
            if (priority != other.priority) {
                return priority > other.priority; // 数值越大 = 优先级越低
            }
            return sequence > other.sequence; // 序号越小（越早）= 优先级越高
        }
    };

    PriorityTaskQueue() = default;

    // 禁止拷贝和移动（线程安全的队列不应被复制）
    PriorityTaskQueue(const PriorityTaskQueue&) = delete;
    PriorityTaskQueue& operator=(const PriorityTaskQueue&) = delete;
    PriorityTaskQueue(PriorityTaskQueue&&) = delete;
    PriorityTaskQueue& operator=(PriorityTaskQueue&&) = delete;

    // Ch6.2.1：带优先级的入队操作。
    // 推送任务：使用移动语义避免拷贝，通知等待线程
    void push(T item, TaskPriority priority) {
        {
            std::lock_guard lock(mutex_);
            queue_.push({std::move(item), priority, seq_++});
        }
        cond_.notify_one();
    }

    // Ch6.2.2：阻塞式获取最高优先级项。
    // 阻塞等待并获取最高优先级项
    PrioritizedItem wait_and_pop() {
        std::unique_lock lock(mutex_);
        cond_.wait(lock, [this] { return !queue_.empty(); });
        auto item = std::move(const_cast<PrioritizedItem&>(queue_.top()));
        queue_.pop();
        return item;
    }

    // Ch6.2.3：带超时的等待。
    // 带超时的阻塞等待，超时返回 nullopt
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

    // Ch6.2.4：非阻塞 try_pop——优先取最高优先级。
    // 非阻塞获取：空队列返回 nullopt
    std::optional<PrioritizedItem> try_pop() {
        std::lock_guard lock(mutex_);
        if (queue_.empty()) {
            return std::nullopt;
        }
        auto item = std::move(const_cast<PrioritizedItem&>(queue_.top()));
        queue_.pop();
        return std::make_optional(std::move(item));
    }

    // Ch6.2.5：弹出特定优先级带的所有项，用于批量处理。
    // 优先级反转避免：将同优先级任务分组在一起。
    // 按优先级批量获取：将同优先级任务归组处理
    template <typename OutputIt>
    size_t pop_by_priority(TaskPriority priority, OutputIt dest, size_t max_items) {
        std::lock_guard lock(mutex_);
        std::vector<PrioritizedItem> temp;
        size_t count = 0;

        // 遍历堆，取出匹配优先级的项，将不匹配的暂存
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
        // 重新插入不同优先级的项
        for (auto& item : temp) {
            queue_.push(std::move(item));
        }
        return count;
    }

    // 检查队列是否为空
    [[nodiscard]] bool empty() const {
        std::lock_guard lock(mutex_);
        return queue_.empty();
    }

    // 查询队列大小
    [[nodiscard]] size_t size() const {
        std::lock_guard lock(mutex_);
        return queue_.size();
    }

    // 通知所有等待线程（在关闭时很有用）
    void notify_all() { cond_.notify_all(); }

private:
    mutable std::mutex mutex_;
    std::priority_queue<PrioritizedItem> queue_;
    std::condition_variable cond_;
    uint64_t seq_{0}; // Ch6.2.7：序号计数器，用于同优先级内的 FIFO 排序
};

} // namespace task_scheduler
