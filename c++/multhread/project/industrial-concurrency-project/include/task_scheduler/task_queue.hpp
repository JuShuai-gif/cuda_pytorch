#pragma once
// Ch6.2：基于锁的线程安全队列
// 一个支持多生产者多消费者（MPMC）的线程安全队列。
// 使用 std::mutex + std::condition_variable（Ch4.1）。
// 支持阻塞（wait_and_pop）和非阻塞（try_pop）操作。
// 基于锁的线程安全队列
// MPMC（多生产者多消费者）模式
// 使用 std::mutex + std::condition_variable

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

    // 不可拷贝（Ch3.2.1：通过删除防止数据竞争）。
    // 禁止拷贝和移动：避免数据竞争
    TaskQueue(const TaskQueue&) = delete;
    TaskQueue& operator=(const TaskQueue&) = delete;
    TaskQueue(TaskQueue&&) = delete;
    TaskQueue& operator=(TaskQueue&&) = delete;

    // Ch6.2.1：入队——获取 mutex，推送到底层队列，通知一个等待者。
    // 内存序：mutex 的 lock 提供 acquire 语义，unlock 提供 release 语义。
    // 推送元素：加锁后移动插入，锁外通知避免"急上加急"
    void push(T item) {
        {
            // Ch3.2.3：std::lock_guard 用于异常安全的锁定（RAII）。
            std::lock_guard lock(mutex_);
            queue_.push(std::move(item));
        }
        // Ch4.1.1：在锁外通知，以避免"急上加急"问题（通知的线程立即被阻塞）。
        cond_.notify_one();
    }

    // Ch6.2.2：阻塞式出队——等待直到有可用项。
    // 使用 std::unique_lock（Ch3.2.6），因为 condition_variable 需要解锁能力。
    // 阻塞等待并出队
    T wait_and_pop() {
        std::unique_lock lock(mutex_);
        // Ch4.1.1：condition_variable::wait 原子地解锁 mutex 并休眠，
        // 然后在返回前重新获取 mutex。
        cond_.wait(lock, [this] { return !queue_.empty(); });
        T item = std::move(queue_.front());
        queue_.pop();
        return item;
    }

    // Ch6.2.3：带超时的等待（Ch4.1.2：带谓词的 wait_for）。
    // 如果超时而未收到项，返回 std::nullopt。
    // 带超时的阻塞等待
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

    // Ch6.2.4：非阻塞 try_pop——立即返回。
    // 如果队列为空返回 std::nullopt。
    // 非阻塞出队：立即返回，空队列返回 nullopt
    std::optional<T> try_pop() {
        std::lock_guard lock(mutex_);
        if (queue_.empty()) {
            return std::nullopt;
        }
        T item = std::move(queue_.front());
        queue_.pop();
        return std::make_optional(std::move(item));
    }

    // Ch6.2.5：批量操作以提高效率。
    // 批量出队：一次获取最多 max_items 个，减少锁开销
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

    // Ch6.2.6：工具查询。
    // 检查是否为空
    [[nodiscard]] bool empty() const {
        std::lock_guard lock(mutex_);
        return queue_.empty();
    }

    // 查询大小
    [[nodiscard]] size_t size() const {
        std::lock_guard lock(mutex_);
        return queue_.size();
    }

    // 唤醒所有等待的消费者（在关闭时很有用，Ch9.2.1）。
    // 通知所有等待线程
    void notify_all() {
        cond_.notify_all();
    }

private:
    // Ch3.2.8：mutable mutex 用于需要加锁的 const 限定方法。
    // mutable 允许 const 方法中加锁
    mutable std::mutex mutex_;
    std::queue<T> queue_;
    std::condition_variable cond_; // Ch4.1.2：等待 mutex_ 上的非空谓词条件
};

} // namespace task_scheduler
