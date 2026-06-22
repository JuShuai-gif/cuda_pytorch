#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>

template <typename T>
class StageQueue {
public:
    void push(const T &item) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(item);
        cv_.notify_one();
    }

    bool pop(T &item) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]() { return !queue_.empty(); });
        item = queue_.front();
        queue_.pop();
        return true;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<T> queue_;
};
