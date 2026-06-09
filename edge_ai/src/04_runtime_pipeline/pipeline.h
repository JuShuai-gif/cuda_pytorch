#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>

// ============================================================================
// BoundedQueue: 线程安全的固定容量队列，用于流水线各阶段
// ============================================================================
class BoundedQueue {
public:
    explicit BoundedQueue(int cap);

    void push(int val);

    bool pop(int &val);

    void set_done();

private:
    std::queue<int> q_;
    std::mutex mtx_;
    std::condition_variable cv_producer_;
    std::condition_variable cv_consumer_;
    int capacity_;
    bool done_;
};
