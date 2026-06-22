#include "pipeline.h"

BoundedQueue::BoundedQueue(int cap) : capacity_(cap), done_(false) {
}

void BoundedQueue::push(int val) {
    std::unique_lock<std::mutex> lk(mtx_);
    cv_producer_.wait(lk, [this]() { return (int)q_.size() < capacity_; });
    q_.push(val);
    cv_consumer_.notify_one();
}

bool BoundedQueue::pop(int &val) {
    std::unique_lock<std::mutex> lk(mtx_);
    cv_consumer_.wait(lk, [this]() { return !q_.empty() || done_; });
    if (q_.empty() && done_) return false;
    val = q_.front();
    q_.pop();
    cv_producer_.notify_one();
    return true;
}

void BoundedQueue::set_done() {
    std::lock_guard<std::mutex> lk(mtx_);
    done_ = true;
    cv_consumer_.notify_all();
}
