#pragma once

#include <array>
#include <atomic>
#include <mutex>
#include <condition_variable>

// ============================================================================
// DoubleBuffer: 线程安全的乒乓缓冲区模板
//
// 生产者写入一个槽位，消费者从另一个槽位读取。
// 两者完成操作后进行交换。
// ============================================================================
template <typename T>
class DoubleBuffer {
public:
    DoubleBuffer() : produce_idx_(0), consume_idx_(0),
                     data_ready_(false), producer_done_(false) {
    }

    // 生产者：获取可写槽位，返回其索引（0 或 1）
    int producer_acquire() {
        std::unique_lock<std::mutex> lk(mtx_);
        cv_producer_.wait(lk, [this]() { return !data_ready_ || producer_done_; });
        return produce_idx_.load(std::memory_order_acquire);
    }

    // 生产者：通知数据已就绪，将槽位交给消费者
    void producer_release(int idx) {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            data_ready_ = true;
            consume_idx_.store(idx, std::memory_order_release);
        }
        cv_consumer_.notify_one();
    }

    // 生产者：消费者完成后切换到另一个缓冲区
    void producer_swap() {
        std::unique_lock<std::mutex> lk(mtx_);
        cv_producer_.wait(lk, [this]() { return !data_ready_ || producer_done_; });
        int curr = produce_idx_.load(std::memory_order_acquire);
        produce_idx_.store(1 - curr, std::memory_order_release);
    }

    // 生产者：通知不再产生数据
    void producer_done() {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            producer_done_ = true;
        }
        cv_consumer_.notify_all();
    }

    // 消费者：获取可读槽位，返回其索引
    int consumer_acquire() {
        std::unique_lock<std::mutex> lk(mtx_);
        cv_consumer_.wait(lk, [this]() { return data_ready_ || producer_done_; });
        if (producer_done_ && !data_ready_) return -1;
        return consume_idx_.load(std::memory_order_acquire);
    }

    // 消费者：通知数据已被消费
    void consumer_release(int idx) {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            data_ready_ = false;
            consume_idx_.store(1 - idx, std::memory_order_release);
        }
        cv_producer_.notify_one();
    }

    // 访问底层缓冲区槽位
    T &buffer(int idx) {
        return buffers_[idx];
    }
    const T &buffer(int idx) const {
        return buffers_[idx];
    }

private:
    std::array<T, 2> buffers_;
    std::atomic<int> produce_idx_;
    std::atomic<int> consume_idx_;
    std::mutex mtx_;
    std::condition_variable cv_producer_;
    std::condition_variable cv_consumer_;
    bool data_ready_;
    bool producer_done_;
};
