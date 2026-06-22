// 02_producer_consumer_queue.cpp - 工业级有界阻塞队列
// BoundedBlockingQueue<T>：支持多生产者多消费者
// 使用两把条件变量：not_full_ 和 not_empty_

#include <condition_variable>
#include <iostream>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <vector>

template <typename T>
class BoundedBlockingQueue {
public:
    explicit BoundedBlockingQueue(size_t capacity)
        : capacity_(capacity) {}

    // 阻塞式入队：队列满时等待
    void push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        // 等待队列有空位
        not_full_.wait(lock, [this]() { return queue_.size() < capacity_; });

        queue_.push(std::move(item));

        // 通知等待取数据的消费者
        not_empty_.notify_one();
    }

    // 阻塞式出队：队列空时等待
    T pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [this]() { return !queue_.empty(); });

        T item = std::move(queue_.front());
        queue_.pop();

        // 通知等待放数据的生产者
        not_full_.notify_one();
        return item;
    }

    // 非阻塞尝试出队
    std::optional<T> try_pop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return std::nullopt;

        T item = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return item;
    }

    // 带超时的出队
    template <typename Rep, typename Period>
    std::optional<T> pop_for(const std::chrono::duration<Rep, Period>& timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!not_empty_.wait_for(lock, timeout, [this]() { return !queue_.empty(); })) {
            return std::nullopt;
        }

        T item = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return item;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

private:
    mutable std::mutex      mutex_;
    std::condition_variable not_full_;
    std::condition_variable not_empty_;
    std::queue<T>           queue_;
    size_t                  capacity_;
};

int main() {
    BoundedBlockingQueue<int> queue(5); // 容量 5

    // 4 个生产者：各生产 20 个数据
    std::vector<std::jthread> producers;
    for (int p = 0; p < 4; ++p) {
        producers.emplace_back([&queue, p]() {
            for (int i = 0; i < 20; ++i) {
                int val = p * 100 + i;
                queue.push(val);
                std::cout << "[P" << p << "] 生产 " << val << "\n";
            }
        });
    }

    // 2 个消费者：各消费 40 个数据
    std::vector<std::jthread> consumers;
    for (int c = 0; c < 2; ++c) {
        consumers.emplace_back([&queue, c]() {
            for (int i = 0; i < 40; ++i) {
                int val = queue.pop();
                std::cout << "[C" << c << "] 消费 " << val << "\n";
            }
        });
    }

    // jthread 自动 join
    std::cout << "[Main] 队列最终大小: " << queue.size() << "\n";
    return 0;
}
