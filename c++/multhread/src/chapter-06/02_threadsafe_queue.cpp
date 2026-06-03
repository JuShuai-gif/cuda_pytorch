// 02_threadsafe_queue.cpp - 线程安全队列（单锁版本）
// ThreadSafeQueue<T>：一把锁保护全部操作
// 支持多生产者多消费者，提供阻塞和非阻塞接口

#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <vector>

template <typename T>
class ThreadSafeQueue {
public:
    ThreadSafeQueue() = default;

    // 禁止拷贝
    ThreadSafeQueue(const ThreadSafeQueue&) = delete;
    ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;

    // 阻塞入队（UML bound queue，但此实现无限容量）
    void push(T value) {
        std::lock_guard<std::mutex> lock(mutex_);
        data_.push(std::move(value));
        cond_var_.notify_one();
    }

    // 阻塞出队（wait_and_pop）
    T wait_and_pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_var_.wait(lock, [this] { return !data_.empty(); });
        T value = std::move(data_.front());
        data_.pop();
        return value;
    }

    // 非阻塞尝试出队
    std::optional<T> try_pop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (data_.empty()) return std::nullopt;
        T value = std::move(data_.front());
        data_.pop();
        return value;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return data_.empty();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return data_.size();
    }

private:
    mutable std::mutex      mutex_;
    std::condition_variable cond_var_;
    std::queue<T>           data_;
};

// ===== 测试 =====
int main() {
    std::cout << "=== ThreadSafeQueue (单锁版本) ===\n";

    ThreadSafeQueue<int> queue;

    const int kNumProducers = 3;
    const int kNumConsumers = 2;
    const int kItemsPerProducer = 20;
    const int kTotalItems = kNumProducers * kItemsPerProducer;

    std::atomic<int> consumed{0};

    // 生产者
    std::vector<std::jthread> producers;
    for (int p = 0; p < kNumProducers; ++p) {
        producers.emplace_back([&, p]() {
            for (int i = 0; i < kItemsPerProducer; ++i) {
                int val = p * 1000 + i;
                queue.push(val);
                std::cout << "[P" << p << "] -> " << val << "\n";
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        });
    }

    // 消费者
    std::vector<std::jthread> consumers;
    for (int c = 0; c < kNumConsumers; ++c) {
        consumers.emplace_back([&]() {
            while (consumed.load() < kTotalItems) {
                auto val = queue.try_pop();
                if (val) {
                    std::cout << "[C" << c << "] <- " << *val << "\n";
                    consumed.fetch_add(1);
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
            }
        });
    }

    // jthread 自动 join
    std::cout << "[Main] 全部消费完毕，最终队列大小: " << queue.size() << "\n";
    return 0;
}
