// 01_pipeline.cpp — Pipeline 模式实现
// 演示: 三阶段流水线 (生成 → 处理 → 消费)

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <syncstream>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

// ===== 有界阻塞队列 (Pipeline 阶段间通信) =====
template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t capacity)
        : capacity_(capacity) {}

    void push(T item) {
        std::unique_lock lock(mtx_);
        not_full_.wait(lock, [this] { return queue_.size() < capacity_; });
        queue_.push(std::move(item));
        not_empty_.notify_one();
    }

    T pop() {
        std::unique_lock lock(mtx_);
        not_empty_.wait(lock, [this] { return !queue_.empty(); });
        T item = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return item;
    }

    bool try_pop(T& item) {
        std::lock_guard lock(mtx_);
        if (queue_.empty()) return false;
        item = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return true;
    }

    void close() {
        closed_ = true;
        not_empty_.notify_all();
    }

    bool is_closed_and_empty() {
        std::lock_guard lock(mtx_);
        return closed_ && queue_.empty();
    }

private:
    size_t capacity_;
    std::queue<T> queue_;
    std::mutex mtx_;
    std::condition_variable not_full_;
    std::condition_variable not_empty_;
    bool closed_ = false;
};

// ===== Pipeline 实现 =====
void demo_pipeline() {
    std::cout << "=== 三阶段 Pipeline ===\n\n";

    BoundedQueue<int> stage1_to_2{4};
    BoundedQueue<int> stage2_to_3{4};

    std::atomic<long long> items_generated{0};
    std::atomic<long long> items_processed{0};
    std::atomic<long long> items_consumed{0};

    const int kTotalItems = 20;

    // Stage 1: 生成数据
    std::jthread generator([&](std::stop_token stoken) {
        for (int i = 0; i < kTotalItems && !stoken.stop_requested(); ++i) {
            std::this_thread::sleep_for(20ms); // 模拟生成耗时
            stage1_to_2.push(i);
            items_generated.fetch_add(1);
            std::osyncstream(std::cout)
                << "  [Stage1] 生成: " << i << "\n";
        }
        stage1_to_2.close();
    });

    // Stage 2: 处理数据
    std::jthread processor([&](std::stop_token stoken) {
        while (!stoken.stop_requested()) {
            int item;
            if (!stage1_to_2.try_pop(item)) {
                if (stage1_to_2.is_closed_and_empty()) break;
                std::this_thread::sleep_for(1ms);
                continue;
            }
            std::this_thread::sleep_for(30ms); // 模拟处理耗时（瓶颈）
            int result = item * item;
            stage2_to_3.push(result);
            items_processed.fetch_add(1);
            std::osyncstream(std::cout)
                << "  [Stage2] 处理: " << item << " -> " << result << "\n";
        }
        stage2_to_3.close();
    });

    // Stage 3: 消费/输出
    std::jthread consumer([&](std::stop_token stoken) {
        while (!stoken.stop_requested()) {
            int item;
            if (!stage2_to_3.try_pop(item)) {
                if (stage2_to_3.is_closed_and_empty()) break;
                std::this_thread::sleep_for(1ms);
                continue;
            }
            items_consumed.fetch_add(1);
            std::osyncstream(std::cout)
                << "  [Stage3] 消费: " << item << "\n";
        }
    });

    generator.join();
    processor.join();
    consumer.join();

    std::cout << "\n  生成: " << items_generated.load()
              << " | 处理: " << items_processed.load()
              << " | 消费: " << items_consumed.load() << "\n";
    std::cout << "  瓶颈在 Stage2 (30ms per item)，"
              << "吞吐量由它决定\n";
}

// ===== 无锁 Pipeline (使用 atomics 传递) =====
void demo_lockfree_pipeline_stages() {
    std::cout << "\n=== 无锁阶段间传递 ===\n";

    // 简单场景: 多个 producer 更新同一个状态，单个 consumer 读取
    std::atomic<int> shared_state{-1};
    std::atomic<bool> done{false};

    const int kProducers = 4;
    std::vector<std::jthread> producers;
    for (int i = 0; i < kProducers; ++i) {
        producers.emplace_back([&, i]() {
            std::this_thread::sleep_for(10ms * (i + 1));
            shared_state.store(i, std::memory_order_release);
            std::osyncstream(std::cout)
                << "  Producer " << i << " 写入状态\n";
        });
    }

    std::jthread observer([&](std::stop_token stoken) {
        while (!done.load(std::memory_order_acquire) &&
               !stoken.stop_requested()) {
            int state = shared_state.load(std::memory_order_acquire);
            if (state >= 0) {
                std::osyncstream(std::cout)
                    << "  Observer 读取: " << state << "\n";
            }
            std::this_thread::sleep_for(5ms);
        }
    });

    for (auto& p : producers) p.join();
    std::this_thread::sleep_for(50ms);
    done.store(true);
    observer.join();
}

int main() {
    demo_pipeline();
    demo_lockfree_pipeline_stages();

    std::cout << "\nPipeline 模式: 阶段间用有界队列解耦，吞吐量由瓶颈决定。\n";
    return 0;
}
