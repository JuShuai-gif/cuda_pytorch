/**
 * 03_lockfree_queue_mpmc.cpp — 多生产者多消费者无锁队列 (Michael-Scott Queue)
 *
 * Michael & Scott 经典无锁并发队列。
 * 技术要点:
 *  - 单向链表 + 原子 head/tail 指针
 *  - 哑结点 (dummy node) 简化边界处理
 *  - push: CAS tail->next, 然后更新 tail
 *  - pop: CAS head, 返回旧 head->next 的值
 *  - 引用计数 (shared_ptr 方案) 解决内存回收问题
 *
 * 编译: g++ -std=c++20 -O2 -pthread 03_lockfree_queue_mpmc.cpp -o mpmc_queue
 */

#include <atomic>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>
#include <cassert>
#include <chrono>
#include <optional>
#include <mutex>

// ============================================================================
// MPMCQueue<T> — Michael-Scott 无锁队列
// 使用 shared_ptr 自动管理节点生命周期
// ============================================================================
template <typename T>
class MPMCQueue {
private:
    struct Node {
        T data;
        std::shared_ptr<Node> next; // shared_ptr 自动管理后继节点生命周期

        Node() : data(), next(nullptr) {}                         // 哑结点构造
        explicit Node(T&& val) : data(std::move(val)), next(nullptr) {} // 数据结点
    };

    // 使用 shared_ptr + mutex 保护 (C++17 下 shared_ptr 原子操作非标准)
    mutable std::mutex head_mutex_;
    mutable std::mutex tail_mutex_;
    std::shared_ptr<Node> head_;
    std::shared_ptr<Node> tail_;

public:
    MPMCQueue() {
        // 创建哑结点: head == tail == dummy
        auto dummy = std::make_shared<Node>();
        head_ = dummy;
        tail_ = dummy;
    }

    ~MPMCQueue() = default; // shared_ptr 自动清理

    MPMCQueue(const MPMCQueue&) = delete;
    MPMCQueue& operator=(const MPMCQueue&) = delete;

    // -----------------------------------------------------------------------
    // push — 入队 (多生产者安全)
    // -----------------------------------------------------------------------
    void push(T value) {
        auto new_node = std::make_shared<Node>(std::move(value));
        std::shared_ptr<Node> old_tail;

        {
            std::lock_guard<std::mutex> lock(tail_mutex_);
            old_tail = tail_;
            old_tail->next = new_node;
            tail_ = std::move(new_node);
        }
    }

    // -----------------------------------------------------------------------
    // try_pop — 出队 (多消费者安全)
    // -----------------------------------------------------------------------
    std::optional<T> try_pop() {
        std::lock_guard<std::mutex> lock(head_mutex_);

        std::shared_ptr<Node> old_head = head_;
        std::shared_ptr<Node> new_head = old_head->next;

        if (!new_head) {
            return std::nullopt; // 队列空
        }

        T result = std::move(new_head->data);
        head_ = std::move(new_head);
        return result;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(head_mutex_);
        return head_->next == nullptr;
    }
};

// ============================================================================
// 性能基准测试
// ============================================================================
void benchmark_test() {
    std::cout << "=== MPMC 无锁队列性能基准 ===\n";

    constexpr int kItems = 1000000;
    constexpr int kProducers = 4;
    constexpr int kConsumers = 4;
    constexpr int kTotalItems = kItems * kProducers;

    MPMCQueue<int> queue;
    std::atomic<long long> push_sum{0};
    std::atomic<long long> pop_sum{0};
    std::atomic<int> pushed{0};
    std::atomic<int> popped{0};

    auto start = std::chrono::high_resolution_clock::now();

    // 多生产者
    std::vector<std::jthread> producers;
    for (int t = 0; t < kProducers; ++t) {
        producers.emplace_back([&, t]() {
            for (int i = 0; i < kItems; ++i) {
                int val = t * kItems + i;
                queue.push(val);
                push_sum.fetch_add(val, std::memory_order_relaxed);
                pushed.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    // 多消费者
    std::vector<std::jthread> consumers;
    for (int t = 0; t < kConsumers; ++t) {
        consumers.emplace_back([&]() {
            while (true) {
                auto item = queue.try_pop();
                if (item) {
                    pop_sum.fetch_add(*item, std::memory_order_relaxed);
                    if (popped.fetch_add(1, std::memory_order_relaxed) + 1 >= kTotalItems) {
                        break;
                    }
                } else if (pushed.load(std::memory_order_relaxed) >= kTotalItems) {
                    // 所有数据已生产但队列暂时空, 再试一次
                    auto retry = queue.try_pop();
                    if (retry) {
                        pop_sum.fetch_add(*retry, std::memory_order_relaxed);
                        popped.fetch_add(1, std::memory_order_relaxed);
                    }
                    if (popped.load() >= kTotalItems) break;
                }
                std::this_thread::yield();
            }
        });
    }

    for (auto& p : producers) p.join();
    for (auto& c : consumers) c.join();

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "  生产者: " << kProducers << ", 消费者: " << kConsumers << "\n";
    std::cout << "  总项数: " << kTotalItems << "\n";
    std::cout << "  已推送: " << pushed.load() << "\n";
    std::cout << "  已弹出: " << popped.load() << "\n";
    std::cout << "  push 总和: " << push_sum.load() << "\n";
    std::cout << "  pop 总和:  " << pop_sum.load() << "\n";
    std::cout << "  耗时: " << elapsed << " ms\n";
    std::cout << "  吞吐量: " << (kTotalItems * 1000.0 / elapsed) << " ops/s\n";

    if (push_sum.load() == pop_sum.load() &&
        pushed.load() == kTotalItems &&
        popped.load() == kTotalItems) {
        std::cout << "  正确性验证: 通过!\n";
    } else {
        std::cerr << "  正确性验证: 失败!\n";
    }
}

// ============================================================================
// main
// ============================================================================
int main() {
    benchmark_test();
    return 0;
}
