// 03_threadsafe_queue_fine.cpp - 线程安全队列（双锁分离版本）
// 头尾各一把锁，push 和 pop 操作可并发执行
// 使用 unique_ptr 管理节点，RAII 无裸指针

#include <atomic>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

template <typename T>
class FineGrainedQueue {
    // 链表节点
    struct Node {
        std::unique_ptr<Node> next;
        T                     data;

        Node() = default;
        explicit Node(T value) : data(std::move(value)) {}
    };

public:
    FineGrainedQueue() {
        // 创建 dummy 头节点，使 head_ 和 tail_ 都指向它
        head_ = std::make_unique<Node>();
        tail_ = head_.get();
    }

    // 禁止拷贝
    FineGrainedQueue(const FineGrainedQueue&) = delete;
    FineGrainedQueue& operator=(const FineGrainedQueue&) = delete;

    // 入队：只锁 tail_mutex_
    void push(T value) {
        auto new_node = std::make_unique<Node>(std::move(value));
        Node* new_tail = new_node.get();

        {
            std::lock_guard<std::mutex> lock(tail_mutex_);
            tail_->next = std::move(new_node); // 链接新节点
            tail_       = new_tail;             // 更新尾指针
        }
        cond_var_.notify_one();
    }

    // 阻塞出队：需要同时操作 head_
    T wait_and_pop() {
        std::unique_lock<std::mutex> head_lock(head_mutex_);
        cond_var_.wait(head_lock, [this] { return head_.get() != get_tail(); });

        return pop_head();
    }

    // 非阻塞出队
    std::unique_ptr<T> try_pop() {
        std::lock_guard<std::mutex> head_lock(head_mutex_);
        if (head_.get() == get_tail()) {
            return nullptr; // 队列空
        }
        return std::make_unique<T>(pop_head());
    }

    bool empty() {
        std::lock_guard<std::mutex> head_lock(head_mutex_);
        return head_.get() == get_tail();
    }

private:
    // 安全获取 tail_（需要 tail_mutex_）
    Node* get_tail() {
        std::lock_guard<std::mutex> tail_lock(tail_mutex_);
        return tail_;
    }

    // 前提：已持有 head_mutex_
    T pop_head() {
        Node* old_head = head_.get();
        head_          = std::move(old_head->next); // 移动 unique_ptr

        T result = std::move(head_->data);
        // old_head 随 scope 结束自动释放
        return result;
    }

    std::mutex              head_mutex_;
    std::unique_ptr<Node>   head_;          // 头节点（dummy）
    std::mutex              tail_mutex_;
    Node*                   tail_;          // 尾指针（裸指针，生命周期由 head_ 管理）
    std::condition_variable cond_var_;
};

// ===== 测试 =====
int main() {
    std::cout << "=== FineGrainedQueue (双锁分离版本) ===\n";

    FineGrainedQueue<int> queue;

    const int kNumProducers    = 4;
    const int kNumConsumers    = 4;
    const int kItemsPerProducer = 25;
    const int kTotalItems       = kNumProducers * kItemsPerProducer;

    std::atomic<int> consumed{0};

    // 生产者
    std::vector<std::jthread> producers;
    for (int p = 0; p < kNumProducers; ++p) {
        producers.emplace_back([&, p]() {
            for (int i = 0; i < kItemsPerProducer; ++i) {
                queue.push(p * 100 + i);
            }
        });
    }

    // 消费者（使用 wait_and_pop 阻塞等待）
    std::vector<std::jthread> consumers;
    for (int c = 0; c < kNumConsumers; ++c) {
        consumers.emplace_back([&]() {
            while (consumed.fetch_add(1) < kTotalItems) {
                int val = queue.wait_and_pop();
                // 消费 val ...
            }
        });
    }

    // jthread 自动 join
    std::cout << "[Main] 全部生产-消费完成\n";
    return 0;
}
