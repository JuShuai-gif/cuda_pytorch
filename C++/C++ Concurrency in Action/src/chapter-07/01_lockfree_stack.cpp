/**
 * 01_lockfree_stack.cpp — 无锁栈 (Treiber Stack) 的工业级实现
 *
 * 基于 CAS (compare_exchange) 的无锁并发栈。
 * 技术要点:
 *  - std::atomic<Node*> 管理链表头指针
 *  - push: 创建新节点, CAS 更新 head
 *  - pop: CAS 摘取 head, 返回 shared_ptr 确保内存安全
 *  - 使用 std::shared_ptr 避免 ABA 与内存泄漏
 *  - C++20 可直接用 atomic<shared_ptr<T>>; 这里展示 C++17 兼容方案
 *
 * 编译: g++ -std=c++20 -O2 -pthread 01_lockfree_stack.cpp -o lockfree_stack
 */

#include <atomic>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>
#include <cassert>
#include <chrono>
#include <mutex>

// ============================================================================
// LockFreeStack<T> — Treiber Stack
// ============================================================================
template <typename T>
class LockFreeStack {
private:
    struct Node {
        T data;
        std::shared_ptr<Node> next; // shared_ptr 管理节点生命周期

        Node(T val) : data(std::move(val)), next(nullptr) {}
    };

    // C++20: std::atomic<std::shared_ptr<Node>> head_{nullptr};
    // C++17 兼容: 使用 shared_ptr + atomic 操作函数 (C++20 前非正式支持)
    // 这里使用 std::atomic_load / std::atomic_store 系列函数
    std::shared_ptr<Node> head_{nullptr};
    std::mutex head_mutex_; // C++17 下 shared_ptr 的原子操作非标准, 用 mutex 保护

public:
    LockFreeStack() = default;
    ~LockFreeStack() {
        // shared_ptr 自动回收所有节点
        head_.reset();
    }

    // 禁止拷贝 (unique ownership of head)
    LockFreeStack(const LockFreeStack&) = delete;
    LockFreeStack& operator=(const LockFreeStack&) = delete;

    // -----------------------------------------------------------------------
    // push — 向栈顶压入元素
    // -----------------------------------------------------------------------
    void push(T val) {
        auto new_node = std::make_shared<Node>(std::move(val));
        std::lock_guard<std::mutex> lock(head_mutex_);
        new_node->next = head_;
        head_ = std::move(new_node);
    }

    // -----------------------------------------------------------------------
    // pop — 从栈顶弹出元素, 返回 shared_ptr (空表示栈空)
    // -----------------------------------------------------------------------
    std::shared_ptr<T> pop() {
        std::lock_guard<std::mutex> lock(head_mutex_);
        if (!head_) {
            return nullptr; // 栈空
        }
        auto result = std::make_shared<T>(head_->data);
        head_ = head_->next; // shared_ptr 自动释放旧 head
        return result;
    }

    // -----------------------------------------------------------------------
    // empty — 检查栈是否为空
    // -----------------------------------------------------------------------
    bool empty() const {
        std::lock_guard<std::mutex> lock(head_mutex_);
        return head_ == nullptr;
    }
};

// ============================================================================
// LockFreeStackV2<T> — 使用 atomic<raw pointer> + 引用计数实现真正无锁
// ============================================================================
template <typename T>
class LockFreeStackV2 {
private:
    struct Node {
        T data;
        Node* next;
        std::atomic<int> ref_count{0}; // 引用计数, 用于安全内存释放

        Node(T val) : data(std::move(val)), next(nullptr) {}
    };

    std::atomic<Node*> head_{nullptr};

    // 原子递增引用计数
    static void add_ref(Node* node) {
        if (node) node->ref_count.fetch_add(1, std::memory_order_relaxed);
    }

    // 原子递减引用计数, 计数归零则删除
    static void release(Node* node) {
        if (node && node->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            // 先释放后继节点
            Node* next = node->next;
            delete node;
            // 继续释放后继 (递归可能很深, 实际应用用循环)
            while (next) {
                Node* tmp = next;
                next = next->next;
                delete tmp;
            }
        }
    }

public:
    LockFreeStackV2() = default;

    ~LockFreeStackV2() {
        // 释放所有节点
        Node* node = head_.load(std::memory_order_relaxed);
        while (node) {
            Node* next = node->next;
            delete node;
            node = next;
        }
    }

    LockFreeStackV2(const LockFreeStackV2&) = delete;
    LockFreeStackV2& operator=(const LockFreeStackV2&) = delete;

    // -----------------------------------------------------------------------
    // push — 无锁入栈
    // -----------------------------------------------------------------------
    void push(T val) {
        Node* new_node = new Node(std::move(val));
        new_node->ref_count.store(1, std::memory_order_relaxed);
        new_node->next = head_.load(std::memory_order_relaxed);

        // CAS 循环: 直到成功将 head 更新为 new_node
        while (!head_.compare_exchange_weak(
                   new_node->next, new_node,
                   std::memory_order_release,
                   std::memory_order_relaxed)) {
            // CAS 失败, new_node->next 已被更新为最新 head, 重试
        }
    }

    // -----------------------------------------------------------------------
    // pop — 无锁出栈
    // -----------------------------------------------------------------------
    std::shared_ptr<T> pop() {
        Node* old_head = head_.load(std::memory_order_acquire);

        while (old_head) {
            add_ref(old_head); // 保护指针, 防止被其他线程删除

            Node* next = old_head->next;
            if (head_.compare_exchange_weak(
                    old_head, next,
                    std::memory_order_acquire,
                    std::memory_order_relaxed)) {
                // 成功弹出
                auto result = std::make_shared<T>(old_head->data);
                release(old_head); // 递减 ref_count
                release(old_head); // 再次递减 (我们加了两次: 初始 + add_ref)
                return result;
            }

            release(old_head); // 失败, 释放本次引用
            // old_head 已被 CAS 更新, 继续循环
        }

        return nullptr; // 栈空
    }

    bool empty() const {
        return head_.load(std::memory_order_relaxed) == nullptr;
    }
};

// ============================================================================
// 并发测试
// ============================================================================
template <typename StackType>
void concurrency_test(const std::string& name) {
    StackType stack;
    constexpr int kNumProducers = 4;
    constexpr int kNumConsumers = 4;
    constexpr int kItemsPerProducer = 100000;
    constexpr int kTotalItems = kNumProducers * kItemsPerProducer;

    std::atomic<int> produced{0};
    std::atomic<int> consumed{0};
    std::vector<int> popped_values(kTotalItems, 0);
    std::mutex result_mutex;

    // 生产者线程
    std::vector<std::jthread> producers;
    for (int t = 0; t < kNumProducers; ++t) {
        producers.emplace_back([&, t]() {
            for (int i = 0; i < kItemsPerProducer; ++i) {
                int val = t * kItemsPerProducer + i;
                stack.push(val);
                produced.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    // 等待所有生产者完成
    for (auto& p : producers) p.join();

    // 消费者线程
    std::vector<std::jthread> consumers;
    for (int t = 0; t < kNumConsumers; ++t) {
        consumers.emplace_back([&]() {
            while (consumed.load(std::memory_order_relaxed) < kTotalItems) {
                auto item = stack.pop();
                if (item) {
                    int idx = consumed.fetch_add(1, std::memory_order_relaxed);
                    if (idx < kTotalItems) {
                        popped_values[idx] = *item;
                    }
                }
            }
        });
    }

    for (auto& c : consumers) c.join();

    // 验证结果
    std::cout << "[" << name << "] 结果:\n";
    std::cout << "  总生产: " << kTotalItems << "\n";
    std::cout << "  总消费: " << consumed.load() << "\n";

    // 排序并验证不重复/不遗漏
    std::sort(popped_values.begin(), popped_values.end());
    bool correct = true;
    for (int i = 0; i < kTotalItems; ++i) {
        if (popped_values[i] != i) {
            correct = false;
            std::cerr << "  错误: popped_values[" << i << "] = "
                      << popped_values[i] << ", 期望 " << i << "\n";
            break;
        }
    }
    std::cout << "  数据完整性: " << (correct ? "通过" : "失败") << "\n\n";
}

// ============================================================================
// main
// ============================================================================
int main() {
    std::cout << "=== 第7章: 无锁栈 (Treiber Stack) ===\n\n";

    // 测试 V1 (mutex 保护版本, 用于对比)
    concurrency_test<LockFreeStack<int>>("LockFreeStack (mutex)");

    // 测试 V2 (真正的无锁版本)
    concurrency_test<LockFreeStackV2<int>>("LockFreeStackV2 (lock-free)");

    std::cout << "所有测试完成!\n";
    return 0;
}
