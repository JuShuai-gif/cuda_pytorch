/**
 * 04_aba_demo.cpp — ABA 问题的演示与解决方案
 *
 * ABA 问题: 线程 A 读取值 X, 线程 B 将 X 改为 Y 再改回 X,
 *           线程 A 的 CAS 成功但状态已不一致。
 *
 * 解决方案:
 *  1. Tagged Pointer (带版本号的指针): 将指针和高位版本号打包
 *  2. Double-width CAS (CMPXCHG16B on x86-64): 原子操作 128 位
 *  3. Hazard Pointer / RCU: 延迟内存回收, 防止地址复用
 *
 * 编译: g++ -std=c++20 -O2 -pthread 04_aba_demo.cpp -o aba_demo
 */

#include <atomic>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>
#include <cstdint>
#include <chrono>
#include <optional>
#include <mutex>

// ============================================================================
// 场景1: 无防护的 ABA 易发栈
// ============================================================================
template <typename T>
class VulnerableStack {
private:
    struct Node {
        T data;
        Node* next;
        Node(T val, Node* n = nullptr) : data(std::move(val)), next(n) {}
    };

    std::atomic<Node*> head_{nullptr};

public:
    void push(T val) {
        auto* node = new Node(std::move(val));
        node->next = head_.load(std::memory_order_relaxed);
        while (!head_.compare_exchange_weak(
                   node->next, node,
                   std::memory_order_release,
                   std::memory_order_relaxed)) {}
    }

    std::optional<T> pop() {
        Node* old_head = head_.load(std::memory_order_acquire);
        while (old_head) {
            if (head_.compare_exchange_weak(
                    old_head, old_head->next,
                    std::memory_order_acquire,
                    std::memory_order_relaxed)) {
                T val = std::move(old_head->data);
                delete old_head; // ⚠️ 危险! 立即释放可能导致 ABA
                return val;
            }
        }
        return std::nullopt;
    }

    ~VulnerableStack() {
        Node* node = head_.load();
        while (node) {
            Node* next = node->next;
            delete node;
            node = next;
        }
    }
};

// ============================================================================
// 场景2: Tagged Pointer — ABA 免疫栈
// 在 x86-64 上指针只使用低 48 位, 高 16 位用做版本计数器
// ============================================================================
template <typename T>
class TaggedPointerStack {
private:
    struct Node {
        T data;
        Node* next;
        Node(T val, Node* n = nullptr) : data(std::move(val)), next(n) {}
    };

    // 带版本号的指针 (tagged pointer)
    // 低 48 位: 实际指针; 高 16 位: 版本计数器
    struct TaggedPtr {
        Node* ptr;
        uint16_t tag;

        bool operator==(const TaggedPtr& other) const {
            return ptr == other.ptr && tag == other.tag;
        }
    };

    // 将 TaggedPtr 打包为 uintptr_t (64 位平台)
    static_assert(sizeof(uintptr_t) == 8, "需要 64 位平台");

    static uintptr_t pack(TaggedPtr tp) {
        return (static_cast<uintptr_t>(tp.tag) << 48) |
               (reinterpret_cast<uintptr_t>(tp.ptr) & 0x0000FFFFFFFFFFFFULL);
    }

    static TaggedPtr unpack(uintptr_t val) {
        return {
            reinterpret_cast<Node*>(val & 0x0000FFFFFFFFFFFFULL),
            static_cast<uint16_t>(val >> 48)
        };
    }

    std::atomic<uintptr_t> head_{0}; // nullptr with tag 0

    // 待回收节点链表 (简化: 用 shared_ptr 管理)
    // 生产环境应用 Hazard Pointer 或 epoch-based reclamation
    struct ReclaimNode {
        Node* node;
        std::shared_ptr<ReclaimNode> next;
    };
    std::shared_ptr<ReclaimNode> reclaim_list_;

public:
    void push(T val) {
        auto* node = new Node(std::move(val));
        uintptr_t old_head = head_.load(std::memory_order_relaxed);
        TaggedPtr old_tp = unpack(old_head);
        TaggedPtr new_tp{node, static_cast<uint16_t>(old_tp.tag + 1)};
        uintptr_t new_head = pack(new_tp);

        node->next = old_tp.ptr;
        while (!head_.compare_exchange_weak(
                   old_head, new_head,
                   std::memory_order_release,
                   std::memory_order_relaxed)) {
            old_tp = unpack(old_head);
            node->next = old_tp.ptr;
            new_tp.tag = old_tp.tag + 1;
            new_head = pack(new_tp);
        }
    }

    std::optional<T> pop() {
        uintptr_t old_head = head_.load(std::memory_order_acquire);
        TaggedPtr old_tp = unpack(old_head);

        while (old_tp.ptr) {
            TaggedPtr new_tp{old_tp.ptr->next, static_cast<uint16_t>(old_tp.tag + 1)};
            uintptr_t new_head = pack(new_tp);

            if (head_.compare_exchange_weak(
                    old_head, new_head,
                    std::memory_order_acquire,
                    std::memory_order_relaxed)) {
                T val = std::move(old_tp.ptr->data);
                // 延迟回收 (简化: 加入 reclaim 链表)
                auto reclaim_node = std::make_shared<ReclaimNode>(
                    ReclaimNode{old_tp.ptr, reclaim_list_});
                reclaim_list_ = reclaim_node;
                return val;
            }
            old_tp = unpack(old_head);
        }
        return std::nullopt;
    }

    ~TaggedPointerStack() {
        // 先释放 reclaim 链表
        reclaim_list_.reset();
        // 再释放残余节点
        uintptr_t val = head_.load();
        Node* node = unpack(val).ptr;
        while (node) {
            Node* next = node->next;
            delete node;
            node = next;
        }
    }
};

// ============================================================================
// 演示: 高并发下 ABA 测试
// ============================================================================
void aba_demonstration() {
    std::cout << "=== ABA 问题演示 ===\n\n";

    std::cout << "ABA 问题场景:\n";
    std::cout << "  1. 线程A 读取 head = 0x1000 (节点 N1)\n";
    std::cout << "  2. 线程B pop N1, delete N1\n";
    std::cout << "  3. 线程B push N2, 恰好分配在 0x1000 地址\n";
    std::cout << "  4. 线程A 的 CAS 比较 head 仍为 0x1000, 成功\n";
    std::cout << "  5. 结果: N1->next 已无效, 数据损坏!\n\n";

    // 使用 Tagged Pointer 栈进行并发测试
    TaggedPointerStack<int> stack;
    constexpr int kThreads = 8;
    constexpr int kOpsPerThread = 50000;

    std::atomic<int> push_count{0};
    std::atomic<int> pop_count{0};
    std::atomic<long long> sum_pushed{0};
    std::atomic<long long> sum_popped{0};

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::jthread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < kOpsPerThread; ++i) {
                if (i % 2 == 0) {
                    int val = t * kOpsPerThread + i;
                    stack.push(val);
                    push_count.fetch_add(1, std::memory_order_relaxed);
                    sum_pushed.fetch_add(val, std::memory_order_relaxed);
                } else {
                    auto item = stack.pop();
                    if (item) {
                        pop_count.fetch_add(1, std::memory_order_relaxed);
                        sum_popped.fetch_add(*item, std::memory_order_relaxed);
                    }
                }
            }
        });
    }

    for (auto& th : threads) th.join();

    // 清空剩余元素
    while (true) {
        auto item = stack.pop();
        if (!item) break;
        pop_count.fetch_add(1, std::memory_order_relaxed);
        sum_popped.fetch_add(*item, std::memory_order_relaxed);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Tagged Pointer 栈并发测试结果:\n";
    std::cout << "  线程数: " << kThreads << "\n";
    std::cout << "  推送次数: " << push_count.load() << "\n";
    std::cout << "  弹出次数: " << pop_count.load() << "\n";
    std::cout << "  推送总和: " << sum_pushed.load() << "\n";
    std::cout << "  弹出总和: " << sum_popped.load() << "\n";
    std::cout << "  耗时: " << elapsed << " ms\n";

    if (sum_pushed.load() == sum_popped.load() &&
        push_count.load() == pop_count.load()) {
        std::cout << "  正确性: 通过! (Tagged Pointer 防护有效)\n";
    } else {
        std::cerr << "  正确性: 失败!\n";
    }
}

// ============================================================================
// main
// ============================================================================
int main() {
    aba_demonstration();
    return 0;
}
