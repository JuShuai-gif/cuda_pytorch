/**
 * 05_hazard_pointer.cpp — Hazard Pointer 的简化实现
 *
 * Hazard Pointer 是一种安全内存回收 (SMR) 机制, 用于无锁数据结构。
 * 核心思想:
 *  - 每个线程维护一组 "危险指针" (hazard pointers), 标记正在访问的节点
 *  - 待删除的节点放入 retire list, 而不是立即释放
 *  - 当 retire list 中某个节点不被任何 hazard pointer 引用时, 才安全释放
 *
 * 技术要点:
 *  - 全局 HP 数组 (每个线程最多 N 个 HP)
 *  - acquire/release 原语
 *  - retire 与 scan 机制
 *
 * 编译: g++ -std=c++20 -O2 -pthread 05_hazard_pointer.cpp -o hazard_pointer
 */

#include <atomic>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>
#include <cassert>
#include <algorithm>
#include <array>
#include <functional>
#include <mutex>

// ============================================================================
// HazardPointer 系统
// ============================================================================
constexpr int kMaxHazardPointers = 100;   // 最大线程数
constexpr int kMaxHPPerThread = 2;        // 每个线程最多持有的 HP 数
constexpr int kRetireThreshold = 64;      // retire list 阈值, 触发 scan

// 危险指针记录
struct HazardPointer {
    std::atomic<void*> ptr{nullptr};
};

// 全局 HP 池
static HazardPointer g_hazard_pointers[kMaxHazardPointers][kMaxHPPerThread];
static std::atomic<int> g_hp_owner[kMaxHazardPointers]{}; // -1 表示空闲

// 线程局部: 该线程拥有的 HP 索引
static thread_local int t_hp_index = -1;

namespace hazard_pointer {

// -----------------------------------------------------------------------
// 注册当前线程, 分配 HP 槽位
// -----------------------------------------------------------------------
inline int register_thread() {
    if (t_hp_index != -1) return t_hp_index;

    for (int i = 0; i < kMaxHazardPointers; ++i) {
        int expected = -1;
        if (g_hp_owner[i].compare_exchange_strong(expected, 0)) {
            t_hp_index = i;
            return i;
        }
    }
    std::cerr << "错误: Hazard Pointer 槽位已满!\n";
    std::terminate();
}

// -----------------------------------------------------------------------
// 注销当前线程
// -----------------------------------------------------------------------
inline void unregister_thread() {
    if (t_hp_index != -1) {
        // 清空该线程所有 HP
        for (int j = 0; j < kMaxHPPerThread; ++j) {
            g_hazard_pointers[t_hp_index][j].ptr.store(nullptr, std::memory_order_release);
        }
        g_hp_owner[t_hp_index].store(-1, std::memory_order_release);
        t_hp_index = -1;
    }
}

// -----------------------------------------------------------------------
// 获取指定 HP 槽位 (hp_idx ∈ [0, kMaxHPPerThread))
// -----------------------------------------------------------------------
inline HazardPointer& get_hp(int hp_idx = 0) {
    assert(t_hp_index >= 0 && hp_idx < kMaxHPPerThread);
    return g_hazard_pointers[t_hp_index][hp_idx];
}

// -----------------------------------------------------------------------
// 保护指针 (acquire)
// 将 ptr 写入 HP, 确保该指针指向的内存不会被释放
// -----------------------------------------------------------------------
inline void protect(void* ptr, int hp_idx = 0) {
    get_hp(hp_idx).ptr.store(ptr, std::memory_order_release);
}

// -----------------------------------------------------------------------
// 清除保护
// -----------------------------------------------------------------------
inline void clear_protect(int hp_idx = 0) {
    get_hp(hp_idx).ptr.store(nullptr, std::memory_order_release);
}

// -----------------------------------------------------------------------
// 检查某个指针是否被任何线程的 HP 引用
// -----------------------------------------------------------------------
inline bool is_protected(void* ptr) {
    for (int i = 0; i < kMaxHazardPointers; ++i) {
        if (g_hp_owner[i].load(std::memory_order_acquire) >= 0) {
            for (int j = 0; j < kMaxHPPerThread; ++j) {
                if (g_hazard_pointers[i][j].ptr.load(std::memory_order_acquire) == ptr) {
                    return true;
                }
            }
        }
    }
    return false;
}

// ============================================================================
// RetireList — 延迟回收列表 (每线程独立)
// ============================================================================
class RetireList {
private:
    struct RetiredNode {
        void* ptr;
        std::function<void(void*)> deleter;
    };

    std::vector<RetiredNode> list_;
    int scan_threshold_;

public:
    explicit RetireList(int threshold = kRetireThreshold)
        : scan_threshold_(threshold) {
        list_.reserve(threshold * 2);
    }

    // 添加待回收节点
    void add(void* ptr, std::function<void(void*)> deleter) {
        list_.push_back({ptr, std::move(deleter)});
        if (list_.size() >= static_cast<size_t>(scan_threshold_)) {
            scan();
        }
    }

    // 扫描并释放不再被保护的节点
    void scan() {
        auto it = list_.begin();
        while (it != list_.end()) {
            if (!is_protected(it->ptr)) {
                it->deleter(it->ptr); // 安全释放
                it = list_.erase(it);
            } else {
                ++it;
            }
        }
    }

    // 强制释放所有 (仅在单线程退出时调用)
    void force_reclaim() {
        scan();
        for (auto& node : list_) {
            node.deleter(node.ptr);
        }
        list_.clear();
    }
};

static thread_local RetireList t_retire_list;

// -----------------------------------------------------------------------
// 公共 retire 接口
// -----------------------------------------------------------------------
inline void retire(void* ptr, std::function<void(void*)> deleter) {
    t_retire_list.add(ptr, std::move(deleter));
}

} // namespace hazard_pointer

// ============================================================================
// 带 Hazard Pointer 保护的无锁栈
// ============================================================================
template <typename T>
class HPProtectedStack {
private:
    struct Node {
        T data;
        Node* next;
        Node(T val, Node* n = nullptr) : data(std::move(val)), next(n) {}
    };

    std::atomic<Node*> head_{nullptr};

public:
    HPProtectedStack() {
        hazard_pointer::register_thread();
    }

    ~HPProtectedStack() {
        Node* node = head_.load();
        while (node) {
            Node* next = node->next;
            delete node;
            node = next;
        }
        hazard_pointer::unregister_thread();
    }

    HPProtectedStack(const HPProtectedStack&) = delete;
    HPProtectedStack& operator=(const HPProtectedStack&) = delete;

    // -------------------------------------------------------------------
    // push — 入栈
    // -------------------------------------------------------------------
    void push(T val) {
        Node* node = new Node(std::move(val));
        node->next = head_.load(std::memory_order_relaxed);
        while (!head_.compare_exchange_weak(
                   node->next, node,
                   std::memory_order_release,
                   std::memory_order_relaxed)) {}
    }

    // -------------------------------------------------------------------
    // pop — 出栈 (Hazard Pointer 保护)
    // -------------------------------------------------------------------
    std::unique_ptr<T> pop() {
        Node* old_head = head_.load(std::memory_order_acquire);

        while (old_head) {
            // 步骤1: 用 HP 保护 old_head
            hazard_pointer::protect(old_head, 0);

            // 步骤2: 重新读取, 确认 head 未被修改
            Node* new_head = head_.load(std::memory_order_acquire);
            if (new_head != old_head) {
                hazard_pointer::clear_protect(0);
                old_head = new_head;
                continue;
            }

            // 步骤3: CAS 尝试弹出
            if (head_.compare_exchange_weak(
                    old_head, old_head->next,
                    std::memory_order_acquire,
                    std::memory_order_relaxed)) {
                hazard_pointer::clear_protect(0);

                auto result = std::make_unique<T>(std::move(old_head->data));

                // 延迟回收: 不是立即 delete, 而是交给 retire list
                hazard_pointer::retire(old_head, [](void* p) {
                    delete static_cast<Node*>(p);
                });

                return result;
            }

            hazard_pointer::clear_protect(0);
            // old_head 已被 CAS 更新, 继续循环
        }

        return nullptr;
    }
};

// ============================================================================
// 并发正确性测试
// ============================================================================
void concurrency_test() {
    std::cout << "=== Hazard Pointer 无锁栈并发测试 ===\n";

    hazard_pointer::register_thread();

    HPProtectedStack<int> stack;
    constexpr int kThreads = 4;
    constexpr int kOpsPerThread = 250000;

    std::atomic<long long> push_sum{0};
    std::atomic<long long> pop_sum{0};
    std::atomic<int> push_count{0};
    std::atomic<int> pop_count{0};

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::jthread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() {
            hazard_pointer::register_thread();

            for (int i = 0; i < kOpsPerThread; ++i) {
                int val = t * kOpsPerThread + i;
                stack.push(val);
                push_sum.fetch_add(val, std::memory_order_relaxed);
                push_count.fetch_add(1, std::memory_order_relaxed);

                // 偶尔 pop
                if (i % 3 == 0) {
                    auto item = stack.pop();
                    if (item) {
                        pop_sum.fetch_add(*item, std::memory_order_relaxed);
                        pop_count.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }

            hazard_pointer::unregister_thread();
        });
    }

    for (auto& th : threads) th.join();

    // 清空剩余
    while (true) {
        auto item = stack.pop();
        if (!item) break;
        pop_sum.fetch_add(*item, std::memory_order_relaxed);
        pop_count.fetch_add(1, std::memory_order_relaxed);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "  线程数: " << kThreads << "\n";
    std::cout << "  推送次数: " << push_count.load() << "\n";
    std::cout << "  弹出次数: " << pop_count.load() << "\n";
    std::cout << "  推送总和: " << push_sum.load() << "\n";
    std::cout << "  弹出总和: " << pop_sum.load() << "\n";
    std::cout << "  耗时: " << elapsed << " ms\n";

    if (push_sum.load() == pop_sum.load() &&
        push_count.load() == pop_count.load()) {
        std::cout << "  正确性: 通过!\n";
    } else {
        std::cerr << "  正确性: 失败!\n";
    }

    hazard_pointer::unregister_thread();
}

// ============================================================================
// main
// ============================================================================
int main() {
    concurrency_test();
    return 0;
}
