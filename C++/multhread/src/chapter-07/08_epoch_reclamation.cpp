// 08_epoch_reclamation.cpp — Epoch-Based 内存回收 (RCU 简化实现)
//
// 无锁数据结构的最大挑战: 何时安全释放被移除的节点?
//  - 不能立即释放: 可能有其他线程正在读取
//  - shared_ptr: 有引用计数开销
//  - Hazard Pointer: 每线程需注册保护指针
//  - Epoch-Based: 基于"代"的批量回收
//
// 原理:
//   1. 维护一个全局 epoch 计数器
//   2. 每个线程记录自己当前所在的 epoch
//   3. 删除节点时放入当前 epoch 的回收列表
//   4. 当所有线程都离开某个 epoch 后，该 epoch 的回收列表可安全删除

#include <atomic>
#include <chrono>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <syncstream>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

// ================================================================
// Epoch-Based Reclamation System
// ================================================================
class EpochSystem {
public:
    static constexpr int kMaxEpochs = 3;
    static constexpr int kMaxThreads = 64;

    EpochSystem() {
        // 初始化线程状态
        for (int i = 0; i < kMaxThreads; ++i) {
            thread_epochs_[i].store(0, std::memory_order_relaxed);
        }
    }

    // 注册一个线程
    int register_thread() {
        int tid = next_tid_.fetch_add(1);
        return tid;
    }

    // 进入临界区 (读操作前调用)
    void enter_critical(int tid) {
        int current_epoch = global_epoch_.load(std::memory_order_acquire);
        thread_epochs_[tid].store(current_epoch, std::memory_order_release);
    }

    // 离开临界区 (读操作后调用)
    void leave_critical(int tid) {
        thread_epochs_[tid].store(0, std::memory_order_relaxed);
    }

    // 将待删除对象加入回收列表
    template <typename T>
    void retire(T* obj, int tid) {
        int epoch = global_epoch_.load(std::memory_order_relaxed);
        std::lock_guard lock(retire_mtx_);
        retire_lists_[epoch].push_back([obj]() { delete obj; });
        retire_count_++;
        (void)tid;

        // 当累积足够多时尝试推进 epoch
        if (retire_count_ >= kRetireThreshold) {
            try_advance_epoch();
        }
    }

    // 手动触发 epoch 推进
    void try_advance_epoch() {
        int current_epoch = global_epoch_.load(std::memory_order_relaxed);
        int next_epoch = (current_epoch + 1) % kMaxEpochs;

        // 检查是否有线程仍在当前 epoch 中
        for (int i = 0; i < kMaxThreads; ++i) {
            int t_epoch = thread_epochs_[i].load(std::memory_order_acquire);
            if (t_epoch == current_epoch) {
                return; // 仍有线程在当前 epoch，无法回收
            }
        }

        // 所有线程都已离开当前 epoch → 安全回收该 epoch 的列表
        if (global_epoch_.compare_exchange_strong(
                current_epoch, next_epoch,
                std::memory_order_release,
                std::memory_order_relaxed)) {
            // 回收 (current_epoch + 1) % kMaxEpochs 的列表
            // (该 epoch 现在不可能有线程驻留)
            int reclaim_epoch = (next_epoch + 1) % kMaxEpochs;
            reclaim_list(reclaim_epoch);
            retire_count_ = 0;
        }
    }

private:
    void reclaim_list(int epoch) {
        std::lock_guard lock(retire_mtx_);
        for (auto& deleter : retire_lists_[epoch]) {
            deleter(); // 安全释放
        }
        retire_lists_[epoch].clear();
    }

    static constexpr int kRetireThreshold = 100;

    std::atomic<int> global_epoch_{0};
    std::atomic<int> thread_epochs_[kMaxThreads];
    std::atomic<int> next_tid_{0};
    std::atomic<int> retire_count_{0};

    std::mutex retire_mtx_;
    std::vector<std::function<void()>> retire_lists_[kMaxEpochs];
};

// ================================================================
// 使用 Epoch 的无锁栈 (完整示例)
// ================================================================
template <typename T>
class EpochProtectedStack {
    struct Node {
        T data;
        Node* next;
        Node(T val, Node* n = nullptr) : data(std::move(val)), next(n) {}
    };

public:
    explicit EpochProtectedStack(EpochSystem& epoch_sys)
        : epoch_sys_(epoch_sys), tid_(epoch_sys_.register_thread()) {}

    void push(T value) {
        auto* node = new Node(std::move(value));
        node->next = head_.load(std::memory_order_relaxed);
        while (!head_.compare_exchange_weak(
            node->next, node,
            std::memory_order_release,
            std::memory_order_relaxed)) {}
    }

    bool try_pop(T& value) {
        epoch_sys_.enter_critical(tid_);

        Node* old_head = head_.load(std::memory_order_acquire);
        while (old_head) {
            if (head_.compare_exchange_weak(
                    old_head, old_head->next,
                    std::memory_order_acquire,
                    std::memory_order_relaxed)) {
                value = std::move(old_head->data);
                epoch_sys_.leave_critical(tid_);
                // 安全回收: 通过 Epoch 系统延迟 delete
                epoch_sys_.retire(old_head, tid_);
                return true;
            }
        }

        epoch_sys_.leave_critical(tid_);
        return false;
    }

private:
    EpochSystem& epoch_sys_;
    int tid_;
    std::atomic<Node*> head_{nullptr};
};

// ================================================================
// 演示
// ================================================================
void demo_epoch_reclamation() {
    std::cout << "=== Epoch-Based 内存回收 ===\n\n";

    EpochSystem epoch_sys;
    EpochProtectedStack<int> stack(epoch_sys);

    const int kThreads = 8;
    const int kOpsPerThread = 50'000;
    std::atomic<long long> pushed{0};
    std::atomic<long long> popped{0};

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::jthread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < kOpsPerThread; ++i) {
                if (i % 2 == 0) {
                    stack.push(t * kOpsPerThread + i);
                    pushed.fetch_add(1);
                } else {
                    int val;
                    if (stack.try_pop(val)) {
                        popped.fetch_add(1);
                    }
                }
            }
        });
    }
    threads.clear();

    // 清空栈
    int val;
    while (stack.try_pop(val)) {
        popped.fetch_add(1);
    }

    // 最终回收
    epoch_sys.try_advance_epoch();

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start);

    std::cout << "  推送: " << pushed.load() << "\n";
    std::cout << "  弹出: " << popped.load() << "\n";
    std::cout << "  耗时: " << elapsed.count() << " ms\n";
    std::cout << "  正确性: " << (pushed == popped ? "PASS" : "FAIL") << "\n";
}

// ================================================================
// 内存回收方案对比
// ================================================================
void comparison() {
    std::cout << "\n=== 无锁内存回收方案对比 ===\n\n";

    std::cout << "  ┌──────────────────┬────────┬──────────┬──────────┐\n";
    std::cout << "  │ 方案              │ 读开销  │ 回收延迟  │ 复杂度    │\n";
    std::cout << "  ├──────────────────┼────────┼──────────┼──────────┤\n";
    std::cout << "  │ shared_ptr       │ 高     │ 即时      │ 低       │\n";
    std::cout << "  │ Hazard Pointer   │ 中     │ 近即时    │ 中       │\n";
    std::cout << "  │ Epoch (RCU)      │ 低     │ 批量      │ 中       │\n";
    std::cout << "  │ 引用计数         │ 高     │ 即时      │ 中       │\n";
    std::cout << "  │ 无回收 (泄漏)    │ 零     │ N/A       │ 低       │\n";
    std::cout << "  └──────────────────┴────────┴──────────┴──────────┘\n\n";

    std::cout << "  Epoch 优势:\n";
    std::cout << "    - 读路径极轻量 (仅 store thread-local epoch)\n";
    std::cout << "    - 批量回收分摊开销\n";
    std::cout << "    - 无需 intrusive list (不像 Hazard Pointer)\n\n";

    std::cout << "  Epoch 劣势:\n";
    std::cout << "    - 回收有延迟 (需等所有线程离开当前 epoch)\n";
    std::cout << "    - 若有线程长时间不离开 epoch，内存堆积\n";
    std::cout << "    - 实现比 shared_ptr 复杂\n\n";

    std::cout << "  适用场景: 读多写少的高并发无锁结构 (如 RCU 链表)\n";
}

int main() {
    demo_epoch_reclamation();
    comparison();
    return 0;
}
