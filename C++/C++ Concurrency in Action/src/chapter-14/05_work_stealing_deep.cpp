// 05_work_stealing_deep.cpp — Work Stealing 深层分析
// 演示: 工作窃取的负载均衡效果、局部性优势

#include <atomic>
#include <chrono>
#include <deque>
#include <functional>
#include <iostream>
#include <mutex>
#include <random>
#include <syncstream>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

// ===== 简化的 Work-Stealing Deque =====
template <typename T>
class WorkStealingDeque {
public:
    // 本地 push/pop (LIFO, 只被 owner 线程调用)
    void push(T item) {
        std::lock_guard lock(mtx_);
        deque_.push_back(std::move(item));
    }

    bool try_pop(T& item) {
        std::lock_guard lock(mtx_);
        if (deque_.empty()) return false;
        item = std::move(deque_.back());
        deque_.pop_back();
        return true;
    }

    // 窃取 (FIFO, 从其他线程调用)
    bool try_steal(T& item) {
        std::lock_guard lock(mtx_);
        if (deque_.empty()) return false;
        item = std::move(deque_.front());
        deque_.pop_front();
        return true;
    }

    bool empty() const {
        std::lock_guard lock(mtx_);
        return deque_.empty();
    }

    size_t size() const {
        std::lock_guard lock(mtx_);
        return deque_.size();
    }

private:
    mutable std::mutex mtx_;
    std::deque<T> deque_;
};

// ===== 带工作窃取的线程池 (简化版) =====
class WorkStealingPool {
public:
    explicit WorkStealingPool(size_t num_threads)
        : num_threads_(num_threads), deques_(num_threads) {
        threads_.reserve(num_threads);
        for (size_t t = 0; t < num_threads; ++t) {
            threads_.emplace_back([this, t]() { worker_loop(t); });
        }
    }

    ~WorkStealingPool() {
        done_ = true;
        for (auto& t : threads_) t.join();
    }

    // 提交任务到指定线程的队列
    void submit(size_t thread_id, std::function<void()> task) {
        deques_[thread_id].push(std::move(task));
        total_submitted_.fetch_add(1);
    }

    size_t stolen_count() const { return stolen_.load(); }
    size_t total_submitted() const { return total_submitted_.load(); }

private:
    void worker_loop(size_t tid) {
        while (!done_.load(std::memory_order_relaxed)) {
            std::function<void()> task;

            // 1. 先尝试本地队列
            if (deques_[tid].try_pop(task)) {
                task();
                continue;
            }

            // 2. 尝试从其他线程窃取
            bool stolen = false;
            for (size_t i = 1; i < num_threads_ && !stolen; ++i) {
                size_t victim = (tid + i) % num_threads_;
                if (deques_[victim].try_steal(task)) {
                    stolen_.fetch_add(1);
                    stolen = true;
                    break;
                }
            }

            if (stolen) {
                task();
            } else {
                // 无事可做，短暂休眠
                std::this_thread::sleep_for(100us);
            }
        }
    }

    size_t num_threads_;
    std::atomic<bool> done_{false};
    std::vector<WorkStealingDeque<std::function<void()>>> deques_;
    std::vector<std::jthread> threads_;
    std::atomic<size_t> stolen_{0};
    std::atomic<size_t> total_submitted_{0};
};

// ===== 演示: 不均衡负载下的工作窃取 =====
void demo_work_stealing_balancing() {
    std::cout << "=== Work Stealing 负载均衡 ===\n";

    const size_t kThreads = 4;
    WorkStealingPool pool(kThreads);

    // 模拟不均衡: 将大量任务提交到线程 0
    const int kTasks = 100;
    for (int i = 0; i < kTasks; ++i) {
        pool.submit(0, [i]() {
            // 模拟不同耗时的任务
            int delay = (i % 5 + 1); // 1-5 ms
            std::this_thread::sleep_for(std::chrono::milliseconds(delay));
        });
    }

    // 等待完成 — 简化: 等待一段时间
    std::this_thread::sleep_for(500ms);

    std::cout << "  总提交: " << pool.total_submitted()
              << " (全部提交到线程 0)\n";
    std::cout << "  被窃取: " << pool.stolen_count()
              << " (其他线程窃取的工作)\n";
    std::cout << "  若没有工作窃取，只有线程 0 会执行任务\n";
    std::cout << "  工作窃取让空闲线程自动从繁忙线程拉取任务\n";
}

// ===== Work Stealing vs 共享队列对比 =====
void demo_ws_vs_shared_queue() {
    std::cout << "\n=== Work Stealing vs 共享队列 ===\n";

    std::cout << "  Work Stealing 优势:\n";
    std::cout << "    1. 本地操作无锁争用 (LIFO pop/push)\n";
    std::cout << "    2. 窃取从另一端 (FIFO)，减少争用\n";
    std::cout << "    3. 任务有局部性 (同线程的数据在缓存中)\n";
    std::cout << "    4. 自动负载均衡\n\n";

    std::cout << "  共享队列问题:\n";
    std::cout << "    1. 每次 push/pop 都有锁争用\n";
    std::cout << "    2. 缓存行在核心间来回弹跳 (cache line bouncing)\n";
    std::cout << "    3. 无局部性保证\n";
}

int main() {
    demo_work_stealing_balancing();
    demo_ws_vs_shared_queue();

    std::cout << "\nWork Stealing 是 C++ 并行标准库 (TBB, PSTL) 的基础调度策略。\n";
    return 0;
}
