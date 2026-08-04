/**
 * 01_simple_thread_pool.cpp — 简单线程池实现
 *
 * 固定数量工作线程 + 线程安全任务队列。
 * 技术要点:
 *  - std::function<void()> 作为任务类型
 *  - std::mutex + std::condition_variable 实现生产者-消费者
 *  - 停止机制: join() 方法 + poison pill 或 stop flag
 *  - 使用 std::jthread (C++20) 或手动 join (C++17)
 *
 * 编译: g++ -std=c++20 -O2 -pthread 01_simple_thread_pool.cpp -o simple_thread_pool
 */

#include <iostream>
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <chrono>
#include <memory>
#include <cassert>

// ============================================================================
// SimpleThreadPool — 固定大小线程池
// ============================================================================
class SimpleThreadPool {
private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;

    mutable std::mutex queue_mutex_;
    std::condition_variable cv_;
    std::atomic<bool> stop_{false};

public:
    // -------------------------------------------------------------------
    // 构造函数: 创建 num_threads 个工作线程
    // -------------------------------------------------------------------
    explicit SimpleThreadPool(size_t num_threads = 0) {
        if (num_threads == 0) {
            num_threads = std::thread::hardware_concurrency();
        }
        if (num_threads == 0) num_threads = 2; // 保底

        workers_.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back(&SimpleThreadPool::worker_loop, this);
        }

        std::cout << "线程池已创建, 工作线程数: " << num_threads << "\n";
    }

    // -------------------------------------------------------------------
    // 析构函数: 停止所有线程并 join
    // -------------------------------------------------------------------
    ~SimpleThreadPool() {
        join();
    }

    // 禁止拷贝/移动
    SimpleThreadPool(const SimpleThreadPool&) = delete;
    SimpleThreadPool& operator=(const SimpleThreadPool&) = delete;

    // -------------------------------------------------------------------
    // submit — 提交任务 (void() 类型)
    // -------------------------------------------------------------------
    void submit(std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (stop_.load(std::memory_order_acquire)) {
                throw std::runtime_error("线程池已停止, 无法提交任务");
            }
            tasks_.push(std::move(task));
        }
        cv_.notify_one();
    }

    // -------------------------------------------------------------------
    // join — 停止线程池, 等待所有任务完成
    // -------------------------------------------------------------------
    void join() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (stop_.load(std::memory_order_acquire)) {
                return; // 已经停止
            }
            stop_.store(true, std::memory_order_release);
        }

        cv_.notify_all();

        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    // -------------------------------------------------------------------
    // 当前队列中的任务数
    // -------------------------------------------------------------------
    size_t pending_tasks() const {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        return tasks_.size();
    }

    // -------------------------------------------------------------------
    // 工作线程数
    // -------------------------------------------------------------------
    size_t worker_count() const {
        return workers_.size();
    }

private:
    // -------------------------------------------------------------------
    // 工作线程主循环
    // -------------------------------------------------------------------
    void worker_loop() {
        while (true) {
            std::function<void()> task;

            {
                std::unique_lock<std::mutex> lock(queue_mutex_);

                // 等待任务或停止信号
                cv_.wait(lock, [this]() {
                    return stop_.load(std::memory_order_acquire) || !tasks_.empty();
                });

                // 如果收到停止信号且队列为空, 退出
                if (stop_.load(std::memory_order_acquire) && tasks_.empty()) {
                    return;
                }

                // 取出任务
                task = std::move(tasks_.front());
                tasks_.pop();
            }

            // 执行任务 (不持锁)
            if (task) {
                task();
            }
        }
    }
};

// ============================================================================
// 使用示例
// ============================================================================
void usage_demo() {
    std::cout << "=== 简单线程池使用示例 ===\n\n";

    SimpleThreadPool pool(4);

    std::atomic<int> counter{0};
    constexpr int kNumTasks = 20;

    // 提交一批任务
    for (int i = 0; i < kNumTasks; ++i) {
        pool.submit([i, &counter]() {
            // 模拟计算
            int result = 0;
            for (int j = 0; j < 100000; ++j) {
                result = (result + j * i) % 10007;
            }

            int count = counter.fetch_add(1, std::memory_order_relaxed) + 1;
            std::cout << "  任务 " << i << " 完成 (第 " << count << " 个)"
                      << "  [线程 " << std::this_thread::get_id() << "]\n";

            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        });
    }

    std::cout << "  已提交 " << kNumTasks << " 个任务, 队列剩余: "
              << pool.pending_tasks() << "\n";

    // 等待所有任务完成
    pool.join();

    std::cout << "\n  所有任务完成! 总计数器: " << counter.load() << "\n";
}

// ============================================================================
// 压力测试
// ============================================================================
void stress_test() {
    std::cout << "\n=== 线程池压力测试 ===\n";

    constexpr int kTasks = 100000;
    SimpleThreadPool pool(8);

    std::atomic<long long> sum{0};
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < kTasks; ++i) {
        pool.submit([i, &sum]() {
            sum.fetch_add(i, std::memory_order_relaxed);
        });
    }

    pool.join();

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // 期望和: 0 + 1 + ... + (kTasks - 1) = kTasks * (kTasks - 1) / 2
    long long expected = static_cast<long long>(kTasks) * (kTasks - 1) / 2;
    long long actual = sum.load();

    std::cout << "  任务数: " << kTasks << "\n";
    std::cout << "  期望和: " << expected << "\n";
    std::cout << "  实际和: " << actual << "\n";
    std::cout << "  耗时: " << elapsed << " ms\n";
    std::cout << "  正确性: " << (expected == actual ? "通过" : "失败") << "\n";
}

// ============================================================================
// main
// ============================================================================
int main() {
    usage_demo();
    stress_test();
    return 0;
}
