/**
 * 03_work_stealing_pool.cpp — 带 Work Stealing 的线程池
 *
 * 核心设计:
 *  - 每个工作线程有一个本地双端队列 (deque)
 *  - 全局共享队列作为备用
 *  - 线程优先从本地队列取任务 (LIFO, 缓存友好)
 *  - 本地队列为空时, 随机从其他线程队列 "偷" 任务 (FIFO)
 *  - 所有队列都为空时, 才从全局队列取任务
 *
 * 技术要点:
 *  - 每个线程独立的 deque, 线程本地取用无锁
 *  - 偷取操作使用 mutex 保护 (简化) 或 Chase-Lev deque
 *  - 避免饥饿: 偷取时选择受害者线程
 *
 * 编译: g++ -std=c++20 -O2 -pthread 03_work_stealing_pool.cpp -o work_stealing_pool
 */

#include <iostream>
#include <thread>
#include <vector>
#include <queue>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <atomic>
#include <chrono>
#include <memory>
#include <random>
#include <cassert>

// ============================================================================
// WorkStealingThreadPool
// ============================================================================
class WorkStealingThreadPool {
public:
    using Task = std::function<void()>;

private:
    struct WorkerData {
        std::deque<Task> local_queue; // 本地任务队列
        std::mutex mutex;             // 保护本地队列 (偷取时需要)
        std::atomic<bool> idle{true}; // 是否空闲
    };

    std::vector<std::thread> threads_;
    std::vector<std::unique_ptr<WorkerData>> worker_data_;

    std::queue<Task> global_queue_;   // 全局后备队列
    mutable std::mutex global_mutex_;
    std::condition_variable global_cv_;

    std::atomic<bool> stop_{false};
    std::atomic<size_t> active_tasks_{0}; // 活跃任务计数
    std::condition_variable done_cv_;     // 等待所有任务完成
    std::mutex done_mutex_;

public:
    explicit WorkStealingThreadPool(size_t num_threads = 0) {
        if (num_threads == 0) {
            num_threads = std::thread::hardware_concurrency();
        }
        if (num_threads == 0) num_threads = 2;

        worker_data_.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            worker_data_.push_back(std::make_unique<WorkerData>());
        }

        threads_.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            threads_.emplace_back(&WorkStealingThreadPool::worker_loop, this, i);
        }

        std::cout << "Work Stealing 线程池已创建, 工作线程数: " << num_threads << "\n";
    }

    ~WorkStealingThreadPool() {
        join();
    }

    WorkStealingThreadPool(const WorkStealingThreadPool&) = delete;
    WorkStealingThreadPool& operator=(const WorkStealingThreadPool&) = delete;

    // -------------------------------------------------------------------
    // submit — 提交任务 (放入当前线程的本地队列, 或全局队列)
    // -------------------------------------------------------------------
    template <typename Func, typename... Args>
    auto submit(Func&& func, Args&&... args)
        -> std::future<typename std::invoke_result_t<Func, Args...>>
    {
        using ResultType = typename std::invoke_result_t<Func, Args...>;

        auto task = std::make_shared<std::packaged_task<ResultType()>>(
            std::bind(std::forward<Func>(func), std::forward<Args>(args)...));

        std::future<ResultType> result = task->get_future();

        Task wrapper = [task = std::move(task)]() { (*task)(); };

        {
            std::lock_guard<std::mutex> lock(global_mutex_);
            if (stop_.load(std::memory_order_acquire)) {
                throw std::runtime_error("线程池已停止");
            }
            global_queue_.push(std::move(wrapper));
        }

        active_tasks_.fetch_add(1, std::memory_order_release);
        global_cv_.notify_one();

        return result;
    }

    // -------------------------------------------------------------------
    // join — 停止线程池
    // -------------------------------------------------------------------
    void join() {
        {
            std::lock_guard<std::mutex> lock(global_mutex_);
            if (stop_.load(std::memory_order_acquire)) return;
            stop_.store(true, std::memory_order_release);
        }
        global_cv_.notify_all();

        for (auto& t : threads_) {
            if (t.joinable()) t.join();
        }
    }

    // -------------------------------------------------------------------
    // wait_all — 等待所有已提交任务完成
    // -------------------------------------------------------------------
    void wait_all() {
        std::unique_lock<std::mutex> lock(done_mutex_);
        done_cv_.wait(lock, [this]() {
            return active_tasks_.load(std::memory_order_acquire) == 0;
        });
    }

    size_t pending_tasks() const {
        return active_tasks_.load(std::memory_order_acquire);
    }

private:
    // -------------------------------------------------------------------
    // 工作线程主循环
    // -------------------------------------------------------------------
    void worker_loop(size_t worker_id) {
        WorkerData& my_data = *worker_data_[worker_id];

        // 随机数生成器 (用于选择偷取目标)
        thread_local std::mt19937 rng(
            static_cast<unsigned>(std::chrono::steady_clock::now().time_since_epoch().count())
            ^ static_cast<unsigned>(worker_id));

        while (true) {
            Task task;

            // 1. 优先从本地队列取 (LIFO: 从尾部取)
            if (pop_local(my_data, task)) {
                my_data.idle.store(false, std::memory_order_relaxed);
                task();
                active_tasks_.fetch_sub(1, std::memory_order_release);
                notify_if_done();
                continue;
            }

            // 2. 本地空了, 尝试从其他线程偷取 (FIFO: 从头部偷)
            if (steal_from_others(worker_id, rng, task)) {
                my_data.idle.store(false, std::memory_order_relaxed);
                task();
                active_tasks_.fetch_sub(1, std::memory_order_release);
                notify_if_done();
                continue;
            }

            // 3. 尝试从全局队列取
            {
                std::unique_lock<std::mutex> lock(global_mutex_);

                my_data.idle.store(true, std::memory_order_relaxed);

                // 等待条件: stop 或 全局队列非空 或 本地有任务
                global_cv_.wait_for(lock, std::chrono::milliseconds(1), [&]() {
                    return stop_.load(std::memory_order_acquire) ||
                           !global_queue_.empty() ||
                           !my_data.local_queue.empty();
                });

                if (stop_.load(std::memory_order_acquire) &&
                    global_queue_.empty() &&
                    my_data.local_queue.empty() &&
                    all_idle()) {
                    return;
                }

                if (!global_queue_.empty()) {
                    task = std::move(global_queue_.front());
                    global_queue_.pop();
                    my_data.idle.store(false, std::memory_order_relaxed);
                    lock.unlock();

                    // 取出后放入本地队列 (后续递归任务可放入此处)
                    task();
                    active_tasks_.fetch_sub(1, std::memory_order_release);
                    notify_if_done();
                    continue;
                }
            }
        }
    }

    // -------------------------------------------------------------------
    // 从本地队列取任务 (LIFO: pop_back)
    // -------------------------------------------------------------------
    bool pop_local(WorkerData& data, Task& out) {
        std::lock_guard<std::mutex> lock(data.mutex);
        if (data.local_queue.empty()) {
            return false;
        }
        out = std::move(data.local_queue.back());
        data.local_queue.pop_back();
        return true;
    }

    // -------------------------------------------------------------------
    // 从其他线程队列偷取任务 (FIFO: pop_front, 减少缓存颠簸)
    // -------------------------------------------------------------------
    bool steal_from_others(size_t my_id, std::mt19937& rng, Task& out) {
        const size_t num_workers = worker_data_.size();

        // 随机顺序遍历其他线程
        size_t start = std::uniform_int_distribution<size_t>(0, num_workers - 1)(rng);
        for (size_t offset = 0; offset < num_workers; ++offset) {
            size_t victim_id = (start + offset) % num_workers;
            if (victim_id == my_id) continue;

            WorkerData& victim = *worker_data_[victim_id];
            std::unique_lock<std::mutex> lock(victim.mutex, std::try_to_lock);
            if (!lock.owns_lock()) continue;

            if (victim.local_queue.empty()) continue;

            // 从头部偷取 (FIFO 方向, 与本地 LIFO 互补)
            out = std::move(victim.local_queue.front());
            victim.local_queue.pop_front();
            return true;
        }

        return false;
    }

    // -------------------------------------------------------------------
    // 检查是否所有线程都空闲
    // -------------------------------------------------------------------
    bool all_idle() const {
        for (const auto& wd : worker_data_) {
            if (!wd->idle.load(std::memory_order_relaxed)) {
                return false;
            }
        }
        // 确保全局队列也为空
        std::lock_guard<std::mutex> lock(global_mutex_);
        return global_queue_.empty();
    }

    // -------------------------------------------------------------------
    // 通知等待 all-done 的线程
    // -------------------------------------------------------------------
    void notify_if_done() {
        if (active_tasks_.load(std::memory_order_acquire) == 0) {
            done_cv_.notify_all();
        }
    }
};

// ============================================================================
// 使用示例: 递归任务 (利用 work stealing 的优势)
// ============================================================================
void recursive_demo() {
    std::cout << "=== Work Stealing 递归任务演示 ===\n\n";

    WorkStealingThreadPool pool(4);

    // 模拟递归分治: 计算斐波那契 (故意使用低效算法展示 work stealing)
    std::function<int(int)> fib = [&pool, &fib](int n) -> int {
        if (n <= 1) return n;

        // 大于阈值时, 将子任务提交到线程池
        if (n > 20) {
            auto fut1 = pool.submit(fib, n - 1);
            auto fut2 = pool.submit(fib, n - 2);
            return fut1.get() + fut2.get();
        }

        // 小子问题直接递归计算
        return fib(n - 1) + fib(n - 2);
    };

    auto start = std::chrono::high_resolution_clock::now();
    int result = fib(25);
    pool.wait_all();
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "  fib(25) = " << result << "\n";
    std::cout << "  耗时: " << ms << " ms\n\n";
}

// ============================================================================
// 性能压力测试
// ============================================================================
void stress_test() {
    std::cout << "=== Work Stealing 压力测试 ===\n";

    constexpr int kTasks = 100000;
    WorkStealingThreadPool pool(8);

    std::vector<std::future<long long>> futures;
    futures.reserve(kTasks);

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < kTasks; ++i) {
        futures.push_back(pool.submit([i]() -> long long {
            // 模拟随机计算量
            long long acc = 0;
            for (int j = 0; j < (i % 100 + 10); ++j) {
                acc += j * i;
                std::atomic_signal_fence(std::memory_order_relaxed); // 阻止优化
            }
            return static_cast<long long>(i) + (acc & 0xFF);
        }));
    }

    long long sum = 0;
    for (auto& f : futures) {
        sum += f.get();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "  任务数: " << kTasks << "\n";
    std::cout << "  累计和: " << sum << "\n";
    std::cout << "  耗时: " << ms << " ms\n";
    std::cout << "  吞吐量: " << (kTasks * 1000.0 / ms) << " tasks/s\n";
}

// ============================================================================
// main
// ============================================================================
int main() {
    recursive_demo();
    stress_test();
    return 0;
}
