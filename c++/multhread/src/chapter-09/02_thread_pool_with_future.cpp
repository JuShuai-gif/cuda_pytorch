/**
 * 02_thread_pool_with_future.cpp — 工业级线程池: 返回 std::future 的 submit
 *
 * 扩展简单线程池, 支持:
 *  - submit 返回 std::future<ResultType>, 可获取异步结果
 *  - 使用 std::packaged_task 封装可调用对象
 *  - 类型擦除: 任务队列存储 std::function<void()>
 *  - 异常传播: packaged_task 自动将异常存入 future
 *
 * 编译: g++ -std=c++20 -O2 -pthread 02_thread_pool_with_future.cpp -o thread_pool_future
 */

#include <iostream>
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <atomic>
#include <chrono>
#include <memory>
#include <type_traits>

// ============================================================================
// FutureThreadPool — 支持 future 的线程池
// ============================================================================
class FutureThreadPool {
private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;

    mutable std::mutex queue_mutex_;
    std::condition_variable cv_;
    std::atomic<bool> stop_{false};

public:
    explicit FutureThreadPool(size_t num_threads = 0) {
        if (num_threads == 0) {
            num_threads = std::thread::hardware_concurrency();
        }
        if (num_threads == 0) num_threads = 2;

        workers_.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back(&FutureThreadPool::worker_loop, this);
        }

        std::cout << "线程池 (带 future) 已创建, 工作线程数: " << num_threads << "\n";
    }

    ~FutureThreadPool() {
        join();
    }

    FutureThreadPool(const FutureThreadPool&) = delete;
    FutureThreadPool& operator=(const FutureThreadPool&) = delete;

    // -------------------------------------------------------------------
    // submit — 提交任务并返回 future
    // 支持普通函数、lambda、std::function 等任何可调用对象
    // -------------------------------------------------------------------
    template <typename Func, typename... Args>
    auto submit(Func&& func, Args&&... args)
        -> std::future<typename std::invoke_result_t<Func, Args...>>
    {
        using ResultType = typename std::invoke_result_t<Func, Args...>;

        // 将调用封装为 packaged_task
        // shared_ptr 保证任务生命周期 (因为 packaged_task 不可拷贝)
        auto task = std::make_shared<std::packaged_task<ResultType()>>(
            std::bind(std::forward<Func>(func), std::forward<Args>(args)...)
        );

        std::future<ResultType> result = task->get_future();

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (stop_.load(std::memory_order_acquire)) {
                throw std::runtime_error("线程池已停止, 无法提交任务");
            }
            // 类型擦除: packaged_task<void()> 通过 lambda 间接调用
            tasks_.emplace([task = std::move(task)]() {
                (*task)();
            });
        }

        cv_.notify_one();
        return result;
    }

    // -------------------------------------------------------------------
    // join — 停止线程池
    // -------------------------------------------------------------------
    void join() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (stop_.load(std::memory_order_acquire)) return;
            stop_.store(true, std::memory_order_release);
        }
        cv_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) worker.join();
        }
    }

    size_t pending_tasks() const {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        return tasks_.size();
    }

    size_t worker_count() const { return workers_.size(); }

private:
    void worker_loop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                cv_.wait(lock, [this]() {
                    return stop_.load(std::memory_order_acquire) || !tasks_.empty();
                });
                if (stop_.load(std::memory_order_acquire) && tasks_.empty()) {
                    return;
                }
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            if (task) task();
        }
    }
};

// ============================================================================
// 使用示例
// ============================================================================
void usage_demo() {
    std::cout << "=== 带 future 的线程池使用示例 ===\n\n";

    FutureThreadPool pool(4);
    std::vector<std::future<int>> futures;

    // 提交多个任务
    for (int i = 0; i < 10; ++i) {
        futures.push_back(pool.submit([i]() -> int {
            // 模拟耗时计算
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            return i * i;
        }));
    }

    std::cout << "  已提交 " << futures.size() << " 个任务\n";

    // 收集结果
    for (size_t i = 0; i < futures.size(); ++i) {
        int result = futures[i].get(); // 阻塞直到结果就绪
        std::cout << "  任务 " << i << " 结果: " << result << "\n";
    }

    std::cout << "\n";
}

// ============================================================================
// 异常传播演示
// ============================================================================
void exception_demo() {
    std::cout << "=== future 异常传播演示 ===\n\n";

    FutureThreadPool pool(2);

    auto normal_future = pool.submit([]() -> int {
        return 42;
    });

    auto exception_future = pool.submit([]() -> int {
        throw std::runtime_error("任务内部异常!");
        return 0;
    });

    // 获取正常结果
    try {
        int val = normal_future.get();
        std::cout << "  正常任务结果: " << val << "\n";
    } catch (const std::exception& e) {
        std::cout << "  意外异常: " << e.what() << "\n";
    }

    // 获取异常结果
    try {
        exception_future.get();
    } catch (const std::exception& e) {
        std::cout << "  捕获到异常: " << e.what() << "\n";
    }

    std::cout << "\n";
}

// ============================================================================
// 性能压力测试
// ============================================================================
void stress_test() {
    std::cout << "=== 带 future 线程池压力测试 ===\n";

    constexpr int kTasks = 50000;
    FutureThreadPool pool(8);

    std::vector<std::future<long long>> futures;
    futures.reserve(kTasks);

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < kTasks; ++i) {
        futures.push_back(pool.submit([i]() -> long long {
            return static_cast<long long>(i);
        }));
    }

    long long sum = 0;
    for (auto& f : futures) {
        sum += f.get();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    long long expected = static_cast<long long>(kTasks) * (kTasks - 1) / 2;

    std::cout << "  任务数: " << kTasks << "\n";
    std::cout << "  期望和: " << expected << "\n";
    std::cout << "  实际和: " << sum << "\n";
    std::cout << "  耗时: " << elapsed << " ms\n";
    std::cout << "  正确性: " << (expected == sum ? "通过" : "失败") << "\n";
}

// ============================================================================
// main
// ============================================================================
int main() {
    usage_demo();
    exception_demo();
    stress_test();
    return 0;
}
