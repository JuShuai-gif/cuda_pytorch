#pragma once
// Ch9.1：线程池——高级线程管理
// 实现一个固定大小的线程池，包含：
//   - Ch8.4：用于负载均衡的工作窃取
//   - Ch9.1.1：submit() 返回 std::future（通过 std::packaged_task）
//   - Ch9.2：用于协作式中断的 stop_token
//   - Ch4.2：std::future/std::promise 用于结果传播
//   - Ch4.4：std::packaged_task 包装可调用对象
//   - Ch3.2：std::mutex 用于队列保护
//   - Ch4.1：std::condition_variable 用于工作线程唤醒

#include "task_scheduler/task_queue.hpp"
#include "task_scheduler/stop_token.hpp"
#include <vector>
#include <thread>
#include <future>
#include <functional>
#include <type_traits>
#include <memory>
#include <atomic>
#include <random>

namespace task_scheduler {

// 线程池：固定大小，每线程本地队列，支持工作窃取
class ThreadPool {
public:
    // Ch9.1：构造函数启动 num_threads 个工作线程。
    // 如果 num_threads == 0，使用 hardware_concurrency（Ch8.4.1）。
    explicit ThreadPool(size_t num_threads = 0);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    // Ch9.1.1：提交一个可调用对象并获取结果的 std::future。
    // 内部使用 std::packaged_task（Ch4.4.1）。
    // 支持任意可调用类型和参数类型（Ch9.1.3：可变参数模板）。
    // 提交任务：返回 future 供调用者获取结果
    template <typename F, typename... Args>
        requires std::invocable<F, Args...>
    auto submit(F&& f, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>> {
        using ResultType = std::invoke_result_t<F, Args...>;

        // Ch4.4.1：packaged_task 包装可调用对象并连接到一个 future。
        // 使用 shared_ptr 是因为 packaged_task 是 move-only 的，而我们需要
        // 将其存储在 std::function 中（后者要求可拷贝）。
        // 用 shared_ptr 包装 packaged_task（因为 std::function 要求可拷贝）
        auto task = std::make_shared<std::packaged_task<ResultType()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        std::future<ResultType> result = task->get_future();

        // Ch9.1.2：提交到全局队列或某个工作线程的本地队列。
        // 为简化起见推送到全局队列；生产环境会使用
        // 轮询分发到本地队列（参见 submit_to_local()）。
        // 推送到全局任务队列
        {
            std::lock_guard lock(queue_mutex_);
            if (stop_source_.stop_requested()) {
                throw std::runtime_error("ThreadPool: cannot submit to stopped pool");
            }
            active_tasks_.fetch_add(1, std::memory_order_release);
            // Ch4.4.2：通过 std::function<void()> 进行类型擦除以统一存储。
            global_queue_.push([task]() { (*task)(); });
        }
        cond_.notify_one();
        return result;
    }

    // Ch9.1.4：提交到特定工作线程的本地队列（用于工作分发）。
    // 提交到指定工作线程的本地队列
    template <typename F, typename... Args>
        requires std::invocable<F, Args...>
    auto submit_to_local(size_t worker_idx, F&& f, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>> {
        using ResultType = std::invoke_result_t<F, Args...>;

        auto task = std::make_shared<std::packaged_task<ResultType()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        std::future<ResultType> result = task->get_future();

        if (worker_idx >= workers_.size()) {
            throw std::out_of_range("ThreadPool: worker index out of range");
        }
        active_tasks_.fetch_add(1, std::memory_order_release);
        workers_[worker_idx]->local_queue.push([task]() { (*task)(); });

        return result;
    }

    // Ch9.1.5：等待直到所有已提交的任务完成（不停止线程池）。
    // 等待所有已提交任务完成
    void wait_for_tasks();

    // Ch9.2.1：请求停止并等待所有线程结束（优雅关闭）。
    // 关闭线程池：请求停止并等待线程 join
    void shutdown();

    // Ch9.2.2：检查是否已请求停止。
    [[nodiscard]] bool is_stopping() const {
        return stop_source_.stop_requested();
    }

    // Ch9.1.6：工作线程数量。
    [[nodiscard]] size_t worker_count() const { return workers_.size(); }

    // Ch9.1.7：待处理任务的大致数量。
    [[nodiscard]] size_t pending_tasks() const;

private:
    // Ch9.1.8：每工作线程的状态。
    // 每个工作线程有自己的本地队列用于工作窃取（Ch8.4.2）。
    // 工作线程的结构定义
    struct Worker {
        size_t index;
        // Ch9.2.3：jthread（C++20）原生支持协作式中断。
        // 我们使用自己的 stop_token 包装器做教学演示。
        std::jthread thread;
        TaskQueue<std::function<void()>> local_queue;
        std::atomic<bool> running{true};
    };

    // Ch9.1.9：主工作循环（在每个线程中运行）。
    // 工作线程主循环
    void worker_loop(size_t worker_idx);

    // Ch8.4.3：工作窃取——尝试从其他工作线程的本地队列获取任务。
    // 工作窃取函数
    bool steal_task(size_t worker_idx, std::function<void()>& task);

    // Ch9.1.10：按优先级尝试获取任务：先自己本地队列，再全局队列，最后窃取。
    // 获取任务：本地 -> 全局 -> 窃取的优先级顺序
    bool get_task(size_t worker_idx, std::function<void()>& task, stop_token st);

    // Ch8.4.1：活动任务计数器，用于 wait_for_tasks。
    std::atomic<size_t> active_tasks_{0};

    std::vector<std::unique_ptr<Worker>> workers_;

    // Ch9.1.11：用于初始提交的全局任务队列。
    // 全局任务队列：初始提交的目标
    TaskQueue<std::function<void()>> global_queue_;
    mutable std::mutex queue_mutex_; // 保护 global_queue_ 同步
    std::condition_variable cond_;

    stop_source stop_source_;

    // Ch9.2.1：信号表示所有工作线程已初始化（防止初始化期间的窃取）。
    std::atomic<bool> pool_ready_{false};
};

} // namespace task_scheduler
