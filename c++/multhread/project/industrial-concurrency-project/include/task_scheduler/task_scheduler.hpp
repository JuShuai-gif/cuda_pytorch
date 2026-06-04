#pragma once
// Ch8 & Ch9：任务调度器——AI/ML 推理的核心编排器
// 集成：
//   - ThreadPool（Ch9.1）用于并行执行
//   - PriorityTaskQueue（Ch6.2+）用于基于优先级的调度
//   - ConcurrentCache（Ch3.3.2）用于缓存推理结果
//   - StopToken（Ch9.2）用于优雅关闭
//   - Logger（Ch11）用于调试和监控
//
// 功能特性：
//   - Ch8.2：任务分解（将批次拆分到各工作线程）
//   - Ch8.3：基于 std::future 的延续风格任务链（Ch4.2.3）
//   - Ch4.2：类似 std::async 的"发射后不管"模式，带 future 跟踪
//   - Ch9.2：在任务边界的协作式中断
//   - Ch3.2：死锁避免——每个组件单锁设计

#include "task_scheduler/thread_pool.hpp"
#include "task_scheduler/priority_task_queue.hpp"
#include "task_scheduler/concurrent_cache.hpp"
#include "task_scheduler/stop_token.hpp"
#include "task_scheduler/logger.hpp"
#include <functional>
#include <future>
#include <vector>
#include <map>
#include <string>
#include <chrono>
#include <atomic>
#include <memory>

namespace task_scheduler {

// Ch8.5：用于调度决策的任务元数据。
// 任务元数据：名称、优先级、超时、提交时间
struct TaskMetadata {
    std::string name;
    TaskPriority priority{TaskPriority::NORMAL};
    std::chrono::milliseconds timeout{0}; // 0 表示无超时
    std::chrono::steady_clock::time_point submit_time;
};

// Ch8.1：TaskScheduler 编排线程池中的任务执行。
// 支持优先级队列、批处理和定时任务。
// 任务调度器：核心编排器，管理优先级队列和线程池
class TaskScheduler {
public:
    // 构造函数：指定线程数和缓存大小
    TaskScheduler(size_t num_threads = 0,
                  size_t cache_size = 1024);
    ~TaskScheduler();

    TaskScheduler(const TaskScheduler&) = delete;
    TaskScheduler& operator=(const TaskScheduler&) = delete;

    // Ch8.2.1：提交单个带优先级的任务。
    // 返回 std::future 用于获取结果（Ch4.2.1）。
    // 提交单个任务：指定优先级、名称、可调用对象和参数
    template <typename F, typename... Args>
        requires std::invocable<F, Args...>
    auto submit(TaskPriority priority, std::string_view name,
                F&& f, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>> {
        using ResultType = std::invoke_result_t<F, Args...>;

        // 用 packaged_task 包装可调用对象，连接到 future
        auto task = std::make_shared<std::packaged_task<ResultType()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        std::future<ResultType> result = task->get_future();

        // 构建任务元数据
        TaskMetadata meta;
        meta.name = name;
        meta.priority = priority;
        meta.submit_time = std::chrono::steady_clock::now();
        meta.timeout = default_timeout_;

        // Ch8.3：入队到优先级队列。
        // 将任务包装后推入优先级队列
        priority_queue_.push(
            [this, task = std::move(task), meta]() mutable {
                Logger::instance().debug(
                    TS_FORMAT("TaskScheduler: executing '{}'", meta.name));
                try {
                    (*task)();
                } catch (const std::exception& e) {
                    Logger::instance().error(
                        TS_FORMAT("Task '{}' failed: {}", meta.name, e.what()));
                    throw;
                }
            },
            priority);

        // Ch8.4.1：分发到线程池执行。
        // 触发分发：将优先级队列中的任务移动到线程池
        dispatch_pending();
        return result;
    }

    // Ch8.2.2：提交一批同优先级的任务。
    // 返回一个 future 向量（Ch4.2.4：shared_future 的替代方案）。
    // 批量提交：一次性提交 count 个相同优先级的任务
    template <typename F, typename... Args>
        requires std::invocable<F, Args...>
    auto submit_batch(TaskPriority priority, std::string_view name_prefix,
                      size_t count, F&& f, Args&&... args)
        -> std::vector<std::future<std::invoke_result_t<F, Args...>>> {
        using ResultType = std::invoke_result_t<F, Args...>;
        std::vector<std::future<ResultType>> futures;
        futures.reserve(count);

        for (size_t i = 0; i < count; ++i) {
            auto task = std::make_shared<std::packaged_task<ResultType()>>(
                std::bind(std::forward<F>(f), std::forward<Args>(args)...));

            futures.push_back(task->get_future());

            std::string task_name = TS_FORMAT("{}_{}", name_prefix, i);
            // 值捕获：task（shared_ptr）和 task_name
            priority_queue_.push(
                [this, task = std::move(task), task_name]() mutable {
                    (*task)();
                },
                priority);
        }

        dispatch_pending();
        return futures;
    }

    // Ch8.3.1：流水线执行——将多个阶段按依赖关系链接。
    // 阶段1 -> 阶段2 -> 阶段3。每个阶段的输出作为下个阶段的输入。
    // 返回最终阶段结果的 future（Ch4.2.2：future 链式调用）。
    // 流水线执行：两阶段任务链
    template <typename Input, typename Output>
    auto submit_pipeline(std::string_view pipeline_name,
                         std::function<Input()> stage1,
                         std::function<Output(Input)> stage2)
        -> std::future<Output> {
        // Ch4.2.3：为简化起见使用 std::async，或提交到线程池以获取更多控制。
        // 这里通过 .then() 模式演示延续调用（Ch4.4.3）。
        // 使用 promise/future 实现阶段间的延续
        auto promise = std::make_shared<std::promise<Output>>();
        auto result = promise->get_future();

        pool_->submit([this, promise, stage1 = std::move(stage1),
                       stage2 = std::move(stage2), pipeline_name]() mutable {
            try {
                Logger::instance().debug(
                    TS_FORMAT("Pipeline '{}': stage1 executing", pipeline_name));
                Input intermediate = stage1();

                Logger::instance().debug(
                    TS_FORMAT("Pipeline '{}': stage2 executing", pipeline_name));
                Output final_result = stage2(std::move(intermediate));

                promise->set_value(std::move(final_result));
            } catch (...) {
                promise->set_exception(std::current_exception());
            }
        });

        return result;
    }

    // Ch8.5.1：定时任务提交（Ch4.1.3：定时等待模式）。
    // 返回一个可停止该定时任务的 stop_source。
    // 定时任务：以固定间隔重复提交回调
    using PeriodicCallback = std::function<void()>;
    stop_source schedule_periodic(PeriodicCallback callback,
                                   std::chrono::milliseconds interval,
                                   TaskPriority priority = TaskPriority::NORMAL,
                                   std::string_view name = "periodic");

    // Ch9.2.1：关闭调度器。
    // 关闭调度器：请求所有子系统停止
    void shutdown();

    // Ch9.2.2：检查调度器是否正在关闭。
    [[nodiscard]] bool is_stopping() const;

    // Ch8.5.2：缓存访问，用于推理结果复用（Ch3.3.2）。
    // 缓存操作：put/get
    template <typename K, typename V>
    void cache_put(const K& key, const V& value) {
        cache_.put(key, value);
    }

    template <typename K, typename V>
    std::optional<V> cache_get(const K& key) {
        return cache_.get(key);
    }

    // Ch9.1.6：等待所有当前任务完成。
    // 等待所有任务完成
    void wait_for_all() { pool_->wait_for_tasks(); }

    // Ch8.5.3：统计信息。
    // 统计：待处理任务数和活跃线程数
    size_t pending_count() const { return priority_queue_.size(); }
    size_t active_threads() const { return pool_->worker_count(); }

private:
    // Ch8.4.1：从优先级队列分发任务到线程池。
    // 分发待处理任务到线程池
    void dispatch_pending();

    // Ch8.4.2：用于定时任务的持续分发循环。
    // 定时任务分发循环
    void dispatch_loop();

    std::unique_ptr<ThreadPool> pool_;
    PriorityTaskQueue<std::function<void()>> priority_queue_;
    ConcurrentCache<std::string, std::string> cache_;

    std::chrono::milliseconds default_timeout_{0};
    std::atomic<bool> running_{true};

    // Ch9.2：定时任务的停止机制。
    stop_source scheduler_stop_source_;
};

} // namespace task_scheduler
