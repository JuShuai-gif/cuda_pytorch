// Ch8 & Ch9：任务调度器实现
// 实现分发逻辑和定时任务调度。

#include "task_scheduler/task_scheduler.hpp"
#include <chrono>
#include "task_scheduler/format_compat.hpp"

namespace task_scheduler {

// 构造函数：初始化线程池和缓存
TaskScheduler::TaskScheduler(size_t num_threads, size_t cache_size)
    : pool_(std::make_unique<ThreadPool>(num_threads))
    , cache_(cache_size) {
    Logger::instance().info(TS_FORMAT(
        "TaskScheduler initialized with {} threads, cache size {}",
        pool_->worker_count(), cache_size));
}

// 析构函数：调用 shutdown 进行优雅关闭
TaskScheduler::~TaskScheduler() {
    shutdown();
}

// Ch8.4.1：从优先级队列分发待处理任务到线程池。
// 每次提交后调用以启动执行。
// 分发待处理任务：从优先级队列取出，轮询分发到工作线程
void TaskScheduler::dispatch_pending() {
    // Ch8.4.2：将优先级队列排空到线程池。
    // 只要队列中有任务就持续分发，并以轮询方式
    // 分发到各工作线程以实现负载均衡。
    size_t dispatched = 0;
    while (true) {
        auto item = priority_queue_.try_pop();
        if (!item) break;

        // Ch9.1.2：分发到特定工作线程以实现负载均衡。
        // 使用跨工作线程的轮询（Ch8.4.5：均匀分布）。
        pool_->submit_to_local(
            dispatched % pool_->worker_count(),
            std::move(item->data));
        ++dispatched;
    }
}

// Ch8.5.1：定时任务调度。
// 使用带有定时等待的后台线程（Ch4.1.3）。
// 定时任务：创建一个后台线程，按固定间隔提交回调
stop_source TaskScheduler::schedule_periodic(
    PeriodicCallback callback,
    std::chrono::milliseconds interval,
    TaskPriority priority,
    std::string_view name) {

    stop_source periodic_stop;
    auto st = periodic_stop.get_token();
    std::string task_name(name);

    // Ch9.1.3：为定时任务启动一个专用线程。
    std::jthread([this, cb = std::move(callback), interval, priority,
                  task_name, st = std::move(st)]() mutable {
        Logger::instance().debug(TS_FORMAT(
            "Periodic task '{}' started (interval: {}ms)",
            task_name, interval.count()));

        while (!st.stop_requested()) {
            // Ch8.5.2：将定时回调作为高优先级任务提交。
            // 我们不等待它——发射后不管。
            if (!scheduler_stop_source_.stop_requested()) {
                submit(priority, task_name, cb);
            }

            // Ch4.1.3：带停止检查的定时等待。
            // 避免忙等待，同时允许响应式停止。
            if (!st.wait_for(interval)) {
                // 超时正常到期，继续循环
                continue;
            }
            break; // 已请求停止
        }

        Logger::instance().debug(
            TS_FORMAT("Periodic task '{}' stopped", task_name));
    }).detach(); // Ch2.3：分离线程——它通过 stop_token 自行清理。

    return periodic_stop;
}

// 关闭调度器：请求所有子系统停止
void TaskScheduler::shutdown() {
    if (!running_.exchange(false)) return; // 已经停止（Ch5.3.2：原子标志）

    Logger::instance().info("TaskScheduler shutting down...");

    // Ch9.2.1：通知所有组件停止。
    scheduler_stop_source_.request_stop();

    // Ch9.2.3：通知优先级队列等待者。
    priority_queue_.notify_all();

    // Ch9.1.7：线程池析构函数处理线程 join。
    pool_->shutdown();

    Logger::instance().info("TaskScheduler shutdown complete");
}

// 检查调度器是否正在关闭
bool TaskScheduler::is_stopping() const {
    return scheduler_stop_source_.stop_requested();
}

} // namespace task_scheduler
