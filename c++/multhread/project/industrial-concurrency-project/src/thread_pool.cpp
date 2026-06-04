// Ch9：线程池实现
// 实现工作线程循环和工作窃取逻辑。
// 演示 Ch9.1（线程池设计）、Ch8.4（工作窃取）、
// Ch4.1（condition_variable）以及 Ch3.2（mutex 模式）。

#include "task_scheduler/thread_pool.hpp"
#include "task_scheduler/logger.hpp"
#include <random>
#include <algorithm>
#include <cassert>

namespace task_scheduler {

// Ch9.1：构造函数启动工作线程。
// Ch8.4.1：如果 num_threads 为 0，使用 hardware_concurrency。
// 构造函数：启动指定数量的工作线程
ThreadPool::ThreadPool(size_t num_threads) {
    size_t count = num_threads > 0 ? num_threads : std::thread::hardware_concurrency();
    count = std::max(count, size_t(1));

    auto st = stop_source_.get_token();

    // 预分配以防止线程启动时 vector 重新分配。
    workers_.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        // 首先 push Worker，确保 workers_[i] 在线程访问它之前存在。
        workers_.push_back(std::make_unique<Worker>());
        workers_.back()->index = i;

        // Ch9.1.2：以工作循环启动每个工作线程。
        workers_.back()->thread = std::jthread([this, i, st] {
            worker_loop(i);
        });
    }

    // 信号表明所有工作线程已初始化，此时可以进行工作窃取（Ch9.2.1）。
    pool_ready_.store(true, std::memory_order_release);
}

// Ch9.1.7：析构函数调用 shutdown 进行优雅清理（RAII）。
// 析构函数：RAII 确保资源正确释放
ThreadPool::~ThreadPool() {
    shutdown();
}

// Ch9.2：优雅关闭过程。
// 关闭过程：请求停止 -> 唤醒所有线程 -> join
void ThreadPool::shutdown() {
    // Ch9.2.1：通知所有工作线程停止。
    stop_source_.request_stop();

    // Ch9.2.3：唤醒所有等待的工作线程，以便它们检查停止标志。
    // 唤醒所有等待线程，让它们检查停止标志
    {
        std::lock_guard lock(queue_mutex_);
    }
    cond_.notify_all();
    for (auto& w : workers_) {
        w->local_queue.notify_all();
    }
    global_queue_.notify_all();

    // Ch9.1.6：jthread 在析构函数中自动 join（RAII）。
    workers_.clear();
}

// Ch9.1.3：工作线程主循环。
// 每个工作线程运行此循环，直到请求停止且所有任务被排空。
// 工作线程主循环：获取任务 -> 执行 -> 检查停止
void ThreadPool::worker_loop(size_t worker_idx) {
    auto st = stop_source_.get_token();

    while (true) {
        std::function<void()> task;

        // Ch9.1.4：尝试获取一个任务。
        if (get_task(worker_idx, task, st)) {
            // Ch9.1.5：执行任务。
            try {
                task();
            } catch (...) {
                Logger::instance().error("Worker thread caught unhandled exception");
            }
            active_tasks_.fetch_sub(1, std::memory_order_release);
            continue;
        }

        // Ch9.2.2：没有可用任务且已请求停止——退出。
        if (st.stop_requested()) {
            break;
        }

        // Ch4.1.2：在 condition_variable 上等待新任务。
        // 等待新任务：超时 10ms 后重新检查
        {
            std::unique_lock lock(queue_mutex_);
            cond_.wait_for(lock, std::chrono::milliseconds(10));
        }
    }
}

// Ch8.4.3：工作窃取实现。
// 每个工作线程首先检查自己的本地队列，然后检查全局队列，
// 最后尝试从另一个随机工作线程的本地队列中窃取任务。
// 获取任务的优先级顺序：本地 -> 全局 -> 窃取
bool ThreadPool::get_task(size_t worker_idx, std::function<void()>& task,
                          stop_token st) {
    auto& local_q = workers_[worker_idx]->local_queue;

    // Ch8.4.2：步骤 1——检查自己的本地队列（最快路径，无竞争）。
    if (auto t = local_q.try_pop()) {
        task = std::move(*t);
        return true;
    }

    // Ch9.1.11：步骤 2——检查全局队列（中等竞争，单 mutex）。
    {
        std::lock_guard lock(queue_mutex_);
        if (auto t = global_queue_.try_pop()) {
            task = std::move(*t);
            return true;
        }
    }

    // Ch8.4.3：步骤 3——从邻居工作线程窃取任务。
    // 随机选择相比顺序窃取减少了竞争（Ch8.4.4）。
    // 仅在线程池完全初始化后进行窃取（pool_ready_ 标志，Ch9.2.1）。
    // 从随机受害者线程窃取
    if (workers_.size() > 1 && pool_ready_.load(std::memory_order_acquire)) {
        size_t start = rand() % workers_.size();
        for (size_t offset = 0; offset < workers_.size(); ++offset) {
            if (st.stop_requested()) return false;
            size_t victim = (start + offset) % workers_.size();
            if (victim == worker_idx) continue;

            // 从受害者的本地队列窃取一个任务。
            if (auto t = workers_[victim]->local_queue.try_pop()) {
                task = std::move(*t);
                return true;
            }
        }
    }

    return false;
}

// Ch8.4.2：窃取任务——get_task 的别名，仅窃取语义。
// 仅窃取：不检查本地或全局队列
bool ThreadPool::steal_task(size_t worker_idx, std::function<void()>& task) {
    if (workers_.size() <= 1) return false;

    size_t start = rand() % workers_.size();
    for (size_t offset = 1; offset < workers_.size(); ++offset) {
        size_t victim = (start + offset) % workers_.size();
        if (victim == worker_idx) continue;
        if (auto t = workers_[victim]->local_queue.try_pop()) {
            task = std::move(*t);
            return true;
        }
    }
    return false;
}

// Ch9.1.8：阻塞直到所有已提交任务完成（active_tasks_ == 0）。
// 等待所有任务完成：自旋等待 active_tasks_ 变为 0
void ThreadPool::wait_for_tasks() {
    // Ch5.3.1：以最小开销自旋等待，偶尔让步。
    while (active_tasks_.load(std::memory_order_acquire) > 0) {
        std::this_thread::yield();
        // Ch9.2.4：检查停止标志以避免无限等待。
        if (stop_source_.stop_requested()) break;
    }
}

// Ch9.1.9：待处理任务的大致数量（不完全精确，Ch6.2.7）。
// 获取待处理任务数（近似值，非精确）
size_t ThreadPool::pending_tasks() const {
    size_t count = 0;
    {
        std::lock_guard lock(queue_mutex_);
        count += global_queue_.size();
    }
    for (auto& w : workers_) {
        count += w->local_queue.size();
    }
    return count + active_tasks_.load(std::memory_order_acquire);
}

} // namespace task_scheduler
