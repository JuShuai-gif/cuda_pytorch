// A small thread pool (synthesized from Ch10 concurrency building blocks).
//
// Fixed number of worker threads pull std::function tasks from a shared
// queue guarded by a mutex + condition variable. submit() returns a
// std::future so callers can collect results and exceptions.

#ifndef CHP_TASK_SYSTEM_HPP
#define CHP_TASK_SYSTEM_HPP

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace chp {

class TaskSystem {
public:
    // num_threads == 0 means hardware_concurrency() (clamped to >= 1).
    explicit TaskSystem(std::size_t num_threads = 0) {
        const auto hw = std::thread::hardware_concurrency();
        const auto n = num_threads == 0
                           ? (hw == 0 ? 1 : hw)
                           : num_threads;
        workers_.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    ~TaskSystem() {
        {
            std::lock_guard<std::mutex> lock{mutex_};
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : workers_) {
            t.join();
        }
    }

    TaskSystem(const TaskSystem&) = delete;
    TaskSystem& operator=(const TaskSystem&) = delete;

    // Submit a callable; returns a future for its result.
    template <typename Fn, typename... Args>
    auto submit(Fn&& fn, Args&&... args) {
        using Result = std::invoke_result_t<Fn, Args...>;
        auto task = std::make_shared<std::packaged_task<Result()>>(
            std::bind(std::forward<Fn>(fn), std::forward<Args>(args)...));
        auto future = task->get_future();
        {
            std::lock_guard<std::mutex> lock{mutex_};
            if (stop_) {
                throw std::runtime_error("submit on stopped TaskSystem");
            }
            tasks_.emplace([task] { (*task)(); });
        }
        cv_.notify_one();
        return future;
    }

    std::size_t size() const { return workers_.size(); }

private:
    void worker_loop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock{mutex_};
                cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                if (stop_ && tasks_.empty()) {
                    return;
                }
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();
        }
    }

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<std::function<void()>> tasks_;
    bool stop_ = false;
};

}  // namespace chp

#endif  // CHP_TASK_SYSTEM_HPP
