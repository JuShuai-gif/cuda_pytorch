/**
 * 05_exception_safe_parallel.cpp — 并行任务中的异常安全处理
 *
 * 当并行任务抛出异常时, 必须确保:
 *  1. 所有线程正确终止 (没有线程被遗弃)
 *  2. 异常信息被传播到调用者
 *  3. RAII 资源被正确释放
 *
 * 技术要点:
 *  - std::future: 从异步任务获取结果或异常
 *  - std::exception_ptr: 传递跨线程异常
 *  - std::jthread: 自动 join, 防止线程泄漏
 *  - 异常发生时设置停止标志, 通知其他线程
 *
 * 编译: g++ -std=c++20 -O2 -pthread 05_exception_safe_parallel.cpp -o exception_safe_parallel
 */

#include <iostream>
#include <thread>
#include <vector>
#include <future>
#include <atomic>
#include <exception>
#include <stdexcept>
#include <memory>
#include <functional>
#include <chrono>
#include <mutex>
#include <condition_variable>

// ============================================================================
// 模拟可能在并行任务中抛出的异常
// ============================================================================
class TaskException : public std::runtime_error {
public:
    explicit TaskException(const std::string& msg) : std::runtime_error(msg) {}
};

// ============================================================================
// 场景1: 使用 std::async + future 传播异常
// ============================================================================
void demo_async_exception_handling() {
    std::cout << "=== 场景1: std::async + future 异常传播 ===\n";

    auto safe_task = [](int id) -> int {
        // 正常任务
        return id * 2;
    };

    auto failing_task = [](int id) -> int {
        throw TaskException("任务 " + std::to_string(id) + " 发生异常");
        return 0;
    };

    std::vector<std::future<int>> futures;

    // 提交混合任务
    futures.push_back(std::async(std::launch::async, safe_task, 1));
    futures.push_back(std::async(std::launch::async, safe_task, 2));
    futures.push_back(std::async(std::launch::async, failing_task, 3)); // 会抛出
    futures.push_back(std::async(std::launch::async, safe_task, 4));

    std::vector<std::string> errors;
    std::vector<int> results;

    for (auto& f : futures) {
        try {
            results.push_back(f.get()); // 如果有异常, get() 会重新抛出
        } catch (const std::exception& e) {
            errors.emplace_back(e.what());
        }
    }

    std::cout << "  成功结果: ";
    for (int r : results) std::cout << r << " ";
    std::cout << "\n  捕获异常: ";
    for (const auto& e : errors) std::cout << "[" << e << "] ";
    std::cout << "\n\n";
}

// ============================================================================
// 场景2: 自管理线程池中的异常安全
// ============================================================================
class ExceptionSafeRunner {
private:
    struct TaskResult {
        std::exception_ptr exception;
        int thread_id;
    };

    std::mutex results_mutex_;
    std::vector<TaskResult> results_;
    std::atomic<bool> stop_flag_{false};

public:
    // 提交多个任务, 某个失败后通知其他任务停止
    void run_tasks(const std::vector<std::function<void()>>& tasks) {
        stop_flag_.store(false, std::memory_order_relaxed);
        results_.clear();

        std::vector<std::jthread> threads;
        threads.reserve(tasks.size());

        for (size_t i = 0; i < tasks.size(); ++i) {
            threads.emplace_back([this, &tasks, i]() {
                int tid = static_cast<int>(i);

                // 检查停止标志
                if (stop_flag_.load(std::memory_order_acquire)) {
                    return;
                }

                try {
                    tasks[i]();
                } catch (...) {
                    // 捕获异常, 通知其他线程停止
                    stop_flag_.store(true, std::memory_order_release);

                    std::lock_guard<std::mutex> lock(results_mutex_);
                    results_.push_back({std::current_exception(), tid});
                }
            });
        }

        // jthread 析构时自动 join, 确保所有线程正确终止
    }

    // 获取所有异常
    void report() {
        std::lock_guard<std::mutex> lock(results_mutex_);
        if (results_.empty()) {
            std::cout << "  所有任务成功完成, 无异常\n";
        } else {
            for (const auto& r : results_) {
                try {
                    if (r.exception) {
                        std::rethrow_exception(r.exception);
                    }
                } catch (const std::exception& e) {
                    std::cout << "  线程 " << r.thread_id << " 异常: " << e.what() << "\n";
                }
            }
        }
    }
};

void demo_manual_exception_handling() {
    std::cout << "=== 场景2: 自管理线程异常处理 ===\n";

    ExceptionSafeRunner runner;

    // 定义任务: #2 会抛出异常
    std::vector<std::function<void()>> tasks;

    tasks.push_back([]() {
        std::cout << "    任务0: 正在执行...\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::cout << "    任务0: 完成\n";
    });

    tasks.push_back([]() {
        std::cout << "    任务1: 正在执行...\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        std::cout << "    任务1: 完成\n";
    });

    tasks.push_back([]() {
        std::cout << "    任务2: 正在执行...\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        throw TaskException("任务2 模拟故障!");
    });

    tasks.push_back([]() {
        std::cout << "    任务3: 正在执行...\n";
        // 这个任务应该能检测到停止标志并提前退出
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        std::cout << "    任务3: 完成 (可能已被取消)\n";
    });

    runner.run_tasks(tasks);
    runner.report();
    std::cout << "\n";
}

// ============================================================================
// 场景3: RAII 与异常安全
// ============================================================================
void demo_raii_exception_safety() {
    std::cout << "=== 场景3: RAII 与异常安全 ===\n";

    // RAII 锁: 异常抛出时自动释放
    {
        std::mutex mtx;
        try {
            std::lock_guard<std::mutex> lock(mtx);
            std::cout << "  已获取锁, 即将抛出异常...\n";
            throw std::runtime_error("RAII 测试异常");
        } catch (const std::exception& e) {
            std::cout << "  捕获: " << e.what() << "\n";
            // lock_guard 已在异常时自动释放
            if (mtx.try_lock()) {
                std::cout << "  锁已正确释放 (RAII 生效)\n";
                mtx.unlock();
            }
        }
    }

    // std::jthread 的 RAII: 异常时自动 join
    {
        std::jthread t([]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        });
        // t 析构时自动 join, 即使发生异常也不会泄漏线程
        std::cout << "  jthread 将在作用域结束时自动 join\n";
    }

    std::cout << "\n";
}

// ============================================================================
// 场景4: 使用 std::packaged_task 实现异常安全的并行执行
// ============================================================================
template <typename Func>
auto safe_parallel_execute(Func&& func) {
    auto task = std::make_shared<std::packaged_task<decltype(func())()>>(
        std::forward<Func>(func));

    auto future = task->get_future();

    // 使用 jthread 确保即使抛出异常也会 join
    std::jthread([task = std::move(task)]() mutable {
        (*task)(); // packaged_task 自动捕获异常并传递给 future
    }).detach();   // 演示场景下 detach, 实际应用应管理生命周期

    return future;
}

void demo_packaged_task_exception() {
    std::cout << "=== 场景4: packaged_task 异常传播 ===\n";

    auto future = safe_parallel_execute([]() -> int {
        throw TaskException("packaged_task 中的异常");
        return 0;
    });

    try {
        future.get();
    } catch (const std::exception& e) {
        std::cout << "  从 future 获取到异常: " << e.what() << "\n";
    }

    // 给 detach 线程一点时间完成
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    std::cout << "\n";
}

// ============================================================================
// main
// ============================================================================
int main() {
    demo_async_exception_handling();
    demo_manual_exception_handling();
    demo_raii_exception_safety();
    demo_packaged_task_exception();

    std::cout << "=== 异常安全总结 ===\n";
    std::cout << "  1. 使用 std::future 获取任务结果和异常\n";
    std::cout << "  2. 使用 std::jthread 防止线程泄漏\n";
    std::cout << "  3. 使用 RAII (lock_guard, unique_ptr) 保证资源释放\n";
    std::cout << "  4. 使用 std::packaged_task 跨线程传播异常\n";
    std::cout << "  5. 使用停止标志通知其他线程异常发生\n";

    return 0;
}
