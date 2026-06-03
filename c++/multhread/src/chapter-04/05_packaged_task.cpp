// 05_packaged_task.cpp - std::packaged_task 将可调用对象包装为异步任务
// packaged_task 绑定到 future/promise，可与线程池配合使用

#include <cmath>
#include <deque>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

// ===== 简易线程池（仅用于演示 packaged_task） =====
class SimpleThreadPool {
public:
    explicit SimpleThreadPool(size_t num_threads) {
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this]() { worker_loop(); });
        }
    }

    ~SimpleThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cond_var_.notify_all();
        // jthread 自动 join
    }

    // 提交一个任务，返回 future 以获取结果
    template <typename F>
    auto submit(F&& f) -> std::future<decltype(f())> {
        using result_type = decltype(f());

        // 用 packaged_task 包装可调用对象
        auto task = std::make_shared<std::packaged_task<result_type()>>(std::forward<F>(f));
        std::future<result_type> future = task->get_future();

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stop_) throw std::runtime_error("线程池已停止，无法提交任务");
            tasks_.push_back([task]() { (*task)(); });
        }
        cond_var_.notify_one();
        return future;
    }

private:
    void worker_loop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cond_var_.wait(lock, [this]() { return stop_ || !tasks_.empty(); });

                if (stop_ && tasks_.empty()) return;

                task = std::move(tasks_.front());
                tasks_.pop_front();
            }
            task();
        }
    }

    std::vector<std::jthread>          workers_;
    std::deque<std::function<void()>>  tasks_;
    std::mutex                         mutex_;
    std::condition_variable            cond_var_;
    bool                               stop_ = false;
};

// ===== 演示 standalone packaged_task （也可直接绑定线程） =====
void demo_standalone() {
    std::cout << "=== standalone packaged_task ===\n";

    // 包装一个 lambda
    std::packaged_task<int(int, int)> task([](int a, int b) {
        std::cout << "[Task] 计算 " << a << " + " << b << "\n";
        return a + b;
    });

    std::future<int> result = task.get_future();

    // task 只能移动不能复制，显式创建线程执行
    std::jthread worker(std::move(task), 10, 20);

    std::cout << "[Main] 结果: " << result.get() << "\n\n";
}

// 演示与线程池配合
void demo_with_threadpool() {
    std::cout << "=== 线程池 + packaged_task ===\n";
    SimpleThreadPool pool(4);

    // 提交多个任务
    std::vector<std::future<int>> results;
    for (int i = 0; i < 8; ++i) {
        results.push_back(pool.submit([i]() -> int {
            std::cout << "[Worker] 处理任务 " << i << "\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            return i * i;
        }));
    }

    // 收集结果
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << "[Main] 任务 " << i << " 结果: " << results[i].get() << "\n";
    }
}

int main() {
    demo_standalone();
    demo_with_threadpool();
    return 0;
}
