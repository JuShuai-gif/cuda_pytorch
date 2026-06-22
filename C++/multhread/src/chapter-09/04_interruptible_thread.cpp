/**
 * 04_interruptible_thread.cpp — 可中断线程实现
 *
 * 当需要提前终止一个阻塞中的线程时, 不能使用 std::terminate 或强制 kill。
 * 正确方案: 在线程的等待点 (condition_variable wait, sleep 等) 检测中断标志。
 *
 * 技术要点:
 *  - interrupt_flag: 原子中断标志 + condition_variable
 *  - interruptible_wait: 在 condition_variable wait 中检测中断
 *  - interruption_point(): 手动插入检查点
 *  - 线程安全: 外部调用 interrupt() 唤醒等待中的线程
 *
 * 编译: g++ -std=c++20 -O2 -pthread 04_interruptible_thread.cpp -o interruptible_thread
 */

#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <chrono>
#include <future>
#include <exception>
#include <memory>
#include <queue>

// ============================================================================
// ThreadInterrupted — 自定义异常, 抛出表示线程被中断
// ============================================================================
class ThreadInterrupted : public std::runtime_error {
public:
    ThreadInterrupted() : std::runtime_error("线程被中断") {}
};

// ============================================================================
// InterruptFlag — 中断标志, 每个线程一个实例
// ============================================================================
class InterruptFlag {
private:
    std::atomic<bool> flag_{false};
    std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable* waiting_on_{nullptr}; // 当前等待的 CV

public:
    InterruptFlag() = default;
    InterruptFlag(const InterruptFlag&) = delete;
    InterruptFlag& operator=(const InterruptFlag&) = delete;

    // 设置中断标志, 并唤醒在 CV 上等待的线程
    void set() {
        flag_.store(true, std::memory_order_release);

        std::lock_guard<std::mutex> lock(mutex_);
        if (waiting_on_) {
            waiting_on_->notify_all();
        }
    }

    // 检查是否已被中断
    bool is_set() const {
        return flag_.load(std::memory_order_acquire);
    }

    // -------------------------------------------------------------------
    // 可中断的 condition_variable wait
    // 参数: 要等待的锁 和 外部 condition_variable
    // -------------------------------------------------------------------
    template <typename Lockable>
    void wait(std::condition_variable& cv, std::unique_lock<Lockable>& lock) {
        {
            std::lock_guard<std::mutex> guard(mutex_);
            waiting_on_ = &cv;
        }

        // 内部循环: 处理虚假唤醒和中途中断
        cv.wait(lock, [this]() {
            return flag_.load(std::memory_order_acquire);
        });

        // 被唤醒后抛出中断异常
        {
            std::lock_guard<std::mutex> guard(mutex_);
            waiting_on_ = nullptr;
        }
        throw ThreadInterrupted();
    }

    // -------------------------------------------------------------------
    // 可中断的 condition_variable wait_for (带超时)
    // -------------------------------------------------------------------
    template <typename Lockable, typename Rep, typename Period>
    bool wait_for(std::condition_variable& cv,
                  std::unique_lock<Lockable>& lock,
                  const std::chrono::duration<Rep, Period>& timeout) {
        {
            std::lock_guard<std::mutex> guard(mutex_);
            waiting_on_ = &cv;
        }

        // 等待超时或中断
        bool result = cv.wait_for(lock, timeout, [this]() {
            return flag_.load(std::memory_order_acquire);
        });

        {
            std::lock_guard<std::mutex> guard(mutex_);
            waiting_on_ = nullptr;
        }

        if (flag_.load(std::memory_order_acquire)) {
            throw ThreadInterrupted();
        }

        return result; // 返回是否超时 (false = timeout)
    }

    // -------------------------------------------------------------------
    // 设置当前等待的 CV (用于自定义等待)
    // -------------------------------------------------------------------
    void set_condition_variable(std::condition_variable& cv) {
        std::lock_guard<std::mutex> guard(mutex_);
        waiting_on_ = &cv;
    }

    void clear_condition_variable() {
        std::lock_guard<std::mutex> guard(mutex_);
        waiting_on_ = nullptr;
    }
};

// ============================================================================
// 线程局部: 每个线程持有自己的中断标志
// ============================================================================
static thread_local InterruptFlag t_local_interrupt_flag;
static thread_local bool t_interrupt_enabled = true;

// ============================================================================
// InterruptFlag 访问接口
// ============================================================================
namespace this_thread_interruptible {

// 获取当前线程的中断标志
inline InterruptFlag& get_flag() {
    return t_local_interrupt_flag;
}

// 中断点: 在代码中显式检查中断请求
inline void interruption_point() {
    if (t_interrupt_enabled && t_local_interrupt_flag.is_set()) {
        throw ThreadInterrupted();
    }
}

// 禁用中断 (RAII)
class disable_interruption {
private:
    bool old_value_;
public:
    disable_interruption() : old_value_(t_interrupt_enabled) {
        t_interrupt_enabled = false;
    }
    ~disable_interruption() {
        t_interrupt_enabled = old_value_;
    }
    disable_interruption(const disable_interruption&) = delete;
    disable_interruption& operator=(const disable_interruption&) = delete;
};

// 可中断的 sleep
template <typename Rep, typename Period>
void sleep_for(const std::chrono::duration<Rep, Period>& duration) {
    if (!t_interrupt_enabled) {
        std::this_thread::sleep_for(duration);
        return;
    }

    std::mutex mtx;
    std::unique_lock<std::mutex> lock(mtx);
    std::condition_variable cv;

    auto& flag = t_local_interrupt_flag;
    flag.set_condition_variable(cv);

    cv.wait_for(lock, duration, [&flag]() {
        return flag.is_set();
    });

    flag.clear_condition_variable();

    if (flag.is_set()) {
        throw ThreadInterrupted();
    }
}

} // namespace this_thread_interruptible

// ============================================================================
// InterruptibleThread — 可中断的线程包装器
// ============================================================================
class InterruptibleThread {
private:
    std::thread thread_;
    InterruptFlag* flag_; // 指向线程内部的中断标志

public:
    template <typename Func>
    explicit InterruptibleThread(Func&& func)
        : flag_(nullptr)
    {
        // 启动线程, 传入中断标志的指针
        thread_ = std::thread([this, f = std::forward<Func>(func)]() mutable {
            flag_ = &this_thread_interruptible::get_flag();
            try {
                f();
            } catch (const ThreadInterrupted&) {
                // 正常中断, 静默退出
            }
        });
    }

    ~InterruptibleThread() {
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    InterruptibleThread(InterruptibleThread&&) = default;
    InterruptibleThread& operator=(InterruptibleThread&&) = default;

    InterruptibleThread(const InterruptibleThread&) = delete;
    InterruptibleThread& operator=(const InterruptibleThread&) = delete;

    // 中断线程
    void interrupt() {
        if (flag_) {
            flag_->set();
        }
    }

    void join() {
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    bool joinable() const {
        return thread_.joinable();
    }
};

// ============================================================================
// 使用演示
// ============================================================================
void demo_basic_interruption() {
    std::cout << "=== 基本中断演示 ===\n\n";

    InterruptibleThread t([]() {
        try {
            std::cout << "  线程启动, 准备睡眠 5 秒...\n";
            this_thread_interruptible::sleep_for(std::chrono::seconds(5));
            std::cout << "  线程正常完成 (未被中断)\n";
        } catch (const ThreadInterrupted&) {
            std::cout << "  线程被中断!\n";
        }
    });

    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::cout << "  主线程发送中断信号...\n";
    t.interrupt();
    t.join();

    std::cout << "\n";
}

void demo_producer_consumer_interruption() {
    std::cout << "=== 生产者-消费者中断演示 ===\n\n";

    std::queue<int> data_queue;
    std::mutex mtx;
    std::condition_variable cv;
    bool done = false;

    // 消费者线程 (可中断)
    InterruptibleThread consumer([&]() {
        auto& flag = this_thread_interruptible::get_flag();

        try {
            while (true) {
                std::unique_lock<std::mutex> lock(mtx);

                // 可中断的等待
                flag.wait(cv, lock);

                while (!data_queue.empty()) {
                    int val = data_queue.front();
                    data_queue.pop();
                    lock.unlock();
                    std::cout << "  消费: " << val << "\n";
                    lock.lock();
                }

                if (done && data_queue.empty()) break;
            }
        } catch (const ThreadInterrupted&) {
            std::cout << "  消费者被中断!\n";
        }
    });

    // 生产者: 放入一些数据然后中断
    for (int i = 0; i < 5; ++i) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            data_queue.push(i);
            std::cout << "  生产: " << i << "\n";
        }
        cv.notify_one();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    std::cout << "  主线程发送中断...\n";
    consumer.interrupt();
    consumer.join();

    std::cout << "\n";
}

void demo_interruption_points() {
    std::cout << "=== 显式中断检查点演示 ===\n\n";

    InterruptibleThread t([]() {
        for (int i = 0; i < 100; ++i) {
            // 在每个循环迭代中检查中断
            this_thread_interruptible::interruption_point();

            // 模拟工作...
            std::cout << "  迭代 " << i << "\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::cout << "  主线程发送中断...\n";
    t.interrupt();
    t.join();

    std::cout << "\n";
}

void demo_disable_interruption() {
    std::cout << "=== 禁用中断区域演示 ===\n\n";

    InterruptibleThread t([]() {
        try {
            for (int i = 0; i < 5; ++i) {
                std::cout << "  迭代 " << i << " (可中断)\n";
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                this_thread_interruptible::interruption_point();
            }

            // 关键区: 中断被禁用
            {
                this_thread_interruptible::disable_interruption guard;
                std::cout << "  进入关键区 (中断已禁用)\n";
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                std::cout << "  退出关键区 (中断恢复)\n";
            }

            // 再次可中断
            this_thread_interruptible::interruption_point();
            std::cout << "  不应该到达这里\n";

        } catch (const ThreadInterrupted&) {
            std::cout << "  线程被中断!\n";
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    std::cout << "  主线程 (在关键区时) 发送中断...\n";
    t.interrupt();
    t.join();

    std::cout << "\n";
}

// ============================================================================
// main
// ============================================================================
int main() {
    demo_basic_interruption();
    demo_producer_consumer_interruption();
    demo_interruption_points();
    demo_disable_interruption();
    return 0;
}
