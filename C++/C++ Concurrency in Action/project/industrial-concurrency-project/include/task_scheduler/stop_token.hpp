#pragma once
// Ch9.2：中断线程 / 停止令牌（C++20 std::stop_token 模式）
// 简化的手动实现，用于演示机制原理。
// 在生产 C++20 项目中，应优先直接使用 <stop_token> 中的 std::stop_token。
// 中断线程 / 停止令牌机制
// 手动简化实现，演示 stop_source/stop_token 的工作原理
// 生产代码中应使用标准库的 std::stop_token

#include <atomic>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <functional>

namespace task_scheduler {

// 前向声明以解决循环依赖。
class stop_token;

// 在中断点抛出的异常（Ch9.2.8：通过异常实现中断）。
// 必须在 stop_token 之前定义，因为 stop_token 使用它。
// 停止请求异常：在中断点抛出，通知调用者线程被请求停止
class StopRequestedException : public std::exception {
public:
    const char* what() const noexcept override { return "Stop requested"; }
};

// Ch9.2.1：Stop source——停止机制的生产者端。
// 线程安全：可以从任何线程调用以请求停止。
// 停止源：生产者端，用于请求停止
class stop_source {
public:
    stop_source() : state_(std::make_shared<State>()) {}

    // Ch9.2.3：不可拷贝但可移动（类似于 std::stop_source）。
    // 不可拷贝，可移动
    stop_source(const stop_source&) = delete;
    stop_source& operator=(const stop_source&) = delete;
    stop_source(stop_source&&) noexcept = default;
    stop_source& operator=(stop_source&&) noexcept = default;

    // 请求停止。如果已停止或没有关联的 token，返回 false。
    // 使用原子标志，线程安全；唤醒所有等待者
    bool request_stop() noexcept {
        auto prev = state_->stopped.exchange(true, std::memory_order_acq_rel);
        if (!prev) {
            state_->cv.notify_all();
        }
        return !prev;
    }

    // 检查是否已请求停止
    bool stop_requested() const noexcept {
        return state_->stopped.load(std::memory_order_acquire);
    }

    // 从此 source 创建一个 token（Ch9.2.2）。
    // 从停止源创建令牌
    stop_token get_token() const noexcept;

private:
    // 共享状态：多个 token 可以共享同一个状态
    struct State {
        // Ch5.3.3：atomic<bool> 作为标志位，此处不需要 memory_order_seq_cst。
        // 原子布尔标志位：标记是否已请求停止
        std::atomic<bool> stopped{false};
        std::mutex mtx;
        std::condition_variable cv;
    };
    std::shared_ptr<State> state_;

    friend class stop_token;
};

// Ch9.2.2：Stop token——消费者端。轻量级，可拷贝。
// 停止令牌：消费者端，轻量且可拷贝
class stop_token {
public:
    stop_token() noexcept = default;
    explicit stop_token(std::shared_ptr<stop_source::State> state) noexcept
        : state_(std::move(state)) {}

    // Ch9.2.4：检查是否已请求停止。
    // 检查停止是否已被请求
    [[nodiscard]] bool stop_requested() const noexcept {
        return state_ && state_->stopped.load(std::memory_order_acquire);
    }

    // 阻塞直到请求停止（Ch9.2.5：使用 condition_variable 等待）。
    // 阻塞当前线程直到停止被请求
    void wait() const {
        if (state_) {
            std::unique_lock lock(state_->mtx);
            state_->cv.wait(lock, [this] { return state_->stopped.load(std::memory_order_acquire); });
        }
    }

    // Ch9.2.6：带超时的等待。
    // 带超时的阻塞等待
    template <typename Rep, typename Period>
    bool wait_for(const std::chrono::duration<Rep, Period>& timeout) const {
        if (state_) {
            std::unique_lock lock(state_->mtx);
            return state_->cv.wait_for(lock, timeout,
                [this] { return state_->stopped.load(std::memory_order_acquire); });
        }
        return false;
    }

    // 注册停止回调（简化版：仅定期检查标志位）。
    // Ch9.2.7：中断点——在安全的取消点调用此方法。
    // 中断点：在安全取消点调用，如果已请求停止则抛出异常
    void interruption_point() const {
        if (stop_requested()) {
            throw StopRequestedException();
        }
    }

    // 永不停止的公共 token（没有关联的实际停止源）。
    // 永不停止的令牌：没有关联的停止源
    static stop_token never_stopping() noexcept { return stop_token{}; }

private:
    std::shared_ptr<stop_source::State> state_;
};

// stop_source::get_token() 的内联实现
inline stop_token stop_source::get_token() const noexcept {
    return stop_token(state_);
}

} // namespace task_scheduler
