#pragma once
// Chapter 9.2: Interrupting Threads / Stop Tokens (C++20 std::stop_token pattern)
// Simplified manual implementation to demonstrate the mechanism.
// In production C++20, prefer std::stop_token from <stop_token> directly.

#include <atomic>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <functional>

namespace task_scheduler {

// Forward declarations to resolve circular dependencies.
class stop_token;

// Exception thrown at interruption points (Ch9.2.8: interruption via exceptions).
// Must be defined before stop_token which uses it.
class StopRequestedException : public std::exception {
public:
    const char* what() const noexcept override { return "Stop requested"; }
};

// Ch9.2.1: Stop source - the producer side of the stop mechanism.
// Thread-safe: can be called from any thread to request stop.
class stop_source {
public:
    stop_source() : state_(std::make_shared<State>()) {}

    // Ch9.2.3: Non-copyable but movable (similar to std::stop_source).
    stop_source(const stop_source&) = delete;
    stop_source& operator=(const stop_source&) = delete;
    stop_source(stop_source&&) noexcept = default;
    stop_source& operator=(stop_source&&) noexcept = default;

    // Request stop. Returns false if already stopped or no associated token.
    bool request_stop() noexcept {
        auto prev = state_->stopped.exchange(true, std::memory_order_acq_rel);
        if (!prev) {
            state_->cv.notify_all();
        }
        return !prev;
    }

    bool stop_requested() const noexcept {
        return state_->stopped.load(std::memory_order_acquire);
    }

    // Create a token from this source (Ch9.2.2).
    stop_token get_token() const noexcept;

private:
    struct State {
        // Ch5.3.3: atomic<bool> for flag, memory_order_seq_cst not needed here.
        std::atomic<bool> stopped{false};
        std::mutex mtx;
        std::condition_variable cv;
    };
    std::shared_ptr<State> state_;

    friend class stop_token;
};

// Ch9.2.2: Stop token - the consumer side. Lightweight, copyable.
class stop_token {
public:
    stop_token() noexcept = default;
    explicit stop_token(std::shared_ptr<stop_source::State> state) noexcept
        : state_(std::move(state)) {}

    // Ch9.2.4: Check if stop has been requested.
    [[nodiscard]] bool stop_requested() const noexcept {
        return state_ && state_->stopped.load(std::memory_order_acquire);
    }

    // Block until stop is requested (Ch9.2.5: waiting with condition_variable).
    void wait() const {
        if (state_) {
            std::unique_lock lock(state_->mtx);
            state_->cv.wait(lock, [this] { return state_->stopped.load(std::memory_order_acquire); });
        }
    }

    // Ch9.2.6: Wait with timeout.
    template <typename Rep, typename Period>
    bool wait_for(const std::chrono::duration<Rep, Period>& timeout) const {
        if (state_) {
            std::unique_lock lock(state_->mtx);
            return state_->cv.wait_for(lock, timeout,
                [this] { return state_->stopped.load(std::memory_order_acquire); });
        }
        return false;
    }

    // Registration for stop callback (simplified: just check flag periodically).
    // Ch9.2.7: Interruption points - call this at safe cancellation points.
    void interruption_point() const {
        if (stop_requested()) {
            throw StopRequestedException();
        }
    }

    // Public token without actual stop source (never stops).
    static stop_token never_stopping() noexcept { return stop_token{}; }

private:
    std::shared_ptr<stop_source::State> state_;
};

inline stop_token stop_source::get_token() const noexcept {
    return stop_token(state_);
}

} // namespace task_scheduler
