#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace edge_vla {
using Clock = std::chrono::steady_clock;

struct Frame { std::uint64_t sequence{}; Clock::time_point captured_at{}; };
struct Action { std::uint64_t sequence{}; Clock::time_point captured_at{}; };

// 容量为一的邮箱：控制系统宁可丢旧帧，也不允许 FIFO 累积延迟。
template <typename T> class LatestValue {
public:
    bool publish(T value) {
        {
            std::lock_guard lock(mutex_);
            if (closed_) return false;
            if (value_) ++overwritten_;
            value_ = std::move(value);
        }
        ready_.notify_one();
        return true;
    }

    std::optional<T> wait_take(std::stop_token stop) {
        std::unique_lock lock(mutex_);
        const bool ready = ready_.wait(lock, stop, [this] { return value_ || closed_; });
        if (!ready || !value_) return std::nullopt;
        auto result = std::move(value_);
        value_.reset();
        return result;
    }

    void close() {
        { std::lock_guard lock(mutex_); closed_ = true; }
        ready_.notify_all();
    }

    [[nodiscard]] std::uint64_t overwritten() const {
        std::lock_guard lock(mutex_);
        return overwritten_;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable_any ready_;
    std::optional<T> value_;
    bool closed_{false};
    std::uint64_t overwritten_{0};
};

struct Report {
    std::uint64_t captured{}, completed{}, dropped{}, deadline_misses{};
    double p50_ms{}, p95_ms{}, p99_ms{};
};

class Pipeline {
public:
    Pipeline(std::chrono::milliseconds frame_period,
             std::chrono::milliseconds inference_time,
             std::chrono::milliseconds deadline)
        : frame_period_(frame_period), inference_time_(inference_time), deadline_(deadline) {}

    Report run_for(std::chrono::milliseconds duration) {
        std::jthread camera([this](std::stop_token s) { capture_loop(s); });
        std::jthread inference([this](std::stop_token s) { inference_loop(s); });
        std::jthread control([this](std::stop_token s) { control_loop(s); });
        std::this_thread::sleep_for(duration);
        camera.request_stop();
        camera.join();
        frames_.close();
        inference.join();                 // drain 最后一帧
        actions_.close();
        control.join();
        return report();
    }

private:
    void capture_loop(std::stop_token stop) {
        auto next = Clock::now();
        while (!stop.stop_requested()) {
            const auto seq = captured_.fetch_add(1, std::memory_order_relaxed);
            if (!frames_.publish(Frame{seq, Clock::now()})) break;
            next += frame_period_;
            std::this_thread::sleep_until(next);
        }
    }

    void inference_loop(std::stop_token stop) {
        while (auto frame = frames_.wait_take(stop)) {
            // 替换为 TensorRT enqueueV3 / LibTorch / ORT。真实 CUDA 后端应使用
            // 非默认 stream，并用 CUDA event 计设备时间。
            std::this_thread::sleep_for(inference_time_);
            if (!actions_.publish(Action{frame->sequence, frame->captured_at})) break;
        }
    }

    void control_loop(std::stop_token stop) {
        while (auto action = actions_.wait_take(stop)) {
            const auto latency = Clock::now() - action->captured_at;
            const auto us = std::chrono::duration_cast<std::chrono::microseconds>(latency).count();
            { std::lock_guard lock(latencies_mutex_); latencies_us_.push_back(us); }
            completed_.fetch_add(1, std::memory_order_relaxed);
            if (latency > deadline_) deadline_misses_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    static double percentile(const std::vector<std::int64_t>& sorted, double p) {
        if (sorted.empty()) return 0.0;
        const auto index = static_cast<std::size_t>(p * static_cast<double>(sorted.size() - 1));
        return static_cast<double>(sorted[index]) / 1000.0;
    }

    Report report() const {
        std::vector<std::int64_t> sorted;
        { std::lock_guard lock(latencies_mutex_); sorted = latencies_us_; }
        std::sort(sorted.begin(), sorted.end());
        return {captured_.load(), completed_.load(),
                frames_.overwritten() + actions_.overwritten(), deadline_misses_.load(),
                percentile(sorted, .50), percentile(sorted, .95), percentile(sorted, .99)};
    }

    std::chrono::milliseconds frame_period_, inference_time_, deadline_;
    LatestValue<Frame> frames_;
    LatestValue<Action> actions_;
    std::atomic<std::uint64_t> captured_{0}, completed_{0}, deadline_misses_{0};
    mutable std::mutex latencies_mutex_;
    std::vector<std::int64_t> latencies_us_;
};
}  // namespace edge_vla

int main(int argc, char** argv) {
    using namespace std::chrono_literals;
    const bool self_test = argc == 2 && std::string_view(argv[1]) == "--self-test";
    edge_vla::Pipeline pipeline{5ms, 12ms, 25ms};
    const auto r = pipeline.run_for(self_test ? 120ms : 1s);
    std::cout << "captured=" << r.captured << " completed=" << r.completed
              << " dropped=" << r.dropped << " deadline_misses=" << r.deadline_misses
              << '\n' << "latency_ms p50=" << r.p50_ms << " p95=" << r.p95_ms
              << " p99=" << r.p99_ms << '\n';
    if (self_test && (r.captured == 0 || r.completed == 0 ||
                      r.completed > r.captured || r.dropped == 0)) {
        std::cerr << "self-test invariant failed\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
