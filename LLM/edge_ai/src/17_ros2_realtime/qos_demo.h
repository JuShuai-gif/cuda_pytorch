#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "spsc_ringbuffer.h"

// ROS2 QoS 策略模拟
// 包含: Reliable (可靠传输), BestEffort (尽力传输), Deadline (截止时间监控)

// ============================================================
// 可靠传输通道: 保证每条消息都送达
// 使用 SPSC 环形缓冲区，满时阻塞等待
// 适用于控制指令 (RELIABLE QoS)
// ============================================================
template <typename T, size_t Depth>
class ReliableChannel {
public:
    ReliableChannel() = default;

    // 发送数据，如果缓冲区满则自旋等待
    void publish(const T &data) {
        while (!buffer_.try_push(data)) {
            // 自旋等待：在实际 ROS2 中，这里会阻塞等待历史消息被消费
            // 对于 RT 控制指令，阻塞是可以接受的
            dropped_count_.fetch_add(0, std::memory_order_relaxed);
        }
        published_count_.fetch_add(1, std::memory_order_relaxed);
    }

    // 接收数据，如果缓冲区空则返回 false
    bool take(T &data) {
        if (buffer_.try_pop(data)) {
            received_count_.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        return false;
    }

    // 一次性取走所有可用数据 (用于 draining)
    size_t take_all(std::vector<T> &out) {
        size_t count = 0;
        T item;
        while (buffer_.try_pop(item)) {
            out.push_back(item);
            ++count;
            received_count_.fetch_add(1, std::memory_order_relaxed);
        }
        return count;
    }

    uint64_t published() const {
        return published_count_.load();
    }
    uint64_t received() const {
        return received_count_.load();
    }
    uint64_t dropped() const {
        return dropped_count_.load();
    }

private:
    SPSCRingBuffer<T, Depth> buffer_;
    std::atomic<uint64_t> published_count_{0};
    std::atomic<uint64_t> received_count_{0};
    std::atomic<uint64_t> dropped_count_{0};
};

// ============================================================
// 尽力传输通道: 不保证送达，缓冲区满时丢弃
// 适用于传感器数据 (BEST_EFFORT QoS)
// ============================================================
template <typename T, size_t Depth>
class BestEffortChannel {
public:
    BestEffortChannel() = default;

    // 发送数据，如果缓冲区满则丢弃最旧的数据
    void publish(const T &data) {
        buffer_.push_overwrite(data);
        published_count_.fetch_add(1, std::memory_order_relaxed);
    }

    // 接收最新数据
    bool take_latest(T &data) {
        if (buffer_.peek_latest(data)) {
            // 取走最新数据并清空缓冲区
            buffer_.drain();
            received_count_.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        return false;
    }

    uint64_t published() const {
        return published_count_.load();
    }
    uint64_t received() const {
        return received_count_.load();
    }

private:
    SPSCRingBuffer<T, Depth> buffer_;
    std::atomic<uint64_t> published_count_{0};
    std::atomic<uint64_t> received_count_{0};
};

// ============================================================
// 截止时间监控器: 检测数据是否在截止时间内到达
// 模拟 ROS2 Deadline QoS
// ============================================================
class DeadlineMonitor {
public:
    // deadline_ns: 截止时间 (纳秒)
    // missed_callback: 超时回调
    DeadlineMonitor(int64_t deadline_ns,
                    std::function<void(int64_t late_by_ns)> missed_callback) : deadline_ns_(deadline_ns), missed_callback_(std::move(missed_callback)) {
    }

    // 记录数据到达时间
    void record_arrival(int64_t timestamp_ns) {
        last_arrival_ns_.store(timestamp_ns, std::memory_order_release);
        arrival_count_.fetch_add(1, std::memory_order_relaxed);
    }

    // 检查是否错过截止时间 (应在控制循环中周期调用)
    void check(int64_t now_ns) {
        int64_t last = last_arrival_ns_.load(std::memory_order_acquire);
        if (last == 0) return; // 尚未收到任何数据

        int64_t elapsed = now_ns - last;
        if (elapsed > deadline_ns_) {
            missed_callback_(elapsed - deadline_ns_);
            miss_count_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    uint64_t miss_count() const {
        return miss_count_.load();
    }
    uint64_t arrival_count() const {
        return arrival_count_.load();
    }

private:
    int64_t deadline_ns_;
    std::function<void(int64_t late_by_ns)> missed_callback_;
    std::atomic<int64_t> last_arrival_ns_{0};
    std::atomic<uint64_t> miss_count_{0};
    std::atomic<uint64_t> arrival_count_{0};
};

// ============================================================
// 速率匹配器: 在不同频率的节点之间匹配数据速率
//
// 200Hz 传感器 -> 30Hz 感知: latest-is-best (取最新帧)
// 30Hz 感知   -> 1kHz 控制: hold-last (保持最近结果)
// ============================================================
class RateMatcher {
public:
    RateMatcher(int source_hz, int target_hz) : source_interval_ns_(1'000'000'000LL / source_hz), target_interval_ns_(1'000'000'000LL / target_hz), ratio_(static_cast<double>(source_hz) / target_hz) {
    }

    // 源端发布数据 (生产者调用)
    void source_publish(int64_t timestamp_ns) {
        latest_source_ts_.store(timestamp_ns, std::memory_order_release);
        source_count_.fetch_add(1, std::memory_order_relaxed);
    }

    // 目标端取数据
    // 模式 selection: true = latest-is-best (取最新, 丢弃旧的)
    //                 false = hold-last (保持最近值, 不清空)
    bool target_consume(int64_t &out_timestamp_ns, bool /*latest_is_best*/) {
        int64_t latest = latest_source_ts_.load(std::memory_order_acquire);

        if (latest == last_consumed_ts_) {
            // 无新数据
            if (last_valid_ts_ != 0) {
                out_timestamp_ns = last_valid_ts_; // hold-last
                return true;
            }
            return false;
        }

        last_consumed_ts_ = latest;
        last_valid_ts_ = latest;
        out_timestamp_ns = latest;
        target_count_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    uint64_t source_count() const {
        return source_count_.load();
    }
    uint64_t target_count() const {
        return target_count_.load();
    }
    double ratio() const {
        return ratio_;
    }

private:
    int64_t source_interval_ns_;
    int64_t target_interval_ns_;
    double ratio_;

    std::atomic<int64_t> latest_source_ts_{0};
    int64_t last_consumed_ts_ = 0;
    int64_t last_valid_ts_ = 0;

    std::atomic<uint64_t> source_count_{0};
    std::atomic<uint64_t> target_count_{0};
};

// ============================================================
// 多速率管线数据结构
// 模拟完整的传感器→感知→控制管线
// ============================================================
struct SensorSample {
    int64_t timestamp_ns;
    int64_t seq;
    double value; // 模拟的传感器读数
};

struct PerceptionResult {
    int64_t timestamp_ns;
    int64_t seq;
    int num_detections;
    double confidence;
};

struct ControlCommand {
    int64_t timestamp_ns;
    int64_t seq;
    double target_position;
    double target_velocity;
};
