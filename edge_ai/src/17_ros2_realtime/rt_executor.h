#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <thread>
#include <vector>

// 实时执行器架构
// 模拟 ros2_control 的线程分离模式：
//   - RTExecutor: 运行在 SCHED_FIFO (实时优先级)，执行控制循环
//   - NonRTThread: 运行在 SCHED_OTHER (普通优先级)，执行感知和日志

// 回调函数类型：接收当前迭代编号，返回执行耗时 (纳秒)
using rt_callback_t = std::function<int64_t(uint64_t iteration)>;

// 单个 RT 回调的注册信息
struct RTCallbackEntry {
    std::string name;
    rt_callback_t callback;
    int64_t period_ns;      // 目标周期 (纳秒)
    uint64_t next_deadline; // 下次截止时间的绝对时间戳
};

// 单次迭代的抖动记录
struct IterationRecord {
    uint64_t iteration;
    int64_t wakeup_error_ns; // 实际唤醒时间与目标的偏差 (正数 = 延迟)
    int64_t execution_ns;    // 回调执行耗时
    bool overrun;            // 是否超时 (执行耗时 > 周期)
};

// 实时执行器：固定频率回调调度器
class RTExecutor {
public:
    RTExecutor();
    ~RTExecutor();

    RTExecutor(const RTExecutor &) = delete;
    RTExecutor &operator=(const RTExecutor &) = delete;

    // 注册一个固定频率的回调
    // period_hz: 目标频率 (Hz)
    void register_callback(const std::string &name, int period_hz,
                           rt_callback_t callback);

    // 启动 RT 循环，运行 duration_seconds 秒
    // 返回抖动记录列表
    std::vector<IterationRecord> run(double duration_seconds);

    // 获取统计摘要
    struct JitterStats {
        std::string name;
        int64_t min_jitter_ns;
        int64_t max_jitter_ns;
        int64_t avg_jitter_ns;
        int64_t p99_jitter_ns;
        uint64_t total_iterations;
        uint64_t overrun_count;
        int target_hz;
    };

    std::vector<JitterStats> get_stats() const;

private:
    // 设置线程为 SCHED_FIFO 实时调度策略
    static bool set_realtime_priority(int priority);

    // 获取单调时钟时间戳 (纳秒)
    static int64_t now_ns();

    std::vector<RTCallbackEntry> callbacks_;
    std::vector<IterationRecord> records_;
    std::atomic<bool> running_{false};

    int64_t min_period_ns_ = 0;
};

// 非实时线程：模拟 ros2_control 中的感知/规划/日志线程
// 运行在 SCHED_OTHER 下，可以被 RT 线程抢占
class NonRTThread {
public:
    NonRTThread(const std::string &name, int64_t work_duration_us,
                double frequency_hz);
    ~NonRTThread();

    NonRTThread(const NonRTThread &) = delete;
    NonRTThread &operator=(const NonRTThread &) = delete;

    void start();
    void stop();
    void join();

    uint64_t iteration_count() const {
        return iterations_.load();
    }
    bool running() const {
        return running_.load();
    }

private:
    void worker_loop();

    std::string name_;
    int64_t work_duration_us_;
    double frequency_hz_;
    std::thread thread_;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> iterations_{0};
};
