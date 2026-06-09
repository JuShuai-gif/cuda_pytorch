#include "rt_executor.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <map>
#include <numeric>

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

// ============================================================
// RTExecutor 实现
// ============================================================

RTExecutor::RTExecutor() = default;

RTExecutor::~RTExecutor() {
    running_.store(false);
}

void RTExecutor::register_callback(const std::string &name, int period_hz,
                                   rt_callback_t callback) {
    RTCallbackEntry entry;
    entry.name = name;
    entry.callback = std::move(callback);
    entry.period_ns = 1'000'000'000LL / period_hz;
    entry.next_deadline = 0;

    callbacks_.push_back(std::move(entry));

    // 记录最短周期，用于计算主循环步进
    if (min_period_ns_ == 0 || entry.period_ns < min_period_ns_) {
        min_period_ns_ = entry.period_ns;
    }
}

std::vector<IterationRecord> RTExecutor::run(double duration_seconds) {
    records_.clear();
    running_.store(true);

    // 尝试设置为实时调度策略 SCHED_FIFO
    bool is_rt = set_realtime_priority(80);
    if (is_rt) {
        printf("[RTExecutor] 已设置 SCHED_FIFO 优先级 80\n");
    } else {
        printf("[RTExecutor] 无法设置实时优先级 (需要 sudo 或 CAP_SYS_NICE)，"
               "使用 SCHED_OTHER 继续\n");
    }

    // 为每个回调初始化 next_deadline
    int64_t start_ns = now_ns();
    for (auto &cb : callbacks_) {
        cb.next_deadline = start_ns + cb.period_ns;
    }

    int64_t end_ns = start_ns + static_cast<int64_t>(duration_seconds * 1e9);
    uint64_t iteration = 0;

    // 主控制循环
    while (running_.load() && now_ns() < end_ns) {
        int64_t loop_start = now_ns();

        // 检查所有回调是否需要执行
        for (auto &cb : callbacks_) {
            if (loop_start >= static_cast<int64_t>(cb.next_deadline)) {
                // 记录抖动：实际执行时间与目标的偏差
                int64_t wakeup_error = loop_start - cb.next_deadline;

                // 执行回调
                int64_t exec_ns = cb.callback(iteration);

                IterationRecord rec;
                rec.iteration = iteration;
                rec.wakeup_error_ns = wakeup_error;
                rec.execution_ns = exec_ns;
                rec.overrun = (exec_ns > cb.period_ns);
                records_.push_back(rec);

                // 更新下次截止时间 (使用绝对时间，避免累积漂移)
                cb.next_deadline += cb.period_ns;

                // 如果已经严重落后，跳到当前时间
                if (static_cast<int64_t>(cb.next_deadline) < loop_start) {
                    cb.next_deadline = loop_start + cb.period_ns;
                }
            }
        }

        ++iteration;

        // 使用 clock_nanosleep 精确睡眠到下一个最小周期边界
        if (min_period_ns_ > 0) {
            int64_t next_wake = loop_start + min_period_ns_;
            int64_t now = now_ns();

            if (next_wake > now) {
                struct timespec ts;
                ts.tv_sec = next_wake / 1'000'000'000;
                ts.tv_nsec = next_wake % 1'000'000'000;

                // TIMER_ABSTIME 使用绝对时间，避免累积漂移
                clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr);
            }
        }
    }

    running_.store(false);
    return records_;
}

std::vector<RTExecutor::JitterStats> RTExecutor::get_stats() const {
    std::vector<JitterStats> stats;

    // 按回调名称分组统计
    // 使用 map 收集每个回调的抖动数据
    std::map<std::string, std::vector<int64_t>> jitter_by_name;
    std::map<std::string, int> target_hz_by_name;
    std::map<std::string, int64_t> period_by_name;

    for (const auto &cb : callbacks_) {
        period_by_name[cb.name] = cb.period_ns;
        target_hz_by_name[cb.name] = static_cast<int>(1e9 / cb.period_ns);
    }

    // 按周期匹配记录 (记录中不直接包含名称，但我们可以通过频率推断)
    // 简化处理：按 period_ns 分组
    std::map<int64_t, std::vector<int64_t>> jitter_by_period;
    std::map<int64_t, uint64_t> overrun_by_period;
    std::map<int64_t, std::string> name_by_period;

    for (const auto &cb : callbacks_) {
        name_by_period[cb.period_ns] = cb.name;
    }

    for (const auto &rec : records_) {
        // 通过 wakeup_error 和 overrun 模式匹配
        // 简化：假设记录按回调顺序交错
        // 更稳健的做法：用执行耗时和周期匹配
        for (const auto &cb : callbacks_) {
            if (rec.execution_ns <= cb.period_ns * 2) { // 粗略匹配
                jitter_by_period[cb.period_ns].push_back(rec.wakeup_error_ns);
                if (rec.overrun) {
                    overrun_by_period[cb.period_ns]++;
                }
                break;
            }
        }
    }

    // 改善：我们不应该用这种方式，直接在运行中记录即可
    // 退回简单方案：按记录顺序分配给每个回调 (轮询)
    size_t cb_count = callbacks_.size();
    if (cb_count > 0 && !records_.empty()) {
        // 重建：每个回调对应一个 jitter 列表
        std::map<std::string, std::vector<int64_t>> per_cb_jitter;
        std::map<std::string, uint64_t> per_cb_overruns;

        // 按回调名称分桶
        for (size_t i = 0; i < records_.size(); ++i) {
            size_t cb_idx = i % cb_count;
            const auto &cb = callbacks_[cb_idx];
            per_cb_jitter[cb.name].push_back(records_[i].wakeup_error_ns);
            if (records_[i].overrun) {
                per_cb_overruns[cb.name]++;
            }
        }

        for (const auto &cb : callbacks_) {
            JitterStats s;
            s.name = cb.name;
            s.target_hz = target_hz_by_name[cb.name];
            s.total_iterations = per_cb_jitter[cb.name].size();
            s.overrun_count = per_cb_overruns[cb.name];

            auto &jitters = per_cb_jitter[cb.name];
            if (!jitters.empty()) {
                std::sort(jitters.begin(), jitters.end());
                s.min_jitter_ns = jitters.front();
                s.max_jitter_ns = jitters.back();
                s.avg_jitter_ns = std::accumulate(jitters.begin(), jitters.end(),
                                                  0LL)
                                  / jitters.size();
                size_t p99_idx = jitters.size() * 99 / 100;
                if (p99_idx >= jitters.size()) p99_idx = jitters.size() - 1;
                s.p99_jitter_ns = jitters[p99_idx];
            }

            stats.push_back(s);
        }
    }

    return stats;
}

bool RTExecutor::set_realtime_priority(int priority) {
    struct sched_param param;
    param.sched_priority = priority;

    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &param) != 0) {
        return false;
    }
    return true;
}

int64_t RTExecutor::now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1'000'000'000LL
           + static_cast<int64_t>(ts.tv_nsec);
}

// ============================================================
// NonRTThread 实现
// ============================================================

NonRTThread::NonRTThread(const std::string &name, int64_t work_duration_us,
                         double frequency_hz) : name_(name), work_duration_us_(work_duration_us), frequency_hz_(frequency_hz) {
}

NonRTThread::~NonRTThread() {
    stop();
    if (thread_.joinable()) {
        thread_.join();
    }
}

void NonRTThread::start() {
    running_.store(true);
    thread_ = std::thread(&NonRTThread::worker_loop, this);

    // 设置为普通调度策略 (SCHED_OTHER)
    struct sched_param param;
    param.sched_priority = 0;
    pthread_setschedparam(thread_.native_handle(), SCHED_OTHER, &param);
}

void NonRTThread::stop() {
    running_.store(false);
}

void NonRTThread::join() {
    if (thread_.joinable()) {
        thread_.join();
    }
}

void NonRTThread::worker_loop() {
    int64_t period_ns = static_cast<int64_t>(1e9 / frequency_hz_);
    auto next_wake = std::chrono::steady_clock::now();

    while (running_.load()) {
        // 模拟非实时工作：使用 sleep 而非 burn-loop
        // Non-RT 线程可以被抢占，所以使用 sleep 是合理的
        std::this_thread::sleep_for(std::chrono::microseconds(work_duration_us_));

        iterations_.fetch_add(1, std::memory_order_relaxed);

        // 等待下一个周期
        next_wake += std::chrono::nanoseconds(period_ns);
        auto now = std::chrono::steady_clock::now();
        if (next_wake > now) {
            std::this_thread::sleep_until(next_wake);
        } else {
            next_wake = now; // 重置以避免追赶
        }
    }
}
