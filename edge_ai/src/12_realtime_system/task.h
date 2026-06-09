#pragma once

#include <cstdint>
#include <string>
#include <vector>

// 任务模型：C（WCET）、T（周期）、D（截止时间）、名称
struct Task {
    std::string name;
    int64_t wcet_us;     // 最坏情况执行时间（C）
    int64_t period_us;   // 周期（T）
    int64_t deadline_us; // 相对截止时间（D），通常等于周期
    int id;              // 唯一 ID

    // 运行时状态（每次模拟重置）
    int64_t next_release_us;   // 下一个绝对释放时间
    int64_t absolute_deadline; // 下一个绝对截止时间
    int64_t remaining_time;    // 当前任务剩余执行时间
    int priority;              // 分配的优先级（RMS 中 0 = 最高，EDF 中为动态）

    Task(std::string n, int64_t c, int64_t p, int64_t d, int i) : name(std::move(n)), wcet_us(c), period_us(p), deadline_us(d), id(i) {
    }
};

// 调度事件：记录在给定时间发生了什么
struct ScheduleEvent {
    int64_t time_us;
    std::string task_name;
    std::string event_type; // "RELEASE"、"START"、"COMPLETE"、"DEADLINE_MISS"、"PREEMPT"
    int job_id;
    int64_t remaining; // 剩余执行时间
};

// 模拟过程中收集的统计信息
struct SchedulerStats {
    int total_jobs = 0;
    int missed_deadlines = 0;
    int preemptions = 0;
    int context_switches = 0;
    int idle_ticks = 0;
    int busy_ticks = 0;
    std::vector<int64_t> response_times;        // 任务完成时间减去释放时间
    std::vector<int64_t> deadline_miss_amounts; // 超限时长
};
