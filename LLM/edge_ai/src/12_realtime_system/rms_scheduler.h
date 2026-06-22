#pragma once

#include "task.h"

#include <vector>

// 调度算法基类
class Scheduler {
public:
    Scheduler(std::vector<Task> tasks, int64_t tick_us = 1000) : tasks_(std::move(tasks)), tick_us_(tick_us) {
    }

    virtual ~Scheduler() = default;

    virtual void run() = 0;

    const std::vector<ScheduleEvent> &events() const {
        return events_;
    }
    const SchedulerStats &stats() const {
        return stats_;
    }

protected:
    int64_t hyperperiod() const {
        if (tasks_.empty()) return 0;
        int64_t h = tasks_[0].period_us;
        for (size_t i = 1; i < tasks_.size(); i++) {
            h = lcm(h, tasks_[i].period_us);
        }
        return h;
    }

    static int64_t gcd(int64_t a, int64_t b) {
        while (b) {
            int64_t t = b;
            b = a % b;
            a = t;
        }
        return a;
    }

    static int64_t lcm(int64_t a, int64_t b) {
        return a / gcd(a, b) * b;
    }

    std::vector<Task> tasks_;
    int64_t tick_us_;
    std::vector<ScheduleEvent> events_;
    SchedulerStats stats_;
};

// 单调速率调度（RMS）
class RMSScheduler : public Scheduler {
public:
    using Scheduler::Scheduler;

    void run() override;

private:
    void print_summary() const;
};
