#pragma once

#include "rms_scheduler.h"

// 最早截止时间优先（EDF）
class EDFScheduler : public Scheduler {
public:
    using Scheduler::Scheduler;

    void run() override;

private:
    void print_summary() const;
};
