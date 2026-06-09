#include "edf_scheduler.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

void EDFScheduler::run() {
    int64_t sim_end = hyperperiod();
    int64_t time = 0;
    int job_counter = 0;

    for (size_t i = 0; i < tasks_.size(); i++) {
        tasks_[i].next_release_us = 0;
    }

    struct Job {
        int task_idx;
        int64_t remaining;
        int64_t absolute_deadline;
        int job_id;
    };

    std::vector<Job> ready_queue;

    auto pick_edf = [&]() -> int {
        int best = -1;
        int64_t earliest_deadline = std::numeric_limits<int64_t>::max();
        for (size_t i = 0; i < ready_queue.size(); i++) {
            if (ready_queue[i].absolute_deadline < earliest_deadline) {
                earliest_deadline = ready_queue[i].absolute_deadline;
                best = static_cast<int>(i);
            }
        }
        return best;
    };

    std::cout << "\n=== EDF（最早截止时间优先）===\n";
    std::cout << "超周期：" << sim_end / 1000.0 << " ms\n";
    std::cout << std::left
              << std::setw(14) << "任务"
              << std::setw(8) << "C(us)"
              << std::setw(8) << "T(us)"
              << std::setw(8) << "D(us)\n";
    std::cout << std::string(38, '-') << "\n";
    for (const auto &t : tasks_) {
        std::cout << std::left
                  << std::setw(14) << t.name
                  << std::setw(8) << t.wcet_us
                  << std::setw(8) << t.period_us
                  << std::setw(8) << t.deadline_us << "\n";
    }

    int active_job = -1;

    while (time <= sim_end) {
        for (size_t i = 0; i < tasks_.size(); i++) {
            if (time >= tasks_[i].next_release_us) {
                Job job;
                job.task_idx = static_cast<int>(i);
                job.remaining = tasks_[i].wcet_us;
                job.absolute_deadline = time + tasks_[i].deadline_us;
                job.job_id = job_counter++;
                ready_queue.push_back(job);
                tasks_[i].next_release_us += tasks_[i].period_us;

                std::stringstream ss;
                ss << tasks_[i].name << "_J" << job.job_id;
                events_.push_back({time, ss.str(), "RELEASE", job.job_id, job.remaining});
                stats_.total_jobs++;
            }
        }

        if (ready_queue.empty()) {
            stats_.idle_ticks++;
            time += tick_us_;
            continue;
        }

        // 截止时间超限检查
        for (size_t i = 0; i < ready_queue.size();) {
            if (time >= ready_queue[i].absolute_deadline && ready_queue[i].remaining > 0) {
                std::stringstream ss;
                ss << tasks_[ready_queue[i].task_idx].name
                   << "_J" << ready_queue[i].job_id;
                events_.push_back({time, ss.str(), "DEADLINE_MISS",
                                   ready_queue[i].job_id, ready_queue[i].remaining});
                stats_.missed_deadlines++;
                stats_.deadline_miss_amounts.push_back(
                    time - ready_queue[i].absolute_deadline);
                ready_queue.erase(ready_queue.begin() + i);
            } else {
                i++;
            }
        }

        if (ready_queue.empty()) {
            stats_.idle_ticks++;
            time += tick_us_;
            continue;
        }

        int picked = pick_edf();
        if (picked == -1) {
            stats_.idle_ticks++;
            time += tick_us_;
            continue;
        }

        if (active_job != picked && active_job != -1) {
            stats_.preemptions++;
            stats_.context_switches++;
            events_.push_back({time, tasks_[ready_queue[picked].task_idx].name,
                               "PREEMPT", -1, 0});
        }

        if (active_job != picked) {
            events_.push_back({time, tasks_[ready_queue[picked].task_idx].name,
                               "START", ready_queue[picked].job_id,
                               ready_queue[picked].remaining});
            if (active_job == -1) stats_.context_switches++;
            active_job = picked;
        }

        ready_queue[picked].remaining -= tick_us_;
        stats_.busy_ticks++;

        if (ready_queue[picked].remaining <= 0) {
            int64_t response = time + tick_us_ - (ready_queue[picked].absolute_deadline - tasks_[ready_queue[picked].task_idx].deadline_us);
            stats_.response_times.push_back(response);

            std::stringstream ss;
            ss << tasks_[ready_queue[picked].task_idx].name
               << "_J" << ready_queue[picked].job_id;
            events_.push_back({time + tick_us_, ss.str(), "COMPLETE",
                               ready_queue[picked].job_id, 0});
            ready_queue.erase(ready_queue.begin() + picked);
            active_job = -1;
            stats_.context_switches++;
        }

        time += tick_us_;
    }

    print_summary();
}

void EDFScheduler::print_summary() const {
    double util = 0;
    for (const auto &t : tasks_) {
        util += (double)t.wcet_us / t.period_us;
    }
    std::cout << "\nEDF 结果：\n";
    std::cout << "  总任务数：         " << stats_.total_jobs << "\n";
    std::cout << "  截止时间超限：     " << stats_.missed_deadlines << "\n";
    std::cout << "  抢占次数：         " << stats_.preemptions << "\n";
    std::cout << "  总利用率：         " << std::fixed
              << std::setprecision(2) << util * 100 << "%\n";
    if (stats_.missed_deadlines == 0) {
        std::cout << "  状态：可调度（EDF 利用率上限为 100%）\n";
    } else {
        std::cout << "  状态：存在截止时间超限（利用率 >100%？）\n";
    }
}
