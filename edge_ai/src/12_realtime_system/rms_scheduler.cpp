#include "rms_scheduler.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

void RMSScheduler::run() {
    int64_t sim_end = hyperperiod();
    int64_t time = 0;
    int job_counter = 0;

    // 按周期排序（周期越短 = 优先级越高）
    std::sort(tasks_.begin(), tasks_.end(), [](const Task &a, const Task &b) {
        return a.period_us < b.period_us;
    });
    for (size_t i = 0; i < tasks_.size(); i++) {
        tasks_[i].priority = static_cast<int>(i); // 0 = 最高
        tasks_[i].next_release_us = 0;
    }

    struct Job {
        int task_idx;
        int64_t remaining;
        int64_t absolute_deadline;
        int job_id;
    };

    std::vector<Job> ready_queue;

    // Lambda：在就绪队列中查找最高优先级的任务
    auto pick_highest = [&]() -> int {
        int best = -1;
        int best_prio = std::numeric_limits<int>::max();
        for (size_t i = 0; i < ready_queue.size(); i++) {
            int prio = tasks_[ready_queue[i].task_idx].priority;
            if (prio < best_prio) {
                best_prio = prio;
                best = static_cast<int>(i);
            }
        }
        return best;
    };

    std::cout << "\n=== RMS（单调速率调度）===\n";
    std::cout << "超周期：" << sim_end / 1000.0 << " ms\n";
    std::cout << std::left
              << std::setw(14) << "任务"
              << std::setw(8) << "C(us)"
              << std::setw(8) << "T(us)"
              << std::setw(8) << "D(us)"
              << std::setw(8) << "优先级\n";
    std::cout << std::string(46, '-') << "\n";
    for (const auto &t : tasks_) {
        std::cout << std::left
                  << std::setw(14) << t.name
                  << std::setw(8) << t.wcet_us
                  << std::setw(8) << t.period_us
                  << std::setw(8) << t.deadline_us
                  << std::setw(8) << t.priority << "\n";
    }

    int active_job = -1;

    while (time <= sim_end) {
        // 在此时刻释放新任务
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

        // 检查截止时间超限
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
                // 不递增 i
            } else {
                i++;
            }
        }

        if (ready_queue.empty()) {
            stats_.idle_ticks++;
            time += tick_us_;
            continue;
        }

        int picked = pick_highest();
        if (picked == -1) {
            stats_.idle_ticks++;
            time += tick_us_;
            continue;
        }

        // 抢占检测
        if (active_job != picked && active_job != -1) {
            stats_.preemptions++;
            stats_.context_switches++;
            events_.push_back({time,
                               tasks_[ready_queue[picked].task_idx].name,
                               "PREEMPT", -1, 0});
        }

        if (active_job != picked) {
            events_.push_back({time,
                               tasks_[ready_queue[picked].task_idx].name,
                               "START", ready_queue[picked].job_id,
                               ready_queue[picked].remaining});
            if (active_job == -1) {
                stats_.context_switches++;
            }
            active_job = picked;
        }

        // 执行一个时钟周期
        ready_queue[picked].remaining -= tick_us_;
        stats_.busy_ticks++;

        // 任务是否完成？
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

void RMSScheduler::print_summary() const {
    std::cout << "\nRMS 结果：\n";
    std::cout << "  总任务数：         " << stats_.total_jobs << "\n";
    std::cout << "  截止时间超限：     " << stats_.missed_deadlines << "\n";
    std::cout << "  抢占次数：         " << stats_.preemptions << "\n";
    std::cout << "  上下文切换：       " << stats_.context_switches << "\n";

    if (!stats_.response_times.empty()) {
        auto sorted = stats_.response_times;
        std::sort(sorted.begin(), sorted.end());
        double sum = 0;
        for (auto v : sorted) sum += v;
        double avg = sum / sorted.size();

        std::cout << "  平均响应时间：     " << std::fixed
                  << std::setprecision(1) << avg / 1000.0 << " ms\n";
        std::cout << "  最大响应时间：     " << sorted.back() / 1000.0 << " ms\n";

        double util = 0;
        for (const auto &t : tasks_) {
            util += (double)t.wcet_us / t.period_us;
        }
        std::cout << "  总利用率：         " << std::fixed
                  << std::setprecision(2) << util * 100 << "%\n";
        double bound = tasks_.size() * (std::pow(2.0, 1.0 / tasks_.size()) - 1.0);
        std::cout << "  RMS 上限（n=" << tasks_.size() << "）："
                  << std::fixed << std::setprecision(2) << bound * 100 << "%\n";
        if (stats_.missed_deadlines == 0) {
            std::cout << "  状态：可调度\n";
        } else {
            std::cout << "  状态：检测到截止时间超限\n";
        }
    }
}
