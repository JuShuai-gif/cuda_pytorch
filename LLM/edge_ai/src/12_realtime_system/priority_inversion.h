#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>

// 用于优先级反转演示的简单共享资源（带互斥锁）
struct SharedResource {
    std::mutex mtx;
    int64_t locked_by_prio = -1; // 持有锁的任务的优先级（用于日志记录）
    int data = 0;
};

// 具有模拟优先级继承语义的互斥锁
class PriorityInheritanceMutex {
public:
    explicit PriorityInheritanceMutex(int ceiling_prio = 0) : owner_prio_(-1), ceiling_prio_(ceiling_prio), locked_(false) {
    }

    void lock(int current_prio) {
        std::unique_lock<std::mutex> lk(mtx_);
        while (locked_) {
            // 等待直到解锁
            cv_.wait(lk);
        }
        locked_ = true;
        owner_prio_ = current_prio;
        // 在真实 RTOS 中：如果 current_prio > inherited_owner_prio，
        // 则将锁持有者的优先级提升到 current_prio（优先级继承）
    }

    void unlock() {
        std::lock_guard<std::mutex> lk(mtx_);
        locked_ = false;
        owner_prio_ = -1;
        cv_.notify_one();
    }

    // 模拟优先级继承：
    // 当任务 A 在此锁上阻塞（该锁由任务 B 持有），
    // 任务 B 的有效优先级提升到 max(A.prio, ceiling_prio_)
    int boost_owner_priority(int blocker_prio, int owner_orig_prio) {
        return std::min(blocker_prio, owner_orig_prio);
    }

private:
    std::mutex mtx_;
    std::condition_variable cv_;
    int owner_prio_;
    int ceiling_prio_; // 优先级上限
    bool locked_;
};

// 优先级反转演示
void demo_priority_inversion();

// 优先级继承解决方案演示
void demo_priority_inheritance();
