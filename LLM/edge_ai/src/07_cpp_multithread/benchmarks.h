#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

// ============================================================================
// 计时器工具
// ============================================================================
class Timer {
public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
    double elapsed_us() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(end - start_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

// ============================================================================
// 辅助函数：打印节标题
// ============================================================================
void print_header(const std::string &title);

// ============================================================================
// 演示 1: 线程池压力测试
// ============================================================================
void demo_thread_pool();

// ============================================================================
// 演示 2: 无锁队列压力测试 (MPMC)
// ============================================================================
void demo_lockfree_queue();

// ============================================================================
// 演示 3: 无锁 vs 基于互斥锁的队列吞吐量对比
// ============================================================================
template <size_t Capacity>
void benchmark_queue_comparison();

// ============================================================================
// 演示 4: 优先级反转模拟
// ============================================================================
void demo_priority_inversion();

// ============================================================================
// 演示 5: 原子内存顺序演示
// ============================================================================
void demo_memory_ordering();

// ============================================================================
// 演示 6: 使用条件变量的生产者-消费者
// ============================================================================
void demo_producer_consumer();
