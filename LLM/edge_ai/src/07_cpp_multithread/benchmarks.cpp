#include "benchmarks.h"
#include "thread_pool.h"
#include "lockfree_queue.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <queue>
#include <random>
#include <sstream>
#include <vector>

// ============================================================================
// 辅助函数：打印节标题
// ============================================================================
void print_header(const std::string &title) {
    std::cout << "\n"
              << std::string(70, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(70, '=') << "\n";
}

// ============================================================================
// 演示 1: 线程池压力测试
// ============================================================================
void demo_thread_pool() {
    print_header("演示 1: 线程池压力测试");

    const int NUM_TASKS = 10'000;
    const int NUM_THREADS = static_cast<int>(std::thread::hardware_concurrency());
    ThreadPool pool(NUM_THREADS);

    std::cout << "  工作线程数: " << pool.worker_count() << "\n";
    std::cout << "  待提交任务数: " << NUM_TASKS << "\n\n";

    // 提交计算其索引哈希值的任务
    std::vector<std::future<uint64_t>> futures;
    futures.reserve(NUM_TASKS);

    Timer timer;
    timer.start();
    for (int i = 0; i < NUM_TASKS; ++i) {
        futures.push_back(pool.submit([i]() -> uint64_t {
            // 模拟有意义的工作：计算一个简单的哈希
            uint64_t h = static_cast<uint64_t>(i);
            for (int k = 0; k < 1000; ++k) {
                h = h * 1103515245 + 12345;
            }
            return h;
        }));
    }

    // 验证所有结果
    uint64_t checksum = 0;
    for (int i = 0; i < NUM_TASKS; ++i) {
        uint64_t result = futures[i].get();
        checksum ^= result;
    }
    double elapsed = timer.elapsed_ms();

    std::cout << "  全部 " << NUM_TASKS << " 个任务已完成\n";
    std::cout << "  校验和: 0x" << std::hex << checksum << std::dec << "\n";
    std::cout << "  总耗时: " << std::fixed << std::setprecision(2)
              << elapsed << " ms\n";
    std::cout << "  吞吐量: " << std::fixed << std::setprecision(0)
              << (NUM_TASKS / elapsed * 1000.0) << " 任务/秒\n";
}

// ============================================================================
// 演示 2: 无锁队列压力测试 (MPMC)
// ============================================================================
void demo_lockfree_queue() {
    print_header("演示 2: 无锁队列 MPMC 压力测试");

    constexpr size_t QUEUE_CAPACITY = 1024;
    constexpr int NUM_PRODUCERS = 4;
    constexpr int NUM_CONSUMERS = 4;
    constexpr int ITEMS_PER_PRODUCER = 250'000;
    constexpr int TOTAL_ITEMS = NUM_PRODUCERS * ITEMS_PER_PRODUCER;

    LockFreeQueue<int, QUEUE_CAPACITY> queue;

    std::atomic<int64_t> produced_total{0};
    std::atomic<int64_t> consumed_total{0};
    std::atomic<int64_t> pop_failures{0};
    std::atomic<int64_t> push_failures{0};
    std::atomic<bool> producers_done{false};

    std::cout << "  队列容量: " << QUEUE_CAPACITY << "\n";
    std::cout << "  生产者: " << NUM_PRODUCERS
              << " x " << ITEMS_PER_PRODUCER << " 项\n";
    std::cout << "  消费者: " << NUM_CONSUMERS << "\n\n";

    // 启动消费者
    std::vector<std::thread> consumers;
    consumers.reserve(NUM_CONSUMERS);
    for (int c = 0; c < NUM_CONSUMERS; ++c) {
        consumers.emplace_back([&, c]() {
            int64_t local_count = 0;
            while (!producers_done.load(std::memory_order_acquire)
                   || !queue.empty()) {
                int val;
                if (queue.try_pop(val)) {
                    ++local_count;
                    // 模拟处理
                    volatile int sink = val * 2;
                    (void)sink;
                } else if (!producers_done.load(std::memory_order_acquire)) {
                    ++pop_failures;
                    std::this_thread::yield();
                }
            }
            consumed_total.fetch_add(local_count, std::memory_order_relaxed);
        });
    }

    Timer timer;
    timer.start();

    // 启动生产者
    std::vector<std::thread> producers;
    producers.reserve(NUM_PRODUCERS);
    for (int p = 0; p < NUM_PRODUCERS; ++p) {
        producers.emplace_back([&, p]() {
            int64_t local_count = 0;
            for (int i = 0; i < ITEMS_PER_PRODUCER; ++i) {
                int item = p * ITEMS_PER_PRODUCER + i;
                while (!queue.try_push(item)) {
                    ++push_failures;
                    std::this_thread::yield();
                }
                ++local_count;
            }
            produced_total.fetch_add(local_count, std::memory_order_relaxed);
        });
    }

    // 等待生产者完成
    for (auto &t : producers) t.join();
    producers_done.store(true, std::memory_order_release);

    // 等待消费者完成
    for (auto &t : consumers) t.join();

    double elapsed = timer.elapsed_ms();

    std::cout << "  已生产: " << produced_total.load() << "\n";
    std::cout << "  已消费: " << consumed_total.load() << "\n";
    std::cout << "  入队失败次数（队列已满）: " << push_failures.load() << "\n";
    std::cout << "  出队失败次数（队列为空）: " << pop_failures.load() << "\n";
    std::cout << "  总耗时: " << std::fixed << std::setprecision(2)
              << elapsed << " ms\n";
    std::cout << "  吞吐量: " << std::fixed << std::setprecision(0)
              << (TOTAL_ITEMS / elapsed * 1000.0) << " 操作/秒\n";
}

// ============================================================================
// 演示 3: 无锁 vs 基于互斥锁的队列吞吐量对比
// ============================================================================
template <size_t Capacity>
void benchmark_queue_comparison() {
    print_header("演示 3: 无锁 vs 基于互斥锁的有界队列吞吐量");

    constexpr int ITEMS = 500'000;
    constexpr int PRODUCERS = 2;
    constexpr int CONSUMERS = 2;

    // 无锁基准测试
    {
        LockFreeQueue<int, Capacity> lf_queue;
        std::atomic<bool> done{false};
        std::atomic<int64_t> total{0};

        auto producer_fn = [&](int id, int count) {
            for (int i = 0; i < count; ++i) {
                int item = id * count + i;
                while (!lf_queue.try_push(item)) {
                    std::this_thread::yield();
                }
            }
        };
        auto consumer_fn = [&](int64_t &local) {
            while (!done.load(std::memory_order_acquire) || !lf_queue.empty()) {
                int val;
                if (lf_queue.try_pop(val)) {
                    ++local;
                } else if (!done.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }
            }
        };

        std::vector<std::thread> threads;
        int per_prod = ITEMS / PRODUCERS;

        Timer t;
        t.start();
        for (int i = 0; i < PRODUCERS; ++i) {
            threads.emplace_back(producer_fn, i, per_prod);
        }
        std::vector<int64_t> consumer_counts(CONSUMERS, 0);
        for (int c = 0; c < CONSUMERS; ++c) {
            threads.emplace_back(consumer_fn, std::ref(consumer_counts[c]));
        }
        for (int i = 0; i < PRODUCERS; ++i) threads[i].join();
        done.store(true, std::memory_order_release);
        for (int c = 0; c < CONSUMERS; ++c) threads[PRODUCERS + c].join();
        double lf_time = t.elapsed_ms();

        int64_t lf_total = 0;
        for (auto c : consumer_counts) lf_total += c;

        std::cout << "  无锁队列:\n";
        std::cout << "    传输项数: " << lf_total << "\n";
        std::cout << "    耗时: " << std::fixed << std::setprecision(2)
                  << lf_time << " ms\n";
        std::cout << "    速率: " << std::fixed << std::setprecision(0)
                  << (ITEMS / lf_time * 1000.0) << " 操作/秒\n\n";
    }

    // 基于互斥锁的有界队列基准测试
    {
        std::queue<int> mq;
        std::mutex mtx;
        std::condition_variable not_full;
        std::condition_variable not_empty;
        size_t max_cap = Capacity - 1;
        bool producers_done = false;
        std::atomic<int64_t> total{0};

        auto producer_fn = [&](int id, int count) {
            for (int i = 0; i < count; ++i) {
                int item = id * count + i;
                std::unique_lock<std::mutex> lock(mtx);
                not_full.wait(lock, [&] { return mq.size() < max_cap; });
                mq.push(item);
                lock.unlock();
                not_empty.notify_one();
            }
        };
        auto consumer_fn = [&](int64_t &local) {
            while (true) {
                std::unique_lock<std::mutex> lock(mtx);
                not_empty.wait_for(lock, std::chrono::milliseconds(1),
                                   [&] { return !mq.empty() || producers_done; });
                if (!mq.empty()) {
                    int val = mq.front();
                    mq.pop();
                    ++local;
                    lock.unlock();
                    not_full.notify_one();
                } else if (producers_done && mq.empty()) {
                    break;
                }
            }
        };

        std::vector<std::thread> threads;
        int per_prod = ITEMS / PRODUCERS;

        Timer t;
        t.start();
        for (int i = 0; i < PRODUCERS; ++i) {
            threads.emplace_back(producer_fn, i, per_prod);
        }
        std::vector<int64_t> consumer_counts(CONSUMERS, 0);
        for (int c = 0; c < CONSUMERS; ++c) {
            threads.emplace_back(consumer_fn, std::ref(consumer_counts[c]));
        }
        for (int i = 0; i < PRODUCERS; ++i) threads[i].join();
        {
            std::lock_guard<std::mutex> lock(mtx);
            producers_done = true;
        }
        not_empty.notify_all();
        for (int c = 0; c < CONSUMERS; ++c) threads[PRODUCERS + c].join();
        double mutex_time = t.elapsed_ms();

        int64_t mutex_total = 0;
        for (auto c : consumer_counts) mutex_total += c;

        std::cout << "  互斥锁有界队列:\n";
        std::cout << "    传输项数: " << mutex_total << "\n";
        std::cout << "    耗时: " << std::fixed << std::setprecision(2)
                  << mutex_time << " ms\n";
        std::cout << "    速率: " << std::fixed << std::setprecision(0)
                  << (ITEMS / mutex_time * 1000.0) << " 操作/秒\n";
    }
}

// 显式模板实例化，Capacity=1024
template void benchmark_queue_comparison<1024>();

// ============================================================================
// 演示 4: 优先级反转模拟
// ============================================================================
void demo_priority_inversion() {
    print_header("演示 4: 优先级反转模拟");

    std::mutex resource;
    std::atomic<bool> start_flag{false};
    std::atomic<int> phase{0};
    std::atomic<int64_t> low_progress{0};
    std::atomic<int64_t> med_progress{0};
    std::atomic<int64_t> high_progress{0};
    std::atomic<int64_t> high_wait_us{0};

    // 低优先级线程：持有共享资源
    std::thread low_thread([&]() {
        while (!start_flag.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        auto t_start = std::chrono::high_resolution_clock::now();
        resource.lock();
        // 模拟长临界区
        for (volatile int i = 0; i < 50'000'000; ++i) {
            low_progress++;
        }
        resource.unlock();
        auto t_end = std::chrono::high_resolution_clock::now();
        (void)t_start;
        (void)t_end;
    });

    // 中优先级线程：抢占低优先级线程（CPU 密集型）
    std::thread med_thread([&]() {
        while (!start_flag.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        auto t_start = std::chrono::high_resolution_clock::now();
        while (phase.load(std::memory_order_relaxed) < 3) {
            for (volatile int i = 0; i < 10'000; ++i) {
                med_progress++;
            }
        }
        auto t_end = std::chrono::high_resolution_clock::now();
        (void)t_start;
        (void)t_end;
    });

    // 高优先级线程：需要低优先级线程持有的资源
    std::thread high_thread([&]() {
        while (!start_flag.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        phase.store(1, std::memory_order_release);

        auto t_start = std::chrono::high_resolution_clock::now();
        resource.lock(); // 因低优先级线程持有锁而阻塞
        auto t_end = std::chrono::high_resolution_clock::now();
        high_wait_us = std::chrono::duration_cast<std::chrono::microseconds>(
                           t_end - t_start)
                           .count();

        // 临界区（较短）
        for (volatile int i = 0; i < 1'000'000; ++i) {
            high_progress++;
        }
        resource.unlock();
        phase.store(3, std::memory_order_release);
    });

    // 设置优先级（需要 SCHED_FIFO；无 root 权限可能失败）
    sched_param param;

    param.sched_priority = 10; // 低
    int ret_low = pthread_setschedparam(low_thread.native_handle(),
                                        SCHED_FIFO, &param);

    param.sched_priority = 50; // 中
    int ret_med = pthread_setschedparam(med_thread.native_handle(),
                                        SCHED_FIFO, &param);

    param.sched_priority = 90; // 高
    int ret_high = pthread_setschedparam(high_thread.native_handle(),
                                         SCHED_FIFO, &param);

    if (ret_low != 0 || ret_med != 0 || ret_high != 0) {
        std::cout << "  注意: 无法设置实时优先级（需要 root 权限或 "
                     "CAP_SYS_NICE）。\n";
        std::cout << "  以默认调度策略运行 - 可能无法观察到优先级反转。\n\n";
    } else {
        std::cout << "  实时优先级已设置: 低=10, 中=50, 高=90\n\n";
    }

    start_flag.store(true, std::memory_order_release);

    low_thread.join();
    med_thread.join();
    high_thread.join();

    std::cout << "  优先级反转场景:\n";
    std::cout << "    1. 低优先级线程获取锁\n";
    std::cout << "    2. 高优先级线程抢占，尝试获取锁 -> 阻塞\n";
    std::cout << "    3. 中优先级线程抢占低优先级线程（而非高优先级，因为高优先级在等待）\n";
    std::cout << "       -> 高优先级线程等待中优先级，尽管中优先级线程优先级更低！\n\n";

    std::cout << "  结果:\n";
    std::cout << "    低优先级进度:   " << low_progress.load() << "\n";
    std::cout << "    中优先级进度:   " << med_progress.load() << "\n";
    std::cout << "    高优先级进度:  " << high_progress.load() << "\n";
    std::cout << "    高优先级等待时间: " << high_wait_us.load() << " us\n";
    std::cout << "\n  解决方案: 使用 PTHREAD_PRIO_INHERIT 互斥锁协议\n";
}

// ============================================================================
// 演示 5: 原子内存顺序演示
// ============================================================================
void demo_memory_ordering() {
    print_header("演示 5: 原子内存顺序");

    // 展示 relaxed 与 seq_cst 的区别
    constexpr int ITERATIONS = 1'000'000;

    // Relaxed 计数器
    {
        std::atomic<int64_t> counter{0};
        std::vector<std::thread> threads;

        Timer t;
        t.start();
        for (int i = 0; i < 4; ++i) {
            threads.emplace_back([&]() {
                for (int k = 0; k < ITERATIONS / 4; ++k) {
                    counter.fetch_add(1, std::memory_order_relaxed);
                }
            });
        }
        for (auto &th : threads) th.join();
        double relaxed_time = t.elapsed_ms();

        std::cout << "  Relaxed 顺序:\n";
        std::cout << "    最终值: " << counter.load() << "\n";
        std::cout << "    耗时: " << std::fixed << std::setprecision(2)
                  << relaxed_time << " ms\n\n";
    }

    // Seq_cst 计数器
    {
        std::atomic<int64_t> counter{0};
        std::vector<std::thread> threads;

        Timer t;
        t.start();
        for (int i = 0; i < 4; ++i) {
            threads.emplace_back([&]() {
                for (int k = 0; k < ITERATIONS / 4; ++k) {
                    counter.fetch_add(1, std::memory_order_seq_cst);
                }
            });
        }
        for (auto &th : threads) th.join();
        double sc_time = t.elapsed_ms();

        std::cout << "  顺序一致性:\n";
        std::cout << "    最终值: " << counter.load() << "\n";
        std::cout << "    耗时: " << std::fixed << std::setprecision(2)
                  << sc_time << " ms\n";
    }

    std::cout << "\n  注意: 在 x86 上，relaxed 和 seq_cst 性能相近"
              << "，因为其强内存模型。\n"
              << "  在 ARM（弱内存模型）上，relaxed 会显著更快。\n";
}

// ============================================================================
// 演示 6: 使用条件变量的生产者-消费者
// ============================================================================
void demo_producer_consumer() {
    print_header("演示 6: 使用条件变量的生产者-消费者");

    constexpr int NUM_ITEMS = 1'000'000;
    std::queue<int> queue;
    std::mutex mtx;
    std::condition_variable cv_producer;
    std::condition_variable cv_consumer;
    constexpr size_t MAX_QUEUE = 100;
    bool producers_done = false;

    std::atomic<int64_t> produced{0};
    std::atomic<int64_t> consumed{0};

    auto producer_fn = [&](int offset, int count) {
        for (int i = 0; i < count; ++i) {
            std::unique_lock<std::mutex> lock(mtx);
            cv_producer.wait(lock, [&] { return queue.size() < MAX_QUEUE; });
            queue.push(offset + i);
            produced++;
            lock.unlock();
            cv_consumer.notify_one();
        }
    };

    auto consumer_fn = [&]() {
        while (true) {
            std::unique_lock<std::mutex> lock(mtx);
            cv_consumer.wait(lock, [&] {
                return !queue.empty() || producers_done;
            });
            if (!queue.empty()) {
                int val = queue.front();
                queue.pop();
                consumed++;
                lock.unlock();
                cv_producer.notify_one();
                volatile int sink = val;
                (void)sink;
            } else if (producers_done && queue.empty()) {
                break;
            }
        }
    };

    Timer t;
    t.start();

    std::thread p1(producer_fn, 0, NUM_ITEMS / 2);
    std::thread p2(producer_fn, NUM_ITEMS / 2, NUM_ITEMS / 2);
    std::thread c1(consumer_fn);
    std::thread c2(consumer_fn);

    p1.join();
    p2.join();

    {
        std::lock_guard<std::mutex> lock(mtx);
        producers_done = true;
    }
    cv_consumer.notify_all();

    c1.join();
    c2.join();

    double elapsed = t.elapsed_ms();

    std::cout << "  已生产项数: " << produced.load() << "\n";
    std::cout << "  已消费项数: " << consumed.load() << "\n";
    std::cout << "  耗时: " << std::fixed << std::setprecision(2)
              << elapsed << " ms\n";
    std::cout << "  吞吐量: " << std::fixed << std::setprecision(0)
              << (NUM_ITEMS / elapsed * 1000.0) << " 项/秒\n";
}
