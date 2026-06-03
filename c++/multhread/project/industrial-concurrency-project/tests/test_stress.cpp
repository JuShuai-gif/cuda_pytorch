// Chapter 10.4: Stress / Concurrency Safety Test
// Validates no deadlocks, no data races, and correct results under heavy load.
// Designed to be run with ThreadSanitizer (TSan) for data race detection.

#include "task_scheduler/task_scheduler.hpp"
#include "task_scheduler/task_queue.hpp"
#include "task_scheduler/priority_task_queue.hpp"
#include "task_scheduler/concurrent_cache.hpp"
#include "task_scheduler/spinlock.hpp"
#include "task_scheduler/logger.hpp"
#include <cassert>
#include <iostream>
#include <vector>
#include <atomic>
#include <random>
#include <chrono>

using namespace task_scheduler;

// Ch10.4.1: High-contention thread pool stress test.
// Submits thousands of tasks with varying execution times.
void stress_thread_pool() {
    std::cout << "  [stress_thread_pool] ";
    ThreadPool pool(8);
    std::atomic<size_t> counter{0};
    constexpr size_t N = 5000;

    std::vector<std::future<size_t>> futures;
    futures.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        futures.push_back(pool.submit([i, &counter]() -> size_t {
            // Vary work amount to trigger work stealing (Ch8.4).
            size_t work = i % 100 + 1;
            volatile size_t dummy = 0;
            for (size_t j = 0; j < work; ++j) {
                dummy += j;
            }
            counter.fetch_add(1, std::memory_order_relaxed);
            return dummy;
        }));
    }

    size_t total = 0;
    for (auto& f : futures) {
        total += f.get();
    }
    assert(counter.load() == N);
    std::cout << "PASSED (counter=" << counter.load() << ")\n";
}

// Ch10.4.2: MPMC queue stress test.
// Many producers and consumers operating simultaneously.
void stress_task_queue() {
    std::cout << "  [stress_task_queue] ";
    TaskQueue<int> q;
    constexpr int N = 10000;
    constexpr int producers = 4;
    constexpr int consumers = 4;

    std::atomic<int> produced{0};
    std::atomic<int> consumed{0};
    std::atomic<bool> done{false};

    // Producers (Ch6.2.1)
    std::vector<std::jthread> producer_threads;
    for (int p = 0; p < producers; ++p) {
        producer_threads.emplace_back([&q, &produced, p] {
            for (int i = 0; i < N / producers; ++i) {
                q.push(p * 10000 + i);
                produced.fetch_add(1);
            }
        });
    }

    // Consumers (Ch6.2.2/Ch6.2.4)
    std::vector<std::jthread> consumer_threads;
    for (int c = 0; c < consumers; ++c) {
        consumer_threads.emplace_back([&q, &consumed, &done, &produced] {
            while (!done.load()) {
                if (auto item = q.try_pop()) {
                    consumed.fetch_add(1);
                } else if (produced.load() >= N && q.empty()) {
                    break;
                } else {
                    std::this_thread::yield();
                }
            }
        });
    }

    for (auto& t : producer_threads) t.join();
    done.store(true);
    q.notify_all();
    for (auto& t : consumer_threads) t.join();

    assert(consumed.load() == N);
    std::cout << "PASSED (consumed=" << consumed.load() << ")\n";
}

// Ch10.4.3: Priority queue stress test.
void stress_priority_queue() {
    std::cout << "  [stress_priority_queue] ";
    PriorityTaskQueue<int> pq;
    constexpr int N = 2000;

    std::atomic<int> submitted{0};
    std::vector<std::jthread> producers;

    for (int p = 0; p < 4; ++p) {
        producers.emplace_back([&pq, &submitted, p] {
            for (int i = 0; i < N / 4; ++i) {
                TaskPriority prio = static_cast<TaskPriority>(rand() % 4);
                pq.push(p * 10000 + i, prio);
                submitted.fetch_add(1);
            }
        });
    }

    // Consumer verifies all items received
    std::atomic<int> received{0};
    auto consumer = std::jthread([&pq, &received, &submitted] {
        while (received.load() < N) {
            if (auto item = pq.try_pop()) {
                received.fetch_add(1);
            } else if (submitted.load() >= N) {
                // Drain remaining after all submitted
                auto last = pq.wait_and_pop_for(std::chrono::milliseconds(10));
                if (last) received.fetch_add(1);
                if (!last && pq.empty()) break;
            } else {
                std::this_thread::yield();
            }
        }
    });

    for (auto& t : producers) t.join();
    consumer.join();

    assert(received.load() == N);
    std::cout << "PASSED (received=" << received.load() << ")\n";
}

// Ch10.4.4: Concurrent cache stress test (Ch3.3.2).
void stress_concurrent_cache() {
    std::cout << "  [stress_concurrent_cache] ";
    ConcurrentCache<int, int> cache(256);
    std::atomic<size_t> ops{0};
    constexpr size_t TARGET = 50000;

    // Reader threads (shared_lock - Ch3.3.2)
    std::vector<std::jthread> readers;
    for (int r = 0; r < 4; ++r) {
        readers.emplace_back([&cache, &ops] {
            while (ops.load() < TARGET) {
                for (int k = 0; k < 128; ++k) {
                    cache.get(k);
                    ops.fetch_add(1);
                }
            }
        });
    }

    // Writer threads (unique_lock - Ch3.3.1)
    std::vector<std::jthread> writers;
    for (int w = 0; w < 2; ++w) {
        writers.emplace_back([&cache] {
            for (int k = 0; k < 256; ++k) {
                cache.put(k, k * 2);
            }
        });
    }

    for (auto& t : readers) t.join();
    for (auto& t : writers) t.join();

    assert(ops.load() >= TARGET);
    std::cout << "PASSED (ops=" << ops.load() << ")\n";
}

// Ch10.4.5: Multi-component integration stress test.
void stress_integration() {
    std::cout << "  [stress_integration] ";
    TaskScheduler scheduler(8, 64);
    std::atomic<size_t> completed{0};
    constexpr size_t N = 500;

    // Submit mixed-priority tasks
    std::vector<std::future<int>> futures;
    for (size_t i = 0; i < N; ++i) {
        TaskPriority prio = static_cast<TaskPriority>(i % 4);
        futures.push_back(
            scheduler.submit(prio, TS_FORMAT("task_{}", i),
                [i, &completed]() -> int {
                    completed.fetch_add(1);
                    return static_cast<int>(i);
                }));
    }

    // Concurrently access cache while tasks run
    std::jthread cache_user([&scheduler, &completed] {
        while (completed.load() < N) {
            scheduler.cache_put(TS_FORMAT("key_{}", rand() % 100),
                              TS_FORMAT("val_{}", rand()));
        }
    });

    for (auto& f : futures) {
        f.get();
    }
    cache_user.join();

    assert(completed.load() == N);
    std::cout << "PASSED (completed=" << completed.load() << ")\n";
}

// Ch10.4.6: Spinlock stress test (Ch5).
void stress_spinlock() {
    std::cout << "  [stress_spinlock] ";
    spinlock sl;
    std::atomic<size_t> counter{0};
    constexpr size_t N = 100000;

    std::vector<std::jthread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&sl, &counter] {
            for (size_t i = 0; i < N / 4; ++i) {
                spinlock_guard guard(sl);
                counter.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    for (auto& t : threads) t.join();
    assert(counter.load() == N);
    std::cout << "PASSED (counter=" << counter.load() << ")\n";
}

int main() {
    std::cout << "=== Stress Tests ===\n";
    std::cout << "NOTE: Run with ThreadSanitizer for data race detection:\n";
    std::cout << "  cmake -DCMAKE_BUILD_TYPE=Tsan .. && make\n\n";

    // Ch9.2: Set logger to minimal output during stress tests.
    Logger::instance().set_level(LogLevel::WARN);

    stress_thread_pool();
    stress_task_queue();
    stress_priority_queue();
    stress_concurrent_cache();
    stress_integration();
    stress_spinlock();

    std::cout << "\n=== All Stress Tests Passed ===\n";
    std::cout << "No deadlocks, no data races, all counters correct.\n";
    return 0;
}
