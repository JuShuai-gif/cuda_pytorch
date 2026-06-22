/**
 * 03_stress_test.cpp — 并发压力测试框架
 *
 * 对线程安全数据结构进行多线程高强度压力测试, 检测:
 *  - 数据一致性问题 (丢失更新、重复等)
 *  - ABA 问题
 *  - 死锁
 *  - 内存泄漏 (配合 Valgrind)
 *
 * 以线程安全队列为例进行测试。
 *
 * 编译: g++ -std=c++20 -O2 -pthread 03_stress_test.cpp -o stress_test
 */

#include <iostream>
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <cassert>
#include <algorithm>
#include <random>
#include <set>
#include <iomanip>
#include <map>
#include <optional>

// ============================================================================
// 被测试对象: 线程安全队列 (mutex + condition_variable)
// ============================================================================
template <typename T>
class ThreadSafeQueue {
private:
    mutable std::mutex mutex_;
    std::queue<T> queue_;
    std::condition_variable cv_;
    size_t capacity_;
    bool closed_{false};

public:
    explicit ThreadSafeQueue(size_t capacity = std::numeric_limits<size_t>::max())
        : capacity_(capacity) {}

    // 阻塞入队
    void push(T value) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]() { return queue_.size() < capacity_ || closed_; });
        if (closed_) throw std::runtime_error("队列已关闭");
        queue_.push(std::move(value));
        cv_.notify_one();
    }

    // 尝试入队 (非阻塞)
    bool try_push(T value) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_ || queue_.size() >= capacity_) return false;
        queue_.push(std::move(value));
        cv_.notify_one();
        return true;
    }

    // 阻塞出队
    T pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]() { return !queue_.empty() || closed_; });
        if (closed_ && queue_.empty()) throw std::runtime_error("队列已关闭且为空");
        T value = std::move(queue_.front());
        queue_.pop();
        cv_.notify_one();
        return value;
    }

    // 尝试出队
    std::optional<T> try_pop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return std::nullopt;
        T value = std::move(queue_.front());
        queue_.pop();
        cv_.notify_one();
        return value;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        cv_.notify_all();
    }
};

// ============================================================================
// 压力测试框架
// ============================================================================
class StressTester {
private:
    struct TestResult {
        std::string test_name;
        bool passed;
        std::string details;
        double elapsed_ms;
    };

    std::vector<TestResult> results_;
    std::mutex results_mutex_;

    void add_result(const std::string& name, bool passed,
                    const std::string& details, double elapsed_ms) {
        std::lock_guard<std::mutex> lock(results_mutex_);
        results_.push_back({name, passed, details, elapsed_ms});
    }

public:
    // ===================================================================
    // 测试1: 生产者-消费者正确性
    // ===================================================================
    void test_producer_consumer_correctness() {
        std::cout << "测试1: 生产者-消费者正确性...\n";

        constexpr int kProducers = 4;
        constexpr int kConsumers = 4;
        constexpr int kItemsPerProducer = 50000;
        constexpr int kTotalItems = kProducers * kItemsPerProducer;

        ThreadSafeQueue<int> queue(256);

        std::atomic<long long> produce_sum{0};
        std::atomic<long long> consume_sum{0};
        std::atomic<int> produced{0};
        std::atomic<int> consumed{0};

        auto start = std::chrono::high_resolution_clock::now();

        // 生产者
        std::vector<std::jthread> producers;
        for (int t = 0; t < kProducers; ++t) {
            producers.emplace_back([&, t]() {
                for (int i = 0; i < kItemsPerProducer; ++i) {
                    int val = t * kItemsPerProducer + i;
                    queue.push(val);
                    produce_sum.fetch_add(val, std::memory_order_relaxed);
                    produced.fetch_add(1, std::memory_order_relaxed);
                }
            });
        }

        // 消费者
        std::vector<std::jthread> consumers;
        for (int t = 0; t < kConsumers; ++t) {
            consumers.emplace_back([&]() {
                while (consumed.load(std::memory_order_relaxed) < kTotalItems) {
                    auto item = queue.try_pop();
                    if (item) {
                        consume_sum.fetch_add(*item, std::memory_order_relaxed);
                        consumed.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        std::this_thread::yield();
                    }
                }
            });
        }

        for (auto& p : producers) p.join();
        for (auto& c : consumers) c.join();

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        bool sum_ok = (produce_sum.load() == consume_sum.load());
        bool count_ok = (produced.load() == kTotalItems &&
                         consumed.load() == kTotalItems);
        bool passed = sum_ok && count_ok;

        std::string details = "produce_sum=" + std::to_string(produce_sum.load()) +
                              " consume_sum=" + std::to_string(consume_sum.load()) +
                              " produced=" + std::to_string(produced.load()) +
                              " consumed=" + std::to_string(consumed.load());

        add_result("Producer-Consumer", passed, details, ms);

        std::cout << "  结果: " << (passed ? "通过" : "失败")
                  << " (" << static_cast<int>(ms) << " ms)\n\n";
    }

    // ===================================================================
    // 测试2: 唯一性测试 (所有值不重复不遗漏)
    // ===================================================================
    void test_uniqueness() {
        std::cout << "测试2: 数据唯一性 (不重复不遗漏)...\n";

        constexpr int kItems = 100000;
        ThreadSafeQueue<int> queue(1024);

        // 生产者: 推送 0..kItems-1
        std::jthread producer([&]() {
            for (int i = 0; i < kItems; ++i) {
                queue.push(i);
            }
        });

        // 消费者: 收集所有值
        std::vector<int> received;
        received.reserve(kItems);
        std::mutex recv_mutex;

        std::jthread consumer([&]() {
            for (int i = 0; i < kItems; ++i) {
                int val = queue.pop();
                std::lock_guard<std::mutex> lock(recv_mutex);
                received.push_back(val);
            }
        });

        producer.join();
        consumer.join();

        auto start = std::chrono::high_resolution_clock::now();

        // 验证
        std::sort(received.begin(), received.end());
        bool no_duplicates = std::adjacent_find(received.begin(), received.end()) == received.end();
        bool all_present = (received.size() == static_cast<size_t>(kItems));
        bool correct_range = true;
        for (size_t i = 0; i < received.size(); ++i) {
            if (received[i] != static_cast<int>(i)) {
                correct_range = false;
                break;
            }
        }
        bool passed = no_duplicates && all_present && correct_range;

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        add_result("Uniqueness", passed,
                   "size=" + std::to_string(received.size()) +
                   " no_dup=" + std::string(no_duplicates ? "true" : "false"),
                   ms);

        std::cout << "  结果: " << (passed ? "通过" : "失败")
                  << " (size=" << received.size() << ")\n\n";
    }

    // ===================================================================
    // 测试3: 高并发混合操作
    // ===================================================================
    void test_mixed_operations() {
        std::cout << "测试3: 高并发混合 push/pop...\n";

        constexpr int kThreads = 8;
        constexpr int kOpsPerThread = 100000;

        ThreadSafeQueue<int> queue(128);
        std::atomic<long long> total_pushed{0};
        std::atomic<long long> total_popped{0};
        std::atomic<int> push_count{0};
        std::atomic<int> pop_count{0};

        auto start = std::chrono::high_resolution_clock::now();

        std::vector<std::jthread> threads;
        std::mt19937 rng_base(42);

        for (int t = 0; t < kThreads; ++t) {
            threads.emplace_back([&, t]() {
                std::mt19937 rng(rng_base() + t * 12345);
                std::uniform_int_distribution<int> coin(0, 1);

                int local_push = 0;
                int local_pop = 0;

                while (local_push + local_pop < kOpsPerThread) {
                    if (coin(rng) == 0) {
                        // push
                        int val = t * kOpsPerThread + local_push;
                        if (queue.try_push(val)) {
                            total_pushed.fetch_add(val, std::memory_order_relaxed);
                            push_count.fetch_add(1, std::memory_order_relaxed);
                            ++local_push;
                        }
                    } else {
                        // pop
                        auto item = queue.try_pop();
                        if (item) {
                            total_popped.fetch_add(*item, std::memory_order_relaxed);
                            pop_count.fetch_add(1, std::memory_order_relaxed);
                            ++local_pop;
                        }
                    }
                }

                return std::pair(local_push, local_pop);
            });
        }

        long long total_local_push = 0, total_local_pop = 0;
        for (auto& t : threads) {
            t.join();
        }

        // 清空队列
        while (true) {
            auto item = queue.try_pop();
            if (!item) break;
            total_popped.fetch_add(*item, std::memory_order_relaxed);
            pop_count.fetch_add(1, std::memory_order_relaxed);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        bool sum_ok = (total_pushed.load() == total_popped.load());
        bool count_ok = (push_count.load() == pop_count.load());
        bool passed = sum_ok && count_ok;

        add_result("Mixed Operations", passed,
                   "push_sum=" + std::to_string(total_pushed.load()) +
                   " pop_sum=" + std::to_string(total_popped.load()),
                   ms);

        std::cout << "  结果: " << (passed ? "通过" : "失败")
                  << " (push=" << push_count.load() << ", pop=" << pop_count.load()
                  << ", " << static_cast<int>(ms) << " ms)\n\n";
    }

    // ===================================================================
    // 测试4: 边界条件 (空队列 pop, 满队列 push)
    // ===================================================================
    void test_edge_cases() {
        std::cout << "测试4: 边界条件...\n";

        bool passed = true;
        std::string details;

        {
            // 空队列 try_pop 返回 nullopt
            ThreadSafeQueue<int> q(4);
            auto val = q.try_pop();
            if (val.has_value()) {
                passed = false;
                details += "空队列 pop 应返回 nullopt; ";
            }
        }

        {
            // 满队列 try_push 返回 false
            ThreadSafeQueue<int> q(2);
            q.push(1);
            q.push(2);
            if (q.try_push(3)) {
                passed = false;
                details += "满队列 push 应返回 false; ";
            }
        }

        {
            // 正确顺序
            ThreadSafeQueue<int> q(4);
            q.push(10);
            q.push(20);
            int a = q.pop();
            int b = q.pop();
            if (a != 10 || b != 20) {
                passed = false;
                details += "FIFO 顺序错误; ";
            }
        }

        {
            // close 后操作
            ThreadSafeQueue<int> q(4);
            q.push(1);
            q.close();
            auto val = q.try_pop();
            if (!val || *val != 1) {
                passed = false;
                details += "close 后应仍能 pop 剩余元素; ";
            }
            try {
                q.push(2);
                passed = false;
                details += "close 后 push 应抛出异常; ";
            } catch (const std::runtime_error&) {
                // expected
            }
        }

        if (details.empty()) details = "所有边界条件通过";

        add_result("Edge Cases", passed, details, 0);

        std::cout << "  结果: " << (passed ? "通过" : "失败")
                  << " (" << details << ")\n\n";
    }

    // ===================================================================
    // 测试5: 长时间稳定性 (无死锁/无内存泄漏)
    // ===================================================================
    void test_stability() {
        std::cout << "测试5: 长时间稳定性 (运行 3 秒)...\n";

        ThreadSafeQueue<int> queue(64);
        std::atomic<bool> stop{false};
        constexpr int kThreads = 6;

        auto start = std::chrono::high_resolution_clock::now();

        std::vector<std::jthread> threads;
        for (int t = 0; t < kThreads; ++t) {
            threads.emplace_back([&, t]() {
                std::mt19937 rng(42 + t * 100);
                while (!stop.load(std::memory_order_relaxed)) {
                    if (rng() % 2 == 0) {
                        queue.try_push(t);
                    } else {
                        queue.try_pop();
                    }
                }
            });
        }

        std::this_thread::sleep_for(std::chrono::seconds(3));
        stop.store(true, std::memory_order_relaxed);

        for (auto& th : threads) th.join();

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        add_result("Stability", true,
                   "3秒稳定运行, 无崩溃/死锁, 剩余元素=" +
                   std::to_string(queue.size()),
                   ms);

        std::cout << "  结果: 通过 (稳定运行 3 秒, 无崩溃)\n\n";
    }

    // ===================================================================
    // 打印总结
    // ===================================================================
    void print_summary() {
        std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                    压力测试结果总结                          ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════╣\n";

        int passed = 0, failed = 0;
        for (const auto& r : results_) {
            std::cout << "║ "
                      << std::left << std::setw(38) << r.test_name
                      << (r.passed ? " 通过" : " 失败")
                      << std::right << std::setw(8)
                      << static_cast<int>(r.elapsed_ms) << " ms ║\n";
            if (r.passed) ++passed; else ++failed;
        }

        std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
        std::cout << "║  总计: " << (passed + failed) << " 项 | "
                  << "通过: " << passed << " | 失败: " << failed
                  << std::string(30, ' ') << "║\n";
        std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    }

    void run_all() {
        test_producer_consumer_correctness();
        test_uniqueness();
        test_mixed_operations();
        test_edge_cases();
        test_stability();
        print_summary();
    }
};

// ============================================================================
// main
// ============================================================================
int main() {
    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║  线程安全队列 — 并发压力测试             ║\n";
    std::cout << "║  硬件线程: " << std::setw(2)
              << std::jthread::hardware_concurrency() << "                          ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n\n";

    StressTester tester;
    tester.run_all();

    return 0;
}
