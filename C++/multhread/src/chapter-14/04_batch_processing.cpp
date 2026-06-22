// 04_batch_processing.cpp — 批量处理优化
// 演示: 批量入队、批量写入、与逐个操作的性能对比

#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <queue>
#include <syncstream>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

// ===== 1. 批量 vs 逐个入队性能对比 =====
void demo_batch_enqueue() {
    std::cout << "=== 批量入队 vs 逐个入队 ===\n";

    const int kTotalItems = 1'000'000;
    const int kBatchSize = 100;
    const int kProducers = 4;

    // 方案 A: 逐个加锁入队
    {
        std::mutex mtx;
        std::queue<int> q;
        std::atomic<int> produced{0};

        auto start = std::chrono::high_resolution_clock::now();

        std::vector<std::jthread> producers;
        for (int p = 0; p < kProducers; ++p) {
            producers.emplace_back([&]() {
                for (int i = 0; i < kTotalItems / kProducers; ++i) {
                    std::lock_guard lock(mtx);
                    q.push(i);
                    produced.fetch_add(1);
                }
            });
        }
        producers.clear();

        auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start);
        std::cout << "  逐个入队: " << elapsed.count()
                  << " ms (锁获取 " << kTotalItems << " 次)\n";
    }

    // 方案 B: 批量入队
    {
        std::mutex mtx;
        std::queue<int> q;
        std::atomic<int> produced{0};

        auto start = std::chrono::high_resolution_clock::now();

        std::vector<std::jthread> producers;
        for (int p = 0; p < kProducers; ++p) {
            producers.emplace_back([&]() {
                std::vector<int> batch;
                batch.reserve(kBatchSize);
                int total = kTotalItems / kProducers;

                for (int i = 0; i < total; ++i) {
                    batch.push_back(i);
                    if (batch.size() >= static_cast<size_t>(kBatchSize)) {
                        std::lock_guard lock(mtx);
                        for (int v : batch) q.push(v);
                        produced.fetch_add(batch.size());
                        batch.clear();
                    }
                }
                // 剩余元素
                if (!batch.empty()) {
                    std::lock_guard lock(mtx);
                    for (int v : batch) q.push(v);
                    produced.fetch_add(batch.size());
                }
            });
        }
        producers.clear();

        auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start);
        std::cout << "  批量入队: " << elapsed.count()
                  << " ms (锁获取 ~"
                  << kTotalItems / kBatchSize << " 次)\n";
    }
}

// ===== 2. 批量处理的典型模式 =====
class BatchProcessor {
public:
    void add_item(int item) {
        buffer_[write_idx_].store(item, std::memory_order_relaxed);
        write_idx_++;

        if (write_idx_ >= kBatchSize) {
            flush_batch();
            write_idx_ = 0;
        }
    }

    ~BatchProcessor() {
        if (write_idx_ > 0) flush_batch();
    }

    int total_processed() const { return total_.load(); }

private:
    static constexpr int kBatchSize = 64;
    std::atomic<int> buffer_[kBatchSize];
    int write_idx_ = 0;
    std::mutex flush_mtx_;
    std::atomic<int> total_{0};

    void flush_batch() {
        std::lock_guard lock(flush_mtx_);
        // 模拟批量处理 (数据库写入 / 网络发送)
        total_.fetch_add(write_idx_, std::memory_order_relaxed);
    }
};

void demo_batch_processor() {
    std::cout << "\n=== 批量处理器模式 ===\n";

    BatchProcessor processor;
    const int kItems = 1000;
    const int kThreads = 4;

    std::vector<std::jthread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < kItems; ++i) {
                processor.add_item(t * kItems + i);
            }
        });
    }
    threads.clear();

    std::cout << "  处理 " << (kItems * kThreads)
              << " 个项目，批量大小 " << 64 << "\n";
    std::cout << "  实际处理: " << processor.total_processed() << "\n";
}

// ===== 3. 批量 vs 逐个: 原子计数器 =====
void demo_batch_atomic_counter() {
    std::cout << "\n=== 批量原子操作 vs 逐个原子操作 ===\n";

    const long long kIters = 2'000'000;
    const int kThreads = 4;

    // 逐个 fetch_add
    {
        std::atomic<long long> counter{0};
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<std::jthread> threads;
        for (int t = 0; t < kThreads; ++t) {
            threads.emplace_back([&]() {
                for (long long i = 0; i < kIters; ++i) {
                    counter.fetch_add(1, std::memory_order_relaxed);
                }
            });
        }
        threads.clear();

        auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start);
        std::cout << "  逐个原子加: " << elapsed.count() << " ms\n";
    }

    // 批量: 每线程维护局部计数器，最后合并
    {
        std::atomic<long long> counter{0};
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<std::jthread> threads;
        for (int t = 0; t < kThreads; ++t) {
            threads.emplace_back([&]() {
                long long local = 0;
                const long long kBatchSize = 1000;
                for (long long i = 0; i < kIters; ++i) {
                    ++local;
                    if (local % kBatchSize == 0) {
                        counter.fetch_add(kBatchSize,
                                          std::memory_order_relaxed);
                        local = 0;
                    }
                }
                if (local > 0) {
                    counter.fetch_add(local, std::memory_order_relaxed);
                }
            });
        }
        threads.clear();

        auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start);
        std::cout << "  批量原子加: " << elapsed.count() << " ms\n";
    }
}

int main() {
    demo_batch_enqueue();
    demo_batch_processor();
    demo_batch_atomic_counter();

    std::cout << "\n批量处理的核心: 用空间换时间，减少同步开销。\n";
    return 0;
}
