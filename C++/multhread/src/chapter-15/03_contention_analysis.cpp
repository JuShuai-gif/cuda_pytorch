// 03_contention_analysis.cpp — 锁竞争分析
// 演示: 不同竞争级别的性能、锁粒度对比、竞争检测

#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace std::chrono_literals;

// ===== 1. 竞争计数器 =====
class ContentionAwareMutex {
public:
    void lock() {
        int spins = 0;
        while (!mtx_.try_lock()) {
            ++spins;
            if (spins > 1000) {
                // 自旋太久，退化为阻塞
                mtx_.lock();
                contention_count_.fetch_add(1);
                return;
            }
            // CPU PAUSE (x86) — 减少功耗和争用
#if defined(__x86_64__) || defined(_M_X64)
            __asm__ volatile("pause" ::: "memory");
#endif
        }
        if (spins > 0) {
            spin_count_.fetch_add(spins);
        }
    }

    void unlock() { mtx_.unlock(); }

    long long contention_count() const { return contention_count_.load(); }
    long long total_spins() const { return spin_count_.load(); }

private:
    std::mutex mtx_;
    std::atomic<long long> contention_count_{0};
    std::atomic<long long> spin_count_{0};
};

// ===== 2. 不同竞争级别性能测试 =====
void demo_contention_levels() {
    std::cout << "=== 竞争级别对性能的影响 ===\n\n";

    const long long kOpsPerThread = 500'000;

    for (int num_threads : {1, 2, 4, 8}) {
        // 方案 A: 粗粒度全局锁 (高竞争)
        {
            std::mutex global_mtx;
            long long shared_data = 0;
            long long total_spins = 0;

            auto start = std::chrono::high_resolution_clock::now();

            std::vector<std::jthread> threads;
            for (int t = 0; t < num_threads; ++t) {
                threads.emplace_back([&]() {
                    for (long long i = 0; i < kOpsPerThread; ++i) {
                        std::lock_guard lock(global_mtx);
                        ++shared_data;
                    }
                });
            }
            threads.clear();

            auto elapsed =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - start);

            std::cout << "  全局锁 " << num_threads << " 线程: "
                      << std::setw(6) << elapsed.count() << " ms | "
                      << "counter=" << shared_data << "\n";
        }

        // 方案 B: 原子操作 (无锁，但仍有竞争)
        {
            std::atomic<long long> shared_data{0};

            auto start = std::chrono::high_resolution_clock::now();

            std::vector<std::jthread> threads;
            for (int t = 0; t < num_threads; ++t) {
                threads.emplace_back([&]() {
                    for (long long i = 0; i < kOpsPerThread; ++i) {
                        shared_data.fetch_add(1,
                            std::memory_order_relaxed);
                    }
                });
            }
            threads.clear();

            auto elapsed =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - start);

            std::cout << "  原子操作 " << num_threads << " 线程: "
                      << std::setw(6) << elapsed.count() << " ms | "
                      << "counter=" << shared_data.load() << "\n";
        }
    }
}

// ===== 3. 锁粒度对比: 粗粒度 vs 细粒度 =====
class CoarseGrainedMap {
public:
    void insert(int key, int value) {
        std::lock_guard lock(mtx_);
        data_[key] = value;
    }

    int get(int key) {
        std::lock_guard lock(mtx_);
        return data_[key];
    }

private:
    std::mutex mtx_;
    std::unordered_map<int, int> data_;
};

class FineGrainedMap {
public:
    void insert(int key, int value) {
        auto& bucket = buckets_[key % kNumBuckets];
        std::lock_guard lock(bucket.mtx);
        bucket.data[key] = value;
    }

    int get(int key) {
        auto& bucket = buckets_[key % kNumBuckets];
        std::lock_guard lock(bucket.mtx);
        return bucket.data[key];
    }

private:
    static constexpr int kNumBuckets = 16;
    struct Bucket {
        std::mutex mtx;
        std::unordered_map<int, int> data;
    };
    std::vector<Bucket> buckets_{kNumBuckets};
};

void demo_lock_granularity() {
    std::cout << "\n=== 锁粒度: 粗粒度 vs 细粒度 ===\n";

    const int kOps = 500'000;
    const int kThreads = 4;

    // 粗粒度
    {
        CoarseGrainedMap map;
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<std::jthread> threads;
        for (int t = 0; t < kThreads; ++t) {
            threads.emplace_back([&, t]() {
                for (int i = 0; i < kOps; ++i) {
                    map.insert(i, t);
                }
            });
        }
        threads.clear();

        auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start);
        std::cout << "  粗粒度 (1 lock):  " << elapsed.count() << " ms\n";
    }

    // 细粒度
    {
        FineGrainedMap map;
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<std::jthread> threads;
        for (int t = 0; t < kThreads; ++t) {
            threads.emplace_back([&, t]() {
                for (int i = 0; i < kOps; ++i) {
                    map.insert(i, t);
                }
            });
        }
        threads.clear();

        auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start);
        std::cout << "  细粒度 (16 locks): " << elapsed.count() << " ms\n";
    }
}

// ===== 4. 读写锁竞争分析 =====
void demo_rwlock_contention() {
    std::cout << "\n=== 读写锁竞争: shared_mutex vs mutex ===\n";

    const long long kReads = 1'000'000;
    const long long kWrites = 10'000;
    const int kReaders = 8;
    const int kWriters = 2;

    // shared_mutex
    {
        std::shared_mutex rw_mtx;
        int data = 0;

        auto start = std::chrono::high_resolution_clock::now();

        std::vector<std::jthread> threads;
        // Reader 线程
        for (int r = 0; r < kReaders; ++r) {
            threads.emplace_back([&]() {
                for (long long i = 0; i < kReads; ++i) {
                    std::shared_lock lock(rw_mtx);
                    volatile int v = data;
                    (void)v;
                }
            });
        }
        // Writer 线程
        for (int w = 0; w < kWriters; ++w) {
            threads.emplace_back([&]() {
                for (long long i = 0; i < kWrites; ++i) {
                    std::unique_lock lock(rw_mtx);
                    ++data;
                }
            });
        }
        threads.clear();

        auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start);

        std::cout << "  shared_mutex (8读2写): " << elapsed.count()
                  << " ms\n";
    }

    // mutex (对比)
    {
        std::mutex mtx;
        int data = 0;

        auto start = std::chrono::high_resolution_clock::now();

        std::vector<std::jthread> threads;
        for (int r = 0; r < kReaders; ++r) {
            threads.emplace_back([&]() {
                for (long long i = 0; i < kReads; ++i) {
                    std::lock_guard lock(mtx);
                    volatile int v = data;
                    (void)v;
                }
            });
        }
        for (int w = 0; w < kWriters; ++w) {
            threads.emplace_back([&]() {
                for (long long i = 0; i < kWrites; ++i) {
                    std::lock_guard lock(mtx);
                    ++data;
                }
            });
        }
        threads.clear();

        auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start);

        std::cout << "  mutex       (8读2写): " << elapsed.count()
                  << " ms\n";
    }
    std::cout << "  结论: 读多写少场景 shared_mutex 显著优于 mutex\n";
}

int main() {
    demo_contention_levels();
    demo_lock_granularity();
    demo_rwlock_contention();

    return 0;
}
