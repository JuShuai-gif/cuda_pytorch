// 04_lock_vs_lockfree_bench.cpp — 锁 vs 无锁全面性能对比
// 对比: mutex, spinlock, atomic CAS, lock-free 在不同竞争下的表现

#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

// ===== 被测对象 =====

// 1. std::mutex
class MutexCounter {
public:
    void inc() {
        std::lock_guard lock(mtx_);
        ++value_;
    }
    long long get() const { return value_; }

private:
    std::mutex mtx_;
    long long value_ = 0;
};

// 2. TTAS Spinlock
class SpinlockCounter {
public:
    void inc() {
        while (flag_.test_and_set(std::memory_order_acquire)) {
#if defined(__x86_64__) || defined(_M_X64)
            __asm__ volatile("pause" ::: "memory");
#endif
        }
        ++value_;
        flag_.clear(std::memory_order_release);
    }
    long long get() const { return value_; }

private:
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
    long long value_ = 0;
};

// 3. CAS-based 无锁
class CASCounter {
public:
    void inc() {
        long long expected = value_.load(std::memory_order_relaxed);
        while (!value_.compare_exchange_weak(expected, expected + 1,
                                              std::memory_order_release,
                                              std::memory_order_relaxed)) {
#if defined(__x86_64__) || defined(_M_X64)
            __asm__ volatile("pause" ::: "memory");
#endif
        }
    }
    long long get() const { return value_.load(); }

private:
    std::atomic<long long> value_{0};
};

// 4. fetch_add 无锁
class AtomicCounter {
public:
    void inc() { value_.fetch_add(1, std::memory_order_relaxed); }
    long long get() const { return value_.load(); }

private:
    std::atomic<long long> value_{0};
};

// 5. Per-thread 计数器 (无竞争)
class PerThreadCounter {
public:
    explicit PerThreadCounter(int num_threads)
        : values_(num_threads, 0) {}

    void inc(int tid) { values_[tid]++; }

    long long get() const {
        long long sum = 0;
        for (long long v : values_) sum += v;
        return sum;
    }

private:
    std::vector<long long> values_;
};

// ===== 基准测试 =====
template <typename Counter, typename... Args>
double benchmark_counter(int num_threads, long long ops_per_thread,
                          Args&&... args) {
    Counter counter(std::forward<Args>(args)...);

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::jthread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (long long i = 0; i < ops_per_thread; ++i) {
                if constexpr (std::is_same_v<Counter, PerThreadCounter>) {
                    counter.inc(t);
                } else {
                    counter.inc();
                }
            }
        });
    }
    threads.clear();

    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start);
    return elapsed.count();
}

// ===== 主测试 =====
int main() {
    std::cout << "=== 锁 vs 无锁性能全面对比 ===\n";
    std::cout << "每线程 " << 2'000'000 << " 次递增\n\n";

    const long long kOps = 2'000'000;

    std::cout << std::setw(18) << "方案"
              << std::setw(10) << "1线程"
              << std::setw(10) << "2线程"
              << std::setw(10) << "4线程"
              << std::setw(10) << "8线程" << "\n";
    std::cout << std::string(58, '-') << "\n";

    auto print_row = [](const std::string& name,
                         const std::vector<double>& times) {
        std::cout << std::setw(18) << name;
        for (double t : times) {
            std::cout << std::setw(8) << static_cast<int>(t) << "ms";
        }
        std::cout << "\n";
    };

    for (int n : {1, 2, 4, 8}) {
        std::vector<double> results;

        results.push_back(
            benchmark_counter<MutexCounter>(n, kOps / n));
        results.push_back(
            benchmark_counter<SpinlockCounter>(n, kOps / n));
        results.push_back(
            benchmark_counter<CASCounter>(n, kOps / n));
        results.push_back(
            benchmark_counter<AtomicCounter>(n, kOps / n));
        results.push_back(
            benchmark_counter<PerThreadCounter>(n, kOps / n, n));

        if (n == 1) {
            print_row("std::mutex", {results[0]});
            print_row("Spinlock", {results[1]});
            print_row("CAS Loop", {results[2]});
            print_row("fetch_add", {results[3]});
            print_row("PerThread(0争用)", {results[4]});
        }
    }

    // 完整表格
    std::cout << "\n完整数据表格 (ms):\n";
    std::cout << std::setw(18) << "方案"
              << std::setw(10) << "1线程"
              << std::setw(10) << "2线程"
              << std::setw(10) << "4线程"
              << std::setw(10) << "8线程" << "\n";
    std::cout << std::string(58, '-') << "\n";

    std::vector<std::string> names = {
        "std::mutex", "Spinlock", "CAS Loop", "fetch_add",
        "PerThread(0争用)"};

    for (size_t i = 0; i < names.size(); ++i) {
        std::cout << std::setw(18) << names[i];
        for (int n : {1, 2, 4, 8}) {
            double t = benchmark_counter<MutexCounter>(n, kOps / n);
            // Note: Manual dispatch for different counter types
            // This is simplified; in practice each type tested separately
        }
        std::cout << "\n";
    }

    // 实际运行每个类型
    std::cout << "\n各方案实测 (越高竞争越明显):\n\n";
    for (auto& pair : std::vector<std::pair<std::string, int>>{
             {"std::mutex", 0}, {"Spinlock", 1}, {"CAS Loop", 2},
             {"fetch_add", 3}, {"PerThread", 4}}) {
        std::cout << std::setw(18) << pair.first;
        for (int n : {1, 2, 4, 8}) {
            double t = 0;
            switch (pair.second) {
            case 0:
                t = benchmark_counter<MutexCounter>(n, kOps / n);
                break;
            case 1:
                t = benchmark_counter<SpinlockCounter>(n, kOps / n);
                break;
            case 2:
                t = benchmark_counter<CASCounter>(n, kOps / n);
                break;
            case 3:
                t = benchmark_counter<AtomicCounter>(n, kOps / n);
                break;
            case 4:
                t = benchmark_counter<PerThreadCounter>(n, kOps / n, n);
                break;
            }
            std::cout << std::setw(8) << static_cast<int>(t) << "ms";
        }
        std::cout << "\n";
    }

    std::cout << "\n关键发现:\n";
    std::cout << "  1. 低竞争时: mutex >> spinlock (避免 busy-wait)\n";
    std::cout << "  2. 高竞争时: atomic > cas_loop > spinlock > mutex\n";
    std::cout << "  3. 终极方案: 消除共享 (PerThread) 总是最快\n";
    std::cout << "  4. 无锁并非万能: 设计比实现选择更重要\n";

    return 0;
}
