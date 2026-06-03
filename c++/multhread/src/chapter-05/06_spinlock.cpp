// 06_spinlock.cpp - 工业级自旋锁实现与性能对比
// 实现 SpinLock（std::atomic_flag），与 std::mutex 做性能基准对比
// 包含：plain spinlock, yield spinlock, exponential backoff spinlock
//       以及 std::mutex 的性能对比

#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

// ===== 1. 基础自旋锁（纯忙等） =====
class PlainSpinLock {
public:
    void lock() {
        while (flag_.test_and_set(std::memory_order_acquire)) {
            // 空循环：高 CPU 占用，低延迟
        }
    }
    void unlock() {
        flag_.clear(std::memory_order_release);
    }

private:
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
};

// ===== 2. Yield 自旋锁（忙等时让出 CPU） =====
class YieldSpinLock {
public:
    void lock() {
        while (flag_.test_and_set(std::memory_order_acquire)) {
            // 让出当前时间片，减少 CPU 浪费
            std::this_thread::yield();
        }
    }
    void unlock() {
        flag_.clear(std::memory_order_release);
    }

private:
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
};

// ===== 3. 指数退避自旋锁 =====
class ExponentialBackoffSpinLock {
public:
    void lock() {
        int delay = 1;
        const int kMaxDelay = 1024;

        while (flag_.test_and_set(std::memory_order_acquire)) {
            // 指数增长的等待时间，减少竞争
            for (int i = 0; i < delay; ++i) {
                // x86 PAUSE 指令等效：减少 CPU 流水线竞争
                asm volatile("pause" ::: "memory");
            }
            delay = std::min(delay * 2, kMaxDelay);
        }
    }
    void unlock() {
        flag_.clear(std::memory_order_release);
    }

private:
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
};

// ===== 4. TTAS (Test-Test-And-Set) 自旋锁 =====
class TTASSpinLock {
public:
    void lock() {
        while (true) {
            // 先测试（读取），避免无效的 test_and_set
            if (!flag_.test(std::memory_order_relaxed)) {
                // 仅在可能成功时才尝试获取
                if (!flag_.test_and_set(std::memory_order_acquire)) {
                    return; // 获取成功
                }
            }
            std::this_thread::yield();
        }
    }
    void unlock() {
        flag_.clear(std::memory_order_release);
    }

private:
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
};

// ===== 性能基准测试 =====
template <typename Lock>
long long benchmark_lock(const std::string& name,
                         int num_threads, int iters_per_thread) {
    Lock       lock;
    long long  shared_counter = 0;

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::jthread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&]() {
            for (int j = 0; j < iters_per_thread; ++j) {
                std::lock_guard<Lock> guard(lock);
                ++shared_counter;
            }
        });
    }
    threads.clear();

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start);

    std::cout << "  " << name << ": " << elapsed.count() << "ms"
              << " (计数器=" << shared_counter
              << ", 期望=" << num_threads * iters_per_thread << ")\n";

    return elapsed.count();
}

int main() {
    const int kThreads = 4;
    const int kIters   = 500000;

    std::cout << "=== 自旋锁性能对比（" << kThreads
              << " 线程 × " << kIters << " 次迭代） ===\n\n";

    // Plain SpinLock（高竞争，CPU 高负载）
    benchmark_lock<PlainSpinLock>("PlainSpinLock          ", kThreads, kIters);

    // Yield SpinLock（调度的开销较大）
    benchmark_lock<YieldSpinLock>("YieldSpinLock          ", kThreads, kIters);

    // TTAS SpinLock（减少缓存一致性流量）
    benchmark_lock<TTASSpinLock>("TTASSpinLock           ", kThreads, kIters);

    // Exponential Backoff（平衡延迟与功耗）
    benchmark_lock<ExponentialBackoffSpinLock>("ExpBackoffSpinLock     ", kThreads, kIters);

    // std::mutex 作为基准对比
    benchmark_lock<std::mutex>("std::mutex             ", kThreads, kIters);

    std::cout << "\n提示:\n"
              << "  - PlainSpinLock: 延迟最低，CPU 占用最高\n"
              << "  - YieldSpinLock: 低 CPU，调度开销大\n"
              << "  - TTASSpinLock: 减少总线竞争，适合多核\n"
              << "  - ExpBackoff:   高竞争下表现好\n"
              << "  - std::mutex:   内核态休眠，长时间等待更优\n";

    return 0;
}
