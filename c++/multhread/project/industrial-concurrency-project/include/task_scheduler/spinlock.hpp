#pragma once
// Ch5：C++ 内存模型与原子类型的操作
// 实现一个带指数退避的 TTAS（Test-Test-And-Set）自旋锁。
// 演示 std::atomic_flag、memory_order_acquire/release 以及 x86 PAUSE 指令用于能效优化。
// C++ 内存模型与原子操作
// 实现 TTAS（测试-测试并设置）自旋锁，包含指数退避
// 演示 std::atomic_flag、memory_order_acquire/release 和 x86 PAUSE 指令

#include <atomic>
#include <thread>

// 根据平台选择适当的 PAUSE/暂停 指令
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #include <immintrin.h>
  #define SPINLOCK_PAUSE() _mm_pause()
#elif defined(__aarch64__) || defined(_M_ARM64)
  #define SPINLOCK_PAUSE() __asm__ __volatile__("yield")
#else
  #define SPINLOCK_PAUSE() ((void)0)
#endif

namespace task_scheduler {

// Ch5.3.2：std::atomic_flag 始终是 lock-free 的，非常适合做自旋锁。
// std::atomic_flag 始终是无锁的，是自旋锁的理想选择
class spinlock {
public:
    spinlock() noexcept : flag_(ATOMIC_FLAG_INIT) {}

    spinlock(const spinlock&) = delete;
    spinlock& operator=(const spinlock&) = delete;

    // 带指数退避的 TTAS 锁（Ch5.3.2 + Ch5.3.4 内存序）。
    // 指数退避：先读（廉价，在 L1 缓存共享状态），再竞争
    void lock() noexcept {
        // Ch5.3.4：memory_order_acquire 确保加锁后能读到前一个持有者写入的值。
        // 使用 memory_order_acquire 保证后续读取能看到之前线程的写入
        for (int backoff = 1; !try_lock(); backoff = std::min(backoff * 2, 4096)) {
            // TTAS：先通过读轮询（廉价，在 L1 缓存中以共享状态保持）
            // 测试阶段：先读标志位（memory_order_relaxed 足够，仅是轮询）
            for (int i = 0; i < backoff; ++i) {
                // Ch5.3.1：memory_order_relaxed 对轮询测试已足够。
                // 此处不需要同步——仅测试标志位。
                if (!flag_.test(std::memory_order_relaxed)) {
                    break; // 竞争可能已结束，现在尝试真正的加锁
                }
                SPINLOCK_PAUSE(); // Ch5.3.5：超线程的暂停提示
            }
        }
    }

    // Ch5.3.2：test_and_set 返回之前的值，并设置为 true。
    // 如果返回 false，表示我们获取了锁。
    // 尝试加锁：如果成功获取锁返回 true
    bool try_lock() noexcept {
        // memory_order_acquire：与 unlock 的 memory_order_release 配对。
        return !flag_.test_and_set(std::memory_order_acquire);
    }

    void unlock() noexcept {
        // Ch5.3.4：memory_order_release 确保在解锁之前的所有写入对下一个获取锁的线程可见。
        // 使用 memory_order_release 保证解锁前的所有写入对其他线程可见
        flag_.clear(std::memory_order_release);
    }

private:
    std::atomic_flag flag_;
};

// RAII 包装器（Ch3.2.1：std::lock_guard 模式，为自旋锁扩展）。
// RAII 守卫：自动加锁和解锁，异常安全
class spinlock_guard {
public:
    explicit spinlock_guard(spinlock& sl) noexcept : sl_(sl) { sl_.lock(); }
    ~spinlock_guard() noexcept { sl_.unlock(); }
    spinlock_guard(const spinlock_guard&) = delete;
    spinlock_guard& operator=(const spinlock_guard&) = delete;
private:
    spinlock& sl_;
};

} // namespace task_scheduler
