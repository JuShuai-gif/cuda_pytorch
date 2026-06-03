#pragma once
// Chapter 5: The C++ Memory Model and Operations on Atomic Types
// Implements a TTAS (Test-Test-And-Set) spinlock with exponential backoff.
// Demonstrates std::atomic_flag, memory_order_acquire/release, and
// x86 PAUSE instruction for power efficiency.

#include <atomic>
#include <thread>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #include <immintrin.h>
  #define SPINLOCK_PAUSE() _mm_pause()
#elif defined(__aarch64__) || defined(_M_ARM64)
  #define SPINLOCK_PAUSE() __asm__ __volatile__("yield")
#else
  #define SPINLOCK_PAUSE() ((void)0)
#endif

namespace task_scheduler {

// Ch5.3.2: std::atomic_flag is always lock-free, ideal for spinlock.
class spinlock {
public:
    spinlock() noexcept : flag_(ATOMIC_FLAG_INIT) {}

    spinlock(const spinlock&) = delete;
    spinlock& operator=(const spinlock&) = delete;

    // TTAS lock with exponential backoff (Ch5.3.2 + Ch5.3.4 memory ordering).
    void lock() noexcept {
        // Ch5.3.4: memory_order_acquire for lock acquisition ensures
        // subsequent reads see writes from the thread that held the lock.
        for (int backoff = 1; !try_lock(); backoff = std::min(backoff * 2, 4096)) {
            // TTAS: spin on read first (cheap, stays in L1 cache in shared state)
            for (int i = 0; i < backoff; ++i) {
                // Ch5.3.1: memory_order_relaxed is sufficient for polling test.
                // No synchronization needed here - just testing the flag.
                if (!flag_.test(std::memory_order_relaxed)) {
                    break; // contention may have ended, try real lock now
                }
                SPINLOCK_PAUSE(); // Ch5.3.5: pause hint for hyperthreading
            }
        }
    }

    // Ch5.3.2: test_and_set returns previous value, sets to true.
    // If false was returned, we acquired the lock.
    bool try_lock() noexcept {
        // memory_order_acquire: pairs with unlock's memory_order_release.
        return !flag_.test_and_set(std::memory_order_acquire);
    }

    void unlock() noexcept {
        // Ch5.3.4: memory_order_release ensures all writes before unlock
        // are visible to the next thread that acquires the lock.
        flag_.clear(std::memory_order_release);
    }

private:
    std::atomic_flag flag_;
};

// RAII wrapper (Ch3.2.1: std::lock_guard pattern, extended for spinlock).
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
