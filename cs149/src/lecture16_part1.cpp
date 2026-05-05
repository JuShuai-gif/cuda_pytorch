/*
 * lecture16_part1.cpp - Lock Implementations
 * Stanford CS149, Fall 2025 - Lecture 16
 *
 * Demonstrates various lock implementations:
 *   1. Test-and-set lock (simple, high coherence traffic)
 *   2. Test-and-test-and-set lock (lower traffic via read-spinning)
 *   3. Ticket lock (FIFO fairness, minimal traffic)
 *   4. CAS-based lock (using compare_exchange_strong)
 *   5. Atomic fetch-and-op built from CAS
 *
 * Compile: g++ -std=c++17 -pthread lecture16_part1.cpp -o lecture16_part1
 * Run: ./lecture16_part1
 */

#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <vector>
#include <chrono>
#include <cassert>

// ============================================================
// Part 1: Test-and-Set Lock
// ============================================================
// Simple spinlock using atomic_flag (hardware test-and-set equivalent).
// Characteristics: low latency under low contention, high interconnect
// traffic under contention (O(P^2) invalidations).

class TASLock {
public:
    void lock() {
        // test_and_set sets the flag to true and returns the old value.
        // If old value was false, we acquired the lock.
        while (flag.test_and_set(std::memory_order_acquire)) {
            // Spin-wait: constantly retrying TAS generates high bus traffic.
        }
    }

    void unlock() {
        flag.clear(std::memory_order_release);
    }

private:
    std::atomic_flag flag = ATOMIC_FLAG_INIT;
};

// ============================================================
// Part 2: Test-and-Test-and-Set Lock
// ============================================================
// Optimized version: first spins reading (low-cost cache hit),
// then attempts test-and-set only when the lock appears free.
// Generates much less interconnect traffic.

class TTASLock {
public:
    void lock() {
        while (true) {
            // Phase 1: Spin on read (stays in Shared state, cache hit)
            while (flag.load(std::memory_order_relaxed)) {
                // Busy-wait without generating bus traffic.
            }
            // Phase 2: Attempt to acquire when lock appears free
            if (!flag.exchange(true, std::memory_order_acquire)) {
                return; // Lock acquired!
            }
            // If exchange fails (another thread grabbed it), retry.
        }
    }

    void unlock() {
        flag.store(false, std::memory_order_release);
    }

private:
    std::atomic<bool> flag{false};
};

// ============================================================
// Part 3: Ticket Lock
// ============================================================
// Provides FIFO fairness. Each thread takes a ticket number.
// A thread can enter the critical section when its ticket
// matches now_serving. Only one invalidation per lock release.
// This is O(P) interconnect traffic vs O(P^2) for TAS.

class TicketLock {
public:
    void lock() {
        // Atomically get a ticket and increment next_ticket
        unsigned int my_ticket = next_ticket.fetch_add(1, std::memory_order_relaxed);
        // Spin until our ticket is called
        while (now_serving.load(std::memory_order_acquire) != my_ticket) {
            // Only reading now_serving - low coherence traffic
        }
    }

    void unlock() {
        // Signal the next waiting thread
        now_serving.fetch_add(1, std::memory_order_release);
    }

private:
    std::atomic<unsigned int> next_ticket{0};
    std::atomic<unsigned int> now_serving{0};
};

// ============================================================
// Part 4: CAS-based Lock (Optimized for Contention)
// ============================================================
// Like TTAS but using compare_exchange_strong instead of exchange.
// First spins reading, then tries CAS. Under contention, the
// shared-read phase significantly reduces bus traffic.
// The CAS version is potentially more efficient than exchange
// because hardware can optimize the compare path.

class CASLock {
public:
    void lock() {
        while (true) {
            // Phase 1: Spin reading (cache-friendly)
            while (locked.load(std::memory_order_relaxed)) {
            }
            // Phase 2: Attempt CAS acquisition
            bool expected = false;
            if (locked.compare_exchange_strong(expected, true,
                    std::memory_order_acquire, std::memory_order_relaxed)) {
                return;
            }
        }
    }

    void unlock() {
        locked.store(false, std::memory_order_release);
    }

private:
    std::atomic<bool> locked{false};
};

// ============================================================
// Part 5: Atomic fetch-and-op built from CAS
// ============================================================
// Lecture shows how to build atomic_min using CAS.
// Pattern: read old value, compute desired new value,
// CAS to install, retry if CAS fails (someone else modified).

// Atomic minimum: atomically sets *addr = min(*addr, x)
void atomic_min(std::atomic<int>& addr, int x) {
    int old_val = addr.load(std::memory_order_relaxed);
    int new_val = std::min(old_val, x);
    // Keep retrying if another thread changed the value
    while (!addr.compare_exchange_weak(old_val, new_val,
            std::memory_order_release, std::memory_order_relaxed)) {
        // CAS failed: old_val now contains the current value.
        // No need to reload - compare_exchange_weak updates old_val on failure.
        new_val = std::min(old_val, x);
    }
}

// Atomic increment using CAS
int atomic_fetch_add_cas(std::atomic<int>& addr, int x) {
    int old_val = addr.load(std::memory_order_relaxed);
    while (!addr.compare_exchange_weak(old_val, old_val + x,
            std::memory_order_release, std::memory_order_relaxed)) {
        // old_val is updated by CAS on failure, so just loop
    }
    return old_val; // Return the value before addition
}

// ============================================================
// Demo: Counter Increment Under Contention
// ============================================================
// Multiple threads increment a shared counter using different
// lock types to demonstrate correctness and performance.

template <typename LockType>
void increment_counter(LockType& lock, int& counter, int iterations) {
    for (int i = 0; i < iterations; ++i) {
        std::lock_guard<LockType> guard(lock);
        ++counter;
    }
}

// TicketLock supports std::lock_guard via custom lock/unlock
// Already compatible since we followed the Lockable concept.

template <typename LockType>
void run_benchmark(const std::string& name, int num_threads, int per_thread) {
    LockType lock;
    int counter = 0;
    std::vector<std::thread> threads;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(increment_counter<LockType>,
                             std::ref(lock), std::ref(counter), per_thread);
    }
    for (auto& t : threads) {
        t.join();
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "[" << name << "] Threads=" << num_threads
              << " Total ops=" << (num_threads * per_thread)
              << " Counter=" << counter
              << " Time=" << duration.count() << "ms" << std::endl;

    // Verify correctness: counter should equal total operations
    assert(counter == num_threads * per_thread);
}

int main() {
    std::cout << "=== CS149 Lecture 16: Lock Implementations ===" << std::endl;
    std::cout << std::endl;

    // ---- Part 5 Demo: atomic fetch-and-op built from CAS ----
    std::cout << "--- Atomic fetch-and-op from CAS ---" << std::endl;
    {
        std::atomic<int> val{42};
        atomic_min(val, 10);
        std::cout << "atomic_min(42, 10) = " << val.load() << " (expected 10)" << std::endl;

        val.store(5);
        atomic_min(val, 10);
        std::cout << "atomic_min(5, 10) = " << val.load() << " (expected 5)" << std::endl;

        val.store(0);
        int old = atomic_fetch_add_cas(val, 5);
        std::cout << "atomic_fetch_add_cas(0, 5): returned " << old
                  << ", new value = " << val.load() << " (expected 0, 5)" << std::endl;
    }

    std::cout << std::endl;
    std::cout << "--- Lock Benchmarks (correctness + timing) ---" << std::endl;

    const int num_threads = 4;
    const int per_thread = 100000;

    run_benchmark<TASLock>("TAS Lock  ", num_threads, per_thread);
    run_benchmark<TTASLock>("TTAS Lock  ", num_threads, per_thread);
    run_benchmark<TicketLock>("Ticket Lock", num_threads, per_thread);
    run_benchmark<CASLock>("CAS Lock   ", num_threads, per_thread);

    std::cout << std::endl;
    std::cout << "All lock implementations verified correct." << std::endl;
    std::cout << "Note: TAS may be slowest under contention (high bus traffic)." << std::endl;
    std::cout << "TTAS and Ticket lock should be fastest (low bus traffic)." << std::endl;

    return 0;
}
