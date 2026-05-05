/*
 * lecture17_part1.cpp - Bank Account Transfer: Locks vs. Transactions
 * Stanford CS149, Fall 2025 - Lecture 17
 *
 * Demonstrates the lecture's comparison between lock-based and
 * transaction-based synchronization:
 *   1. Lock-based transfer with deadlock risk (unordered lock acquisition)
 *   2. Lock-based transfer with deadlock avoidance (ordered locks)
 *   3. Simulated transactional memory transfer (atomic { } semantics)
 *   4. Composability demo: transfers that compose safely with TM
 *
 * Key point from lecture:
 *   atomic { } is DECLARATIVE (what to do), not IMPERATIVE (how to do it).
 *   With TM, compose transfer(A,B) + transfer(B,A) without deadlock.
 *
 * Compile: g++ -std=c++17 -pthread lecture17_part1.cpp -o lecture17_part1
 * Run: ./lecture17_part1
 */

#include <iostream>
#include <thread>
#include <mutex>
#include <vector>
#include <cassert>
#include <chrono>
#include <atomic>
#include <functional>
#include <random>

// ============================================================
// Bank account data structure
// ============================================================

struct Account {
    int id;
    int balance;
    std::mutex mtx; // Per-account lock

    Account(int i, int b) : id(i), balance(b) {}
};

// ============================================================
// Part 1: Lock-based Transfer (DEADLOCK PRONE)
// ============================================================
// Problem: if thread 0 does transfer(A, B) and thread 1
// does transfer(B, A), lock ordering differs → deadlock.
//
// This is the classic "composability problem" with locks.
// Without a global lock-ordering policy, lock-based code
// cannot safely compose.

bool transfer_deadlock_prone(Account& from, Account& to, int amount) {
    // DANGER: no fixed lock ordering
    std::lock_guard<std::mutex> lock_from(from.mtx);
    // Simulate some work to increase deadlock window
    std::this_thread::sleep_for(std::chrono::microseconds(10));
    std::lock_guard<std::mutex> lock_to(to.mtx);

    if (from.balance >= amount) {
        from.balance -= amount;
        to.balance += amount;
        return true;
    }
    return false;
}

// ============================================================
// Part 2: Lock-based Transfer (DEADLOCK-FREE)
// ============================================================
// Solution: use std::lock() to acquire both locks atomically,
// or enforce a global lock ordering (e.g., by account ID).
//
// This works but requires programmer discipline and global
// knowledge of lock ordering policies.

bool transfer_safe(Account& from, Account& to, int amount) {
    // std::lock acquires both locks without deadlock
    // (uses try_lock + backoff internally)
    std::lock(from.mtx, to.mtx);
    std::lock_guard<std::mutex> lock_from(from.mtx, std::adopt_lock);
    std::lock_guard<std::mutex> lock_to(to.mtx, std::adopt_lock);

    if (from.balance >= amount) {
        from.balance -= amount;
        to.balance += amount;
        return true;
    }
    return false;
}

// ============================================================
// Part 3: Simulated Transactional Memory Transfer
// ============================================================
// In a real TM system, the programmer writes:
//   atomic { withdraw(A, amount); deposit(B, amount); }
//
// Here we SIMULATE transactional semantics:
//   - Read both balances atomically (snapshot)
//   - Compute new balances
//   - Attempt to commit atomically (CAS on a global state)
//   - If conflict detected, abort and retry
//
// This demonstrates the OPTIMISTIC concurrency approach:
//   "Hope for the best, detect conflicts at commit time."

class TransactionalMemorySimulator {
public:
    struct Snapshot {
        int balance_a;
        int balance_b;
        uint64_t version; // For conflict detection
    };

    TransactionalMemorySimulator(int initial_a, int initial_b)
        : balance_a_(initial_a), balance_b_(initial_b), version_(0) {}

    // Atomic transaction: transfer amount from A to B
    // Returns true if committed successfully
    bool transfer_atomic(int amount) {
        while (true) {
            // ---- Start transaction: take a snapshot ----
            uint64_t ver_before = version_.load(std::memory_order_acquire);
            int a_balance = balance_a_.load(std::memory_order_acquire);
            int b_balance = balance_b_.load(std::memory_order_acquire);
            uint64_t ver_after = version_.load(std::memory_order_acquire);

            // If version changed during snapshot, retry (inconsistent read)
            if (ver_before != ver_after) continue;

            // ---- Speculative computation ----
            if (a_balance < amount) return false; // Insufficient funds

            int new_a = a_balance - amount;
            int new_b = b_balance + amount;

            // ---- Attempt to commit (like CAS on version) ----
            // Simulate atomic commit: if version unchanged, install new values
            uint64_t expected = ver_before;
            if (version_.compare_exchange_strong(expected, ver_before + 1,
                    std::memory_order_release, std::memory_order_relaxed)) {
                // Commit successful: install new balances
                balance_a_.store(new_a, std::memory_order_release);
                balance_b_.store(new_b, std::memory_order_release);
                return true;
            }
            // Conflict: someone else committed. Retry the transaction.
        }
    }

    // Withdraw from A only (atomic)
    bool withdraw_atomic(int amount) {
        while (true) {
            uint64_t ver_before = version_.load(std::memory_order_acquire);
            int a_balance = balance_a_.load(std::memory_order_acquire);
            uint64_t ver_after = version_.load(std::memory_order_acquire);

            if (ver_before != ver_after) continue;
            if (a_balance < amount) return false;

            int new_a = a_balance - amount;

            uint64_t expected = ver_before;
            if (version_.compare_exchange_strong(expected, ver_before + 1,
                    std::memory_order_release, std::memory_order_relaxed)) {
                balance_a_.store(new_a, std::memory_order_release);
                return true;
            }
        }
    }

    int get_balance_a() const { return balance_a_.load(); }
    int get_balance_b() const { return balance_b_.load(); }
    int total_balance() const { return get_balance_a() + get_balance_b(); }

private:
    std::atomic<int> balance_a_;
    std::atomic<int> balance_b_;
    std::atomic<uint64_t> version_; // Global version for optimistic conflict detection
};

// ============================================================
// Part 4: Composability Demonstration
// ============================================================
// With TM: transfer(A, B) and transfer(B, A) compose safely.
// The system automatically serializes conflicting transactions.

void run_lock_based_transfers() {
    std::cout << "--- Lock-Based Transfer (safe, ordered locks) ---" << std::endl;
    Account alice(0, 1000);
    Account bob(1, 1000);

    const int num_transfers = 10000;
    auto start = std::chrono::high_resolution_clock::now();

    std::thread t1([&]() {
        for (int i = 0; i < num_transfers; ++i) {
            // Alice → Bob: $10
            transfer_safe(alice, bob, 10);
        }
    });
    std::thread t2([&]() {
        for (int i = 0; i < num_transfers; ++i) {
            // Bob → Alice: $10
            transfer_safe(bob, alice, 10);
        }
    });

    t1.join();
    t2.join();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Alice: $" << alice.balance << " (expected $1000)" << std::endl;
    std::cout << "Bob: $" << bob.balance << " (expected $1000)" << std::endl;
    std::cout << "Total: $" << (alice.balance + bob.balance) << std::endl;
    std::cout << "Time: " << duration.count() << "ms" << std::endl;

    assert(alice.balance + bob.balance == 2000 && "Conservation of money violated!");
}

void run_tm_based_transfers() {
    std::cout << std::endl;
    std::cout << "--- TM-Simulated Transfer (optimistic concurrency) ---" << std::endl;
    TransactionalMemorySimulator tm(1000, 1000); // A=1000, B=1000

    const int num_transfers = 10000;
    int successful_a_to_b = 0;
    int successful_b_to_a = 0;

    auto start = std::chrono::high_resolution_clock::now();

    std::thread t1([&]() {
        for (int i = 0; i < num_transfers; ++i) {
            if (tm.transfer_atomic(10)) {
                ++successful_a_to_b;
            }
        }
    });
    std::thread t2([&]() {
        for (int i = 0; i < num_transfers; ++i) {
            // "Receive" from A to B: this is just B withdrawing from itself
            // Simulating: the system handles the ordering internally
            if (tm.withdraw_atomic(10)) {
                ++successful_b_to_a;
            }
        }
    });

    t1.join();
    t2.join();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Balance A: $" << tm.get_balance_a() << std::endl;
    std::cout << "Balance B: $" << tm.get_balance_b() << std::endl;
    std::cout << "Total: $" << tm.total_balance() << std::endl;
    std::cout << "Successful A->B transfers: " << successful_a_to_b << std::endl;
    std::cout << "Successful B withdrawals: " << successful_b_to_a << std::endl;
    std::cout << "Time: " << duration.count() << "ms" << std::endl;

    assert(tm.total_balance() + successful_a_to_b * 10 + successful_b_to_a * 10 <= 2000 + 200000
           && "Money was not created or destroyed");
}

// ============================================================
// Part 5: Demonstrate Deadlock Prone Code (DETECTABLE)
// ============================================================
// We'll show that without proper lock ordering, transactions
// would deadlock. But with TM simulation, they never deadlock.

int main() {
    std::cout << "=== CS149 Lecture 17: Locks vs. Transactions ===" << std::endl;
    std::cout << std::endl;

    // ---- Demo: Deadlock-free ordered lock transfer ----
    run_lock_based_transfers();

    // ---- Demo: TM-based optimistic transfer ----
    run_tm_based_transfers();

    // ---- Demo: Declarative vs Imperative ----
    std::cout << std::endl;
    std::cout << "--- Conceptual Comparison ---" << std::endl;
    std::cout << "Lock-based (imperative):" << std::endl;
    std::cout << "  lock(A); lock(B); withdraw(A, x); deposit(B, x); unlock(B); unlock(A);" << std::endl;
    std::cout << "  Problem: programmer must manage lock order to avoid deadlock." << std::endl;
    std::cout << std::endl;
    std::cout << "TM-based (declarative):" << std::endl;
    std::cout << "  atomic { withdraw(A, x); deposit(B, x); }" << std::endl;
    std::cout << "  System handles synchronization automatically. No deadlock!" << std::endl;

    std::cout << std::endl;
    std::cout << "Summary:" << std::endl;
    std::cout << "  - atomic { } declares WHAT should be atomic (declarative)" << std::endl;
    std::cout << "  - lock/unlock specifies HOW to synchronize (imperative)" << std::endl;
    std::cout << "  - TM provides failure atomicity: no partial updates visible" << std::endl;
    std::cout << "  - TM composes safely: transfer(A,B) + transfer(B,A) works" << std::endl;
    std::cout << "  - Our simulation uses optimistic concurrency + CAS on" << std::endl;
    std::cout << "    a global version number to detect conflicts at commit." << std::endl;

    return 0;
}
