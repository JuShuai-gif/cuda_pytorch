// lecture15_part3.cpp — CS149 Lecture 15: Memory Fences and Data Races
// Demonstrates fence instructions, data race detection concepts,
// happens-before analysis, and the x86 fence family.
// Compile: g++ -std=c++17 -O2 -pthread lecture15_part3.cpp -o lecture15_part3
// Run:     ./lecture15_part3

#include <iostream>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cassert>

// ============================================================================
// Part 1: Memory Fence (Barrier) Explanation
// ============================================================================

void explain_fences() {
    std::cout << "=== CS149 Lecture 15: Memory Fences and Data Races ===" << std::endl;
    std::cout << std::endl;
    std::cout << "Memory fences (barriers) prevent reordering of memory operations." << std::endl;
    std::cout << "They are the programmer's escape hatch when the consistency" << std::endl;
    std::cout << "model is too relaxed." << std::endl;
    std::cout << std::endl;
    std::cout << "x86 fence instructions:" << std::endl;
    std::cout << "  mfence: all prior loads+stores complete before any subsequent" << std::endl;
    std::cout << "          load or store begins.  (full barrier)" << std::endl;
    std::cout << "  lfence: all prior loads complete before any subsequent load." << std::endl;
    std::cout << "  sfence: all prior stores complete before any subsequent store." << std::endl;
    std::cout << std::endl;
    std::cout << "C++11 equivalent:" << std::endl;
    std::cout << "  std::atomic_thread_fence(std::memory_order_seq_cst);  // mfence" << std::endl;
    std::cout << "  std::atomic_thread_fence(std::memory_order_acquire);  // load fence" << std::endl;
    std::cout << "  std::atomic_thread_fence(std::memory_order_release);  // store fence" << std::endl;
}

// ============================================================================
// Part 2: Using std::atomic_thread_fence to Fix Dekker's Pattern
// ============================================================================
// Without fences, relaxed atomics allow reordering → (r1=0, r2=0) possible.
// Adding seq_cst fences between the store and load PREVENTS this reordering.

void demo_fence_fixing_dekker() {
    std::cout << std::endl;
    std::cout << "=== Fence Fix for Dekker's Pattern ===" << std::endl;
    std::cout << std::endl;

    std::atomic<int> A{0}, B{0};
    int zero_zero = 0;
    const int N = 500'000;

    for (int trial = 0; trial < N; ++trial) {
        A.store(0, std::memory_order_relaxed);
        B.store(0, std::memory_order_relaxed);

        int r1 = -1, r2 = -1;
        std::thread t0([&]() {
            A.store(1, std::memory_order_relaxed);    // (1) store to A
            // Fence: all prior stores complete before subsequent loads
            std::atomic_thread_fence(std::memory_order_seq_cst);
            r1 = B.load(std::memory_order_relaxed);    // (2) load from B
        });
        std::thread t1([&]() {
            B.store(1, std::memory_order_relaxed);    // (3) store to B
            std::atomic_thread_fence(std::memory_order_seq_cst);
            r2 = A.load(std::memory_order_relaxed);    // (4) load from A
        });
        t0.join(); t1.join();

        if (r1 == 0 && r2 == 0) zero_zero++;
    }

    std::cout << "Dekker with relaxed atomics + seq_cst fence between store and load:" << std::endl;
    std::cout << "  Trials: " << N << std::endl;
    std::cout << "  r1=0 && r2=0: " << zero_zero;
    if (zero_zero == 0)
        std::cout << " — fence prevents reordering! SC-like behavior." << std::endl;
    else
        std::cout << " — reordering still occurred." << std::endl;
}

// ============================================================================
// Part 3: Happens-Before Analysis
// ============================================================================
// The happens-before relation is the fundamental tool for reasoning about
// concurrent program outcomes. If a cycle exists in the happens-before graph,
// the outcome is impossible.
//
// Happens-before edges:
//   1. Program order (po): operations in the same thread happen in order
//   2. Synchronizes-with (sw): release store → acquire load, lock release → lock acquire
//   3. Transitive closure: if A hb B and B hb C, then A hb C

void explain_happens_before() {
    std::cout << std::endl;
    std::cout << "=== Happens-Before Analysis ===" << std::endl;
    std::cout << std::endl;
    std::cout << "Dekker's pattern: r1=0, r2=0 — why impossible under SC?" << std::endl;
    std::cout << std::endl;
    std::cout << "  Thread 0:      Thread 1:" << std::endl;
    std::cout << "    (1) A = 1      (3) B = 1" << std::endl;
    std::cout << "    (2) r1 = B     (4) r2 = A" << std::endl;
    std::cout << std::endl;
    std::cout << "To get r1=0, r2=0, we need:" << std::endl;
    std::cout << "  (2) reads 0 → (2) happens before (3) in the coherence order" << std::endl;
    std::cout << "  (4) reads 0 → (4) happens before (1) in the coherence order" << std::endl;
    std::cout << std::endl;
    std::cout << "Happens-before edges:" << std::endl;
    std::cout << "  program order: (1) → (2) and (3) → (4)" << std::endl;
    std::cout << "  coherence:     (2) → (3) (r1=0, so read of B before write of B)" << std::endl;
    std::cout << "  coherence:     (4) → (1) (r2=0, so read of A before write of A)" << std::endl;
    std::cout << std::endl;
    std::cout << "Cycle: (1) → (2) → (3) → (4) → (1)  ← IMPOSSIBLE!" << std::endl;
    std::cout << "Therefore (r1=0, r2=0) cannot happen under SC." << std::endl;
}

// ============================================================================
// Part 4: Data Race Demonstration
// ============================================================================
// A data race: two threads access the same memory location concurrently,
// at least one is a write, and the accesses are not ordered by synchronization.
// Programs with data races have UNDEFINED BEHAVIOR in C++11.

void demonstrate_data_race() {
    std::cout << std::endl;
    std::cout << "=== Data Races ===" << std::endl;
    std::cout << std::endl;

    // ---- Unsafe: data race on plain int ----
    int shared_counter = 0;
    const int N_INC = 1'000'000;
    std::atomic<bool> done{false};

    std::thread t0([&]() {
        for (int i = 0; i < N_INC; ++i)
            shared_counter++;  // DATA RACE: non-atomic write + concurrent writes
        done.store(true, std::memory_order_release);
    });
    std::thread t1([&]() {
        for (int i = 0; i < N_INC; ++i)
            shared_counter++;  // DATA RACE
    });
    t0.join(); t1.join();

    std::cout << "Data race example (unsynchronized plain int):" << std::endl;
    std::cout << "  Expected: " << (2 * N_INC) << std::endl;
    std::cout << "  Got:      " << shared_counter << std::endl;
    std::cout << "  Lost updates due to data race! (Undefined Behavior in C++)" << std::endl;

    // ---- Safe: atomic with relaxed (no race, correct count) ----
    std::atomic<int> safe_counter{0};
    std::thread t2([&]() {
        for (int i = 0; i < N_INC; ++i)
            safe_counter.fetch_add(1, std::memory_order_relaxed);
    });
    std::thread t3([&]() {
        for (int i = 0; i < N_INC; ++i)
            safe_counter.fetch_add(1, std::memory_order_relaxed);
    });
    t2.join(); t3.join();

    std::cout << std::endl;
    std::cout << "Race-free example (atomic<int> with relaxed):" << std::endl;
    std::cout << "  Expected: " << (2 * N_INC) << std::endl;
    std::cout << "  Got:      " << safe_counter.load() << std::endl;
    std::cout << "  Correct! Atomic operations are indivisible — no lost updates." << std::endl;
}

// ============================================================================
// Part 5: Conflicting Accesses Classification
// ============================================================================
void classify_conflicts() {
    std::cout << std::endl;
    std::cout << "=== Conflicting Access Classification ===" << std::endl;
    std::cout << std::endl;
    std::cout << "Two memory accesses by different threads conflict if:" << std::endl;
    std::cout << "  1. They access the same memory location" << std::endl;
    std::cout << "  2. At least one is a write" << std::endl;
    std::cout << std::endl;
    std::cout << "Synchronized program:" << std::endl;
    std::cout << "  Conflicting accesses ordered by synchronization (fence," << std::endl;
    std::cout << "  release/acquire, lock, barrier) → NO data race." << std::endl;
    std::cout << std::endl;
    std::cout << "Unsynchronized program:" << std::endl;
    std::cout << "  Conflicting accesses NOT ordered → DATA RACE." << std::endl;
    std::cout << "  Output depends on relative thread speeds (non-deterministic)." << std::endl;
    std::cout << std::endl;
    std::cout << "In practice:" << std::endl;
    std::cout << "  Most programs use synchronization libraries (locks, barriers)." << std::endl;
    std::cout << "  Ad-hoc shared variable access without synchronization → bug!" << std::endl;
}

// ============================================================================
// Part 6: The Lock as a Fence
// ============================================================================
// Locks (mutex) implicitly include acquire and release fences:
//   lock()   → acquire fence (all subsequent ops stay after lock acquire)
//   unlock() → release fence (all prior ops complete before unlock)
// This is why correctly locked programs appear sequentially consistent.

void demo_lock_as_fence() {
    std::cout << std::endl;
    std::cout << "=== Lock as Implicit Fence ===" << std::endl;
    std::cout << std::endl;

    int shared_data = 0;
    std::mutex mtx;
    const int N = 100'000;

    auto worker = [&](int id) {
        for (int i = 0; i < N; ++i) {
            std::lock_guard<std::mutex> lk(mtx);
            // lock() has acquire semantics — data loaded here is fresh
            shared_data++;
            // unlock() has release semantics — all writes visible to next locker
        }
    };

    std::thread t0(worker, 0);
    std::thread t1(worker, 1);
    std::thread t2(worker, 2);
    std::thread t3(worker, 3);

    t0.join(); t1.join(); t2.join(); t3.join();

    std::cout << "4 threads, " << N << " increments each." << std::endl;
    std::cout << "Expected: " << (4 * N) << std::endl;
    std::cout << "Got:      " << shared_data << std::endl;
    std::cout << "Correct! The mutex provides acquire/release fences implicitly." << std::endl;
    std::cout << "This is why DRF (Data-Race-Free) programs are SC." << std::endl;
}

// ============================================================================
// Part 7: Litmus Test Summary
// ============================================================================
void litmus_test_summary() {
    std::cout << std::endl;
    std::cout << "=== Common Consistency Litmus Tests ===" << std::endl;
    std::cout << std::endl;

    std::cout << "1. Store Buffering (Dekker / SB):" << std::endl;
    std::cout << "   P0: X=1; r1=Y   P1: Y=1; r2=X" << std::endl;
    std::cout << "   SC: (r1,r2) ≠ (0,0)   TSO/x86: (0,0) allowed" << std::endl;
    std::cout << std::endl;

    std::cout << "2. Message Passing (MP):" << std::endl;
    std::cout << "   P0: X=1; Y=1   P1: r1=Y; r2=X" << std::endl;
    std::cout << "   SC/TSO: r1=1 ⇒ r2=1   Relaxed: r1=1, r2=0 possible" << std::endl;
    std::cout << "   Fix: release store for Y, acquire load for Y" << std::endl;
    std::cout << std::endl;

    std::cout << "3. Store Buffer / IRIW (Independent Reads Independent Writes):" << std::endl;
    std::cout << "   P0: X=1   P1: Y=1   P2: r1=X; r2=Y   P3: r3=Y; r4=X" << std::endl;
    std::cout << "   SC: P2 and P3 must agree on order of writes" << std::endl;
    std::cout << "   TSO: P2 and P3 may disagree (x86 actually forbids this!)" << std::endl;
    std::cout << std::endl;

    std::cout << "4. Coherence (same address):" << std::endl;
    std::cout << "   P0: X=1   P1: X=2   P2: r1=X; r2=X" << std::endl;
    std::cout << "   All models: r1=1,r2=2 or r1=2,r2=2 or r1=2,r2=1 allowed" << std::endl;
    std::cout << "   But r1=2,r2=1 after the values were written → coherence violation!" << std::endl;
}

// ============================================================================
// main
// ============================================================================
int main() {
    explain_fences();
    demo_fence_fixing_dekker();
    explain_happens_before();
    demonstrate_data_race();
    classify_conflicts();
    demo_lock_as_fence();
    litmus_test_summary();

    std::cout << std::endl;
    std::cout << "=== Summary ===" << std::endl;
    std::cout << "1. Memory fences prevent reordering at specific points." << std::endl;
    std::cout << "2. Happens-before analysis: cycles = impossible outcomes." << std::endl;
    std::cout << "3. Data races → undefined behavior in C++11. Use atomics/locks." << std::endl;
    std::cout << "4. DRF programs are SC — synchronization libraries handle ordering." << std::endl;
    std::cout << "5. C++11: 'SC for DRF' is the contract between language and programmer." << std::endl;

    return 0;
}
