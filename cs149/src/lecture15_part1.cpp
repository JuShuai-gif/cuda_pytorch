// lecture15_part1.cpp — CS149 Lecture 15: SC vs TSO vs PSO Memory Consistency
// Demonstrates Dekker's pattern under sequential consistency and relaxed models.
// Compile: g++ -std=c++17 -O2 -pthread lecture15_part1.cpp -o lecture15_part1
// Run:     ./lecture15_part1

#include <iostream>
#include <thread>
#include <atomic>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cassert>

// ============================================================================
// Part 1: Conceptual Explanation of Memory Consistency Models
// ============================================================================

void explain_consistency_models() {
    std::cout << "=== CS149 Lecture 15: Memory Consistency Models ===" << std::endl;
    std::cout << std::endl;
    std::cout << "Coherence (per-address):" << std::endl;
    std::cout << "  All processors agree on order of reads/writes to SAME address." << std::endl;
    std::cout << std::endl;
    std::cout << "Consistency (cross-address):" << std::endl;
    std::cout << "  When do writes to X become visible relative to reads/writes to Y?" << std::endl;
    std::cout << std::endl;
    std::cout << "Four memory operation orderings:" << std::endl;
    std::cout << "  W→R: write must complete before subsequent read" << std::endl;
    std::cout << "  R→R: read must complete before subsequent read" << std::endl;
    std::cout << "  R→W: read must complete before subsequent write" << std::endl;
    std::cout << "  W→W: write must complete before subsequent write" << std::endl;
    std::cout << std::endl;
    std::cout << "Consistency models by which orderings they relax:" << std::endl;
    std::cout << std::endl;
    std::cout << std::left
              << std::setw(18) << "Model"
              << std::setw(12) << "W→R"
              << std::setw(12) << "R→R"
              << std::setw(12) << "R→W"
              << std::setw(12) << "W→W"
              << "Example" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << std::left
              << std::setw(18) << "SC (Sequential)"
              << std::setw(12) << "✓"
              << std::setw(12) << "✓"
              << std::setw(12) << "✓"
              << std::setw(12) << "✓"
              << "Idealized; slow" << std::endl;
    std::cout << std::left
              << std::setw(18) << "TSO (Total Store)"
              << std::setw(12) << "✗"
              << std::setw(12) << "✓"
              << std::setw(12) << "✓"
              << std::setw(12) << "✓"
              << "x86, SPARC" << std::endl;
    std::cout << std::left
              << std::setw(18) << "PSO (Partial)"
              << std::setw(12) << "✗"
              << std::setw(12) << "✓"
              << std::setw(12) << "✓"
              << std::setw(12) << "✗"
              << "SPARC PSO" << std::endl;
    std::cout << std::left
              << std::setw(18) << "WO/RC (Weak)"
              << std::setw(12) << "✗"
              << std::setw(12) << "✗"
              << std::setw(12) << "✗"
              << std::setw(12) << "✗"
              << "ARM, PowerPC" << std::endl;
}

// ============================================================================
// Part 2: Dekker's Pattern — the Classic Consistency Litmus Test
// ============================================================================
// Initially: A = 0, B = 0
//
// Thread 0:              Thread 1:
//   A = 1;                 B = 1;
//   r1 = B;                r2 = A;
//
// Possible outcomes:
//   SC:     (r1,r2) = (0,1) or (1,0) or (1,1) — NEVER (0,0)
//   TSO/x86:(r1,r2) can be (0,0) — reads move ahead of buffered writes
//
// Proof that (0,0) is impossible under SC:
//   If r1=0, then Thread 0's read of B happened before Thread 1's write to B.
//   By program order, Thread 0's write to A happened before its read of B.
//   So Thread 0's write to A happened before Thread 1's write to B.
//   Thread 1's read of A must happen after its write to B (program order in Thread 1).
//   But if Thread 0's write to A happened before Thread 1's read of A starts,
//   then r2 must be 1 — contradiction with r2=0. Therefore (0,0) impossible.
//   (The happens-before graph has a cycle → impossible.)

void dekker_sc(int num_trials) {
    std::atomic<int> A{0}, B{0};
    int sc_violations = 0;  // should be 0 under SC (seq_cst atomics)

    for (int trial = 0; trial < num_trials; ++trial) {
        A.store(0, std::memory_order_seq_cst);
        B.store(0, std::memory_order_seq_cst);

        int r1 = -1, r2 = -1;
        std::thread t0([&]() {
            A.store(1, std::memory_order_seq_cst);
            r1 = B.load(std::memory_order_seq_cst);
        });
        std::thread t1([&]() {
            B.store(1, std::memory_order_seq_cst);
            r2 = A.load(std::memory_order_seq_cst);
        });
        t0.join(); t1.join();

        if (r1 == 0 && r2 == 0)
            sc_violations++;
    }

    std::cout << "Dekker's pattern with SC (seq_cst atomics):" << std::endl;
    std::cout << "  Trials: " << num_trials << std::endl;
    std::cout << "  r1=0 && r2=0 (SC violation): " << sc_violations;
    if (sc_violations == 0)
        std::cout << " — SC holds (never saw 0,0)" << std::endl;
    else
        std::cout << " — UNEXPECTED!" << std::endl;
}

void dekker_relaxed(int num_trials) {
    std::atomic<int> A{0}, B{0};
    int zero_zero = 0;

    for (int trial = 0; trial < num_trials; ++trial) {
        A.store(0, std::memory_order_relaxed);
        B.store(0, std::memory_order_relaxed);

        int r1 = -1, r2 = -1;
        std::thread t0([&]() {
            A.store(1, std::memory_order_relaxed);
            r1 = B.load(std::memory_order_relaxed);
        });
        std::thread t1([&]() {
            B.store(1, std::memory_order_relaxed);
            r2 = A.load(std::memory_order_relaxed);
        });
        t0.join(); t1.join();

        if (r1 == 0 && r2 == 0)
            zero_zero++;
    }

    std::cout << "Dekker's pattern with RELAXED atomics:" << std::endl;
    std::cout << "  Trials: " << num_trials << std::endl;
    std::cout << "  r1=0 && r2=0 (TSO/relaxed allows): " << zero_zero
              << " (" << std::fixed << std::setprecision(1)
              << (100.0 * zero_zero / num_trials) << "%)" << std::endl;
    std::cout << "  This outcome is IMPOSSIBLE under SC, but occurs under TSO/relaxed!" << std::endl;
}

// ============================================================================
// Part 3: Write Buffer Simulation (Conceptual)
// ============================================================================
// Under TSO, each processor has a write buffer:
//   Writes go to the write buffer (fast), reads check the buffer first.
//   Writes drain to cache/memory asynchronously.
// This means: a processor's own reads can move ahead of its pending writes.

void explain_write_buffer() {
    std::cout << std::endl;
    std::cout << "=== Write Buffer (TSO Motivation) ===" << std::endl;
    std::cout << std::endl;
    std::cout << "Problem with SC: a write may take 100s of cycles (cache miss," << std::endl;
    std::cout << "coherence traffic). Under SC, the processor must STALL until" << std::endl;
    std::cout << "the write completes before issuing the next instruction." << std::endl;
    std::cout << std::endl;
    std::cout << "Write buffer solution:" << std::endl;
    std::cout << "  Store A=1 → write buffer (fast, ~1 cycle)" << std::endl;
    std::cout << "  Load B    → read from cache (doesn't wait for A's write)" << std::endl;
    std::cout << "  Write buffer drains to cache/memory in background" << std::endl;
    std::cout << std::endl;
    std::cout << "This is why r1=r2=0 is possible: each processor reads the" << std::endl;
    std::cout << "other's variable before the other's write leaves the buffer." << std::endl;
    std::cout << std::endl;
    std::cout << "Every modern processor uses write buffers!" << std::endl;
    std::cout << "  x86: TSO-like (incompletely specified)" << std::endl;
    std::cout << "  ARM: very relaxed (weaker than TSO)" << std::endl;
}

// ============================================================================
// Part 4: Store Buffer — Classic Dekker with plain variables (x86 behavior)
// ============================================================================
// On x86, even plain (non-atomic) variables may exhibit TSO behavior:
// the compiler can reorder, and the hardware definitely reorders W→R.

void dekker_plain_vars(int num_trials) {
    // volatile prevents compiler reordering but NOT hardware reordering
    volatile int A = 0, B = 0;
    int zero_zero = 0;

    for (int trial = 0; trial < num_trials; ++trial) {
        A = 0; B = 0;

        int r1 = -1, r2 = -1;
        std::thread t0([&]() {
            A = 1;
            r1 = B;
        });
        std::thread t1([&]() {
            B = 1;
            r2 = A;
        });
        t0.join(); t1.join();

        if (r1 == 0 && r2 == 0)
            zero_zero++;
    }

    std::cout << "Dekker's pattern with plain volatile int (x86 TSO):" << std::endl;
    std::cout << "  Trials: " << num_trials << std::endl;
    std::cout << "  r1=0 && r2=0: " << zero_zero
              << " (" << std::fixed << std::setprecision(1)
              << (100.0 * zero_zero / num_trials) << "%)" << std::endl;
    std::cout << "  (0,0) is possible on x86 due to store buffer / TSO!" << std::endl;
}

// ============================================================================
// Part 5: Store Forwarding (TSO detail)
// ============================================================================
// Under TSO, a processor can read its own writes from the store buffer
// before they are globally visible. This is called "store-to-load forwarding."
//
// Example:
//   Thread 0:              Thread 1:
//     X = 1;     (in sb)     r1 = X;   (might see old value 0)
//     r2 = X;    (reads from sb → 1)    r2 = Y;
//
// Thread 0 sees its OWN write immediately (store forwarding),
// but Thread 1 might not see it yet (write still in buffer).

void demo_store_forwarding() {
    std::cout << std::endl;
    std::cout << "=== Store-to-Load Forwarding (TSO) ===" << std::endl;
    std::cout << std::endl;

    std::atomic<int> X{0}, Y{0};
    int seen_stale = 0;
    int trials = 100000;

    for (int t = 0; t < trials; ++t) {
        X.store(0, std::memory_order_relaxed);
        Y.store(0, std::memory_order_relaxed);

        int r1 = -1, r2 = -1, r3 = -1;

        std::thread t0([&]() {
            X.store(1, std::memory_order_relaxed);  // goes to store buffer
            r1 = X.load(std::memory_order_relaxed);  // store forwarding → 1
            r2 = Y.load(std::memory_order_relaxed);
        });
        std::thread t1([&]() {
            Y.store(1, std::memory_order_relaxed);
            r3 = X.load(std::memory_order_relaxed);  // might see 0 (buffered in t0)
        });
        t0.join(); t1.join();

        if (r3 == 0 && r1 == 1)
            seen_stale++;
    }

    std::cout << "Thread 0 writes X=1, then reads X (gets 1 via store forwarding)." << std::endl;
    std::cout << "Thread 1 reads X — might see stale 0 (still in T0's buffer)." << std::endl;
    std::cout << "  Trials: " << trials << std::endl;
    std::cout << "  T1 saw stale X=0: " << seen_stale
              << " (" << std::fixed << std::setprecision(1)
              << (100.0 * seen_stale / trials) << "%)" << std::endl;
}

// ============================================================================
// Part 6: PSO Example — flag before data
// ============================================================================
// Under PSO (Partial Store Order), writes can be reordered W→W.
// This means a flag variable might become visible BEFORE the data it guards.
//
//   Thread 0 (P0):         Thread 1 (P1):
//     A = 1;   (data)        while (flag == 0);
//     flag = 1; (flag)        print A;  ← might print 0!
//
// Under PSO, the write to 'flag' could drain from the write buffer
// before the write to 'A' — P1 sees flag=1 but A still 0.

void demo_pso_flag_data() {
    std::cout << std::endl;
    std::cout << "=== PSO: Flag-Before-Data Hazard ===" << std::endl;
    std::cout << std::endl;

    std::atomic<int> A{0}, flag{0};
    int stale_reads = 0;
    int trials = 100000;

    for (int t = 0; t < trials; ++t) {
        A.store(0, std::memory_order_relaxed);
        flag.store(0, std::memory_order_relaxed);

        std::thread t0([&]() {
            // Under PSO, these writes could be reordered!
            A.store(1, std::memory_order_relaxed);       // data
            flag.store(1, std::memory_order_relaxed);     // flag
        });
        std::thread t1([&]() {
            while (flag.load(std::memory_order_relaxed) == 0)
                ;  // spin
            if (A.load(std::memory_order_relaxed) == 0)
                stale_reads++;  // saw flag=1 but data still 0!
        });
        t0.join(); t1.join();
    }

    std::cout << "Thread 0: writes A=1, then flag=1." << std::endl;
    std::cout << "Thread 1: spins on flag, then reads A." << std::endl;
    std::cout << "  Trials: " << trials << std::endl;
    std::cout << "  P1 saw flag=1 but A=0: " << stale_reads
              << " (" << std::fixed << std::setprecision(1)
              << (100.0 * stale_reads / trials) << "%)" << std::endl;
    std::cout << "  Fix: use release store for flag, acquire load for flag." << std::endl;
}

// ============================================================================
// Part 7: Synchronized = Data-Race-Free = SC
// ============================================================================
void demo_synchronized_is_sc() {
    std::cout << std::endl;
    std::cout << "=== SC for DRF (Data-Race-Free) ===" << std::endl;
    std::cout << std::endl;
    std::cout << "C++11 / Java memory model guarantee:" << std::endl;
    std::cout << "  If your program is data-race-free (all shared accesses" << std::endl;
    std::cout << "  are ordered by synchronization), then it behaves as if" << std::endl;
    std::cout << "  executed under Sequential Consistency." << std::endl;
    std::cout << std::endl;
    std::cout << "This means library code (locks, barriers, atomics)" << std::endl;
    std::cout << "handles all the messy memory ordering for you!" << std::endl;
    std::cout << std::endl;
    std::cout << "Data race definition:" << std::endl;
    std::cout << "  Two accesses to the same memory location," << std::endl;
    std::cout << "  at least one is a write," << std::endl;
    std::cout << "  not ordered by synchronization (fence, lock, etc.)." << std::endl;
}

// ============================================================================
// main
// ============================================================================
int main() {
    explain_consistency_models();
    std::cout << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << std::endl;

    // Dekker's pattern experiments
    int n = 1000000;
    dekker_sc(n);
    std::cout << std::endl;
    dekker_relaxed(n);
    std::cout << std::endl;

    explain_write_buffer();
    std::cout << std::endl;

    dekker_plain_vars(n / 100);   // fewer trials for volatile (slower)
    std::cout << std::endl;

    demo_store_forwarding();
    std::cout << std::endl;

    demo_pso_flag_data();
    std::cout << std::endl;

    demo_synchronized_is_sc();

    std::cout << std::endl;
    std::cout << "=== Summary ===" << std::endl;
    std::cout << "1. SC: all 4 orderings maintained — intuitive but slow." << std::endl;
    std::cout << "2. TSO: relaxes W→R (write buffer) — x86 model." << std::endl;
    std::cout << "3. PSO: also relaxes W→W — flag-before-data hazard." << std::endl;
    std::cout << "4. DRF programs get SC for free — use synchronization!" << std::endl;

    return 0;
}
