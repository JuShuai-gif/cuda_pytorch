// lecture15_part2.cpp — CS149 Lecture 15: C++11 Atomics and Memory Ordering
// Demonstrates memory_order_relaxed, acquire, release, acq_rel, seq_cst.
// Compile: g++ -std=c++17 -O2 -pthread lecture15_part2.cpp -o lecture15_part2
// Run:     ./lecture15_part2

#include <iostream>
#include <thread>
#include <atomic>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cassert>
#include <mutex>

// ============================================================================
// C++11 Memory Ordering Reference
// ============================================================================
//
// std::memory_order_relaxed:
//   No ordering constraints. Only atomicity guaranteed.
//   Compiler/hardware can reorder freely.
//
// std::memory_order_acquire:
//   For loads. No reads/writes AFTER this load can be reordered BEFORE it.
//   ("Acquire" the visibility of writes from the releasing thread.)
//
// std::memory_order_release:
//   For stores. No reads/writes BEFORE this store can be reordered AFTER it.
//   ("Release" all prior writes to be visible to acquiring threads.)
//
// std::memory_order_acq_rel:
//   Both acquire and release semantics. For read-modify-write (RMW) ops.
//
// std::memory_order_seq_cst:
//   Sequential consistency. Total global order of all seq_cst operations.
//   The default and the strongest (most expensive).

// ============================================================================
// Part 1: Relaxed Ordering — Atomic Counters
// ============================================================================
// relaxed is sufficient for simple counters where only atomicity matters,
// not ordering relative to other memory operations.

void demo_relaxed_counter() {
    std::cout << "=== Part 1: Relaxed Atomic Counter ===" << std::endl;
    std::cout << std::endl;
    std::cout << "relaxed ordering: only guarantees atomicity of the operation." << std::endl;
    std::cout << "No ordering guarantees relative to other memory accesses." << std::endl;
    std::cout << "Perfect for simple counters, statistics, etc." << std::endl;
    std::cout << std::endl;

    std::atomic<long long> counter{0};
    const int N_THREADS = 4;
    const long long N_INC = 25'000'000;

    auto worker = [&]() {
        for (long long i = 0; i < N_INC; ++i)
            counter.fetch_add(1, std::memory_order_relaxed);
    };

    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> threads;
    for (int i = 0; i < N_THREADS; ++i)
        threads.emplace_back(worker);
    for (auto& th : threads) th.join();
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Threads: " << N_THREADS << ", increments/thread: " << N_INC << std::endl;
    std::cout << "Expected: " << (N_THREADS * N_INC) << std::endl;
    std::cout << "Got:      " << counter.load() << std::endl;
    std::cout << "Time:     " << std::fixed << std::setprecision(1) << ms << " ms" << std::endl;
}

// ============================================================================
// Part 2: Acquire-Release — Message Passing
// ============================================================================
// The classic use case for acquire/release: one thread produces data,
// signals via an atomic flag; another thread spins on the flag and reads data.
//
// Producer:                         Consumer:
//   data = 42;  (non-atomic)          while (!flag.load(acquire));
//   flag.store(true, release);        int x = data;  // guaranteed to be 42
//
// release ensures data=42 is visible before flag=true.
// acquire ensures the read of flag happens before the read of data.

void demo_acquire_release_message() {
    std::cout << std::endl;
    std::cout << "=== Part 2: Acquire-Release Message Passing ===" << std::endl;
    std::cout << std::endl;

    // Shared variables
    int data = 0;  // non-atomic! protected by flag ordering
    std::atomic<bool> flag{false};

    std::thread producer([&]() {
        data = 42;                                    // (A) non-atomic write
        flag.store(true, std::memory_order_release);   // (B) release: makes (A) visible
    });

    std::thread consumer([&]() {
        while (!flag.load(std::memory_order_acquire))  // (C) acquire: synchronizes with (B)
            ;
        int x = data;                                  // (D) guaranteed to see 42!
        std::cout << "Consumer read data = " << x << std::endl;
    });

    producer.join();
    consumer.join();

    std::cout << "Release-store synchronizes-with acquire-load:" << std::endl;
    std::cout << "  All writes BEFORE release are visible AFTER acquire." << std::endl;
    std::cout << "  This is the fundamental building block of lock-free programming." << std::endl;
}

// ============================================================================
// Part 3: Sequential Consistency — Strongest Model
// ============================================================================
// seq_cst operations have a single total order visible to all threads.
// This is the default for std::atomic (e.g., atomic<int> x; x.store(1);).
// More expensive than acquire/release but easier to reason about.

void demo_seq_cst_ordering() {
    std::cout << std::endl;
    std::cout << "=== Part 3: Sequential Consistency (seq_cst) ===" << std::endl;
    std::cout << std::endl;

    // The Dekker's pattern is impossible to observe (0,0) with seq_cst.
    std::atomic<int> X{0}, Y{0};
    std::atomic<int> Z{0};  // witness variable

    int zero_zero = 0;
    const int N = 500'000;

    for (int trial = 0; trial < N; ++trial) {
        X.store(0, std::memory_order_seq_cst);
        Y.store(0, std::memory_order_seq_cst);

        int r1 = -1, r2 = -1;
        std::thread t0([&]() {
            X.store(1, std::memory_order_seq_cst);
            r1 = Y.load(std::memory_order_seq_cst);
        });
        std::thread t1([&]() {
            Y.store(1, std::memory_order_seq_cst);
            r2 = X.load(std::memory_order_seq_cst);
        });
        t0.join(); t1.join();

        if (r1 == 0 && r2 == 0) zero_zero++;
    }

    std::cout << "Dekker's pattern with seq_cst (N=" << N << ")" << std::endl;
    std::cout << "  r1=0 && r2=0 occurrences: " << zero_zero;
    if (zero_zero == 0)
        std::cout << " — impossible under SC, as expected." << std::endl;
    else
        std::cout << " — UNEXPECTED under SC!" << std::endl;

    std::cout << std::endl;
    std::cout << "seq_cst is the default memory_order for std::atomic." << std::endl;
    std::cout << "It provides the strongest guarantees but may have" << std::endl;
    std::cout << "higher overhead (especially on ARM, where it requires" << std::endl;
    std::cout << "explicit DMB (Data Memory Barrier) instructions)." << std::endl;
}

// ============================================================================
// Part 4: Acq-Rel for Read-Modify-Write (compare_exchange_weak)
// ============================================================================
// RMW operations like compare_exchange, fetch_add can use acq_rel
// to get both acquire (load phase) and release (store phase) semantics.

void demo_rmw_acq_rel() {
    std::cout << std::endl;
    std::cout << "=== Part 4: acq_rel for Read-Modify-Write ===" << std::endl;
    std::cout << std::endl;

    std::atomic<int> counter{0};
    std::atomic<bool> ready{false};
    int observed = 0;

    std::thread producer([&]() {
        // Use fetch_add with acq_rel: the store part has release semantics
        counter.fetch_add(1, std::memory_order_acq_rel);
        counter.fetch_add(1, std::memory_order_acq_rel);
        counter.fetch_add(1, std::memory_order_acq_rel);
    });

    std::thread consumer([&]() {
        // Spin until counter reaches 3
        int prev = counter.load(std::memory_order_acquire);
        while (prev < 3) {
            // compare_exchange_weak is an RMW — use acq_rel for both phases
            counter.compare_exchange_weak(prev, prev, std::memory_order_acq_rel);
            prev = counter.load(std::memory_order_acquire);
        }
        observed = counter.load(std::memory_order_acquire);
    });

    producer.join();
    consumer.join();

    std::cout << "fetch_add with acq_rel: both acquire and release semantics" << std::endl;
    std::cout << "  Consumer observed counter = " << observed << " (expected 3)" << std::endl;
    std::cout << "  acq_rel is the natural choice for RMW operations." << std::endl;
}

// ============================================================================
// Part 5: Memory Ordering Cost Comparison
// ============================================================================
void demo_ordering_cost() {
    std::cout << std::endl;
    std::cout << "=== Part 5: Memory Ordering Cost Comparison ===" << std::endl;
    std::cout << std::endl;

    const long long N = 50'000'000;
    std::atomic<long long> sum{0};

    auto bench = [&](std::memory_order mo, const char* name) {
        sum.store(0, std::memory_order_relaxed);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (long long i = 0; i < N; ++i)
            sum.fetch_add(1, mo);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "  " << std::left << std::setw(20) << name
                  << std::fixed << std::setprecision(1) << std::setw(10) << ms << " ms";
        if (mo == std::memory_order_relaxed)
            std::cout << " (baseline)";
        std::cout << std::endl;
    };

    std::cout << "fetch_add performance (" << (N / 1'000'000) << "M ops, single thread):" << std::endl;
    bench(std::memory_order_relaxed, "relaxed");
    bench(std::memory_order_acquire, "acquire");   // not ideal for RMW, but legal
    bench(std::memory_order_release, "release");   // not ideal for RMW, but legal
    bench(std::memory_order_acq_rel, "acq_rel");
    bench(std::memory_order_seq_cst, "seq_cst");

    std::cout << std::endl;
    std::cout << "On x86: relaxed, acquire, release, and acq_rel have similar cost" << std::endl;
    std::cout << "(x86 provides strong ordering by default)." << std::endl;
    std::cout << "seq_cst may require mfence on x86, making it more expensive." << std::endl;
    std::cout << "On ARM: acquire/release require explicit barriers, larger differences." << std::endl;
}

// ============================================================================
// Part 6: Memory Ordering Cheat Sheet
// ============================================================================
void print_cheat_sheet() {
    std::cout << std::endl;
    std::cout << "=== C++11 Memory Ordering Cheat Sheet ===" << std::endl;
    std::cout << std::endl;
    std::cout << std::left
              << std::setw(18) << "Order"
              << std::setw(12) << "Load"
              << std::setw(12) << "Store"
              << std::setw(51) << "Use Case" << std::endl;
    std::cout << std::string(93, '-') << std::endl;
    std::cout << std::left
              << std::setw(18) << "relaxed"
              << std::setw(12) << "✓"
              << std::setw(12) << "✓"
              << "Counters, statistics (no ordering needed)" << std::endl;
    std::cout << std::left
              << std::setw(18) << "acquire"
              << std::setw(12) << "✓"
              << std::setw(12) << "—"
              << "Consumer reads; pairs with release store" << std::endl;
    std::cout << std::left
              << std::setw(18) << "release"
              << std::setw(12) << "—"
              << std::setw(12) << "✓"
              << "Producer writes; pairs with acquire load" << std::endl;
    std::cout << std::left
              << std::setw(18) << "acq_rel"
              << std::setw(12) << "—"
              << std::setw(12) << "—"
              << "RMW ops only (fetch_add, compare_exchange)" << std::endl;
    std::cout << std::left
              << std::setw(18) << "seq_cst"
              << std::setw(12) << "✓"
              << std::setw(12) << "✓"
              << "Sequential consistency (default, strongest)" << std::endl;
}

// ============================================================================
// main
// ============================================================================
int main() {
    demo_relaxed_counter();
    demo_acquire_release_message();
    demo_seq_cst_ordering();
    demo_rmw_acq_rel();
    demo_ordering_cost();
    print_cheat_sheet();

    std::cout << std::endl;
    std::cout << "=== Key Takeaway ===" << std::endl;
    std::cout << "Start with seq_cst (default). Optimize to acquire/release" << std::endl;
    std::cout << "only after profiling shows it matters. Never use relaxed" << std::endl;
    std::cout << "unless you can PROVE no ordering is needed." << std::endl;

    return 0;
}
