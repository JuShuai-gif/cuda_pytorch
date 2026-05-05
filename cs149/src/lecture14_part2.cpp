// lecture14_part2.cpp — CS149 Lecture 14: MESI Protocol + False Sharing Demo
// Demonstrates MESI (adds Exclusive state) and the false sharing problem.
// Compile: g++ -std=c++17 -O2 -pthread lecture14_part2.cpp -o lecture14_part2
// Run:     ./lecture14_part2

#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <vector>
#include <atomic>
#include <cstring>
#include <cassert>

// ============================================================================
// Part 1: MESI Protocol Explanation
// ============================================================================
void explain_mesi() {
    std::cout << "=== CS149 Lecture 14: MESI (MESI, not Messi!) ===" << std::endl;
    std::cout << std::endl;
    std::cout << "MESI adds the E (Exclusive Clean) state to MSI." << std::endl;
    std::cout << std::endl;
    std::cout << "MESI States:" << std::endl;
    std::cout << "  M (Modified):   Dirty, exclusive — only copy, can write freely" << std::endl;
    std::cout << "  E (Exclusive):  Clean, exclusive — only copy, memory is up-to-date" << std::endl;
    std::cout << "  S (Shared):     Clean, shared — memory is up-to-date" << std::endl;
    std::cout << "  I (Invalid):    Not present" << std::endl;
    std::cout << std::endl;
    std::cout << "Key MESI innovation: E→M upgrade requires NO bus transaction!" << std::endl;
    std::cout << std::endl;
    std::cout << "MSI read-then-write (common case):" << std::endl;
    std::cout << "  PrRd: I→S via BusRd (1 transaction)" << std::endl;
    std::cout << "  PrWr: S→M via BusRdX (1 transaction) → 2 total" << std::endl;
    std::cout << std::endl;
    std::cout << "MESI read-then-write (no sharing):" << std::endl;
    std::cout << "  PrRd: I→E via BusRd (1 transaction, no other cache has line)" << std::endl;
    std::cout << "  PrWr: E→M silently (0 transactions) → 1 total!" << std::endl;
    std::cout << std::endl;
    std::cout << "How does cache know to enter E (not S) on BusRd?" << std::endl;
    std::cout << "  A shared line on the bus indicates whether another cache" << std::endl;
    std::cout << "  also has the line. If no cache asserts 'shared', enter E." << std::endl;

    std::cout << std::endl;
    std::cout << "MESI State Transitions:" << std::endl;
    std::cout << "  I + PrRd → E (if exclusive) or S (if shared)" << std::endl;
    std::cout << "  I + PrWr → M (via BusRdX)" << std::endl;
    std::cout << "  E + PrRd → E (hit, no bus)" << std::endl;
    std::cout << "  E + PrWr → M (silent upgrade, no bus!)" << std::endl;
    std::cout << "  S + PrRd → S (hit)" << std::endl;
    std::cout << "  S + PrWr → M (via BusRdX, upgrade)" << std::endl;
    std::cout << "  M + PrRd → M (hit)" << std::endl;
    std::cout << "  M + PrWr → M (hit)" << std::endl;
    std::cout << "  E/S + BusRdX → I (invalidate)" << std::endl;
    std::cout << "  M + BusRd → S (downgrade, BusWB)" << std::endl;
    std::cout << "  M + BusRdX → I (invalidate, BusWB)" << std::endl;
}

// ============================================================================
// Part 2: False Sharing Demo
// ============================================================================

constexpr int CACHE_LINE_SIZE = 64;   // typical x86 cache line
constexpr long long N_ITERATIONS = 50'000'000LL;   // many iterations to see the false sharing effect

// ----- Version 1: Unpadded — susceptible to false sharing -----
// Adjacent array elements share the same 64-byte cache line.
// When Thread 0 writes counter[0] and Thread 1 writes counter[1],
// both modify the SAME cache line → ping-pong invalidation traffic.
void worker_false_share(volatile long long* counter, long long n) {
    for (long long i = 0; i < n; ++i)
        (*counter)++;
}

// ----- Version 2: Padded — each counter occupies its own cache line -----
// sizeof(PaddedCounter) = CACHE_LINE_SIZE (64 bytes).
// Each thread writes to a different cache line → NO coherence traffic.
struct PaddedCounter {
    long long counter;
    char padding[CACHE_LINE_SIZE - sizeof(long long)];
};
static_assert(sizeof(PaddedCounter) == CACHE_LINE_SIZE,
              "PaddedCounter must be exactly one cache line");

void worker_no_false_share(volatile long long* counter, long long n) {
    for (long long i = 0; i < n; ++i)
        (*counter)++;
}

double time_execution(bool padded, int num_threads) {
    std::vector<std::thread> threads;

    // Allocate counters
    std::vector<long long> counters_unpadded(num_threads, 0);
    std::vector<PaddedCounter> counters_padded(num_threads);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < num_threads; ++t) {
        volatile long long* ptr;
        if (padded)
            ptr = &counters_padded[t].counter;
        else
            ptr = &counters_unpadded[t];

        threads.emplace_back(worker_no_false_share, ptr, N_ITERATIONS);
    }

    for (auto& th : threads) th.join();

    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

void demo_false_sharing() {
    std::cout << "================================================================" << std::endl;
    std::cout << "=== FALSE SHARING DEMO ===" << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << std::endl;
    std::cout << "Scenario: " << N_ITERATIONS << " increments per thread across multiple threads." << std::endl;
    std::cout << "          Each thread writes to its OWN counter (no data sharing)." << std::endl;
    std::cout << "          But adjacent counters may share a cache line → false sharing." << std::endl;
    std::cout << std::endl;

    std::cout << "Unpadded: sizeof(long long) = " << sizeof(long long) << " bytes" << std::endl;
    std::cout << "  Counter[0] and Counter[1] are " << sizeof(long long)
              << " bytes apart → SAME cache line!" << std::endl;
    std::cout << std::endl;

    std::cout << "Padded: sizeof(PaddedCounter) = " << sizeof(PaddedCounter) << " bytes" << std::endl;
    std::cout << "  Each counter is on its OWN cache line → NO false sharing." << std::endl;
    std::cout << std::endl;

    // Test with different thread counts
    unsigned int hw_threads = std::thread::hardware_concurrency();
    std::cout << "Hardware concurrency: " << hw_threads << std::endl;
    std::cout << std::endl;

    std::cout << std::left
              << std::setw(14) << "Threads"
              << std::setw(18) << "Unpadded (s)"
              << std::setw(16) << "Padded (s)"
              << std::setw(12) << "Speedup" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    for (int nt : {1, 2, 4, std::min(8, (int)hw_threads)}) {
        double t_unpadded = time_execution(false, nt);
        double t_padded   = time_execution(true,  nt);
        double speedup    = t_unpadded / t_padded;

        std::cout << std::left << std::fixed << std::setprecision(2)
                  << std::setw(14) << nt
                  << std::setw(18) << t_unpadded
                  << std::setw(16) << t_padded
                  << std::setw(10) << speedup << "x" << std::endl;
    }

    std::cout << std::endl;
    std::cout << "=== Why False Sharing Hurts ===" << std::endl;
    std::cout << "Cache line ping-pongs between cores:" << std::endl;
    std::cout << "  P0 writes Counter[0] → cache line moves to P0's cache (M)" << std::endl;
    std::cout << "  P1 writes Counter[1] → cache line moves to P1's cache (M)" << std::endl;
    std::cout << "    → P0's line invalidated (BusRdX)" << std::endl;
    std::cout << "  P0 writes Counter[0] → cache line moves back to P0" << std::endl;
    std::cout << "    → P1's line invalidated" << std::endl;
    std::cout << "  ...thousands of cycles wasted on coherence traffic!" << std::endl;
    std::cout << std::endl;
    std::cout << "Solution: pad per-thread data to CACHE_LINE_SIZE (64 bytes)." << std::endl;
}

// ============================================================================
// Part 3: AMAT (Average Memory Access Time) in Multiprocessors
// ============================================================================
void explain_amat() {
    std::cout << std::endl;
    std::cout << "=== AMAT in Multiprocessor Systems ===" << std::endl;
    std::cout << std::endl;
    std::cout << "AMAT = Σ (frequency × latency) for each access type" << std::endl;
    std::cout << std::endl;
    std::cout << "Uniprocessor access sources: Register, L1, L2, Main Memory" << std::endl;
    std::cout << "Multiprocessor adds: L3 shared, L3 modified (remote)" << std::endl;
    std::cout << std::endl;
    std::cout << "Core i7 Xeon 5500 Series approximate latencies:" << std::endl;
    std::cout << "  L1 hit:                    ~4 cycles" << std::endl;
    std::cout << "  L2 hit:                    ~10 cycles" << std::endl;
    std::cout << "  L3 hit (unshared):         ~40 cycles" << std::endl;
    std::cout << "  L3 hit (shared, other core): ~65 cycles" << std::endl;
    std::cout << "  L3 hit (modified, other core): ~75 cycles" << std::endl;
    std::cout << "  Local DRAM:                ~120 cycles" << std::endl;
    std::cout << "  Remote DRAM (NUMA):        ~400 cycles" << std::endl;
    std::cout << std::endl;
    std::cout << "Key insight: AMAT_multiprocessor > AMAT_uniprocessor because:" << std::endl;
    std::cout << "  1. Higher miss rates (shared cache capacity divided)" << std::endl;
    std::cout << "  2. Coherence misses (false sharing, true sharing)" << std::endl;
    std::cout << "  3. Remote NUMA access latency" << std::endl;
}

// ============================================================================
// main
// ============================================================================
int main() {
    explain_mesi();

    std::cout << std::endl;
    demo_false_sharing();

    explain_amat();

    return 0;
}
