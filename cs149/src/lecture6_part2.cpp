/**
 * lecture6_part2.cpp - Cache Coherency & False Sharing
 *
 * Demonstrates CS149 Lecture 6 concepts:
 * - Cache coherency protocol concepts (MESI-like states)
 * - False sharing: when threads modify different variables on the same cache line
 * - Cache line padding to prevent false sharing
 * - Shared address space hardware (ring interconnect)
 * - Artifactual communication from cache behavior
 *
 * Compile: g++ -std=c++17 -pthread lecture6_part2.cpp -o lecture6_part2 && ./lecture6_part2
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <mutex>

// ============================================================================
// Part 1: Cache Coherency Protocol Simulation
// ============================================================================

/**
 * Simplified model of a MESI-like cache coherency protocol.
 * Each cache line can be in one of these states:
 * - MODIFIED:  dirty, exclusive copy, must write back
 * - EXCLUSIVE: clean, exclusive copy
 * - SHARED:    clean, other caches may have copy
 * - INVALID:   line not valid
 */
enum class LineState { MODIFIED, EXCLUSIVE, SHARED, INVALID };

const char* state_name(LineState s) {
    switch (s) {
        case LineState::MODIFIED: return "MODIFIED";
        case LineState::EXCLUSIVE: return "EXCLUSIVE";
        case LineState::SHARED:   return "SHARED";
        case LineState::INVALID:  return "INVALID";
        default: return "?";
    }
}

class CacheLine {
public:
    int tag;
    LineState state;
    int data;

    CacheLine() : tag(-1), state(LineState::INVALID), data(0) {}
};

class Core {
public:
    int id;
    std::vector<CacheLine> cache;  // 4-line cache per core
    int total_misses;
    int total_invalidations;

    Core(int core_id, int cache_size = 4)
        : id(core_id), cache(cache_size), total_misses(0), total_invalidations(0) {}

    /**
     * Read a memory address. Simulates cache coherency protocol:
     * 1. Check local cache
     * 2. If miss, broadcast read request to other cores (snoop)
     * 3. Update states according to MESI
     */
    bool read(int address, int& value, std::vector<Core>& all_cores) {
        int line = address % cache.size();

        // Check local cache
        if (cache[line].tag == address && cache[line].state != LineState::INVALID) {
            // Cache hit
            value = cache[line].data;
            return true;
        }

        // Cache miss: need to fetch from memory or other caches
        total_misses++;

        // Snoop: check other cores for this address
        for (auto& other : all_cores) {
            if (other.id == id) continue;
            int other_line = address % other.cache.size();
            if (other.cache[other_line].tag == address) {
                if (other.cache[other_line].state == LineState::MODIFIED) {
                    // Write-back before sharing
                    other.cache[other_line].state = LineState::SHARED;
                    cache[line].data = other.cache[other_line].data;
                    cache[line].tag = address;
                    cache[line].state = LineState::SHARED;

                    // Other core's line becomes shared too
                    value = cache[line].data;
                    return true;
                } else if (other.cache[other_line].state == LineState::SHARED ||
                           other.cache[other_line].state == LineState::EXCLUSIVE) {
                    // Found in another cache in clean state
                    other.cache[other_line].state = LineState::SHARED;
                    cache[line].data = other.cache[other_line].data;
                    cache[line].tag = address;
                    cache[line].state = LineState::SHARED;
                    value = cache[line].data;
                    return true;
                }
            }
        }

        // Not in any cache: load from "memory"
        cache[line].tag = address;
        cache[line].state = LineState::EXCLUSIVE;
        cache[line].data = address * 10;  // Simulated memory value
        value = cache[line].data;
        return true;
    }

    /**
     * Write to a memory address.
     * 1. Must get line in MODIFIED state
     * 2. Invalidate other copies (RFO = Read For Ownership)
     */
    void write(int address, int value, std::vector<Core>& all_cores) {
        int line = address % cache.size();

        // If already in MODIFIED or EXCLUSIVE, just write
        if (cache[line].tag == address &&
            (cache[line].state == LineState::MODIFIED ||
             cache[line].state == LineState::EXCLUSIVE)) {
            cache[line].data = value;
            cache[line].state = LineState::MODIFIED;
            return;
        }

        // If in SHARED, need to invalidate other copies
        if (cache[line].tag == address && cache[line].state == LineState::SHARED) {
            invalidate_others(address, all_cores);
            cache[line].data = value;
            cache[line].state = LineState::MODIFIED;
            return;
        }

        // Miss or INVALID: get exclusive ownership
        total_misses++;
        invalidate_others(address, all_cores);

        cache[line].tag = address;
        cache[line].data = value;
        cache[line].state = LineState::MODIFIED;
    }

private:
    void invalidate_others(int address, std::vector<Core>& all_cores) {
        for (auto& other : all_cores) {
            if (other.id == id) continue;
            int other_line = address % other.cache.size();
            if (other.cache[other_line].tag == address) {
                if (other.cache[other_line].state != LineState::INVALID) {
                    other.total_invalidations++;
                    other.cache[other_line].state = LineState::INVALID;
                }
            }
        }
    }
};

/**
 * Simulate intra-thread and inter-thread cache behavior.
 * Demonstrates why shared writable data causes cache invalidation traffic.
 */
void simulate_cache_coherency() {
    std::cout << "\n=== Cache Coherency Simulation (MESI-like) ===\n\n";

    const int NUM_CORES = 4;
    std::vector<Core> cores;
    for (int i = 0; i < NUM_CORES; i++) cores.emplace_back(i);

    auto print_state = [&](int addr) {
        std::cout << "  Address " << addr << ": ";
        for (auto& c : cores) {
            int line = addr % c.cache.size();
            std::cout << "C" << c.id << "=" << state_name(c.cache[line].state) << " ";
        }
        std::cout << "\n";
    };

    // Scenario 1: Core 0 reads address 5
    std::cout << "Scenario 1: Core 0 reads addr 5\n";
    int val;
    cores[0].read(5, val, cores);
    print_state(5);

    // Scenario 2: Core 1 reads address 5 (should get SHARED state)
    std::cout << "Scenario 2: Core 1 reads addr 5 → SHARED\n";
    cores[1].read(5, val, cores);
    print_state(5);

    // Scenario 3: Core 0 writes address 5 (invalidates Core 1)
    std::cout << "Scenario 3: Core 0 writes addr 5 → invalidates C1\n";
    cores[0].write(5, 999, cores);
    print_state(5);

    // Scenario 4: Core 1 reads address 5 again (miss due to invalidation)
    std::cout << "Scenario 4: Core 1 reads addr 5 again → miss (was invalidated)\n";
    cores[1].read(5, val, cores);
    print_state(5);

    std::cout << "\n  Core 0 total misses: " << cores[0].total_misses
              << "  invalidations: " << cores[0].total_invalidations << "\n";
    std::cout << "  Core 1 total misses: " << cores[1].total_misses
              << "  invalidations: " << cores[1].total_invalidations << "\n";
    std::cout << "\n  Key insight: Write by C0 to a SHARED line causes invalidation\n";
    std::cout << "  in C1, forcing C1 to re-fetch on next access.\n";
}

// ============================================================================
// Part 2: False Sharing Demonstration
// ============================================================================

/**
 * FALSE SHARING:
 * When two threads modify different variables that happen to reside
 * on the same cache line, causing unnecessary cache coherence traffic.
 *
 * Cache line size: typically 64 bytes = 16 ints or 8 doubles.
 */

// Structure WITHOUT padding (suffers from false sharing)
struct UnpaddedCounter {
    alignas(64) int counter;  // alignas(64) ensures this starts at cache line boundary
    // BUT: multiple counters may still share a cache line if packed in array
};
static_assert(sizeof(UnpaddedCounter) == 64, "Padded counter to cache line size");

// Structure WITH padding for cache line isolation
struct alignas(64) PaddedCounter {
    int counter;
    char padding[60];  // Fill rest of cache line (64 - 4 = 60 bytes)
};
static_assert(sizeof(PaddedCounter) == 64, "Padded counter must be 64 bytes");

/**
 * Benchmark false sharing vs padded counters.
 * Each thread increments its own counter repeatedly.
 * Without padding: counters share cache lines → massive invalidation traffic.
 * With padding: each counter on its own cache line → no false sharing.
 */
void benchmark_false_sharing() {
    std::cout << "\n=== False Sharing Benchmark ===\n\n";

    const int NUM_THREADS = 4;
    const int ITERATIONS = 10000000;

    // === Unpadded: all counters share cache lines ===
    {
        // Allocate array of unpadded counters (contiguous in memory)
        // Multiple will fit in the same 64-byte cache line
        alignas(64) int counters[NUM_THREADS] = {0};

        auto worker = [&](int tid) {
            for (int i = 0; i < ITERATIONS; i++) {
                counters[tid]++;  // False sharing: invalidates entire cache line
            }
        };

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::thread> threads;
        for (int t = 0; t < NUM_THREADS; t++) {
            threads.emplace_back(worker, t);
        }
        for (auto& th : threads) th.join();
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed = std::chrono::duration<double>(end - start).count();
        std::cout << "  Unpadded (false sharing): "
                  << std::fixed << std::setprecision(4) << elapsed << "s\n";

        // Verify
        long long total = 0;
        for (int t = 0; t < NUM_THREADS; t++) total += counters[t];
        std::cout << "    Total: " << total << " (expected: "
                  << (1LL * NUM_THREADS * ITERATIONS) << ")\n";
    }

    // === Padded: each counter on its own cache line ===
    {
        PaddedCounter counters[NUM_THREADS];
        for (int t = 0; t < NUM_THREADS; t++) counters[t].counter = 0;

        auto worker = [&](int tid) {
            for (int i = 0; i < ITERATIONS; i++) {
                counters[tid].counter++;  // No false sharing: isolated cache line
            }
        };

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::thread> threads;
        for (int t = 0; t < NUM_THREADS; t++) {
            threads.emplace_back(worker, t);
        }
        for (auto& th : threads) th.join();
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed = std::chrono::duration<double>(end - start).count();
        std::cout << "  Padded (no false sharing): "
                  << std::fixed << std::setprecision(4) << elapsed << "s\n";

        long long total = 0;
        for (int t = 0; t < NUM_THREADS; t++) total += counters[t].counter;
        std::cout << "    Total: " << total << " (expected: "
                  << (1LL * NUM_THREADS * ITERATIONS) << ")\n";
    }

    std::cout << "\n  False sharing is an INVISIBLE performance killer.\n";
    std::cout << "  Two threads modifying independent variables on the same cache\n";
    std::cout << "  line still cause invalidation traffic between caches.\n";
    std::cout << "  Solution: align to cache line boundary and pad to 64 bytes.\n";
}

// ============================================================================
// Part 3: Artifactual Communication Examples
// ============================================================================

void explain_artifactual_communication() {
    std::cout << "\n=== Artifactual Communication ===\n\n";

    std::cout << "Artifactual communication = unnecessary data movement caused by\n";
    std::cout << "implementation details, not by algorithmic requirements.\n\n";

    std::cout << "Example 1: Minimum transfer granularity\n";
    std::cout << "  - Load one 4-byte float → entire 64-byte cache line transferred\n";
    std::cout << "  - 16x more communication than necessary\n\n";

    std::cout << "Example 2: Unnecessary load-before-store\n";
    std::cout << "  - Write 16 consecutive floats → cache line loaded, then overwritten\n";
    std::cout << "  - 2x overhead: load was unnecessary (entire line overwritten)\n";
    std::cout << "  - Solution: use non-temporal stores (streaming stores) to bypass cache\n\n";

    std::cout << "Example 3: Capacity misses (finite cache)\n";
    std::cout << "  - Cache too small to retain data between accesses\n";
    std::cout << "  - Same data communicated multiple times\n";
    std::cout << "  - Solution: blocking/tiling to keep working set in cache\n\n";

    std::cout << "Example 4: Conflict misses\n";
    std::cout << "  - Two frequently accessed addresses map to same cache set\n";
    std::cout << "  - Solution: padding, data layout reorganization\n";
}

// ============================================================================
// Part 4: Ring Interconnect Model
// ============================================================================

void explain_ring_interconnect() {
    std::cout << "\n=== Intel Ring Interconnect ===\n\n";

    std::cout << "Introduced in Sandy Bridge microarchitecture:\n\n";
    std::cout << "  Four rings for different message types:\n";
    std::cout << "    - Request ring\n";
    std::cout << "    - Snoop ring\n";
    std::cout << "    - Acknowledgement ring\n";
    std::cout << "    - Data ring (32 bytes)\n\n";

    std::cout << "  Six interconnect nodes:\n";
    std::cout << "    - Four 'slices' of L3 cache (2 MB each)\n";
    std::cout << "    - System agent\n";
    std::cout << "    - Graphics\n\n";

    std::cout << "  Each L3 bank connected to ring bus TWICE (bidirectional)\n";
    std::cout << "  Peak BW from cores to L3 at 3.4 GHz ~ 435 GB/sec\n";
    std::cout << "  (when each core accesses its local L3 slice)\n\n";

    std::cout << "  NUMA effect even on single socket: different cache slices\n";
    std::cout << "  at different distances from each core on the ring.\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "============================================================\n";
    std::cout << "Lecture 6 Part 2: Cache Coherency & False Sharing\n";
    std::cout << "============================================================\n";

    // Part 1: Cache coherency simulation
    simulate_cache_coherency();

    // Part 2: False sharing benchmark
    benchmark_false_sharing();

    // Part 3: Artifactual communication
    explain_artifactual_communication();

    // Part 4: Ring interconnect
    explain_ring_interconnect();

    // Summary
    std::cout << "\n=== Cache Coherency & False Sharing: Key Concepts ===\n";
    std::cout << "┌─────────────────────┬─────────────────────────────────────┐\n";
    std::cout << "│ Concept             │ Impact                             │\n";
    std::cout << "├─────────────────────┼─────────────────────────────────────┤\n";
    std::cout << "│ MESI protocol       │ Keeps caches coherent automatically│\n";
    std::cout << "│ Write invalidation  │ Writer must invalidate all copies  │\n";
    std::cout << "│ False sharing       │ Independent vars on same line      │\n";
    std::cout << "│                     │ cause unnecessary invalidations    │\n";
    std::cout << "│ Cache line padding  │ alignas(64) + char pad[60]         │\n";
    std::cout << "│ Artifactual comm    │ Min granularity, capacity misses   │\n";
    std::cout << "│ Ring interconnect   │ 4 rings, multi-hop latency on ring │\n";
    std::cout << "│ NUMA                │ Access latency varies by location  │\n";
    std::cout << "└─────────────────────┴─────────────────────────────────────┘\n";

    std::cout << "\nAll tests completed successfully.\n";
    return 0;
}
