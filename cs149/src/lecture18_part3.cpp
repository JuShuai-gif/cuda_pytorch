/*
 * lecture18_part3.cpp - Hardware Transactional Memory (HTM) Simulation
 * Stanford CS149, Fall 2025 - Lecture 18
 *
 * Simulates cache-based Hardware Transactional Memory:
 *
 * Key HTM concepts from the lecture:
 *   1. Data versioning in caches (write buffer or undo log in cache lines)
 *   2. R/W bits on cache lines to track read-set and write-set membership
 *   3. Conflict detection through the cache coherence protocol:
 *      - BusRd to W-line → read-write conflict
 *      - BusRdX to R-line → write-read conflict
 *      - BusRdX to W-line → write-write conflict
 *   4. Two-phase commit: validate then gang-clear R/W bits
 *   5. Fast abort: invalidate write-set, gang-clear R/W bits,
 *      restore register checkpoint
 *
 * Also demonstrates Intel Haswell RTM-like semantics:
 *   - Fallback path (lock-based) when hardware aborts
 *   - Transaction may abort for any reason (eviction, interrupt, etc.)
 *
 * Compile: g++ -std=c++17 -pthread lecture18_part3.cpp -o lecture18_part3
 * Run: ./lecture18_part3
 */

#include <iostream>
#include <thread>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <atomic>
#include <mutex>
#include <cassert>
#include <random>
#include <chrono>

// ============================================================
// Simulated Cache Line with R/W Transaction Bits
// ============================================================
// Represents one cache line with HTM metadata.
// In real hardware: MESI state + R bit + W bit + Tag + Data.
// Here we simulate this structure per "address".

enum class CacheState { INVALID, SHARED, MODIFIED };

struct CacheLine {
    int addr;                // Address this line caches
    int data;                // The cached data
    CacheState mesi;         // MESI coherence state
    bool r_bit;              // Read-set membership (set on loads in transaction)
    bool w_bit;              // Write-set membership (set on stores in transaction)
    bool dirty;              // Whether data differs from "memory"

    CacheLine() : addr(-1), data(0), mesi(CacheState::INVALID),
                  r_bit(false), w_bit(false), dirty(false) {}
};

// ============================================================
// Hardware Transactional Memory Simulator
// ============================================================
// Simulates a cache with HTM support.
// Transactions track reads and writes via R/W bits on cache lines.
// Conflict detection is triggered by coherence lookups (BusRd/BusRdX).

class HardwareTransactionalMemory {
public:
    static constexpr int CACHE_SIZE = 16; // Small cache for simulation

    HardwareTransactionalMemory() {
        // Initialize global memory
        for (int i = 0; i < 256; ++i) {
            main_memory_[i] = 0;
        }
    }

    // ============================================================
    // HTM Operations (per-core view)
    // ============================================================

    // XBEGIN: Start a hardware transaction
    bool xbegin(int core_id, int fallback_addr = -1) {
        if (txn_active_[core_id]) {
            std::cerr << "Error: Core " << core_id
                      << " already in a transaction!" << std::endl;
            return false;
        }

        // Checkpoint register state (in real HW)
        // Initialize transactional cache state
        txn_active_[core_id] = true;
        txn_fallback_[core_id] = fallback_addr;

        // Save register checkpoint values for abort recovery
        checkpoint_[core_id] = reg_state_[core_id];

        // Gang-clear all R/W bits (start fresh)
        for (auto& line : caches_[core_id]) {
            line.r_bit = false;
            line.w_bit = false;
        }

        return true;
    }

    // XLOAD: Load within a transaction
    // Sets R bit on the loaded cache line.
    // Conflict detection: if another core issues BusRdX to this line,
    // it's a write-read conflict that aborts us.
    int xload(int core_id, int addr) {
        assert(txn_active_[core_id] && "Not in a transaction");

        // Find or load the cache line
        int cache_idx = find_or_load_line(core_id, addr);

        if (cache_idx < 0) {
            // Cache miss → in real HTM, eviction might abort transaction
            // Intel RTM: eviction of ANY line in read/write set → abort
            std::cout << "  [Core " << core_id << "] XLOAD addr[" << addr
                      << "] → CACHE MISS, aborting transaction!" << std::endl;
            xabort(core_id);
            return -1;
        }

        auto& line = caches_[core_id][cache_idx];

        // Conflict check: does another core have this line in its write set?
        if (check_conflict(core_id, addr, true)) { // true = read operation
            std::cout << "  [Core " << core_id << "] XLOAD addr[" << addr
                      << "] → READ-WRITE CONFLICT, aborting!" << std::endl;
            xabort(core_id);
            return -1;
        }

        // Mark as part of read set
        line.r_bit = true;

        // Load data from cache
        return line.data;
    }

    // XSTORE: Store within a transaction
    // Sets W bit on the stored cache line.
    // For lazy versioning: buffer in cache, don't write to main memory yet.
    // For eager versioning: write to memory, keep undo log in separate cache line.
    bool xstore(int core_id, int addr, int value) {
        assert(txn_active_[core_id] && "Not in a transaction");

        // Find or load the cache line
        int cache_idx = find_or_load_line(core_id, addr);

        if (cache_idx < 0) {
            std::cout << "  [Core " << core_id << "] XSTORE addr[" << addr
                      << "] → CACHE MISS, aborting!" << std::endl;
            xabort(core_id);
            return false;
        }

        auto& line = caches_[core_id][cache_idx];

        // Conflict check
        if (check_conflict(core_id, addr, false)) { // false = write operation
            std::cout << "  [Core " << core_id << "] XSTORE addr[" << addr
                      << "] → WRITE CONFLICT, aborting!" << std::endl;
            xabort(core_id);
            return false;
        }

        // Lazy versioning: buffer write in cache (mark dirty, set W bit)
        // Main memory NOT updated until commit
        line.w_bit = true;
        line.dirty = true;
        line.data = value;
        line.mesi = CacheState::MODIFIED;

        return true;
    }

    // XCOMMIT: Commit a transaction (two-phase commit)
    // Phase 1: Validate - request exclusive access to write-set lines
    //          (in real HW: upgrade to RdX for all W-bit lines)
    // Phase 2: Commit - gang-clear R/W bits, write-set data becomes valid/dirty
    bool xcommit(int core_id) {
        assert(txn_active_[core_id] && "Not in a transaction");

        std::cout << "  [Core " << core_id << "] XCOMMIT starting..." << std::endl;

        // Phase 1: Validate - check for conflicts on write-set
        // In real HW: issue BusRdX for each W-bit line to get exclusive access
        for (auto& line : caches_[core_id]) {
            if (line.w_bit) {
                // Check if any other core is reading or writing this line
                for (int other = 0; other < MAX_CORES; ++other) {
                    if (other == core_id || !txn_active_[other]) continue;
                    for (auto& other_line : caches_[other]) {
                        if (other_line.addr == line.addr &&
                            (other_line.r_bit || other_line.w_bit)) {
                            std::cout << "  [Core " << core_id << "] XCOMMIT FAIL: "
                                      << "addr[" << line.addr << "] conflict with Core "
                                      << other << std::endl;
                            xabort(core_id);
                            return false;
                        }
                    }
                }
            }
        }

        // Phase 2: Commit - flush dirty lines to main memory
        for (auto& line : caches_[core_id]) {
            if (line.w_bit && line.dirty) {
                main_memory_[line.addr] = line.data;
                std::cout << "  [Core " << core_id << "] COMMIT: flushed addr["
                          << line.addr << "] = " << line.data << " to memory" << std::endl;
            }
        }

        // Gang-clear R/W bits
        for (auto& line : caches_[core_id]) {
            line.r_bit = false;
            line.w_bit = false;
            // W-bit lines become normal dirty cache lines (valid committed data)
        }

        txn_active_[core_id] = false;
        std::cout << "  [Core " << core_id << "] XCOMMIT success!" << std::endl;
        return true;
    }

    // XABORT: Abort a transaction
    // 1. Invalidate write-set (discard dirty data in W-bit lines)
    // 2. Gang-clear R/W bits
    // 3. Restore register checkpoint
    void xabort(int core_id) {
        assert(txn_active_[core_id] && "Not in a transaction");

        std::cout << "  [Core " << core_id << "] XABORT! Rolling back..." << std::endl;

        // Invalidate write-set lines (discard uncommitted writes)
        for (auto& line : caches_[core_id]) {
            if (line.w_bit) {
                line.mesi = CacheState::INVALID;
                line.dirty = false;
                line.data = 0; // Discard
            }
        }

        // Gang-clear R/W bits
        for (auto& line : caches_[core_id]) {
            line.r_bit = false;
            line.w_bit = false;
        }

        // Restore register checkpoint
        reg_state_[core_id] = checkpoint_[core_id];

        txn_active_[core_id] = false;

        // Check if we should fall back to lock-based code
        if (txn_fallback_[core_id] >= 0) {
            std::cout << "  [Core " << core_id << "] Falling back to lock-based path."
                      << std::endl;
        }
    }

    // ============================================================
    // Cache coherence simulation
    // ============================================================
    bool check_conflict(int core_id, int addr, bool is_read) {
        // Simulate coherence snoops:
        // BusRd (shared request) to W-line → read-write conflict
        // BusRdX (exclusive request) to R-line → write-read conflict
        // BusRdX (exclusive request) to W-line → write-write conflict

        for (int other = 0; other < MAX_CORES; ++other) {
            if (other == core_id || !txn_active_[other]) continue;

            for (auto& other_line : caches_[other]) {
                if (other_line.addr != addr) continue;

                if (is_read) {
                    // We are READING: conflict if another core has W-bit set
                    if (other_line.w_bit) {
                        return true; // BusRd to W-line → R-W conflict
                    }
                } else {
                    // We are WRITING (BusRdX): conflict if other has R or W bit
                    if (other_line.r_bit || other_line.w_bit) {
                        return true; // BusRdX to R/W-line → W-R or W-W conflict
                    }
                }
            }
        }
        return false; // No conflict
    }

    // ============================================================
    // Fallback path: lock-based execution
    // ============================================================
    // Intel RTM requires a fallback path when hardware transactions
    // repeatedly abort (e.g., cache evictions, interrupts).
    void fallback_transfer(int core_id, int from_addr, int to_addr, int amount) {
        std::lock_guard<std::mutex> guard(fallback_lock_);

        // Safe, lock-based execution
        main_memory_[from_addr] -= amount;
        main_memory_[to_addr] += amount;

        std::cout << "  [Core " << core_id << "] Fallback: transferred " << amount
                  << " from addr[" << from_addr << "] to addr[" << to_addr << "]" << std::endl;
    }

    // ============================================================
    // Intel RTM-style optimistic transaction with fallback
    // ============================================================
    void rtm_transfer(int core_id, int from_addr, int to_addr, int amount) {
        int max_retries = 3;
        for (int attempt = 0; attempt < max_retries; ++attempt) {
            if (xbegin(core_id)) {
                // Optimistic path: use hardware transaction
                int from_val = xload(core_id, from_addr);
                if (from_val < 0) continue; // Aborted, retry

                int to_val = xload(core_id, to_addr);
                if (to_val < 0) continue; // Aborted, retry

                bool ok1 = xstore(core_id, from_addr, from_val - amount);
                if (!ok1) continue;

                bool ok2 = xstore(core_id, to_addr, to_val + amount);
                if (!ok2) continue;

                if (xcommit(core_id)) {
                    return; // Success!
                }
                // Commit failed, retry or fallback
            }
        }

        // Fallback: all HTM attempts failed
        fallback_transfer(core_id, from_addr, to_addr, amount);
    }

    // ============================================================
    // Utility
    // ============================================================
    int read_memory(int addr) const { return main_memory_[addr]; }
    void write_memory(int addr, int value) { main_memory_[addr] = value; }

    static constexpr int MAX_CORES = 4;

private:
    int find_or_load_line(int core_id, int addr) {
        // Check if line already in cache
        for (int i = 0; i < CACHE_SIZE; ++i) {
            if (caches_[core_id][i].addr == addr &&
                caches_[core_id][i].mesi != CacheState::INVALID) {
                return i;
            }
        }

        // Find empty line (or evict one)
        for (int i = 0; i < CACHE_SIZE; ++i) {
            if (caches_[core_id][i].mesi == CacheState::INVALID) {
                caches_[core_id][i].addr = addr;
                caches_[core_id][i].data = main_memory_[addr];
                caches_[core_id][i].mesi = CacheState::SHARED;
                caches_[core_id][i].dirty = false;
                caches_[core_id][i].r_bit = false;
                caches_[core_id][i].w_bit = false;
                return i;
            }
        }

        // Evict LRU-like: evict first non-transactional line
        for (int i = 0; i < CACHE_SIZE; ++i) {
            if (!caches_[core_id][i].r_bit && !caches_[core_id][i].w_bit) {
                // If dirty, write back
                if (caches_[core_id][i].dirty) {
                    main_memory_[caches_[core_id][i].addr] = caches_[core_id][i].data;
                }
                caches_[core_id][i].addr = addr;
                caches_[core_id][i].data = main_memory_[addr];
                caches_[core_id][i].mesi = CacheState::SHARED;
                caches_[core_id][i].dirty = false;
                return i;
            }
        }

        // All lines are in transaction read/write set → eviction aborts!
        return -1;
    }

    int main_memory_[256];
    CacheLine caches_[MAX_CORES][CACHE_SIZE];
    bool txn_active_[MAX_CORES] = {false};
    int txn_fallback_[MAX_CORES] = {-1, -1, -1, -1};
    int reg_state_[MAX_CORES] = {0};
    int checkpoint_[MAX_CORES] = {0};
    std::mutex fallback_lock_;
};

// ============================================================
// Demonstration
// ============================================================

void demo_htm_instructions() {
    std::cout << "=== HTM: Single-Core Transaction Execution ===" << std::endl;
    std::cout << std::endl;

    HardwareTransactionalMemory htm;
    htm.write_memory(100, 10); // addr[100] = 10
    htm.write_memory(200, 20); // addr[200] = 20

    std::cout << "Initial memory: addr[100]=" << htm.read_memory(100)
              << ", addr[200]=" << htm.read_memory(200) << std::endl;
    std::cout << std::endl;

    // Xbegin → Xload → Xstore → Xcommit
    std::cout << "[Transaction on Core 0]" << std::endl;
    htm.xbegin(0);

    int a = htm.xload(0, 100);
    std::cout << "  XLOAD addr[100] = " << a << " (R-bit set on cache line 100)" << std::endl;

    int b = htm.xload(0, 200);
    std::cout << "  XLOAD addr[200] = " << b << " (R-bit set on cache line 200)" << std::endl;

    htm.xstore(0, 100, a * 10); // 10 → 100
    std::cout << "  XSTORE addr[100] = " << a * 10
              << " (W-bit set, buffered in cache, NOT in memory yet)" << std::endl;

    std::cout << "  Memory during transaction: addr[100]=" << htm.read_memory(100)
              << " (still old value - lazy versioning!)" << std::endl;

    htm.xcommit(0);
    std::cout << "  Memory after commit: addr[100]=" << htm.read_memory(100)
              << " (now visible)" << std::endl;
}

void demo_htm_conflict() {
    std::cout << std::endl;
    std::cout << "=== HTM: Cross-Core Conflict Detection ===" << std::endl;
    std::cout << std::endl;

    HardwareTransactionalMemory htm;
    htm.write_memory(50, 100); // Shared data at addr[50] = 100

    // Core 0 starts transaction, reads addr[50]
    htm.xbegin(0);
    int val0 = htm.xload(0, 50);
    std::cout << "Core 0 reads addr[50] = " << val0 << " (R-bit set)" << std::endl;

    // Core 1 starts transaction, tries to WRITE addr[50]
    // This generates BusRdX on the coherence bus
    // Core 0's cache sees BusRdX to its R-bit line → WRITE-READ conflict
    htm.xbegin(1);
    std::cout << "Core 1 tries to write addr[50] = 200..." << std::endl;
    htm.xstore(1, 50, 200); // This should succeed for Core 1

    // Core 0's transaction is now in conflict
    // When Core 0 tries to xload again, it will detect the conflict
    std::cout << "Core 0 tries to read addr[50] again..." << std::endl;
    int val0b = htm.xload(0, 50);
    if (val0b < 0) {
        std::cout << "Core 0's transaction was aborted (WR conflict detected)!" << std::endl;
    }

    // Core 1 can commit
    htm.xcommit(1);
    std::cout << "Core 1 committed. addr[50] = " << htm.read_memory(50) << std::endl;
}

void demo_rtm_fallback() {
    std::cout << std::endl;
    std::cout << "=== Intel RTM: Fallback Path ===" << std::endl;
    std::cout << std::endl;

    std::cout << "Intel Haswell RTM provides:" << std::endl;
    std::cout << "  xbegin(fallback_addr) - start transaction" << std::endl;
    std::cout << "  xend                  - commit transaction" << std::endl;
    std::cout << "  xabort                - explicit abort" << std::endl;
    std::cout << std::endl;
    std::cout << "Key limitation: NO forward progress guarantee." << std::endl;
    std::cout << "  - Cache eviction of any line in read/write set → abort" << std::endl;
    std::cout << "  - Interrupts, page faults, context switches → abort" << std::endl;
    std::cout << "  - MUST provide lock-based fallback path" << std::endl;
    std::cout << std::endl;

    HardwareTransactionalMemory htm;
    htm.write_memory(10, 1000);
    htm.write_memory(20, 500);

    // Simulate RTM style: try HTM, fallback to locks
    std::cout << "Transfer $100 from addr[10] to addr[20]:" << std::endl;
    htm.rtm_transfer(0, 10, 20, 100);

    std::cout << "Result: addr[10]=" << htm.read_memory(10)
              << ", addr[20]=" << htm.read_memory(20) << std::endl;
}

int main() {
    std::cout << "=== CS149 Lecture 18: Hardware Transactional Memory ===" << std::endl;
    std::cout << std::endl;

    std::cout << "Key HTM Architecture Concepts:" << std::endl;
    std::cout << "  1. Data versioning in caches (lazy via write buffering)" << std::endl;
    std::cout << "  2. R/W bits per cache line track transactional read/write sets" << std::endl;
    std::cout << "  3. Conflict detection via coherence protocol:" << std::endl;
    std::cout << "     - BusRd to W-line → R-W conflict" << std::endl;
    std::cout << "     - BusRdX to R-line → W-R conflict" << std::endl;
    std::cout << "     - BusRdX to W-line → W-W conflict" << std::endl;
    std::cout << "  4. Two-phase commit: validate + gang-clear R/W bits" << std::endl;
    std::cout << "  5. Fast abort: invalidate W-set + restore register checkpoint" << std::endl;
    std::cout << std::endl;

    demo_htm_instructions();
    demo_htm_conflict();
    demo_rtm_fallback();

    std::cout << std::endl;
    std::cout << "HTM Performance (from lecture):" << std::endl;
    std::cout << "  - 2x-7x over STM performance" << std::endl;
    std::cout << "  - Within 10% of sequential for single thread" << std::endl;
    std::cout << "  - Near-ideal speedup on benchmarks like Vacation" << std::endl;
    std::cout << std::endl;
    std::cout << "HTM Limitations:" << std::endl;
    std::cout << "  - L1 cache size bounds transaction working set" << std::endl;
    std::cout << "  - Spurious aborts from interrupts, page faults, etc." << std::endl;
    std::cout << "  - Requires lock-based fallback for progress guarantee" << std::endl;
    std::cout << "  - Intel optimization guide (Ch. 12): guidelines for" << std::endl;
    std::cout << "    increasing transaction success probability" << std::endl;

    return 0;
}
