/*
 * lecture17_part2.cpp - TM Data Versioning: Eager vs Lazy
 * Stanford CS149, Fall 2025 - Lecture 17
 *
 * Simulates the two data versioning strategies for transactional memory:
 *   1. Eager versioning (undo-log based):
 *      - Write directly to memory, maintain undo log for rollback
 *      - Fast commit (data already in place), slow abort (must undo)
 *      - Poor fault tolerance (system crash during transaction loses state)
 *
 *   2. Lazy versioning (write-buffer based):
 *      - Buffer writes in transaction-local storage
 *      - Flush buffer to memory on commit
 *      - Fast abort (just clear buffer), slow commit (must flush)
 *      - Good fault tolerance (no partial writes visible)
 *
 * This simulation uses a simple key-value store with transactional
 * semantics to compare both approaches.
 *
 * Compile: g++ -std=c++17 -pthread lecture17_part2.cpp -o lecture17_part2
 * Run: ./lecture17_part2
 */

#include <iostream>
#include <thread>
#include <vector>
#include <unordered_map>
#include <cassert>
#include <chrono>
#include <atomic>
#include <mutex>
#include <shared_mutex>

// ============================================================
// Shared memory: a simple key-value store
// ============================================================
// Protected by readers-writer lock to allow simulating
// concurrent transactions that may conflict.

class SharedMemory {
public:
    void write(int addr, int value) {
        std::unique_lock lock(mtx_);
        memory_[addr] = value;
    }

    int read(int addr) {
        std::shared_lock lock(mtx_);
        auto it = memory_.find(addr);
        return (it != memory_.end()) ? it->second : 0;
    }

    std::unordered_map<int, int> snapshot() const {
        // For demonstration: take a full snapshot
        std::shared_lock lock(mtx_);
        return memory_;
    }

private:
    std::unordered_map<int, int> memory_;
    mutable std::shared_mutex mtx_;
};

// ============================================================
// Part 1: Eager Versioning Transaction (Undo-Log Based)
// ============================================================
// Philosophy: "Write to memory immediately, hoping transaction
// won't abort. Deal with aborts when you have to."
//
// On write: update memory in place, log old value to undo log.
// On commit: clear undo log (nothing to do - data already in memory).
// On abort: replay undo log in reverse to restore old values.

class EagerTransaction {
public:
    enum State { ACTIVE, COMMITTED, ABORTED };

    EagerTransaction(SharedMemory& mem) : mem_(mem), state_(ACTIVE) {}

    // Write: eager - directly write to shared memory, save undo info
    void write(int addr, int value) {
        assert(state_ == ACTIVE && "Transaction already terminated");

        // Save old value for potential undo
        int old_value = mem_.read(addr);
        undo_log_.push_back({addr, old_value});

        // Write immediately to shared memory (eager)
        mem_.write(addr, value);

        write_set_.push_back(addr);
    }

    // Read: just read from shared memory (eager versioning)
    int read(int addr) {
        assert(state_ == ACTIVE && "Transaction already terminated");
        read_set_.push_back(addr);
        return mem_.read(addr);
    }

    // Commit: just mark committed (data already in memory)
    void commit() {
        assert(state_ == ACTIVE && "Transaction already terminated");
        state_ = COMMITTED;
        undo_log_.clear(); // Discard undo log
        std::cout << "  [Eager] Committed. Undo log cleared." << std::endl;
    }

    // Abort: replay undo log in reverse order
    void abort_txn() {
        assert(state_ == ACTIVE && "Transaction already terminated");
        std::cout << "  [Eager] Aborting! Replaying undo log..." << std::endl;

        // Replay in REVERSE order (last write undone first)
        for (auto it = undo_log_.rbegin(); it != undo_log_.rend(); ++it) {
            std::cout << "    Undo: addr[" << it->addr << "] ← " << it->old_value << std::endl;
            mem_.write(it->addr, it->old_value);
        }
        state_ = ABORTED;
    }

    State get_state() const { return state_; }
    const std::vector<int>& get_read_set() const { return read_set_; }
    const std::vector<int>& get_write_set() const { return write_set_; }

private:
    struct UndoEntry {
        int addr;
        int old_value;
    };

    SharedMemory& mem_;
    State state_;
    std::vector<int> read_set_;
    std::vector<int> write_set_;
    std::vector<UndoEntry> undo_log_;
};

// ============================================================
// Part 2: Lazy Versioning Transaction (Write-Buffer Based)
// ============================================================
// Philosophy: "Only write to memory when you have to (at commit)."
//
// On write: buffer the write locally, do not touch shared memory.
// On read: check write buffer first (read-your-own-writes), then shared memory.
// On commit: flush write buffer to shared memory.
// On abort: just clear the write buffer (nothing to undo).

class LazyTransaction {
public:
    enum State { ACTIVE, COMMITTED, ABORTED };

    LazyTransaction(SharedMemory& mem) : mem_(mem), state_(ACTIVE) {}

    // Write: lazy - buffer the write, don't touch shared memory yet
    void write(int addr, int value) {
        assert(state_ == ACTIVE && "Transaction already terminated");
        write_buffer_[addr] = value;
        write_set_.push_back(addr);
    }

    // Read: check write buffer first (read-your-own-writes),
    // then fall back to shared memory.
    int read(int addr) {
        assert(state_ == ACTIVE && "Transaction already terminated");
        read_set_.push_back(addr);

        // Check if this address was written by this transaction
        auto it = write_buffer_.find(addr);
        if (it != write_buffer_.end()) {
            return it->second; // Read own uncommitted write
        }
        return mem_.read(addr); // Read from shared memory
    }

    // Commit: flush write buffer to shared memory
    void commit() {
        assert(state_ == ACTIVE && "Transaction already terminated");
        std::cout << "  [Lazy] Committing. Flushing write buffer..." << std::endl;

        for (const auto& [addr, value] : write_buffer_) {
            std::cout << "    Flush: addr[" << addr << "] ← " << value << std::endl;
            mem_.write(addr, value);
        }
        state_ = COMMITTED;
        write_buffer_.clear();
    }

    // Abort: just discard the write buffer (fast!)
    void abort_txn() {
        assert(state_ == ACTIVE && "Transaction already terminated");
        std::cout << "  [Lazy] Aborting! Discarding write buffer (nothing to undo)." << std::endl;
        write_buffer_.clear();
        state_ = ABORTED;
    }

    State get_state() const { return state_; }
    const std::vector<int>& get_read_set() const { return read_set_; }
    const std::vector<int>& get_write_set() const { return write_set_; }

private:
    SharedMemory& mem_;
    State state_;
    std::vector<int> read_set_;
    std::vector<int> write_set_;
    std::unordered_map<int, int> write_buffer_; // Local write buffer
};

// ============================================================
// Demonstration
// ============================================================

void demo_eager_versioning() {
    std::cout << "=== Eager Versioning (Undo-Log Based) ===" << std::endl;

    SharedMemory mem;
    mem.write(100, 10); // Initialize: addr[100] = 10
    mem.write(200, 20); // Initialize: addr[200] = 20

    std::cout << "Initial state: addr[100]=" << mem.read(100)
              << ", addr[200]=" << mem.read(200) << std::endl;

    {
        EagerTransaction txn(mem);
        txn.write(100, 42); // Eager: writes 42 to memory immediately, saves undo(100,10)
        txn.write(200, 99); // Eager: writes 99 to memory immediately, saves undo(200,20)
        std::cout << "After eager writes: addr[100]=" << mem.read(100)
                  << " (immediately visible!), addr[200]=" << mem.read(99) << std::endl;

        // Simulate: we decide to abort
        txn.abort_txn();
    }

    std::cout << "After abort: addr[100]=" << mem.read(100)
              << " (restored from undo log), addr[200]=" << mem.read(200)
              << " (restored)" << std::endl;
    std::cout << std::endl;

    // Commit scenario
    {
        EagerTransaction txn(mem);
        txn.write(100, 500); // Eager write
        txn.commit(); // Fast: just clear undo log
    }

    std::cout << "After commit: addr[100]=" << mem.read(100) << " (value stays)" << std::endl;
    std::cout << std::endl;
}

void demo_lazy_versioning() {
    std::cout << "=== Lazy Versioning (Write-Buffer Based) ===" << std::endl;

    SharedMemory mem;
    mem.write(100, 10); // Initialize
    mem.write(200, 20); // Initialize

    std::cout << "Initial state: addr[100]=" << mem.read(100)
              << ", addr[200]=" << mem.read(200) << std::endl;

    {
        LazyTransaction txn(mem);
        txn.write(100, 42); // Lazy: buffers write, shared memory unchanged
        txn.write(200, 99); // Lazy: buffers write, shared memory unchanged
        std::cout << "After lazy writes (NOT YET in memory): addr[100]=" << mem.read(100)
                  << " (still old value!), addr[200]=" << mem.read(200) << std::endl;

        // Read-your-own-writes: transaction sees its own buffered writes
        std::cout << "Transaction reads: addr[100]=" << txn.read(100)
                  << " (sees own write), addr[200]=" << txn.read(200)
                  << " (sees own write)" << std::endl;

        // Simulate abort
        txn.abort_txn();
    }

    std::cout << "After abort: addr[100]=" << mem.read(100)
              << " (unchanged - buffer was just discarded)" << std::endl;
    std::cout << "              addr[200]=" << mem.read(200)
              << " (unchanged)" << std::endl;
    std::cout << std::endl;

    // Commit scenario
    {
        LazyTransaction txn(mem);
        txn.write(100, 500); // Buffered
        txn.write(200, 600); // Buffered
        txn.commit(); // Slow: flush buffer to memory
    }

    std::cout << "After commit: addr[100]=" << mem.read(100)
              << " (now visible), addr[200]=" << mem.read(200)
              << " (now visible)" << std::endl;
    std::cout << std::endl;
}

void demo_comparison() {
    std::cout << "=== Comparison Summary ===" << std::endl;
    std::cout << "┌─────────────────┬──────────────────────┬──────────────────────┐" << std::endl;
    std::cout << "│     Aspect      │  Eager (Undo-Log)    │  Lazy (Write-Buffer) │" << std::endl;
    std::cout << "├─────────────────┼──────────────────────┼──────────────────────┤" << std::endl;
    std::cout << "│ Commit speed    │ Fast (data in memory)│ Slow (flush buffer)  │" << std::endl;
    std::cout << "│ Abort speed     │ Slow (replay undo)   │ Fast (discard buffer)│" << std::endl;
    std::cout << "│ Fault tolerance │ Poor (partial writes)│ Good (all-or-nothing)│" << std::endl;
    std::cout << "│ Write visibility │ Immediate (to all)  │ Delayed (commit time)│" << std::endl;
    std::cout << "│ Isolation        │ Weaker (writes leak) │ Stronger (buffered)  │" << std::endl;
    std::cout << "│ Per-write overhead│ Log old value      │ Buffer new value     │" << std::endl;
    std::cout << "└─────────────────┴──────────────────────┴──────────────────────┘" << std::endl;
}

int main() {
    std::cout << "=== CS149 Lecture 17: TM Data Versioning ===" << std::endl;
    std::cout << std::endl;

    demo_eager_versioning();
    demo_lazy_versioning();
    demo_comparison();

    return 0;
}
