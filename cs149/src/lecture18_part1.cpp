/*
 * lecture18_part1.cpp - Software Transactional Memory (STM) Implementation
 * Stanford CS149, Fall 2025 - Lecture 18
 *
 * Implements a software transactional memory system based on the Intel
 * McRT STM algorithm described in the lecture:
 *   - Eager versioning (undo-log based): writes go directly to memory
 *   - Optimistic reads: validate after each read
 *   - Pessimistic writes: acquire lock before writing
 *   - Timestamp-based version tracking per object
 *
 * Key data structures:
 *   - Transaction Descriptor (per-thread): read-set, write-set, undo-log
 *   - Transaction Record (per-object): writer lock + version number
 *   - Global timestamp: incremented by 2 on each writing commit
 *     (LSb = write-lock bit, MS bits = version number)
 *
 * STM Operations:
 *   STM Read: direct read → validate (unlocked, version ≤ local ts) → insert in read-set
 *   STM Write: validate → acquire lock → create undo-log entry → write in place
 *   STM Commit: increment global ts by 2 → validate read-set → release locks with new version
 *
 * Compile: g++ -std=c++17 -pthread lecture18_part1.cpp -o lecture18_part1
 * Run: ./lecture18_part1
 */

#include <iostream>
#include <thread>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>

// ============================================================
// Transaction Record (per-object)
// ============================================================
// 64-bit record encoding:
//   LSb (bit 0): 0 = writer-locked, 1 = not locked
//   MS bits (63:1): timestamp (version number) of last commit if unlocked
//                   OR pointer-like ID to owner transaction if locked
//
// This is analogous to how Intel McRT STM packs lock state and version
// into a single word.

using TxVersion = uint64_t;

// Encode/decode the transaction record
inline bool is_locked(TxVersion rec) {
    return (rec & 1ULL) == 0; // LSb = 0 means locked
}

inline uint64_t get_version(TxVersion rec) {
    return rec >> 1; // Strip LSb to get version
}

inline TxVersion make_locked(int owner_id) {
    // Locked: LSb=0, rest = owner_id shifted up by 1
    return (static_cast<uint64_t>(owner_id + 1) << 1);
}

inline TxVersion make_unlocked(uint64_t version) {
    // Unlocked: LSb=1, rest = version shifted up
    return (version << 1) | 1ULL;
}

// ============================================================
// STM System
// ============================================================

class SoftwareTransactionalMemory {
public:
    SoftwareTransactionalMemory() : global_timestamp_(0) {}

    // ---- STM Read (Optimistic) ----
    // 1. Direct read of memory location (eager)
    // 2. Validate read data: check unlocked and version ≤ local timestamp
    // 3. If validation fails, validate entire read-set for consistency
    // 4. Insert in read set, return value
    int stm_read(int txn_id, int obj_id) {
        int value = memory_[obj_id]; // Direct read (eager versioning)

        // Check if the object is locked by another transaction
        TxVersion rec = records_[obj_id];
        if (is_locked(rec) && get_version(rec) != static_cast<uint64_t>(txn_id + 1)) {
            // Object is locked by another transaction → abort
            std::cout << "  [STM Read] Txn " << txn_id << " read obj[" << obj_id
                      << "] FAIL: locked by Txn " << (get_version(rec) - 1) << std::endl;
            return -1; // Signal conflict
        }

        // Check version: data should not be newer than our local timestamp
        uint64_t obj_version = get_version(rec);
        uint64_t local_ts = local_timestamps_[txn_id];
        if (!is_locked(rec) && obj_version > local_ts) {
            // Data is newer than our snapshot → need to validate entire read-set
            if (!validate_read_set(txn_id)) {
                std::cout << "  [STM Read] Txn " << txn_id << " read obj[" << obj_id
                          << "] FAIL: version " << obj_version << " > local ts "
                          << local_ts << std::endl;
                return -1; // Validation failed → abort
            }
            // Update local timestamp after successful validation
            local_timestamps_[txn_id] = global_timestamp_.load(std::memory_order_acquire);
        }

        // Insert into read-set
        read_sets_[txn_id].insert(obj_id);
        return value;
    }

    // ---- STM Write (Pessimistic) ----
    // 1. Validate data (check unlocked, version ≤ local timestamp)
    // 2. Acquire lock on the object
    // 3. Create undo-log entry (save old value)
    // 4. Write data in place (eager versioning)
    bool stm_write(int txn_id, int obj_id, int new_value) {
        TxVersion rec = records_[obj_id];

        // Conflict check: is object locked by another transaction?
        if (is_locked(rec) && get_version(rec) != static_cast<uint64_t>(txn_id + 1)) {
            std::cout << "  [STM Write] Txn " << txn_id << " write obj[" << obj_id
                      << "] FAIL: locked by Txn " << (get_version(rec) - 1) << std::endl;
            return false; // Write-write conflict
        }

        // Version check for unlocked objects
        if (!is_locked(rec)) {
            uint64_t obj_version = get_version(rec);
            uint64_t local_ts = local_timestamps_[txn_id];
            if (obj_version > local_ts) {
                if (!validate_read_set(txn_id)) {
                    std::cout << "  [STM Write] Txn " << txn_id << " write obj[" << obj_id
                              << "] FAIL: validation failed" << std::endl;
                    return false;
                }
                local_timestamps_[txn_id] = global_timestamp_.load(std::memory_order_acquire);
            }
        }

        // Acquire write lock
        uint64_t old_locked_rec = make_locked(txn_id);
        records_[obj_id] = old_locked_rec;

        // Save old value to undo-log (for eager versioning rollback)
        int old_value = memory_[obj_id];
        undo_logs_[txn_id].push_back({obj_id, old_value});

        // Write in place (eager versioning)
        memory_[obj_id] = new_value;

        // Insert into write-set
        write_sets_[txn_id].insert(obj_id);

        return true;
    }

    // ---- STM Commit ----
    // 1. Atomically increment global timestamp by 2
    // 2. If pre-incremented (old) ts > local_ts, validate read-set
    //    (check for recently committed transactions)
    // 3. For each item in write-set: release lock, set version = global ts
    bool stm_commit(int txn_id) {
        // Step 1: Increment global timestamp by 2
        // (LSb used for write-lock, MS bits are version number)
        uint64_t old_global_ts = global_timestamp_.fetch_add(2, std::memory_order_acq_rel);
        uint64_t new_global_ts = old_global_ts + 2;

        // Step 2: Check if any transaction committed since our last validation
        if (old_global_ts > local_timestamps_[txn_id]) {
            if (!validate_read_set(txn_id)) {
                std::cout << "  [STM Commit] Txn " << txn_id
                          << " FAIL: read-set validation failed at commit time." << std::endl;
                // Rollback: replay undo log
                rollback(txn_id);
                return false;
            }
        }

        // Step 3: Release write locks and stamp with new version
        for (int obj_id : write_sets_[txn_id]) {
            records_[obj_id] = make_unlocked(new_global_ts);
        }

        // Clear transaction state
        read_sets_[txn_id].clear();
        write_sets_[txn_id].clear();
        undo_logs_[txn_id].clear();
        local_timestamps_[txn_id] = new_global_ts;

        std::cout << "  [STM Commit] Txn " << txn_id << " committed. Global ts = "
                  << new_global_ts << std::endl;
        return true;
    }

    // ---- STM Abort / Rollback ----
    void rollback(int txn_id) {
        std::cout << "  [STM Rollback] Txn " << txn_id << " rolling back..." << std::endl;

        // Replay undo log in reverse order (eager versioning)
        auto& log = undo_logs_[txn_id];
        for (auto it = log.rbegin(); it != log.rend(); ++it) {
            memory_[it->obj_id] = it->old_value;
        }

        // Release locks
        for (int obj_id : write_sets_[txn_id]) {
            uint64_t old_version = original_versions_[txn_id][obj_id];
            records_[obj_id] = make_unlocked(old_version);
        }

        // Clear state
        read_sets_[txn_id].clear();
        write_sets_[txn_id].clear();
        undo_logs_[txn_id].clear();
        original_versions_[txn_id].clear();
    }

    // ---- Read-Set Validation ----
    // Check that all objects in the read-set are still at a version
    // consistent with our local timestamp.
    bool validate_read_set(int txn_id) {
        for (int obj_id : read_sets_[txn_id]) {
            TxVersion rec = records_[obj_id];
            if (is_locked(rec)) {
                // If locked, check if it's locked by US (re-read is ok)
                if (get_version(rec) != static_cast<uint64_t>(txn_id + 1)) {
                    return false; // Locked by another transaction
                }
            } else {
                uint64_t obj_version = get_version(rec);
                if (obj_version > local_timestamps_[txn_id]) {
                    return false; // Data has been updated since we read it
                }
            }
        }
        return true;
    }

    // ---- Initialization helpers ----
    void init_object(int obj_id, int value) {
        memory_[obj_id] = value;
        records_[obj_id] = make_unlocked(0); // Initial version = 0
    }

    void init_txn(int txn_id) {
        local_timestamps_[txn_id] = global_timestamp_.load(std::memory_order_acquire);
        read_sets_[txn_id].clear();
        write_sets_[txn_id].clear();
        undo_logs_[txn_id].clear();
        original_versions_[txn_id].clear();
    }

    int get_value(int obj_id) const {
        auto it = memory_.find(obj_id);
        return (it != memory_.end()) ? it->second : 0;
    }

    uint64_t get_global_ts() const {
        return global_timestamp_.load();
    }

private:
    struct UndoEntry {
        int obj_id;
        int old_value;
    };

    std::unordered_map<int, int> memory_;                          // Shared memory
    std::unordered_map<int, TxVersion> records_;                   // Per-object TxRecords
    std::atomic<uint64_t> global_timestamp_;                       // Global timestamp

    // Per-transaction state
    std::unordered_map<int, uint64_t> local_timestamps_;           // Local timestamp
    std::unordered_map<int, std::unordered_set<int>> read_sets_;   // Read-set per txn
    std::unordered_map<int, std::unordered_set<int>> write_sets_;  // Write-set per txn
    std::unordered_map<int, std::vector<UndoEntry>> undo_logs_;    // Undo log per txn
    std::unordered_map<int, std::unordered_map<int, uint64_t>> original_versions_;
};

// ============================================================
// Demonstration: Copy object from foo to bar (Lecture example)
// ============================================================
// X1: copies object foo into object bar
// X2: reads bar's fields
// Expected: X2 should see bar as [0,0] or [9,7], never mixed

void demo_stm_copy_example() {
    std::cout << "=== STM Copy Example (from Lecture 18) ===" << std::endl;
    std::cout << "  X1 copies foo(x=9,y=7) into bar(x=0,y=0)" << std::endl;
    std::cout << "  X2 reads bar's fields" << std::endl;
    std::cout << "  Expected: bar = [0,0] (before X1 commits)" << std::endl;
    std::cout << "           or bar = [9,7] (after X1 commits)" << std::endl;
    std::cout << "           NEVER: bar = [9,0] or [0,7]" << std::endl;
    std::cout << std::endl;

    SoftwareTransactionalMemory stm;

    // Object layout: foo = obj 1 (x), obj 2 (y); bar = obj 3 (x), obj 4 (y)
    stm.init_object(1, 9);  // foo.x
    stm.init_object(2, 7);  // foo.y
    stm.init_object(3, 0);  // bar.x
    stm.init_object(4, 0);  // bar.y

    std::cout << "Initial: foo=(x=" << stm.get_value(1) << ", y=" << stm.get_value(2) << ")" << std::endl;
    std::cout << "         bar=(x=" << stm.get_value(3) << ", y=" << stm.get_value(4) << ")" << std::endl;
    std::cout << std::endl;

    // X1: copy foo → bar
    std::cout << "[Transaction X1: copy foo → bar]" << std::endl;
    stm.init_txn(1);

    int foo_x = stm.stm_read(1, 1); // Read foo.x
    int foo_y = stm.stm_read(1, 2); // Read foo.y
    std::cout << "  X1 read foo: x=" << foo_x << ", y=" << foo_y << std::endl;

    assert(foo_x == 9 && foo_y == 7);

    bool w1 = stm.stm_write(1, 3, foo_x); // Write bar.x = foo.x
    assert(w1);
    bool w2 = stm.stm_write(1, 4, foo_y); // Write bar.y = foo.y
    assert(w2);
    std::cout << "  X1 wrote bar: x=" << foo_x << ", y=" << foo_y << std::endl;

    // Before commit, another thread reads bar (should see old values)
    std::cout << std::endl;
    std::cout << "[Before X1 commit: check bar values]" << std::endl;
    std::cout << "  bar.x=" << stm.get_value(3) << " (eager write - already updated!)" << std::endl;
    std::cout << "  bar.y=" << stm.get_value(4) << " (eager write - already updated!)" << std::endl;

    bool committed = stm.stm_commit(1);
    assert(committed);
    std::cout << std::endl;

    std::cout << "After commit: bar=(x=" << stm.get_value(3) << ", y=" << stm.get_value(4) << ")" << std::endl;
}

// ============================================================
// Demonstration: Conflict and Rollback
// ============================================================
void demo_stm_conflict() {
    std::cout << std::endl;
    std::cout << "=== STM Conflict and Rollback ===" << std::endl;
    std::cout << "  Txn 1 writes obj[10] = 42" << std::endl;
    std::cout << "  Txn 2 tries to write obj[10] = 99 (conflict!)" << std::endl;
    std::cout << std::endl;

    SoftwareTransactionalMemory stm;
    stm.init_object(10, 0);

    // Txn 1 acquires write lock on obj 10
    std::cout << "[Txn 1: write obj[10] = 42]" << std::endl;
    stm.init_txn(1);
    bool w1_ok = stm.stm_write(1, 10, 42);
    std::cout << "  Txn 1 wrote obj[10]=42: " << (w1_ok ? "OK" : "FAIL") << std::endl;
    std::cout << "  obj[10] in memory = " << stm.get_value(10) << " (eager: immediately visible)" << std::endl;

    // Txn 2 tries to write obj 10 → should detect lock conflict
    std::cout << std::endl;
    std::cout << "[Txn 2: try write obj[10] = 99]" << std::endl;
    stm.init_txn(2);
    bool w2_ok = stm.stm_write(2, 10, 99);
    std::cout << "  Txn 2 write obj[10]=99: " << (w2_ok ? "OK" : "FAIL (expected)")
              << std::endl;

    // Txn 1 commits
    std::cout << std::endl;
    stm.stm_commit(1);

    // Now Txn 2 can retry
    std::cout << std::endl;
    std::cout << "[Txn 2 retries after abort]" << std::endl;
    stm.init_txn(2);
    w2_ok = stm.stm_write(2, 10, 99);
    std::cout << "  Txn 2 write obj[10]=99: " << (w2_ok ? "OK" : "FAIL") << std::endl;
    stm.stm_commit(2);

    std::cout << "Final: obj[10] = " << stm.get_value(10) << std::endl;
}

// ============================================================
// Demonstration: Timestamp-based Version Tracking
// ============================================================
void demo_version_tracking() {
    std::cout << std::endl;
    std::cout << "=== Timestamp-Based Version Tracking ===" << std::endl;
    std::cout << std::endl;

    SoftwareTransactionalMemory stm;
    stm.init_object(1, 100);

    std::cout << "Initial global timestamp: " << stm.get_global_ts() << std::endl;

    // Txn 1: read obj 1, write obj 1, commit
    stm.init_txn(1);
    int v = stm.stm_read(1, 1);
    std::cout << "Txn 1 reads obj[1] = " << v << " (local ts=" << stm.get_global_ts() << ")" << std::endl;

    stm.stm_write(1, 1, 200);
    stm.stm_commit(1);
    std::cout << "After Txn 1 commit, global ts = " << stm.get_global_ts() << std::endl;

    // Txn 2: read obj 1 - version check should pass (local ts updated on init)
    stm.init_txn(2);
    v = stm.stm_read(2, 1);
    std::cout << "Txn 2 reads obj[1] = " << v << " (should be 200)" << std::endl;
    assert(v == 200);

    std::cout << "Version tracking and read validation working correctly." << std::endl;
}

int main() {
    std::cout << "=== CS149 Lecture 18: Software Transactional Memory ===" << std::endl;
    std::cout << std::endl;

    std::cout << "Implementation based on Intel McRT STM:" << std::endl;
    std::cout << "  - Eager versioning (undo-log)" << std::endl;
    std::cout << "  - Optimistic reads (validate after read)" << std::endl;
    std::cout << "  - Pessimistic writes (acquire lock before write)" << std::endl;
    std::cout << "  - Timestamp-based version tracking" << std::endl;
    std::cout << "  - Global timestamp incremented by 2 (LSb = write-lock)" << std::endl;
    std::cout << std::endl;

    demo_stm_copy_example();
    demo_stm_conflict();
    demo_version_tracking();

    std::cout << std::endl;
    std::cout << "Summary:" << std::endl;
    std::cout << "  - STM barriers (StmRead/StmWrite) are inserted by compiler" << std::endl;
    std::cout << "  - Transaction descriptor tracks read-set, write-set, undo-log" << std::endl;
    std::cout << "  - Transaction record per object packs lock + version in a word" << std::endl;
    std::cout << "  - STM overhead: 2-8x per thread (why HTM is preferred)" << std::endl;

    return 0;
}
