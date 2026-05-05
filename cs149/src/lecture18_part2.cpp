/*
 * lecture18_part2.cpp - STM Compiler Optimizations: Barrier Decomposition
 * Stanford CS149, Fall 2025 - Lecture 18
 *
 * Demonstrates the STM compiler optimization techniques from the lecture:
 *
 * Problem: Monolithic STM barriers (tmTxnBegin/tmTxnCommit) hide
 *   redundant logging and locking from the compiler.
 *
 * Optimization: Decompose barriers into fine-grained operations:
 *   - txnOpenForWrite(obj): acquire write lock on object (once)
 *   - txnLogObjectInt(&field, obj): save undo-log entry
 *   - txnOpenForRead(obj): register object in read-set (once)
 *
 * With decomposed barriers, the compiler can:
 *   1. Remove redundant OpenForWrite calls (open once, write many)
 *   2. Remove redundant OpenForRead calls (open once, read many)
 *   3. Hoist barrier calls out of loops
 *   4. Merge consecutive undo-log entries on same object
 *
 * Result: <40% overhead over sequential, <30% over lock-based.
 *
 * Compile: g++ -std=c++17 -pthread lecture18_part2.cpp -o lecture18_part2
 * Run: ./lecture18_part2
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
// Optimized STM System with Decomposed Barriers
// ============================================================
// Instead of monolithic barrier calls, we decompose into:
//   openForWrite → logField → writeField → commit

class OptimizedSTM {
public:
    // ---- Decomposed STM barriers ----

    // Open an object for write (acquire lock once, may be called multiple
    // times but the compiler can eliminate duplicate calls)
    bool openForWrite(int txn_id, int obj_id) {
        // If already opened for write, skip (redundant call elimination)
        if (write_opened_[txn_id].count(obj_id)) {
            return true; // Already opened
        }

        uint64_t& rec = records_[obj_id];
        if (is_locked_by_other(rec, txn_id)) {
            std::cout << "  [OpenForWrite] Txn " << txn_id << " obj[" << obj_id
                      << "] locked by another. Conflict!" << std::endl;
            return false;
        }

        // Save original version for rollback
        if (!write_opened_[txn_id].count(obj_id)) {
            original_versions_[txn_id][obj_id] = get_obj_version(rec);
        }

        // Acquire write lock
        rec = make_locked(txn_id);
        write_opened_[txn_id].insert(obj_id);
        return true;
    }

    // Log a field for undo (save old value before modifying)
    void logField(int txn_id, int obj_id, const std::string& field_name, int old_value) {
        undo_logs_[txn_id].push_back({obj_id, field_name, old_value});
    }

    // Write a field (eager: directly to memory, but we simulate here)
    void writeField(int obj_id, int value) {
        memory_[obj_id] = value;
    }

    // Open an object for read (register in read-set once)
    bool openForRead(int txn_id, int obj_id) {
        // If already opened, skip
        if (read_opened_[txn_id].count(obj_id)) {
            return true;
        }

        uint64_t rec = records_[obj_id];
        if (is_locked_by_other(rec, txn_id)) {
            return false; // Conflict
        }

        read_opened_[txn_id].insert(obj_id);
        return true;
    }

    // Read a field value
    int readField(int obj_id) {
        auto it = memory_.find(obj_id);
        return (it != memory_.end()) ? it->second : 0;
    }

    // Commit: validate read-set, release locks
    bool commit(int txn_id) {
        // Validate read-set
        for (int obj_id : read_opened_[txn_id]) {
            uint64_t rec = records_[obj_id];
            if (is_locked_by_other(rec, txn_id)) {
                std::cout << "  [Commit] Txn " << txn_id << " FAIL: read-set conflict on obj["
                          << obj_id << "]" << std::endl;
                rollback(txn_id);
                return false;
            }
        }

        // Release write locks and stamp with new version
        uint64_t new_version = global_version_.fetch_add(1) + 1;
        for (int obj_id : write_opened_[txn_id]) {
            records_[obj_id] = make_unlocked(new_version);
        }

        std::cout << "  [Commit] Txn " << txn_id << " committed. New version = "
                  << new_version << std::endl;

        // Clean up
        read_opened_[txn_id].clear();
        write_opened_[txn_id].clear();
        undo_logs_[txn_id].clear();
        original_versions_[txn_id].clear();
        return true;
    }

    // Rollback (replay undo-log)
    void rollback(int txn_id) {
        auto& log = undo_logs_[txn_id];
        for (auto it = log.rbegin(); it != log.rend(); ++it) {
            memory_[it->obj_id] = it->old_value;
        }
        // Release locks, restore versions
        for (int obj_id : write_opened_[txn_id]) {
            records_[obj_id] = make_unlocked(original_versions_[txn_id][obj_id]);
        }
        read_opened_[txn_id].clear();
        write_opened_[txn_id].clear();
        undo_logs_[txn_id].clear();
        original_versions_[txn_id].clear();
    }

    void init_object(int obj_id, int value) {
        memory_[obj_id] = value;
        records_[obj_id] = make_unlocked(0);
    }

    int get_value(int obj_id) const {
        auto it = memory_.find(obj_id);
        return (it != memory_.end()) ? it->second : 0;
    }

private:
    struct UndoEntry {
        int obj_id;
        std::string field_name;
        int old_value;
    };

    bool is_locked_by_other(uint64_t rec, int txn_id) {
        if ((rec & 1ULL) == 0) { // locked
            uint64_t owner = rec >> 1;
            return owner != static_cast<uint64_t>(txn_id + 1);
        }
        return false;
    }

    uint64_t get_obj_version(uint64_t rec) {
        return rec >> 1;
    }

    static uint64_t make_locked(int owner_id) {
        return (static_cast<uint64_t>(owner_id + 1) << 1);
    }

    static uint64_t make_unlocked(uint64_t version) {
        return (version << 1) | 1ULL;
    }

    std::unordered_map<int, int> memory_;
    std::unordered_map<int, uint64_t> records_;
    std::atomic<uint64_t> global_version_{0};

    std::unordered_map<int, std::unordered_set<int>> read_opened_;
    std::unordered_map<int, std::unordered_set<int>> write_opened_;
    std::unordered_map<int, std::vector<UndoEntry>> undo_logs_;
    std::unordered_map<int, std::unordered_map<int, uint64_t>> original_versions_;
};

// ============================================================
// Before Optimization: Monolithic Barrier Version
// ============================================================
// Simulates the naive STM instrumentation where every memory
// access triggers a full barrier call.

class MonolithicSTM {
public:
    bool txnBegin(int txn_id) {
        active_txns_[txn_id] = true;
        return true;
    }

    void txnWrite(int txn_id, int obj_id, int& field, int value) {
        // Monolithic: does openForWrite + logField + write in one call
        // No opportunity for the compiler to eliminate redundancies
        barrier_count_++;
        field = value;
    }

    int txnRead(int txn_id, int obj_id, int field) {
        barrier_count_++;
        return field;
    }

    bool txnCommit(int txn_id) {
        barrier_count_++;
        active_txns_[txn_id] = false;
        return true;
    }

    int get_barrier_count() const { return barrier_count_; }

private:
    std::unordered_map<int, bool> active_txns_;
    int barrier_count_ = 0;
};

// ============================================================
// After Optimization: Decomposed Barrier Version
// ============================================================
// Shows how the compiler can optimize when barriers are decomposed.

void demo_barrier_decomposition() {
    std::cout << "=== STM Barrier Decomposition Optimization ===" << std::endl;
    std::cout << std::endl;

    // Sample code we want to make transactional:
    // atomic {
    //     a.x = t1;
    //     a.y = t2;
    //     if (a.z == 0) {
    //         a.x = 0;
    //         a.z = t3;
    //     }
    // }
    //
    // Object layout: obj_a.x=1, obj_a.y=2, obj_a.z=3
    const int OBJ_A_X = 1;
    const int OBJ_A_Y = 2;
    const int OBJ_A_Z = 3;

    // ======== Monolithic Barriers (Before) ========
    std::cout << "--- Monolithic Barriers (Naive Instrumentation) ---" << std::endl;
    {
        MonolithicSTM stm;
        int t1 = 10, t2 = 20, t3 = 30;
        int a_x = 0, a_y = 0, a_z = 0;

        stm.txnBegin(0);

        // Every access is wrapped in a monolithic barrier call
        stm.txnWrite(0, OBJ_A_X, a_x, t1);  // barrier #1
        stm.txnWrite(0, OBJ_A_Y, a_y, t2);  // barrier #2
        if (stm.txnRead(0, OBJ_A_Z, a_z) == 0) {  // barrier #3
            stm.txnWrite(0, OBJ_A_X, a_x, 0);  // barrier #4
            stm.txnWrite(0, OBJ_A_Z, a_z, t3);  // barrier #5
        }

        stm.txnCommit(0);  // barrier #6
        std::cout << "Total barrier calls: " << stm.get_barrier_count()
                  << " (monolithic: every access = one barrier)" << std::endl;
    }

    // ======== Decomposed Barriers (After) ========
    std::cout << std::endl;
    std::cout << "--- Decomposed Barriers (After Compiler Optimization) ---" << std::endl;
    {
        OptimizedSTM stm;
        stm.init_object(OBJ_A_X, 0);
        stm.init_object(OBJ_A_Y, 0);
        stm.init_object(OBJ_A_Z, 0);

        int txn_id = 0;
        int t1 = 10, t2 = 20, t3 = 30;

        // Compiler optimization: hoist OpenForWrite above multiple writes
        // Open obj_a for write ONCE (not once per field)
        bool ok = stm.openForWrite(txn_id, 1); // obj_a base - open once!
        assert(ok);
        std::cout << "OpenForWrite(obj_a): called ONCE for entire object" << std::endl;

        // Log and write a.x = t1
        stm.logField(txn_id, OBJ_A_X, "x", stm.get_value(OBJ_A_X));
        stm.writeField(OBJ_A_X, t1);

        // Write a.y = t2 (no need to reopen - compiler eliminated redundant open)
        stm.logField(txn_id, OBJ_A_Y, "y", stm.get_value(OBJ_A_Y));
        stm.writeField(OBJ_A_Y, t2);

        // Read a.z
        bool rd_ok = stm.openForRead(txn_id, OBJ_A_Z);
        assert(rd_ok);
        int a_z = stm.readField(OBJ_A_Z);

        std::cout << "Read a.z = " << a_z << " (OpenForRead called ONCE)" << std::endl;

        if (a_z == 0) {
            // Write a.x = 0 (already opened for write - no redundant open!)
            stm.logField(txn_id, OBJ_A_X, "x", t1);
            stm.writeField(OBJ_A_X, 0);

            // Write a.z = t3 (need to open - but compiler can merge with write open above)
            stm.openForWrite(txn_id, OBJ_A_Z);
            stm.logField(txn_id, OBJ_A_Z, "z", a_z);
            stm.writeField(OBJ_A_Z, t3);
        }

        stm.commit(txn_id);

        std::cout << std::endl;
        std::cout << "Result: a.x=" << stm.get_value(OBJ_A_X)
                  << " a.y=" << stm.get_value(OBJ_A_Y)
                  << " a.z=" << stm.get_value(OBJ_A_Z) << std::endl;

        int barrier_calls = 1 + 2 + 1 + (a_z == 0 ? 1 : 0) + 1;
        std::cout << "Total decomposed barrier calls: ~" << barrier_calls
                  << " (vs 6 monolithic)" << std::endl;
    }
}

// ============================================================
// Optimization: Eliminating Redundant OpenForRead/OpenForWrite
// ============================================================
void demo_redundant_elimination() {
    std::cout << std::endl;
    std::cout << "=== Redundant Open Elimination ===" << std::endl;
    std::cout << std::endl;

    OptimizedSTM stm;
    stm.init_object(1, 0);
    stm.init_object(2, 0);

    std::cout << "The optimizer detects that multiple field accesses" << std::endl;
    std::cout << "to the same object don't need repeated OpenForWrite/OpenForRead:" << std::endl;
    std::cout << std::endl;

    int txn_id = 0;

    // First OpenForWrite on obj 1 - real call
    bool ok1 = stm.openForWrite(txn_id, 1);
    std::cout << "OpenForWrite(txn, obj1): " << (ok1 ? "called (first time)" : "FAIL") << std::endl;

    // Second OpenForWrite on obj 1 - should be eliminated by optimizer
    bool ok2 = stm.openForWrite(txn_id, 1);
    std::cout << "OpenForWrite(txn, obj1): " << (ok2 ? "no-op (already opened)" : "FAIL")
              << " ← compiler eliminates this!" << std::endl;

    stm.logField(txn_id, 1, "field", 0);
    stm.writeField(1, 42);
    stm.logField(txn_id, 1, "field2", 42);
    stm.writeField(1, 99);

    std::cout << "Wrote obj1 = 99 (two writes, one OpenForWrite)" << std::endl;

    // First OpenForRead on obj 2 - real call
    bool rd1 = stm.openForRead(txn_id, 2);
    std::cout << "OpenForRead(txn, obj2): " << (rd1 ? "called (first time)" : "FAIL") << std::endl;

    // Second OpenForRead on obj 2 - eliminated
    bool rd2 = stm.openForRead(txn_id, 2);
    std::cout << "OpenForRead(txn, obj2): " << (rd2 ? "no-op (already opened)" : "FAIL")
              << " ← compiler eliminates this!" << std::endl;

    stm.commit(txn_id);

    std::cout << std::endl;
    std::cout << "This is how the compiler reduces per-thread overhead" << std::endl;
    std::cout << "from 2-8x down to <40% over sequential execution." << std::endl;
}

int main() {
    std::cout << "=== CS149 Lecture 18: STM Compiler Optimizations ===" << std::endl;
    std::cout << std::endl;

    demo_barrier_decomposition();
    demo_redundant_elimination();

    std::cout << std::endl;
    std::cout << "Summary of STM Optimization Techniques:" << std::endl;
    std::cout << "  1. Decompose monolithic barriers (tmWr/tmRd) into" << std::endl;
    std::cout << "     fine-grained primitives (OpenForWrite, LogField, OpenForRead)" << std::endl;
    std::cout << "  2. Eliminate redundant OpenForWrite calls: open once per" << std::endl;
    std::cout << "     object, not once per field write" << std::endl;
    std::cout << "  3. Eliminate redundant OpenForRead calls: open once per" << std::endl;
    std::cout << "     object, not once per field read" << std::endl;
    std::cout << "  4. Hoist barrier calls out of loops" << std::endl;
    std::cout << "  5. Merge consecutive undo-log entries on same object" << std::endl;
    std::cout << std::endl;
    std::cout << "Result: <40% overhead over sequential, <30% over lock-based." << std::endl;

    return 0;
}
