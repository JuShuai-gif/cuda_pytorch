/*
 * lecture17_part3.cpp - TM Conflict Detection: Pessimistic vs Optimistic
 * Stanford CS149, Fall 2025 - Lecture 17
 *
 * Demonstrates the two conflict detection strategies for TM:
 *   1. Pessimistic (eager) detection:
 *      - Check for conflicts on EVERY load and store
 *      - On conflict: stall or abort immediately
 *      - Good: detect conflicts early, avoid wasted work
 *      - Bad: per-operation overhead, no forward progress guarantee
 *
 *   2. Optimistic (lazy) detection:
 *      - Check for conflicts only at commit time
 *      - On conflict: abort the non-committing transaction
 *      - Good: no per-operation overhead, forward progress guarantee
 *      - Bad: may waste work on doomed transactions
 *
 * Compile: g++ -std=c++17 -pthread lecture17_part3.cpp -o lecture17_part3
 * Run: ./lecture17_part3
 */

#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <cassert>
#include <chrono>
#include <unordered_map>
#include <unordered_set>

// ============================================================
// Shared transactional memory with conflict tracking
// ============================================================
// Uses per-address metadata to track which transaction "owns"
// each address for read or write.

struct TxRecord {
    int writer_id = -1;          // ID of transaction currently writing (-1 = none)
    std::unordered_set<int> reader_ids; // IDs of transactions currently reading
    std::mutex mtx;              // Protect this record
};

class TransactionalMemory {
public:
    TransactionalMemory() : global_version_(0) {}

    // Transaction descriptor
    struct TxnDesc {
        int id;
        std::unordered_set<int> read_set;  // Addresses read
        std::unordered_set<int> write_set; // Addresses written
        bool active = true;
    };

    // ---- Pessimistic (Eager) Conflict Detection ----

    // Read with pessimistic detection
    // Checks for write-write or write-read conflicts at read time
    bool read_pessimistic(TxnDesc& txn, int addr, int& out_value) {
        auto& record = records_[addr];
        std::lock_guard<std::mutex> guard(record.mtx);

        // Conflict: another transaction is writing to this address
        if (record.writer_id != -1 && record.writer_id != txn.id) {
            std::cout << "  [Pessimistic] Conflict on READ: Txn " << txn.id
                      << " reads addr[" << addr << "], but Txn " << record.writer_id
                      << " is writing it. Aborting!" << std::endl;
            return false; // Conflict detected → abort
        }

        // No conflict: register as reader
        record.reader_ids.insert(txn.id);
        txn.read_set.insert(addr);
        out_value = memory_[addr];
        return true;
    }

    // Write with pessimistic detection
    // Checks for any conflict (R-W, W-R, W-W) at write time
    bool write_pessimistic(TxnDesc& txn, int addr, int value) {
        auto& record = records_[addr];
        std::lock_guard<std::mutex> guard(record.mtx);

        // Conflict: another writer holds this address
        if (record.writer_id != -1 && record.writer_id != txn.id) {
            std::cout << "  [Pessimistic] Conflict on WRITE: Txn " << txn.id
                      << " writes addr[" << addr << "], but Txn " << record.writer_id
                      << " already owns it. Aborting!" << std::endl;
            return false;
        }

        // Conflict: other readers exist
        if (!record.reader_ids.empty()) {
            bool has_other_readers = false;
            for (int rid : record.reader_ids) {
                if (rid != txn.id) {
                    has_other_readers = true;
                    break;
                }
            }
            if (has_other_readers) {
                std::cout << "  [Pessimistic] Conflict on WRITE: Txn " << txn.id
                          << " writes addr[" << addr << "], but other txns are reading it."
                          << " Aborting!" << std::endl;
                return false;
            }
        }

        // Acquire write ownership
        record.writer_id = txn.id;
        record.reader_ids.clear(); // Writer takes exclusive access
        txn.write_set.insert(addr);
        return true; // No conflict at detection time
    }

    // Pessimistic commit
    bool commit_pessimistic(TxnDesc& txn) {
        // Final conflict check (should already be clean with pessimistic detection)
        for (int addr : txn.write_set) {
            auto& record = records_[addr];
            std::lock_guard<std::mutex> guard(record.mtx);
            if (record.writer_id != txn.id) {
                std::cout << "  [Pessimistic] Commit conflict on addr[" << addr << "]!"
                          << std::endl;
                return false;
            }
        }
        // Release locks and update memory
        for (int addr : txn.write_set) {
            // (memory update happens in the actual write path for eager versioning)
            auto& record = records_[addr];
            std::lock_guard<std::mutex> guard(record.mtx);
            record.writer_id = -1;
            record.reader_ids.clear();
        }
        return true;
    }

    // ---- Optimistic (Lazy) Conflict Detection ----

    // Read with optimistic detection (just record, no conflict check)
    bool read_optimistic(TxnDesc& txn, int addr, int& out_value) {
        txn.read_set.insert(addr);
        out_value = memory_[addr];

        // Record this read in the TxRecord (for conflict detection at commit)
        auto& record = records_[addr];
        std::lock_guard<std::mutex> guard(record.mtx);
        record.reader_ids.insert(txn.id);
        return true; // Optimistic: always succeeds at read time
    }

    // Write with optimistic detection (just record, no conflict check)
    bool write_optimistic(TxnDesc& txn, int addr, int value) {
        txn.write_set.insert(addr);

        auto& record = records_[addr];
        std::lock_guard<std::mutex> guard(record.mtx);
        // Conflict: another writer?
        if (record.writer_id != -1 && record.writer_id != txn.id) {
            return false; // Write-write conflict with active writer (pessimistic on writes)
        }
        record.writer_id = txn.id;
        return true;
    }

    // Optimistic commit: check ALL conflicts at commit time
    bool commit_optimistic(TxnDesc& txn) {
        // Check 1: For each write, no other transaction has read or written it
        for (int addr : txn.write_set) {
            auto& record = records_[addr];
            std::lock_guard<std::mutex> guard(record.mtx);

            // A committed transaction may have already modified this
            // In a real STM, we'd use version numbers/timestamps
            if (record.writer_id != txn.id) {
                std::cout << "  [Optimistic] Commit FAIL: addr[" << addr
                          << "] modified by another txn." << std::endl;
                // Clean up optimistic reads
                for (int r_addr : txn.read_set) {
                    auto& r_rec = records_[r_addr];
                    std::lock_guard<std::mutex> r_guard(r_rec.mtx);
                    r_rec.reader_ids.erase(txn.id);
                }
                return false;
            }
        }

        // Check 2: For each read, no other transaction has written it
        for (int addr : txn.read_set) {
            auto& record = records_[addr];
            std::lock_guard<std::mutex> guard(record.mtx);
            // If the data was read but is now writer-locked by another...
            if (record.writer_id != -1 && record.writer_id != txn.id) {
                std::cout << "  [Optimistic] Commit FAIL: addr[" << addr
                          << "] was read but another txn wrote it." << std::endl;
                // Clean up
                for (int r_addr : txn.read_set) {
                    auto& r_rec = records_[r_addr];
                    std::lock_guard<std::mutex> r_guard(r_rec.mtx);
                    r_rec.reader_ids.erase(txn.id);
                }
                return false;
            }
        }

        // Commit succeeds: make writes visible, release locks
        for (int addr : txn.write_set) {
            auto& record = records_[addr];
            std::lock_guard<std::mutex> guard(record.mtx);
            memory_[addr] = write_buffer_[addr];
            record.writer_id = -1;
            record.reader_ids.clear();
        }
        for (int addr : txn.read_set) {
            auto& record = records_[addr];
            std::lock_guard<std::mutex> guard(record.mtx);
            record.reader_ids.erase(txn.id);
        }

        std::cout << "  [Optimistic] Commit SUCCESS!" << std::endl;
        return true;
    }

    // Store a value for optimistic write buffering
    void buffer_write(int addr, int value) {
        write_buffer_[addr] = value;
    }

    // Direct memory access for non-transactional or eager writes
    void direct_write(int addr, int value) {
        memory_[addr] = value;
    }

    int direct_read(int addr) const {
        auto it = memory_.find(addr);
        return (it != memory_.end()) ? it->second : 0;
    }

    void reset() {
        memory_.clear();
        write_buffer_.clear();
        records_.clear();
    }

private:
    std::unordered_map<int, int> memory_;          // Actual shared memory
    std::unordered_map<int, int> write_buffer_;    // Write buffer for optimistic
    std::unordered_map<int, TxRecord> records_;    // Per-address metadata
    std::atomic<uint64_t> global_version_;
};

// ============================================================
// Demonstration
// ============================================================

int main() {
    std::cout << "=== CS149 Lecture 17: TM Conflict Detection ===" << std::endl;
    std::cout << std::endl;

    // ---- Scenario: Two transactions, one reads+write A, another writes B ----
    // This should succeed with BOTH approaches (no real conflict)
    std::cout << "--- Scenario 1: No Real Conflict ---" << std::endl;
    std::cout << "  Txn 0: reads A, writes A=42" << std::endl;
    std::cout << "  Txn 1: writes B=99" << std::endl;
    std::cout << "  Expected: both commit (A and B are different addresses)" << std::endl;
    std::cout << std::endl;

    {
        TransactionalMemory tm;
        tm.direct_write(0, 10); // Initialize A=10 at addr 0
        tm.direct_write(1, 20); // Initialize B=20 at addr 1

        // Pessimistic approach
        {
            std::cout << "[Pessimistic Detection]" << std::endl;
            TransactionalMemory::TxnDesc txn0{0};
            TransactionalMemory::TxnDesc txn1{1};

            int val_a;
            bool r_ok = tm.read_pessimistic(txn0, 0, val_a);
            assert(r_ok);
            std::cout << "  Txn 0 read A=" << val_a << std::endl;

            bool w_ok0 = tm.write_pessimistic(txn0, 0, 42);
            assert(w_ok0);
            tm.direct_write(0, 42);
            std::cout << "  Txn 0 wrote A=42" << std::endl;

            bool w_ok1 = tm.write_pessimistic(txn1, 1, 99);
            assert(w_ok1);
            tm.direct_write(1, 99);
            std::cout << "  Txn 1 wrote B=99" << std::endl;

            tm.commit_pessimistic(txn0);
            tm.commit_pessimistic(txn1);
            std::cout << "  Result: A=" << tm.direct_read(0) << " B=" << tm.direct_read(1) << std::endl;
        }

        tm.reset();
        tm.direct_write(0, 10);
        tm.direct_write(1, 20);

        // Optimistic approach
        {
            std::cout << "[Optimistic Detection]" << std::endl;
            TransactionalMemory::TxnDesc txn0{0};
            TransactionalMemory::TxnDesc txn1{1};

            int val_a;
            tm.read_optimistic(txn0, 0, val_a);
            std::cout << "  Txn 0 read A=" << val_a << std::endl;

            tm.write_optimistic(txn0, 0, 42);
            tm.buffer_write(0, 42);
            std::cout << "  Txn 0 buffer-wrote A=42" << std::endl;

            tm.write_optimistic(txn1, 1, 99);
            tm.buffer_write(1, 99);
            std::cout << "  Txn 1 buffer-wrote B=99" << std::endl;

            tm.commit_optimistic(txn0);
            tm.commit_optimistic(txn1);
            std::cout << "  Result: A=" << tm.direct_read(0) << " B=" << tm.direct_read(1) << std::endl;
        }
    }

    // ---- Scenario 2: Write-Write Conflict ----
    std::cout << std::endl;
    std::cout << "--- Scenario 2: Write-Write Conflict on A ---" << std::endl;
    std::cout << "  Txn 0: writes A=42" << std::endl;
    std::cout << "  Txn 1: writes A=99 (conflict with Txn 0!)" << std::endl;
    std::cout << std::endl;

    {
        TransactionalMemory tm;
        tm.direct_write(0, 10);

        // Pessimistic: detects conflict IMMEDIATELY on the second write
        {
            std::cout << "[Pessimistic Detection]" << std::endl;
            TransactionalMemory::TxnDesc txn0{0};
            TransactionalMemory::TxnDesc txn1{1};

            bool w0 = tm.write_pessimistic(txn0, 0, 42);
            std::cout << "  Txn 0 write A=42: " << (w0 ? "OK" : "CONFLICT") << std::endl;

            bool w1 = tm.write_pessimistic(txn1, 0, 99);
            std::cout << "  Txn 1 write A=99: " << (w1 ? "OK" : "CONFLICT (detected immediately!)")
                      << std::endl;
            std::cout << "  → Pessimistic detected the conflict at write time!" << std::endl;
        }
    }

    {
        TransactionalMemory tm;
        tm.direct_write(0, 10);

        // Optimistic: detects conflict at COMMIT time
        {
            std::cout << "[Optimistic Detection]" << std::endl;
            TransactionalMemory::TxnDesc txn0{0};
            TransactionalMemory::TxnDesc txn1{1};

            bool w0 = tm.write_optimistic(txn0, 0, 42);
            tm.buffer_write(0, 42);
            std::cout << "  Txn 0 write A=42: " << (w0 ? "OK (optimistic)" : "CONFLICT") << std::endl;

            // Txn 1 tries to write A: should fail even in optimistic (writer conflict)
            bool w1 = tm.write_optimistic(txn1, 0, 99);
            std::cout << "  Txn 1 write A=99: " << (w1 ? "OK" : "CONFLICT (writer already active)")
                      << std::endl;
        }
    }

    // ---- Scenario 3: Read-Write Conflict ----
    std::cout << std::endl;
    std::cout << "--- Scenario 3: Read-Write Conflict ---" << std::endl;
    std::cout << "  Txn 0: reads A" << std::endl;
    std::cout << "  Txn 1: writes A=99" << std::endl;
    std::cout << "  Pessimistic: Txn 1's write detects readers → abort" << std::endl;
    std::cout << "  Optimistic: Txn 0 reads, Txn 1 writes; at commit, " << std::endl;
    std::cout << "    if Txn 0 commits first → OK; if Txn 1 commits first," << std::endl;
    std::cout << "    Txn 0's read set is stale → abort Txn 0" << std::endl;
    std::cout << std::endl;

    // ---- Summary ----
    std::cout << "=== Comparison Summary ===" << std::endl;
    std::cout << "┌─────────────────────┬──────────────────────────┬──────────────────────────┐" << std::endl;
    std::cout << "│      Aspect         │  Pessimistic (Eager)     │  Optimistic (Lazy)       │" << std::endl;
    std::cout << "├─────────────────────┼──────────────────────────┼──────────────────────────┤" << std::endl;
    std::cout << "│ When to check       │ Every load/store         │ At commit time only      │" << std::endl;
    std::cout << "│ Early abort?        │ Yes (less wasted work)   │ No (may waste work)      │" << std::endl;
    std::cout << "│ Forward progress    │ No guarantee (livelock)  │ Guaranteed               │" << std::endl;
    std::cout << "│ Communication       │ Fine-grained (per op)    │ Bulk (at commit)         │" << std::endl;
    std::cout << "│ Overhead            │ Per-operation overhead   │ Low until commit         │" << std::endl;
    std::cout << "│ Stalling possible?  │ Yes (stall instead of    │ No (always abort on      │" << std::endl;
    std::cout << "│                     │  abort in some cases)    │  conflict at commit)     │" << std::endl;
    std::cout << "└─────────────────────┴──────────────────────────┴──────────────────────────┘" << std::endl;

    return 0;
}
