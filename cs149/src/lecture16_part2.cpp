/*
 * lecture16_part2.cpp - Fine-Grained Locking: Hand-over-Hand Linked List
 * Stanford CS149, Fall 2025 - Lecture 16
 *
 * Demonstrates fine-grained synchronization on a sorted linked list:
 *   1. Single global lock (coarse-grained) - simple but serializes all operations
 *   2. Hand-over-hand locking (fine-grained) - enables parallelism on
 *      different parts of the list
 *
 * Key insight of hand-over-hand locking:
 *   - Lock current node, lock next node, then unlock previous node
 *   - Deadlock-free because locks are always acquired in list traversal order
 *     (no circular wait condition)
 *
 * Compile: g++ -std=c++17 -pthread lecture16_part2.cpp -o lecture16_part2
 * Run: ./lecture16_part2
 */

#include <iostream>
#include <thread>
#include <climits>
#include <mutex>
#include <vector>
#include <cassert>
#include <memory>
#include <functional>

// ============================================================
// Node structure with per-node lock for fine-grained locking
// ============================================================
struct FGLNode {
    int value;
    std::unique_ptr<FGLNode> next;
    std::mutex lock;  // Per-node lock for hand-over-hand locking

    FGLNode(int v) : value(v), next(nullptr) {}
};

// ============================================================
// Fine-Grained Linked List (Hand-over-Hand Locking)
// ============================================================
// Each node has its own mutex. During traversal, threads lock
// the current node, then the next node, then unlock the current.
// This allows concurrent operations on different parts of the list.
//
// Deadlock-free guarantee: locks are always acquired in order
// of list traversal (head -> tail). No circular wait possible.

class FineGrainedList {
public:
    FineGrainedList() {
        // Sentinel head node (value = -1, never removed)
        head_ = std::make_unique<FGLNode>(-1);
        // Sentinel tail node (value = INT_MAX for simplicity)
        head_->next = std::make_unique<FGLNode>(INT_MAX);
    }

    // Insert value while maintaining sorted order.
    // Uses hand-over-hand locking during traversal.
    bool insert(int value) {
        // Step 1: Lock the list head lock (or first real lock)
        head_->lock.lock();

        FGLNode* prev = head_.get();
        FGLNode* cur = prev->next.get();

        // Lock the first real node
        cur->lock.lock();

        // Step 2: Hand-over-hand traversal
        while (cur->value < value) {
            FGLNode* old_prev = prev;
            prev = cur;
            cur = cur->next.get();
            // Lock the new current node
            cur->lock.lock();
            // Release the old previous node's lock
            old_prev->lock.unlock();
        }

        // Step 3: Check for duplicate
        if (cur->value == value) {
            prev->lock.unlock();
            cur->lock.unlock();
            return false; // Duplicate, not inserted
        }

        // Step 4: Create and insert new node
        auto new_node = std::make_unique<FGLNode>(value);
        new_node->next.reset(prev->next.release()); // Take over cur
        // We hold prev->lock, so no other thread can access prev->next
        prev->next.reset(new_node.release());

        // Step 5: Release remaining locks
        prev->lock.unlock();
        cur->lock.unlock();
        return true;
    }

    // Delete a value from the sorted list.
    // Uses hand-over-hand locking during traversal.
    bool remove(int value) {
        head_->lock.lock();

        FGLNode* prev = head_.get();
        FGLNode* cur = prev->next.get();

        if (cur == nullptr) {
            head_->lock.unlock();
            return false;
        }

        cur->lock.lock();

        // Hand-over-hand traversal
        while (cur->value < value) {
            FGLNode* old_prev = prev;
            prev = cur;
            cur = cur->next.get();
            if (cur == nullptr) {
                old_prev->lock.unlock();
                prev->lock.unlock();
                return false;
            }
            cur->lock.lock();
            old_prev->lock.unlock();
        }

        // Found the node to delete?
        if (cur->value == value) {
            // prev->next = cur->next (bypass cur)
            prev->next.reset(cur->next.release());
            prev->lock.unlock();
            cur->lock.unlock();
            // cur is now deleted (unique_ptr releases)
            return true;
        }

        // Not found
        prev->lock.unlock();
        cur->lock.unlock();
        return false;
    }

    // Check if value exists in the list (non-modifying, uses locks for safety)
    bool contains(int value) {
        head_->lock.lock();

        FGLNode* prev = head_.get();
        FGLNode* cur = prev->next.get();
        if (cur) cur->lock.lock();

        while (cur && cur->value < value) {
            FGLNode* old_prev = prev;
            prev = cur;
            cur = cur->next.get();
            if (cur) cur->lock.lock();
            old_prev->lock.unlock();
        }

        bool found = (cur && cur->value == value);

        prev->lock.unlock();
        if (cur) cur->lock.unlock();
        return found;
    }

    // Print the list (not thread-safe, for debugging only)
    void print() const {
        FGLNode* cur = head_->next.get();
        std::cout << "List: ";
        while (cur) {
            std::cout << cur->value << " ";
            cur = cur->next.get();
        }
        std::cout << std::endl;
    }

private:
    std::unique_ptr<FGLNode> head_;
};

// ============================================================
// Coarse-Grained Linked List (Single Global Lock)
// ============================================================
// For comparison: the simplest safe implementation.
// All operations serialized by a single mutex.

class CoarseGrainedList {
public:
    bool insert(int value) {
        std::lock_guard<std::mutex> guard(lock_);
        return insert_unsafe(value);
    }

    bool remove(int value) {
        std::lock_guard<std::mutex> guard(lock_);
        return remove_unsafe(value);
    }

    bool contains(int value) {
        std::lock_guard<std::mutex> guard(lock_);
        return contains_unsafe(value);
    }

private:
    struct Node {
        int value;
        std::unique_ptr<Node> next;
        Node(int v) : value(v), next(nullptr) {}
    };

    std::unique_ptr<Node> head_ = std::make_unique<Node>(-1);
    std::mutex lock_;

    bool insert_unsafe(int value) {
        Node* prev = head_.get();
        while (prev->next && prev->next->value < value) {
            prev = prev->next.get();
        }
        if (prev->next && prev->next->value == value) return false;
        auto new_node = std::make_unique<Node>(value);
        new_node->next.reset(prev->next.release());
        prev->next.reset(new_node.release());
        return true;
    }

    bool remove_unsafe(int value) {
        Node* prev = head_.get();
        while (prev->next && prev->next->value < value) {
            prev = prev->next.get();
        }
        if (prev->next && prev->next->value == value) {
            prev->next.reset(prev->next->next.release());
            return true;
        }
        return false;
    }

    bool contains_unsafe(int value) {
        Node* cur = head_->next.get();
        while (cur) {
            if (cur->value == value) return true;
            cur = cur->next.get();
        }
        return false;
    }
};

// ============================================================
// Demonstration: Concurrent Insertions
// ============================================================

void concurrent_inserts(FineGrainedList& list, int thread_id, int count) {
    // Each thread inserts values from a different range to minimize
    // contention and demonstrate parallel insertions on different
    // parts of the list.
    int base = thread_id * 1000;
    for (int i = 0; i < count; ++i) {
        list.insert(base + i);
    }
}

void concurrent_mixed_ops(FineGrainedList& list, int thread_id, int count) {
    int base = thread_id * 1000;
    for (int i = 0; i < count; ++i) {
        int val = base + i;
        // All threads do insert, then contains, then remove
        list.insert(val);
        bool found = list.contains(val);
        assert(found && "Value should be found after insert");
        bool removed = list.remove(val);
        assert(removed && "Value should be removable");
    }
}

int main() {
    std::cout << "=== CS149 Lecture 16: Fine-Grained Locking ===" << std::endl;
    std::cout << std::endl;

    // ---- Demo 1: Fine-grained insert operations ----
    std::cout << "--- Demo 1: Concurrent Insertions (Fine-Grained) ---" << std::endl;
    {
        FineGrainedList list;
        const int num_threads = 4;
        const int inserts_per_thread = 500;
        std::vector<std::thread> threads;

        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back(concurrent_inserts, std::ref(list), i, inserts_per_thread);
        }
        for (auto& t : threads) t.join();

        // Verify: each thread inserted values [i*1000, i*1000+500)
        std::cout << "Inserted " << (num_threads * inserts_per_thread) << " values." << std::endl;

        // Spot-check some values
        for (int i = 0; i < num_threads; ++i) {
            assert(list.contains(i * 1000) && "First insert of each thread must exist");
            assert(list.contains(i * 1000 + inserts_per_thread - 1) &&
                   "Last insert of each thread must exist");
        }
        std::cout << "All spot-checks passed!" << std::endl;
    }

    // ---- Demo 2: Fine-grained mixed operations ----
    std::cout << std::endl;
    std::cout << "--- Demo 2: Mixed Insert/Contains/Remove (Fine-Grained) ---" << std::endl;
    {
        FineGrainedList list;
        const int num_threads = 4;
        const int ops_per_thread = 500;
        std::vector<std::thread> threads;

        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back(concurrent_mixed_ops, std::ref(list), i, ops_per_thread);
        }
        for (auto& t : threads) t.join();

        // After all threads finish, the list should be empty (all removed)
        for (int i = 0; i < num_threads; ++i) {
            assert(!list.contains(i * 1000) && "Values should have been removed");
        }
        std::cout << "All values correctly removed. Fine-grained locking works!" << std::endl;
    }

    // ---- Demo 3: Coarse-grained comparison ----
    std::cout << std::endl;
    std::cout << "--- Demo 3: Coarse-Grained List (Correctness Check) ---" << std::endl;
    {
        CoarseGrainedList list;
        list.insert(5);
        list.insert(3);
        list.insert(7);
        list.insert(1);

        assert(list.contains(5) && "5 should be in list");
        assert(list.contains(3) && "3 should be in list");
        assert(!list.contains(99) && "99 should not be in list");

        list.remove(3);
        assert(!list.contains(3) && "3 should be removed");

        std::cout << "Coarse-grained list operations correct." << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Summary:" << std::endl;
    std::cout << "  - Fine-grained (hand-over-hand) locking enables concurrent" << std::endl;
    std::cout << "    operations on different parts of the linked list." << std::endl;
    std::cout << "  - Deadlock is avoided because locks are always acquired" << std::endl;
    std::cout << "    in list traversal order (no circular wait)." << std::endl;
    std::cout << "  - Trade-off: more lock/unlock overhead vs. better parallelism." << std::endl;

    return 0;
}
