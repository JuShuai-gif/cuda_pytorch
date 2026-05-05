/*
 * lecture16_part4.cpp - Lock-Free Queues
 * Stanford CS149, Fall 2025 - Lecture 16
 *
 * Demonstrates lock-free queue designs from the lecture:
 *   1. Single Reader, Single Writer Bounded Queue (ring buffer)
 *   2. Single Reader, Single Writer Unbounded Queue (linked-list based)
 *
 * Key insight: These queues are lock-free because there is exactly
 * ONE producer and ONE consumer. Head is only modified by consumer,
 * tail is only modified by producer. No CAS or locks needed!
 * Threads never block waiting for each other.
 *
 * Assumption: sequentially consistent memory system (or proper
 * memory fences / C++11 atomic<>).
 *
 * Compile: g++ -std=c++17 -pthread lecture16_part4.cpp -o lecture16_part4
 * Run: ./lecture16_part4
 */

#include <iostream>
#include <thread>
#include <atomic>
#include <vector>
#include <cassert>
#include <memory>
#include <chrono>

// ============================================================
// Part 1: Single Reader, Single Writer Bounded Queue
// ============================================================
// Fixed-size ring buffer. Producer writes to tail, consumer
// reads from head. No locks because head and tail are each
// modified by only one thread.
//
// Full condition: tail == MOD_N(head - 1) (one slot wasted)
// Empty condition: head == tail

template <typename T, size_t N>
class BoundedSPSCQueue {
public:
    BoundedSPSCQueue() : head_(0), tail_(0) {}

    // Producer: return false if queue is full
    bool push(T value) {
        size_t current_tail = tail_.load(std::memory_order_relaxed);
        size_t next_tail = (current_tail + 1) % N;

        // Queue is full if next_tail would catch up to head
        if (next_tail == head_.load(std::memory_order_acquire)) {
            return false; // Full
        }

        data_[current_tail] = value;
        tail_.store(next_tail, std::memory_order_release);
        return true;
    }

    // Consumer: return false if queue is empty
    bool pop(T& value) {
        size_t current_head = head_.load(std::memory_order_relaxed);

        // Queue is empty if head == tail
        if (current_head == tail_.load(std::memory_order_acquire)) {
            return false; // Empty
        }

        value = data_[current_head];
        head_.store((current_head + 1) % N, std::memory_order_release);
        return true;
    }

private:
    T data_[N];
    // pad to avoid false sharing (head and tail on different cache lines)
    alignas(64) std::atomic<size_t> head_;
    alignas(64) std::atomic<size_t> tail_;
};

// ============================================================
// Part 2: Single Reader, Single Writer Unbounded Queue
// ============================================================
// Linked-list based queue from the lecture (Dr. Dobbs Journal).
// Key design:
//   - Head points to element BEFORE the head of the queue
//   - Tail points to last element added
//   - Node allocation and deletion performed by the SAME thread
//     (producer allocates nodes AND reclaims consumed nodes)
//
// This avoids the ABA problem because the consumer only moves head,
// and the producer manages all memory.

template <typename T>
class UnboundedSPSCQueue {
private:
    struct Node {
        T value;
        Node* next;
        Node() : next(nullptr) {} // Sentinel constructor
        Node(T v) : value(v), next(nullptr) {}
    };

public:
    UnboundedSPSCQueue() {
        // Sentinel node: head and tail both point to it initially
        Node* sentinel = new Node();
        head_ = sentinel;
        tail_ = sentinel;
        reclaim_ = sentinel;
    }

    ~UnboundedSPSCQueue() {
        // Clean up remaining nodes
        while (reclaim_ != nullptr) {
            Node* tmp = reclaim_;
            reclaim_ = reclaim_->next;
            delete tmp;
        }
    }

    // Producer: add value to queue.
    // Also reclaims consumed nodes (producer owns memory management).
    void push(T value) {
        Node* n = new Node(value); // Producer allocates
        n->next = nullptr;

        // Link new node after tail
        tail_->next = n;
        tail_ = n;

        // Reclaim consumed nodes (nodes between reclaim_ and head_)
        // Only the producer reclaims, avoiding ABA issues.
        while (reclaim_ != head_) {
            Node* tmp = reclaim_;
            reclaim_ = tmp->next;
            delete tmp;
        }
    }

    // Consumer: return false if queue is empty
    bool pop(T& value) {
        Node* current_head = head_;
        Node* next = current_head->next;

        // Queue is empty if head's next is null (head == tail)
        if (next == nullptr) {
            return false;
        }

        // Read value from the node AFTER head (head is always sentinel)
        value = next->value;
        head_ = next;
        // Note: we don't delete the old head here.
        // The producer thread handles reclamation in push().
        return true;
    }

private:
    // Head: always points to a sentinel node (element before first real node)
    // Tail: points to the last real node added
    // Reclaim: producer uses this to track nodes that can be freed
    Node* head_;      // modified only by consumer
    Node* tail_;      // modified only by producer
    Node* reclaim_;   // modified only by producer
};

// ============================================================
// Demonstration
// ============================================================

void producer_bounded(BoundedSPSCQueue<int, 16>& q, int count) {
    int pushed = 0;
    while (pushed < count) {
        if (q.push(pushed)) {
            ++pushed;
        }
        // If full, spin-wait (would be wasted CPU in real code)
    }
    std::cout << "Producer finished pushing " << pushed << " items." << std::endl;
}

void consumer_bounded(BoundedSPSCQueue<int, 16>& q, int count) {
    int popped = 0;
    int sum = 0;
    while (popped < count) {
        int val;
        if (q.pop(val)) {
            sum += val;
            ++popped;
        }
    }
    std::cout << "Consumer popped " << popped << " items, sum=" << sum << std::endl;
}

void producer_unbounded(UnboundedSPSCQueue<int>& q, int count) {
    for (int i = 0; i < count; ++i) {
        q.push(i);
    }
    std::cout << "Unbounded producer finished pushing " << count << " items." << std::endl;
}

void consumer_unbounded(UnboundedSPSCQueue<int>& q, int count) {
    int popped = 0;
    int sum = 0;
    while (popped < count) {
        int val;
        if (q.pop(val)) {
            sum += val;
            ++popped;
        }
    }
    std::cout << "Unbounded consumer popped " << popped << " items, sum=" << sum << std::endl;
}

int main() {
    std::cout << "=== CS149 Lecture 16: Lock-Free Queues ===" << std::endl;
    std::cout << std::endl;

    const int num_items = 100000;

    // ---- Demo 1: Bounded SPSC Queue ----
    std::cout << "--- Demo 1: Bounded SPSC Queue (ring buffer, capacity=16) ---" << std::endl;
    {
        BoundedSPSCQueue<int, 16> q;
        auto start = std::chrono::high_resolution_clock::now();

        std::thread producer(producer_bounded, std::ref(q), num_items);
        std::thread consumer(consumer_bounded, std::ref(q), num_items);
        producer.join();
        consumer.join();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Time: " << duration.count() << "ms" << std::endl;
    }

    // ---- Demo 2: Unbounded SPSC Queue ----
    std::cout << std::endl;
    std::cout << "--- Demo 2: Unbounded SPSC Queue (linked-list) ---" << std::endl;
    {
        UnboundedSPSCQueue<int> q;
        auto start = std::chrono::high_resolution_clock::now();

        std::thread producer(producer_unbounded, std::ref(q), num_items);
        std::thread consumer(consumer_unbounded, std::ref(q), num_items);
        producer.join();
        consumer.join();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Time: " << duration.count() << "ms" << std::endl;
    }

    // ---- Demo 3: Correctness verification ----
    std::cout << std::endl;
    std::cout << "--- Demo 3: Correctness Test ---" << std::endl;
    {
        BoundedSPSCQueue<int, 8> q;
        // Push 1,2,3,4,5
        for (int i = 1; i <= 5; ++i) {
            assert(q.push(i) && "Should succeed");
        }
        // Pop in FIFO order
        for (int expected = 1; expected <= 5; ++expected) {
            int val;
            assert(q.pop(val) && "Should succeed");
            assert(val == expected && "FIFO order");
        }
        // Should be empty now
        int dummy;
        assert(!q.pop(dummy) && "Should be empty");
        std::cout << "Bounded queue FIFO order verified." << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Summary:" << std::endl;
    std::cout << "  - SPSC (Single Producer, Single Consumer) queues need NO" << std::endl;
    std::cout << "    locks or CAS. Each end is modified by only one thread." << std::endl;
    std::cout << "  - Bounded: ring buffer, needs memory fences for visibility." << std::endl;
    std::cout << "  - Unbounded: linked-list. Producer manages memory reclamation" << std::endl;
    std::cout << "    to avoid ABA and use-after-free issues." << std::endl;
    std::cout << "  - For multiple producers/consumers, see Michael-Scott queue." << std::endl;

    return 0;
}
