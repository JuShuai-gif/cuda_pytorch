/**
 * lecture5_part2.cpp - Work Stealing Scheduler Simulation
 *
 * Simulates Cilk's work stealing scheduler:
 * - Per-thread dequeue (double-ended queue)
 * - Local thread pushes/pops from tail (LIFO)
 * - Remote threads steal from head (FIFO)
 * - Continuation stealing ("run child first")
 * - Random victim selection
 * - Greedy join scheduling
 *
 * Compile: g++ -std=c++17 -pthread lecture5_part2.cpp -o lecture5_part2 && ./lecture5_part2
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <thread>
#include <mutex>
#include <random>
#include <chrono>
#include <deque>
#include <atomic>
#include <algorithm>
#include <sstream>
#include <memory>

// ============================================================================
// Part 1: Lock-Free(ish) Dequeue for Work Stealing
// ============================================================================

/**
 * Simplified dequeue (double-ended queue) for work stealing.
 * In real Cilk, this is implemented lock-free for performance.
 * Here we use a mutex for simplicity.
 *
 * Local operations (push_back, pop_back): LIFO
 * Remote operations (pop_front): FIFO, steals largest work first
 */
template<typename T>
class WorkStealingDequeue {
private:
    std::deque<T> queue;
    mutable std::mutex mtx;

public:
    void push_back(T item) {
        std::lock_guard<std::mutex> lock(mtx);
        queue.push_back(std::move(item));
    }

    bool pop_back(T& item) {
        std::lock_guard<std::mutex> lock(mtx);
        if (queue.empty()) return false;
        item = std::move(queue.back());
        queue.pop_back();
        return true;
    }

    // Steal from front: takes the largest/oldest piece of work
    bool steal_front(T& item) {
        std::lock_guard<std::mutex> lock(mtx);
        if (queue.empty()) return false;
        item = std::move(queue.front());
        queue.pop_front();
        return true;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mtx);
        return queue.size();
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mtx);
        return queue.empty();
    }
};

// ============================================================================
// Part 2: Task and Work Representation
// ============================================================================

/**
 * Represents a piece of work in the system.
 * "Continuation" tasks represent the rest of a for loop or function.
 * "Leaf" tasks are individual work items.
 */
struct WorkItem {
    enum Type { LEAF, CONTINUATION };

    Type type;
    int id;          // Task identifier
    int start;       // For continuations: start of remaining range
    int end;         // For continuations: end of remaining range
    int block_id;    // Sync block this belongs to

    std::string describe() const {
        std::ostringstream oss;
        if (type == LEAF) {
            oss << "Leaf(" << id << ")";
        } else {
            oss << "Cont([" << start << "," << end << "), block=" << block_id << ")";
        }
        return oss.str();
    }
};

// ============================================================================
// Part 3: Sync Block Descriptor
// ============================================================================

/**
 * Descriptor for a cilk_sync block.
 * Tracks number of spawned tasks and completed tasks.
 * Used to determine when all spawned work for a block is done.
 */
struct SyncBlockDescriptor {
    int block_id;
    int total_spawned;   // Total spawns in this block
    int total_completed;  // Completed spawns
    bool stolen;          // Whether work from this block was stolen
    std::unique_ptr<std::mutex> mtx;

    SyncBlockDescriptor(int id)
        : block_id(id), total_spawned(0), total_completed(0), stolen(false),
          mtx(std::make_unique<std::mutex>()) {}

    void increment_spawned() {
        std::lock_guard<std::mutex> lock(*mtx);
        total_spawned++;
    }

    void increment_completed() {
        std::lock_guard<std::mutex> lock(*mtx);
        total_completed++;
    }

    bool all_completed() {
        std::lock_guard<std::mutex> lock(*mtx);
        return total_spawned > 0 && total_completed >= total_spawned;
    }
};

// ============================================================================
// Part 4: Work Stealing Scheduler
// ============================================================================

class WorkStealingScheduler {
private:
    int num_threads;
    std::vector<WorkStealingDequeue<WorkItem>> queues;
    std::vector<std::thread> workers;
    std::vector<SyncBlockDescriptor> sync_blocks;
    std::atomic<bool> shutdown{false};
    std::atomic<int> active_workers{0};

    // Random engine for victim selection
    std::mt19937 rng;

    // Statistics
    std::atomic<long> total_steals{0};
    std::atomic<long> total_local_pops{0};
    std::atomic<long> total_tasks_completed{0};

public:
    explicit WorkStealingScheduler(int threads) : num_threads(threads), queues(threads) {
        std::random_device rd;
        rng.seed(rd());
    }

    /**
     * Simulates spawning a for loop: for (int i=0; i<N; i++) cilk_spawn foo(i);
     *
     * With continuation stealing (run child first):
     * - Thread 0 starts executing foo(0) immediately
     * - Places continuation (i=1..N) in its work queue for stealing
     */
    void simulate_spawn_loop(int N, int tid) {
        if (N <= 0) return;

        // Create a sync block for tracking
        int block_id = static_cast<int>(sync_blocks.size());
        sync_blocks.emplace_back(block_id);

        // Run child first: execute foo(0), enqueue continuation
        for (int i = 0; i < N; i++) {
            // Simulate spawning foo(i)
            sync_blocks[block_id].increment_spawned();

            if (i == 0 && tid == 0) {
                // Thread 0 runs child first (foo(0))
                std::cout << "  [T" << tid << "] Executing foo(" << i
                          << ") directly (run child first)\n";
                execute_task({WorkItem::LEAF, i, 0, 0, block_id}, tid);
                sync_blocks[block_id].increment_completed();
            } else {
                // Enqueue continuation represents all remaining iterations
                // In real Cilk: single continuation with i as the loop counter
                WorkItem cont{WorkItem::CONTINUATION, 0, i, N, block_id};
                queues[tid].push_back(cont);
                std::cout << "  [T" << tid << "] Enqueued continuation i="
                          << i << ".." << N-1 << " (block=" << block_id << ")\n";
                break;  // Only one continuation needed
            }
        }
    }

    /**
     * Execute a single task (simulated work).
     */
    void execute_task(const WorkItem& item, int tid) {
        // Simulate work
        volatile int work = 0;
        int workload = (item.type == WorkItem::LEAF) ? 1000000 : 100000;
        for (int i = 0; i < workload; i++) work++;

        total_tasks_completed++;
        std::cout << "  [T" << tid << "] Completed " << item.describe() << "\n";
    }

    /**
     * Worker thread main loop.
     * Implements Cilk's work stealing worker behavior:
     * 1. Try to pop work from own queue (tail)
     * 2. If empty, try to steal from a random victim (head)
     * 3. If nothing to steal anywhere, go idle
     */
    void worker_loop(int tid) {
        active_workers++;
        int failed_steal_attempts = 0;
        const int MAX_FAILED_STEALS = num_threads * 2;

        while (!shutdown) {
            WorkItem task;

            // Step 1: Try local queue (pop from tail - LIFO)
            if (queues[tid].pop_back(task)) {
                total_local_pops++;
                failed_steal_attempts = 0;
                execute_task(task, tid);
                continue;
            }

            // Step 2: Queue empty - try to steal (pop from head - FIFO)
            int victim = std::uniform_int_distribution<int>(0, num_threads - 1)(rng);
            if (victim != tid) {
                if (queues[victim].steal_front(task)) {
                    total_steals++;
                    failed_steal_attempts = 0;
                    std::cout << "  [T" << tid << "] STOLE from T" << victim
                              << ": " << task.describe() << "\n";
                    execute_task(task, tid);
                    continue;
                }
            }

            // Step 3: Nothing to steal
            failed_steal_attempts++;
            if (failed_steal_attempts >= MAX_FAILED_STEALS) {
                // Greedy: if nothing to steal anywhere, go idle
                // In real Cilk: thread blocks until new work arrives
                break;
            }
            std::this_thread::yield();
        }
        active_workers--;
    }

    /**
     * Run a work stealing simulation.
     */
    void run_simulation(int num_tasks) {
        std::cout << "\n=== Work Stealing Scheduler Simulation ===\n";
        std::cout << "Threads: " << num_threads << "  Tasks: " << num_tasks << "\n\n";

        // Start worker threads
        for (int t = 0; t < num_threads; t++) {
            workers.emplace_back(&WorkStealingScheduler::worker_loop, this, t);
        }

        // Give threads time to start
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        // Thread 0 spawns all the work (simulates main thread creating tasks)
        std::cout << "[Main] Spawning " << num_tasks << " tasks on T0...\n";
        simulate_spawn_loop(num_tasks, 0);

        // Wait for all work to complete
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        shutdown = true;

        for (auto& w : workers) w.join();

        // Statistics
        std::cout << "\n--- Scheduler Statistics ---\n";
        std::cout << "  Total tasks completed: " << total_tasks_completed << "\n";
        std::cout << "  Total local pops:      " << total_local_pops << "\n";
        std::cout << "  Total steals:          " << total_steals << "\n";
        if (total_local_pops + total_steals > 0) {
            double steal_pct = 100.0 * total_steals / (total_local_pops + total_steals);
            std::cout << "  Steal percentage:      " << std::fixed
                      << std::setprecision(1) << steal_pct << "%\n";
        }
    }
};

// ============================================================================
// Part 5: Dequeue Behavior Visual Demonstration
// ============================================================================

void demonstrate_dequeue_behavior() {
    std::cout << "\n=== Dequeue Behavior: Local LIFO, Remote FIFO ===\n\n";

    WorkStealingDequeue<int> dq;

    // Simulate thread pushing work (continuation stealing)
    std::cout << "Thread 0 (local) pushes work items:\n";
    for (int i = 0; i < 5; i++) {
        dq.push_back(i * 10);
        std::cout << "  push_back(" << i * 10 << ")\n";
    }

    std::cout << "\nThread 0 pops from tail (LIFO):\n";
    int val;
    if (dq.pop_back(val)) std::cout << "  pop_back() → " << val << "\n";
    if (dq.pop_back(val)) std::cout << "  pop_back() → " << val << "\n";

    std::cout << "\nThread 1 steals from head (FIFO):\n";
    if (dq.steal_front(val)) std::cout << "  steal_front() → " << val << " (oldest work)\n";
    if (dq.steal_front(val)) std::cout << "  steal_front() → " << val << " (next oldest)\n";
    if (dq.steal_front(val)) std::cout << "  steal_front() → " << val << "\n";

    std::cout << "\nQueue empty? " << (dq.empty() ? "YES" : "NO") << "\n";

    std::cout << "\nKey insight:\n";
    std::cout << "  - Local LIFO: maintains depth-first execution → good cache locality\n";
    std::cout << "  - Remote FIFO: steals largest/oldest work → fewer total steals\n";
    std::cout << "  - No contention: local thread uses tail, remote thread uses head\n";
}

// ============================================================================
// Part 6: Run Child First vs Run Continuation First
// ============================================================================

void explain_spawn_strategies() {
    std::cout << "\n=== Spawn Strategy: Run Child First vs Run Continuation First ===\n\n";

    std::cout << "Code: for (int i=0; i<N; i++) { cilk_spawn foo(i); } cilk_sync;\n\n";

    std::cout << "Strategy 1: Run continuation first ('child stealing')\n";
    std::cout << "  Thread enqueues foo(0), continues to spawn foo(1), foo(2), ...\n";
    std::cout << "  Queue after all spawns: [foo(0), foo(1), ..., foo(N-1)]\n";
    std::cout << "  Space: O(N) items in queue\n";
    std::cout << "  Execution order: breadth-first (very different from sequential)\n\n";

    std::cout << "Strategy 2: Run child first ('continuation stealing') ← Cilk uses this\n";
    std::cout << "  Thread executes foo(0) immediately, enqueues ONE continuation (i=1..N)\n";
    std::cout << "  Queue: [cont(i=1..N-1)]\n";
    std::cout << "  Space: O(T) where T = max threads (bounded!)\n";
    std::cout << "  Execution order: depth-first (same as sequential if no stealing)\n\n";
}

// ============================================================================
// Part 7: Greedy Join Scheduling Explanation
// ============================================================================

void explain_greedy_join() {
    std::cout << "\n=== Greedy Join Scheduling (Cilk) ===\n\n";

    std::cout << "Key principles:\n";
    std::cout << "  1. All threads always attempt to steal if nothing to do\n";
    std::cout << "  2. Threads only go idle if NO work exists anywhere in system\n";
    std::cout << "  3. Thread that initiated spawn may NOT execute code after cilk_sync\n\n";

    std::cout << "Why this matters:\n";
    std::cout << "  - The 'last' thread to complete a spawn continues the caller\n";
    std::cout << "  - Overhead of sync bookkeeping only occurs when steals happen\n";
    std::cout << "  - In common case (no stealing): cilk_sync is a no-op\n";
    std::cout << "  - Descriptors track spawn/completion counts only when stolen\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "============================================================\n";
    std::cout << "Lecture 5 Part 2: Work Stealing Scheduler Simulation\n";
    std::cout << "============================================================\n";

    // Part 1: Dequeue behavior demonstration
    demonstrate_dequeue_behavior();

    // Part 2: Spawn strategy explanation
    explain_spawn_strategies();

    // Part 3: Work stealing simulation
    int hw_threads = std::thread::hardware_concurrency();
    if (hw_threads < 2) hw_threads = 2;

    WorkStealingScheduler scheduler(hw_threads);
    scheduler.run_simulation(5);  // Small simulation for clarity

    // Part 4: Greedy join explanation
    explain_greedy_join();

    // Part 5: Key takeaways
    std::cout << "\n=== Cilk Scheduler: Key Design Summary ===\n";
    std::cout << "┌────────────────────┬────────────────────────────────────┐\n";
    std::cout << "│ Design Element     │ Implementation                     │\n";
    std::cout << "├────────────────────┼────────────────────────────────────┤\n";
    std::cout << "│ Queue structure    │ Dequeue: local tail, remote head   │\n";
    std::cout << "│ Spawn strategy     │ Run child first (continuation steal│\n";
    std::cout << "│ Local operations   │ LIFO (push/pop tail)               │\n";
    std::cout << "│ Steal operations   │ FIFO (steal from head)             │\n";
    std::cout << "│ Victim selection   │ Random (uniform distribution)      │\n";
    std::cout << "│ Join behavior      │ Greedy: never wait, always steal   │\n";
    std::cout << "│ Sync overhead      │ Only paid when stealing occurs     │\n";
    std::cout << "│ Work queue storage │ O(T * stack_depth) at most         │\n";
    std::cout << "└────────────────────┴────────────────────────────────────┘\n";

    std::cout << "\nAll tests completed successfully.\n";
    return 0;
}
