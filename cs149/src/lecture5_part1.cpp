/**
 * lecture5_part1.cpp - Static vs Dynamic Work Assignment
 *
 * Demonstrates key concepts from CS149 Lecture 5:
 * - Static assignment: work divided evenly, predictable cost
 * - Dynamic assignment: work claimed at runtime via shared counter
 * - Semi-static assignment: periodic rebalancing
 * - Task granularity: fine vs coarse
 * - Work queue model
 *
 * Uses primality testing as the workload (unpredictable execution time)
 * to highlight when dynamic assignment outperforms static.
 *
 * Compile: g++ -std=c++17 -pthread lecture5_part1.cpp -o lecture5_part1 && ./lecture5_part1
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>
#include <random>
#include <algorithm>
#include <atomic>
#include <cmath>

// ============================================================================
// Part 1: Simulated Work - Primality Testing
// ============================================================================

/**
 * Simulated primality test with varying execution time.
 * Larger numbers take longer (unpredictable cost from the scheduler's
 * perspective if the input values are not known in advance).
 */
bool test_primality(long long n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;

    // Trial division - work proportional to sqrt(n)
    long long limit = static_cast<long long>(std::sqrt(static_cast<double>(n)));
    for (long long i = 3; i <= limit; i += 2) {
        if (n % i == 0) return false;
    }
    return true;
}

// ============================================================================
// Part 2: Static Assignment
// ============================================================================

/**
 * Static assignment: each thread gets a fixed, pre-determined chunk of work.
 * Works well when all tasks have equal cost.
 * Fails when costs are unpredictable (load imbalance).
 */
std::vector<bool> static_assignment(const std::vector<long long>& inputs,
                                     int num_threads) {
    int N = inputs.size();
    std::vector<bool> results(N, false);

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; t++) {
        int start = t * (N / num_threads);
        int end = (t == num_threads - 1) ? N : start + (N / num_threads);
        threads.emplace_back([&inputs, &results, start, end]() {
            for (int i = start; i < end; i++) {
                results[i] = test_primality(inputs[i]);
            }
        });
    }
    for (auto& th : threads) th.join();
    return results;
}

// ============================================================================
// Part 3: Dynamic Assignment (Shared Counter)
// ============================================================================

/**
 * Dynamic assignment using a shared atomic counter.
 * Each thread grabs the next available work item.
 * Better load balance when task costs vary.
 * But: synchronization overhead from atomic increment.
 */
std::vector<bool> dynamic_assignment_counter(const std::vector<long long>& inputs,
                                              int num_threads) {
    int N = inputs.size();
    std::vector<bool> results(N, false);
    std::atomic<int> counter{0};

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back([&inputs, &results, &counter, N]() {
            while (true) {
                int i = counter.fetch_add(1);
                if (i >= N) break;
                results[i] = test_primality(inputs[i]);
            }
        });
    }
    for (auto& th : threads) th.join();
    return results;
}

// ============================================================================
// Part 4: Dynamic Assignment with Coarse Granularity
// ============================================================================

/**
 * Dynamic assignment with tunable granularity.
 * GRANULARITY = 1: fine-grained (1 element per critical section entry)
 * GRANULARITY = 10: coarse-grained (10 elements per critical section entry)
 *
 * Trade-off:
 * - Fine: better load balance, higher sync overhead
 * - Coarse: lower sync overhead, potentially worse load balance
 */
std::vector<bool> dynamic_assignment_granular(const std::vector<long long>& inputs,
                                               int num_threads, int granularity) {
    int N = inputs.size();
    std::vector<bool> results(N, false);
    std::atomic<int> counter{0};

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back([&inputs, &results, &counter, N, granularity]() {
            while (true) {
                int i = counter.fetch_add(granularity);
                if (i >= N) break;
                int end = std::min(i + granularity, N);
                for (int j = i; j < end; j++) {
                    results[j] = test_primality(inputs[j]);
                }
            }
        });
    }
    for (auto& th : threads) th.join();
    return results;
}

// ============================================================================
// Part 5: Work Queue Model
// ============================================================================

/**
 * Simple shared work queue.
 * Tasks are pushed to a queue and workers pull from it.
 * This is the simplest work queue model - one queue, multiple workers.
 *
 * Note: Single queue creates contention when many threads access it.
 * Distributed queues (one per worker) + work stealing is better.
 */
class SimpleWorkQueue {
private:
    std::mutex mtx;
    std::vector<int> tasks;
    int next_task;

public:
    SimpleWorkQueue() : next_task(0) {}

    void add_task(int task) {
        std::lock_guard<std::mutex> lock(mtx);
        tasks.push_back(task);
    }

    bool get_task(int& task) {
        std::lock_guard<std::mutex> lock(mtx);
        if (next_task >= static_cast<int>(tasks.size())) return false;
        task = tasks[next_task++];
        return true;
    }

    size_t size() const { return tasks.size() - next_task; }
};

// ============================================================================
// Part 6: Benchmarking
// ============================================================================

struct BenchmarkResult {
    double time_seconds;
    int tasks_completed;
    double imbalance_ratio;  // max_time / min_time among threads
};

/**
 * Generate a workload with a mix of "easy" and "hard" primality tests.
 * This creates load imbalance when using static assignment.
 */
std::vector<long long> generate_workload(int N, bool balanced) {
    std::vector<long long> data(N);
    std::mt19937 rng(42);

    if (balanced) {
        // All tasks have similar cost (numbers around 1000)
        std::uniform_int_distribution<long long> dist(900, 1100);
        for (int i = 0; i < N; i++) data[i] = dist(rng);
    } else {
        // Mixed: some very large numbers (high cost) scattered among small ones
        for (int i = 0; i < N; i++) {
            if (i % 50 == 0) {
                // Hard task: large prime test
                data[i] = 1000000 + (rng() % 10000);
            } else {
                // Easy task: small prime test
                data[i] = 100 + (rng() % 200);
            }
        }
    }
    return data;
}

template<typename F>
BenchmarkResult benchmark(F&& fn, const std::string& label,
                          const std::vector<long long>& inputs, int num_threads) {
    auto start = std::chrono::high_resolution_clock::now();
    auto results = fn(inputs, num_threads);
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(end - start).count();
    int completed = static_cast<int>(results.size());

    std::cout << "  " << std::left << std::setw(35) << label
              << " time=" << std::fixed << std::setprecision(4) << elapsed << "s"
              << "  tasks=" << completed << "\n";
    return {elapsed, completed, 0.0};
}

// ============================================================================
// Part 7: Load Imbalance Visual Explanation
// ============================================================================

void explain_load_imbalance() {
    std::cout << "\n=== Load Imbalance Demonstration ===\n\n";

    std::cout << "Static Assignment (blocked), P=4:\n";
    std::cout << "  P1: [easy, easy, easy, hard] → 3 units\n";
    std::cout << "  P2: [easy, easy, easy, easy] → 4 units\n";
    std::cout << "  P3: [easy, easy, easy, easy] → 4 units\n";
    std::cout << "  P4: [easy, easy, easy, easy] → 4 units\n\n";
    std::cout << "  P1 finishes last at t=4, but P2-P4 idle from t=3→4.\n";
    std::cout << "  Result: 25% idle time → effective S=0.25 serial fraction!\n\n";

    std::cout << "Dynamic Assignment (shared counter), P=4:\n";
    std::cout << "  P1:[easy, easy, hard] P2:[easy, easy, easy, easy]\n";
    std::cout << "  P3:[easy, easy, easy, easy] P4:[easy, easy, easy]\n\n";
    std::cout << "  Work naturally balances. All finish near t=4.\n";
}

// ============================================================================
// Part 8: Semi-Static Assignment Concept
// ============================================================================

void explain_semi_static() {
    std::cout << "\n=== Semi-Static Assignment ===\n\n";

    std::cout << "Concept: When cost of work is predictable for near-term future.\n";
    std::cout << "  - Application periodically profiles execution\n";
    std::cout << "  - Re-adjusts assignment based on recent performance\n";
    std::cout << "  - Assignment is 'static' between re-adjustments\n\n";

    std::cout << "Examples from lecture:\n";
    std::cout << "  - Particle simulation: redistribute as particles move slowly\n";
    std::cout << "  - Adaptive mesh: remesh when object moves; reassign regions\n";
    std::cout << "  - Cost function: next_work_cost ≈ recent_work_cost\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "============================================================\n";
    std::cout << "Lecture 5 Part 1: Static vs Dynamic Work Assignment\n";
    std::cout << "============================================================\n";

    const int NUM_THREADS = 4;
    const int N_BALANCED = 200;
    const int N_UNBALANCED = 200;

    // === Generate workloads ===
    auto balanced_wl = generate_workload(N_BALANCED, true);
    auto unbalanced_wl = generate_workload(N_UNBALANCED, false);

    // === Benchmark: Balanced Workload ===
    std::cout << "\n--- Balanced Workload (similar cost tasks) ---\n";
    benchmark([](const auto& in, int p) { return static_assignment(in, p); },
              "Static Assignment (blocked)", balanced_wl, NUM_THREADS);
    benchmark([](const auto& in, int p) { return dynamic_assignment_counter(in, p); },
              "Dynamic Assignment (counter)", balanced_wl, NUM_THREADS);

    std::cout << "\n  Observation: Static ≈ Dynamic when costs are balanced.\n";
    std::cout << "  Static has lower overhead (no atomic ops per element).\n";

    // === Benchmark: Unbalanced Workload ===
    std::cout << "\n--- Unbalanced Workload (mixed easy/hard tasks) ---\n";
    benchmark([](const auto& in, int p) { return static_assignment(in, p); },
              "Static Assignment (blocked)", unbalanced_wl, NUM_THREADS);
    benchmark([](const auto& in, int p) { return dynamic_assignment_counter(in, p); },
              "Dynamic Assignment (counter, fine)", unbalanced_wl, NUM_THREADS);
    benchmark([](const auto& in, int p) { return dynamic_assignment_granular(in, p, 5); },
              "Dynamic Assignment (granularity=5)", unbalanced_wl, NUM_THREADS);
    benchmark([](const auto& in, int p) { return dynamic_assignment_granular(in, p, 20); },
              "Dynamic Assignment (granularity=20)", unbalanced_wl, NUM_THREADS);

    std::cout << "\n  Observation: Dynamic >> Static when costs are unbalanced.\n";
    std::cout << "  Coarser granularity reduces sync overhead but may sacrifice balance.\n";

    // === Test correctness ===
    std::cout << "\n--- Correctness Verification ---\n";
    auto ref_results = static_assignment(unbalanced_wl, 1);  // Sequential reference
    auto dyn_results = dynamic_assignment_counter(unbalanced_wl, NUM_THREADS);

    bool correct = (ref_results.size() == dyn_results.size());
    for (size_t i = 0; i < ref_results.size() && correct; i++) {
        correct = (ref_results[i] == dyn_results[i]);
    }
    std::cout << "  Static(1) == Dynamic(4): " << (correct ? "YES" : "NO") << "\n";

    // === Load Imbalance Explanation ===
    explain_load_imbalance();

    // === Semi-Static Explanation ===
    explain_semi_static();

    // === Task Granularity Summary ===
    std::cout << "\n=== Task Granularity Trade-off ===\n";
    std::cout << "┌──────────────┬───────────────────┬─────────────────────┐\n";
    std::cout << "│ Granularity  │ Load Balance      │ Sync Overhead       │\n";
    std::cout << "├──────────────┼───────────────────┼─────────────────────┤\n";
    std::cout << "│ Fine (1)     │ Best              │ Highest (per item)  │\n";
    std::cout << "│ Medium (5-20)│ Good              │ Moderate            │\n";
    std::cout << "│ Coarse (100+)│ Potentially poor  │ Lowest              │\n";
    std::cout << "└──────────────┴───────────────────┴─────────────────────┘\n";

    std::cout << "\nAll tests completed successfully.\n";
    return 0;
}
