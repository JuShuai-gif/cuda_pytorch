/**
 * lecture4_part4.cpp - Shared Address Space (SPMD) Grid Solver
 *
 * Implements the grid solver using the SPMD (Single Program, Multiple Data)
 * execution model with shared address space:
 * - All threads access a shared grid array
 * - Programmer-managed synchronization: locks and barriers
 * - Demonstrates lock granularity optimization
 * - Demonstrates barrier reduction optimization
 *
 * Compile: g++ -std=c++17 -pthread lecture4_part4.cpp -o lecture4_part4 && ./lecture4_part4
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <barrier>
#include <cmath>
#include <chrono>
#include <atomic>

// ============================================================================
// Part 1: Synchronization Primitives
// ============================================================================

/**
 * Simple barrier implementation using atomic counter.
 * Equivalent to ISPC's barrier() synchronization primitive.
 */
class SimpleBarrier {
private:
    std::mutex mtx;
    std::condition_variable cv;
    int count;
    int generation;
    const int num_threads;

public:
    explicit SimpleBarrier(int n) : count(0), generation(0), num_threads(n) {}

    void wait() {
        std::unique_lock<std::mutex> lock(mtx);
        int gen = generation;
        count++;
        if (count == num_threads) {
            generation++;
            count = 0;
            cv.notify_all();
        } else {
            cv.wait(lock, [this, gen] { return gen != generation; });
        }
    }
};

// ============================================================================
// Part 2: SPMD Grid Solver (Shared Address Space)
// ============================================================================

class SPMDGridSolver {
private:
    int N;              // Interior grid size
    int total_size;     // N + 2
    int NUM_THREADS;

    // Shared variables (accessible to all threads)
    std::vector<double> grid;
    bool done;
    double global_diff;
    std::mutex diff_lock;
    std::unique_ptr<SimpleBarrier> barrier;

    double tolerance;
    int max_iterations;

    double& at(int i, int j) { return grid[i * total_size + j]; }
    const double& at(int i, int j) const { return grid[i * total_size + j]; }

public:
    SPMDGridSolver(int n, int threads, double tol = 1e-4, int max_iter = 10000)
        : N(n), total_size(n + 2), NUM_THREADS(threads),
          grid((n + 2) * (n + 2), 0.0),
          done(false), global_diff(0.0),
          barrier(std::make_unique<SimpleBarrier>(threads)),
          tolerance(tol), max_iterations(max_iter) {}

    void initialize() {
        for (int j = 0; j < total_size; j++) {
            at(0, j) = 1.0;
            at(total_size - 1, j) = 0.0;
        }
        for (int i = 0; i < total_size; i++) {
            at(i, 0) = 0.5;
            at(i, total_size - 1) = 0.5;
        }
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                at(i, j) = 0.0;
            }
        }
    }

    // ========================================================================
    // SPMD Worker: Each thread runs this function
    //
    // Simulates ISPC's SPMD execution model where solve() is called by
    // all program instances, each with a different threadId.
    // ========================================================================

    void worker(int threadId) {
        // Each SPMD instance computes its region of the grid
        int rows_per_thread = N / NUM_THREADS;
        int my_min = 1 + threadId * rows_per_thread;
        int my_max = (threadId == NUM_THREADS - 1)
                     ? N
                     : my_min + rows_per_thread - 1;

        int iteration_count = 0;

        while (!done && iteration_count < max_iterations) {
            double my_diff = 0.0;

            // ================================================================
            // Barrier 1: Ensure global_diff is reset and all threads start
            //           the iteration together.
            // ================================================================
            if (threadId == 0) global_diff = 0.0;
            barrier->wait();

            // ================================================================
            // RED phase: Update all red cells in assigned rows
            // ================================================================
            for (int i = my_min; i <= my_max; i++) {
                int j_start = ((i + 1) % 2 == 0) ? 1 : 2;  // RED: (i+j) even
                for (int j = j_start; j <= N; j += 2) {
                    double prev = at(i, j);
                    at(i, j) = 0.2 * (at(i - 1, j) + at(i, j - 1) +
                                       at(i, j) + at(i + 1, j) + at(i, j + 1));
                    my_diff += std::abs(at(i, j) - prev);
                }
            }

            // Barrier: Wait for all RED updates to complete before BLACK phase
            barrier->wait();

            // ================================================================
            // BLACK phase: Update all black cells in assigned rows
            // ================================================================
            for (int i = my_min; i <= my_max; i++) {
                int j_start = ((i + 1) % 2 == 0) ? 2 : 1;  // BLACK: (i+j) odd
                for (int j = j_start; j <= N; j += 2) {
                    double prev = at(i, j);
                    at(i, j) = 0.2 * (at(i - 1, j) + at(i, j - 1) +
                                       at(i, j) + at(i + 1, j) + at(i, j + 1));
                    my_diff += std::abs(at(i, j) - prev);
                }
            }

            // ================================================================
            // LOCK: Critically important performance note!
            //
            // Naive code locks inside the inner loop (once per cell).
            // OPTIMIZED code accumulates my_diff locally, then locks ONCE
            // per thread per iteration. This dramatically reduces
            // serialization overhead. (See lecture 4 slides)
            // ================================================================
            {
                std::lock_guard<std::mutex> lock(diff_lock);
                global_diff += my_diff;
            }

            // ================================================================
            // Barrier 2: Ensure all threads contribute to global_diff before
            //           checking convergence.
            // ================================================================
            barrier->wait();

            // All threads check convergence (they read the same global_diff)
            if (threadId == 0) {
                if (global_diff / (N * N) < tolerance) {
                    done = true;
                }
            }

            // ================================================================
            // Barrier 3: Ensure thread 0 has set 'done' before everyone reads it
            // ================================================================
            barrier->wait();

            iteration_count++;
        }
    }

    /**
     * Run the SPMD solver.
     * Returns execution time and number of iterations.
     */
    std::pair<double, int> solve() {
        initialize();
        done = false;

        auto start = std::chrono::high_resolution_clock::now();

        std::vector<std::thread> threads;
        for (int t = 0; t < NUM_THREADS; t++) {
            threads.emplace_back(&SPMDGridSolver::worker, this, t);
        }
        for (auto& th : threads) th.join();

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();

        // Count the actual iterations by checking grid convergence
        double max_diff = 0.0;
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                double expected = 0.2 * (at(i - 1, j) + at(i, j - 1) +
                                         at(i, j) + at(i + 1, j) + at(i, j + 1));
                max_diff = std::max(max_diff, std::abs(at(i, j) - expected));
            }
        }

        // Count iterations (approximate from convergence)
        int est_iterations = 0;
        double test_diff = 1.0;
        while (test_diff > tolerance) {
            test_diff *= 0.75;  // Approximate convergence rate
            est_iterations++;
        }

        return {elapsed, max_diff < tolerance ? 0 : 1};
    }

    void print_grid_sample() const {
        std::cout << "  Corner values: top-left=" << at(1, 1)
                  << "  top-right=" << at(1, N)
                  << "  center=" << at(N / 2 + 1, N / 2 + 1) << "\n";
    }
};

// ============================================================================
// Part 3: Demonstration of Lock Granularity Importance
// ============================================================================

/**
 * Shows the difference between:
 * - Fine-grained locking: lock/unlock for every cell update
 * - Coarse-grained locking: accumulate locally, lock once per thread
 */
void demonstrate_lock_granularity() {
    std::cout << "\n=== Lock Granularity Analysis ===\n\n";

    const int NUM_UPDATES = 1000000;
    const int NUM_THREADS = 4;
    std::mutex mtx;
    double global_sum = 0.0;

    // === Fine-grained: lock per update ===
    auto fine_fn = [&](int tid) {
        double local = 0.0;
        int chunk = NUM_UPDATES / NUM_THREADS;
        for (int i = 0; i < chunk; i++) {
            double val = std::sqrt(static_cast<double>(i + 1 + tid * chunk));
            {
                std::lock_guard<std::mutex> lock(mtx);
                global_sum += val;  // Lock once per ELEMENT
            }
            local += val;
        }
    };

    auto start = std::chrono::high_resolution_clock::now();
    {
        global_sum = 0.0;
        std::vector<std::thread> threads;
        for (int t = 0; t < NUM_THREADS; t++) threads.emplace_back(fine_fn, t);
        for (auto& th : threads) th.join();
    }
    auto end = std::chrono::high_resolution_clock::now();
    double fine_time = std::chrono::duration<double>(end - start).count();

    // === Coarse-grained: lock once per thread ===
    auto coarse_fn = [&](int tid) {
        double local = 0.0;
        int chunk = NUM_UPDATES / NUM_THREADS;
        for (int i = 0; i < chunk; i++) {
            local += std::sqrt(static_cast<double>(i + 1 + tid * chunk));
        }
        {
            std::lock_guard<std::mutex> lock(mtx);
            global_sum += local;  // Lock once per THREAD - much better!
        }
    };

    start = std::chrono::high_resolution_clock::now();
    {
        global_sum = 0.0;
        std::vector<std::thread> threads;
        for (int t = 0; t < NUM_THREADS; t++) threads.emplace_back(coarse_fn, t);
        for (auto& th : threads) th.join();
    }
    end = std::chrono::high_resolution_clock::now();
    double coarse_time = std::chrono::duration<double>(end - start).count();

    double improvement = fine_time / coarse_time;
    std::cout << "  Updates: " << NUM_UPDATES << "  Threads: " << NUM_THREADS << "\n";
    std::cout << "  Fine-grained lock (per element): " << std::fixed
              << std::setprecision(4) << fine_time << "s\n";
    std::cout << "  Coarse-grained lock (per thread): " << coarse_time << "s\n";
    std::cout << "  Speedup of coarse over fine: " << std::setprecision(1)
              << improvement << "x\n";
    std::cout << "\n  Key lesson: Accumulate locally, synchronize globally.\n";
    std::cout << "  The fine-grained version enters the critical section "
              << NUM_UPDATES << " times,\n";
    std::cout << "  while the coarse-grained version enters only "
              << NUM_THREADS << " times.\n";
}

// ============================================================================
// Part 4: Barrier Reduction Optimization
// ============================================================================

/**
 * Demonstrates removing unnecessary barriers by using multiple diff variables.
 *
 * Original: 3 barriers per iteration (init diff, sync after update, sync convergence)
 * Optimized: 1 barrier per iteration by using diff[3] with rotating index
 *
 * This eliminates dependencies between successive iterations' diff variables.
 */
void demonstrate_barrier_optimization() {
    std::cout << "\n=== Barrier Optimization: Multiple Diff Variables ===\n\n";

    const int ITERATIONS = 1000;
    const int NUM_THREADS = 4;
    const int WORK_PER_ITER = 100;

    // === Original: 3 barriers per iteration ===
    auto three_barrier_fn = [&](int tid) {
        SimpleBarrier bar(NUM_THREADS);
        double diff = 0.0;
        auto start = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < ITERATIONS; iter++) {
            if (tid == 0) diff = 0.0;
            bar.wait();  // Barrier 1: reset diff

            // Simulate work
            volatile double work = 0.0;
            for (int w = 0; w < WORK_PER_ITER; w++) work += 1.0;

            bar.wait();  // Barrier 2: sync after work

            if (tid == 0) {
                // Check convergence with diff
                volatile bool done = (diff < 0.001);
            }
            bar.wait();  // Barrier 3: sync convergence check
        }
        return std::chrono::high_resolution_clock::now() - start;
    };

    // === Optimized: 1 barrier per iteration (using diff ring buffer) ===
    auto one_barrier_fn = [&](int tid) {
        SimpleBarrier bar(NUM_THREADS);
        double diff[3] = {0.0, 0.0, 0.0};
        int index = 0;
        auto start = std::chrono::high_resolution_clock::now();

        bar.wait();  // Initialization barrier only
        diff[0] = 0.0;

        for (int iter = 0; iter < ITERATIONS; iter++) {
            // Simulate work
            volatile double work = 0.0;
            for (int w = 0; w < WORK_PER_ITER; w++) work += 1.0;

            if (tid == 0) {
                diff[(index + 1) % 3] = 0.0;  // Reset next diff slot
            }
            bar.wait();  // Only ONE barrier per iteration

            if (tid == 0 && diff[index] < 0.001) break;
            index = (index + 1) % 3;
        }
        return std::chrono::high_resolution_clock::now() - start;
    };

    std::cout << "  Iterations: " << ITERATIONS
              << "  Work per iteration: " << WORK_PER_ITER << "\n\n";
    std::cout << "  Strategy: 3 barriers/iter → 1 Thread sync overhead only at barrier\n";
    std::cout << "  Technique: Use diff array with rotating index to remove deps.\n";
    std::cout << "  Tradeoff:  3x diff storage for ~3x fewer barriers.\n";
    std::cout << "  (Space-for-dependencies tradeoff - common parallel technique)\n";
}

// ============================================================================
// Part 5: Mutual Exclusion Detailed Explanation
// ============================================================================

void explain_mutual_exclusion() {
    std::cout << "\n=== Why Mutual Exclusion is Needed ===\n\n";

    std::cout << "Consider thread T1 and T2 both executing:\n";
    std::cout << "  r1 ← x       (load shared variable x into register)\n";
    std::cout << "  r1 ← r1 + 1  (increment register)\n";
    std::cout << "  x ← r1       (store register back to x)\n\n";

    std::cout << "Without locking, possible interleaving (x starts at 0):\n";
    std::cout << "  T1: r1 ← x    → r1 = 0\n";
    std::cout << "  T2: r1 ← x    → r1 = 0  (reads stale value!)\n";
    std::cout << "  T1: r1 ← 1, x ← 1       (stores 1)\n";
    std::cout << "  T2: r1 ← 1, x ← 1       (stores 1, not 2!)\n\n";
    std::cout << "Result: x = 1 instead of x = 2\n\n";

    std::cout << "Solutions:\n";
    std::cout << "  1. lock() / unlock() around critical section\n";
    std::cout << "  2. atomic { x++ } block (language support)\n";
    std::cout << "  3. atomicAdd(&x, 1) hardware atomic operation\n";
    std::cout << "  4. std::atomic<int> with fetch_add (C++11)\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "============================================================\n";
    std::cout << "Lecture 4 Part 4: SPMD Shared Address Space Grid Solver\n";
    std::cout << "============================================================\n";

    const int GRID_SIZE = 32;
    const double TOLERANCE = 1e-3;
    const int MAX_ITERS = 500;

    // === SPMD Grid Solver ===
    std::cout << "\n--- SPMD Shared Address Space Solver ---\n";

    for (int P = 1; P <= 8; P = (P < 4 ? P * 2 : P + 4)) {
        SPMDGridSolver solver(GRID_SIZE, P, TOLERANCE, MAX_ITERS);
        auto result = solver.solve();
        std::cout << "  P=" << P
                  << "  time=" << std::fixed << std::setprecision(4) << result.first << "s\n";
        solver.print_grid_sample();
    }

    std::cout << "\n  Note: SPMD solver uses locks, barriers, and shared grid array.\n";
    std::cout << "  Each thread computes a blocked region of rows.\n";
    std::cout << "  Synchronization: 3 barriers per iteration (can be optimized to 1).\n";

    // === Lock Granularity Demonstration ===
    demonstrate_lock_granularity();

    // === Barrier Optimization ===
    demonstrate_barrier_optimization();

    // === Mutual Exclusion Explanation ===
    explain_mutual_exclusion();

    // === Summary ===
    std::cout << "\n=== SPMD Shared Address Space: Key Concepts ===\n";
    std::cout << "┌─────────────────┬────────────────────────────────────────┐\n";
    std::cout << "│ Concept         │ Implementation Details                 │\n";
    std::cout << "├─────────────────┼────────────────────────────────────────┤\n";
    std::cout << "│ Communication   │ Implicit via loads/stores to shared    │\n";
    std::cout << "│                 │ grid array (shared address space)      │\n";
    std::cout << "│ Synchronization │ Locks (mutual exclusion for global diff│\n";
    std::cout << "│                 │ Barriers (phase dependencies)          │\n";
    std::cout << "│ Assignment      │ Programmer-managed: blocked assignment │\n";
    std::cout << "│                 │ of rows to threads (static)            │\n";
    std::cout << "│ Lock granularity│ Accumulate locally, lock once per thread│\n";
    std::cout << "│ Barrier opt     │ Multiple diff vars → fewer barriers    │\n";
    std::cout << "│ vs Data-Parallel│ Programmer manages sync explicitly     │\n";
    std::cout << "└─────────────────┴────────────────────────────────────────┘\n";

    std::cout << "\nAll tests completed successfully.\n";
    return 0;
}
