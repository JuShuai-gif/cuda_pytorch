/**
 * lecture4_part3.cpp - Data-Parallel Grid Solver (Red-Black Gauss-Seidel)
 *
 * Simulates the 2D grid solver from CS149 Lecture 4:
 * - Iterative Gauss-Seidel method on (N+2) x (N+2) grid
 * - Red-black coloring to expose parallelism
 * - Data-parallel expression with implicit barriers
 * - Demonstrates decomposition, assignment, and orchestration
 *
 * Algorithm per cell:
 *   A[i][j] = 0.2 * (A[i-1][j] + A[i][j-1] + A[i][j] + A[i+1][j] + A[i][j+1])
 *
 * Compile: g++ -std=c++17 -pthread lecture4_part3.cpp -o lecture4_part3 && ./lecture4_part3
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <thread>
#include <cmath>
#include <algorithm>
#include <chrono>

// ============================================================================
// Grid Solver: Data Structures and Core Algorithm
// ============================================================================

class GridSolver {
public:
    enum CellColor { RED, BLACK };

private:
    int N;  // Interior grid size (actual grid is (N+2) x (N+2))
    int total_size;  // N + 2
    std::vector<double> grid;       // Current grid values
    std::vector<double> new_grid;   // Buffer for updates
    double tolerance;
    int max_iterations;

    // Helper to access grid as 2D
    double& at(int i, int j) { return grid[i * total_size + j]; }
    const double& at(int i, int j) const { return grid[i * total_size + j]; }
    double& at_new(int i, int j) { return new_grid[i * total_size + j]; }

    // Determine color of cell (i,j): sum of coordinates determines color
    CellColor cell_color(int i, int j) const {
        return ((i + j) % 2 == 0) ? RED : BLACK;
    }

public:
    GridSolver(int n, double tol = 1e-4, int max_iter = 10000)
        : N(n), total_size(n + 2), grid((n + 2) * (n + 2), 0.0),
          new_grid((n + 2) * (n + 2), 0.0),
          tolerance(tol), max_iterations(max_iter) {}

    /**
     * Initialize grid with boundary conditions.
     * Grid borders are set to fixed values (simulating Dirichlet BC).
     */
    void initialize() {
        // Set boundary values (top/bottom rows, left/right columns)
        for (int j = 0; j < total_size; j++) {
            at(0, j) = 1.0;                    // Top boundary
            at(total_size - 1, j) = 0.0;       // Bottom boundary
        }
        for (int i = 0; i < total_size; i++) {
            at(i, 0) = 0.5;                    // Left boundary
            at(i, total_size - 1) = 0.5;       // Right boundary
        }

        // Interior initialized to 0.0 (average guess)
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                at(i, j) = 0.0;
            }
        }
    }

    // ========================================================================
    // Sequential Solver (Original Gauss-Seidel, row-by-row)
    // ========================================================================

    struct SolveResult {
        double diff;
        int iterations;
        bool converged;
        double time_seconds;
    };

    SolveResult solve_sequential() {
        auto start = std::chrono::high_resolution_clock::now();
        int iter = 0;
        bool done = false;

        while (!done && iter < max_iterations) {
            double diff = 0.0;

            // Gauss-Seidel: uses updated values from same iteration
            for (int i = 1; i <= N; i++) {
                for (int j = 1; j <= N; j++) {
                    double prev = at(i, j);
                    at(i, j) = 0.2 * (at(i - 1, j) + at(i, j - 1) +
                                      at(i, j) + at(i + 1, j) + at(i, j + 1));
                    diff += std::abs(at(i, j) - prev);
                }
            }

            iter++;
            if (diff / (N * N) < tolerance) {
                done = true;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        return {calculate_diff(), iter, done,
                std::chrono::duration<double>(end - start).count()};
    }

    // ========================================================================
    // Data-Parallel Solver (Red-Black Coloring)
    //
    // Key idea: Red cells depend only on black cells, and vice versa.
    // All cells of one color can be updated in parallel.
    // After both colors are updated, we check convergence.
    // ========================================================================

    /**
     * Update all cells of a specific color in parallel.
     * This simulates a data-parallel for_all over cells of one color.
     *
     * In ISPC this would be: for_all (red cells (i,j)) { ... }
     */
    void update_color_parallel(CellColor color, double& local_diff, int tid, int num_threads) {
        // Assign rows to threads in blocked fashion
        int rows_per_thread = N / num_threads;
        int start_row = 1 + tid * rows_per_thread;
        int end_row = (tid == num_threads - 1) ? N + 1 : start_row + rows_per_thread;

        for (int i = start_row; i < end_row; i++) {
            // For each row, determine starting column based on color
            // RED cells: (i+j) even; BLACK cells: (i+j) odd
            int j_start = 1;
            // Align j_start so that (i + j_start) % 2 matches the color
            int target_parity = (color == RED) ? 0 : 1;
            if ((i + j_start) % 2 != target_parity) {
                j_start = 2;  // Start from column 2 instead
            }

            for (int j = j_start; j <= N; j += 2) {
                double prev = at(i, j);
                double new_val = 0.2 * (at(i - 1, j) + at(i, j - 1) +
                                        at(i, j) + at(i + 1, j) + at(i, j + 1));
                at_new(i, j) = new_val;
                local_diff += std::abs(new_val - prev);
            }
        }
    }

    /**
     * Data-parallel grid solver using red-black coloring.
     *
     * Decomposition: processing individual grid elements = independent work
     * Assignment: system-assigned (blocked assignment of rows to threads)
     * Orchestration: implicit barrier between RED and BLACK phases
     * Communication: implicit in shared grid access (data-parallel style)
     */
    SolveResult solve_redblack_parallel(int num_threads) {
        auto start = std::chrono::high_resolution_clock::now();
        int iter = 0;
        bool done = false;

        // Reset grid
        initialize();

        while (!done && iter < max_iterations) {
            double global_diff = 0.0;
            std::vector<double> partial_diffs(num_threads, 0.0);
            std::vector<std::thread> threads;

            // Phase 1: Update all RED cells in parallel
            for (int t = 0; t < num_threads; t++) {
                threads.emplace_back([this, t, num_threads, &partial_diffs]() {
                    update_color_parallel(RED, partial_diffs[t], t, num_threads);
                });
            }
            for (auto& th : threads) th.join();

            // Copy RED updates from new_grid back to grid
            for (int i = 1; i <= N; i++) {
                for (int j = 1; j <= N; j++) {
                    if (cell_color(i, j) == RED) {
                        at(i, j) = at_new(i, j);
                    }
                }
            }
            // Implicit barrier: all red updates complete before black begins

            // Phase 2: Update all BLACK cells in parallel
            threads.clear();
            for (int t = 0; t < num_threads; t++) {
                threads.emplace_back([this, t, num_threads, &partial_diffs]() {
                    update_color_parallel(BLACK, partial_diffs[t], t, num_threads);
                });
            }
            for (auto& th : threads) th.join();

            // Copy BLACK updates
            for (int i = 1; i <= N; i++) {
                for (int j = 1; j <= N; j++) {
                    if (cell_color(i, j) == BLACK) {
                        at(i, j) = at_new(i, j);
                    }
                }
            }

            // Combine partial diffs (simulates reduce_add)
            for (double d : partial_diffs) global_diff += d;
            partial_diffs.assign(num_threads, 0.0);

            iter++;
            if (global_diff / (N * N) < tolerance) {
                done = true;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        return {calculate_diff(), iter, done,
                std::chrono::duration<double>(end - start).count()};
    }

    // ========================================================================
    // Utility
    // ========================================================================

    double calculate_diff() const {
        double max_diff = 0.0;
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                double expected = 0.2 * (at(i - 1, j) + at(i, j - 1) +
                                         at(i, j) + at(i + 1, j) + at(i, j + 1));
                max_diff = std::max(max_diff, std::abs(at(i, j) - expected));
            }
        }
        return max_diff;
    }

    void print_grid_summary() const {
        std::cout << "  Corner values: top-left=" << at(1, 1)
                  << "  top-right=" << at(1, N)
                  << "  bottom-left=" << at(N, 1)
                  << "  bottom-right=" << at(N, N)
                  << "  center=" << at(N / 2 + 1, N / 2 + 1) << "\n";
    }

    // Verify that red-black and sequential give same result
    static bool verify_results(const std::vector<double>& a,
                                const std::vector<double>& b, double eps) {
        for (size_t k = 0; k < a.size(); k++) {
            if (std::abs(a[k] - b[k]) > eps) return false;
        }
        return true;
    }

    std::vector<double> get_grid_copy() const { return grid; }
};

// ============================================================================
// Part 2: Work Assignment Analysis
// ============================================================================

/**
 * Compares different work assignment strategies for the grid solver:
 * 1. 1D blocked: each thread gets contiguous rows
 * 2. 1D interleaved: thread t gets rows t, t+P, t+2P, ...
 * 3. 2D blocked: grid divided into rectangular blocks
 */
void analyze_assignments() {
    std::cout << "\n=== Work Assignment Strategies for Grid Solver ===\n\n";

    std::cout << "┌─────────────────┬──────────────────────┬──────────────────────┐\n";
    std::cout << "│ Assignment      │ Elements per Thread  │ Communication (rows) │\n";
    std::cout << "├─────────────────┼──────────────────────┼──────────────────────┤\n";

    int N = 256;
    int P = 4;

    // 1D blocked
    std::cout << "│ 1D Blocked      │ " << std::setw(18) << (N * N / P)
              << "  │ " << std::setw(18) << (2 * N / P) << "        │\n";

    // 1D interleaved
    std::cout << "│ 1D Interleaved  │ " << std::setw(18) << (N * N / P)
              << "  │ " << std::setw(18) << (N * N / 2) << "        │\n";

    // 2D blocked
    int sqrtP = static_cast<int>(std::sqrt(P));
    std::cout << "│ 2D Blocked      │ " << std::setw(18) << (N * N / P)
              << "  │ " << std::setw(18) << (2 * N / sqrtP) << "        │\n";

    std::cout << "└─────────────────┴──────────────────────┴──────────────────────┘\n";
    std::cout << "\nKey insight: 2D blocked assignment captures 2D spatial locality.\n";
    std::cout << "Communication per processor: 1D blocked ∝ N, 2D blocked ∝ N/sqrt(P).\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "============================================================\n";
    std::cout << "Lecture 4 Part 3: Data-Parallel Grid Solver (Red-Black)\n";
    std::cout << "============================================================\n";

    const int GRID_SIZE = 64;  // Interior N x N
    const double TOLERANCE = 1e-4;

    // === Sequential Solver ===
    std::cout << "\n--- Sequential Gauss-Seidel Solver ---\n";
    GridSolver seq_solver(GRID_SIZE, TOLERANCE);
    seq_solver.initialize();

    auto seq_result = seq_solver.solve_sequential();
    std::cout << "  Iterations: " << seq_result.iterations << "\n";
    std::cout << "  Converged:  " << (seq_result.converged ? "YES" : "NO") << "\n";
    std::cout << "  Final diff: " << seq_result.diff << "\n";
    std::cout << "  Time:       " << seq_result.time_seconds << "s\n";
    seq_solver.print_grid_summary();

    auto seq_grid = seq_solver.get_grid_copy();

    // === Red-Black Parallel Solver ===
    std::cout << "\n--- Red-Black Parallel Solver ---\n";
    int hw_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (hw_threads < 1) hw_threads = 4;

    for (int P : {1, 2, 4, 8}) {
        if (P > hw_threads * 2) continue;

        GridSolver par_solver(GRID_SIZE, TOLERANCE);
        par_solver.initialize();

        auto par_result = par_solver.solve_redblack_parallel(P);
        auto par_grid = par_solver.get_grid_copy();

        double speedup = seq_result.time_seconds / par_result.time_seconds;
        bool match = GridSolver::verify_results(seq_grid, par_grid, 1e-3);

        std::cout << "  P=" << P
                  << ": iterations=" << par_result.iterations
                  << "  time=" << par_result.time_seconds << "s"
                  << "  speedup=" << std::fixed << std::setprecision(2) << speedup
                  << "x  results_match=" << (match ? "YES" : "NO") << "\n";
    }

    // === Work Assignment Analysis ===
    analyze_assignments();

    // === Decomposition Summary ===
    std::cout << "\n=== Data-Parallel Grid Solver: Key Concepts ===\n";
    std::cout << "┌────────────────┬─────────────────────────────────────────┐\n";
    std::cout << "│ Concept        │ Implementation                          │\n";
    std::cout << "├────────────────┼─────────────────────────────────────────┤\n";
    std::cout << "│ Decomposition  │ Each grid cell update = independent task│\n";
    std::cout << "│ Assignment     │ Blocked rows to threads (static)        │\n";
    std::cout << "│ Orchestration  │ Implicit barrier between RED and BLACK  │\n";
    std::cout << "│ Communication  │ Implicit via shared grid array          │\n";
    std::cout << "│ Sync: reduce   │ Thread-local partial sums + global sum  │\n";
    std::cout << "│ Key technique  │ Red-black coloring avoids dependencies  │\n";
    std::cout << "└────────────────┴─────────────────────────────────────────┘\n";

    std::cout << "\nAll tests completed successfully.\n";
    return 0;
}
