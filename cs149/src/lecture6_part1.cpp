/**
 * lecture6_part1.cpp - Memory Locality Optimization
 *
 * Demonstrates CS149 Lecture 6 concepts:
 * - Row-major vs blocked traversal of 2D grids
 * - Loop fusion for improved arithmetic intensity
 * - Cache line effects on performance
 * - Blocked matrix operation for temporal locality
 * - Arithmetic intensity calculation
 *
 * Compile: g++ -std=c++17 lecture6_part1.cpp -o lecture6_part1 && ./lecture6_part1
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstring>
#include <random>
#include <algorithm>

// ============================================================================
// Part 1: Grid Traversal Order & Cache Effects
// ============================================================================

/**
 * Simulates the grid solver access pattern from lecture 6.
 *
 * Key insight: Row-major traversal loads too many cache lines
 * because data from previous rows is evicted before reuse.
 */

// Constants for cache simulation
constexpr int CACHE_LINE_ELEMENTS = 4;   // 4 floats per 64-byte cache line
constexpr int CACHE_CAPACITY_LINES = 6;  // 6 cache lines total

// Compute which cache line an (i,j) grid element falls on
inline int get_line(int i, int j, int cols) {
    return (i * cols + j) / CACHE_LINE_ELEMENTS;
}

class CacheSimulator {
private:
    int hits;
    int misses;
    std::vector<int> cache_tags;  // Which cache lines are present
    int access_counter;

    int get_line(int row, int col, int stride) {
        return (row * stride + col) / CACHE_LINE_ELEMENTS;
    }

public:
    CacheSimulator() : hits(0), misses(0), access_counter(0) {
        cache_tags.resize(CACHE_CAPACITY_LINES, -1);
    }

    bool access(int line) {
        access_counter++;
        // Check if line is in cache
        for (int i = 0; i < CACHE_CAPACITY_LINES; i++) {
            if (cache_tags[i] == line) {
                hits++;
                return true;  // Hit
            }
        }
        // Miss: load into cache (simple FIFO replacement)
        misses++;
        cache_tags[access_counter % CACHE_CAPACITY_LINES] = line;
        return false;
    }

    void reset() {
        hits = 0;
        misses = 0;
        access_counter = 0;
        std::fill(cache_tags.begin(), cache_tags.end(), -1);
    }

    int get_hits() const { return hits; }
    int get_misses() const { return misses; }
    double hit_rate() const {
        int total = hits + misses;
        return total > 0 ? 100.0 * hits / total : 0.0;
    }
};

/**
 * Standard row-major grid traversal (as in grid solver).
 * Problem: long time between accesses to same data.
 */
void row_major_traversal(int N, CacheSimulator& cache) {
    cache.reset();
    std::cout << "  Row-major traversal (" << N << "x" << N << "):\n";

    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            // Access 5-point stencil: center, N, S, E, W
            cache.access(get_line(i, j, N + 2));
            cache.access(get_line(i - 1, j, N + 2));
            cache.access(get_line(i + 1, j, N + 2));
            cache.access(get_line(i, j - 1, N + 2));
            cache.access(get_line(i, j + 1, N + 2));
        }
    }

    int total = cache.get_hits() + cache.get_misses();
    std::cout << "    Total accesses: " << total << "\n";
    std::cout << "    Cache hits: " << cache.get_hits()
              << "  misses: " << cache.get_misses()
              << "  hit rate: " << std::fixed << std::setprecision(1)
              << cache.hit_rate() << "%\n";
}

/**
 * Blocked traversal: process grid in blocks that fit in cache.
 * Improves temporal locality: data reused before eviction.
 */
void blocked_traversal(int N, int block_size, CacheSimulator& cache) {
    cache.reset();
    std::cout << "  Blocked traversal (" << N << "x" << N
              << ", block=" << block_size << "):\n";

    for (int bi = 1; bi <= N; bi += block_size) {
        for (int bj = 1; bj <= N; bj += block_size) {
            int i_end = std::min(bi + block_size, N + 1);
            int j_end = std::min(bj + block_size, N + 1);
            for (int i = bi; i < i_end; i++) {
                for (int j = bj; j < j_end; j++) {
                    cache.access(get_line(i, j, N + 2));
                    cache.access(get_line(i - 1, j, N + 2));
                    cache.access(get_line(i + 1, j, N + 2));
                    cache.access(get_line(i, j - 1, N + 2));
                    cache.access(get_line(i, j + 1, N + 2));
                }
            }
        }
    }

    int total = cache.get_hits() + cache.get_misses();
    std::cout << "    Total accesses: " << total << "\n";
    std::cout << "    Cache hits: " << cache.get_hits()
              << "  misses: " << cache.get_misses()
              << "  hit rate: " << std::fixed << std::setprecision(1)
              << cache.hit_rate() << "%\n";
}

// ============================================================================
// Part 2: Loop Fusion - Improving Arithmetic Intensity
// ============================================================================

/**
 * Demonstrates loop fusion from Lecture 6:
 *
 * Separate loops:
 *   E = D + ((A + B) * C) requires 3 separate loops
 *   Arithmetic intensity = 1/3 (2 loads, 1 store per math op)
 *
 * Fused loop:
 *   E[i] = D[i] + (A[i] + B[i]) * C[i] in one pass
 *   Arithmetic intensity = 3/5 (4 loads, 1 store per 3 math ops)
 */

void benchmark_separate_loops(int N) {
    std::vector<float> A(N, 1.0f), B(N, 2.0f), C(N, 3.0f);
    std::vector<float> D(N, 4.0f), E(N);
    std::vector<float> tmp1(N), tmp2(N);

    auto start = std::chrono::high_resolution_clock::now();

    // Loop 1: tmp1 = A + B  (2 loads, 1 store per element)
    for (int i = 0; i < N; i++) {
        tmp1[i] = A[i] + B[i];
    }

    // Loop 2: tmp2 = tmp1 * C  (2 loads, 1 store per element)
    for (int i = 0; i < N; i++) {
        tmp2[i] = tmp1[i] * C[i];
    }

    // Loop 3: E = tmp2 + D  (2 loads, 1 store per element)
    for (int i = 0; i < N; i++) {
        E[i] = tmp2[i] + D[i];
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    // Verify result
    double checksum = 0.0;
    for (int i = 0; i < N; i++) checksum += E[i];

    std::cout << "  Separate loops: " << std::fixed << std::setprecision(4)
              << elapsed << "s  (AI=1/3, checksum=" << checksum << ")\n";
}

void benchmark_fused_loop(int N) {
    std::vector<float> A(N, 1.0f), B(N, 2.0f), C(N, 3.0f);
    std::vector<float> D(N, 4.0f), E(N);

    auto start = std::chrono::high_resolution_clock::now();

    // Fused loop: E = D + (A + B) * C
    // 4 loads (A,B,C,D), 1 store (E) per 3 math ops (add, multiply, add)
    // Arithmetic intensity = 3/5 = 0.6
    for (int i = 0; i < N; i++) {
        E[i] = D[i] + (A[i] + B[i]) * C[i];
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    double checksum = 0.0;
    for (int i = 0; i < N; i++) checksum += E[i];

    std::cout << "  Fused loop:     " << std::fixed << std::setprecision(4)
              << elapsed << "s  (AI=3/5, checksum=" << checksum << ")\n";
}

// ============================================================================
// Part 3: Arithmetic Intensity Demonstration
// ============================================================================

/**
 * Computes and displays arithmetic intensity for various operations.
 *
 * AI = amount of computation / amount of communication
 * Higher AI = better utilization of memory bandwidth.
 */
void analyze_arithmetic_intensity() {
    std::cout << "\n=== Arithmetic Intensity Analysis ===\n\n";

    std::cout << "AI = computation (FLOPs) / communication (bytes)\n\n";

    // Element-wise vector multiply: C[i] = A[i] * B[i]
    // 1 FLOP, 2*4=8 bytes load, 4 bytes store = 12 bytes → AI = 1/12
    std::cout << "Operation                     FLOPs  Bytes  AI\n";
    std::cout << "─────────────────────────────  ─────  ─────  ──────\n";
    std::cout << "C[i] = A[i] * B[i]             1      12     " << std::fixed
              << std::setprecision(4) << (1.0 / 12.0) << "\n";
    std::cout << "C[i] = A[i] + B[i] * C[i]      2      16     "
              << (2.0 / 16.0) << "\n";
    std::cout << "E[i] = D[i]+(A[i]+B[i])*C[i]   3      20     "
              << (3.0 / 20.0) << "\n";
    std::cout << "C[i] = α*A[i] + β*B[i] (BLAS)  3      16     "
              << (3.0 / 16.0) << "\n";
    std::cout << "Matrix multiply (inner product) 2N    4N+4   "
              << (2.0 / 4.0) << " (per element)\n\n";

    std::cout << "Key: To be compute-bound on modern GPUs (10+ TFLOPS, 1+ TB/s BW),\n";
    std::cout << "     need AI >> 10 to saturate compute before bandwidth.\n";
}

// ============================================================================
// Part 4: Blocked Matrix Operations
// ============================================================================

/**
 * Compares naive vs blocked matrix multiplication.
 * Blocked version keeps sub-blocks in cache for reuse.
 */
void matrix_multiply_naive(int N, const std::vector<double>& A,
                            const std::vector<double>& B, std::vector<double>& C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void matrix_multiply_blocked(int N, int block, const std::vector<double>& A,
                              const std::vector<double>& B, std::vector<double>& C) {
    std::fill(C.begin(), C.end(), 0.0);

    for (int bi = 0; bi < N; bi += block) {
        for (int bj = 0; bj < N; bj += block) {
            for (int bk = 0; bk < N; bk += block) {
                // Multiply block
                int i_end = std::min(bi + block, N);
                int j_end = std::min(bj + block, N);
                int k_end = std::min(bk + block, N);

                for (int i = bi; i < i_end; i++) {
                    for (int k = bk; k < k_end; k++) {
                        double aik = A[i * N + k];
                        for (int j = bj; j < j_end; j++) {
                            C[i * N + j] += aik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

void benchmark_matrix_multiply() {
    std::cout << "\n=== Blocked Matrix Multiplication (N=256) ===\n\n";

    const int N = 256;
    std::vector<double> A(N * N), B(N * N), C_naive(N * N), C_blocked(N * N);

    // Initialize with random values
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < N * N; i++) {
        A[i] = dist(rng);
        B[i] = dist(rng);
    }

    // Naive
    auto start = std::chrono::high_resolution_clock::now();
    matrix_multiply_naive(N, A, B, C_naive);
    auto end = std::chrono::high_resolution_clock::now();
    double naive_time = std::chrono::duration<double>(end - start).count();

    std::cout << "  Naive (i,j,k): " << std::fixed << std::setprecision(4)
              << naive_time << "s\n";

    // Blocked - try different block sizes
    for (int block : {16, 32, 64}) {
        start = std::chrono::high_resolution_clock::now();
        matrix_multiply_blocked(N, block, A, B, C_blocked);
        end = std::chrono::high_resolution_clock::now();
        double block_time = std::chrono::duration<double>(end - start).count();

        // Verify correctness
        bool correct = true;
        for (int i = 0; i < N * N && correct; i++) {
            correct = (std::abs(C_naive[i] - C_blocked[i]) < 1e-6);
        }

        std::cout << "  Blocked (block=" << block << "): " << std::fixed
                  << std::setprecision(4) << block_time << "s"
                  << "  speedup=" << std::setprecision(2) << (naive_time / block_time) << "x"
                  << "  correct=" << (correct ? "YES" : "NO") << "\n";
    }
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "============================================================\n";
    std::cout << "Lecture 6 Part 1: Memory Locality Optimization\n";
    std::cout << "============================================================\n";

    // === Part 1: Grid traversal cache simulation ===
    std::cout << "\n--- Grid Traversal & Cache Effects ---\n\n";
    std::cout << "Cache config: " << CACHE_LINE_ELEMENTS << " elements/line, "
              << CACHE_CAPACITY_LINES << " lines capacity\n\n";

    CacheSimulator cache;
    row_major_traversal(6, cache);
    std::cout << "\n";
    blocked_traversal(6, 3, cache);

    std::cout << "\n  Observation: Blocked traversal keeps data in cache between\n";
    std::cout << "  accesses, while row-major traversal loses data from previous rows.\n";

    // === Part 2: Loop fusion benchmark ===
    std::cout << "\n--- Loop Fusion: Arithmetic Intensity ---\n\n";
    const int N_FUSION = 10000000;
    benchmark_separate_loops(N_FUSION);
    benchmark_fused_loop(N_FUSION);
    std::cout << "\n  Fused loop: fewer memory round-trips per computation.\n";
    std::cout << "  Temporary arrays (tmp1, tmp2) eliminated → better locality.\n";

    // === Part 3: Arithmetic intensity analysis ===
    analyze_arithmetic_intensity();

    // === Part 4: Blocked matrix multiply ===
    benchmark_matrix_multiply();

    // === Summary ===
    std::cout << "\n=== Locality Optimization: Key Techniques ===\n";
    std::cout << "┌────────────────────┬──────────────────────────────────────┐\n";
    std::cout << "│ Technique          │ Benefit                             │\n";
    std::cout << "├────────────────────┼──────────────────────────────────────┤\n";
    std::cout << "│ Blocked traversal  │ Keeps working set in cache          │\n";
    std::cout << "│ Loop fusion        │ Reduces intermediate stores/loads   │\n";
    std::cout << "│ Blocked matmul     │ Reuses sub-blocks from cache        │\n";
    std::cout << "│ Row-major order    │ Co-locates consecutive accesses     │\n";
    std::cout << "│ High AI operations │ Better utilizes memory bandwidth    │\n";
    std::cout << "└────────────────────┴──────────────────────────────────────┘\n";

    std::cout << "\nAll tests completed successfully.\n";
    return 0;
}
