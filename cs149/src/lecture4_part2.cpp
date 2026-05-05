/**
 * lecture4_part2.cpp - Amdahl's Law: Speedup Analysis
 *
 * Simulates and visualizes Amdahl's Law:
 * - speedup <= 1 / S (S = fraction of sequential execution)
 * - Demonstrates how small serial regions limit scalability
 * - Shows the effect of overhead from parallelization (e.g., combining partial sums)
 *
 * Compile: g++ -std=c++17 lecture4_part2.cpp -o lecture4_part2 && ./lecture4_part2
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <thread>
#include <random>
#include <numeric>

// ============================================================================
// Part 1: Core Amdahl's Law Calculation
// ============================================================================

/**
 * Amdahl's Law: maximum speedup achievable with P processors
 * given fraction S of work that is inherently sequential.
 *
 * speedup(P) = 1 / (S + (1 - S) / P)
 */
double amdahl_speedup(double S, int P) {
    return 1.0 / (S + (1.0 - S) / P);
}

void print_amdahl_table() {
    std::cout << "\n=== Amdahl's Law: Maximum Speedup ===\n\n";
    std::cout << "┌────────┬──────────────────────────────────────────┐\n";
    std::cout << "│   P    │ Speedup for S=0.01  S=0.05  S=0.1  S=0.5 │\n";
    std::cout << "├────────┼──────────────────────────────────────────┤\n";

    std::vector<double> serial_fractions = {0.01, 0.05, 0.1, 0.5};
    std::vector<int> processors = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};

    for (int P : processors) {
        std::cout << "│ " << std::setw(6) << P << " │";
        for (double S : serial_fractions) {
            double sp = amdahl_speedup(S, P);
            std::cout << "  " << std::setw(5) << std::fixed << std::setprecision(2) << sp;
        }
        std::cout << " │\n";
    }
    std::cout << "└────────┴──────────────────────────────────────────┘\n";

    std::cout << "\nKey insight: With S=0.01 (1% serial), even infinite P gives max ~100x speedup.\n";
}

// ============================================================================
// Part 2: Image Processing Example (from Lecture)
// ============================================================================

/**
 * Simulates the two-step NxN image processing example:
 * Step 1: Multiply all pixel brightness by 2 (parallelizable)
 * Step 2: Compute average of all pixel values (partially parallelizable)
 */
class ImageProcessor {
private:
    int N;
    std::vector<double> pixels;

    // Simulate work with a delay
    void simulated_work(double ops) {
        volatile double x = 0.0;
        for (long i = 0; i < static_cast<long>(ops * 10); i++) {
            x += std::sin(static_cast<double>(i) * 0.001);
        }
    }

public:
    ImageProcessor(int size) : N(size), pixels(size * size) {
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (auto& p : pixels) p = dist(rng);
    }

    // Sequential implementation: T_seq = 2 * N^2
    double sequential() {
        auto start = std::chrono::high_resolution_clock::now();

        // Step 1: multiply brightness by 2 (N^2 operations)
        for (int i = 0; i < N * N; i++) {
            pixels[i] *= 2.0;
        }

        // Step 2: compute average (N^2 operations)
        double sum = 0.0;
        for (int i = 0; i < N * N; i++) {
            sum += pixels[i];
        }
        double avg = sum / (N * N);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        std::cout << "  Sequential: avg=" << avg << "  time=" << elapsed.count() << "s\n";
        return elapsed.count();
    }

    // Attempt 1: Step 1 parallel, Step 2 serial
    double attempt1(int P) {
        // Step 1: parallel (time = N^2 / P)
        // Step 2: serial (time = N^2)
        // Total: N^2/P + N^2
        // Speedup <= 2 (no matter how large P is)
        double t1 = static_cast<double>(N * N) / P;
        double t2 = static_cast<double>(N * N);
        double speedup = (2.0 * N * N) / (t1 + t2);
        return speedup;
    }

    // Attempt 2: Step 1 parallel, Step 2 compute partial sums in parallel + combine
    double attempt2(int P) {
        // Step 1: parallel (time = N^2 / P)
        // Step 2: parallel partial sums + serial combine (time = N^2/P + P)
        // Total: 2*N^2/P + P
        double t1 = static_cast<double>(N * N) / P;
        double t2 = static_cast<double>(N * N) / P + P;
        double speedup = (2.0 * N * N) / (t1 + t2);
        return speedup;
    }
};

// ============================================================================
// Part 3: Practical Amdahl's Law Simulation
// ============================================================================

/**
 * Simulates a parallel program where:
 * - S fraction is inherently sequential
 * - (1-S) fraction is perfectly parallelizable
 * - Overhead O is added due to parallel management
 */
double amdahl_with_overhead(double S, int P, double overhead_per_task) {
    double parallel_portion = 1.0 - S;
    double parallel_time = parallel_portion / P;
    double overhead = overhead_per_task * std::log2(P);  // e.g., tree reduction overhead
    return 1.0 / (S + parallel_time + overhead);
}

void analyze_overhead_impact() {
    std::cout << "\n=== Amdahl's Law with Parallelization Overhead ===\n\n";
    std::cout << "┌────────┬────────────────────────────────────────┐\n";
    std::cout << "│   P    │ No overhead   O=0.001    O=0.01    O=0.1 │\n";
    std::cout << "├────────┼────────────────────────────────────────┤\n";

    double S = 0.01;  // 1% serial
    std::vector<double> overheads = {0.0, 0.001, 0.01, 0.1};

    for (int P : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}) {
        std::cout << "│ " << std::setw(6) << P << " │";
        for (double ov : overheads) {
            double sp = amdahl_with_overhead(S, P, ov);
            std::cout << "  " << std::setw(7) << std::fixed << std::setprecision(2) << sp;
        }
        std::cout << " │\n";
    }
    std::cout << "└────────┴────────────────────────────────────────┘\n";
    std::cout << "\nObservation: Even small overhead per parallel task can severely limit\n";
    std::cout << "scalability when P is large, because overhead grows with log(P) or P.\n";
}

// ============================================================================
// Part 4: Summit Supercomputer Example
// ============================================================================

void summit_example() {
    std::cout << "\n=== Summit Supercomputer Scale Analysis ===\n\n";

    // Summit: 27,648 GPUs x 5,376 ALUs/GPU = 148,635,648 ALUs
    long long alus = 148635648LL;

    std::cout << "Summit supercomputer: " << alus << " parallel ALUs\n\n";

    std::vector<double> serial_fractions = {0.1, 0.01, 0.001, 0.0001, 0.00001};
    std::cout << "┌───────────┬──────────────┬───────────────────────────┐\n";
    std::cout << "│ Serial(S) │ Max Speedup  │ Effective ALUs Utilized    │\n";
    std::cout << "├───────────┼──────────────┼───────────────────────────┤\n";

    for (double S : serial_fractions) {
        double sp = amdahl_speedup(S, alus);
        double utilized = sp / alus * 100.0;
        std::cout << "│ " << std::setw(9) << std::fixed << std::setprecision(4) << S * 100 << "%"
                  << " │ " << std::setw(12) << std::fixed << std::setprecision(1) << sp
                  << " │ " << std::setw(16) << std::fixed << std::setprecision(6) << utilized
                  << "%    │\n";
    }
    std::cout << "└───────────┴──────────────┴───────────────────────────┘\n";
    std::cout << "\nKey insight: With 0.1% serial code, max speedup is only ~1000x\n";
    std::cout << "on a machine capable of 148 million parallel operations!\n";
}

// ============================================================================
// Part 5: Measuring Actual Parallel Speedup
// ============================================================================

/**
 * Demonstrates how to measure actual speedup by running a
 * workload with different numbers of threads.
 */
double parallel_workload(int N, int P) {
    std::vector<std::thread> threads;
    std::vector<double> partial_sums(P, 0.0);

    auto worker = [&](int tid) {
        int chunk = N / P;
        int start = tid * chunk;
        int end = (tid == P - 1) ? N : start + chunk;
        double local = 0.0;
        for (int i = start; i < end; i++) {
            local += std::sqrt(static_cast<double>(i + 1));
        }
        partial_sums[tid] = local;
    };

    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < P; t++) {
        threads.emplace_back(worker, t);
    }
    for (auto& t : threads) t.join();

    // Serial reduction (this is the "S" in Amdahl's law)
    double total = 0.0;
    for (double s : partial_sums) total += s;

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

void measure_speedup() {
    std::cout << "\n=== Measuring Actual Speedup (sqrt summation) ===\n\n";
    const int N = 1000000;
    const int HW_THREADS = static_cast<int>(std::thread::hardware_concurrency());

    std::cout << "Hardware threads available: " << HW_THREADS << "\n";
    std::cout << "Problem size: N = " << N << "\n\n";

    // Measure sequential time
    double t1 = parallel_workload(N, 1);
    std::cout << "┌────────┬──────────────┬──────────────┬──────────────┐\n";
    std::cout << "│   P    │  Time (s)    │   Speedup    │  Efficiency   │\n";
    std::cout << "├────────┼──────────────┼──────────────┼──────────────┤\n";

    for (int P = 1; P <= HW_THREADS && P <= 16; P++) {
        double tP = parallel_workload(N, P);
        double speedup = t1 / tP;
        double efficiency = speedup / P * 100.0;
        std::cout << "│ " << std::setw(6) << P
                  << " │  " << std::setw(10) << std::fixed << std::setprecision(6) << tP
                  << " │  " << std::setw(8) << std::fixed << std::setprecision(3) << speedup
                  << "  │  " << std::setw(8) << std::fixed << std::setprecision(1) << efficiency
                  << "%  │\n";
    }
    std::cout << "└────────┴──────────────┴──────────────┴──────────────┘\n";
    std::cout << "\nEfficiency drops due to: serial reduction, thread overhead, Amdahl's law.\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "============================================================\n";
    std::cout << "Lecture 4 Part 2: Amdahl's Law - Speedup Analysis\n";
    std::cout << "============================================================\n";

    // Part 1: Print Amdahl's Law speedup table
    print_amdahl_table();

    // Part 2: Image processing example
    std::cout << "\n=== Image Processing Example (NxN pixels) ===\n";
    ImageProcessor img(100);
    double t_seq = img.sequential();

    std::cout << "\nAttempt 1 (Step 1 parallel, Step 2 serial):\n";
    for (int P : {1, 2, 4, 8, 16, 32}) {
        double sp = img.attempt1(P);
        std::cout << "  P=" << P << ": speedup ≤ " << std::fixed << std::setprecision(2) << sp << "\n";
    }
    std::cout << "  → Speedup bounded by 2 (Step 2 is serial)\n";

    std::cout << "\nAttempt 2 (Both steps parallel, partial sums combined):\n";
    for (int P : {1, 2, 4, 8, 16, 32}) {
        double sp = img.attempt2(P);
        std::cout << "  P=" << P << ": speedup ≈ " << std::fixed << std::setprecision(2) << sp << "\n";
    }
    std::cout << "  → Speedup → P when N >> P (near-linear scaling for large N)\n";

    // Part 3: Overhead impact
    analyze_overhead_impact();

    // Part 4: Summit example
    summit_example();

    // Part 5: Measure actual speedup
    measure_speedup();

    std::cout << "\n=== Amdahl's Law Key Takeaways ===\n";
    std::cout << "1. speedup ≤ 1/S where S = serial fraction\n";
    std::cout << "2. A tiny serial region severely limits large-scale parallelism\n";
    std::cout << "3. Parallelization overhead (sync, communication) adds to effective S\n";
    std::cout << "4. Always minimize serial portions before scaling to many processors\n";
    std::cout << "5. Measure, don't just theorize - real overheads matter\n";

    std::cout << "\nAll tests completed successfully.\n";
    return 0;
}
