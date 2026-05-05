// lecture1_part1.cpp - Speedup Demo: Parallel Sum with Timing
// =============================================================================
// Key concepts from CS149 Lecture 1:
//  - Speedup formula: speedup(P) = T(1) / T(P)
//  - Amdahl's Law: speedup limited by the serial fraction
//  - Communication overhead limits speedup
//  - Work imbalance limits speedup
//
// Compile: g++ -std=c++17 -O2 -pthread lecture1_part1.cpp -o lecture1_part1
// =============================================================================

#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <random>
#include <cmath>

using namespace std::chrono;

// ---------------------------------------------------------------------------
// Sequential sum (baseline: 1 processor)
// ---------------------------------------------------------------------------
double sequential_sum(const std::vector<double>& data) {
    double total = 0.0;
    for (double val : data) {
        total += val;
    }
    return total;
}

// ---------------------------------------------------------------------------
// Parallel sum with explicit thread management
// Each thread sums a contiguous chunk of the array
// ---------------------------------------------------------------------------
double parallel_sum_chunks(const std::vector<double>& data, int num_threads) {
    size_t n = data.size();
    size_t chunk_size = (n + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;
    std::vector<double> partial_sums(num_threads, 0.0);

    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back([&, t]() {
            size_t start = t * chunk_size;
            size_t end = std::min(start + chunk_size, n);
            double local_sum = 0.0;
            for (size_t i = start; i < end; i++) {
                local_sum += data[i];
            }
            partial_sums[t] = local_sum;
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    // Final reduction (serial step - this is the communication cost)
    double total = 0.0;
    for (double s : partial_sums) {
        total += s;
    }
    return total;
}

// ---------------------------------------------------------------------------
// Simulate unbalanced work distribution (some threads get more work)
// ---------------------------------------------------------------------------
double parallel_sum_unbalanced(const std::vector<double>& data, int num_threads) {
    size_t n = data.size();
    // Give later threads increasingly more work to simulate imbalance
    std::vector<size_t> chunks(num_threads, 0);
    size_t total_assigned = 0;
    for (int t = 0; t < num_threads; t++) {
        // Thread t gets (t+1) times the base chunk size
        chunks[t] = (t + 1) * (n / (num_threads * (num_threads + 1) / 2));
        if (t == num_threads - 1) {
            chunks[t] = n - total_assigned; // last thread gets remainder
        }
        total_assigned += chunks[t];
    }
    // Ensure all data is covered
    if (total_assigned < n) chunks.back() += n - total_assigned;

    std::vector<std::thread> threads;
    std::vector<double> partial_sums(num_threads, 0.0);

    size_t offset = 0;
    for (int t = 0; t < num_threads; t++) {
        size_t chunk = chunks[t];
        size_t start = offset;
        threads.emplace_back([&, t, start, chunk]() {
            double local_sum = 0.0;
            for (size_t i = start; i < start + chunk && i < data.size(); i++) {
                local_sum += data[i];
            }
            partial_sums[t] = local_sum;
        });
        offset += chunk;
    }

    for (auto& th : threads) {
        th.join();
    }

    double total = 0.0;
    for (double s : partial_sums) {
        total += s;
    }
    return total;
}

// ---------------------------------------------------------------------------
// Amdahl's Law calculator
// S_perf(p) = 1 / (1 - f_perf + f_perf / p)
// where f_perf = fraction of work that is parallelizable
// ---------------------------------------------------------------------------
double amdahl_speedup(int processors, double parallel_fraction) {
    return 1.0 / ((1.0 - parallel_fraction) + parallel_fraction / processors);
}

// ---------------------------------------------------------------------------
// Benchmark helper: measures execution time of a function
// ---------------------------------------------------------------------------
template<typename Func, typename... Args>
double benchmark(Func func, Args&&... args) {
    auto start = high_resolution_clock::now();
    func(std::forward<Args>(args)...);
    auto end = high_resolution_clock::now();
    return duration_cast<microseconds>(end - start).count() / 1000.0; // ms
}

// =============================================================================
int main() {
    std::cout << "=== CS149 Lecture 1: Speedup & Parallel Efficiency ===\n" << std::endl;

    // ---- Part 1: Parallel Speedup Measurement ----
    std::cout << "[1] Measuring parallel speedup for array sum\n" << std::endl;

    const size_t N = 100'000'000; // 100 million elements
    std::cout << "    Array size: " << N << " doubles (" 
              << (N * sizeof(double) / (1024.0 * 1024.0)) << " MB)\n" << std::endl;

    // Generate random data
    std::vector<double> data(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < N; i++) {
        data[i] = dist(rng);
    }

    // Benchmark sequential sum
    double seq_time = benchmark([&]() { 
        volatile double r = sequential_sum(data); 
    });

    // Run multiple times for stable results
    double seq_sum = sequential_sum(data);
    std::cout << "    Sequential sum = " << std::fixed << std::setprecision(1) 
              << seq_sum << "\n";
    std::cout << "    Sequential time = " << seq_time << " ms\n" << std::endl;

    // Measure parallel speedup with varying thread counts
    std::cout << "    " << std::left << std::setw(10) << "Threads"
              << std::setw(14) << "Time(ms)" 
              << std::setw(10) << "Speedup"
              << std::setw(12) << "Efficiency" << std::endl;
    std::cout << "    " << std::string(46, '-') << std::endl;

    int max_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (max_threads == 0) max_threads = 8;

    for (int p = 1; p <= max_threads; p++) {
        double par_time = benchmark([&]() {
            volatile double r = parallel_sum_chunks(data, p);
        });
        double speedup = seq_time / par_time;
        double efficiency = speedup / p * 100.0;

        std::cout << "    " << std::left << std::setw(10) << p
                  << std::setw(14) << std::fixed << std::setprecision(2) << par_time
                  << std::setw(10) << std::setprecision(2) << speedup << "x"
                  << std::setw(12) << std::setprecision(1) << efficiency << "%"
                  << std::endl;
    }

    // ---- Part 2: Amdahl's Law Visualization ----
    std::cout << "\n[2] Amdahl's Law: theoretical speedup limits\n" << std::endl;
    std::cout << "    Speedup(P) = 1 / ((1 - f_par) + f_par / P)\n" << std::endl;

    std::vector<double> parallel_fractions = {0.50, 0.75, 0.90, 0.95, 0.99};
    std::vector<int> processor_counts = {1, 2, 4, 8, 16, 32, 64, 128, 1024};

    std::cout << "    " << std::setw(8) << "P";
    for (double f : parallel_fractions) {
        std::cout << std::setw(10) << ("f=" + std::to_string(static_cast<int>(f*100)) + "%");
    }
    std::cout << std::endl;
    std::cout << "    " << std::string(58, '-') << std::endl;

    for (int p : processor_counts) {
        std::cout << "    " << std::setw(8) << p;
        for (double f : parallel_fractions) {
            double sp = amdahl_speedup(p, f);
            std::cout << std::setw(10) << std::fixed << std::setprecision(2) << sp;
        }
        std::cout << std::endl;
    }

    // ---- Part 3: Work Imbalance Demonstration ----
    std::cout << "\n[3] Work imbalance impact on speedup\n" << std::endl;
    std::cout << "    Balanced vs Unbalanced work distribution (4 threads):\n" << std::endl;

    double bal_time = benchmark([&]() {
        volatile double r = parallel_sum_chunks(data, 4);
    });
    double unbal_time = benchmark([&]() {
        volatile double r = parallel_sum_unbalanced(data, 4);
    });

    std::cout << "    Balanced chunks:   " << std::fixed << std::setprecision(2) 
              << bal_time << " ms\n";
    std::cout << "    Unbalanced chunks: " << std::setprecision(2) 
              << unbal_time << " ms\n";
    std::cout << "    Imbalance slowdown: " << std::setprecision(1) 
              << (unbal_time / bal_time - 1.0) * 100 << "%\n";

    // ---- Part 4: Key Takeaways ----
    std::cout << "\n[4] Key Takeaways from Lecture 1\n";
    std::cout << "    - Speedup = T(1) / T(P), upper bounded by P\n";
    std::cout << "    - Communication overhead limits real speedup\n";
    std::cout << "    - Amdahl's Law: serial fraction dominates at scale\n";
    std::cout << "    - Work imbalance reduces efficiency (idle processors)\n";
    std::cout << "    - FAST != EFFICIENT (2x speedup on 10 cores = 20% efficient)\n";

    return 0;
}
