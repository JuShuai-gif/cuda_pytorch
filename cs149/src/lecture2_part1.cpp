// lecture2_part1.cpp - Multi-Core Execution & Parallelism Patterns
// =============================================================================
// Key concepts from CS149 Lecture 2:
//  - Three forms of parallel execution:
//    1. Superscalar: exploit ILP within one instruction stream (automated by HW)
//    2. SIMD: multiple ALUs controlled by same instruction (within a core)
//    3. Multi-core: multiple cores, each running independent instruction streams
//  - Multi-core era: use transistors for more cores instead of fancier single cores
//  - Data-parallel expression: forall construct (independent loop iterations)
//  - Expressing parallelism: C++ threads vs. data-parallel abstractions
//  - Coherent control flow needed for SIMD efficiency
//  - SIMD on CPUs: AVX2 (256-bit → 8x32bit), AVX512, ARM Neon
//
// Compile: g++ -std=c++17 -O2 lecture2_part1.cpp -o lecture2_part1
// =============================================================================

#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <functional>
#include <future>
#include <algorithm>

// =============================================================================
// sin(x) Taylor expansion: sin(x) = x - x^3/3! + x^5/5! - x^7/7! + ...
// (This is the example program from the lecture)
// =============================================================================
float sin_taylor(float x, int terms) {
    float value = x;
    float numer = x * x * x;       // x^3
    float denom = 6.0f;            // 3!
    int sign = -1;

    for (int j = 1; j <= terms; j++) {
        value += sign * numer / denom;
        numer *= x * x;
        denom *= (2 * j + 2) * (2 * j + 3);
        sign *= -1;
    }
    return value;
}

// ---------------------------------------------------------------------------
// Sequential sinx: one element at a time on one core
// ---------------------------------------------------------------------------
void sinx_sequential(int N, int terms, const float* x, float* result) {
    for (int i = 0; i < N; i++) {
        result[i] = sin_taylor(x[i], terms);
    }
}

// ---------------------------------------------------------------------------
// Parallel sinx using C++ threads (multi-core)
// Splits work across threads manually
// ---------------------------------------------------------------------------
void sinx_parallel_threads(int N, int terms, const float* x, float* result,
                            int num_threads) {
    int chunk = (N + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back([=, &result]() {
            int start = t * chunk;
            int end = std::min(start + chunk, N);
            for (int i = start; i < end; i++) {
                result[i] = sin_taylor(x[i], terms);
            }
        });
    }

    for (auto& th : threads) th.join();
}

// ---------------------------------------------------------------------------
// Data-parallel expression: simulates "forall" construct
// This is Kayvon's fictitious forall: programmer declares iterations are independent
// ---------------------------------------------------------------------------
void sinx_parallel_forall(int N, int terms, const float* x, float* result,
                           int num_threads) {
    // The "forall" abstraction: the programmer says "these iterations are independent"
    // The runtime/compiler decides how to map iterations to execution resources
    // Here we simulate automatic decomposition into chunks

    int chunk = (N + num_threads - 1) / num_threads;
    std::vector<std::future<void>> futures;

    for (int t = 0; t < num_threads; t++) {
        int start = t * chunk;
        int end = std::min(start + chunk, N);
        futures.push_back(std::async(std::launch::async, [=, &result]() {
            for (int i = start; i < end; i++) {
                result[i] = sin_taylor(x[i], terms);
            }
        }));
    }

    for (auto& f : futures) f.wait();
}

// ---------------------------------------------------------------------------
// SIMD simulation (8-wide, like AVX2)
// Manually process 8 elements at a time to simulate SIMD vector operations
// ---------------------------------------------------------------------------
void sinx_simd_8wide(int N, int terms, const float* x, float* result) {
    // Process 8 elements at a time (simulating 8-wide SIMD)
    for (int i = 0; i < N; i += 8) {
        float values[8];
        int end = std::min(i + 8, N);

        // "Vector load": load 8 elements
        for (int k = 0; k < 8 && (i + k) < N; k++) {
            values[k] = x[i + k];
        }

        // "Vector compute": compute sin for each element
        for (int k = 0; k < 8 && (i + k) < N; k++) {
            values[k] = sin_taylor(values[k], terms);
        }

        // "Vector store": store 8 results
        for (int k = 0; k < 8 && (i + k) < N; k++) {
            result[i + k] = values[k];
        }
    }
}

// ---------------------------------------------------------------------------
// Combined multi-core + SIMD: 4 cores × 8-wide SIMD = 32 elements in parallel
// (Matches the lecture example: 4-core Intel CPU with AVX2)
// ---------------------------------------------------------------------------
void sinx_multicore_simd(int N, int terms, const float* x, float* result) {
    const int SIMD_WIDTH = 8;
    const int NUM_CORES = 4;
    int total_parallelism = SIMD_WIDTH * NUM_CORES;

    // Each core processes chunks of SIMD_WIDTH elements
    int elements_per_core = ((N + SIMD_WIDTH - 1) / SIMD_WIDTH + NUM_CORES - 1) / NUM_CORES
                            * SIMD_WIDTH;

    std::vector<std::thread> threads;
    for (int t = 0; t < NUM_CORES; t++) {
        threads.emplace_back([=, &result]() {
            int start = t * elements_per_core;
            int end = std::min(start + elements_per_core, N);
            // Each core runs an 8-wide SIMD inner loop
            for (int i = start; i < end; i += SIMD_WIDTH) {
                for (int k = 0; k < SIMD_WIDTH && (i + k) < N; k++) {
                    result[i + k] = sin_taylor(x[i + k], terms);
                }
            }
        });
    }

    for (auto& th : threads) th.join();
}

// ---------------------------------------------------------------------------
// Benchmark helper
// ---------------------------------------------------------------------------
template<typename Func, typename... Args>
double benchmark_ms(Func func, Args&&... args) {
    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    func(std::forward<Args>(args)...);
    auto end = high_resolution_clock::now();
    return duration_cast<microseconds>(end - start).count() / 1000.0;
}

// =============================================================================
int main() {
    std::cout << "=== CS149 Lecture 2: Multi-Core & SIMD Parallelism ===\n" << std::endl;

    // ---- Part 1: The sin(x) Taylor expansion program ----
    std::cout << "[1] Example program: sin(x) Taylor expansion\n" << std::endl;
    std::cout << "    sin(x) = x - x^3/3! + x^5/5! - x^7/7! + ...\n" << std::endl;

    // Test correctness
    float test_x = 0.5f;
    float result = sin_taylor(test_x, 5);
    std::cout << "    sin(0.5) ≈ " << std::fixed << std::setprecision(6) 
              << result << " (std::sin = " << std::sin(test_x) << ")\n" << std::endl;

    // ---- Part 2: Benchmark different execution strategies ----
    const int N = 1'000'000;
    const int TERMS = 5;

    std::vector<float> x(N);
    std::vector<float> y(N, 0.0f);

    // Fill input with values in [-π, π]
    for (int i = 0; i < N; i++) {
        x[i] = (static_cast<float>(i) / N - 0.5f) * 2.0f * static_cast<float>(M_PI);
    }

    std::cout << "[2] Performance comparison (N=" << N << " elements, " << TERMS 
              << " Taylor terms)\n" << std::endl;

    std::cout << "    " << std::setw(30) << "Strategy" 
              << std::setw(12) << "Time(ms)" 
              << std::setw(10) << "Speedup" << std::endl;
    std::cout << "    " << std::string(52, '-') << std::endl;

    // Sequential baseline
    std::fill(y.begin(), y.end(), 0.0f);
    double seq_time = benchmark_ms(sinx_sequential, N, TERMS, x.data(), y.data());
    std::cout << "    " << std::setw(30) << "Sequential (1 core)" 
              << std::setw(12) << std::fixed << std::setprecision(2) << seq_time
              << std::setw(10) << "1.00x" << std::endl;

    // Multi-core: 2 threads
    std::fill(y.begin(), y.end(), 0.0f);
    double par2_time = benchmark_ms(sinx_parallel_threads, N, TERMS, 
                                     x.data(), y.data(), 2);
    std::cout << "    " << std::setw(30) << "Multi-core (2 threads)" 
              << std::setw(12) << std::fixed << std::setprecision(2) << par2_time
              << std::setw(10) << std::setprecision(2) << (seq_time / par2_time) << "x" 
              << std::endl;

    // Multi-core: 4 threads
    std::fill(y.begin(), y.end(), 0.0f);
    double par4_time = benchmark_ms(sinx_parallel_threads, N, TERMS, 
                                     x.data(), y.data(), 4);
    std::cout << "    " << std::setw(30) << "Multi-core (4 threads)" 
              << std::setw(12) << std::setprecision(2) << par4_time
              << std::setw(10) << std::setprecision(2) << (seq_time / par4_time) << "x" 
              << std::endl;

    // SIMD 8-wide (single core)
    std::fill(y.begin(), y.end(), 0.0f);
    double simd_time = benchmark_ms(sinx_simd_8wide, N, TERMS, x.data(), y.data());
    std::cout << "    " << std::setw(30) << "SIMD 8-wide (1 core)" 
              << std::setw(12) << std::setprecision(2) << simd_time
              << std::setw(10) << std::setprecision(2) << (seq_time / simd_time) << "x" 
              << std::endl;

    // Multi-core + SIMD (4 cores × 8-wide)
    std::fill(y.begin(), y.end(), 0.0f);
    double combined_time = benchmark_ms(sinx_multicore_simd, N, TERMS, x.data(), y.data());
    std::cout << "    " << std::setw(30) << "4 cores × 8-wide SIMD" 
              << std::setw(12) << std::setprecision(2) << combined_time
              << std::setw(10) << std::setprecision(2) << (seq_time / combined_time) << "x" 
              << std::endl;

    // ---- Part 3: Three forms of parallel execution ----
    std::cout << "\n[3] Three Forms of Parallel Execution\n" << std::endl;
    std::cout << "    ┌─────────────────┬──────────────────────────────────┐\n";
    std::cout << "    │ Superscalar      │ Exploit ILP within one stream    │\n";
    std::cout << "    │                  │ (HW discovers at runtime)         │\n";
    std::cout << "    ├─────────────────┼──────────────────────────────────┤\n";
    std::cout << "    │ SIMD             │ Multiple ALUs, one instruction   │\n";
    std::cout << "    │                  │ (compiler generates vector ops)  │\n";
    std::cout << "    ├─────────────────┼──────────────────────────────────┤\n";
    std::cout << "    │ Multi-core       │ Multiple independent streams    │\n";
    std::cout << "    │                  │ (software creates threads)       │\n";
    std::cout << "    └─────────────────┴──────────────────────────────────┘\n" << std::endl;

    // ---- Part 4: Compute throughput examples ----
    std::cout << "[4] Real-world processor throughput examples\n" << std::endl;
    std::cout << "    Intel i7-7700K (4 cores × 8-wide AVX2 × 3 ALUs × 4.2 GHz):\n";
    double i7_flops = 4.0 * 8.0 * 3.0 * 4.2e9;
    std::cout << "    → " << std::fixed << std::setprecision(0) << i7_flops / 1e9 
              << " GFLOPs (approx 400 GFLOPs per lecture)\n" << std::endl;

    std::cout << "    NVIDIA V100 (80 SMs × 64 fp32 ALUs × 1.6 GHz):\n";
    double v100_flops = 80.0 * 64.0 * 1.6e9;
    std::cout << "    → " << std::fixed << std::setprecision(0) << v100_flops / 1e12 
              << " TFLOPs (~16 TFLOPs per lecture)\n" << std::endl;

    // ---- Part 5: Coherent vs. Divergent execution ----
    std::cout << "[5] Instruction Stream Coherence\n" << std::endl;
    std::cout << "    Coherent execution: same instruction sequence applies\n"
              << "    to many data elements (NEEDED for SIMD efficiency)\n" << std::endl;
    std::cout << "    Divergent execution: different control flow per element\n"
              << "    → SIMD lanes masked off → reduced throughput\n" << std::endl;

    // ---- Part 6: Key Takeaways ----
    std::cout << "\n[6] Key Takeaways from Lecture 2\n" << std::endl;
    std::cout << "    - Pre multi-core: transistors used for fancy single-core (OoO, branch pred)\n";
    std::cout << "    - Multi-core era: transistors used for more simpler cores\n";
    std::cout << "    - Superscalar: automatic ILP within a core (HW-driven)\n";
    std::cout << "    - SIMD: amortizes control cost over many ALUs (compiler-driven)\n";
    std::cout << "    - Multi-core: thread-level parallelism (programmer-driven)\n";
    std::cout << "    - SIMD on CPUs: AVX2 (256-bit), AVX512 (512-bit), Neon (128-bit)\n";
    std::cout << "    - GPUs: extreme SIMD width (8-32) + many cores\n";
    std::cout << "    - Combined: multi-core × SIMD × frequency = peak throughput\n";

    return 0;
}
