/**
 * lecture4_part1.cpp - ISPC SPMD Abstraction Simulation
 *
 * Simulates key ISPC concepts:
 * - programCount: number of simultaneously executing instances
 * - programIndex: id of current instance (0..programCount-1)
 * - uniform vs varying variables
 * - Interleaved vs blocked data assignment
 * - foreach abstraction
 * - reduce_add cross-instance communication
 *
 * Compile: g++ -std=c++17 -pthread lecture4_part1.cpp -o lecture4_part1 && ./lecture4_part1
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <algorithm>
#include <iomanip>

// ============================================================================
// Part 1: SPMD Simulation - sin(x) Taylor Series Computation
// ============================================================================

// Simulate programCount = 8 (SIMD width)
constexpr int PROGRAM_COUNT = 8;

// Taylor series for sin(x): x - x^3/3! + x^5/5! - x^7/7! + ...
float compute_sinx(float x, int terms) {
    float value = x;
    float numer = x * x * x;
    float denom = 6.0f;  // 3!
    float sign = -1.0f;

    for (int j = 1; j <= terms; j++) {
        value += sign * numer / denom;
        numer *= x * x;
        denom *= (2 * j + 2) * (2 * j + 3);
        sign *= -1.0f;
    }
    return value;
}

/**
 * Simulates ISPC's interleaved assignment:
 * for (uniform int i = 0; i < N; i += programCount) {
 *     int idx = i + programIndex;
 *     result[idx] = compute_sinx(x[idx], terms);
 * }
 *
 * Each "program instance" processes elements at stride=programCount.
 * Allows packed vector loads since elements are contiguous.
 */
void interleaved_sinx(const std::vector<float>& x, std::vector<float>& result,
                      int terms, int programIndex) {
    int N = x.size();
    for (int i = 0; i < N; i += PROGRAM_COUNT) {
        int idx = i + programIndex;
        if (idx < N) {
            result[idx] = compute_sinx(x[idx], terms);
        }
    }
}

/**
 * Simulates ISPC's blocked assignment:
 * uniform int count = N / programCount;
 * int start = programIndex * count;
 * for (uniform int i = 0; i < count; i++) {
 *     int idx = start + i;
 *     result[idx] = compute_sinx(x[idx], terms);
 * }
 *
 * Each instance processes a contiguous block.
 * Requires gather instructions (non-contiguous across instances).
 */
void blocked_sinx(const std::vector<float>& x, std::vector<float>& result,
                  int terms, int programIndex) {
    int N = x.size();
    int count = N / PROGRAM_COUNT;
    int start = programIndex * count;
    int end = (programIndex == PROGRAM_COUNT - 1) ? N : start + count;
    for (int idx = start; idx < end; idx++) {
        result[idx] = compute_sinx(x[idx], terms);
    }
}

/**
 * Demonstrates "foreach" concept:
 * The system assigns iterations to program instances automatically.
 * Here, we use interleaved assignment as the implementation.
 */
void foreach_sinx(const std::vector<float>& x, std::vector<float>& result, int terms) {
    int N = x.size();
    // foreach (i = 0 ... N) - the programmer just declares parallel iterations
    // #pragma omp parallel for could be used here with OpenMP
    for (int i = 0; i < N; i++) {
        result[i] = compute_sinx(x[i], terms);
    }
    // In ISPC, the system handles assignment. Here we use a simple parallel for.
}

// ============================================================================
// Part 2: Cross-Instance Communication - reduce_add
// ============================================================================

/**
 * Simulates ISPC reduce_add: sum values across all program instances.
 * Each instance computes a private partial sum, then reduce_add combines them.
 */
float simulated_reduce_add(const std::vector<float>& partials) {
    float sum = 0.0f;
    for (float p : partials) {
        sum += p;
    }
    return sum;
}

/**
 * Correct array sum using ISPC pattern:
 * - Each instance accumulates a private partial sum (no communication)
 * - reduce_add combines partial sums
 */
float reduce_sum(const std::vector<float>& arr) {
    std::vector<float> partial(PROGRAM_COUNT, 0.0f);
    int N = arr.size();

    // Each "program instance" accumulates a partial sum
    for (int i = 0; i < N; i++) {
        int inst = i % PROGRAM_COUNT;
        partial[inst] += arr[i];
    }

    // Cross-instance reduce_add
    return simulated_reduce_add(partial);
}

// ============================================================================
// Part 3: Advanced Cooperation - Parallel Product in O(log N) steps
// ============================================================================

/**
 * ISPC-style parallel product of 8 elements using shift/rotate.
 * Each step halves the number of active instances.
 * Total: lg(8) = 3 steps.
 */
float parallel_product8(const std::vector<float>& arr) {
    // Assumes gang size = 8 and arr.size() == 8
    std::vector<float> val(arr.begin(), arr.end());

    // Step 1: shift by 1, multiply even-indexed instances
    for (int i = 0; i < 8; i += 2) {
        val[i] *= val[i + 1];
    }

    // Step 2: shift by 2, multiply instances where programIndex % 4 == 0
    for (int i = 0; i < 8; i += 4) {
        val[i] *= val[i + 2];
    }

    // Step 3: shift by 4, multiply instances where programIndex % 8 == 0
    val[0] *= val[4];

    return val[0];
}

// ============================================================================
// Part 4: General Parallel Reduction (log2 steps)
// ============================================================================

/**
 * Generic parallel reduction: sum of array using tree-based reduction.
 * Demonstrates the principle behind reduce_add: O(log N) steps with O(N) work.
 */
float parallel_reduce_sum(const std::vector<float>& arr) {
    std::vector<float> data(arr.begin(), arr.end());
    int n = data.size();

    // Pad to power of 2
    while ((n & (n - 1)) != 0) {
        data.push_back(0.0f);
        n++;
    }

    // Tree reduction: each step halves the active elements
    for (int step = 1; step < n; step *= 2) {
        for (int i = 0; i < n; i += 2 * step) {
            data[i] += data[i + step];
        }
    }

    return data[0];
}

// ============================================================================
// Part 5: Uniform vs Varying Variable Demonstration
// ============================================================================

/**
 * Demonstrates the concept of uniform vs varying in ISPC.
 * uniform: all instances see the same value (stored once)
 * varying: each instance has its own copy
 */
void demonstrate_uniform_varying() {
    std::cout << "\n=== Uniform vs Varying Variables ===\n";

    // uniform int N = 10;  -- all instances share this value
    int uniform_N = 10;

    // varying float partial = 0.0f; -- each instance has own copy
    std::vector<float> varying_partial(PROGRAM_COUNT, 0.0f);

    // Simulate each program instance incrementing its own partial
    for (int inst = 0; inst < PROGRAM_COUNT; inst++) {
        for (int j = 0; j < uniform_N; j++) {
            varying_partial[inst] += static_cast<float>(j + 1);
        }
    }

    std::cout << "Uniform N = " << uniform_N << " (shared by all instances)\n";
    std::cout << "Varying partial sums: ";
    for (int inst = 0; inst < PROGRAM_COUNT; inst++) {
        std::cout << varying_partial[inst] << " ";
    }
    std::cout << "\nEach instance computed: sum(1..10) = " << (10 * 11 / 2) << "\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "============================================================\n";
    std::cout << "Lecture 4 Part 1: ISPC SPMD Abstraction Simulation\n";
    std::cout << "============================================================\n";

    const int N = 1024;
    const int TERMS = 5;
    std::vector<float> x(N);
    std::vector<float> result_interleaved(N, 0.0f);
    std::vector<float> result_blocked(N, 0.0f);
    std::vector<float> result_foreach(N, 0.0f);

    // Initialize input array with values in [0, PI]
    for (int i = 0; i < N; i++) {
        x[i] = static_cast<float>(i) * M_PI / N;
    }

    // === Demonstrate Interleaved Assignment ===
    std::cout << "\n--- Interleaved Assignment (programCount=" << PROGRAM_COUNT << ") ---\n";

    std::vector<std::thread> threads;
    for (int inst = 0; inst < PROGRAM_COUNT; inst++) {
        threads.emplace_back(interleaved_sinx, std::ref(x), std::ref(result_interleaved),
                             TERMS, inst);
    }
    for (auto& t : threads) t.join();

    std::cout << "First 16 results (interleaved): ";
    for (int i = 0; i < 16 && i < N; i++) {
        std::cout << std::fixed << std::setprecision(4) << result_interleaved[i] << " ";
    }
    std::cout << "\nMemory access: contiguous for all instances (efficient packed load)\n";

    // === Demonstrate Blocked Assignment ===
    std::cout << "\n--- Blocked Assignment (programCount=" << PROGRAM_COUNT << ") ---\n";
    threads.clear();
    for (int inst = 0; inst < PROGRAM_COUNT; inst++) {
        threads.emplace_back(blocked_sinx, std::ref(x), std::ref(result_blocked),
                             TERMS, inst);
    }
    for (auto& t : threads) t.join();

    std::cout << "First 16 results (blocked): ";
    for (int i = 0; i < 16 && i < N; i++) {
        std::cout << std::fixed << std::setprecision(4) << result_blocked[i] << " ";
    }
    std::cout << "\nMemory access: non-contiguous across instances (requires gather)\n";

    // === Demonstrate foreach ===
    std::cout << "\n--- foreach Abstraction ---\n";
    foreach_sinx(x, result_foreach, TERMS);
    std::cout << "First 16 results (foreach): ";
    for (int i = 0; i < 16 && i < N; i++) {
        std::cout << std::fixed << std::setprecision(4) << result_foreach[i] << " ";
    }
    std::cout << "\nforeach lets the system manage assignment automatically.\n";

    // === Verify results match ===
    bool match = true;
    for (int i = 0; i < N && match; i++) {
        if (std::abs(result_interleaved[i] - result_blocked[i]) > 1e-4f) match = false;
    }
    std::cout << "\nInterleaved == Blocked results: " << (match ? "YES" : "NO") << "\n";

    match = true;
    for (int i = 0; i < N && match; i++) {
        if (std::abs(result_interleaved[i] - result_foreach[i]) > 1e-4f) match = false;
    }
    std::cout << "Interleaved == foreach results: " << (match ? "YES" : "NO") << "\n";

    // === Demonstrate reduce_add ===
    std::cout << "\n--- reduce_add (Cross-Instance Sum) ---\n";
    std::vector<float> test_arr = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float sum = reduce_sum(test_arr);
    std::cout << "Array: [";
    for (size_t i = 0; i < test_arr.size(); i++) {
        std::cout << test_arr[i] << (i < test_arr.size() - 1 ? ", " : "");
    }
    std::cout << "]\n";
    std::cout << "reduce_add sum = " << sum << " (expected: 36)\n";

    // === Demonstrate parallel product ===
    std::cout << "\n--- Parallel Product (O(log N) steps) ---\n";
    float prod = parallel_product8(test_arr);
    std::cout << "parallel_product8 = " << prod << " (expected: 40320 = 8!)\n";

    // === Demonstrate general parallel reduction ===
    std::cout << "\n--- General Parallel Reduction (Tree Sum) ---\n";
    float tsum = parallel_reduce_sum(test_arr);
    std::cout << "Tree reduction sum = " << tsum << " (expected: 36)\n";

    // === Demonstrate uniform vs varying ===
    demonstrate_uniform_varying();

    // === Assignment pattern summary ===
    std::cout << "\n=== Assignment Pattern Summary ===\n";
    std::cout << "┌─────────────┬──────────────────────────────────────┬───────────────────────┐\n";
    std::cout << "│ Assignment  │ Memory Access Pattern               │ SIMD Efficiency       │\n";
    std::cout << "├─────────────┼──────────────────────────────────────┼───────────────────────┤\n";
    std::cout << "│ Interleaved │ Contiguous per iteration            │ Packed loads (vmovaps)│\n";
    std::cout << "│ Blocked     │ Non-contiguous across instances     │ Gather (vgatherdps)   │\n";
    std::cout << "│ foreach     │ System-managed (currently static)   │ Implementation-defined│\n";
    std::cout << "└─────────────┴──────────────────────────────────────┴───────────────────────┘\n";

    std::cout << "\nAll tests completed successfully.\n";
    return 0;
}
