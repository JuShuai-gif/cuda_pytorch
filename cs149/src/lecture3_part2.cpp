// lecture3_part2.cpp - SPMD/ISPC Programming Model Simulation
// =============================================================================
// Key concepts from CS149 Lecture 3:
//  - ISPC: Intel SPMD Program Compiler
//  - SPMD: Single Program, Multiple Data
//  - Gang abstraction: programCount instances running concurrently
//  - programCount: number of simultaneously executing instances in the gang
//  - programIndex: ID of the current instance (0 to programCount-1)
//  - uniform: variable with same value across all instances (optimization)
//  - varying: variable that differs per instance (default)
//  - Interleaved assignment: idx = i + programIndex (contiguous vector load)
//  - Blocked assignment: start = programIndex * count (needs gather/scatter)
//  - foreach: ISPC abstraction for parallel iteration
//  - reduce_add(): cross-instance reduction primitive
//  - Abstraction vs. implementation: SPMD model vs. SIMD hardware
//
// Compile: g++ -std=c++17 -O2 lecture3_part2.cpp -o lecture3_part2
// =============================================================================

#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cassert>
#include <numeric>

// =============================================================================
// ISPC-like Gang abstraction simulation
// =============================================================================
class ISPCGang {
public:
    int programCount; // number of instances in the gang

    explicit ISPCGang(int count) : programCount(count) {
        instances_.resize(count);
    }

    // Each instance stores its local (varying) variables
    struct Instance {
        int programIndex;
        float value = 0.0f;
        float numer = 0.0f;
        float partial = 0.0f;
        // ... other per-instance state
    };

    Instance& instance(int idx) { return instances_[idx]; }
    const Instance& instance(int idx) const { return instances_[idx]; }

    // Cross-instance reduction: sum all instances' partial values
    float reduce_add(float* partials) {
        float sum = 0.0f;
        for (int i = 0; i < programCount; i++) {
            sum += partials[i];
        }
        return sum;
    }

    // Barrier: all instances synchronize (conceptual)
    void barrier() {
        // In real ISPC, this is guaranteed by SIMD lockstep execution
    }

private:
    std::vector<Instance> instances_;
};

// ---------------------------------------------------------------------------
// sin(x) Taylor expansion (same as Lecture 2's function)
// ---------------------------------------------------------------------------
float sin_taylor(float x, int terms) {
    float value = x;
    float numer = x * x * x;
    float denom = 6.0f;
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
// ISPC-style sinx with interleaved assignment
// Matches the lecture's ispc_sinx() with programCount and programIndex
// ---------------------------------------------------------------------------
void ispc_sinx_interleaved(int N, int terms, const float* x, float* result) {
    const int PROGRAM_COUNT = 8;
    ISPCGang gang(PROGRAM_COUNT);

    // Simulate the gang executing the ISPC function
    // In ISPC: for (uniform int i=0; i<N; i+=programCount)
    for (int i = 0; i < N; i += PROGRAM_COUNT) {
        // All programCount instances execute in parallel (SIMD)
        for (int pi = 0; pi < PROGRAM_COUNT && (i + pi) < N; pi++) {
            int idx = i + pi; // idx = i + programIndex
            float value = x[idx];
            float numer = x[idx] * x[idx] * x[idx];
            // uniform variables (same across all instances)
            float denom = 6.0f;
            float sign_val = -1.0f;

            for (int j = 1; j <= terms; j++) {
                value += sign_val * numer / denom;
                numer *= x[idx] * x[idx];
                denom *= (2 * j + 2) * (2 * j + 3);
                sign_val *= -1.0f;
            }
            result[idx] = value;
        }
    }
}

// ---------------------------------------------------------------------------
// ISPC-style sinx with blocked assignment (version 2 from the lecture)
// ---------------------------------------------------------------------------
void ispc_sinx_blocked(int N, int terms, const float* x, float* result) {
    const int PROGRAM_COUNT = 8;
    int count = N / PROGRAM_COUNT; // uniform int count = N / programCount

    // Simulate: each instance processes a contiguous block
    for (int pi = 0; pi < PROGRAM_COUNT; pi++) {
        int start = pi * count; // int start = programIndex * count

        for (int j = 0; j < count; j++) {
            int idx = start + j;
            float value = x[idx];
            float numer = x[idx] * x[idx] * x[idx];
            float denom = 6.0f;
            float sign_val = -1.0f;

            for (int k = 1; k <= terms; k++) {
                value += sign_val * numer / denom;
                numer *= x[idx] * x[idx];
                denom *= (2 * k + 2) * (2 * k + 3);
                sign_val *= -1.0f;
            }
            result[idx] = value;
        }
    }
}

// ---------------------------------------------------------------------------
// ISPC foreach abstraction simulation
// Programmer writes foreach, implementation decides assignment
// ---------------------------------------------------------------------------
void ispc_foreach_sinx(int N, int terms, const float* x, float* result) {
    // foreach implementation: interleave iterations onto program instances
    const int PROGRAM_COUNT = 8;

    for (int loop_i = 0; loop_i < N; loop_i += PROGRAM_COUNT) {
        for (int pi = 0; pi < PROGRAM_COUNT && (loop_i + pi) < N; pi++) {
            int i = loop_i + pi;
            float value = x[i];
            float numer = x[i] * x[i] * x[i];
            float denom = 6.0f;
            float sign_val = -1.0f;

            for (int j = 1; j <= terms; j++) {
                value += sign_val * numer / denom;
                numer *= x[i] * x[i];
                denom *= (2 * j + 2) * (2 * j + 3);
                sign_val *= -1.0f;
            }
            result[i] = value;
        }
    }
}

// ---------------------------------------------------------------------------
// ISPC reduce_add simulation: sum all elements of an array
// (matches the lecture's correct sum_array implementation)
// ---------------------------------------------------------------------------
float ispc_sum_array(int N, const float* x) {
    const int PROGRAM_COUNT = 8;
    std::vector<float> partials(PROGRAM_COUNT, 0.0f);

    // foreach (i = 0 ... N)
    for (int loop_i = 0; loop_i < N; loop_i += PROGRAM_COUNT) {
        for (int pi = 0; pi < PROGRAM_COUNT && (loop_i + pi) < N; pi++) {
            int i = loop_i + pi;
            partials[pi] += x[i]; // each instance accumulates private partial
        }
    }

    // reduce_add: cross-instance sum
    float sum = 0.0f;
    for (float p : partials) sum += p;
    return sum;
}

// ---------------------------------------------------------------------------
// ISPC cross-instance operations: reduce_min simulation
// ---------------------------------------------------------------------------
float ispc_reduce_min(const std::vector<float>& values) {
    if (values.empty()) return 0.0f;
    float min_val = values[0];
    for (float v : values) min_val = std::min(min_val, v);
    return min_val;
}

// ---------------------------------------------------------------------------
// ISPC shift/rotate operation: pass value to instance i+offset
// ---------------------------------------------------------------------------
std::vector<float> ispc_rotate(const std::vector<float>& values, int offset) {
    int n = static_cast<int>(values.size());
    std::vector<float> result(n);
    for (int i = 0; i < n; i++) {
        result[(i + offset) % n] = values[i];
    }
    return result;
}

// ---------------------------------------------------------------------------
// ISPC broadcast: broadcast value from one instance to all
// ---------------------------------------------------------------------------
float ispc_broadcast(const std::vector<float>& values, int index) {
    assert(index >= 0 && index < static_cast<int>(values.size()));
    return values[index];
}

// ---------------------------------------------------------------------------
// Product of 8 elements in log2(8) = 3 steps using ISPC-style cooperation
// (matches the lecture's vec8product example)
// ---------------------------------------------------------------------------
float ispc_vec8product(const float* x) {
    const int PROGRAM_COUNT = 8;
    std::vector<float> val(PROGRAM_COUNT);

    // Step 1: each instance loads its value
    for (int pi = 0; pi < PROGRAM_COUNT; pi++) {
        val[pi] = x[pi];
    }

    // Step 2: shift by 1, multiply even-indexed pairs
    auto val2 = ispc_rotate(val, 1);
    for (int pi = 0; pi < PROGRAM_COUNT; pi++) {
        if (pi % 2 == 0) val[pi] = val[pi] * val2[pi];
    }

    // Step 3: shift by 2, multiply every 4th
    val2 = ispc_rotate(val, 2);
    for (int pi = 0; pi < PROGRAM_COUNT; pi++) {
        if (pi % 4 == 0) val[pi] = val[pi] * val2[pi];
    }

    // Step 4: shift by 4, multiply every 8th (final result in instance 0)
    val2 = ispc_rotate(val, 4);
    for (int pi = 0; pi < PROGRAM_COUNT; pi++) {
        if (pi % 8 == 0) val[pi] = val[pi] * val2[pi];
    }

    return val[0];
}

// ---------------------------------------------------------------------------
// Demonstrate interleaved vs blocked assignment
// ---------------------------------------------------------------------------
void demo_assignment_strategies(int N) {
    std::cout << "[1] Interleaved vs. Blocked Assignment (programCount=8)\n" << std::endl;

    const int PC = 8;
    int elements_per_instance = N / PC;

    // Show which element each program instance handles
    std::cout << "    Interleaved assignment (idx = i + programIndex):\n";
    std::cout << "    ";
    for (int pi = 0; pi < PC; pi++) {
        std::cout << "PI" << pi << "     ";
    }
    std::cout << "\n    ";
    for (int pi = 0; pi < PC; pi++) std::cout << "--------";
    std::cout << std::endl;

    for (int loop_i = 0; loop_i < N; loop_i += PC) {
        std::cout << "    ";
        for (int pi = 0; pi < PC; pi++) {
            int idx = loop_i + pi;
            if (idx < N)
                std::cout << std::setw(4) << idx << "    ";
            else
                std::cout << "  -     ";
        }
        std::cout << std::endl;
    }

    std::cout << "\n    → Contiguous memory access: vector load (vmovaps) works efficiently\n" 
              << std::endl;

    // Blocked assignment
    std::cout << "    Blocked assignment (start = programIndex * count):\n";
    std::cout << "    ";
    for (int pi = 0; pi < PC; pi++) {
        std::cout << "PI" << pi << "     ";
    }
    std::cout << "\n    ";
    for (int pi = 0; pi < PC; pi++) std::cout << "--------";
    std::cout << std::endl;

    int count = N / PC;
    for (int j = 0; j < count; j++) {
        std::cout << "    ";
        for (int pi = 0; pi < PC; pi++) {
            int idx = pi * count + j;
            std::cout << std::setw(4) << idx << "    ";
        }
        std::cout << std::endl;
    }

    std::cout << "\n    → Non-contiguous: needs gather instruction (vgatherdps)\n";
    std::cout << "    → Gather is more complex and more costly\n" << std::endl;
}

// ---------------------------------------------------------------------------
// Correct vs. incorrect ISPC sum implementations
// ---------------------------------------------------------------------------
void demo_ispc_sum() {
    std::cout << "[2] ISPC sum: correct vs. incorrect implementations\n" << std::endl;

    const int N = 1024;
    std::vector<float> x(N);
    for (int i = 0; i < N; i++) x[i] = static_cast<float>(i + 1);

    float expected = static_cast<float>(N * (N + 1) / 2);

    // Correct: v每个实例私有partial，然后reduce_add
    float result = ispc_sum_array(N, x.data());
    std::cout << "    Correct (private partials + reduce_add):\n";
    std::cout << "    Sum = " << std::fixed << std::setprecision(0) << result 
              << " (expected: " << expected << ") ✓\n" << std::endl;

    std::cout << "    Why incorrect versions fail:\n";
    std::cout << "    - sum of type 'float' (varying): each instance has its own sum,\n"
              << "      but there's no way to combine them\n";
    std::cout << "    - sum of type 'uniform float': all instances share one sum,\n"
              << "      but x[i] has different values per instance → data race\n";
    std::cout << "    - Both generate compile-time type errors in ISPC\n" << std::endl;
}

// ---------------------------------------------------------------------------
// Demonstrate the vec8product (lg(N) parallel product)
// ---------------------------------------------------------------------------
void demo_vec8product() {
    std::cout << "[3] Advanced ISPC cooperation: vec8product\n" << std::endl;

    float x[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float expected = 1*2*3*4*5*6*7*8.0f; // 40320.0

    float result = ispc_vec8product(x);

    std::cout << "    Input:  [1, 2, 3, 4, 5, 6, 7, 8]\n";
    std::cout << "    Product = " << std::fixed << std::setprecision(0) << result 
              << " (expected: " << expected << ")\n";
    std::cout << "    Steps: lg(8) = 3 (using shift + conditional multiply)\n" 
              << std::endl;
}

// =============================================================================
int main() {
    std::cout << "=== CS149 Lecture 3: ISPC/SPMD Programming Model Simulation ===\n" << std::endl;

    const int N = 64; // small N for clear demonstration

    // ---- Part 1: Interleaved vs Blocked Assignment ----
    demo_assignment_strategies(N);

    // ---- Part 2: ISPC Sum ----
    demo_ispc_sum();

    // ---- Part 3: vec8product ----
    demo_vec8product();

    // ---- Part 4: ISPC Key Concepts ----
    std::cout << "[4] ISPC Key Concepts Summary\n" << std::endl;
    std::cout << "    ┌─────────────────────┬──────────────────────────────────────┐\n";
    std::cout << "    │ programCount        │ Number of instances per gang         │\n";
    std::cout << "    │ programIndex        │ ID of current instance (0..PC-1)     │\n";
    std::cout << "    │ uniform             │ Same value for all instances         │\n";
    std::cout << "    │ varying (default)   │ Different value per instance         │\n";
    std::cout << "    │ foreach             │ Parallel iteration (gang-scheduled)  │\n";
    std::cout << "    │ reduce_add()        │ Cross-instance sum (uniform result)  │\n";
    std::cout << "    │ broadcast()         │ Send value from one inst to all      │\n";
    std::cout << "    │ rotate()            │ Pass value to instance i+offset      │\n";
    std::cout << "    │ SPMD                │ Programming abstraction              │\n";
    std::cout << "    │ SIMD                │ Hardware implementation              │\n";
    std::cout << "    └─────────────────────┴──────────────────────────────────────┘\n" << std::endl;

    // ---- Part 5: Key Takeaways ----
    std::cout << "[5] Key Takeaways from Lecture 3 (ISPC)\n" << std::endl;
    std::cout << "    - SPMD: programmer thinks in terms of programCount logical threads\n";
    std::cout << "    - SIMD: compiler emits vector instructions (AVX2, Neon, etc.)\n";
    std::cout << "    - Interleaved assignment: good for contiguous memory (vector loads)\n";
    std::cout << "    - Blocked assignment: may need gather/scatter (more expensive)\n";
    std::cout << "    - foreach: raises abstraction level (think iteration, not instances)\n";
    std::cout << "    - uniform variables: optimization, not needed for correctness\n";
    std::cout << "    - Cross-instance ops enable intra-gang communication\n";
    std::cout << "    - Abstraction vs. implementation is key to understanding\n";
    std::cout << "    - ISPC tasks: separate mechanism for multi-core parallelism\n";

    return 0;
}
