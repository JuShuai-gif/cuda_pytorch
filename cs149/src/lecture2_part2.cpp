// lecture2_part2.cpp - SIMD Execution & Conditional Masking Simulation
// =============================================================================
// Key concepts from CS149 Lecture 2:
//  - SIMD: Single Instruction, Multiple Data
//  - Idea: amortize cost/complexity of managing an instruction stream across many ALUs
//  - AVX intrinsics: __m256, _mm256_load_ps, _mm256_mul_ps, etc.
//  - Conditional execution in SIMD: mask (discard) output of ALUs
//  - Coherent execution: same instruction sequence for all data elements
//  - Divergent execution: different control flow → masked lanes → reduced throughput
//  - Explicit SIMD (CPU): compiler generates vector instructions
//  - Implicit SIMD (GPU): hardware runs same instruction on multiple threads
//
// Compile: g++ -std=c++17 -O2 lecture2_part2.cpp -o lecture2_part2
// =============================================================================

#include <iostream>
#include <vector>
#include <iomanip>
#include <cstring>
#include <cassert>
#include <random>
#include <chrono>

// ---------------------------------------------------------------------------
// SIMD Vector abstraction (simulating 8-wide SIMD like AVX2)
// Each vector holds 8 elements of type T
// ---------------------------------------------------------------------------
template<typename T, int WIDTH = 8>
class SIMDVector {
public:
    static constexpr int width = WIDTH;
    T data[WIDTH];

    SIMDVector() {
        for (int i = 0; i < WIDTH; i++) data[i] = 0;
    }

    explicit SIMDVector(T val) {
        for (int i = 0; i < WIDTH; i++) data[i] = val;
    }

    // Load from memory (simulates _mm256_load_ps)
    static SIMDVector load(const T* ptr) {
        SIMDVector v;
        for (int i = 0; i < WIDTH; i++) v.data[i] = ptr[i];
        return v;
    }

    // Store to memory (simulates _mm256_store_ps)
    void store(T* ptr) const {
        for (int i = 0; i < WIDTH; i++) ptr[i] = data[i];
    }

    // Element-wise multiplication
    SIMDVector operator*(const SIMDVector& other) const {
        SIMDVector result;
        for (int i = 0; i < WIDTH; i++) result.data[i] = data[i] * other.data[i];
        return result;
    }

    // Element-wise addition
    SIMDVector operator+(const SIMDVector& other) const {
        SIMDVector result;
        for (int i = 0; i < WIDTH; i++) result.data[i] = data[i] + other.data[i];
        return result;
    }

    // Element-wise division
    SIMDVector operator/(const SIMDVector& other) const {
        SIMDVector result;
        for (int i = 0; i < WIDTH; i++) result.data[i] = data[i] / other.data[i];
        return result;
    }

    // Broadcast scalar to all lanes (simulates _mm256_set1_ps)
    static SIMDVector broadcast(T val) {
        return SIMDVector(val);
    }

    void print(const char* label = "") const {
        if (label[0]) std::cout << label << " = [";
        else std::cout << "[";
        for (int i = 0; i < WIDTH; i++) {
            std::cout << std::setw(6) << std::setprecision(1) << std::fixed << data[i];
            if (i < WIDTH - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
};

// ---------------------------------------------------------------------------
// SIMD execution with condition mask
// Masks are bit vectors: bit i = 1 means lane i is active
// ---------------------------------------------------------------------------
class SIMDMask {
public:
    unsigned int bits; // 8-bit mask for 8-wide SIMD

    SIMDMask() : bits(0) {}
    explicit SIMDMask(unsigned int b) : bits(b) {}

    // Create mask from comparison: lane i active if pred[i] is true
    template<int WIDTH>
    static SIMDMask from_comparison(const bool pred[WIDTH]) {
        unsigned int m = 0;
        for (int i = 0; i < WIDTH; i++) {
            if (pred[i]) m |= (1u << i);
        }
        return SIMDMask(m);
    }

    // Count active lanes
    int popcount() const {
        return __builtin_popcount(bits);
    }

    // Inverse mask
    SIMDMask operator~() const {
        return SIMDMask(~bits & 0xFF);
    }

    bool operator[](int i) const {
        return (bits >> i) & 1;
    }

    void print() const {
        std::cout << "[";
        for (int i = 0; i < 8; i++) {
            std::cout << ((bits >> i) & 1 ? "T" : "F");
            if (i < 7) std::cout << ",";
        }
        std::cout << "] (" << popcount() << "/8 active)\n";
    }
};

// ---------------------------------------------------------------------------
// Simulate the conditional execution example from Lecture 2:
//
//   forall (int i from 0 to N) {
//       float t = x[i];
//       <unconditional code>
//       if (t > 0.0) {
//           t = t * t;          // active: lanes where t > 0
//           t = t * 50.0;
//           t = t + 100.0;
//       } else {
//           t = t + 30.0;       // active: lanes where t <= 0
//           t = t / 10.0;
//       }
//       <resume unconditional code>
//       y[i] = t;
//   }
// ---------------------------------------------------------------------------
void demo_simd_conditional_execution() {
    using Vec = SIMDVector<float, 8>;

    std::cout << "[1] SIMD Conditional Execution (Lecture 2 example)\n" << std::endl;

    // Input data: mix of positive and negative values
    float input[8] = {-1.0f, 0.5f, -0.3f, 2.0f, -0.8f, 1.5f, 0.0f, -0.1f};

    Vec t = Vec::load(input);
    std::cout << "    Input values:\n    ";
    t.print();

    // ---- Unconditional code (all lanes active) ----
    std::cout << "\n    [Unconditional: all 8 lanes active]\n";

    // ---- Conditional: if (t > 0.0) ----
    bool pred_true[8], pred_false[8];
    for (int i = 0; i < 8; i++) {
        pred_true[i] = (t.data[i] > 0.0f);
        pred_false[i] = !pred_true[i];
    }

    SIMDMask mask_true = SIMDMask::from_comparison<8>(pred_true);
    SIMDMask mask_false = ~mask_true;

    std::cout << "    Condition (t > 0.0): ";
    mask_true.print();

    // Simulate "then" branch: only lanes where mask is true execute
    // (In real SIMD, masked lanes still execute but results are discarded)
    Vec t_branch = t;
    std::cout << "\n    [THEN branch: " << mask_true.popcount() << " active lanes]\n";
    
    // t = t * t (masked)
    for (int i = 0; i < 8; i++) {
        if (mask_true[i]) t_branch.data[i] = t_branch.data[i] * t_branch.data[i];
    }
    // t = t * 50.0
    for (int i = 0; i < 8; i++) {
        if (mask_true[i]) t_branch.data[i] = t_branch.data[i] * 50.0f;
    }
    // t = t + 100.0
    for (int i = 0; i < 8; i++) {
        if (mask_true[i]) t_branch.data[i] = t_branch.data[i] + 100.0f;
    }

    std::cout << "    After THEN:  ";
    t_branch.print();

    // Simulate "else" branch: only lanes where mask is false execute
    std::cout << "\n    [ELSE branch: " << mask_false.popcount() << " active lanes]\n";
    
    // t = t + 30.0
    for (int i = 0; i < 8; i++) {
        if (mask_false[i]) t_branch.data[i] = t_branch.data[i] + 30.0f;
    }
    // t = t / 10.0
    for (int i = 0; i < 8; i++) {
        if (mask_false[i]) t_branch.data[i] = t_branch.data[i] / 10.0f;
    }

    std::cout << "    After ELSE:  ";
    t_branch.print();

    // Calculate efficiency
    int total_ops = 8 * 5; // 8 lanes × 5 operations (3 then + 2 else)
    int useful_ops = mask_true.popcount() * 3 + mask_false.popcount() * 2;
    double efficiency = static_cast<double>(useful_ops) / total_ops * 100.0;

    std::cout << "\n    Efficiency: " << std::fixed << std::setprecision(1) 
              << efficiency << "% (" << useful_ops << "/" << total_ops 
              << " useful operations)\n" << std::endl;
}

// ---------------------------------------------------------------------------
// Worst-case divergent execution demo:
// Only 1 of 8 lanes takes the "then" branch → 1/8 efficiency
// ---------------------------------------------------------------------------
void demo_worst_case_divergence() {
    using Vec = SIMDVector<float, 8>;

    std::cout << "[2] Worst-Case Divergent Execution (1 of 8 lanes diverges)\n" << std::endl;

    // Only lane 0 is positive, all others negative
    float input[8] = {1.0f, -0.5f, -0.3f, -2.0f, -0.8f, -1.5f, -0.2f, -0.1f};

    Vec t = Vec::load(input);
    std::cout << "    Input: ";
    t.print();

    bool pred[8];
    for (int i = 0; i < 8; i++) pred[i] = (t.data[i] > 0.0f);
    SIMDMask mask = SIMDMask::from_comparison<8>(pred);
    std::cout << "    Mask:  ";
    mask.print();

    // Simulate 3 operations in THEN branch (1 lane active → 3 useful out of 24 total)
    int total_ops = 8 * 3;
    int useful_ops = mask.popcount() * 3;
    double eff = static_cast<double>(useful_ops) / total_ops * 100.0;

    std::cout << "\n    THEN efficiency: " << std::fixed << std::setprecision(1) 
              << eff << "% (" << useful_ops << "/" << total_ops 
              << " useful operations)\n";

    // Overall: THEN (3 ops) + ELSE (2 ops) = 5 ops × 8 lanes = 40 total
    // Useful: 1 lane runs THEN=3, 7 lanes run ELSE=2 = 3+14=17
    int total_all = 8 * 5;
    int useful_all = mask.popcount() * 3 + (8 - mask.popcount()) * 2;
    double eff_all = static_cast<double>(useful_all) / total_all * 100.0;

    std::cout << "    Overall efficiency: " << eff_all << "% (" 
              << useful_all << "/" << total_all << ")\n" << std::endl;

    std::cout << "    Key insight: even with worst-case divergence (1/8),\n"
              << "    overall efficiency is 42.5% because BOTH branches are needed.\n"
              << "    Actually worst case: nested if/else chain that only 1 lane\n"
              << "    takes the most expensive path.\n" << std::endl;
}

// ---------------------------------------------------------------------------
// Compare coherent vs. divergent execution
// ---------------------------------------------------------------------------
void demo_coherent_vs_divergent() {
    std::cout << "[3] Coherent vs. Divergent Execution\n" << std::endl;

    std::cout << "    ┌──────────────────────────────────────────────────────┐\n";
    std::cout << "    │ Coherent execution (GOOD for SIMD):                   │\n";
    std::cout << "    │   All lanes take the same path through code           │\n";
    std::cout << "    │   → 100% SIMD utilization                             │\n";
    std::cout << "    │   Example: sin(x) Taylor series for all elements      │\n";
    std::cout << "    │   (same number of iterations, same operations)        │\n";
    std::cout << "    ├──────────────────────────────────────────────────────┤\n";
    std::cout << "    │ Divergent execution (BAD for SIMD):                   │\n";
    std::cout << "    │   Different lanes take different paths                │\n";
    std::cout << "    │   → Lanes masked off → reduced throughput             │\n";
    std::cout << "    │   Example: conditional per element (if x[i] > 0)      │\n";
    std::cout << "    │   Worst case: 1/WIDTH peak performance                │\n";
    std::cout << "    └──────────────────────────────────────────────────────┘\n" << std::endl;

    std::cout << "    Note: divergent execution is NOT a problem for multi-core\n";
    std::cout << "    execution, since each core can independently fetch/decode\n";
    std::cout << "    different instructions.\n" << std::endl;

    // Job simulator: coherent vs divergent throughput
    const int N = 1'000'000;
    std::vector<float> coherent_data(N, 1.0f);
    std::vector<float> divergent_data(N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < N; i++) divergent_data[i] = dist(rng);

    // Coherent: all data same → predictable branch
    {
        using namespace std::chrono;
        auto start = high_resolution_clock::now();
        float sum = 0;
        for (int i = 0; i < N; i++) {
            if (coherent_data[i] > 0) {
                sum += coherent_data[i];
            } else {
                sum -= coherent_data[i];
            }
        }
        auto end = high_resolution_clock::now();
        double time1 = duration_cast<microseconds>(end - start).count() / 1000.0;
        std::cout << "    Coherent data (branch predictable): " 
                  << std::fixed << std::setprecision(1) << time1 << " ms, sum="
                  << sum << std::endl;
    }

    // Divergent: random data → unpredictable branch
    {
        using namespace std::chrono;
        auto start = high_resolution_clock::now();
        float sum = 0;
        for (int i = 0; i < N; i++) {
            if (divergent_data[i] > 0) {
                sum += divergent_data[i];
            } else {
                sum -= divergent_data[i];
            }
        }
        auto end = high_resolution_clock::now();
        double time2 = duration_cast<microseconds>(end - start).count() / 1000.0;
        std::cout << "    Divergent data (branch unpredictable): " 
                  << std::fixed << std::setprecision(1) << time2 << " ms, sum="
                  << sum << std::endl;
    }

    std::cout << "\n    This demonstrates how branch prediction interacts with\n"
              << "    coherent vs. divergent data patterns on superscalar CPUs.\n"
              << "    (SIMD masking avoids the branch prediction problem)\n" << std::endl;
}

// =============================================================================
int main() {
    std::cout << "=== CS149 Lecture 2: SIMD Execution & Conditional Masking ===\n" << std::endl;

    demo_simd_conditional_execution();
    demo_worst_case_divergence();
    demo_coherent_vs_divergent();

    // ---- Additional: SIMD terminology reference ----
    std::cout << "[4] SIMD Terminology Reference\n" << std::endl;
    std::cout << "    ┌──────────────────────┬───────────────────────────────────────┐\n";
    std::cout << "    │ Intel AVX2           │ 256-bit, 8×32-bit or 4×64-bit         │\n";
    std::cout << "    │ Intel AVX512         │ 512-bit, 16×32-bit                    │\n";
    std::cout << "    │ ARM Neon             │ 128-bit, 4×32-bit                     │\n";
    std::cout << "    │ Explicit SIMD        │ Compiler generates vector instructions │\n";
    std::cout << "    │ Implicit SIMD (GPU)  │ HW runs same instr on multiple threads │\n";
    std::cout << "    │ Coherent execution   │ Same instruction sequence for all data │\n";
    std::cout << "    │ Divergent execution  │ Different control flow per data item   │\n";
    std::cout << "    └──────────────────────┴───────────────────────────────────────┘\n";

    return 0;
}
