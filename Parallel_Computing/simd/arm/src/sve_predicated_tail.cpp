/**
 * sve_predicated_tail.cpp -- SVE's Killer Feature: Zero-Cost Tail Handling
 *
 * Demonstrates the fundamental difference between NEON-style fixed-width
 * SIMD and SVE's scalable, predicated approach for non-aligned sizes.
 *
 * Scenario: N = 1000003 (prime number, not a multiple of any SIMD width)
 *
 * Approach 1 (NEON-style): Main loop processes in blocks of 4, then a
 *   separate scalar tail loop handles the last 3 elements. This means:
 *   - Extra branch for tail check
 *   - Extra code in I-cache for tail handling
 *   - For very small N, the tail can be a large fraction of work
 *
 * Approach 2 (SVE-style): Single predicated loop with svwhilelt_b32.
 *   The predicate automatically masks inactive lanes in the last iteration.
 *   No separate tail code, no extra branches. Zero-cost tail.
 *
 * The SVE approach is especially valuable for small vectors where the
 * tail would otherwise dominate execution time.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif

#include <arm_neon.h>

extern "C" {
#include "../../common/benchmark.h"
#include "../../common/cpu_features.h"
#include "../../common/check.h"
#include "../../common/random_data.h"
#include "../../common/aligned_buffer.h"
}

// Prime number not a multiple of any common SIMD width
static const int N_PRIME = 1000003;

// ============================================================================
// Scalar baseline
// ============================================================================

static void scalar_add(float* __restrict c,
                        const float* __restrict a,
                        const float* __restrict b, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// ============================================================================
// NEON-style: Main loop (4-wide) + Scalar tail
// ============================================================================
//
// Classic approach: process as many full vectors as possible, then
// handle the remaining elements with a scalar loop. This is how NEON
// (and SSE/AVX) code has been written for decades.

static int neon_style_add(float* __restrict c,
                           const float* __restrict a,
                           const float* __restrict b, int n) {
    int main_loop_count = 0;
    int i = 0;

    // Main loop: process 4 floats at a time with NEON
    for (; i + 3 < n; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        vst1q_f32(&c[i], vaddq_f32(va, vb));
        main_loop_count++;
    }

    // Scalar tail: handle remaining (n % 4) elements
    // This loop executes at most 3 times
    int tail_count = 0;
    for (; i < n; i++) {
        c[i] = a[i] + b[i];
        tail_count++;
    }

    // Return loop counts for analysis
    // Pack into an int: main_loop_count * 1000000 + tail_count
    // (safe since both are small relative to this encoding)
    return main_loop_count * 100 + tail_count;
}

// ============================================================================
// SVE-style: Pure predicated loop, no tail
// ============================================================================
//
// Single loop that handles all elements. The svwhilelt predicate
// automatically generates the correct mask for the final iteration,
// even if fewer elements remain than the vector width.
//
// Zero scalar tail code = smaller I-cache footprint, fewer branches.

#ifdef __ARM_FEATURE_SVE
static int sve_style_add(float* __restrict c,
                          const float* __restrict a,
                          const float* __restrict b, int n) {
    int loop_count = 0;
    int i = 0;

    // Single predicated loop: no tail code at all
    while (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);

        svfloat32_t va = svld1(pg, &a[i]);
        svfloat32_t vb = svld1(pg, &b[i]);

        svfloat32_t vc = svadd_x(pg, va, vb);

        svst1(pg, &c[i], vc);

        i += svcntw();
        loop_count++;
    }

    return loop_count;
}
#endif // __ARM_FEATURE_SVE

// ============================================================================
// Benchmark wrappers
// ============================================================================

static float* g_pt_a = nullptr;
static float* g_pt_b = nullptr;
static float* g_pt_c = nullptr;
static int    g_pt_n = 0;
static volatile int g_pt_count = 0;

__attribute__((noinline))
static void bench_scalar() { scalar_add(g_pt_c, g_pt_a, g_pt_b, g_pt_n); }

__attribute__((noinline))
static void bench_neon_style() {
    g_pt_count = neon_style_add(g_pt_c, g_pt_a, g_pt_b, g_pt_n);
}

#ifdef __ARM_FEATURE_SVE
__attribute__((noinline))
static void bench_sve_style() {
    g_pt_count = sve_style_add(g_pt_c, g_pt_a, g_pt_b, g_pt_n);
}
#endif

// ============================================================================
// Print loop analysis for a given N
// ============================================================================

static void analyze_loops(int n) {
    printf("\n=== Loop Analysis for N=%d ===\n", n);

    // NEON-style analysis (compile-time known: 4-wide)
    int neon_main_iters = n / 4;
    int neon_tail = n % 4;
    printf("  NEON-style (4-wide):\n");
    printf("    Main loop: %d iterations (16 bytes each)\n", neon_main_iters);
    printf("    Tail loop: %d iterations (scalar)\n", neon_tail);
    printf("    Total:     %d loop bodies + 1 tail branch\n",
           neon_main_iters + (neon_tail > 0 ? 1 : 0));

    // SVE-style analysis (width unknown at compile time)
#ifdef __ARM_FEATURE_SVE
    if (cpu_has_sve()) {
        int sve_width = (int)svcntw();
        int sve_iters = (n + sve_width - 1) / sve_width;
        int sve_last_mask = n % sve_width;
        if (sve_last_mask == 0) sve_last_mask = sve_width;

        printf("\n  SVE-style (%d-wide, runtime):\n", sve_width);
        printf("    Predicated loop: %d iterations\n", sve_iters);
        printf("    Last iteration: %d of %d lanes active (predicate masks rest)\n",
               sve_last_mask, sve_width);
        printf("    Tail code:      NONE (predicate handles edges)\n");

        printf("\n  Code size comparison (conceptual):\n");
        printf("    NEON-style: ~30 instructions (main loop ~10, tail ~8, prologue ~12)\n");
        printf("    SVE-style:  ~15 instructions (predicated loop ~10, prologue ~5)\n");
        printf("    Reduction:   ~50%% less I-cache pressure for small kernels\n");
    }
#else
    (void)n;
    printf("\n  SVE analysis unavailable (compile with -march=armv8-a+sve)\n");
#endif
}

// ============================================================================
// Demonstrate tail dominance for small N
// ============================================================================

static void demonstrate_small_n() {
    printf("\n=== Small-N Demonstration: Why Predicated Tail Matters ===\n");

    // For very small N, the scalar tail dominates NEON-style performance
    int small_sizes[] = {3, 4, 5, 7, 9, 15, 16, 17, 31, 63, 127, 255};
    int num_sizes = sizeof(small_sizes) / sizeof(small_sizes[0]);

    printf("  %-8s %-20s %-20s\n", "N", "NEON-style", "SVE-style");
    printf("  %-8s %-20s %-20s\n", "--------",
           "--------------------", "--------------------");

    for (int si = 0; si < num_sizes; si++) {
        int n = small_sizes[si];
        int neon_main = n / 4;
        int neon_tail = n % 4;
        double neon_tail_pct = (neon_main + neon_tail > 0)
            ? 100.0 * neon_tail / (neon_main + neon_tail) : 0.0;

#ifdef __ARM_FEATURE_SVE
        int sve_width = cpu_has_sve() ? (int)svcntw() : 4;
#else
        int sve_width = 4;
#endif
        int sve_iters = (n + sve_width - 1) / sve_width;

        char neon_buf[32], sve_buf[32];
        snprintf(neon_buf, sizeof(neon_buf), "%d main + %d tail (%.0f%%)",
                 neon_main, neon_tail, neon_tail_pct);
        snprintf(sve_buf, sizeof(sve_buf), "%d predicated (0 tail)",
                 sve_iters);

        printf("  %-8d %-20s %-20s\n", n, neon_buf, sve_buf);
    }

    printf("\n  Observation: For N <= 4, NEON-style is 100%% scalar tail.\n");
    printf("  For N <= 8, tail is 25-50%% of the work.\n");
    printf("  SVE has 0%% tail overhead at any N -- the predicate handles it.\n");
}

// ============================================================================
// Main
// ============================================================================

int main() {
    cpu_print_features();

    int n = N_PRIME;
    g_pt_n = n;
    g_pt_a = ALIGNED_ALLOC(float, n, 64);
    g_pt_b = ALIGNED_ALLOC(float, n, 64);
    g_pt_c = ALIGNED_ALLOC(float, n, 64);

    fill_random_f32(g_pt_a, n);
    fill_random_f32(g_pt_b, n);

    // --- Loop analysis (no SVE instructions) ---
    analyze_loops(n);

    // --- Small N demonstration ---
    demonstrate_small_n();

    // --- Correctness ---
    printf("\n=== Correctness Checks ===\n");
    float* ref = ALIGNED_ALLOC(float, n, 64);
    scalar_add(ref, g_pt_a, g_pt_b, n);

    int neon_count = neon_style_add(g_pt_c, g_pt_a, g_pt_b, n);
    CHECK_NEAR_ARRAY(g_pt_c, ref, n, 1e-6f, "NEON-style add vs scalar");
    printf("  NEON-style loop counts: main=%d, tail=%d, total=%d\n",
           neon_count / 100, neon_count % 100,
           neon_count / 100 + (neon_count % 100 > 0 ? 1 : 0));

#ifdef __ARM_FEATURE_SVE
    if (cpu_has_sve()) {
        int sve_count = sve_style_add(g_pt_c, g_pt_a, g_pt_b, n);
        CHECK_NEAR_ARRAY(g_pt_c, ref, n, 1e-6f, "SVE-style add vs scalar");
        printf("  SVE-style loop count: %d (no tail)\n", sve_count);

        printf("\n  Loop code size: NEON=%d bytes+tail, SVE=%d bytes (no tail, unified)\n",
               (int)(n/4 * 4 * 4 + 4 * 3 * 4),  // rough: 4 instr * 4 bytes * iters
               (int)(((n + (int)svcntw() - 1) / (int)svcntw()) * 5 * 4));
    }
#endif

    // --- Benchmark ---
    printf("\n=== Benchmark: Vector Add (N=%d, prime) ===\n", n);
    size_t bytes = (size_t)n * 3 * sizeof(float);

#ifdef __ARM_FEATURE_SVE
    int num_r = cpu_has_sve() ? 3 : 2;
    benchmark_result_t results[3];
    int ri = 0;

    BENCH_COMPUTE(bench_scalar(), n, bytes, 30, results[ri]);
    results[ri++].name = "scalar add (baseline)";

    BENCH_COMPUTE(bench_neon_style(), n, bytes, 30, results[ri]);
    results[ri++].name = "NEON-style (main+tail)";

    if (cpu_has_sve()) {
        BENCH_COMPUTE(bench_sve_style(), n, bytes, 30, results[ri]);
        results[ri++].name = "SVE-style (predicated)";
    }

    bench_report(results, (size_t)ri);

    printf("\n=== Why SVE Predicated Tail is Superior ===\n");
    printf("  1. Zero scalar code:       No I-cache pollution from tail loop\n");
    printf("  2. No branch mispredicts:   Tail check branch eliminated\n");
    printf("  3. Vector ISA consistent:   Only one loop, any N works\n");
    printf("  4. Small N efficiency:      For N < vector width, SVE still\n");
    printf("                              uses vector ops; NEON falls back to\n");
    printf("                              scalar entirely\n");
    printf("  5. Code maintenance:        One loop to write, test, debug\n");
    printf("                              vs NEON's main loop + tail loop\n");
    printf("  6. Forward-compatible:      Same SVE code runs on 128, 256,\n");
    printf("                              512, or 2048-bit hardware unchanged\n");

    printf("\n  NOTE: For N=%d, the NEON tail is only %d elements (%.4f%%),\n",
           n, n % 4, 100.0 * (n % 4) / (double)n);
    printf("        so the practical benefit here is small. The real win\n");
    printf("        comes at small N or when vectorizing complex kernels\n");
    printf("        where the tail code would be large and error-prone.\n");
#else
    benchmark_result_t results[2];
    BENCH_COMPUTE(bench_scalar(), n, bytes, 30, results[0]);
    results[0].name = "scalar add (baseline)";
    BENCH_COMPUTE(bench_neon_style(), n, bytes, 30, results[1]);
    results[1].name = "NEON-style (main+tail)";
    bench_report(results, 2);
    printf("  SVE code not compiled in (use -march=armv8-a+sve)\n");
#endif

    printf("\n  Verification: %d\n", (int)g_pt_count);

    ALIGNED_FREE(ref);
    ALIGNED_FREE(g_pt_a);
    ALIGNED_FREE(g_pt_b);
    ALIGNED_FREE(g_pt_c);
    return 0;
}
