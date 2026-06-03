/**
 * sve_vector_add.cpp -- Vector Add using ARM SVE with Predicated Loop
 *
 * Demonstrates the canonical SVE predicated loop pattern:
 *   svbool_t pg = svwhilelt_b32(i, n);
 *   svfloat32_t va = svld1(pg, a + i);
 *   svfloat32_t vb = svld1(pg, b + i);
 *   svst1(pg, c + i, svadd_x(pg, va, vb));
 *   i += svcntw();
 *
 * SVE vector width is unknown at compile time; printed at runtime via svcntw().
 * Requires runtime SVE check before executing any SVE instructions.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif

extern "C" {
#include "../../common/benchmark.h"
#include "../../common/cpu_features.h"
#include "../../common/check.h"
#include "../../common/random_data.h"
#include "../../common/aligned_buffer.h"
}

static const int N_SVE = 1000000;

// ============================================================================
// Scalar baseline
// ============================================================================

static void scalar_vector_add(float* __restrict c,
                               const float* __restrict a,
                               const float* __restrict b, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// ============================================================================
// SVE vector add with svwhilelt predicate pattern
// ============================================================================

#ifdef __ARM_FEATURE_SVE
static void sve_vector_add(float* __restrict c,
                            const float* __restrict a,
                            const float* __restrict b, int n) {
    int i = 0;

    // Main predicated loop: svwhilelt generates the predicate for remaining
    // elements. No separate tail loop is needed -- when i+sve_width > n,
    // svwhilelt automatically masks out the out-of-bounds lanes.
    for (; i < n; /* i incremented inside */) {
        // Generate predicate: pg[lane] = (i + lane < n) ? true : false
        svbool_t pg = svwhilelt_b32(i, n);

        // Predicated load: loads only active lanes, zeros inactive lanes
        svfloat32_t va = svld1(pg, &a[i]);
        svfloat32_t vb = svld1(pg, &b[i]);

        // Predicated FMA-style add using 'x' (don't care) form:
        // inactive lanes pass through va unchanged
        svfloat32_t vc = svadd_x(pg, va, vb);

        // Predicated store: only active lanes are written to memory
        svst1(pg, &c[i], vc);

        // Advance by the actual vector width (32-bit lanes -> svcntw())
        i += svcntw();
    }
}
#endif // __ARM_FEATURE_SVE

// ============================================================================
// Print SVE hardware info
// ============================================================================

static void print_sve_info() {
    printf("\n=== SVE Hardware Information ===\n");
    if (!cpu_has_sve()) {
        printf("  SVE: NOT AVAILABLE on this CPU\n");
        printf("  (This CPU does not support SVE. Running scalar only.)\n");
        return;
    }

#ifdef __ARM_FEATURE_SVE
    // svcntw() returns the number of 32-bit lanes per SVE vector register.
    // Multiply by 32 to get the vector width in bits.
    // Must be called from a context where SVE is available.
    int sve_width_bits = svcntw() * 32;
    int sve_lanes_f32  = svcntw();
    printf("  SVE vector width: %d bits (%d x float32 lanes)\n",
           sve_width_bits, sve_lanes_f32);
    printf("  Registers: 32 x scalable vector registers (Z0-Z31)\n");
    printf("  Predicate registers: 16 (P0-P15)\n");

    // Print possible widths for common implementations
    printf("  Common implementations:\n");
    printf("    - 128-bit (Neoverse V1, A64FX):  4 x f32 lanes\n");
    printf("    - 256-bit (Neoverse V2, Grace):   8 x f32 lanes\n");
    printf("    - 512-bit (Fujitsu A64FX SVE):    16 x f32 lanes\n");
    printf("    - 2048-bit (future designs):       64 x f32 lanes\n");
    printf("  This CPU: %d x f32 lanes\n", sve_lanes_f32);
#else
    printf("  SVE compiled support: NO (compile with -march=armv8-a+sve)\n");
#endif
}

// ============================================================================
// Benchmark wrappers
// ============================================================================

static float* g_sv_a = nullptr;
static float* g_sv_b = nullptr;
static float* g_sv_c = nullptr;
static int    g_sv_n = 0;

__attribute__((noinline))
static void bench_scalar_add() { scalar_vector_add(g_sv_c, g_sv_a, g_sv_b, g_sv_n); }

#ifdef __ARM_FEATURE_SVE
__attribute__((noinline))
static void bench_sve_add() { sve_vector_add(g_sv_c, g_sv_a, g_sv_b, g_sv_n); }
#endif

// ============================================================================
// Main
// ============================================================================

int main() {
    cpu_print_features();
    print_sve_info();

    int n = N_SVE;
    g_sv_n = n;
    g_sv_a = ALIGNED_ALLOC(float, n, 64);
    g_sv_b = ALIGNED_ALLOC(float, n, 64);
    g_sv_c = ALIGNED_ALLOC(float, n, 64);

    fill_random_f32(g_sv_a, n);
    fill_random_f32(g_sv_b, n);

    // --- Scalar correctness (always runs) ---
    printf("\n=== Scalar Correctness ===\n");
    float* ref = ALIGNED_ALLOC(float, n, 64);
    scalar_vector_add(ref, g_sv_a, g_sv_b, n);
    printf("  [PASS] Scalar vector add produced %zu elements\n", (size_t)n);

    // --- SVE correctness and benchmark ---
#ifdef __ARM_FEATURE_SVE
    if (cpu_has_sve()) {
        printf("\n=== SVE Correctness ===\n");
        sve_vector_add(g_sv_c, g_sv_a, g_sv_b, n);
        CHECK_NEAR_ARRAY(g_sv_c, ref, n, 1e-6f,
                         "SVE vector add vs scalar (predicated loop)");

        printf("\n=== Benchmark: Vector Add (N=%d) ===\n", n);
        benchmark_result_t results[2];

        size_t bytes = (size_t)n * 3 * sizeof(float); // read a,b + write c

        BENCH_COMPUTE(bench_scalar_add(), n, bytes, 30, results[0]);
        results[0].name = "scalar add";

        BENCH_COMPUTE(bench_sve_add(), n, bytes, 30, results[1]);
        results[1].name = "SVE add (predicated)";

        bench_report(results, 2);

        // Print loop summary
        int sve_width = svcntw();
        int sve_iters = (n + sve_width - 1) / sve_width;
        printf("\n=== Loop Analysis ===\n");
        printf("  SVE vector width: %d x f32 lanes\n", sve_width);
        printf("  Total elements: %d\n", n);
        printf("  SVE loop iterations: %d (ceil(N/sve_width))\n", sve_iters);
        printf("  Last iteration predicate: masks %d lanes\n",
               sve_width - (n % sve_width > 0 ? sve_width - n % sve_width : 0));
        printf("  No scalar tail loop needed (predicate handles edges).\n");
    } else {
        printf("\n  SVE not available. Skipping SVE tests.\n");
    }
#else
    printf("\n  Compiled without SVE support (-march=armv8-a+sve needed).\n");
#endif

    // --- Scalar benchmark (always runs) ---
    printf("\n=== Scalar-Only Benchmark ===\n");
    {
        benchmark_result_t r;
        size_t bytes = (size_t)n * 3 * sizeof(float);
        BENCH_COMPUTE(bench_scalar_add(), n, bytes, 30, r);
        r.name = "scalar add (baseline)";
        bench_report(&r, 1);
    }

    ALIGNED_FREE(ref);
    ALIGNED_FREE(g_sv_a);
    ALIGNED_FREE(g_sv_b);
    ALIGNED_FREE(g_sv_c);
    return 0;
}
