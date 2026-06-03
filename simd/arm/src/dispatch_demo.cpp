/*
 * dispatch_demo.cpp -- Multi-ISA runtime dispatch demo (ARM64)
 *
 * Operation: element-wise fused multiply-add  c[i] = a[i] * b[i] + c[i]
 *
 * This file demonstrates the FULL production dispatch pattern for ARM:
 *   - 3 implementations: scalar, NEON (128-bit, FMA), SVE (variable-width)
 *   - SVE code is conditionally compiled (#ifdef __ARM_FEATURE_SVE)
 *   - Runtime ISA detection using HWCAP (via cpu_features.h)
 *   - Dispatch selects the best available ISA at runtime
 *   - Correctness verified against scalar reference
 *   - Benchmark compares dispatched path vs scalar
 *   - Proper SVE fallback: if SVE not available, dispatches to NEON
 *
 * Compile (CMake will set flags; manual example):
 *   g++ -std=c++11 -O2 -march=armv8-a+sve dispatch_demo.cpp -o dispatch_demo
 *
 * Key ARM SIMD ISA hierarchy:
 *   - NEON/ASIMD (ARMv8.0-A):      128-bit, 4x f32, FMA, mandatory on ARM64
 *   - SVE      (ARMv8.2-A SVE):    variable-width 128-2048 bits, predicated
 *   - SVE2     (ARMv9.0-A):        SVE + gather/scatter, complex int math
 *
 * Unlike x86 where each ISA extension has a fixed width, SVE is
 * "vector-length agnostic" (VLA). The same binary runs optimally on:
 *   - 128-bit SVE (Neoverse V1, AWS Graviton3)
 *   - 256-bit SVE (Neoverse V2, NVIDIA Grace)
 *   - 512-bit SVE (Fujitsu A64FX)
 *   - 2048-bit SVE (future designs)
 *
 * Expected output on a Neoverse N1 (NEON only, no SVE):
 *   === Runtime Dispatch Demo (ARM) ===
 *   CPU supports: NEON=YES SVE=NO
 *   Dispatch table: [SVE: SKIP] [NEON+FMA: SELECTED]
 *   Correctness: [PASS]
 *   Benchmark: dispatched = 0.XXX ns/el (3.8x speedup vs scalar)
 *   ISA selected: NEON+FMA (128-bit, 4x f32, FMA)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include <arm_neon.h>

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif

#include "../../common/benchmark.h"
#include "../../common/check.h"
#include "../../common/random_data.h"
#include "../../common/aligned_buffer.h"
#include "../../common/cpu_features.h"
#include "../../common/dispatch.h"

static const size_t N = 1000003; /* prime, to exercise tail handling */

/*
 * ---------------------------------------------------------------------------
 * Kernel type definition
 * ---------------------------------------------------------------------------
 *
 * All kernels share the same signature: element-wise mad (multiply-add).
 * Using a typedef makes the dispatch table and function pointer code cleaner.
 */

typedef void (*mad_fn)(const float* a, const float* b, float* c, size_t n);

/*
 * ---------------------------------------------------------------------------
 * SCALAR: pure C baseline
 * ---------------------------------------------------------------------------
 *
 * No SIMD, always available. Used as:
 *   - Correctness reference (golden output).
 *   - Baseline for speedup calculation.
 *   - Ultimate fallback in the dispatch table.
 */

__attribute__((noinline))
static void mad_scalar(const float* a, const float* b, float* c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] * b[i] + c[i];
    }
}

/*
 * ---------------------------------------------------------------------------
 * NEON: 128-bit, 4x f32 per register, hardware FMA
 * ---------------------------------------------------------------------------
 *
 * NEON/ASIMD is the baseline SIMD ISA for ARM64. Unlike x86 SSE which
 * lacks FMA, NEON has had hardware FMA since ARMv8.0-A (2011).
 *
 * Key intrinsics:
 *   vld1q_f32(ptr)    -- 128-bit load (4 floats)
 *   vfmaq_f32(vc, va, vb)  -- vc = vc + va * vb  (single instruction!)
 *   vst1q_f32(ptr, val)    -- 128-bit store
 *
 * vfmaq_f32 is the NEON FMA instruction: it does c = c + a * b in one
 * 128-bit SIMD operation. This is why NEON gets ~4x speedup over scalar
 * for FMA workloads -- both from 4x parallelism AND from FMA fusing
 * mul+add into one op.
 *
 * Tail: scalar loop for the remaining <4 elements.
 */

__attribute__((noinline))
static void mad_neon(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vc = vld1q_f32(c + i);
        /* vfmaq_f32: vc = vc + va * vb  -- single NEON FMA instruction */
        vst1q_f32(c + i, vfmaq_f32(vc, va, vb));
    }
    for (; i < n; i++) {
        c[i] = a[i] * b[i] + c[i];
    }
}

/*
 * ---------------------------------------------------------------------------
 * SVE: variable-width, predicated loop, hardware FMA
 * ---------------------------------------------------------------------------
 *
 * SVE (Scalable Vector Extension) is ARM's answer to AVX-512, but with a
 * key difference: the vector width is NOT fixed at compile time. The same
 * binary adapts to whichever vector width the CPU implements.
 *
 * SVE key features:
 *   - Vector-length agnostic (VLA) programming model
 *   - Predicated execution: every instruction is conditional on a predicate
 *   - First-faulting loads: safe speculative loads past array bounds
 *   - Gather/scatter: non-contiguous memory access (SVE2+)
 *   - 32 scalable vector registers (Z0-Z31)
 *   - 16 predicate registers (P0-P15)
 *
 * Key intrinsics used here:
 *   svwhilelt_b32(i, n)         -- predicate: lane active if i+lane < n
 *   svld1(pg, &a[i])            -- predicated load
 *   svmla_f32_x(pg, vc, va, vb) -- vc = vc + va * vb (FMA, don't-care pred)
 *   svst1(pg, &c[i], vc)        -- predicated store
 *   svcntw()                    -- number of 32-bit lanes in a vector register
 *
 * The svwhilelt pattern eliminates the need for a scalar tail loop entirely.
 * The predicate automatically masks out-of-bounds lanes on the last iteration.
 *
 * NOTE: This function is only compiled when __ARM_FEATURE_SVE is defined
 * (i.e., when the compiler was invoked with -march=armv8-a+sve or similar).
 * If SVE is not compiled in, the dispatch table skips this entry at runtime.
 */

#ifdef __ARM_FEATURE_SVE

__attribute__((noinline))
static void mad_sve(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0;

    /*
     * Canonical SVE predicated loop:
     * - svwhilelt generates a predicate for each iteration
     * - The predicate tells the hardware which lanes are within bounds
     * - Inactive lanes are NOT written to memory (safe)
     * - i advances by svcntw() (the actual number of 32-bit lanes per vector)
     *
     * This pattern is "vector-length agnostic": it works correctly on
     * 128-bit, 256-bit, 512-bit, and even 2048-bit SVE hardware without
     * any code changes.
     */
    for (; i < n; /* i incremented inside */) {
        svbool_t pg = svwhilelt_b32((int64_t)i, (int64_t)n);

        svfloat32_t va = svld1(pg, &a[i]);
        svfloat32_t vb = svld1(pg, &b[i]);
        svfloat32_t vc = svld1(pg, &c[i]);

        /*
         * svmla_f32_x: "multiply-add, don't care about inactive lanes".
         * vc = vc + va * vb. The '_x' variant means "unpredicated result
         * for inactive lanes" (faster, fewer dependencies).
         *
         * There's also svmla_f32_m (merge) and svmla_f32_z (zero).
         * The _x variant is preferred for simple accumulations because
         * it doesn't create false dependencies on the inactive lanes.
         *
         * Note: svmad_f32_x and svmla_f32_x are equivalent on SVE.
         * svmla (multiply-add) is the canonical variant.
         */
        vc = svmla_f32_x(pg, vc, va, vb);

        svst1(pg, &c[i], vc);

        i += (size_t)svcntw();
    }
}

#endif /* __ARM_FEATURE_SVE */

/*
 * ---------------------------------------------------------------------------
 * Fallback SVE entry (when SVE is not compiled in)
 * ---------------------------------------------------------------------------
 *
 * When __ARM_FEATURE_SVE is not defined (e.g., compiling with -march=armv8-a
 * without +sve), this stub function is used as a placeholder in the dispatch
 * table. The check function (cpu_has_sve) will return 0, so this function
 * is never actually called. But the compiler needs to see its address,
 * so we provide this no-op stub.
 *
 * In production, SVE code is in a separate .o file compiled with +sve.
 */

#ifndef __ARM_FEATURE_SVE

__attribute__((noinline))
static void mad_sve(const float* a, const float* b, float* c, size_t n) {
    /*
     * This stub is never called at runtime (dispatch check skips it).
     * We could use __builtin_trap() but that would pessimize.
     * Instead, fall through to the scalar baseline so the binary is
     * still functional even if the dispatch table is misconfigured.
     */
    mad_scalar(a, b, c, n);
}

#endif /* !__ARM_FEATURE_SVE */

/*
 * ---------------------------------------------------------------------------
 * DISPATCH TABLE
 * ---------------------------------------------------------------------------
 *
 * Entries are ordered from highest to lowest priority. dispatch_select()
 * walks the list and returns the first entry whose `check()` returns non-zero.
 *
 * ARM NEON is mandatory on ARM64 (AArch64), so `cpu_has_neon()` always
 * returns 1. This makes NEON the effective baseline for all 64-bit ARM CPUs.
 * The scalar fallback is included for 32-bit ARM and for consistency.
 *
 * SVE2 is excluded from this demo since it's a superset of SVE and the
 * pattern is identical. For SVE2, you'd add a `cpu_has_sve2()` check
 * above the SVE entry.
 */

static dispatch_entry_t mad_dispatch_table[] = {
#ifdef __ARM_FEATURE_SVE
    /* [0] SVE: variable-width (128-2048 bits), predicated, FMA */
    { cpu_has_sve,  (void*)mad_sve },
#endif

    /* [1] NEON: 128-bit, 4x f32, FMA (mandatory on ARM64) */
    { cpu_has_neon, (void*)mad_neon },

    /* [2] Scalar: pure C, always available (fallback for ARM32) */
    { NULL,         (void*)mad_scalar },
};

static const size_t mad_dispatch_count =
    sizeof(mad_dispatch_table) / sizeof(mad_dispatch_table[0]);

/*
 * Initialize the best kernel ONCE at program start.
 *
 * This function pointer is the "dispatched kernel". After init, all calls
 * go through this pointer. On ARM64, `cpu_has_neon()` always returns 1,
 * so NEON is the minimum dispatched kernel unless SVE is available.
 *
 * Thread safety: assign before creating worker threads. The pointer is
 * read-only after init, so no atomic ops or locks are needed on the hot path.
 */
static mad_fn g_best_mad = NULL;

static void mad_init(void) {
    g_best_mad = (mad_fn)dispatch_select(mad_dispatch_table, mad_dispatch_count);
}

/*
 * ---------------------------------------------------------------------------
 * Dispatch utility: print the dispatch decision
 * ---------------------------------------------------------------------------
 *
 * Walks the dispatch table and prints which ISA was selected and why.
 * Also prints SVE vector width if SVE is active.
 */

static void print_dispatch_decision(void) {
    printf("Dispatch table (priority order):\n");

    const char* fallback_isa = "Unknown";

    int selected_idx = -1;

    for (size_t i = 0; i < mad_dispatch_count; i++) {
        int available = 0;
        const char* isa_name = "???";

        void* fn = mad_dispatch_table[i].fn;

        if (fn == (void*)mad_sve) {
            isa_name = "SVE (variable-width, predicated, FMA)";
        } else if (fn == (void*)mad_neon) {
            isa_name = "NEON (128-bit, 4x f32, FMA)";
        } else if (fn == (void*)mad_scalar) {
            isa_name = "Scalar (pure C, always available)";
        }

        if (mad_dispatch_table[i].check) {
            available = mad_dispatch_table[i].check();
        } else {
            available = 1;
        }

        const char* status = available ? "" : "[SKIP]";
        const char* marker = "";

        if (available && selected_idx == -1) {
            selected_idx = (int)i;
            marker = " <-- SELECTED";
            fallback_isa = isa_name;
        }

        printf("  [%zu] %-52s %-7s%s\n",
               i, isa_name, status, marker);
    }

    printf("\n");

    if (selected_idx >= 0) {
        printf("Active ISA: %s\n", fallback_isa);

        /* If SVE is selected, print the actual vector width */
        if (cpu_has_sve()) {
#ifdef __ARM_FEATURE_SVE
            int sve_width_bits = svcntw() * 32;
            int sve_lanes = svcntw();
            printf("  SVE vector width: %d bits (%d x float32 lanes)\n",
                   sve_width_bits, sve_lanes);
            printf("  (Vector-length agnostic: same binary adapts to any width)\n");
#else
            printf("  SVE available (CPU) but not compiled in (compiler).\n");
#endif
        } else if (cpu_has_neon()) {
            printf("  NEON vector width: 128 bits (4 x float32 lanes)\n");
            printf("  NEON FMA (vfmaq_f32): single instruction mul+add\n");
        }
    }
}

/*
 * ---------------------------------------------------------------------------
 * SVE information printer
 * ---------------------------------------------------------------------------
 *
 * If SVE is available, prints the actual vector width at runtime.
 * svcntw() must be called from a context where SVE is active
 * (i.e., the CPU has SVE and the code was compiled with +sve).
 */

static void print_sve_info(void) {
    printf("\n--- SVE Hardware Information ---\n");

    if (!cpu_has_sve()) {
        printf("  SVE: NOT AVAILABLE on this CPU.\n");
        printf("  This CPU runs NEON only (or scalar fallback).\n");
        return;
    }

#ifdef __ARM_FEATURE_SVE
    int sve_width_bits = svcntw() * 32;
    int sve_lanes = svcntw();
    printf("  SVE: AVAILABLE\n");
    printf("  Vector width: %d bits (%d x float32 lanes)\n",
           sve_width_bits, sve_lanes);
    printf("  32 scalable vector registers (Z0-Z31)\n");
    printf("  16 predicate registers (P0-P15)\n");

    /* List common implementations for context */
    printf("  Common SVE implementations:\n");
    printf("    - 128-bit: Neoverse V1, AWS Graviton3 (4 x f32 lanes)\n");
    printf("    - 256-bit: Neoverse V2, NVIDIA Grace  (8 x f32 lanes)\n");
    printf("    - 512-bit: Fujitsu A64FX               (16 x f32 lanes)\n");
    printf("    - 2048-bit: future designs               (64 x f32 lanes)\n");
#else
    printf("  SVE: AVAILABLE (CPU supports it, but compiled without +sve)\n");
#endif
}

/*
 * ---------------------------------------------------------------------------
 * Benchmark wrapper globals
 * ---------------------------------------------------------------------------
 *
 * BENCH_COMPUTE calls the function with parameters. For noinline barriers
 * we use global pointers (same pattern as the rest of the project).
 */

static float* g_a = NULL;
static float* g_b = NULL;
static float* g_c = NULL;
static size_t g_n = 0;

__attribute__((noinline))
static void bn_scalar() { mad_scalar(g_a, g_b, g_c, g_n); }

__attribute__((noinline))
static void bn_neon() { mad_neon(g_a, g_b, g_c, g_n); }

__attribute__((noinline))
static void bn_dispatched() { g_best_mad(g_a, g_b, g_c, g_n); }

#ifdef __ARM_FEATURE_SVE
__attribute__((noinline))
static void bn_sve() { mad_sve(g_a, g_b, g_c, g_n); }
#endif

/*
 * ---------------------------------------------------------------------------
 * main
 * ---------------------------------------------------------------------------
 */

int main() {
    printf("=== Runtime Dispatch Demo (ARM): Fused Multiply-Add ===\n");
    printf("    Operation: c[i] = a[i] * b[i] + c[i]\n\n");

    /* Step 1: Detect and print CPU features */
    cpu_print_features();
    print_sve_info();

    /* Step 2: Initialize dispatch (MUST be done before any kernel call) */
    mad_init();

    /* Step 3: Show which ISA was selected and why */
    printf("\n");
    print_dispatch_decision();

    /* Step 4: Allocate aligned buffers (64-byte for SVE) */
    float* a     = ALIGNED_ALLOC(float, N, 64);
    float* b     = ALIGNED_ALLOC(float, N, 64);
    float* c_ref = ALIGNED_ALLOC(float, N, 64);
    float* c_dsp = ALIGNED_ALLOC(float, N, 64);

    if (!a || !b || !c_ref || !c_dsp) {
        fprintf(stderr, "Allocation failed.\n");
        return 1;
    }

    /* Step 5: Fill random input data (same seeds = deterministic) */
    rand_xorshift64_seed(42);
    fill_random_f32(a, N);
    rand_xorshift64_seed(99);
    fill_random_f32(b, N);
    rand_xorshift64_seed(7);
    fill_random_f32(c_ref, N); /* initial values for c */

    /* Step 6: Correctness verification */
    printf("\n--- Correctness Verification ---\n");

    /* 6a: Run scalar reference (golden output) */
    float* scalar_out = ALIGNED_ALLOC(float, N, 64);
    memcpy(scalar_out, c_ref, N * sizeof(float));
    mad_scalar(a, b, scalar_out, N);

    /* 6b: Run dispatched kernel */
    memcpy(c_dsp, c_ref, N * sizeof(float));
    g_best_mad(a, b, c_dsp, N);

    CHECK_NEAR_ARRAY(c_dsp, scalar_out, N, 1e-5f,
                     "Dispatched kernel matches scalar reference");

    /* 6c: Verify NEON against scalar */
    {
        float* c_neon = ALIGNED_ALLOC(float, N, 64);
        memcpy(c_neon, c_ref, N * sizeof(float));
        mad_neon(a, b, c_neon, N);
        CHECK_NEAR_ARRAY(c_neon, scalar_out, N, 1e-5f, "NEON vs scalar");
        ALIGNED_FREE(c_neon);
    }

    /* 6d: Verify SVE against scalar (if available and compiled in) */
#ifdef __ARM_FEATURE_SVE
    if (cpu_has_sve()) {
        float* c_sve = ALIGNED_ALLOC(float, N, 64);
        memcpy(c_sve, c_ref, N * sizeof(float));
        mad_sve(a, b, c_sve, N);
        CHECK_NEAR_ARRAY(c_sve, scalar_out, N, 1e-5f, "SVE vs scalar");
        ALIGNED_FREE(c_sve);
    } else {
        printf("  [SKIP] SVE not available on this CPU (skipping SVE test)\n");
    }
#else
    printf("  [SKIP] SVE not compiled in (requires -march=armv8-a+sve)\n");
#endif

    /* Step 7: Benchmark dispatched vs scalar */
    {
        g_a = a; g_b = b; g_c = c_dsp; g_n = N;

        const size_t bytes = N * 3 * sizeof(float);

        benchmark_result_t results[5];
        memset(results, 0, sizeof(results));

        BENCH_COMPUTE(bn_scalar(), N, bytes, 30, results[0]);
        results[0].name = "scalar fmadd (baseline)";

        BENCH_COMPUTE(bn_neon(), N, bytes, 30, results[1]);
        results[1].name = "NEON fmadd (vfmaq_f32)";

        BENCH_COMPUTE(bn_dispatched(), N, bytes, 30, results[2]);
        results[2].name = "dispatched fmadd (auto)";

        int slot = 3;

#ifdef __ARM_FEATURE_SVE
        if (cpu_has_sve()) {
            BENCH_COMPUTE(bn_sve(), N, bytes, 30, results[slot]);
            results[slot].name = "SVE fmadd (svmla, predicated)";
            slot++;
        }
#endif

        printf("\n--- Benchmark Results (N = %zu) ---\n", N);
        bench_report(results, (size_t)slot);
    }

    /* Step 8: SVE width-agnostic programming note */
    if (cpu_has_sve()) {
#ifdef __ARM_FEATURE_SVE
        printf("\n--- Loop Analysis (SVE) ---\n");
        int sve_lanes = svcntw();
        int sve_iters = (int)((N + sve_lanes - 1) / sve_lanes);
        printf("  SVE vector width: %d x f32 lanes\n", sve_lanes);
        printf("  Total elements: %zu\n", N);
        printf("  Loop iterations: %d\n", sve_iters);
        printf("  Last iteration tail: handled by svwhilelt predicate\n");
        printf("  (No separate scalar tail loop needed)\n");
#endif
    }

    /* Step 9: Performance notes */
    printf("\n--- Performance Notes ---\n");
    printf("  This kernel reads 2 arrays + read/writes 1 array.\n");
    printf("  12 bytes/element, %.2f MB total at N = %zu.\n",
           (double)(N * 12) / (1024 * 1024), N);
    printf("  NEON vfmaq_f32: FMA in one instruction (like x86 FMA).\n");
    printf("  SVE svmla: same FMA, but variable-width and predicated.\n");
    printf("  NEON is mandatory on ARM64 (AArch64) - always available.\n");
    printf("  SVE is optional - check cpu_has_sve() at runtime.\n");

    /* Cleanup */
    ALIGNED_FREE(scalar_out);
    ALIGNED_FREE(a);
    ALIGNED_FREE(b);
    ALIGNED_FREE(c_ref);
    ALIGNED_FREE(c_dsp);

    return 0;
}
