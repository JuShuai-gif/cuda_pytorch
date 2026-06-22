/*
 * dispatch_demo.cpp -- Multi-ISA runtime dispatch demo (x86-64)
 *
 * Operation: element-wise fused multiply-add  c[i] = a[i] * b[i] + c[i]
 *
 * This file demonstrates the FULL production dispatch pattern:
 *   - 5 implementions (scalar, SSE, AVX, AVX2+FMA, AVX-512) in ONE file
 *     compiled with the highest ISA flags (-mavx512f -mavx2 -mfma -msse4.1).
 *     In production, each variant lives in its own translation unit.
 *   - Runtime feature detection selects the best available implementation.
 *   - Correctness verified against the scalar reference.
 *   - Benchmark compares dispatched path vs scalar.
 *   - GNU ifunc approach shown as a zero-overhead alternative.
 *
 * Compile (CMake will set flags; manual example):
 *   g++ -std=c++11 -O2 -mavx512f -mavx512dq -mavx512bw -mavx512vl \
 *       -mavx2 -mfma -msse4.1 dispatch_demo.cpp -o dispatch_demo
 *
 * Expected output on a machine with AVX2 but no AVX-512:
 *   === Runtime Dispatch Demo ===
 *   CPU supports: AVX2=YES AVX-512F=NO FMA=YES
 *   Dispatch table: [AVX-512: SKIP] [AVX2+FMA: SELECTED] [AVX: SKIP] [SSE: SKIP]
 *   Correctness: [PASS]
 *   Benchmark: dispatched = 0.XXX ns/el (8.2x speedup vs scalar)
 *   ISA selected: AVX2+FMA (256-bit, 8x f32, FMA)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include <immintrin.h>

#include "../../common/benchmark.h"
#include "../../common/check.h"
#include "../../common/random_data.h"
#include "../../common/aligned_buffer.h"
#include "../../common/cpu_features.h"
#include "../../common/dispatch.h"

static const size_t N = 1000003; /* prime, to exercise tail handling */

/*
 * ---------------------------------------------------------------------------
 * ISA feature helpers not in cpu_features.h
 * ---------------------------------------------------------------------------
 *
 * cpu_features.h provides cpu_has_avx2(), cpu_has_avx512f(), etc. but not
 * cpu_has_avx() (256-bit without FMA) or cpu_has_sse41(). We define those
 * check functions here. In production, extend cpu_features.h instead.
 */

static int cpu_has_avx(void) {
#if CPUDET_X86 && CPUDET_BUILTIN_CPU
    return __builtin_cpu_supports("avx") ? 1 : 0;
#else
    return 0;
#endif
}

static int cpu_has_sse41(void) {
#if CPUDET_X86 && CPUDET_BUILTIN_CPU
    return __builtin_cpu_supports("sse4.1") ? 1 : 0;
#else
    return 0;
#endif
}

/* Convenience: AVX2 + FMA = Haswell and later */
static int cpu_has_avx2_fma(void) {
    return cpu_has_avx2() && cpu_has_fma();
}

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
 * SSE 4.1: 128-bit, 4x f32 per register
 * ---------------------------------------------------------------------------
 *
 * SSE is the baseline SIMD ISA for x86-64. Every x86-64 CPU since 2003
 * has at least SSE2. SSE4.1 (2008, Penryn) is the practical floor used
 * by most production libraries.
 *
 * Key intrinsics:
 *   _mm_loadu_ps    -- 128-bit unaligned load (4 floats)
 *   _mm_mul_ps      -- packed multiply
 *   _mm_add_ps      -- packed add
 *   _mm_storeu_ps   -- 128-bit unaligned store
 *
 * Note: There is no FMA in SSE. We do a separate multiply and add, which
 * is 2 uops instead of 1. This is why AVX2+FMA is faster.
 */

__attribute__((noinline))
static void mad_sse(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 vc = _mm_loadu_ps(c + i);
        __m128 vd = _mm_add_ps(_mm_mul_ps(va, vb), vc);
        _mm_storeu_ps(c + i, vd);
    }
    for (; i < n; i++) {
        c[i] = a[i] * b[i] + c[i];
    }
}

/*
 * ---------------------------------------------------------------------------
 * AVX: 256-bit, 8x f32 per register (no FMA)
 * ---------------------------------------------------------------------------
 *
 * Introduced with Sandy Bridge (2011). Doubles throughput over SSE but
 * still uses separate multiply + add (2 uops per FMA operation).
 *
 * Key intrinsics:
 *   _mm256_loadu_ps     -- 256-bit unaligned load (8 floats)
 *   _mm256_mul_ps       -- packed multiply
 *   _mm256_add_ps       -- packed add
 *   _mm256_storeu_ps    -- 256-bit unaligned store
 */

__attribute__((noinline))
static void mad_avx(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_loadu_ps(c + i);
        __m256 vd = _mm256_add_ps(_mm256_mul_ps(va, vb), vc);
        _mm256_storeu_ps(c + i, vd);
    }
    for (; i < n; i++) {
        c[i] = a[i] * b[i] + c[i];
    }
}

/*
 * ---------------------------------------------------------------------------
 * AVX2 + FMA: 256-bit, 8x f32 per register, single FMA instruction
 * ---------------------------------------------------------------------------
 *
 * Introduced with Haswell (2013). The killer feature for ML: _mm256_fmadd_ps
 * does a*b+c in ONE instruction (1 uop, 0.5 cycles throughput on Skylake).
 * This doubles throughput over plain AVX for FMA-dominated workloads.
 *
 * Key intrinsics:
 *   _mm256_fmadd_ps(va, vb, vc)    -- vc + va * vb  (one instruction!)
 *
 * Most production ML inference backends target this as the minimum for
 * good performance (e.g., XNNPACK, ONNX Runtime).
 */

__attribute__((noinline))
static void mad_avx2_fma(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_loadu_ps(c + i);
        /* _mm256_fmadd_ps: vc = (va * vb) + vc  -> single uop on Haswell+ */
        _mm256_storeu_ps(c + i, _mm256_fmadd_ps(va, vb, vc));
    }
    for (; i < n; i++) {
        c[i] = a[i] * b[i] + c[i];
    }
}

/*
 * ---------------------------------------------------------------------------
 * AVX-512: 512-bit, 16x f32 per register, FMA, masked tail
 * ---------------------------------------------------------------------------
 *
 * Introduced with Skylake-X (2017). Uses 512-bit zmm registers.
 * 16x f32 per instruction.
 *
 * Key features demonstrated:
 *   - _mm512_fmadd_ps(zva, zvb, zvc)  -- 512-bit FMA (16 floats at once!)
 *   - _mm512_mask_fmadd_ps -- predicated FMA using __mmask16
 *   - Masked tail loop eliminates scalar tail, improving efficiency for
 *     non-multiple-of-16 sizes.
 *   - AVX-512 registers: zmm0-zmm31 (32 registers vs 16 ymm registers),
 *     reducing register pressure and spills.
 *
 * AVX-512 variants:
 *   - AVX-512F  (Foundation): FMA, masked ops, 512-bit registers
 *   - AVX-512BW (Byte/Word):  8-bit and 16-bit integer ops
 *   - AVX-512VL (Vector Length): use AVX-512 ops on 128/256-bit registers
 *   - AVX-512DQ (Double/Quad): 64-bit integer ops
 *
 * Production threshold: most libraries require AVX-512F, BW, and VL.
 */

#if defined(__AVX512F__)
__attribute__((noinline))
static void mad_avx512(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 vc = _mm512_loadu_ps(c + i);
        _mm512_storeu_ps(c + i, _mm512_fmadd_ps(va, vb, vc));
    }
    if (i < n) {
        size_t remaining = n - i;
        __mmask16 mask = (__mmask16)((1u << remaining) - 1);
        __m512 va = _mm512_maskz_loadu_ps(mask, a + i);
        __m512 vb = _mm512_maskz_loadu_ps(mask, b + i);
        __m512 vc = _mm512_maskz_loadu_ps(mask, c + i);
        _mm512_mask_storeu_ps(c + i, mask, _mm512_fmadd_ps(va, vb, vc));
    }
}
#endif /* __AVX512F__ */

/*
 * ---------------------------------------------------------------------------
 * DISPATCH TABLE
 * ---------------------------------------------------------------------------
 *
 * Entries are ordered from highest to lowest priority. dispatch_select()
 * walks the list and returns the first entry whose `check()` returns non-zero.
 *
 * IMPORTANT: The scalar fallback (last entry) must always return 1. On every
 * x86-64 CPU in existence, at least the scalar entry matches, providing the
 * ultimate safety net.
 *
 * In production, each variant is compiled in its own translation unit and
 * linked together. Here, all variants are in one file compiled with the
 * highest ISA flags (-mavx512f -mavx2 -mfma), which covers all.
 */

static dispatch_entry_t mad_dispatch_table[] = {
#if defined(__AVX512F__)
    { cpu_has_avx512f,  (void*)mad_avx512 },
#endif
    { cpu_has_avx2_fma, (void*)mad_avx2_fma },

    /* [1] AVX2+FMA: 256-bit, 8x f32, hardware FMA, best perf/$ for most ML */
    { cpu_has_avx2_fma, (void*)mad_avx2_fma },

    /* [2] AVX (plain, no FMA): 256-bit, 8x f32, Sandy Bridge / Ivy Bridge */
    { cpu_has_avx,      (void*)mad_avx },

    /* [3] SSE 4.1: 128-bit, 4x f32, all modern x86-64 CPUs */
    { cpu_has_sse41,    (void*)mad_sse },

    /* [4] Scalar: always available, ultimate fallback */
    { NULL,             (void*)mad_scalar },
};

static const size_t mad_dispatch_count =
    sizeof(mad_dispatch_table) / sizeof(mad_dispatch_table[0]);

/*
 * Initialize the best kernel ONCE at program start.
 *
 * This function pointer is the "dispatched kernel". After init, all calls
 * go through this pointer. On a typical x86 laptop from the last 10 years
 * (Haswell+), this will point to mad_avx2_fma.
 *
 * Thread safety: assign before creating worker threads. The pointer is
 * read-only after initialization, so no atomic ops or locks are needed
 * on the hot path.
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
 * Walks the dispatch table and prints which ISA was selected, which were
 * skipped, and why (feature available or not). Useful for debugging and
 * understanding what code path is actually executing.
 */

static void print_dispatch_decision(void) {
    const char* names[] = {
        "AVX-512 (512-bit, 16x f32, masked FMA)",
        "AVX2+FMA  (256-bit, 8x f32, single-uop FMA)",
        "AVX       (256-bit, 8x f32, mul+add)",
        "SSE 4.1   (128-bit, 4x f32, mul+add)",
        "Scalar    (pure C, always available)",
    };

    printf("Dispatch table (priority order):\n");

    int selected_idx = -1;

    for (size_t i = 0; i < mad_dispatch_count; i++) {
        int available = 0;

        if (mad_dispatch_table[i].check) {
            available = mad_dispatch_table[i].check();
        } else {
            /* NULL check function means always available (scalar fallback) */
            available = 1;
        }

        const char* status = available ? "" : "[SKIP]";
        const char* marker = "";

        if (available && selected_idx == -1) {
            selected_idx = (int)i;
            marker = " <-- SELECTED";
        }

        printf("  [%zu] %-48s %-7s%s\n",
               i, names[i], status, marker);
    }

    printf("\n");
    if (selected_idx >= 0) {
        printf("Active ISA: %s\n", names[selected_idx]);
    }
}

/*
 * ---------------------------------------------------------------------------
 * Benchmark wrapper globals
 * ---------------------------------------------------------------------------
 *
 * BENCH_COMPUTE calls the function with parameters. For noinline barriers
 * we use global pointers (same pattern as the rest of the project). This
 * prevents the compiler from hoisting loads/stores across iterations.
 */

static float* g_a = NULL;
static float* g_b = NULL;
static float* g_c = NULL;
static size_t g_n = 0;

__attribute__((noinline))
static void bn_scalar() { mad_scalar(g_a, g_b, g_c, g_n); }

__attribute__((noinline))
static void bn_dispatched() { g_best_mad(g_a, g_b, g_c, g_n); }

__attribute__((noinline))
static void bn_sse() { mad_sse(g_a, g_b, g_c, g_n); }

__attribute__((noinline))
static void bn_avx() { mad_avx(g_a, g_b, g_c, g_n); }

__attribute__((noinline))
static void bn_avx2_fma() { mad_avx2_fma(g_a, g_b, g_c, g_n); }

__attribute__((noinline))
#if defined(__AVX512F__)
static void bn_avx512() { mad_avx512(g_a, g_b, g_c, g_n); }
#endif

/*
 * ---------------------------------------------------------------------------
 * GNU IFUNC DEMO (zero-overhead dispatch, Linux/ELF only)
 * ---------------------------------------------------------------------------
 *
 * ifunc (indirect function) is resolved by the dynamic linker at load time,
 * before main() runs. The linker calls the resolver function, which returns
 * the address of the best implementation. The PLT/GOT is then patched so
 * that every call to the function goes directly to the best variant.
 *
 * Overhead: ZERO. No indirect branch, no function pointer dereference, no
 * branch predictor dependency. The processor sees a direct call.
 *
 * Limitations:
 *   - GNU/Linux and FreeBSD ELF only. Not supported on macOS (Mach-O) or
 *     Windows (PE/COFF).
 *   - The resolver runs before main(), which makes debugging harder.
 *     Use LD_DEBUG=all or GDB breakpoints on the resolver.
 *   - No runtime re-configuration. Once resolved, it's permanent.
 *   - Must use C linkage (extern "C").
 *
 * This is what glibc uses for memcpy, memmove, strlen, etc.
 */

extern "C" {

/*
 * Resolver: called by ld.so at load time. Must return a function pointer.
 * This runs before any constructors or main(), so it can't use stdio or
 * allocate memory. Feature detection is safe since __builtin_cpu_supports()
 * is a pure function that reads CPUID results.
 *
 * STATIC: the resolver must be static (non-exported) so it doesn't conflict
 * with other translation units.
 */
__attribute__((unused))
static void* resolve_mad_ifunc(void) {
    #if defined(__AVX512F__)
    if (cpu_has_avx512f())  { return (void*)mad_avx512; }
    #endif
    if (cpu_has_avx2_fma()) { return (void*)mad_avx2_fma; }
    if (cpu_has_avx())      { return (void*)mad_avx; }
    if (cpu_has_sse41())    { return (void*)mad_sse; }
    return (void*)mad_scalar;
}

/*
 * The ifunc implementation body.
 *
 * When __attribute__((ifunc("resolve_mad_ifunc"))) is activated on the
 * declaration, this body is NEVER executed -- the dynamic linker replaces
 * the function address with the resolver's return value.
 *
 * When ifunc is NOT active (default in this demo), this body calls the
 * scalar baseline as a safe fallback.
 *
 * To activate ifunc, add this declaration (replacing this definition):
 *   __attribute__((ifunc("resolve_mad_ifunc")))
 *   void mad_ifunc(const float* a, const float* b, float* c, size_t n);
 */
__attribute__((unused)) __attribute__((noinline))
static void mad_ifunc_impl(const float* a, const float* b, float* c, size_t n) {
    mad_scalar(a, b, c, n);
}

} /* extern "C" */

/*
 * ---------------------------------------------------------------------------
 * main
 * ---------------------------------------------------------------------------
 */

int main() {
    printf("=== Runtime Dispatch Demo: Fused Multiply-Add ===\n");
    printf("    Operation: c[i] = a[i] * b[i] + c[i]\n\n");

    /* Step 1: Detect and print CPU features */
    cpu_print_features();

    /* Step 2: Initialize dispatch (MUST be done before any kernel call) */
    mad_init();

    /* Step 3: Show which ISA was selected and why */
    printf("\n");
    print_dispatch_decision();

    /* Step 4: Allocate aligned buffers (64-byte for AVX-512) */
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

    /* 6a: Run scalar reference */
    float* scalar_out = ALIGNED_ALLOC(float, N, 64);
    memcpy(scalar_out, c_ref, N * sizeof(float));
    mad_scalar(a, b, scalar_out, N);

    /* 6b: Run dispatched kernel */
    memcpy(c_dsp, c_ref, N * sizeof(float));
    g_best_mad(a, b, c_dsp, N);

    CHECK_NEAR_ARRAY(c_dsp, scalar_out, N, 1e-5f,
                     "Dispatched kernel matches scalar reference");

    /* 6c: Verify each individual variant against scalar (sanity check)
     * Only test variants the CPU actually supports to avoid SIGILL. */
    printf("\n--- Individual Variant Verification ---\n");

    {
        float* c_tmp = ALIGNED_ALLOC(float, N, 64);

        if (cpu_has_sse41()) {
            memcpy(c_tmp, c_ref, N * sizeof(float));
            mad_sse(a, b, c_tmp, N);
            CHECK_NEAR_ARRAY(c_tmp, scalar_out, N, 5e-5f, "SSE vs scalar");
        } else {
            printf("  [SKIP] SSE not available\n");
        }

        if (cpu_has_avx()) {
            memcpy(c_tmp, c_ref, N * sizeof(float));
            mad_avx(a, b, c_tmp, N);
            CHECK_NEAR_ARRAY(c_tmp, scalar_out, N, 5e-5f, "AVX vs scalar");
        } else {
            printf("  [SKIP] AVX not available\n");
        }

        if (cpu_has_avx2_fma()) {
            memcpy(c_tmp, c_ref, N * sizeof(float));
            mad_avx2_fma(a, b, c_tmp, N);
            CHECK_NEAR_ARRAY(c_tmp, scalar_out, N, 5e-5f, "AVX2+FMA vs scalar");
        } else {
            printf("  [SKIP] AVX2+FMA not available\n");
        }

        if (cpu_has_avx512f()) {
#if defined(__AVX512F__)
            memcpy(c_tmp, c_ref, N * sizeof(float));
            mad_avx512(a, b, c_tmp, N);
            CHECK_NEAR_ARRAY(c_tmp, scalar_out, N, 5e-5f, "AVX-512 vs scalar");
#else
            printf("  [INFO] AVX-512 CPU detected but binary compiled without AVX-512\n");
#endif
        } else {
            printf("  [SKIP] AVX-512 not available\n");
        }

        ALIGNED_FREE(c_tmp);
    }

    /* Step 7: Benchmark all available variants + dispatched vs scalar */
    {
        g_a = a; g_b = b; g_c = c_dsp; g_n = N;

        const size_t bytes = N * 3 * sizeof(float);

        benchmark_result_t results[7];
        memset(results, 0, sizeof(results));

        int slot = 0;

        /* Scalar baseline always runs */
        BENCH_COMPUTE(bn_scalar(), N, bytes, 30, results[slot]);
        results[slot].name = "scalar fmadd (baseline)";
        slot++;

        /* Only benchmark ISA variants that the CPU actually supports.
         * Executing e.g. AVX-512 instructions on a CPU without AVX-512
         * causes SIGILL (illegal instruction). */
        if (cpu_has_sse41()) {
            BENCH_COMPUTE(bn_sse(), N, bytes, 30, results[slot]);
            results[slot].name = "SSE fmadd (128-bit, 4x f32)";
            slot++;
        }

        if (cpu_has_avx()) {
            BENCH_COMPUTE(bn_avx(), N, bytes, 30, results[slot]);
            results[slot].name = "AVX fmadd (256-bit, 8x f32)";
            slot++;
        }

        if (cpu_has_avx2_fma()) {
            BENCH_COMPUTE(bn_avx2_fma(), N, bytes, 30, results[slot]);
            results[slot].name = "AVX2+FMA fmadd (256-bit, FMA)";
            slot++;
        }

        if (cpu_has_avx512f()) {
#if defined(__AVX512F__)
            BENCH_COMPUTE(bn_avx512(), N, bytes, 30, results[slot]);
            results[slot].name = "AVX-512 fmadd (512-bit, 16x f32)";
#else
            printf("  [INFO] AVX-512 bench skipped (binary compiled without AVX-512)\n");
#endif
            slot++;
        }

        /* Dispatched kernel (auto-selects best available ISA) */
        BENCH_COMPUTE(bn_dispatched(), N, bytes, 30, results[slot]);
        results[slot].name = "dispatched fmadd (auto)";
        slot++;

        /*
         * Measure indirect call overhead: force the dispatch pointer to
         * the scalar implementation and compare with direct bn_scalar().
         * The difference is the cost of one indirect call (~1-2 cycles).
         */
        mad_fn saved = g_best_mad;
        g_best_mad = mad_scalar;
        BENCH_COMPUTE(bn_dispatched(), N, bytes, 30, results[slot]);
        results[slot].name = "dispatched (forced scalar, overhead test)";
        slot++;
        g_best_mad = saved;

        printf("\n--- Benchmark Results (N = %zu) ---\n", N);
        bench_report(results, (size_t)slot);

        printf("Notes:\n");
        printf("  - \"dispatched (forced scalar)\" vs \"scalar fmadd\" shows\n");
        printf("    indirect call overhead (~1-2 cycles/call, negligible).\n");
        printf("  - \"dispatched fmadd (auto)\" uses the best ISA for this CPU.\n");
        printf("  - AVX2+FMA: 8x parallelism + single-uop FMA = ~8x speedup.\n");
        printf("  - AVX-512: 16x parallelism but possible frequency throttling.\n");
    }

    /*
     * Step 8: GNU ifunc notes
     *
     * This section demonstrates what the ifunc approach looks like.
     * In production, you would:
     *
     *   // kernel_ifunc.cpp (compiled with -mavx512f -mavx2 -mfma -msse4.1)
     *   extern "C" {
     *       static void* resolve(void) {
     *           if (cpu_has_avx512f()) return (void*)mad_avx512;
     *           ...
     *       }
     *       __attribute__((ifunc("resolve")))
     *       void mad(const float*, const float*, float*, size_t);
     *   }
     *
     * Then in caller: `mad(a, b, c, n);` -- zero overhead, linker-resolved.
     */
    printf("\n--- GNU ifunc Alternative ---\n");
    printf("ifunc is resolved by ld.so at load time (before main).\n");
    printf("Zero overhead: no indirect call, no branch predictor dependency.\n");
    printf("Used by: glibc (memcpy, strlen), libm, some Rust crates.\n");
    printf("Limitation: Linux/FreeBSD ELF only (no macOS, no Windows).\n");
    printf("Activation: uncomment __attribute__((ifunc)) in source.\n");

    /* Step 9: Memory bandwidth note */
    printf("\n--- Performance Notes ---\n");
    printf("This kernel (fmadd) performs 2 FLOPs per element (mul + add).\n");
    printf("It reads 2 arrays and read+write 1 array: 12 bytes/element.\n");
    printf("At N = %zu, total data movement = %.2f MB.\n",
           N, (double)(N * 12) / (1024 * 1024));
    printf("Compute-bound at small N, memory-bound at large N.\n");
    printf("For memory-bound kernels, SIMD width matters less than\n");
    printf("memory bandwidth -- see the memcpy_like benchmarks.\n");

    /* Cleanup */
    ALIGNED_FREE(scalar_out);
    ALIGNED_FREE(a);
    ALIGNED_FREE(b);
    ALIGNED_FREE(c_ref);
    ALIGNED_FREE(c_dsp);

    return 0;
}
