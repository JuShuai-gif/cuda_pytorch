#pragma once
#include "cpu_features.h"
#include <stddef.h>

/*
 * dispatch.h -- Runtime multi-ISA dispatch framework.
 *
 * =========================================================================
 * WHY RUNTIME DISPATCH?
 * =========================================================================
 *
 * A single binary cannot be compiled with -mavx512f, -mavx2, -msse4.1
 * simultaneously. The compiler would mix instructions, potentially emitting
 * AVX-512 in a function meant for SSE. Multi-ISA dispatch solves this by:
 *
 *   1. Compiling each kernel implementation in a separate translation unit
 *      with its own ISA flags (e.g., -mavx2, -mavx512f).
 *   2. At program start, detecting which ISA the host CPU actually supports.
 *   3. Selecting the best implementation via a function pointer.
 *
 * Production libraries that use this pattern:
 *   - XNNPACK (TensorFlow Lite / PyTorch CPU backend)
 *   - OpenBLAS / BLIS
 *   - FFmpeg (libavcodec)
 *   - libjpeg-turbo
 *   - Eigen
 *   - Intel oneDNN / oneMKL
 *   - glibc (memcpy, memmove, strlen, etc.)
 *
 * =========================================================================
 * DISPATCH PATTERNS
 * =========================================================================
 *
 * PATTERN 1: Function pointer dispatch (this framework)
 *   - Dispatch is called ONCE at init, result stored in a function pointer.
 *   - Overhead: one indirect call per kernel invocation (~2 cycles on modern
 *     x86 with good branch predictor). Negligible for kernels processing
 *     >~100 elements.
 *   - Thread-safe IF init is done before spawning threads (recommended).
 *   - Pros: simple, portable, debuggable.
 *   - Cons: 2-cycle overhead per call; I-cache miss possible if dispatch
 *     resolution crosses page boundaries.
 *
 * PATTERN 2: GNU ifunc (indirect function, ELF-only)
 *   - The dynamic linker calls the resolver function at load time, before
 *     main(). The resolver returns the address of the best implementation.
 *   - Zero runtime overhead: no indirect call, no branch, no I-cache penalty.
 *     The PLT/GOT is patched to point directly to the best implementation.
 *   - Usage in a .c file:
 *       typedef void (*kernel_fn)(const float*, const float*, float*, size_t);
 *       __attribute__((ifunc("resolve_kernel")))
 *       void kernel(...);
 *       static void* resolve_kernel(void) {
 *           if (cpu_has_avx512f()) return &kernel_avx512;
 *           if (cpu_has_avx2())    return &kernel_avx2;
 *           return &kernel_scalar;
 *       }
 *   - Pros: zero overhead, resolved before any threads exist.
 *   - Cons: ELF-only (Linux, FreeBSD). Not portable to macOS/Windows.
 *           Harder to debug (resolver runs very early). Doesn't work if you
 *           need runtime re-configuration.
 *
 * PATTERN 3: Per-object-file dispatch (XNNPACK style)
 *   - Each ISA variant is compiled in a separate .o file with its own flags.
 *   - All .o files are linked into the final binary.
 *   - Init function populates a global ops table, which the scheduler uses
 *     to pick the right microkernel for each operation.
 *   - This is effectively Pattern 1 with a centralized ops table.
 *
 * WHEN TO USE WHICH:
 *   - Small, frequently called kernels (e.g., memcpy, strlen): use ifunc.
 *   - Medium/large kernels (>100 elements per call): use function pointers.
 *   - Libraries with many operations: use the ops table pattern.
 *   - Cross-platform code: use function pointers; ifunc is GNU-specific.
 *
 * =========================================================================
 * OVERHEAD ANALYSIS
 * =========================================================================
 *
 * Indirect call overhead on modern x86 (Skylake-X / Zen 4):
 *   - BTB (Branch Target Buffer) hit:  ~1-2 cycles
 *   - BTB miss:                        ~10-15 cycles + branch mispredict penalty
 *   - I-cache miss (cold dispatch):    ~100-200 cycles (first call after
 *                                      context switch or TLB flush)
 *
 * For a kernel processing N elements at ~1 ns/el:
 *   - N=10:  2-cycle overhead = ~20%  -> use ifunc
 *   - N=100: 2-cycle overhead = ~2%   -> acceptable for function pointers
 *   - N=1000:2-cycle overhead = ~0.2% -> negligible
 *
 * Branch predictor tip: since the dispatch function pointer is invariant
 * after init, the indirect branch predictor will learn the target and achieve
 * near-zero overhead after the first few calls.
 *
 * I-cache tip: group all variants of a kernel together in the same page to
 * reduce I-cache pressure. Use -fno-inline-functions-called-once and
 * __attribute__((noinline)) on each variant to prevent the compiler from
 * inlining them into the caller, which would defeat the I-cache benefit.
 *
 * =========================================================================
 * THREAD SAFETY
 * =========================================================================
 *
 * The dispatch function pointer is a global variable assigned ONCE at
 * program initialization, before any worker threads are spawned.
 *
 * This is the "init-before-threads" pattern and is the standard approach
 * in production libraries:
 *
 *   static kernel_fn g_best_kernel = NULL;
 *
 *   // Called from main() before any pthread_create() or omp parallel.
 *   void init_kernels(void) {
 *       g_best_kernel = dispatch_get_kernel();
 *   }
 *
 *   // Called from worker threads. g_best_kernel is read-only after init.
 *   void kernel_entry(...) {
 *       g_best_kernel(...);
 *   }
 *
 * For libraries where init order is unpredictable (e.g., a .so loaded by
 * dlopen in a multi-threaded program), use pthread_once() or C11 call_once():
 *
 *   static kernel_fn g_best_kernel = NULL;
 *   static pthread_once_t g_init_once = PTHREAD_ONCE_INIT;
 *   static void do_init(void) { g_best_kernel = dispatch_get_kernel(); }
 *   void kernel_entry(...) {
 *       pthread_once(&g_init_once, do_init);
 *       g_best_kernel(...);
 *   }
 *
 * NOTE: pthread_once adds ~20-30 cycles on the hot path (an atomic read
 * of the done flag). For latency-sensitive code, prefer init-before-threads.
 */

/*
 * Convenience macro: define a dispatch table entry.
 *
 * Usage:
 *   dispatch_entry_t table[] = {
 *       DISPATCH_ENTRY(cpu_has_avx512f, kernel_avx512),
 *       DISPATCH_ENTRY(cpu_has_avx2,    kernel_avx2),
 *       DISPATCH_ENTRY(cpu_has_sse,     kernel_sse),
 *   };
 */
#define DISPATCH_ENTRY(isa_check_fn, kernel_fn) \
    { (int (*)(void))(isa_check_fn), (void*)(kernel_fn) }

/*
 * A single entry in a dispatch table.
 *
 * `check` is a function that returns 1 if the ISA feature is available,
 * 0 otherwise. Usually points to cpu_has_avx2(), cpu_has_sve(), etc.
 *
 * `fn` is the kernel function pointer, cast to void*. In a strongly-typed
 * codebase, you can wrap dispatch_select() in a typed getter:
 *
 *   typedef void (*add_fn)(const float*, const float*, float*, size_t);
 *   static add_fn dispatch_get_add(void) {
 *       dispatch_entry_t table[] = { ... };
 *       return (add_fn)dispatch_select(table, count);
 *   }
 */
typedef struct {
    int (*check)(void);
    void* fn;
} dispatch_entry_t;

/*
 * Walk the dispatch table from highest to lowest priority (entries should be
 * ordered from most-capable to least-capable ISA). Returns the first entry
 * whose `check()` returns non-zero. If none match, returns the last entry
 * as a safety fallback (which should be the scalar baseline).
 *
 * Complexity: O(n) in the number of entries. Tables typically have 4-6
 * entries, so this is essentially free.
 *
 * This function is NOT thread-safe by itself. Call it during single-threaded
 * init, or protect it with a mutex / pthread_once if needed.
 */
static inline void* dispatch_select(const dispatch_entry_t* entries, size_t count) {
    if (count == 0) return NULL;

    for (size_t i = 0; i < count; i++) {
        if (entries[i].check && entries[i].check()) {
            return entries[i].fn;
        }
    }

    /*
     * Fallback: if no ISA feature matched (should not happen with a
     * well-constructed table where the last entry is the scalar baseline),
     * return the first entry's function pointer as a last resort.
     */
    return entries[count - 1].fn;
}

/*
 * =========================================================================
 * EXAMPLE DISPATCH TABLES
 * =========================================================================
 *
 * === x86 dispatch table (highest to lowest priority) ===
 *
 * dispatch_entry_t x86_table[] = {
 *     // [0] AVX-512: 512-bit, 16x f32, zmm registers, masked ops
 *     DISPATCH_ENTRY(cpu_has_avx512f, kernel_avx512),
 *
 *     // [1] AVX2+FMA: 256-bit, 8x f32, FMA (a*b+c in one instruction)
 *     // Must check both AVX2 and FMA. Some early AVX2 CPUs (e.g., some
 *     // Ivy Bridge EX variants) had AVX but not FMA. FMA is standard from
 *     // Haswell onward.
 *     { [](void)->int{ return cpu_has_avx2() && cpu_has_fma(); },
 *       (void*)kernel_avx2_fma },
 *
 *     // [2] AVX: 256-bit, 8x f32, no FMA (Sandy Bridge / Ivy Bridge)
 *     DISPATCH_ENTRY(cpu_has_avx, kernel_avx),
 *
 *     // [3] SSE 4.2: 128-bit, 4x f32 (Nehalem / Westmere, all x86-64)
 *     // This is the de-facto baseline for x86-64. Every x86-64 CPU has
 *     // at least SSE2, and virtually all have SSE4.2.
 *     DISPATCH_ENTRY(cpu_has_sse42, kernel_sse),
 *
 *     // [4] Scalar: pure C, no SIMD. Truly portable, runs everywhere.
 *     // Always returns 1, so it's the ultimate fallback.
 *     { [](void)->int{ return 1; }, (void*)kernel_scalar },
 * };
 *
 * === ARM dispatch table ===
 *
 * dispatch_entry_t arm_table[] = {
 *     // [0] SVE2: variable-width (128-2048 bits), enhanced predicate ops
 *     DISPATCH_ENTRY(cpu_has_sve2, kernel_sve2),
 *
 *     // [1] SVE: variable-width (128-2048 bits), predicated loops
 *     DISPATCH_ENTRY(cpu_has_sve, kernel_sve),
 *
 *     // [2] NEON/ASIMD: 128-bit, 4x f32, available on all ARMv8-A CPUs
 *     // On ARM64, NEON is mandatory (it's part of the base ARMv8-A ISA).
 *     // So cpu_has_neon() always returns 1 on ARM64, making this the
 *     // effective baseline for 64-bit ARM.
 *     DISPATCH_ENTRY(cpu_has_neon, kernel_neon),
 *
 *     // [3] Scalar: pure C fallback
 *     { [](void)->int{ return 1; }, (void*)kernel_scalar },
 * };
 *
 * Note: SVE implementations exist at 128-bit (Neoverse V1), 256-bit
 * (Neoverse V2), and 512-bit (Fujitsu A64FX). The same SVE binary runs
 * optimally on all widths thanks to the "vector-length agnostic" (VLA)
 * programming model. SVE2 adds gather/scatter, complex integer math,
 * and other enhancements over SVE.
 */
