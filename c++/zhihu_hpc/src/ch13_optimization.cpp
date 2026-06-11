// Chapter 13: Generating multiple versions of critical code for different
// instruction sets ("为不同指令集生成多个版本的关键代码")
//
// This file consolidates two CPU dispatching strategies:
//   13.1 - Function-pointer dispatch on first call (lazy runtime dispatch)
//   13.2 - GNU IFUNC dispatch via __attribute__((ifunc("resolver")))
//          (dynamic linker resolves the function at load time)
//
// The "critical function" demonstrated is a float dot-product (inner product)
// of two vectors.  Three implementations are provided:
//   Generic    – plain scalar loop, works on any x86 CPU
//   SSE2       – 4-wide SIMD using SSE2 intrinsics
//   AVX        – 8-wide SIMD using AVX intrinsics
//
// Compile:
//   g++ -std=c++11 -O2 ch13_optimization.cpp -o ch13_optimization
// Run:
//   ./ch13_optimization

#include <iostream>
#include <iomanip>
#include <cstring>
#include <cstdlib>
#include <cmath>

// ---- SIMD intrinsic headers ------------------------------------------------
#include <emmintrin.h>  // SSE2
#include <immintrin.h>  // AVX / AVX2

// ============================================================================
// Utility: CPU feature detection helpers
// ============================================================================

// Print which instruction sets are supported by the current CPU.
// Uses GCC builtins (available since GCC 4.8); requires __builtin_cpu_init()
// to be called once before any query.
static void print_cpu_features() {
    __builtin_cpu_init();
    std::cout << "CPU features (GCC builtins):\n";
    std::cout << "  SSE2:    " << (__builtin_cpu_supports("sse2") ? "yes" : "no") << "\n";
    std::cout << "  SSE3:    " << (__builtin_cpu_supports("sse3") ? "yes" : "no") << "\n";
    std::cout << "  SSSE3:   " << (__builtin_cpu_supports("ssse3") ? "yes" : "no") << "\n";
    std::cout << "  SSE4.1:  " << (__builtin_cpu_supports("sse4.1") ? "yes" : "no") << "\n";
    std::cout << "  SSE4.2:  " << (__builtin_cpu_supports("sse4.2") ? "yes" : "no") << "\n";
    std::cout << "  AVX:     " << (__builtin_cpu_supports("avx") ? "yes" : "no") << "\n";
    std::cout << "  AVX2:    " << (__builtin_cpu_supports("avx2") ? "yes" : "no") << "\n";
}

// Manual CPUID-based detection – included for pedagogical completeness.
// Equivalent to the GCC builtin approach but visible inline.
enum CpuLevel {
    CPU_BASELINE = 0,
    CPU_SSE2 = 2,
    CPU_SSE3 = 3,
    CPU_SSSE3 = 4,
    CPU_SSE41 = 5,
    CPU_SSE42 = 6,
    CPU_AVX = 7,
    CPU_AVX2 = 8
};

#ifndef _MSC_VER
#include <cpuid.h>
static void do_cpuid(int info[4], int func_id) {
    __cpuid_count(func_id, 0, info[0], info[1], info[2], info[3]);
}
#else
#include <intrin.h>
static void do_cpuid(int info[4], int func_id) {
    __cpuid(info, func_id);
}
#endif

static CpuLevel detect_cpu_level() {
    int info[4];
    do_cpuid(info, 0);
    int max_std = info[0];

    if (max_std < 1)
        return CPU_BASELINE;

    do_cpuid(info, 1);
    bool has_sse2 = (info[3] & (1u << 26)) != 0;
    bool has_sse3 = (info[2] & (1u << 0)) != 0;
    bool has_ssse3 = (info[2] & (1u << 9)) != 0;
    bool has_sse41 = (info[2] & (1u << 19)) != 0;
    bool has_sse42 = (info[2] & (1u << 20)) != 0;
    bool has_avx = (info[2] & (1u << 28)) != 0;
    bool has_osxsave = (info[2] & (1u << 27)) != 0;

    // AVX also requires OS XSAVE/XRSTOR support.
    bool avx_usable = has_avx && has_osxsave;

    // AVX2 is reported in leaf 7, sub-leaf 0, EBX bit 5.
    bool has_avx2 = false;
    if (max_std >= 7) {
        do_cpuid(info, 7);
        has_avx2 = (info[1] & (1u << 5)) != 0;
    }

    if (avx_usable && has_avx2)
        return CPU_AVX2;
    if (avx_usable)
        return CPU_AVX;
    if (has_sse42)
        return CPU_SSE42;
    if (has_sse41)
        return CPU_SSE41;
    if (has_ssse3)
        return CPU_SSSE3;
    if (has_sse3)
        return CPU_SSE3;
    if (has_sse2)
        return CPU_SSE2;
    return CPU_BASELINE;
}

static const char* cpu_level_name(CpuLevel lvl) {
    switch (lvl) {
        case CPU_AVX2:
            return "AVX2";
        case CPU_AVX:
            return "AVX";
        case CPU_SSE42:
            return "SSE4.2";
        case CPU_SSE41:
            return "SSE4.1";
        case CPU_SSSE3:
            return "SSSE3";
        case CPU_SSE3:
            return "SSE3";
        case CPU_SSE2:
            return "SSE2";
        default:
            return "Baseline";
    }
}

// ============================================================================
// Section 13.1 – Function-pointer dispatch on first call
// ============================================================================
//
// Pattern:
//   1. A global function pointer initially points to a dispatcher.
//   2. On the *first* call the dispatcher detects the CPU, updates the
//      pointer to the best implementation, and then calls it.
//   3. Every subsequent call goes directly to the optimal version.
//
// This is a lazy, zero-overhead-after-first-call approach.  The only cost
// is one extra indirection in the hot path (dereferencing the pointer),
// which modern CPUs predict perfectly.

// ---- 13.1 Implementations --------------------------------------------------

// Generic scalar dot product – no SIMD, works everywhere.
static float Dot_13_1_Generic(const float* a, const float* b, std::size_t n) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// SSE2 dot product – processes 4 floats per iteration.
// The target attribute tells GCC to compile only this function with SSE2.
__attribute__((target("sse2"))) static float Dot_13_1_SSE2(const float* a, const float* b,
                                                           std::size_t n) {
    __m128 sum4 = _mm_setzero_ps();
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        sum4 = _mm_add_ps(sum4, _mm_mul_ps(va, vb));
    }
    // Horizontal sum of the 4-wide accumulator (SSE2-only instructions).
    __m128 hi = _mm_movehl_ps(sum4, sum4);       // [d,c, -, -] <- [d,c,b,a]
    __m128 s = _mm_add_ps(sum4, hi);             // [b+d, a+c, -, -]
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));  // [a+b+c+d, ...]
    float result = _mm_cvtss_f32(s);

    // Scalar tail.
    for (; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// AVX dot product – processes 8 floats per iteration.
// The target attribute tells GCC to compile only this function with AVX.
__attribute__((target("avx"))) static float Dot_13_1_AVX(const float* a, const float* b,
                                                         std::size_t n) {
    __m256 sum8 = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum8 = _mm256_add_ps(sum8, _mm256_mul_ps(va, vb));
    }
    // Horizontal sum of the 8-wide accumulator.
    __m128 lo = _mm256_castps256_ps128(sum8);    // lower 128 bits
    __m128 hi = _mm256_extractf128_ps(sum8, 1);  // upper 128 bits
    __m128 sum4 = _mm_add_ps(lo, hi);            // 4-wide partial sum
    __m128 h2 = _mm_movehl_ps(sum4, sum4);
    __m128 s = _mm_add_ps(sum4, h2);
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    float result = _mm_cvtss_f32(s);

    // Scalar tail.
    for (; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// ---- 13.1 Dispatch machinery -----------------------------------------------

// Function pointer type.
typedef float (*DotFunc_13_1)(const float*, const float*, std::size_t);

// Forward-declare the dispatcher (needed to initialise the pointer).
static float Dot_13_1_Dispatch(const float* a, const float* b, std::size_t n);

// Global function pointer – initially points at the dispatcher.
static DotFunc_13_1 g_Dot_13_1 = Dot_13_1_Dispatch;

// Called exactly once (on the first invocation).  Detects CPU capabilities
// and updates the function pointer.  Then calls the chosen implementation.
static float Dot_13_1_Dispatch(const float* a, const float* b, std::size_t n) {
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx")) {
        g_Dot_13_1 = Dot_13_1_AVX;
        std::cout << "[13.1 dispatch] selected AVX version\n";
    } else if (__builtin_cpu_supports("sse2")) {
        g_Dot_13_1 = Dot_13_1_SSE2;
        std::cout << "[13.1 dispatch] selected SSE2 version\n";
    } else {
        g_Dot_13_1 = Dot_13_1_Generic;
        std::cout << "[13.1 dispatch] selected Generic version\n";
    }
    return g_Dot_13_1(a, b, n);
}

// Public entry point for the 13.1 scheme.
static float Dot_13_1(const float* a, const float* b, std::size_t n) {
    return g_Dot_13_1(a, b, n);
}

// ============================================================================
// Section 13.2 – GNU IFUNC dispatch (dynamic linker resolves at load time)
// ============================================================================
//
// Pattern:
//   1. Provide one implementation per ISA level (with target attributes).
//   2. Write a *resolver* function that returns a pointer to the best one.
//   3. Declare the entry-point function with __attribute__((ifunc("resolver"))).
//
// The dynamic linker calls the resolver once at load time (before main())
// and patches the GOT/PLT so every call goes directly to the chosen version.
// No runtime overhead: the call is a direct jump with no indirection.

// ---- 13.2 Implementations --------------------------------------------------

extern "C" {

// Generic – no target attribute, plain scalar code.
static float Dot_13_2_Generic(const float* a, const float* b, std::size_t n) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// SSE2 version.
__attribute__((target("sse2"))) static float Dot_13_2_SSE2(const float* a, const float* b,
                                                           std::size_t n) {
    __m128 sum4 = _mm_setzero_ps();
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        sum4 = _mm_add_ps(sum4, _mm_mul_ps(va, vb));
    }
    __m128 hi = _mm_movehl_ps(sum4, sum4);
    __m128 s = _mm_add_ps(sum4, hi);
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    float result = _mm_cvtss_f32(s);
    for (; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// AVX version.
__attribute__((target("avx"))) static float Dot_13_2_AVX(const float* a, const float* b,
                                                         std::size_t n) {
    __m256 sum8 = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum8 = _mm256_add_ps(sum8, _mm256_mul_ps(va, vb));
    }
    __m128 lo = _mm256_castps256_ps128(sum8);
    __m128 hi = _mm256_extractf128_ps(sum8, 1);
    __m128 sum4 = _mm_add_ps(lo, hi);
    __m128 h2 = _mm_movehl_ps(sum4, sum4);
    __m128 s = _mm_add_ps(sum4, h2);
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    float result = _mm_cvtss_f32(s);
    for (; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// The resolver: called by the dynamic linker at load time, before main().
// WARNING: Do NOT use std::cout or any nontrivial C++ facilities here –
// the C++ runtime may not have been initialised yet.
// The resolver writes its choice to a global so main() can report it.
static int g_13_2_resolved_level = 0;  // 0=generic, 2=sse2, 7=avx

static void* resolve_Dot_13_2(void) {
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx")) {
        g_13_2_resolved_level = 7;
        return reinterpret_cast<void*>(Dot_13_2_AVX);
    }
    if (__builtin_cpu_supports("sse2")) {
        g_13_2_resolved_level = 2;
        return reinterpret_cast<void*>(Dot_13_2_SSE2);
    }
    g_13_2_resolved_level = 0;
    return reinterpret_cast<void*>(Dot_13_2_Generic);
}

// The IFUNC entry point – callers use this name as if it were a normal
// function.  The ifunc attribute tells the linker to resolve it via the
// resolver above at load time.
float Dot_13_2(const float* a, const float* b, std::size_t n)
    __attribute__((ifunc("resolve_Dot_13_2")));

}  // extern "C"

// ============================================================================
// Test & verification
// ============================================================================

// Reference uses float accumulator (same as the SIMD implementations).
// Different addition orders may introduce small ULP differences.
static float reference_dot(const float* a, const float* b, std::size_t n) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Tolerance accounts for up to ~0.05 ULP drift from SIMD reordering over
// ~10k elements.  An absolute bound of 0.5 is conservative.
static bool approx_equal(float x, float y) {
    float abs_err = std::fabs(x - y);
    (void)x;  // unused in release builds that might warn
    return abs_err <= 0.5f;
}

static void fill_random(float* arr, std::size_t n, float scale) {
    for (std::size_t i = 0; i < n; ++i) {
        // Simple deterministic pseudo-random sequence.
        arr[i] = static_cast<float>((i * 1103515245u + 12345u) & 0x7FFFFFFFu) /
                 static_cast<float>(0x7FFFFFFFu) * scale;
    }
}

static bool test_dispatch(const char* label, float (*fn)(const float*, const float*, std::size_t),
                          const float* a, const float* b, std::size_t n) {
    float result = fn(a, b, n);
    float ref = reference_dot(a, b, n);
    bool ok = approx_equal(result, ref);
    std::cout << "  " << label << ": result=" << result << "  ref=" << ref << "  "
              << (ok ? "PASS" : "FAIL") << "\n";
    return ok;
}

// ============================================================================
// main
// ============================================================================

int main() {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "============================================================\n";
    std::cout << "  Chapter 13: CPU Dispatching for Multiple Instruction Sets\n";
    std::cout << "============================================================\n\n";

    // ---- Print CPU features -------------------------------------------------
    print_cpu_features();
    std::cout << "CPU level (CPUID manual): " << cpu_level_name(detect_cpu_level()) << "\n\n";

    // ---- Prepare test data --------------------------------------------------
    const std::size_t N = 10007;  // prime length to exercise tail handling
    float* vec_a = new float[N];
    float* vec_b = new float[N];
    fill_random(vec_a, N, 10.0f);
    fill_random(vec_b, N, 5.0f);

    std::cout << "Test data: two float vectors of length " << N << "\n\n";

    // ---- 13.1 – Function-pointer dispatch on first call ---------------------
    std::cout << "--- Section 13.1: Function-pointer dispatch on first call ---\n";

    // First call triggers dispatch.
    float r1_first = Dot_13_1(vec_a, vec_b, N);
    std::cout << "  First call result: " << r1_first << "\n";

    // Second call goes directly to the optimal version.
    float r1_second = Dot_13_1(vec_a, vec_b, N);
    std::cout << "  Second call result: " << r1_second << "  (no dispatch message printed)\n";

    bool ok1 = test_dispatch("13.1 dispatched", Dot_13_1, vec_a, vec_b, N);
    std::cout << "\n";

    // ---- 13.2 – IFUNC dispatch ----------------------------------------------
    std::cout << "--- Section 13.2: IFUNC dispatch (resolved at load time) ---\n";

    // The dynamic linker resolved Dot_13_2 before main() – report choice.
    const char* ifunc_choice = "Generic";
    if (g_13_2_resolved_level >= 7)
        ifunc_choice = "AVX";
    else if (g_13_2_resolved_level >= 2)
        ifunc_choice = "SSE2";
    std::cout << "[13.2 IFUNC resolver] -> " << ifunc_choice << "\n";

    // Every call to Dot_13_2 goes directly to the chosen implementation.
    float r2 = Dot_13_2(vec_a, vec_b, N);
    std::cout << "  IFUNC call result: " << r2 << "  (no dispatch overhead)\n";

    bool ok2 = test_dispatch("13.2 IFUNC", Dot_13_2, vec_a, vec_b, N);
    std::cout << "\n";

    // ---- Summary ------------------------------------------------------------
    std::cout << "------------------------------------------------------------\n";
    std::cout << "Summary:\n";
    std::cout << "  13.1 (function-pointer): " << (ok1 ? "PASS" : "FAIL") << "\n";
    std::cout << "  13.2 (IFUNC)           : " << (ok2 ? "PASS" : "FAIL") << "\n";

    delete[] vec_a;
    delete[] vec_b;

    if (ok1 && ok2) {
        std::cout << "\nAll checks passed.\n";
        return 0;
    } else {
        std::cout << "\nSome checks FAILED.\n";
        return 1;
    }
}
