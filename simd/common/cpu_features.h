#pragma once

#include <stdio.h>

/*
 * cpu_features.h -- Runtime CPU SIMD feature detection.
 *
 * Detects available SIMD instruction sets:
 *   x86: AVX2, AVX-512 (F/BW/VL), FMA
 *   ARM: NEON/ASIMD, SVE, SVE2
 *
 * Detection methods:
 *   GCC/Clang on x86: __builtin_cpu_supports()
 *   Linux ARM64: getauxval(AT_HWCAP) / getauxval(AT_HWCAP2)
 *   Other platforms: return 0
 */

/* Pull in getauxval for ARM64 Linux detection */
#if defined(__aarch64__) && defined(__linux__)
#include <sys/auxv.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ---- platform flags ---- */

#if defined(__x86_64__) || defined(__i386__) || defined(_M_IX86) || defined(_M_AMD64)
#define CPUDET_X86 1
#else
#define CPUDET_X86 0
#endif

#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
#define CPUDET_ARM64 1
#else
#define CPUDET_ARM64 0
#endif

#if defined(__GNUC__) || defined(__clang__)
#define CPUDET_BUILTIN_CPU 1
#else
#define CPUDET_BUILTIN_CPU 0
#endif

/* ---- detection functions ---- */

static inline int cpu_has_avx2(void) {
#if CPUDET_X86 && CPUDET_BUILTIN_CPU
    return __builtin_cpu_supports("avx2") ? 1 : 0;
#else
    return 0;
#endif
}

static inline int cpu_has_avx512f(void) {
#if CPUDET_X86 && CPUDET_BUILTIN_CPU
    return __builtin_cpu_supports("avx512f") ? 1 : 0;
#else
    return 0;
#endif
}

static inline int cpu_has_avx512bw(void) {
#if CPUDET_X86 && CPUDET_BUILTIN_CPU
    return __builtin_cpu_supports("avx512bw") ? 1 : 0;
#else
    return 0;
#endif
}

static inline int cpu_has_avx512vl(void) {
#if CPUDET_X86 && CPUDET_BUILTIN_CPU
    return __builtin_cpu_supports("avx512vl") ? 1 : 0;
#else
    return 0;
#endif
}

static inline int cpu_has_fma(void) {
#if CPUDET_X86 && CPUDET_BUILTIN_CPU
    return __builtin_cpu_supports("fma") ? 1 : 0;
#else
    return 0;
#endif
}

static inline int cpu_has_neon(void) {
#if CPUDET_ARM64
    return 1;
#else
    return 0;
#endif
}

static inline int cpu_has_sve(void) {
#if CPUDET_ARM64 && defined(__linux__) && defined(HWCAP_SVE)
    unsigned long hwcap = getauxval(AT_HWCAP);
    return (hwcap & HWCAP_SVE) ? 1 : 0;
#elif CPUDET_ARM64 && defined(__APPLE__)
    return 0;
#else
    return 0;
#endif
}

static inline int cpu_has_sve2(void) {
#if CPUDET_ARM64 && defined(__linux__) && defined(HWCAP2_SVE2)
    unsigned long hwcap2 = getauxval(AT_HWCAP2);
    return (hwcap2 & HWCAP2_SVE2) ? 1 : 0;
#else
    return 0;
#endif
}

/* ---- print all detected features ---- */

static inline void cpu_print_features(void) {
    printf("=== CPU SIMD Feature Detection ===\n");

#if CPUDET_X86
    printf("  Platform: x86-64\n");
    printf("  AVX2:        %s\n", cpu_has_avx2()      ? "YES" : "NO");
    printf("  AVX-512F:    %s\n", cpu_has_avx512f()   ? "YES" : "NO");
    printf("  AVX-512BW:   %s\n", cpu_has_avx512bw()  ? "YES" : "NO");
    printf("  AVX-512VL:   %s\n", cpu_has_avx512vl()  ? "YES" : "NO");
    printf("  FMA:         %s\n", cpu_has_fma()        ? "YES" : "NO");
#elif CPUDET_ARM64
    printf("  Platform: ARM64 (AArch64)\n");
    printf("  NEON/ASIMD:  %s\n", cpu_has_neon()       ? "YES" : "NO");
    printf("  SVE:         %s\n", cpu_has_sve()        ? "YES" : "NO");
    printf("  SVE2:        %s\n", cpu_has_sve2()       ? "YES" : "NO");
#else
    printf("  Platform: Unknown / Unsupported\n");
    printf("  AVX2:        NO\n");
    printf("  AVX-512F:    NO\n");
    printf("  AVX-512BW:   NO\n");
    printf("  AVX-512VL:   NO\n");
    printf("  FMA:         NO\n");
    printf("  NEON/ASIMD:  NO\n");
    printf("  SVE:         NO\n");
    printf("  SVE2:        NO\n");
#endif
    printf("===================================\n");
}

#ifdef __cplusplus
}
#endif
