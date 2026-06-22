/*
 * portable_simd_demo.cpp -- Single-source multi-ISA SIMD
 *
 * Compile with different flags to target different ISAs:
 *   g++ -msse4.2  portable_simd_demo.cpp  -> SSE (4-wide)
 *   g++ -mavx2    portable_simd_demo.cpp  -> AVX2 (8-wide)
 *   g++ -mavx512f portable_simd_demo.cpp  -> AVX-512 (16-wide)
 *
 * The same source code adapts to the available SIMD width.
 *
 * Pattern: abstract SIMD width and type behind macros,
 * then write the algorithm once using those abstractions.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "../../common/benchmark.h"
#include "../../common/check.h"
#include "../../common/random_data.h"
#include "../../common/aligned_buffer.h"
#include "../../common/cpu_features.h"

// --- SIMD Abstraction Layer ---

#if defined(__AVX512F__)
  #define SIMD_WIDTH 16
  #define SIMD_FLOAT __m512
  #define SIMD_LOAD(ptr)      _mm512_loadu_ps(ptr)
  #define SIMD_STORE(ptr, v)  _mm512_storeu_ps(ptr, v)
  #define SIMD_ADD(a, b)      _mm512_add_ps(a, b)
  #define SIMD_MUL(a, b)      _mm512_mul_ps(a, b)
  #define SIMD_FMADD(a, b, c) _mm512_fmadd_ps(a, b, c)
  #define SIMD_SET1(v)        _mm512_set1_ps(v)
  #define SIMD_MAX(a, b)      _mm512_max_ps(a, b)
  #define SIMD_MIN(a, b)      _mm512_min_ps(a, b)
  #define ISA_NAME "AVX-512 (512-bit, 16x f32)"
  #include <immintrin.h>
#elif defined(__AVX2__)
  #define SIMD_WIDTH 8
  #define SIMD_FLOAT __m256
  #define SIMD_LOAD(ptr)      _mm256_loadu_ps(ptr)
  #define SIMD_STORE(ptr, v)  _mm256_storeu_ps(ptr, v)
  #define SIMD_ADD(a, b)      _mm256_add_ps(a, b)
  #define SIMD_MUL(a, b)      _mm256_mul_ps(a, b)
  #define SIMD_FMADD(a, b, c) _mm256_add_ps(_mm256_mul_ps(a, b), c)
  #define SIMD_SET1(v)        _mm256_set1_ps(v)
  #define SIMD_MAX(a, b)      _mm256_max_ps(a, b)
  #define SIMD_MIN(a, b)      _mm256_min_ps(a, b)
  #define ISA_NAME "AVX2 (256-bit, 8x f32)"
  #include <immintrin.h>
#elif defined(__SSE4_2__)
  #define SIMD_WIDTH 4
  #define SIMD_FLOAT __m128
  #define SIMD_LOAD(ptr)      _mm_loadu_ps(ptr)
  #define SIMD_STORE(ptr, v)  _mm_storeu_ps(ptr, v)
  #define SIMD_ADD(a, b)      _mm_add_ps(a, b)
  #define SIMD_MUL(a, b)      _mm_mul_ps(a, b)
  #define SIMD_FMADD(a, b, c) _mm_add_ps(_mm_mul_ps(a, b), c)
  #define SIMD_SET1(v)        _mm_set1_ps(v)
  #define SIMD_MAX(a, b)      _mm_max_ps(a, b)
  #define SIMD_MIN(a, b)      _mm_min_ps(a, b)
  #define ISA_NAME "SSE 4.2 (128-bit, 4x f32)"
  #include <xmmintrin.h>
  #include <emmintrin.h>
  #include <pmmintrin.h>
  #include <smmintrin.h>
#else
  #error "No SIMD ISA detected. Compile with -msse4.2, -mavx2, or -mavx512f"
#endif

// --- The actual algorithm (written once, adapts to any SIMD width) ---

// Scalar baseline
static void scalar_fmadd(const float* a, const float* b, float* c, size_t n) {
    for (size_t i = 0; i < n; i++) c[i] = a[i] * b[i] + c[i];
}

// Portable SIMD -- same source code for any ISA width
__attribute__((noinline))
static void simd_fmadd(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0;
    for (; i + SIMD_WIDTH <= n; i += SIMD_WIDTH) {
        SIMD_FLOAT va = SIMD_LOAD(a + i);
        SIMD_FLOAT vb = SIMD_LOAD(b + i);
        SIMD_FLOAT vc = SIMD_LOAD(c + i);
        SIMD_STORE(c + i, SIMD_FMADD(va, vb, vc));
    }
    for (; i < n; i++) c[i] = a[i] * b[i] + c[i];
}

// Also demonstrate: ReLU (same code for all widths)
__attribute__((noinline))
static void simd_relu(float* x, size_t n) {
    SIMD_FLOAT zero = SIMD_SET1(0.0f);
    size_t i = 0;
    for (; i + SIMD_WIDTH <= n; i += SIMD_WIDTH) {
        SIMD_FLOAT v = SIMD_LOAD(x + i);
        SIMD_STORE(x + i, SIMD_MAX(v, zero));
    }
    for (; i < n; i++) if (x[i] < 0) x[i] = 0;
}

// Also: clamp (same code for all widths)
__attribute__((noinline))
static void simd_clamp(float* x, size_t n, float lo, float hi) {
    SIMD_FLOAT vlo = SIMD_SET1(lo);
    SIMD_FLOAT vhi = SIMD_SET1(hi);
    size_t i = 0;
    for (; i + SIMD_WIDTH <= n; i += SIMD_WIDTH) {
        SIMD_FLOAT v = SIMD_LOAD(x + i);
        v = SIMD_MIN(SIMD_MAX(v, vlo), vhi);
        SIMD_STORE(x + i, v);
    }
    for (; i < n; i++) {
        if (x[i] < lo) x[i] = lo;
        else if (x[i] > hi) x[i] = hi;
    }
}

int main() {
    printf("=== Portable SIMD Demo ===\n");
    printf("Compiled for: %s\n", ISA_NAME);
    printf("SIMD_WIDTH: %d floats per operation\n\n", SIMD_WIDTH);

    const size_t N = 1000003;
    float* a = ALIGNED_ALLOC(float, N, 64);
    float* b = ALIGNED_ALLOC(float, N, 64);
    float* c_scalar = ALIGNED_ALLOC(float, N, 64);
    float* c_simd = ALIGNED_ALLOC(float, N, 64);

    fill_random_f32(a, N);
    fill_random_f32(b, N);

    // Test: fmadd
    fill_constant_f32(c_scalar, N, 1.0f);
    memcpy(c_simd, c_scalar, N * sizeof(float));
    scalar_fmadd(a, b, c_scalar, N);
    simd_fmadd(a, b, c_simd, N);
    CHECK_NEAR_ARRAY(c_simd, c_scalar, N, 1e-5f, "portable fmadd");

    // Test: ReLU (negative values)
    for (size_t i = 0; i < N; i++) a[i] = (i % 2) ? -a[i] : a[i]; // make half negative
    memcpy(c_simd, a, N * sizeof(float));
    simd_relu(c_simd, N);
    for (size_t i = 0; i < N; i++)
        if (c_simd[i] < 0) { printf("  [FAIL] ReLU has negative value\n"); return 1; }
    printf("  [PASS] portable relu (all values >= 0)\n");

    // Test: clamp
    fill_random_f32(a, N);
    float lo = -0.5f, hi = 0.5f;
    for (size_t i = 0; i < N; i++) c_scalar[i] = a[i];
    memcpy(c_simd, a, N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        if (c_scalar[i] < lo) c_scalar[i] = lo;
        else if (c_scalar[i] > hi) c_scalar[i] = hi;
    }
    simd_clamp(c_simd, N, lo, hi);
    CHECK_NEAR_ARRAY(c_simd, c_scalar, N, 1e-5f, "portable clamp");

    // Benchmark
    const size_t bytes = N * 3 * sizeof(float);
    benchmark_result_t results[2];
    BENCH_COMPUTE(scalar_fmadd(a, b, c_scalar, N), N, bytes, 30, results[0]);
    results[0].name = "scalar fmadd";
    BENCH_COMPUTE(simd_fmadd(a, b, c_simd, N), N, bytes, 30, results[1]);
    results[1].name = "simd fmadd";

    printf("\n--- Benchmark (%s) ---\n", ISA_NAME);
    bench_report(results, 2);

    printf("\nKey insight:\n");
    printf("  Same source code, different SIMD widths at compile time.\n");
    printf("  Compile with -mavx2 to get 8-wide, -mavx512f to get 16-wide.\n");
    printf("  For production, combine with runtime dispatch (dispatch_demo.cpp)\n");
    printf("  to compile N object files, one per ISA, then select at runtime.\n");

    ALIGNED_FREE(a); ALIGNED_FREE(b);
    ALIGNED_FREE(c_scalar); ALIGNED_FREE(c_simd);
    return 0;
}
