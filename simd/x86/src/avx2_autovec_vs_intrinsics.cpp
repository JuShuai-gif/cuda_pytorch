/**
 * AVX2 Auto-Vectorization vs Hand-Written Intrinsics
 *
 * Compares auto-vectorization (-O2 -mavx2) vs hand-written AVX2 intrinsics
 * across 4 test cases:
 *   1. Simple vector add (auto-vec should match)
 *   2. Vector scale: a*x + y (FMA vs mul+add)
 *   3. Sum reduction (auto-vec does decent)
 *   4. Clamp operation (auto-vec may not pattern-match)
 *
 * Key points:
 *   - __attribute__((noinline)) on test functions
 *   - __restrict to help compiler alias analysis
 *   - Uses common headers for benchmark, check, cpu features, aligned allocation
 */

#include <immintrin.h>
#include <cstdio>
#include <cstdlib>

#include "../../common/benchmark.h"
#include "../../common/check.h"
#include "../../common/cpu_features.h"
#include "../../common/random_data.h"
#include "../../common/aligned_buffer.h"

#define N 1000000

/* ======================================================================== */
/*  Test 1: Vector Add                                                      */
/* ======================================================================== */

__attribute__((noinline))
void vadd_auto(const float *__restrict a, const float *__restrict b,
               float *__restrict c, int n) {
    for (int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

__attribute__((noinline))
void vadd_avx2(const float *a, const float *b, float *c, int n) {
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(c + i, _mm256_add_ps(va, vb));
    }
    for (; i < n; i++)
        c[i] = a[i] + b[i];
}

/* ======================================================================== */
/*  Test 2: Vector Scale (ax + y)                                           */
/* ======================================================================== */

__attribute__((noinline))
void vscal_auto(const float *__restrict a, const float *__restrict x,
                const float *__restrict y, float *__restrict out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = a[i] * x[i] + y[i];
}

__attribute__((noinline))
void vscal_avx2_fma(const float *a, const float *x, const float *y,
                    float *out, int n) {
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        _mm256_storeu_ps(out + i, _mm256_fmadd_ps(va, vx, vy));
    }
    for (; i < n; i++)
        out[i] = a[i] * x[i] + y[i];
}

__attribute__((noinline))
void vscal_avx2_muladd(const float *a, const float *x, const float *y,
                       float *out, int n) {
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        __m256 prod = _mm256_mul_ps(va, vx);
        _mm256_storeu_ps(out + i, _mm256_add_ps(prod, vy));
    }
    for (; i < n; i++)
        out[i] = a[i] * x[i] + y[i];
}

/* ======================================================================== */
/*  Test 3: Sum Reduction                                                   */
/* ======================================================================== */

__attribute__((noinline))
float sum_auto(const float *__restrict a, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++)
        sum += a[i];
    return sum;
}

__attribute__((noinline))
float sum_avx2(const float *a, int n) {
    __m256 vsum0 = _mm256_setzero_ps();
    __m256 vsum1 = _mm256_setzero_ps();
    __m256 vsum2 = _mm256_setzero_ps();
    __m256 vsum3 = _mm256_setzero_ps();

    int i = 0;
    for (; i + 31 < n; i += 32) {
        vsum0 = _mm256_add_ps(vsum0, _mm256_loadu_ps(a + i));
        vsum1 = _mm256_add_ps(vsum1, _mm256_loadu_ps(a + i + 8));
        vsum2 = _mm256_add_ps(vsum2, _mm256_loadu_ps(a + i + 16));
        vsum3 = _mm256_add_ps(vsum3, _mm256_loadu_ps(a + i + 24));
    }
    for (; i + 7 < n; i += 8)
        vsum0 = _mm256_add_ps(vsum0, _mm256_loadu_ps(a + i));

    vsum0 = _mm256_add_ps(vsum0, vsum1);
    vsum0 = _mm256_add_ps(vsum0, vsum2);
    vsum0 = _mm256_add_ps(vsum0, vsum3);

    __m128 lo = _mm256_castps256_ps128(vsum0);
    __m128 hi = _mm256_extractf128_ps(vsum0, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float result = _mm_cvtss_f32(sum128);

    for (; i < n; i++)
        result += a[i];
    return result;
}

/* ======================================================================== */
/*  Test 4: Clamp (min/max)                                                 */
/* ======================================================================== */

__attribute__((noinline))
void clamp_auto(const float *__restrict a, float *__restrict out,
                float lo, float hi, int n) {
    for (int i = 0; i < n; i++) {
        float v = a[i];
        out[i] = (v < lo) ? lo : ((v > hi) ? hi : v);
    }
}

__attribute__((noinline))
void clamp_avx2(const float *a, float *out, float lo, float hi, int n) {
    __m256 vlo = _mm256_set1_ps(lo);
    __m256 vhi = _mm256_set1_ps(hi);

    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 v = _mm256_loadu_ps(a + i);
        v = _mm256_max_ps(v, vlo);
        v = _mm256_min_ps(v, vhi);
        _mm256_storeu_ps(out + i, v);
    }
    for (; i < n; i++) {
        float v = a[i];
        out[i] = (v < lo) ? lo : ((v > hi) ? hi : v);
    }
}

/* ======================================================================== */
/*  Main                                                                    */
/* ======================================================================== */

int main() {
    cpu_print_features();

    if (!cpu_has_avx2()) {
        printf("AVX2 not supported on this CPU. Exiting.\n");
        return 1;
    }

    printf("\n=== Auto-Vectorization vs Hand-Written AVX2 Intrinsics ===\n");
    printf("Compiled with -O2 -mavx2 (adjust as needed)\n");
    printf("N = %d\n", N);

    float *a = ALIGNED_ALLOC(float, N, 32);
    float *b = ALIGNED_ALLOC(float, N, 32);
    float *x = ALIGNED_ALLOC(float, N, 32);
    float *y = ALIGNED_ALLOC(float, N, 32);
    float *c = ALIGNED_ALLOC(float, N, 32);
    float *d = ALIGNED_ALLOC(float, N, 32);

    fill_range_f32(a, N, 0.0f, 128.0f);
    fill_range_f32(b, N, 0.0f, 128.0f);
    fill_range_f32(x, N, 2.0f, 3.0f);
    fill_range_f32(y, N, 0.0f, 10.0f);

    int iters = 200;

    /* -------- Test 1: Vector Add -------- */
    {
        printf("\n--- Test 1: Vector Add (c[i] = a[i] + b[i]) ---\n");

        benchmark_result_t r[2];

        BENCH_COMPUTE(vadd_auto(a, b, c, N), N, N * 12, iters, r[0]);
        r[0].name = "vadd_auto";

        BENCH_COMPUTE(vadd_avx2(a, b, d, N), N, N * 12, iters, r[1]);
        r[1].name = "vadd_avx2";

        printf("  Auto-vec:  %7.1f us\n", r[0].elapsed_ns / 1e3);
        printf("  AVX2:      %7.1f us\n", r[1].elapsed_ns / 1e3);
        printf("  Verdict:   %s\n",
               (r[1].elapsed_ns < r[0].elapsed_ns * 1.1)
                   ? "Auto-vec ~= intrinsics (expected)"
                   : "Significant difference");

        CHECK_NEAR_ARRAY(c, d, N, 1e-5f, "Test 1: vadd correctness");

        bench_report(r, 2);
    }

    /* -------- Test 2: Vector Scale -------- */
    {
        printf("\n--- Test 2: Vector Scale (out[i] = a[i]*x[i] + y[i]) ---\n");

        benchmark_result_t r[3];

        BENCH_COMPUTE(vscal_auto(a, x, y, d, N), N, N * 16, iters, r[0]);
        r[0].name = "vscal_auto";

        BENCH_COMPUTE(vscal_avx2_fma(a, x, y, c, N), N, N * 16, iters, r[1]);
        r[1].name = "vscal_avx2_fma";

        BENCH_COMPUTE(vscal_avx2_muladd(a, x, y, d, N), N, N * 16, iters, r[2]);
        r[2].name = "vscal_mul_add";

        printf("  Auto-vec:     %7.1f us\n", r[0].elapsed_ns / 1e3);
        printf("  AVX2 FMA:     %7.1f us  (%.2fx vs auto)\n",
               r[1].elapsed_ns / 1e3, r[0].elapsed_ns / r[1].elapsed_ns);
        printf("  AVX2 mul+add: %7.1f us  (%.2fx vs auto)\n",
               r[2].elapsed_ns / 1e3, r[0].elapsed_ns / r[2].elapsed_ns);
        printf("  FMA saves 1 instruction (4 ops/cycle vs 2 for mul+add).\n");
        printf("  Verdict:     %s\n",
               (r[1].elapsed_ns < r[0].elapsed_ns * 1.02)
                   ? "Auto-vec may use FMA already"
                   : "Intrinsics FMA has edge");

        CHECK_NEAR_ARRAY(c, d, N, 1e-4f, "Test 2: vscal correctness");

        bench_report(r, 3);
    }

    /* -------- Test 3: Sum Reduction -------- */
    {
        printf("\n--- Test 3: Sum Reduction ---\n");

        benchmark_result_t r[2];
        volatile float sink;

        BENCH_COMPUTE(sink = sum_auto(a, N), N, N * 4, iters * 5, r[0]);
        r[0].name = "sum_auto";

        BENCH_COMPUTE(sink = sum_avx2(a, N), N, N * 4, iters * 5, r[1]);
        r[1].name = "sum_avx2_4acc";

        printf("  Auto-vec:     %7.1f us  (result: %.2f)\n",
               r[0].elapsed_ns / 1e3, sum_auto(a, N));
        printf("  AVX2 4-acc:   %7.1f us  (result: %.2f)\n",
               r[1].elapsed_ns / 1e3, sum_avx2(a, N));
        printf("  Speedup: %.2fx\n", r[0].elapsed_ns / r[1].elapsed_ns);
        printf("  Auto-vec does decent reduction; multi-accumulator hides latency.\n");

        /* Sum of N floats accumulates ~N * machine_epsilon error.
         * With N=1e6, expected error is O(100). Tolerance scales with N. */
        CHECK_NEAR(sum_auto(a, N), sum_avx2(a, N), 1e-3f * (float)N, "Test 3: sum reduction correctness");

        bench_report(r, 2);
    }

    /* -------- Test 4: Clamp -------- */
    {
        printf("\n--- Test 4: Clamp (min/max) ---\n");

        benchmark_result_t r[2];

        BENCH_COMPUTE(clamp_auto(a, c, 10.0f, 100.0f, N), N, N * 8, iters, r[0]);
        r[0].name = "clamp_auto";

        BENCH_COMPUTE(clamp_avx2(a, d, 10.0f, 100.0f, N), N, N * 8, iters, r[1]);
        r[1].name = "clamp_avx2";

        printf("  Auto-vec:     %7.1f us  (branching may prevent vectorization)\n",
               r[0].elapsed_ns / 1e3);
        printf("  AVX2:         %7.1f us  (_mm256_max_ps + _mm256_min_ps)\n",
               r[1].elapsed_ns / 1e3);
        printf("  Speedup: %.2fx\n", r[0].elapsed_ns / r[1].elapsed_ns);
        printf("  Branchless clamp prevents scalarization.\n");

        CHECK_NEAR_ARRAY(c, d, N, 1e-5f, "Test 4: clamp correctness");

        bench_report(r, 2);
    }

    /* -------- Summary -------- */
    printf("\n=== Summary ===\n");
    printf("  %-30s %s\n", "Pattern", "Auto-vec vs Intrinsics");
    printf("  %-30s %s\n", "------------------------------",
           "-------------------------");
    printf("  %-30s %s\n", "Simple vector add", "~Equal (compiler is good)");
    printf("  %-30s %s\n", "ax+y (FMA)", "Auto-vec uses FMA if available");
    printf("  %-30s %s\n", "Sum reduction", "Auto-vec decent; 4-acc wins");
    printf("  %-30s %s\n", "Clamp (branchless)", "Intrinsics clearly win");
    printf("  %-30s %s\n", "Gather/scatter", "Need intrinsics");
    printf("  %-30s %s\n", "Specialized (exp, maddubs)", "Need intrinsics");

    printf("\n--- When to use __restrict ---\n");
    printf("  __restrict tells the compiler pointers don't alias.\n");
    printf("  Without it, compiler assumes writes to `out` could modify\n");
    printf("  `a` or `b`, preventing vectorization or requiring reloads.\n");
    printf("  Always use __restrict on output pointers in auto-vec loops.\n");

    ALIGNED_FREE(a);
    ALIGNED_FREE(b);
    ALIGNED_FREE(x);
    ALIGNED_FREE(y);
    ALIGNED_FREE(c);
    ALIGNED_FREE(d);

    return 0;
}
