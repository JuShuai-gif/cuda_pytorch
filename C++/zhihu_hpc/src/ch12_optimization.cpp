// Chapter: 使用向量操作 (Using Vector Operations / SIMD)
// Consolidated examples 12.1 - 12.9 into a single compilable program
// Compile: g++ -std=c++11 -O2 -msse4.1 ch12_optimization.cpp -o ch12_optimization

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>

// SSE2 / SSE3 / SSE4.1 intrinsics
#include <emmintrin.h>
#include <pmmintrin.h>
#include <smmintrin.h>

// ============================================================================
// Example 12.5: Cross-platform alignment macro
// ============================================================================
#ifdef _MSC_VER
#define Align16(X) __declspec(align(16)) X
#define Align16Struct __declspec(align(16))
#else
#define Align16(X) X __attribute__((aligned(16)))
#define Align16Struct __attribute__((aligned(16)))
#endif

// ============================================================================
// Timing utility
// ============================================================================
static inline double get_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ============================================================================
// Constants
// ============================================================================
const int N_SHORT = 256;  // used for short-int vectorized loops (multiple of 8)
const int N_FLOAT = 100;  // used for sum loops (multiple of 4)

// ============================================================================
// Example 12.1a / 12.1b: Automatic vectorization with __restrict
// Using __restrict tells the compiler that pointers do not alias,
// enabling auto-vectorization.
// ============================================================================

// 12.1a - scalar loop (no __restrict, compiler may not vectorize)
void AddTwo_scalar(int size, int* aa, int* bb) {
    for (int i = 0; i < size; ++i) {
        aa[i] = bb[i] + 2;
    }
}

// 12.1b - loop with __restrict to enable auto-vectorization
void AddTwo_restrict(int size, int* __restrict aa, int* __restrict bb) {
    for (int i = 0; i < size; ++i) {
        aa[i] = bb[i] + 2;
    }
}

// ============================================================================
// Example 12.2: Struct alignment with __attribute__((aligned(16)))
// An aligned struct of 4 floats allows the compiler to use SIMD loads/stores.
// ============================================================================
struct Align16Struct S2 {
    float a, b, c, d;
};

void FuncS2() {
    S2 x, y;
    x.a = 1.0f;
    x.b = 2.0f;
    x.c = 3.0f;
    x.d = 4.0f;
    y.a = 1.5f;
    y.b = 2.5f;
    y.c = 3.5f;
    y.d = 4.5f;

    x.a = y.a + 1.0f;
    x.b = y.b + 2.0f;
    x.c = y.c + 3.0f;
    x.d = y.d + 4.0f;

    // Prevent compiler from optimizing away the computation
    volatile float sink = x.a + x.b + x.c + x.d;
    (void)sink;
}

// ============================================================================
// Example 12.4a: Loop with branch (scalar baseline)
// ============================================================================
void SelectAddMul_scalar(short int aa[], short int bb[], short int cc[]) {
    for (int i = 0; i < N_SHORT; ++i) {
        aa[i] = (bb[i] > 0) ? (cc[i] + 2) : (bb[i] * cc[i]);
    }
}

// ============================================================================
// Example 12.4b: Manual SSE2 vectorization of the branched loop
// Uses _mm_cmpgt_epi16, _mm_and_si128, _mm_andnot_si128, _mm_or_si128
// ============================================================================

// Inline helpers for unaligned load/store (allows compiler to optimize)
static inline __m128i LoadVectorU(void const* p) {
    return _mm_loadu_si128((__m128i const*)p);
}

static inline void StoreVectorU(void* d, __m128i const& x) {
    _mm_storeu_si128((__m128i*)d, x);
}

// Inline helpers for aligned load/store (requires 16-byte alignment)
static inline __m128i LoadVectorA(void const* p) {
    return _mm_load_si128((__m128i const*)p);
}

static inline void StoreVectorA(void* d, __m128i const& x) {
    _mm_store_si128((__m128i*)d, x);
}

void SelectAddMul_SSE2(short int aa[], short int bb[], short int cc[]) {
    __m128i zero = _mm_set1_epi16(0);
    __m128i two = _mm_set1_epi16(2);

    for (int i = 0; i < N_SHORT; i += 8) {
        __m128i b = LoadVectorU(bb + i);
        __m128i c = LoadVectorU(cc + i);
        __m128i c2 = _mm_add_epi16(c, two);
        __m128i bc = _mm_mullo_epi16(b, c);
        __m128i mask = _mm_cmpgt_epi16(b, zero);
        c2 = _mm_and_si128(c2, mask);
        bc = _mm_andnot_si128(mask, bc);
        __m128i a = _mm_or_si128(c2, bc);
        StoreVectorU(aa + i, a);
    }
}

// ============================================================================
// Example 12.4c: SSE4.1 version using _mm_blendv_epi8
// Cleaner than SSE2: one blend instruction replaces AND/ANDNOT/OR
// ============================================================================
void SelectAddMul_SSE41(short int aa[], short int bb[], short int cc[]) {
    __m128i zero = _mm_set1_epi16(0);
    __m128i two = _mm_set1_epi16(2);

    for (int i = 0; i < N_SHORT; i += 8) {
        __m128i b = LoadVectorU(bb + i);
        __m128i c = LoadVectorU(cc + i);
        __m128i c2 = _mm_add_epi16(c, two);
        __m128i bc = _mm_mullo_epi16(b, c);
        __m128i mask = _mm_cmpgt_epi16(b, zero);
        // _mm_blendv_epi8: select byte from c2 when mask bit is set, else from bc
        __m128i a = _mm_blendv_epi8(bc, c2, mask);
        StoreVectorU(aa + i, a);
    }
}

// ============================================================================
// Example 12.4d: Intel vector class library version (Is16vec8)
// Requires Intel C++ compiler or classic icc-compatible headers (dvec.h).
// Wrapped in #ifdef guard since dvec.h is not part of GCC.
// ============================================================================
#ifdef HAS_INTEL_DVEC
#include <dvec.h>

void SelectAddMul_IntelVC(short int aa[], short int bb[], short int cc[]) {
    Is16vec8 zero(0, 0, 0, 0, 0, 0, 0, 0);
    Is16vec8 two(2, 2, 2, 2, 2, 2, 2, 2);

    for (int i = 0; i < N_SHORT; i += 8) {
        Is16vec8 b = LoadVectorU(bb + i);
        Is16vec8 c = LoadVectorU(cc + i);
        Is16vec8 a = select_gt(b, zero, c + two, b * c);
        StoreVectorU(aa + i, a);
    }
}
#else
void SelectAddMul_IntelVC(short int[], short int[], short int[]) {
    std::fprintf(stderr, "[skip] Intel dvec.h not available; using SSE4.1 fallback\n");
}
#endif

// ============================================================================
// Example 12.4e: VCL (Vector Class Library) version (Vec16s)
// Requires Agner Fog's vectorclass.h.
// Wrapped in #ifdef guard.
// ============================================================================
#ifdef HAS_VCL
#include "vectorclass.h"

void SelectAddMul_VCL(short int aa[], short int bb[], short int cc[]) {
    Vec16s a, b, c;
    for (int i = 0; i < N_SHORT; i += 16) {
        b.load(bb + i);
        c.load(cc + i);
        a = select(b > 0, c + 2, b * c);
        a.store(aa + i);
    }
}
#else
void SelectAddMul_VCL(short int[], short int[], short int[]) {
    std::fprintf(stderr, "[skip] VCL vectorclass.h not available; using SSE4.1 fallback\n");
}
#endif

// ============================================================================
// Example 12.6: VCL polynomial function (Vec4f)
// Uses VCL vector types directly as function parameters.
// ============================================================================
#ifdef HAS_VCL
#include "vectorclass.h"

Vec4f polynomial_VCL(Vec4f const& x) {
    // polynomial(x) = 2.5*x^2 - 8*x + 2
    return (2.5f * x - 8.0f) * x + 2.0f;
}
#endif

// Scalar equivalent for comparison
float polynomial_scalar(float x) {
    return (2.5f * x - 8.0f) * x + 2.0f;
}

// ============================================================================
// Example 12.7: CPU instruction set auto-dispatch mechanism
// Uses __builtin_cpu_supports (GCC 4.8+) for runtime CPU detection.
// Falls back to compile-time detection via preprocessor macros.
// ============================================================================

// Forward declarations for dispatch variants
typedef void SelectAddMulFunc(short int aa[], short int bb[], short int cc[]);

static SelectAddMulFunc* g_SelectAddMul_ptr = nullptr;

static void SelectAddMul_dispatch_setup() {
#if defined(__SSE4_1__) || defined(__AVX__) || defined(__AVX2__)
    // Use GCC builtin to detect supported instruction sets at runtime
    bool has_avx2 = __builtin_cpu_supports("avx2");
    bool has_sse4_1 = __builtin_cpu_supports("sse4.1");

    if (has_avx2) {
        // AVX2 would use Vec16s with 16-wide vectors, but since we don't
        // have a separate AVX2 implementation, fall to SSE4.1
        g_SelectAddMul_ptr = &SelectAddMul_SSE41;
    } else if (has_sse4_1) {
        g_SelectAddMul_ptr = &SelectAddMul_SSE41;
    } else {
        g_SelectAddMul_ptr = &SelectAddMul_SSE2;
    }
#else
    // Assume at least SSE2 is available
    g_SelectAddMul_ptr = &SelectAddMul_SSE2;
#endif
}

// Dispatcher function: on first call, detects CPU and sets the function pointer
void SelectAddMul_dispatch(short int aa[], short int bb[], short int cc[]) {
    if (g_SelectAddMul_ptr == nullptr) {
        SelectAddMul_dispatch_setup();
    }
    (*g_SelectAddMul_ptr)(aa, bb, cc);
}

// ============================================================================
// Example 12.8a / 12.8b: Sum loop - scalar vs unrolled
// 12.8a: Simple scalar accumulation
// 12.8b: Loop unrolled by 4 for better instruction-level parallelism
// ============================================================================

float sum_scalar(const float a[], int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += a[i];
    }
    return sum;
}

float sum_unrolled4(const float a[], int n) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    int i = 0;
    for (; i + 3 < n; i += 4) {
        s0 += a[i];
        s1 += a[i + 1];
        s2 += a[i + 2];
        s3 += a[i + 3];
    }
    // Handle remaining elements
    for (; i < n; ++i) {
        s0 += a[i];
    }
    return (s0 + s1) + (s2 + s3);
}

// ============================================================================
// Example 12.9a: Taylor series - scalar Exp() approximation
// exp(x) ≈ Σ x^n / n!  for n = 0..16
// ============================================================================
float Exp_scalar(float x) {
    float xn = x;
    float sum = 1.0f;   // n=0 term
    float nfac = 1.0f;  // 0!

    for (int n = 1; n <= 16; ++n) {
        sum += xn / nfac;
        xn *= x;
        nfac *= (n + 1);
    }
    return sum;
}

// ============================================================================
// Example 12.9b: Taylor series - SSE3 vectorized Exp()
// Uses precomputed 1/n! table and vectorized Horner-like accumulation
// ============================================================================

// Sum the 4 floats of an __m128 using SSE3 horizontal adds
static inline float hadd_sse3(__m128 const& x) {
    __m128 s;
    s = _mm_hadd_ps(x, x);  // [a+b, c+d, a+b, c+d]
    s = _mm_hadd_ps(s, s);  // [a+b+c+d, ...]
    return _mm_cvtss_f32(s);
}

float Exp_vectorized(float x) {
    // Precomputed 1/n! values (aligned for SIMD loads)
    Align16(const float coef[16]) = {1.0f,
                                     1.0f / 2.0f,
                                     1.0f / 6.0f,
                                     1.0f / 24.0f,
                                     1.0f / 120.0f,
                                     1.0f / 720.0f,
                                     1.0f / 5040.0f,
                                     1.0f / 40320.0f,
                                     1.0f / 362880.0f,
                                     1.0f / 3628800.0f,
                                     1.0f / 39916800.0f,
                                     1.0f / 4.790016E8f,
                                     1.0f / 6.22702E9f,
                                     1.0f / 8.71782E10f,
                                     1.0f / 1.30767E12f,
                                     1.0f / 2.09227E13f};

    float x2 = x * x;
    float x4 = x2 * x2;

    // _mm_set_ps maps args to lanes [3,2,1,0] = first arg is highest lane.
    // This matches Intel F32vec4 constructor convention.
    // xxn = [x^4, x^3, x^2, x^1] in lanes 3..0 so that lane0=x, lane1=x^2, ...
    __m128 xxn = _mm_set_ps(x4, x2 * x, x2, x);
    __m128 xx4 = _mm_set1_ps(x4);
    __m128 s = _mm_set_ps(0.0f, 0.0f, 0.0f, 1.0f);  // lane0 = 1.0 (n=0 term)

    for (int i = 0; i < 16; i += 4) {
        // s += x^n * (1/n!)  for 4 terms at once
        s = _mm_add_ps(s, _mm_mul_ps(xxn, _mm_load_ps(coef + i)));
        xxn = _mm_mul_ps(xxn, xx4);  // next four powers: x^5..x^8, etc.
    }
    return hadd_sse3(s);
}

// ============================================================================
// Verification helpers
// ============================================================================

static bool verify_select_add_mul(SelectAddMulFunc* ref, SelectAddMulFunc* test,
                                  const char* label) {
    Align16(short int aa_ref[N_SHORT]);
    Align16(short int aa_tst[N_SHORT]);
    Align16(short int bb[N_SHORT]);
    Align16(short int cc[N_SHORT]);

    // Fill with deterministic test data: alternating positive/negative
    for (int i = 0; i < N_SHORT; ++i) {
        bb[i] = (short int)((i % 2 == 0) ? (i + 1) : -(i + 1));
        cc[i] = (short int)(i * 3 + 1);
    }
    std::memcpy(aa_ref, bb, sizeof(bb));  // copy bb as initial
    std::memcpy(aa_tst, bb, sizeof(bb));

    ref(aa_ref, bb, cc);
    test(aa_tst, bb, cc);

    for (int i = 0; i < N_SHORT; ++i) {
        if (aa_ref[i] != aa_tst[i]) {
            std::fprintf(stderr, "[FAIL] %s: mismatch at index %d: ref=%d test=%d\n", label, i,
                         (int)aa_ref[i], (int)aa_tst[i]);
            return false;
        }
    }
    std::printf("[ OK ] %s: all %d elements match\n", label, N_SHORT);
    return true;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    const int N_LARGE = 1024 * 1024;  // 1M elements for timing
    const int N_REPEAT = 100;

    std::printf("=== Chapter 12: Vector Operations - Consolidated Examples ===\n");
    std::printf("Compiled with: -std=c++11 -O2 -msse4.1\n");
    std::printf("\n");

    // ---- 12.1: Automatic vectorization with __restrict ----
    std::printf("--- 12.1: Automatic vectorization (__restrict) ---\n");
    {
        int* aa = new int[N_LARGE];
        int* bb = new int[N_LARGE];
        for (int i = 0; i < N_LARGE; ++i)
            bb[i] = i;

        double t0 = get_time_sec();
        for (int r = 0; r < N_REPEAT; ++r)
            AddTwo_scalar(N_LARGE, aa, bb);
        double t1 = get_time_sec();
        for (int r = 0; r < N_REPEAT; ++r)
            AddTwo_restrict(N_LARGE, aa, bb);
        double t2 = get_time_sec();

        // Warm prevention: use result
        volatile int sink = aa[0];
        (void)sink;

        std::printf("  scalar   (no __restrict): %.4f ms\n", (t1 - t0) * 1000.0);
        std::printf("  restrict (__restrict)  : %.4f ms\n", (t2 - t1) * 1000.0);

        delete[] aa;
        delete[] bb;
    }
    std::printf("\n");

    // ---- 12.2: Struct alignment ----
    std::printf("--- 12.2: Struct alignment __attribute__((aligned(16))) ---\n");
    {
        // Verify alignment
        S2 s_instance;
        std::printf("  sizeof(S2) = %zu, alignment = %zu (should be 16)\n", sizeof(S2),
                    alignof(S2));
        // Run the function
        double t0 = get_time_sec();
        for (int r = 0; r < N_LARGE; ++r)
            FuncS2();
        double t1 = get_time_sec();
        std::printf("  FuncS2: %zu iterations in %.4f ms\n", (size_t)N_LARGE, (t1 - t0) * 1000.0);
    }
    std::printf("\n");

    // ---- 12.4: Branch loop - scalar vs SSE2 vs SSE4.1 ----
    std::printf("--- 12.4: SelectAddMul (scalar vs SSE2 vs SSE4.1) ---\n");

    // Verify correctness first
    verify_select_add_mul(&SelectAddMul_scalar, &SelectAddMul_SSE2, "SSE2");
    verify_select_add_mul(&SelectAddMul_scalar, &SelectAddMul_SSE41, "SSE4.1");

    {
        Align16(short int aa[N_SHORT]);
        Align16(short int bb[N_SHORT]);
        Align16(short int cc[N_SHORT]);

        for (int i = 0; i < N_SHORT; ++i) {
            bb[i] = (short int)((i % 2 == 0) ? (i + 1) : -(i + 1));
            cc[i] = (short int)(i * 3 + 1);
        }

        const int REP = 100000;

        double t0 = get_time_sec();
        for (int r = 0; r < REP; ++r)
            SelectAddMul_scalar(aa, bb, cc);
        double t1 = get_time_sec();
        for (int r = 0; r < REP; ++r)
            SelectAddMul_SSE2(aa, bb, cc);
        double t2 = get_time_sec();
        for (int r = 0; r < REP; ++r)
            SelectAddMul_SSE41(aa, bb, cc);
        double t3 = get_time_sec();

        volatile short int sink = aa[0];
        (void)sink;

        std::printf("  scalar : %.4f ms\n", (t1 - t0) * 1000.0);
        std::printf("  SSE2   : %.4f ms\n", (t2 - t1) * 1000.0);
        std::printf("  SSE4.1 : %.4f ms\n", (t3 - t2) * 1000.0);
    }
    std::printf("\n");

    // ---- 12.7: CPU dispatch ----
    std::printf("--- 12.7: CPU instruction set dispatch ---\n");
    {
        Align16(short int aa[N_SHORT]);
        Align16(short int bb[N_SHORT]);
        Align16(short int cc[N_SHORT]);

        for (int i = 0; i < N_SHORT; ++i) {
            bb[i] = (short int)((i % 2 == 0) ? (i + 1) : -(i + 1));
            cc[i] = (short int)(i * 3 + 1);
        }

        SelectAddMul_dispatch(aa, bb, cc);

#if defined(__SSE4_1__)
        std::printf("  Dispatcher selected: SSE4.1 (compiled with -msse4.1)\n");
#else
        std::printf("  Dispatcher selected: SSE2\n");
#endif
        // Verify result matches scalar
        bool ok = true;
        Align16(short int aa_ref[N_SHORT]);
        SelectAddMul_scalar(aa_ref, bb, cc);
        for (int i = 0; i < N_SHORT; ++i) {
            if (aa_ref[i] != aa[i]) {
                std::fprintf(stderr, "[FAIL] dispatch: mismatch at %d\n", i);
                ok = false;
                break;
            }
        }
        if (ok)
            std::printf("  [ OK ] dispatch result matches scalar\n");
    }
    std::printf("\n");

    // ---- 12.8: Sum loop - scalar vs unrolled ----
    std::printf("--- 12.8: Sum loop (scalar vs unrolled by 4) ---\n");
    {
        float* a = new float[N_LARGE];
        for (int i = 0; i < N_LARGE; ++i)
            a[i] = (float)(i % 100) * 0.01f;

        float result_s, result_u;

        double t0 = get_time_sec();
        for (int r = 0; r < N_REPEAT; ++r)
            result_s = sum_scalar(a, N_LARGE);
        double t1 = get_time_sec();
        for (int r = 0; r < N_REPEAT; ++r)
            result_u = sum_unrolled4(a, N_LARGE);
        double t2 = get_time_sec();

        std::printf("  scalar  sum = %.6f, time = %.4f ms\n", (double)result_s, (t1 - t0) * 1000.0);
        std::printf("  unrolled sum = %.6f, time = %.4f ms\n", (double)result_u,
                    (t2 - t1) * 1000.0);

        double diff = std::fabs((double)result_s - (double)result_u);
        if (diff > 1.0)
            std::fprintf(stderr, "[WARN] sum_scalar and sum_unrolled4 differ by %g\n", diff);
        else
            std::printf("  [ OK ] sums match within tolerance (diff = %g)\n", diff);

        delete[] a;
    }
    std::printf("\n");

    // ---- 12.9: Taylor series Exp() - scalar vs SSE3 vectorized ----
    std::printf("--- 12.9: Taylor series Exp(x) (scalar vs SSE3 vectorized) ---\n");
    {
        const int N_VALUES = 1024;
        float* inputs = new float[N_VALUES];
        float* out_s = new float[N_VALUES];
        float* out_v = new float[N_VALUES];

        for (int i = 0; i < N_VALUES; ++i)
            inputs[i] = (float)(i % 20) * 0.05f;  // x in [0, 0.95]

        const int REP = 50000;

        // Scalar
        double t0 = get_time_sec();
        for (int r = 0; r < REP; ++r) {
            for (int i = 0; i < N_VALUES; ++i)
                out_s[i] = Exp_scalar(inputs[i]);
        }
        double t1 = get_time_sec();
        // Vectorized
        for (int r = 0; r < REP; ++r) {
            for (int i = 0; i < N_VALUES; ++i)
                out_v[i] = Exp_vectorized(inputs[i]);
        }
        double t2 = get_time_sec();

        std::printf("  scalar     : %.4f ms\n", (t1 - t0) * 1000.0);
        std::printf("  vectorized : %.4f ms\n", (t2 - t1) * 1000.0);

        // Verify accuracy
        double max_error = 0.0;
        for (int i = 0; i < N_VALUES; ++i) {
            double err = std::fabs((double)out_s[i] - (double)out_v[i]);
            if (err > max_error)
                max_error = err;
        }
        std::printf("  max abs error (scalar vs vectorized): %e\n", max_error);

        // Compare against std::exp for the first few values
        std::printf("  sample values (x, scalar, vectorized, std::exp):\n");
        for (int i = 0; i < 5; ++i) {
            std::printf("    x=%.3f | scalar=%.8f | vec=%.8f | std::exp=%.8f\n", (double)inputs[i],
                        (double)out_s[i], (double)out_v[i], std::exp((double)inputs[i]));
        }

        delete[] inputs;
        delete[] out_s;
        delete[] out_v;
    }
    std::printf("\n");

    // ---- 12.6: VCL polynomial (scalar baseline) ----
    std::printf("--- 12.6: Polynomial 2.5*x^2 - 8*x + 2 (scalar) ---\n");
    {
        const int REP = 10000000;
        float x = 3.14f;
        volatile float y;

        double t0 = get_time_sec();
        for (int r = 0; r < REP; ++r)
            y = polynomial_scalar(x);
        double t1 = get_time_sec();

        std::printf("  scalar: %.4f ms  (result = %.6f)\n", (t1 - t0) * 1000.0,
                    (double)polynomial_scalar(x));
#ifdef HAS_VCL
        // If VCL is available, also test the vector version
        Vec4f vx(3.14f);
        Vec4f vy = polynomial_VCL(vx);
        std::printf("  VCL   : result = [%.6f, %.6f, %.6f, %.6f]\n", (double)vy[0], (double)vy[1],
                    (double)vy[2], (double)vy[3]);
#endif
    }
    std::printf("\n");

    // ---- Summary ----
    std::printf("=== All examples completed successfully ===\n");

    return 0;
}
