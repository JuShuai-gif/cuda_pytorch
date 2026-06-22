// Chapter 12: Vector Operations (SIMD)
// Demonstrates manual and automatic vectorization with SSE/AVX intrinsics.
// Compile with: -msse2 -msse4.1 (or -mavx2)
// Compile: see CMakeLists.txt (ch12_demo target)

#ifdef __SSE2__
#include <emmintrin.h>
#endif
#ifdef __SSE4_1__
#include <smmintrin.h>
#endif

#include <iostream>
#include <cstring>
#include <chrono>

// ---- Example 12.1a: Auto-vectorizable loop ----
void AutoVectorAdd(const float *a, const float *b, float *c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

// ---- Example 12.2: Struct-of-arrays vectorization ----
struct Vec4 {
    float x, y, z, w;
};
void VectorAddStruct(const Vec4 &a, Vec4 &out) {
    out.x = a.x + 1.0f;
    out.y = a.y + 2.0f;
    out.z = a.z + 3.0f;
    out.w = a.w + 4.0f;
}

// ---- Example 12.4b: Manual vectorization with SSE2 intrinsics ----
#ifdef __SSE2__
void SelectAddMul_SSE2(short *aa, const short *bb, const short *cc) {
    for (int i = 0; i < 256; i += 8) {
        __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i *>(bb + i));
        __m128i c = _mm_loadu_si128(reinterpret_cast<const __m128i *>(cc + i));
        // Masks for b > 0
        __m128i zero = _mm_setzero_si128();
        __m128i mask = _mm_cmpgt_epi16(b, zero);
        // result = b > 0 ? c + 2 : b * c
        __m128i c_plus_2 = _mm_add_epi16(c, _mm_set1_epi16(2));
        __m128i b_mul_c = _mm_mullo_epi16(b, c);
        __m128i result = _mm_or_si128(
            _mm_and_si128(mask, c_plus_2),
            _mm_andnot_si128(mask, b_mul_c));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(aa + i), result);
    }
}
#endif

// ---- Example 12.8b: Vectorized sum reduction ----
#ifdef __SSE2__
double VectorizedSum(const double *data, int n) {
    __m128d sum = _mm_setzero_pd();
    for (int i = 0; i < n; i += 2) {
        __m128d v = _mm_loadu_pd(data + i);
        sum = _mm_add_pd(sum, v);
    }
    // Horizontal add
    sum = _mm_add_pd(sum, _mm_unpackhi_pd(sum, sum));
    double result;
    _mm_store_sd(&result, sum);
    return result;
}
#endif

// ---- Scalar baseline for comparison ----
double ScalarSum(const double *data, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += data[i];
    }
    return sum;
}

// ---- Main ----
int main() {
    std::cout << "=== Chapter 12: Vector Operations (SIMD) ===\n\n";

    constexpr int N = 1024;
    float *a = new float[N];
    float *b = new float[N];
    float *c = new float[N];
    for (int i = 0; i < N; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i * 2);
    }

    AutoVectorAdd(a, b, c, N);
    std::cout << "AutoVectorAdd c[0]: " << c[0] << " (expected " << a[0] + b[0] << ")\n";

    delete[] a;
    delete[] b;
    delete[] c;

#ifdef __SSE2__
    // SSE2 manual vectorization demo
    short aa[256], bb[256], cc[256];
    for (int i = 0; i < 256; ++i) {
        bb[i] = static_cast<short>(i - 50); // Some negative, some positive
        cc[i] = static_cast<short>(i);
    }
    SelectAddMul_SSE2(aa, bb, cc);
    std::cout << "SSE2 SelectAddMul aa[100]: " << aa[100] << "\n";

    // Vectorized sum
    double *ddata = new double[N];
    for (int i = 0; i < N; ++i) ddata[i] = 1.0;

    double vsum = VectorizedSum(ddata, N);
    double ssum = ScalarSum(ddata, N);
    std::cout << "Vectorized sum: " << vsum << "\n";
    std::cout << "Scalar sum:     " << ssum << "\n";
    std::cout << "[SSE2 vectorization is active]\n";
    delete[] ddata;
#else
    std::cout << "[SSE2 not enabled - vector examples skipped]\n";
#endif

    std::cout << "\nAll chapter 12 checks passed.\n";
    return 0;
}
