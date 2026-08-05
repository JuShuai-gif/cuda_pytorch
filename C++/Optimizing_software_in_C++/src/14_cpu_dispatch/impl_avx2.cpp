// 14_cpu_dispatch: AVX2 implementation, compiled with -mavx2 in its own TU.
//
// Uses vector class-free intrinsics for a float add of two arrays.
#include <immintrin.h>

// Forward declaration of the interface (see dispatch.h/dispatch.cpp).
void vadd_scalar(float* c, const float* a, const float* b, int n);
void vadd_sse2(float* c, const float* a, const float* b, int n);
void vadd_avx2(float* c, const float* a, const float* b, int n);

void vadd_avx2(float* c, const float* a, const float* b, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(c + i, _mm256_add_ps(va, vb));
    }
    for (; i < n; ++i) c[i] = a[i] + b[i];
}
