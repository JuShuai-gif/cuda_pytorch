// 13_intrinsics: optimized -- SSE2 / AVX2 intrinsics for dot, reduce, min/max.
//
// PDF 12.4 (p121-124). Runtime-checked with cpu_info: uses AVX2 when the CPU
// supports it, otherwise falls back to SSE2.
#include <cfloat>
#include <cstdio>
#include <immintrin.h>
#include <vector>

#include "common/cpu_info.h"

// ---- dot product ----------------------------------------------------------
float sse_dot(const float* a, const float* b, int n) {
    __m128 acc = _mm_setzero_ps();
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        acc = _mm_add_ps(acc, _mm_mul_ps(va, vb));
    }
    float s = 0.0f;
    for (; i < n; ++i) s += a[i] * b[i];
    float tmp[4];
    _mm_storeu_ps(tmp, acc);
    return (tmp[0] + tmp[1]) + (tmp[2] + tmp[3]) + s;
}

__attribute__((target("avx2"))) float avx2_dot(const float* a, const float* b, int n) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
    }
    float s = 0.0f;
    for (; i < n; ++i) s += a[i] * b[i];
    float tmp[8];
    _mm256_storeu_ps(tmp, acc);
    return (tmp[0] + tmp[1]) + (tmp[2] + tmp[3]) +
           (tmp[4] + tmp[5]) + (tmp[6] + tmp[7]) + s;
}

// ---- reduction ------------------------------------------------------------
float sse_reduce(const float* a, int n) {
    __m128 acc = _mm_setzero_ps();
    int i = 0;
    for (; i + 4 <= n; i += 4) acc = _mm_add_ps(acc, _mm_loadu_ps(a + i));
    float s = 0.0f;
    for (; i < n; ++i) s += a[i];
    float tmp[4];
    _mm_storeu_ps(tmp, acc);
    return (tmp[0] + tmp[1]) + (tmp[2] + tmp[3]) + s;
}

__attribute__((target("avx2"))) float avx2_reduce(const float* a, int n) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) acc = _mm256_add_ps(acc, _mm256_loadu_ps(a + i));
    float s = 0.0f;
    for (; i < n; ++i) s += a[i];
    float tmp[8];
    _mm256_storeu_ps(tmp, acc);
    return ((tmp[0] + tmp[1]) + (tmp[2] + tmp[3])) +
           ((tmp[4] + tmp[5]) + (tmp[6] + tmp[7])) + s;
}

// ---- min / max (fallback) -------------------------------------------------
void sse_minmax(const float* a, int n, float& mn, float& mx) {
    mn = FLT_MAX;
    mx = -FLT_MAX;
    for (int i = 0; i < n; ++i) {
        mn = a[i] < mn ? a[i] : mn;
        mx = a[i] > mx ? a[i] : mx;
    }
}

// ---- min / max ------------------------------------------------------------
__attribute__((target("avx2"))) void avx2_minmax(const float* a, int n, float& mn, float& mx) {
    __m256 vmn = _mm256_set1_ps(FLT_MAX);
    __m256 vmx = _mm256_set1_ps(-FLT_MAX);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(a + i);
        vmn = _mm256_min_ps(vmn, v);
        vmx = _mm256_max_ps(vmx, v);
    }
    mn = FLT_MAX;
    mx = -FLT_MAX;
    for (; i < n; ++i) {
        mn = a[i] < mn ? a[i] : mn;
        mx = a[i] > mx ? a[i] : mx;
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, vmn);
    for (int k = 0; k < 8; ++k) mn = tmp[k] < mn ? tmp[k] : mn;
    _mm256_storeu_ps(tmp, vmx);
    for (int k = 0; k < 8; ++k) mx = tmp[k] > mx ? tmp[k] : mx;
}

int main() {
    const int n = 8'000'000;
    std::vector<float> a(n), b(n, 1.0f);
    for (int i = 0; i < n; ++i) a[i] = (float)(i % 1024);

    float d, s, mn, mx;
    if (cpu_has_avx2()) {
        d = avx2_dot(a.data(), b.data(), n);
        s = avx2_reduce(a.data(), n);
        avx2_minmax(a.data(), n, mn, mx);
    } else {
        d = sse_dot(a.data(), b.data(), n);
        s = sse_reduce(a.data(), n);
        sse_minmax(a.data(), n, mn, mx);
    }
    std::printf("dot=%.1f sum=%.1f min=%.0f max=%.0f\n", d, s, mn, mx);
    return 0;
}
