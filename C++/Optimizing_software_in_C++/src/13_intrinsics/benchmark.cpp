// 13_intrinsics: compare scalar vs SSE2 vs AVX2 on the same operations.
//
// PDF 12.4 (p121-124). The AVX2 path is only used when the CPU supports it
// (checked at runtime). Results must match the scalar version (checksum).
#include <cfloat>
#include <cstdio>
#include <immintrin.h>
#include <vector>

#include "common/benchmark.h"
#include "common/cpu_info.h"

// --- scalar reference implementations -------------------------------------
float scalar_dot(const float* a, const float* b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}
float scalar_reduce(const float* a, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += a[i];
    return s;
}

// --- SSE2 ------------------------------------------------------------------
float sse_dot(const float* a, const float* b, int n) {
    __m128 acc = _mm_setzero_ps();
    int i = 0;
    for (; i + 4 <= n; i += 4)
        acc = _mm_add_ps(acc, _mm_mul_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
    float s = 0.0f;
    for (; i < n; ++i) s += a[i] * b[i];
    float t[4];
    _mm_storeu_ps(t, acc);
    return (t[0] + t[1]) + (t[2] + t[3]) + s;
}

// --- AVX2 ------------------------------------------------------------------
__attribute__((target("avx2"))) float avx2_dot(const float* a, const float* b, int n) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8)
        acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(a + i),
                                               _mm256_loadu_ps(b + i)));
    float s = 0.0f;
    for (; i < n; ++i) s += a[i] * b[i];
    float t[8];
    _mm256_storeu_ps(t, acc);
    return ((t[0]+t[1])+(t[2]+t[3])) + ((t[4]+t[5])+(t[6]+t[7])) + s;
}

__attribute__((target("avx2"))) float avx2_reduce(const float* a, int n) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) acc = _mm256_add_ps(acc, _mm256_loadu_ps(a + i));
    float s = 0.0f;
    for (; i < n; ++i) s += a[i];
    float t[8];
    _mm256_storeu_ps(t, acc);
    return ((t[0]+t[1])+(t[2]+t[3])) + ((t[4]+t[5])+(t[6]+t[7])) + s;
}

int main() {
    const int n = 8'000'000;
    std::vector<float> a(n), b(n, 1.0f);
    for (int i = 0; i < n; ++i) a[i] = (float)(i % 1024);

    cpu_print_info();
    std::printf("\n");

    bench("scalar_dot",    [&] { return scalar_dot(a.data(), b.data(), n); });
    bench("sse_dot",       [&] { return sse_dot(a.data(), b.data(), n); });
    if (cpu_has_avx2()) {
        bench("avx2_dot",  [&] { return avx2_dot(a.data(), b.data(), n); });
    }
    bench("scalar_reduce", [&] { return scalar_reduce(a.data(), n); });
    if (cpu_has_avx2()) {
        bench("avx2_reduce",[&] { return avx2_reduce(a.data(), n); });
    }

    // checksum: all implementations must agree (within FP tolerance).
    float d0 = scalar_dot(a.data(), b.data(), n);
    float d1 = sse_dot(a.data(), b.data(), n);
    float d2 = cpu_has_avx2() ? avx2_dot(a.data(), b.data(), n) : d0;
    std::printf("\nchecksums: scalar=%.1f sse=%.1f avx2=%.1f\n", d0, d1, d2);
    return 0;
}
