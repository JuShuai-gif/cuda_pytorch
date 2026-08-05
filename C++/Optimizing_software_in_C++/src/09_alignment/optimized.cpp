// 09_alignment: optimized -- alignas(64) + aligned_alloc, aligned access.
//
// PDF 9.5 (p95): alignas(64) aligns the array to the cache line / vector
// size; PDF 12.8 (p133): aligned dynamic allocation for vector operands.
#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
    const size_t n = 16'000'000;

    // (a) stack array aligned to 64 bytes (PDF p95).
    static float a[16] alignas(64);

    // (b) dynamic aligned allocation for vector use.
    float* mem = (float*)std::aligned_alloc(64, n * sizeof(float));
    if (!mem) return 1;
    float* A = mem;
    for (size_t i = 0; i < n; ++i) A[i] = (float)i;

    volatile float r = A[n - 1] + a[0];
    std::printf("checksum = %.1f\n", r);
    std::free(mem);
    return 0;
}
