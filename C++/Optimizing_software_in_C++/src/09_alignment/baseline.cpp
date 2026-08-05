// 09_alignment: baseline -- unaligned SIMD loads (loadu) on a malloc buffer.
//
// PDF 9.5 (p95) and 12.3 (p118): aligned access is preferable; unaligned
// vector loads are slower on some CPUs (especially Atom/older Intel).
#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
    const size_t n = 16'000'000;
    // malloc generally returns 16-byte-aligned memory; simulate a misaligned
    // pointer by offsetting by one element.
    std::vector<float> b(n + 1, 1.0f);
    std::vector<float> a(n + 1, 0.0f);
    float* A = a.data() + 1;   // offset -> A is 4 bytes misaligned
    float* B = b.data() + 1;

    for (size_t i = 0; i < n; ++i) A[i] = B[i] + 2.0f;

    volatile float r = A[n - 1];
    std::printf("checksum = %.1f\n", r);
    return 0;
}
