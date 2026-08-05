// 09_alignment: compare aligned vs misaligned SIMD loads and the address.
//
// PDF 9.5 (p95), 12.8 (p133). On modern x86 the penalty for misaligned
// *loads* is small or zero when the line split is handled by hardware;
// aligned access is still the "free optimization". We print the alignment
// of each buffer and time aligned vs misaligned vector code.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "common/benchmark.h"

int main() {
    const size_t n = 8'000'000;

    std::vector<float> aligned(n, 1.0f);
    std::vector<float> out_aligned(n, 0.0f);
    std::vector<float> misaligned(n + 2, 1.0f);   // offset below
    std::vector<float> out_mis(n + 2, 0.0f);

    float* A = aligned.data();
    float* B = misaligned.data() + 1;             // +4 bytes
    float* C = out_aligned.data();
    float* D = out_mis.data() + 1;

    std::printf("A aligned by %zu;  B offset pointer aligned by %zu\n",
                (size_t)((char*)A - (char*)0) % 64,
                (size_t)((char*)B - (char*)0) % 64);

    bench("aligned_copy", [&] {
        for (size_t i = 0; i < n; ++i) C[i] = A[i] + 2.0f;
        return C[n - 1];
    });

    bench("misaligned_copy", [&] {
        for (size_t i = 0; i < n; ++i) D[i] = B[i] + 2.0f;
        return D[n - 1];
    });

    std::printf("\nresults: %.1f %.1f\n", C[n - 1], D[n - 1]);
    return 0;
}
