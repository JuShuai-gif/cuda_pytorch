// 12_auto_vectorization: loop that is hard to vectorize (aliasing + branch).
//
// PDF 12.3 (p119): obstacles to auto-vectorization include pointer aliasing,
// branches with possible side effects, and calls to external functions.
#include <cstdio>
#include <cstdlib>
#include <vector>

// No __restrict__: a[] and b[] could overlap, so the compiler must be
// conservative. The data-dependent branch also blocks clean vectorization.
void add_conditional(float* a, float* b, int n) {
    for (int i = 0; i < n; ++i) {
        if (b[i] > 0.5f) {
            a[i] = b[i] + 2.0f;      // branch inside loop
        } else {
            a[i] = b[i] * 2.0f;
        }
    }
}

int main() {
    const int n = 8'000'000;
    std::vector<float> a(n), b(n, 1.0f);
    add_conditional(a.data(), b.data(), n);
    volatile float r = a[n - 1];
    std::printf("checksum = %.1f\n", r);
    return 0;
}
