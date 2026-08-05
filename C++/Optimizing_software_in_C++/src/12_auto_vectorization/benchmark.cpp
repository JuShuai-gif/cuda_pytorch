// 12_auto_vectorization: compare a vectorized loop vs the branchy version.
//
// PDF 12.3 (p118-121). The key takeaway: with -mavx2 the compiler vectorizes
// the first loop; the branchy one needs -fno-trapping-math or manual work.
#include <cstdio>
#include <vector>

#include "common/benchmark.h"

void add_two(float* __restrict__ a, const float* __restrict__ b, int n) {
    for (int i = 0; i < n; ++i) a[i] = b[i] + 2.0f;
}

void add_conditional(float* a, float* b, int n) {
    for (int i = 0; i < n; ++i) {
        if (b[i] > 0.5f) a[i] = b[i] + 2.0f;
        else a[i] = b[i] * 2.0f;
    }
}

int main() {
    const int n = 8'000'000;
    std::vector<float> a1(n), a2(n), b(n, 1.0f);

    bench("add_two (restrict)", [&] {
        add_two(a1.data(), b.data(), n);
        return a1[n - 1];
    });

    bench("add_conditional", [&] {
        add_conditional(a2.data(), b.data(), n);
        return a2[n - 1];
    });

    // identical math for this input
    std::printf("\nresults: %.1f %.1f\n", a1[n - 1], a2[n - 1]);
    return 0;
}
