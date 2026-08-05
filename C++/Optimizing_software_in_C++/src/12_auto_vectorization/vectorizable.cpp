// 12_auto_vectorization: vectorizable loop (restrict + aligned arrays).
//
// PDF 12.3 (p118-121): __restrict__ tells the compiler pointers do not
// alias; compile with -mavx2 to vectorize 8 floats per instruction.
#include <cstdio>
#include <cstdlib>
#include <vector>

void add_two(float* __restrict__ a, const float* __restrict__ b, int n) {
    for (int i = 0; i < n; ++i) a[i] = b[i] + 2.0f;
}

int main() {
    const int n = 8'000'000;
    std::vector<float> a(n), b(n, 1.0f);
    add_two(a.data(), b.data(), n);
    volatile float r = a[n - 1];
    std::printf("checksum = %.1f\n", r);
    return 0;
}
