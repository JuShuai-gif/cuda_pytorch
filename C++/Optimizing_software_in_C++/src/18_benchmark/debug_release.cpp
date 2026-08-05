// 18_benchmark: Debug vs Release -- the SAME source behaves completely
// differently, proving that speed tests must use an optimized build
// (PDF p85: debugging and optimization are incompatible).
//
// Built twice: 18_debug (-O0) and 18_release (-O3).
#include <cstdio>

double work(int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += (double)i * 1.0000001;
    return s;
}

int main() {
    // Without using the result in a way the compiler can see, -O3 may
    // remove the whole computation. We print it, so it stays.
    std::printf("result = %.6f\n", work(100'000'000));
    return 0;
}
