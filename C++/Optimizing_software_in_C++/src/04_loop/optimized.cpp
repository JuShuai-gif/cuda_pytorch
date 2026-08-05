// 04_loop: optimized -- 4 accumulators break the dependency chain.
//
// PDF 11 (p113-114): four independent partial sums keep the FP adder(s) busy.
#include <cstdio>
#include <vector>

double sum_parallel(const std::vector<double>& a) {
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    size_t i = 0;
    for (; i + 3 < a.size(); i += 4) {
        s0 += a[i];
        s1 += a[i + 1];
        s2 += a[i + 2];
        s3 += a[i + 3];
    }
    for (; i < a.size(); ++i) s0 += a[i];
    return (s0 + s1) + (s2 + s3);
}

int main() {
    std::vector<double> a(16'000'000, 1.0);
    volatile double r = sum_parallel(a);
    std::printf("checksum = %.0f\n", r);
    return 0;
}
