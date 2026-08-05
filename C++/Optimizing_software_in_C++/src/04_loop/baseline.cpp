// 04_loop: baseline -- serial accumulation (loop-carried dependency chain).
//
// PDF 11 (p113-114): `sum += list[i]` is a long dependency chain; each
// addition waits for the previous one.
#include <cstdio>
#include <vector>

double sum_serial(const std::vector<double>& a) {
    double sum = 0.0;
    for (double x : a) sum += x;
    return sum;
}

int main() {
    std::vector<double> a(16'000'000, 1.0);
    volatile double r = sum_serial(a);
    std::printf("checksum = %.0f\n", r);
    return 0;
}
