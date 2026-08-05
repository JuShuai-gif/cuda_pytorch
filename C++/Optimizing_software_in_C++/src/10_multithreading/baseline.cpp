// 10_multithreading: baseline -- serial sum.
#include <cstdio>
#include <vector>

double serial_sum(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x;
    return s;
}

int main() {
    std::vector<double> v(64'000'000, 1.0);
    volatile double r = serial_sum(v);
    std::printf("checksum = %.0f\n", r);
    return 0;
}
