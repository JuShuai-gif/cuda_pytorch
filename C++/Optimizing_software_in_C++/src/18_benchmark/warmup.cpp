// 18_benchmark: cold vs warm -- the first call pays cache + branch misses.
//
// PDF p168: the first measurement is the "worst case" (cold), later ones the
// "best case" (warm). Which matters depends on how the function is used.
#include <chrono>
#include <cstdio>
#include <vector>

double sum(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x;
    return s;
}

int main() {
    std::vector<double> v(64'000'000, 1.0);

    // No warm-up: readings should start high and drop.
    for (int k = 0; k < 8; ++k) {
        auto t0 = std::chrono::steady_clock::now();
        volatile double r = sum(v);
        auto t1 = std::chrono::steady_clock::now();
        (void)r;
        std::printf("call %d: %8.2f us\n", k,
            std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    return 0;
}
