// 18_benchmark: dead code elimination -- a benchmark that measures nothing.
//
// In -O3, a computation whose result is never used can be removed entirely.
// This program demonstrates the classic pitfall: "optimizing" an already
// eliminated loop.
#include <cstdio>

// The compiler is allowed to remove this loop entirely when the result is
// unused (as in main()). We print both a "volatile" and a non-volatile
// version to show the effect.
double work_volatile(int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += (double)i;
    return s;
}

int main() {
    // Case 1: result folded into a volatile sink -> work survives.
    volatile double sink = work_volatile(100'000'000);
    std::printf("work kept: %.0f\n", sink);

    // Case 2: with the PRINT above removed, a similar loop would vanish.
    // Look at the assembly of this file to see how little is left.
    double dummy = 0.0;
    for (int i = 0; i < 100'000'000; ++i) dummy += (double)i;
    // `dummy` is never printed/used -> the whole loop may be removed.
    (void)dummy;
    return 0;
}
