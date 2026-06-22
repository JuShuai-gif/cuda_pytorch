#include <iostream>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cstdlib>

using namespace std;
using namespace std::chrono;

const int ITERATIONS = 50000000;

// --- Timing helper ---
template <typename F>
double measure_time(const char* label, int iterations, F fn) {
    // Volatile input prevents compile-time constant folding
    volatile double x = 1.234;
    volatile double sink;
    (void)sink;

    auto start = high_resolution_clock::now();
    for (int k = 0; k < iterations; ++k) {
        sink = fn(x);
    }
    auto end = high_resolution_clock::now();
    double elapsed = duration_cast<nanoseconds>(end - start).count() / 1e6 / iterations;
    cout << "  " << left << setw(50) << label << ": " << fixed << setprecision(6) << elapsed
         << " ms (avg over " << iterations << " runs)" << endl;
    return elapsed;
}

// ============================================================
// Example 15.1a: Direct pow(x,10) call using std::pow
// ============================================================
__attribute__((noinline)) double xpow10_a(double x) {
    return pow(x, 10.0);
}

// ============================================================
// Example 15.1b: Integer power using binary exponentiation loop
// ============================================================
__attribute__((noinline)) double ipow(double x, unsigned int n) {
    double y = 1.0;
    while (n != 0) {
        if (n & 1)
            y *= x;
        x *= x;
        n >>= 1;
    }
    return y;
}

__attribute__((noinline)) double xpow10_b(double x) {
    return ipow(x, 10);
}

// ============================================================
// Example 15.1c: Manually unrolled x^10 calculation
// ============================================================
__attribute__((noinline)) double xpow10_c(double x) {
    double x2 = x * x;     // x^2
    double x4 = x2 * x2;   // x^4
    double x8 = x4 * x4;   // x^8
    double x10 = x8 * x2;  // x^10
    return x10;
}

// ============================================================
// Example 15.1d: Template metaprogramming for compile-time
//               integer power
// ============================================================

// General case: N is NOT a power of 2.
// Split N into N1 (rightmost 1-bit) and N-N1 (power of 2 portion).
template <bool IsPowerOf2, int N>
class powN {
public:
    static double p(double x) {
#define N1 (N & (N - 1))
        return powN<(N1 & (N1 - 1)) == 0, N1>::p(x) * powN<true, N - N1>::p(x);
#undef N1
    }
};

// Partial specialization: N is a power of 2.
// Recursively halve the exponent by squaring.
template <int N>
class powN<true, N> {
public:
    static double p(double x) { return powN<true, N / 2>::p(x) * powN<true, N / 2>::p(x); }
};

// Full specialization: base case N = 1.
template <>
class powN<true, 1> {
public:
    static double p(double x) { return x; }
};

// Full specialization: base case N = 0 (safety stop).
template <>
class powN<true, 0> {
public:
    static double p(double x) { return 1.0; }
};

// User-facing function template.
template <int N>
static inline double IntegerPower(double x) {
    return powN<(N & (N - 1)) == 0, N>::p(x);
}

__attribute__((noinline)) double xpow10_d(double x) {
    return IntegerPower<10>(x);
}

// ============================================================
// Correctness verification
// ============================================================
void verify_correctness() {
    const double test_vals[] = {0.0, 1.0, 2.0, 3.0, 5.0, -1.0, -2.0, 1.234, 0.5};
    const int n_tests = sizeof(test_vals) / sizeof(test_vals[0]);
    bool all_pass = true;

    cout << "\nCorrectness check (pow(x,10) via std::pow as reference):" << endl;
    for (int i = 0; i < n_tests; ++i) {
        double x = test_vals[i];
        double ref = pow(x, 10.0);
        double r_a = xpow10_a(x);
        double r_b = xpow10_b(x);
        double r_c = xpow10_c(x);
        double r_d = xpow10_d(x);

        double eps = 1e-12;
        bool ok_a = abs(r_a - ref) < eps * max(1.0, abs(ref));
        bool ok_b = abs(r_b - ref) < eps * max(1.0, abs(ref));
        bool ok_c = abs(r_c - ref) < eps * max(1.0, abs(ref));
        bool ok_d = abs(r_d - ref) < eps * max(1.0, abs(ref));

        if (!ok_a || !ok_b || !ok_c || !ok_d)
            all_pass = false;

        cout << "  x=" << setw(6) << x << "  a:" << (ok_a ? "PASS" : "FAIL")
             << "  b:" << (ok_b ? "PASS" : "FAIL") << "  c:" << (ok_c ? "PASS" : "FAIL")
             << "  d:" << (ok_d ? "PASS" : "FAIL") << endl;
    }
    cout << (all_pass ? "  All PASS" : "  Some FAILED") << endl;
}

// ============================================================
// Main
// ============================================================
int main() {
    cout << "============================================" << endl;
    cout << "  Ch15 Optimization: x^10 Benchmark" << endl;
    cout << "  Comparing 4 approaches (metaprogramming)" << endl;
    cout << "  Iterations per method: " << ITERATIONS << endl;
    cout << "============================================" << endl;

    verify_correctness();

    cout << "\nPerformance benchmark:" << endl;

    measure_time("15.1a: Direct std::pow(x,10)", ITERATIONS, [](double x) { return xpow10_a(x); });

    measure_time("15.1b: Binary exponentiation loop", ITERATIONS,
                 [](double x) { return xpow10_b(x); });

    measure_time("15.1c: Manually unrolled", ITERATIONS, [](double x) { return xpow10_c(x); });

    measure_time("15.1d: Template metaprogramming", ITERATIONS,
                 [](double x) { return xpow10_d(x); });

    cout << "\nDone." << endl;
    return 0;
}
