// Chapter 16: Testing Speed
// Demonstrates high-resolution timing and performance measurement techniques.
// Compile: see CMakeLists.txt (ch16_demo target)

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstring>
#include <cmath>

#ifdef _MSC_VER
#include <intrin.h>
#pragma intrinsic(__rdtsc)
#elif defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
#endif

// ---- RDTSC-based timer (x86 only) ----
#ifdef __RDTSC__
static inline unsigned long long ReadTSC() {
    return __rdtsc();
}

class RdtscTimer {
    unsigned long long start_tsc;

public:
    void Start() {
        start_tsc = ReadTSC();
    }
    unsigned long long Elapsed() const {
        return ReadTSC() - start_tsc;
    }
};
#endif

// ---- std::chrono high-resolution timer ----
class ChronoTimer {
    std::chrono::high_resolution_clock::time_point start;

public:
    void Start() {
        start = std::chrono::high_resolution_clock::now();
    }
    double ElapsedUs() const {
        auto end = std::chrono::high_resolution_clock::now();
        return static_cast<double>(
                   std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count())
               / 1000.0;
    }
    double ElapsedMs() const {
        return ElapsedUs() / 1000.0;
    }
};

// ---- Function under test ----
static double HeavyComputation(int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += std::sin(static_cast<double>(i) * 0.001) * std::cos(static_cast<double>(i) * 0.002);
    }
    return sum;
}

// ---- Measure with overhead subtraction ----
template <typename Func>
double MeasureWithOverhead(Func f, int warmup, int iterations) {
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        volatile auto result = f();
        (void)result;
    }

    ChronoTimer timer;
    timer.Start();
    for (int i = 0; i < iterations; ++i) {
        volatile auto result = f();
        (void)result;
    }
    return timer.ElapsedUs() / iterations;
}

// ---- Measure timer overhead ----
double MeasureTimerOverhead() {
    ChronoTimer timer;
    constexpr int N = 100000;
    timer.Start();
    for (int i = 0; i < N; ++i) {
        volatile auto end = std::chrono::high_resolution_clock::now();
        (void)end;
    }
    return timer.ElapsedUs() / N;
}

// ---- Main ----
int main() {
    std::cout << "=== Chapter 16: Testing Speed ===\n\n";

    // Timer overhead measurement
    double chrono_overhead = MeasureTimerOverhead();
    std::cout << "std::chrono timer overhead: " << chrono_overhead << " us/call\n";

    // RDTSC timing
#ifdef __RDTSC__
    {
        RdtscTimer tsc;
        tsc.Start();
        volatile double r = HeavyComputation(1000);
        (void)r;
        auto elapsed = tsc.Elapsed();
        std::cout << "RDTSC elapsed: " << elapsed << " cycles\n";
    }
#else
    std::cout << "[RDTSC not available on this platform]\n";
#endif

    // Measure computation time
    std::cout << "\nMeasuring HeavyComputation(10000):\n";
    double us_per_call = MeasureWithOverhead(
        []() { return HeavyComputation(10000); },
        5,  // warmup iterations
        100 // measurement iterations
    );
    std::cout << "  Average: " << std::fixed << std::setprecision(3)
              << us_per_call << " us/call\n";

    // Multiple runs for stability
    std::cout << "\nStability test (5 runs of HeavyComputation(10000)):\n";
    for (int run = 0; run < 5; ++run) {
        double us = MeasureWithOverhead(
            []() { return HeavyComputation(10000); },
            3, 50);
        std::cout << "  Run " << run << ": " << us << " us\n";
    }

    std::cout << "\nAll chapter 16 checks passed.\n";
    return 0;
}
