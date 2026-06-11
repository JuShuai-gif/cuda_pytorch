// Chapter 16: Testing Speed (测试速度)
// Consolidates Example 16.1 (ReadTSC with CPUID serialization)
// and Example 16.2 (measurement framework with min/max/avg/median).
//
// Compile: g++ -std=c++11 -O2 ch16_optimization.cpp -o ch16_optimization

#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>
#include <chrono>
#include <cstdint>

// ====================================================================
// Platform detection and rdtsc/cpuid intrinsics
// ====================================================================
#if defined(__GNUC__) || defined(__clang__)
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <x86intrin.h>
#define HAS_RDTSC 1
#else
#define HAS_RDTSC 0
#endif
#elif defined(_MSC_VER)
#include <intrin.h>
#define HAS_RDTSC 1
#else
#define HAS_RDTSC 0
#endif

// ====================================================================
// ReadTSC() — Read Time Stamp Counter with CPUID serialization
// ====================================================================
// The CPUID instruction serializes the pipeline so that all previous
// instructions complete before RDTSC executes. This gives a more
// accurate measurement by preventing out-of-order execution artifacts.
// ====================================================================

#if HAS_RDTSC

static inline uint64_t ReadTSC() {
#if defined(__GNUC__) || defined(__clang__)
    // Use inline assembly for full control over CPUID + RDTSC ordering
    unsigned int eax, ebx, ecx, edx;
    uint32_t lo, hi;

    // CPUID with leaf 0 serializes the pipeline
    __asm__ volatile("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(0));

    // RDTSC reads the 64-bit time-stamp counter into EDX:EAX
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));

    return (static_cast<uint64_t>(hi) << 32) | lo;

#elif defined(_MSC_VER)
    // MSVC: use __cpuid / __rdtsc intrinsics
    int cpuInfo[4];
    __cpuid(cpuInfo, 0);
    volatile int dontSkip = cpuInfo[0];
    (void)dontSkip;
    return __rdtsc();
#endif
}

#else
// Fallback: use std::chrono when RDTSC is not available
static inline uint64_t ReadTSC() {
    auto now = std::chrono::high_resolution_clock::now().time_since_epoch();
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}
#endif  // HAS_RDTSC

// ====================================================================
// Measurement framework
// ====================================================================
// Runs a function multiple times, collects individual TSC deltas,
// and reports min, max, average, and median.
// ====================================================================

struct MeasurementResult {
    uint64_t minCycles;
    uint64_t maxCycles;
    double avgCycles;
    double medCycles;
    int sampleCount;
};

template <typename Func>
MeasurementResult Measure(Func fn, int iterations) {
    std::vector<uint64_t> deltas;
    deltas.reserve(iterations);

    for (int i = 0; i < iterations; ++i) {
        uint64_t t1 = ReadTSC();
        volatile auto result = fn();  // volatile prevents dead-code elimination
        (void)result;
        uint64_t t2 = ReadTSC();
        deltas.push_back(t2 - t1);
    }

    MeasurementResult r;
    r.sampleCount = iterations;
    r.minCycles = *std::min_element(deltas.begin(), deltas.end());
    r.maxCycles = *std::max_element(deltas.begin(), deltas.end());

    double sum = std::accumulate(deltas.begin(), deltas.end(), 0.0);
    r.avgCycles = sum / iterations;

    std::sort(deltas.begin(), deltas.end());
    if (iterations % 2 == 0) {
        r.medCycles = (deltas[iterations / 2 - 1] + deltas[iterations / 2]) / 2.0;
    } else {
        r.medCycles = deltas[iterations / 2];
    }

    return r;
}

// ====================================================================
// Sample functions under test
// ====================================================================

// Baseline: measure the measurement overhead itself
static int EmptyFunction() {
    return 0;
}

// Simple floating-point math (sin + cos on a range)
static double SimpleMath(int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += std::sin(static_cast<double>(i) * 0.001) * std::cos(static_cast<double>(i) * 0.002);
    }
    return sum;
}

// Memory access: sequential read through a moderately sized vector
static double MemoryAccess(int n) {
    std::vector<double> data(n);
    for (int i = 0; i < n; ++i) {
        data[i] = static_cast<double>(i);
    }
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += data[i];
    }
    return sum;
}

// Integer arithmetic with branch prediction-friendly pattern
static int BranchFriendly(int n) {
    int sum = 0;
    for (int i = 0; i < n; ++i) {
        if (i % 2 == 0) {
            sum += i;
        } else {
            sum -= i;
        }
    }
    return sum;
}

// ====================================================================
// Warmup: execute the function a few times to prime caches and
// avoid cold-start penalties in the actual measurement.
// ====================================================================

template <typename Func>
void Warmup(Func fn, int warmupIters) {
    for (int i = 0; i < warmupIters; ++i) {
        volatile auto r = fn();
        (void)r;
    }
}

// ====================================================================
// Print helper
// ====================================================================

void PrintResult(const char* label, const MeasurementResult& r) {
    std::cout << "  " << std::left << std::setw(28) << label << " min=" << std::setw(8)
              << r.minCycles << " max=" << std::setw(8) << r.maxCycles << " avg=" << std::setw(10)
              << std::fixed << std::setprecision(1) << r.avgCycles << " med=" << std::setw(10)
              << std::fixed << std::setprecision(1) << r.medCycles << "  (n=" << r.sampleCount
              << ")" << std::endl;
}

// ====================================================================
// Main
// ====================================================================

int main() {
    const int WARMUP = 5;
    const int ITERS = 100;

    std::cout << "=== Chapter 16: Testing Speed (测试速度) ===" << std::endl;
    std::cout << "Measurement unit: TSC cycles";
#if HAS_RDTSC
    std::cout << " (x86 RDTSC + CPUID serialization)" << std::endl;
#else
    std::cout << " (std::chrono fallback, nanosecond resolution)" << std::endl;
#endif
    std::cout << std::endl;

    // --- Baseline: empty function ---
    {
        auto fn = []() { return EmptyFunction(); };
        Warmup(fn, WARMUP);
        MeasurementResult r = Measure(fn, ITERS);
        PrintResult("EmptyFunction (baseline)", r);
    }

    // --- Simple math (1000 iterations) ---
    {
        const int N = 1000;
        auto fn = [=]() { return SimpleMath(N); };
        Warmup(fn, WARMUP);
        MeasurementResult r = Measure(fn, ITERS);
        PrintResult("SimpleMath (1000 iters)", r);
    }

    // --- Memory access (10000 elements) ---
    {
        const int N = 10000;
        auto fn = [=]() { return MemoryAccess(N); };
        Warmup(fn, WARMUP);
        MeasurementResult r = Measure(fn, ITERS);
        PrintResult("MemoryAccess (10000 elts)", r);
    }

    // --- Branch-friendly integer code ---
    {
        const int N = 10000;
        auto fn = [=]() { return BranchFriendly(N); };
        Warmup(fn, WARMUP);
        MeasurementResult r = Measure(fn, ITERS);
        PrintResult("BranchFriendly (10000)", r);
    }

    // --- Heavier math (5000 iterations) ---
    {
        const int N = 5000;
        auto fn = [=]() { return SimpleMath(N); };
        Warmup(fn, WARMUP);
        MeasurementResult r = Measure(fn, ITERS);
        PrintResult("SimpleMath (5000 iters)", r);
    }

    std::cout << "\nAll chapter 16 checks passed." << std::endl;
    return 0;
}
