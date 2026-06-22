// Chapter 15: Metaprogramming
// Demonstrates template metaprogramming for compile-time computation.
// Compile: see CMakeLists.txt (ch15_demo target)

#include <iostream>
#include <chrono>

// ---- Example 15.1a: Runtime pow with loop ----
double PowRuntime(double x, unsigned int n) {
    double y = 1.0;
    for (unsigned int i = 0; i < n; ++i) {
        y *= x;
    }
    return y;
}

// ---- Example 15.1b: Optimized runtime pow (binary exponentiation) ----
double PowBinary(double x, unsigned int n) {
    double y = 1.0;
    while (n) {
        if (n & 1) y *= x;
        x *= x;
        n >>= 1;
    }
    return y;
}

// ---- Example 15.1d: Compile-time pow via template metaprogramming ----
template <unsigned int N>
struct PowN {
    static double value(double x) {
        return (N & 1 ? x : 1.0) * PowN<(N >> 1)>::value(x * x);
    }
};
template <>
struct PowN<0> {
    static double value(double) {
        return 1.0;
    }
};

// ---- Compile-time factorial ----
template <unsigned int N>
struct Factorial {
    static constexpr unsigned long long value = N * Factorial<N - 1>::value;
};
template <>
struct Factorial<0> {
    static constexpr unsigned long long value = 1;
};

// ---- Compile-time Fibonacci ----
template <unsigned int N>
struct Fibonacci {
    static constexpr unsigned long long value =
        Fibonacci<N - 1>::value + Fibonacci<N - 2>::value;
};
template <>
struct Fibonacci<0> {
    static constexpr unsigned long long value = 0;
};
template <>
struct Fibonacci<1> {
    static constexpr unsigned long long value = 1;
};

// ---- constexpr alternative (C++14+) ----
constexpr double PowConstexpr(double x, unsigned int n) {
    double y = 1.0;
    for (unsigned int i = 0; i < n; ++i) {
        y *= x;
    }
    return y;
}

constexpr unsigned long long FactorialConstexpr(unsigned int n) {
    unsigned long long result = 1;
    for (unsigned int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

// ---- Main ----
int main() {
    std::cout << "=== Chapter 15: Metaprogramming ===\n\n";

    double x = 2.0;
    constexpr unsigned int N = 10;

    std::cout << "Pow(" << x << ", " << N << "):\n";
    std::cout << "  Runtime loop:         " << PowRuntime(x, N) << "\n";
    std::cout << "  Binary exponentiation: " << PowBinary(x, N) << "\n";
    std::cout << "  Template meta:         " << PowN<N>::value(x) << "\n";
    std::cout << "  constexpr:             " << PowConstexpr(x, N) << "\n";

    std::cout << "\nCompile-time values:\n";
    std::cout << "  Factorial<10>::value  = " << Factorial<10>::value << "\n";
    std::cout << "  FactorialConstexpr(10) = " << FactorialConstexpr(10) << "\n";
    std::cout << "  Fibonacci<20>::value   = " << Fibonacci<20>::value << "\n";

    // constexpr can be used in compile-time contexts
    constexpr double compile_time_pow = PowConstexpr(3.0, 4);
    constexpr unsigned long long compile_time_fact = FactorialConstexpr(8);
    std::cout << "\n  constexpr PowConstexpr(3, 4)  = " << compile_time_pow << "\n";
    std::cout << "  constexpr FactorialConstexpr(8) = " << compile_time_fact << "\n";

    // Benchmark: runtime vs compile-time
    constexpr int ITERS = 1000000;
    volatile double result = 0.0;

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; ++i) {
        result = PowRuntime(2.0, 10);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; ++i) {
        result = PowBinary(2.0, 10);
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; ++i) {
        result = PowN<10>::value(2.0); // Template is fully unrolled at compile time
    }
    auto t4 = std::chrono::high_resolution_clock::now();

    using namespace std::chrono;
    auto loop_us = duration_cast<microseconds>(t2 - t1).count();
    auto binary_us = duration_cast<microseconds>(t3 - t2).count();
    auto meta_us = duration_cast<microseconds>(t4 - t3).count();

    std::cout << "\nBenchmark (1M iterations, pow(2.0, 10)):\n";
    std::cout << "  Loop method:   " << loop_us << " us\n";
    std::cout << "  Binary method: " << binary_us << " us\n";
    std::cout << "  Template meta: " << meta_us << " us\n";

    std::cout << "\nAll chapter 15 checks passed.\n";
    return 0;
}
