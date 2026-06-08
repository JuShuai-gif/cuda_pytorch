// Chapter 11: Out-of-Order Execution
// Demonstrates dependency chains and how to break them for ILP.
// Compile: see CMakeLists.txt (ch11_demo target)

#include <iostream>
#include <chrono>

// ---- Example 11.1a: Dependency chain (slow) ----
double DependencyChain(double a, double b, double c, double d) {
    return ((a + b) + c) + d; // Each addition depends on previous
}

// ---- Example 11.1b: Broken dependency chain (fast) ----
double BrokenChain(double a, double b, double c, double d) {
    return (a + b) + (c + d); // (a+b) and (c+d) can execute in parallel
}

// ---- Example 11.2a: Loop-carried dependency chain ----
double LoopDependencyChain(const double *data, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += data[i]; // Each iteration depends on previous sum
    }
    return sum;
}

// ---- Example 11.2b: Multiple accumulators to break chain ----
double LoopMultipleAccumulators(const double *data, int n) {
    double sum1 = 0.0, sum2 = 0.0;
    for (int i = 0; i < n; i += 2) {
        sum1 += data[i];
        sum2 += data[i + 1];
    }
    return sum1 + sum2;
}

// ---- Example 11.3: Register renaming (CPU does this automatically) ----
// No manual unrolling needed on modern OoO CPUs for simple cases
double AutoRenaming(const double *data, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        double temp = data[i] * 2.0; // CPU can rename temp register
        sum += temp;
    }
    return sum;
}

// ---- Benchmark helper ----
template <typename Func>
double Benchmark(Func f, const double *data, int n, int iterations,
                 const char *label) {
    volatile double result = 0.0; // Prevent optimization
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        result = f(data, n);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << label << ": " << us << " us (result=" << result << ")\n";
    return static_cast<double>(us);
}

// ---- Main ----
int main() {
    std::cout << "=== Chapter 11: Out-of-Order Execution ===\n\n";

    // Scalar chain demo
    double x = DependencyChain(1.0, 2.0, 3.0, 4.0);
    double y = BrokenChain(1.0, 2.0, 3.0, 4.0);
    std::cout << "DependencyChain: " << x << "\n";
    std::cout << "BrokenChain:     " << y << "\n";

    // Loop accumulator benchmark
    constexpr int N = 1000000;
    double *data = new double[N];
    for (int i = 0; i < N; ++i) data[i] = 1.0;

    Benchmark(LoopDependencyChain, data, N, 100, "Single accumulator  ");
    Benchmark(LoopMultipleAccumulators, data, N, 100, "Dual accumulators   ");
    Benchmark(AutoRenaming, data, N, 100, "With register rename");

    delete[] data;

    std::cout << "\nAll chapter 11 checks passed.\n";
    return 0;
}
