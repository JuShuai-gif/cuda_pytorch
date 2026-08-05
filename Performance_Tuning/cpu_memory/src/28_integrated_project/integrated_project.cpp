// Experiment 28: Integrated project - matrix multiplication optimization chain.
//
// Runs baseline -> transposed -> blocked -> (optionally SIMD if enabled at
// build time) and reports time, GFLOPS, and checksum per stage.
// Environment and per-stage results are printed so that any claims are
// reproducible and tied to this machine.
//
// Reference: PDF 6.2.1, A.1, Table 6.2.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "benchmark.h"

static constexpr int N = 1024;
static constexpr int kRounds = 3;

static std::vector<double> A, B, T, C;

static void init() {
    A.assign((size_t)N * N, 1.0);
    B.assign((size_t)N * N, 1.0);
    T.assign((size_t)N * N, 0.0);
    C.assign((size_t)N * N, 0.0);
}

static double gflops(double ms) {
    return 2.0 * (double)N * N * N / (ms * 1e-3) / 1e9;
}

// Baseline: naive ijk, B accessed by columns.
static void baseline() {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < N; ++k)
                C[(size_t)i * N + j] += A[(size_t)i * N + k] * B[(size_t)k * N + j];
}

// Transpose B, then ijk (both sequential).
static void transposed() {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) T[(size_t)j * N + i] = B[(size_t)i * N + j];
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < N; ++k)
                C[(size_t)i * N + j] += A[(size_t)i * N + k] * T[(size_t)k * N + j];
}

// Blocked (no copy needed beyond T which is optional here): use blocked ijk
// over transposed B.
static void blocked(int SM) {
    for (int i = 0; i < N; i += SM)
        for (int j = 0; j < N; j += SM)
            for (int k = 0; k < N; k += SM)
                for (int i2 = 0; i2 < SM; ++i2)
                    for (int k2 = 0; k2 < SM; ++k2) {
                        double a = A[(size_t)(i + i2) * N + (k + k2)];
                        for (int j2 = 0; j2 < SM; ++j2)
                            C[(size_t)(i + i2) * N + (j + j2)] +=
                                a * T[(size_t)(k + k2) * N + (j + j2)];
                    }
}

static double run_stage(const char* name, void (*fn)(void)) {
    std::fill(C.begin(), C.end(), 0.0);
    fn();
    double ck = C[0];  // checksum of first element (sum of all products)
    std::fill(C.begin(), C.end(), 0.0);
    auto res = bm::time_rounds(kRounds, fn);
    std::printf("%-16s median=%-10.3f ms  %-8.3f GFLOPS  checksum=%.0f\n",
                name, res.median_ms, gflops(res.median_ms), ck);
    return res.median_ms;
}

int main() {
    std::printf("Experiment 28: matrix multiplication optimization chain\n");
    init();

    // Transpose once for the blocked/T-based stages.
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) T[(size_t)j * N + i] = B[(size_t)i * N + j];

    run_stage("baseline", baseline);
    run_stage("transposed", transposed);
    run_stage("blocked(64)", [] { blocked(64); });
    run_stage("blocked(32)", [] { blocked(32); });

    std::printf("\nNOTE: results are machine-specific (CPU, compiler, flags).\n"
                "SIMD stages require ENABLE_AVX2/AVX512 and runtime detection;\n"
                "see src/README.md. Do not compare against PDF Table 6.2\n"
                "(measured on 2007 Core 2) without recording this environment.\n");
    return 0;
}
