// Experiment 10: Cache blocking.
//
// Matrix multiplication with a transposed B matrix, sweeping the block
// size (8/16/32/64/128). Measures GFLOPS and cache behavior vs block size.
//
// Reference: PDF 6.2.1 (sub-matrix optimization), Table 6.2.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "benchmark.h"

static constexpr int N = 1024;
static constexpr int kRounds = 3;

int main() {
    std::printf("Experiment 10: cache blocking (%dx%d matrix)\n", N, N);
    std::vector<double> A((size_t)N * N, 1.0);
    std::vector<double> B((size_t)N * N, 1.0);
    std::vector<double> T((size_t)N * N, 0.0);
    std::vector<double> C((size_t)N * N, 0.0);

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) T[(size_t)j * N + i] = B[(size_t)i * N + j];

    auto gemm = [&](int SM) {
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
        bm::compiler_barrier();
    };

    std::printf("%-10s %-12s %-12s %-14s\n", "block", "time_ms", "GFLOPS",
                "checksum");
    for (int sm : {8, 16, 32, 64, 128}) {
        std::fill(C.begin(), C.end(), 0.0);
        gemm(sm);  // warmup
        std::fill(C.begin(), C.end(), 0.0);
        auto res = bm::time_rounds(kRounds, [&] { gemm(sm); });
        double flops = 2.0 * (double)N * N * N;
        double gflops = flops / (res.median_ms * 1e-3) / 1e9;
        std::printf("%-10d %-12.3f %-12.3f %-14.0f\n", sm, res.median_ms,
                    gflops, C[0]);
    }
    return 0;
}
