/**
 * Naive Scaled Dot-Product Attention in C++.
 *
 * Purpose:
 * - Serve as a CPU baseline for performance comparison
 * - Demonstrate the O(N^2) memory and compute pattern
 * - Provide a reference for correctness verification
 *
 * This is a single-threaded, unoptimized implementation.
 * We deliberately materialize the full NxN attention matrix
 * to show why GPU kernel fusion is necessary.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <random>
#include <vector>

using Float = float;

// ----------------------------------------------------------------------
// Helper: Row-wise Softmax
// ----------------------------------------------------------------------
static void softmax_row(Float *row, int N) {
    // Find max for numerical stability
    Float max_val = row[0];
    for (int i = 1; i < N; ++i)
        if (row[i] > max_val) max_val = row[i];

    // exp and sum
    Float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        row[i] = std::exp(row[i] - max_val);
        sum += row[i];
    }

    // Normalize
    Float inv_sum = 1.0f / sum;
    for (int i = 0; i < N; ++i)
        row[i] *= inv_sum;
}

// ----------------------------------------------------------------------
// Naive Attention
//
// Q: [N, d_k]  (row-major)
// K: [N, d_k]  (row-major)
// V: [N, d_v]  (row-major)
// O: [N, d_v]  (output, row-major)
// ----------------------------------------------------------------------
static void naive_attention(
    const Float *Q, const Float *K, const Float *V,
    Float *O,
    int N, int d_k, int d_v) {
    // S = Q @ K^T  (N x N)
    std::vector<Float> S(N * N);
    Float scale = 1.0f / std::sqrt(static_cast<Float>(d_k));

    // Step 1: S[i][j] = (1/sqrt(d_k)) * sum_{m} Q[i][m] * K[j][m]
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            Float dot = 0.0f;
            for (int m = 0; m < d_k; ++m) {
                dot += Q[i * d_k + m] * K[j * d_k + m];
            }
            S[i * N + j] = dot * scale;
        }
    }

    // Step 2: P = softmax(S) applied row-wise
    std::vector<Float> P = S; // copy (in practice we'd do in-place)
    for (int i = 0; i < N; ++i)
        softmax_row(P.data() + i * N, N);

    // Step 3: O = P @ V  (N x d_v)
    std::fill(O, O + N * d_v, 0.0f);
    for (int i = 0; i < N; ++i) {
        for (int m = 0; m < d_v; ++m) {
            Float accum = 0.0f;
            for (int j = 0; j < N; ++j) {
                accum += P[i * N + j] * V[j * d_v + m];
            }
            O[i * d_v] = accum;
        }
    }
}

// ----------------------------------------------------------------------
// Benchmark helper
// ----------------------------------------------------------------------
struct BenchResult {
    double qk_ms;
    double softmax_ms;
    double pv_ms;
    double total_ms;
};

static BenchResult benchmark(const Float *Q, const Float *K, const Float *V,
                             Float *O, int N, int d_k, int d_v, int warmup, int iters) {
    // Warmup
    for (int w = 0; w < warmup; ++w)
        naive_attention(Q, K, V, O, N, d_k, d_v);

    // Timed runs — measure total
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i)
        naive_attention(Q, K, V, O, N, d_k, d_v);
    auto t1 = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

    // For step breakdown we approximate: QK^T dominates
    double flops_qk = 2.0 * N * N * d_k; // multiply-add
    double flops_softmax = 4.0 * N * N;  // max + exp + sum + div
    double flops_pv = 2.0 * N * N * d_v; // multiply-add
    double total_flops = flops_qk + flops_softmax + flops_pv;

    // Proportionally split time
    double qk_frac = flops_qk / total_flops;
    double sm_frac = flops_softmax / total_flops;
    double pv_frac = flops_pv / total_flops;

    return {
        total_ms * qk_frac,
        total_ms * sm_frac,
        total_ms * pv_frac,
        total_ms};
}

// ----------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------
int main() {
    std::mt19937 rng(42);
    std::normal_distribution<Float> dist(0.0f, 1.0f);

    std::cout << "Naive Attention (C++ CPU, single-threaded)\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << std::left
              << std::setw(8) << "seq_len"
              << std::setw(12) << "S_mem(KB)"
              << std::setw(10) << "QK^T(ms)"
              << std::setw(12) << "Softmax(ms)"
              << std::setw(10) << "PV(ms)"
              << std::setw(12) << "Total(ms)"
              << std::setw(12) << "FLOP/s" << "\n";
    std::cout << std::string(80, '-') << "\n";

    for (int N : {64, 128, 256, 512, 1024, 2048, 4096}) {
        int d_k = 64;
        int d_v = 64;

        // Allocate
        std::vector<Float> Q(N * d_k), K(N * d_k), V(N * d_v), O(N * d_v);
        for (auto &x : Q) x = dist(rng);
        for (auto &x : K) x = dist(rng);
        for (auto &x : V) x = dist(rng);

        // Benchmark
        int warmup = N <= 512 ? 10 : 2;
        int iters = N <= 512 ? 50 : 10;
        auto res = benchmark(Q.data(), K.data(), V.data(), O.data(),
                             N, d_k, d_v, warmup, iters);

        // FLOPs
        double total_flops = 2.0 * N * N * d_k + 4.0 * N * N + 2.0 * N * N * d_v;
        double gflops = total_flops / (res.total_ms * 1e6);

        double mem_kb = (2.0 * N * N * sizeof(Float)) / 1024.0; // S + P

        std::cout << std::left
                  << std::setw(8) << N
                  << std::setw(12) << std::fixed << std::setprecision(0) << mem_kb
                  << std::setw(10) << std::fixed << std::setprecision(2) << res.qk_ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << res.softmax_ms
                  << std::setw(10) << std::fixed << std::setprecision(2) << res.pv_ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << res.total_ms
                  << std::setw(12) << std::fixed << std::setprecision(1) << gflops << "\n";

        // Too slow beyond 4096 on CPU
        if (N >= 4096) break;
    }

    std::cout << std::string(80, '=') << "\n";
    std::cout << "\nSummary:\n";
    std::cout << "  Time complexity:  O(N^2)\n";
    std::cout << "  Memory complexity: O(N^2) for the NxN attention matrix\n";
    std::cout << "  This is why we need GPU + Kernel Fusion (FlashAttention).\n";

    return 0;
}
