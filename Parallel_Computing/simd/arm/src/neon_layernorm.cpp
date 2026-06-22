/**
 * neon_layernorm.cpp -- Layer Normalization with ARM NEON
 *
 * Algorithm:
 *   mean  = sum(x) / N
 *   var   = sum((x - mean)^2) / N
 *   out   = (x - mean) / sqrt(var + eps) * gamma + beta
 *
 * NEON optimization strategies:
 *   1. 3-Pass NEON: Pass1=sum (for mean), Pass2=sum-of-squares (for var),
 *                    Pass3=normalize. Stores mean between passes.
 *   2. 2-Pass NEON: Pass1=sum+sum-of-squares combined (compute mean+var
 *                    simultaneously using Welford-style), Pass2=normalize.
 *                    Fewer memory passes but uses recomputation.
 *
 * For typical hidden dimensions (N=1024):
 *   - 3-pass is often faster because it avoids recomputing (x-mean) per element
 *   - 2-pass saves one full memory pass but requires more arithmetic in pass 1
 *
 * Simplified: gamma=1, beta=0, eps=1e-5
 */

#include <arm_neon.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

extern "C" {
#include "../../common/benchmark.h"
#include "../../common/cpu_features.h"
#include "../../common/check.h"
#include "../../common/random_data.h"
#include "../../common/aligned_buffer.h"
}

static const int   HIDDEN_DIM = 1024;     // typical transformer hidden dimension
static const float LAYERNORM_EPS = 1e-5f;
static const int   BENCH_REPEATS = 50000;  // repeat small kernel for measurable time

// ============================================================================
// Scalar baseline (3-pass, reference)
// ============================================================================

static void scalar_layernorm(float* __restrict out,
                              const float* __restrict in, int n) {
    // Pass 1: compute sum for mean
    float sum_x = 0.0f;
    for (int i = 0; i < n; i++) sum_x += in[i];
    float mean = sum_x / (float)n;

    // Pass 2: compute variance
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = in[i] - mean;
        sum_sq += d * d;
    }
    float var = sum_sq / (float)n;
    float inv_std = 1.0f / sqrtf(var + LAYERNORM_EPS);

    // Pass 3: normalize (gamma=1, beta=0)
    for (int i = 0; i < n; i++) {
        out[i] = (in[i] - mean) * inv_std;
    }
}

// ============================================================================
// NEON 3-Pass: separate sum, sum-of-squares, normalize passes
// ============================================================================
//
// Three full memory passes over the data. Stores mean and inv_std as scalars
// between passes. Each pass is vectorized with vaddvq_f32 reductions.
// This keeps each pass simple and maximizes throughput within each pass.

static void neon_layernorm_3pass(float* __restrict out,
                                  const float* __restrict in, int n) {
    float32x4_t zero = vdupq_n_f32(0.0f);

    // Pass 1: compute sum -> mean
    float32x4_t vsum = zero;
    int i = 0;
    for (; i + 3 < n; i += 4) {
        vsum = vaddq_f32(vsum, vld1q_f32(&in[i]));
    }
    float sum_x = vaddvq_f32(vsum);
    for (; i < n; i++) sum_x += in[i];
    float mean = sum_x / (float)n;

    // Pass 2: compute sum of squared differences -> variance
    float32x4_t vmean = vdupq_n_f32(mean);
    float32x4_t vsq = zero;
    i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t d = vsubq_f32(vld1q_f32(&in[i]), vmean);
        vsq = vmlaq_f32(vsq, d, d);  // vsq += d * d
    }
    float sum_sq = vaddvq_f32(vsq);
    for (; i < n; i++) {
        float d = in[i] - mean;
        sum_sq += d * d;
    }
    float var = sum_sq / (float)n;
    float inv_std = 1.0f / sqrtf(var + LAYERNORM_EPS);

    // Pass 3: normalize
    float32x4_t vinv = vdupq_n_f32(inv_std);
    i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t d = vsubq_f32(vld1q_f32(&in[i]), vmean);
        vst1q_f32(&out[i], vmulq_f32(d, vinv));
    }
    for (; i < n; i++) {
        out[i] = (in[i] - mean) * inv_std;
    }
}

// ============================================================================
// NEON 2-Pass: compute sum and sum-of-squares in one pass
// ============================================================================
//
// Uses Welford-style approach: accumulate sum and sum_of_squares in one pass.
// Then compute variance: var = (sum_sq - sum^2/N) / N
// Then another pass for normalization. Total: 2 passes over data.
//
// Trade-off: Saves 1 memory pass vs 3-pass, but has numeric precision
// concern with the (sum_sq - sum^2/N) formula (catastrophic cancellation
// when variance is small relative to mean).

static void neon_layernorm_2pass(float* __restrict out,
                                  const float* __restrict in, int n) {
    float32x4_t zero = vdupq_n_f32(0.0f);

    // Pass 1: accumulate sum and sum-of-squares simultaneously
    float32x4_t vsum = zero;
    float32x4_t vsq  = zero;
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(&in[i]);
        vsum = vaddq_f32(vsum, v);
        vsq  = vmlaq_f32(vsq, v, v);  // vsq += v * v
    }
    float sum_x = vaddvq_f32(vsum);
    float sum_sq_raw = vaddvq_f32(vsq);
    for (; i < n; i++) {
        sum_x += in[i];
        sum_sq_raw += in[i] * in[i];
    }
    float mean = sum_x / (float)n;
    // var = E[x^2] - E[x]^2 = sum_sq_raw/N - mean^2
    float var = sum_sq_raw / (float)n - mean * mean;
    if (var < 0.0f) var = 0.0f;  // guard against numeric issues
    float inv_std = 1.0f / sqrtf(var + LAYERNORM_EPS);

    // Pass 2: normalize
    float32x4_t vmean = vdupq_n_f32(mean);
    float32x4_t vinv  = vdupq_n_f32(inv_std);
    i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t d = vsubq_f32(vld1q_f32(&in[i]), vmean);
        vst1q_f32(&out[i], vmulq_f32(d, vinv));
    }
    for (; i < n; i++) {
        out[i] = (in[i] - mean) * inv_std;
    }
}

// ============================================================================
// Benchmark wrappers
// ============================================================================

static float* g_ln_in  = nullptr;
static float* g_ln_out = nullptr;
static int    g_ln_n   = 0;

__attribute__((noinline))
static void bench_scalar() {
    for (int r = 0; r < BENCH_REPEATS; r++)
        scalar_layernorm(g_ln_out, g_ln_in, g_ln_n);
}

__attribute__((noinline))
static void bench_neon_3pass() {
    for (int r = 0; r < BENCH_REPEATS; r++)
        neon_layernorm_3pass(g_ln_out, g_ln_in, g_ln_n);
}

__attribute__((noinline))
static void bench_neon_2pass() {
    for (int r = 0; r < BENCH_REPEATS; r++)
        neon_layernorm_2pass(g_ln_out, g_ln_in, g_ln_n);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    cpu_print_features();

    int n = HIDDEN_DIM;
    g_ln_n = n;
    g_ln_in  = ALIGNED_ALLOC(float, n, 64);
    g_ln_out = ALIGNED_ALLOC(float, n, 64);

    fill_random_f32(g_ln_in, n);

    // --- Correctness ---
    printf("\n=== Correctness Checks (N=%d) ===\n", n);
    float* ref = ALIGNED_ALLOC(float, n, 64);
    scalar_layernorm(ref, g_ln_in, n);

    float* test3 = ALIGNED_ALLOC(float, n, 64);
    neon_layernorm_3pass(test3, g_ln_in, n);
    CHECK_NEAR_ARRAY(test3, ref, n, 1e-4f, "NEON 3-pass LayerNorm vs scalar");

    float* test2 = ALIGNED_ALLOC(float, n, 64);
    neon_layernorm_2pass(test2, g_ln_in, n);
    // 2-pass has slightly lower precision due to cancellation; relax tolerance
    CHECK_NEAR_ARRAY(test2, ref, n, 5e-4f, "NEON 2-pass LayerNorm vs scalar");

    // Print actual computed statistics for inspection
    {
        float mean = 0, var = 0;
        for (int i = 0; i < n; i++) mean += g_ln_in[i];
        mean /= n;
        for (int i = 0; i < n; i++) { float d = g_ln_in[i] - mean; var += d*d; }
        var /= n;
        printf("\n  Input statistics: mean=%.6f, var=%.6f, std=%.6f\n",
               mean, var, sqrtf(var));
    }

    // --- Benchmark ---
    printf("\n=== Benchmark: LayerNorm (N=%d, repeated %dx) ===\n",
           n, BENCH_REPEATS);
    benchmark_result_t results[3];

    size_t nelem_total = (size_t)n * BENCH_REPEATS;
    size_t bytes_total = nelem_total * 2 * sizeof(float); // read in, write out

    BENCH_COMPUTE(bench_scalar(), nelem_total, bytes_total, 30, results[0]);
    results[0].name = "scalar (3-pass)";

    BENCH_COMPUTE(bench_neon_3pass(), nelem_total, bytes_total, 30,
                  results[1]);
    results[1].name = "NEON 3-pass";

    BENCH_COMPUTE(bench_neon_2pass(), nelem_total, bytes_total, 30,
                  results[2]);
    results[2].name = "NEON 2-pass";

    bench_report(results, 3);

    // --- Analysis ---
    printf("=== Analysis ===\n");
    printf("  3-pass: 3 memory passes. Lower IPC due to memory stalls,\n");
    printf("          but simple vectorized reductions.\n");
    printf("  2-pass: 2 memory passes. Saves one pass but uses\n");
    printf("          E[x^2]-E[x]^2 formula which may lose precision.\n");
    printf("  For N=1024 (fits in L1 cache), 2-pass is typically faster\n");
    printf("  since memory bandwidth is not the bottleneck.\n");

    ALIGNED_FREE(ref);
    ALIGNED_FREE(test3);
    ALIGNED_FREE(test2);
    ALIGNED_FREE(g_ln_in);
    ALIGNED_FREE(g_ln_out);
    return 0;
}
