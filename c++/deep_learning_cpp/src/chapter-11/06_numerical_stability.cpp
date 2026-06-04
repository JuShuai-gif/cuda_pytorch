/*
 * numerical_stability.cpp
 * Chapter 11: Debugging and Retraining Deployed Models
 *
 * Numerical bugs appear as unstable probabilities, NaN/Inf propagation,
 * or output differences across devices and precisions (FP32/FP16/BF16).
 *
 * This file covers:
 *   - Stable log-sum-exp (numerically stable softmax)
 *   - CUDA error check macro (diagnostic, compiled conditionally)
 *   - Numerical instability detection patterns
 *   - IEEE-754 special value diagnostics
 *
 * PDF pages: 437-440 (book pp. 437-440)
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// ================================================================
// Conditional CUDA support
// Set USE_CUDA=1 to compile with CUDA error checking
// ================================================================

#ifdef USE_CUDA
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err__ = (call);                                    \
        if (err__ != cudaSuccess) {                                    \
            std::cerr << "CUDA error " << cudaGetErrorString(err__)    \
                      << " @ " << __FILE__ << ":" << __LINE__ << "\n"; \
            assert(false);                                             \
        }                                                              \
    } while (0)
#else
// Mock for compilation without CUDA
#define CUDA_CHECK(call) ((void)(call))
#endif

// ================================================================
// 1. Stable log-sum-exp (PDF p. 438)
//    Computes log(sum(exp(x_i))) without overflow/underflow
//    Used as a building block for numerically stable softmax
// ================================================================

float stable_logsumexp(const float *x, int n) {
    float m = x[0];
    for (int i = 1; i < n; ++i) {
        m = std::max(m, x[i]);
    }
    double acc = 0.0;
    for (int i = 0; i < n; ++i) {
        acc += std::exp(double(x[i] - m));
    }
    return m + float(std::log(acc));
}

// Convenience version using vector
float stable_logsumexp(const std::vector<float> &x) {
    return stable_logsumexp(x.data(), static_cast<int>(x.size()));
}

// ================================================================
// 2. Stable softmax using log-sum-exp
// ================================================================

std::vector<float> stable_softmax(const std::vector<float> &logits) {
    float lse = stable_logsumexp(logits);
    std::vector<float> probs(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(float(double(logits[i]) - double(lse)));
    }
    return probs;
}

// ================================================================
// 3. Epsilon clamp for division and sqrt
// ================================================================

inline double safe_divide(double num, double den, double epsilon = 1e-12) {
    return num / std::max(std::abs(den), epsilon);
}

inline double safe_sqrt(double x) {
    return std::sqrt(std::max(x, 0.0));
}

// ================================================================
// 4. IEEE-754 special value diagnostics
// ================================================================

void diagnose_special_values(const std::vector<float> &data,
                             const std::string &label) {
    int n_nan = 0, n_inf = 0, n_subnormal = 0, n_zero = 0;
    float min_val = INFINITY, max_val = -INFINITY;

    for (float v : data) {
        if (std::isnan(v))
            n_nan++;
        else if (std::isinf(v))
            n_inf++;
        else if (v == 0.0f)
            n_zero++;

        if (std::fpclassify(v) == FP_SUBNORMAL) n_subnormal++;
        if (std::isfinite(v) && v < min_val) min_val = v;
        if (std::isfinite(v) && v > max_val) max_val = v;
    }

    std::cout << "  " << label << " diagnostics:\n";
    std::cout << "    n=" << data.size()
              << " NaN=" << n_nan
              << " Inf=" << n_inf
              << " Zero=" << n_zero
              << " Subnormal=" << n_subnormal << "\n";
    if (n_nan + n_inf > 0) {
        std::cout << "    !! Unstable values detected !!\n";
    } else {
        std::cout << "    min=" << min_val << " max=" << max_val << "\n";
    }
}

// ================================================================
// 5. Numerical precision comparison (FP32 vs simulated FP16)
//    Helps diagnose quantization-related instability
// ================================================================

float to_fp16_sim(float x) {
    // Simulate FP16 range limitations for diagnosis
    if (std::isnan(x) || std::isinf(x)) return x;
    if (x > 65504.0f) return INFINITY;
    if (x < -65504.0f) return -INFINITY;
    return x;
}

struct PrecisionComparison {
    double max_diff = 0.0;
    int count = 0;
    int mismatches = 0;

    void compare(const std::vector<float> &a, const std::vector<float> &b,
                 double tolerance = 1e-3) {
        for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
            count++;
            double diff = std::abs(double(a[i]) - double(b[i]));
            if (diff > max_diff) max_diff = diff;
            if (diff > tolerance * std::max(1.0, double(std::abs(a[i])))) {
                mismatches++;
            }
        }
    }

    void report() const {
        std::cout << "  max_diff=" << max_diff
                  << " mismatches=" << mismatches << "/" << count
                  << " (" << (100.0 * mismatches / std::max(count, 1)) << "%)\n";
        if (max_diff > 1e-2) {
            std::cout << "  >> FP16 precision may be degrading outputs\n";
        }
    }
};

// ================================================================
// Main demonstration
// ================================================================

int main() {
    std::cout << "=== Chapter 11: Numerical Stability ===\n\n";

    // --- Stable log-sum-exp ---
    std::cout << "1. Stable log-sum-exp\n";
    std::vector<float> logits = {100.0f, 200.0f, -100.0f, 0.0f};
    float lse = stable_logsumexp(logits);
    std::cout << "   logits=[100, 200, -100, 0] -> log-sum-exp=" << lse << "\n";

    // --- Stable softmax ---
    std::cout << "\n2. Stable softmax vs naive softmax\n";
    std::vector<float> extreme_logits = {1000.0f, 900.0f, -1000.0f};
    auto probs = stable_softmax(extreme_logits);
    std::cout << "   logits=[1000, 900, -1000]\n";
    std::cout << "   probs=[";
    for (size_t i = 0; i < probs.size(); ++i)
        std::cout << std::fixed << std::setprecision(6)
                  << probs[i] << (i < probs.size() - 1 ? ", " : "");
    std::cout << "]\n";

    // --- Special value diagnostics ---
    std::cout << "\n3. IEEE-754 float diagnostics\n";
    std::vector<float> dirty_data = {1.0f, NAN, 3.0f, INFINITY, 0.0f, -INFINITY};
    diagnose_special_values(dirty_data, "Dirty tensor");

    // Clean data
    std::vector<float> clean_data = {0.01f, 0.5f, 0.99f, 0.33f, 0.67f};
    diagnose_special_values(clean_data, "Clean tensor");

    // --- FP32 vs FP16 simulation ---
    std::cout << "\n4. FP32 vs FP16 precision comparison\n";
    std::vector<float> fp32_outs = {0.001f, 0.998f, 0.001f, 0.5f,
                                    0.99999f, 1e-8f, 2.5e-5f};
    std::vector<float> fp16_outs;
    for (float v : fp32_outs) fp16_outs.push_back(to_fp16_sim(v));

    PrecisionComparison cmp;
    cmp.compare(fp32_outs, fp16_outs, 0.01);
    cmp.report();

    // Show actual differences
    std::cout << "   element-wise comparison:\n";
    for (size_t i = 0; i < fp32_outs.size(); ++i) {
        double diff = std::abs(double(fp32_outs[i]) - double(fp16_outs[i]));
        std::cout << "     [" << i << "] FP32=" << std::setw(10) << fp32_outs[i]
                  << " FP16_sim=" << std::setw(10) << fp16_outs[i]
                  << " diff=" << diff << "\n";
    }

    // --- CUDA_CHECK demo (mock) ---
    std::cout << "\n5. CUDA error check macro (mock mode)\n";
    std::cout << "   CUDA_CHECK skips actual CUDA calls when USE_CUDA is not defined.\n";
    std::cout << "   Build with: -DUSE_CUDA=1 to enable CUDA error checking.\n";

    // --- Diagnostic runbook ---
    std::cout << "\n6. Numerical debugging checklists\n";
    std::cout << "   Quick checks when output looks wrong:\n";
    std::cout << "   1. Use numerically stable forms (log-sum-exp for softmax)\n";
    std::cout << "   2. Add epsilon clamps for division/sqrt\n";
    std::cout << "   3. Compare FP32 with FP16/BF16 outputs\n";
    std::cout << "   4. Disable aggressive fast-math optimizations during diagnosis\n";
    std::cout << "   5. Check denormals / flush-to-zero behavior on your platform\n";

    // --- Simulated bug fix example ---
    std::cout << "\n7. Simulated INT8 quantization stability check\n";
    std::cout << "   INT8 quantized confidences may saturate at 0 or 1\n";
    std::cout << "   for large logits while FP32 reference is well-behaved.\n";
    std::cout << "   Fix: rescale, recalibrate, or use mixed precision at vulnerable layers.\n";

    std::cout << "\n=== Numerical stability demo complete ===\n";
    return 0;
}
