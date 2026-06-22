/*
 * reproducible_debugging.cpp
 * Chapter 11: Debugging and Retraining Deployed Models
 *
 * A bug that cannot be reproduced is difficult to fix with confidence.
 * This file implements:
 *   - ReproPack: compact bundle for capturing failure context
 *   - Deterministic RNG seeding (fixed seed for replay)
 *   - NaN/Inf sentinel checks after numerically risky operations
 *   - Minimal sanity-check helper for input validation
 *
 * PDF pages: 435-437 (book pp. 435-437)
 *
 * Operational rule: store only what is needed to reproduce the failure.
 * Capture normalized tensors (with sensitive fields redacted), or
 * stable hashes plus a deterministic reconstruction method.
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// ================================================================
// 1. ReproPack: minimum metadata to reproduce a failure
// ================================================================

struct ReproPack {
    std::string model_ver;
    std::string preprocess_ver;
    std::string device;
    std::string flags;
    uint64_t seed;
    // Optional: small binary blob for normalized inputs, or hashes
    // plus a reconstruction method

    std::string summary() const {
        std::ostringstream os;
        os << "ReproPack{model=" << model_ver
           << ", preproc=" << preprocess_ver
           << ", device=" << device
           << ", seed=" << seed
           << ", flags=" << flags << "}";
        return os.str();
    }
};

inline std::mt19937 make_rng(uint64_t seed) {
    return std::mt19937{static_cast<unsigned>(seed)};
}

// ================================================================
// 2. NaN/Inf sentinel check
//    Place after normalization, division, log, exp, or softmax
// ================================================================

template <typename T>
bool has_nan_inf(const T *data, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(static_cast<double>(data[i]))) return true;
    }
    return false;
}

template <typename T>
bool has_nan_inf(const std::vector<T> &data) {
    return has_nan_inf(data.data(), data.size());
}

// ================================================================
// 3. Softmax with NaN guard and memory of last stable output
//    When NaN is detected, fall back to cached safe output
// ================================================================

class SafeSoftmax {
public:
    std::vector<double> cached_output;

    std::vector<double> compute(const std::vector<double> &logits) {
        // Stable softmax: subtract max, exp, normalize
        double max_val = *std::max_element(logits.begin(), logits.end());
        std::vector<double> output(logits.size());
        double sum = 0.0;

        for (size_t i = 0; i < logits.size(); ++i) {
            output[i] = std::exp(logits[i] - max_val);
            sum += output[i];
        }

        if (std::isfinite(sum) && sum > 0.0) {
            for (auto &v : output) v /= sum;
            if (!has_nan_inf(output)) {
                cached_output = output; // Cache safe output
                return output;
            }
        }

        // Fallback: return cached output or uniform distribution
        if (!cached_output.empty()) {
            std::cerr << "Warning: NaN detected, using cached softmax output\n";
            return cached_output;
        }

        // Last resort: uniform distribution
        std::cerr << "Warning: NaN detected, no cache, using uniform fallback\n";
        std::vector<double> uniform(logits.size(), 1.0 / logits.size());
        return uniform;
    }
};

// ================================================================
// 4. Input sanity check
//    Validate feature ranges, shapes, and absence of illegal values
// ================================================================

struct DataContractCheck {
    // Check that all values are within expected range
    static bool in_range(const std::vector<double> &data,
                         double min_val, double max_val) {
        for (size_t i = 0; i < data.size(); ++i) {
            if (data[i] < min_val || data[i] > max_val) {
                std::cerr << "DataContract violation: data[" << i
                          << "]=" << data[i] << " outside [" << min_val
                          << ", " << max_val << "]\n";
                return false;
            }
        }
        return true;
    }

    // Check for schema mismatch: expected vs actual dimensions
    static bool check_shape(size_t expected, size_t actual,
                            const std::string &feature_name) {
        if (expected != actual) {
            std::cerr << "Schema mismatch for " << feature_name
                      << ": expected " << expected
                      << " got " << actual << "\n";
            return false;
        }
        return true;
    }

    // Check for unknown categorical tokens
    static bool check_unknown_rate(size_t unknown_count, size_t total_count,
                                   double threshold = 0.05) {
        double rate = (total_count > 0) ? static_cast<double>(unknown_count) / total_count : 0.0;
        if (rate > threshold) {
            std::cerr << "Unknown token rate " << (rate * 100)
                      << "% exceeds threshold " << (threshold * 100) << "%\n";
            return false;
        }
        return true;
    }
};

// ================================================================
// Main demonstration
// ================================================================

int main() {
    std::cout << "=== Chapter 11: Reproducible Debugging ===\n\n";

    // --- ReproPack demo ---
    std::cout << "1. ReproPack: capture failure context\n";
    ReproPack pack{"v3.1.0", "schema_v5", "cuda:0", "--batch_size=8 --fp16", 12345UL};
    std::cout << "   " << pack.summary() << "\n";
    std::cout << "   RNG state: mt19937 with seed=" << pack.seed << "\n";

    // Demonstrate deterministic RNG
    auto rng1 = make_rng(pack.seed);
    auto rng2 = make_rng(pack.seed);
    std::cout << "   Deterministic replay: ";
    std::cout << "rng1[0]=" << rng1() << " rng2[0]=" << rng2() << " -> "
              << (rng1() == rng2() ? "same" : "different") << "\n";

    // --- NaN guard demo ---
    std::cout << "\n2. NaN/Inf sentinel check\n";
    std::vector<double> clean_data = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> inf_data = {1.0, INFINITY, 3.0};
    std::vector<double> nan_data = {1.0, NAN, 3.0};

    std::cout << "   clean: " << (has_nan_inf(clean_data) ? "BAD" : "ok") << "\n";
    std::cout << "   inf:   " << (has_nan_inf(inf_data) ? "BAD" : "ok") << "\n";
    std::cout << "   nan:   " << (has_nan_inf(nan_data) ? "BAD" : "ok") << "\n";

    // --- Safe softmax with fallback ---
    std::cout << "\n3. SafeSoftmax: NaN-resistant with fallback\n";
    SafeSoftmax safe_sm;
    std::vector<double> logits = {2.0, 1.0, 0.1, -1.0, -5.0};
    auto probs = safe_sm.compute(logits);
    std::cout << "   logits=[2.0, 1.0, 0.1, -1.0, -5.0]\n";
    std::cout << "   probs=[";
    for (size_t i = 0; i < probs.size(); ++i) {
        std::cout << probs[i] << (i < probs.size() - 1 ? ", " : "");
    }
    std::cout << "]\n";

    // Test with extreme values that could overflow
    std::cout << "\n   Extreme logits test:\n";
    std::vector<double> extreme = {1e10, -1e10, 0.0};
    auto safe_probs = safe_sm.compute(extreme);
    std::cout << "   logits=[1e10, -1e10, 0.0] -> probs=[";
    for (size_t i = 0; i < safe_probs.size(); ++i) {
        std::cout << safe_probs[i] << (i < safe_probs.size() - 1 ? ", " : "");
    }
    std::cout << "]\n";

    // --- Data contract checks ---
    std::cout << "\n4. Data contract validation\n";
    std::vector<double> prices = {9.99, 29.99, 150.00, 0.01, 500.0};
    std::cout << "   Range check [0, 1000]: "
              << (DataContractCheck::in_range(prices, 0.0, 1000.0) ? "PASS" : "FAIL")
              << "\n";
    std::cout << "   Shape check (expect 5, got 5): "
              << (DataContractCheck::check_shape(5, 5, "price_vector") ? "PASS" : "FAIL")
              << "\n";
    std::cout << "   Unknown token rate (3/100): "
              << (DataContractCheck::check_unknown_rate(3, 100, 0.05) ? "PASS" : "FAIL")
              << "\n";

    // --- Simulated incident: reproduce with ReproPack ---
    std::cout << "\n5. Reproduce with ReproPack\n";
    {
        // Engineer receives ReproPack from production alert
        ReproPack incident{"v3.1.0", "schema_v5", "cuda:0",
                           "--batch_size=8 --fp16", 12345UL};
        std::cout << "   Incident pack: " << incident.summary() << "\n";

        // Fix seeds to replay deterministically
        auto rng = make_rng(incident.seed);
        std::cout << "   Deterministic seed applied: " << incident.seed << "\n";

        // Record shapes and NaN guard
        std::vector<double> offending_input = {100.0, 0.0, -50.0, 3.0e40, 0.001};
        std::cout << "   Recorded input shape: " << offending_input.size() << "\n";
        std::cout << "   NaN/inf check on input: "
                  << (has_nan_inf(offending_input) ? "FAIL" : "PASS") << "\n";
    }

    std::cout << "\n=== Reproducible debugging demo complete ===\n";
    std::cout << "\nOperational rules:\n";
    std::cout << "  - Fix seeds for deterministic replay\n";
    std::cout << "  - Record shapes and data types explicitly\n";
    std::cout << "  - Use sentinel checks (NaN/Inf/range) at transform boundaries\n";
    std::cout << "  - In canary: fail fast. In production: fallback safely.\n";

    return 0;
}
