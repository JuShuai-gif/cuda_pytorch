/*
 * shap_kernelshap.cpp
 * Chapter 13: Explainability and Transparency
 *
 * KernelSHAP estimates Shapley values - additive feature attributions
 * from cooperative game theory. Each feature's contribution is its
 * average marginal contribution across all possible coalitions.
 *
 * This file covers:
 *   - Coalition mask sampling (binary masks for feature presence)
 *   - Shapley kernel weighting (emphasizes intermediate-sized coalitions)
 *   - Masked input building (instance vs background)
 *   - Weighted regression for phi values
 *   - Additivity verification (phi0 + sum(phi) ≈ f(x))
 *
 * PDF pages: 531-535 (book pp. 531-535)
 *
 * Key formula: φ_j = Σ w(S) · [f(S ∪ {j}) - f(S)]
 *   where w(S) = (M-1) / (C(M,k) · k · (M-k))
 */

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// ================================================================
// 1. Configuration (PDF p. 532)
// ================================================================

struct ShapConfig {
    int n_coalitions = 512;
    int background_K = 64;
    double ridge_lambda = 1e-6;
    unsigned seed = 1337;
};

// ================================================================
// 2. Output (PDF p. 532)
// ================================================================

struct ShapValues {
    std::vector<double> phi; // per-feature attributions
    double phi0 = 0.0;       // baseline (expected model output)
};

// ================================================================
// 3. Black-box model (same opaque interface as LIME)
// ================================================================

struct Model {
    Eigen::VectorXd true_weights;

    explicit Model(int d) : true_weights(d) {
        std::mt19937 rng(123);
        std::normal_distribution<double> N(0, 2);
        for (int i = 0; i < d; ++i) true_weights[i] = N(rng);
    }

    double score(const Eigen::VectorXd &x) const {
        return true_weights.dot(x);
    }

    void score_batch(const std::vector<Eigen::VectorXd> &X,
                     std::vector<double> &out) const {
        out.resize(X.size());
        for (size_t i = 0; i < X.size(); ++i) out[i] = score(X[i]);
    }
};

// ================================================================
// 4. Coalition mask sampling (PDF p. 532)
//    Binary mask z ∈ {0,1}^M: 1 = from instance, 0 = from background
// ================================================================

std::vector<uint8_t> sample_mask(int M, std::mt19937 &rng) {
    std::uniform_int_distribution<int> size_dist(1, std::max(1, M - 1));
    int k = size_dist(rng);
    std::vector<uint8_t> z(M, 0);
    std::vector<int> idx(M);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);
    for (int i = 0; i < k; ++i) z[idx[i]] = 1;
    return z;
}

// ================================================================
// 5. Build KernelSHAP design matrix (PDF p. 533)
// ================================================================

void build_kernelshap_design(
    int M,
    const ShapConfig &cfg,
    Eigen::MatrixXd &Z,
    Eigen::VectorXd &w,
    std::vector<std::vector<uint8_t>> &masks) {
    Z.resize(cfg.n_coalitions, M + 1);
    w.resize(cfg.n_coalitions);
    masks.clear();
    masks.reserve(cfg.n_coalitions);
    std::mt19937 rng(cfg.seed);

    // Binomial coefficient helper
    auto comb = [](int n, int k) -> double {
        if (k < 0 || k > n) return 0.0;
        k = std::min(k, n - k);
        double c = 1.0;
        for (int i = 1; i <= k; ++i) c = c * (n - k + i) / i;
        return c;
    };

    for (int i = 0; i < cfg.n_coalitions; ++i) {
        auto z = sample_mask(M, rng);
        int k = std::accumulate(z.begin(), z.end(), 0);

        // Shapley kernel weight: emphasizes intermediate coalition sizes
        double omega = (M > 1) ? static_cast<double>(M - 1) / (comb(M, k) * k * (M - k)) : 1.0;

        masks.push_back(z);
        w[i] = std::max(omega, 1e-12);
        Z(i, 0) = 1.0; // intercept (phi0)
        for (int j = 0; j < M; ++j) {
            Z(i, j + 1) = static_cast<double>(z[j]);
        }
    }
}

// ================================================================
// 6. Build masked inputs (PDF pp. 533-534)
//    x̃(z, b) = x ⊙ z + b ⊙ (1 - z)
//    Each mask creates a hybrid: instance features where z=1,
//    background features where z=0
// ================================================================

std::vector<Eigen::VectorXd> build_masked_batch(
    const Eigen::VectorXd &x,
    const std::vector<Eigen::VectorXd> &background,
    const std::vector<std::vector<uint8_t>> &masks) {
    std::vector<Eigen::VectorXd> batch;
    batch.reserve(masks.size());

    for (size_t i = 0; i < masks.size(); ++i) {
        const auto &z = masks[i];
        const auto &b = background[i % background.size()];
        Eigen::VectorXd xmask = x;
        for (int j = 0; j < x.size(); ++j) {
            if (!z[j]) xmask[j] = b[j];
        }
        batch.push_back(std::move(xmask));
    }
    return batch;
}

// ================================================================
// 7. Solve weighted regression (PDF p. 534)
// ================================================================

ShapValues solve_kernelshap(
    const Eigen::MatrixXd &Z,
    const Eigen::VectorXd &w,
    const std::vector<double> &y,
    double ridge_lambda) {
    using namespace Eigen;
    VectorXd vy = Map<const VectorXd>(y.data(),
                                      static_cast<Eigen::Index>(y.size()));
    MatrixXd W = w.asDiagonal();
    MatrixXd ZtWZ = Z.transpose() * W * Z;

    for (int j = 0; j < Z.cols(); ++j) {
        ZtWZ(j, j) += ridge_lambda;
    }

    VectorXd theta = ZtWZ.ldlt().solve(Z.transpose() * W * vy);

    ShapValues out;
    out.phi0 = theta[0];
    out.phi.resize(Z.cols() - 1);
    for (int j = 0; j < static_cast<int>(out.phi.size()); ++j) {
        out.phi[j] = theta[j + 1];
    }
    return out;
}

// ================================================================
// 8. Full KernelSHAP pipeline
// ================================================================

ShapValues kernelshap_explain(
    const Eigen::VectorXd &x,
    const Model &model,
    const std::vector<Eigen::VectorXd> &background,
    const ShapConfig &cfg) {
    int M = x.size();
    Eigen::MatrixXd Z;
    Eigen::VectorXd w;
    std::vector<std::vector<uint8_t>> masks;

    // Build design matrix with coalition masks and Shapley kernel weights
    build_kernelshap_design(M, cfg, Z, w, masks);

    // Create masked inputs: each coalition = hybrid of instance + background
    auto batch = build_masked_batch(x, background, masks);

    // Score all coalitions in one batch
    std::vector<double> y;
    model.score_batch(batch, y);

    // Solve weighted regression to recover Shapley values
    return solve_kernelshap(Z, w, y, cfg.ridge_lambda);
}

// ================================================================
// 9. Display helpers
// ================================================================

void print_shap_values(const ShapValues &sv,
                       const std::vector<std::string> &feature_names,
                       double model_output) {
    std::cout << "  Baseline (phi0): " << sv.phi0 << "\n";

    // Sort by absolute importance
    std::vector<std::pair<int, double>> ranked;
    for (size_t j = 0; j < sv.phi.size(); ++j) {
        ranked.push_back({static_cast<int>(j), sv.phi[j]});
    }
    std::sort(ranked.begin(), ranked.end(),
              [](auto &a, auto &b) {
                  return std::fabs(a.second) > std::fabs(b.second);
              });

    double sum_phi = sv.phi0;
    std::cout << "\n  Waterfall (from baseline to prediction):\n";
    std::cout << "  " << std::setw(22) << "phi0 (baseline)";
    std::cout << std::setw(12) << sv.phi0 << "\n";

    for (auto &[j, val] : ranked) {
        sum_phi += val;
        std::cout << "  " << std::setw(22) << feature_names[j]
                  << std::setw(12) << std::showpos << val
                  << std::noshowpos << " -> " << sum_phi << "\n";
    }
    std::cout << "  " << std::setw(22) << "f(x) final"
              << std::setw(12) << model_output << "\n";

    // Additivity check
    double additivity_err = std::abs(sv.phi0 + std::accumulate(sv.phi.begin(), sv.phi.end(), 0.0) - model_output);
    std::cout << "\n  Additivity error = " << additivity_err
              << (additivity_err < 1e-3 ? " (PASS)" : " (FAIL)") << "\n";
}

// ================================================================
// Main demonstration
// ================================================================

int main() {
    std::cout << "=== Chapter 13: KernelSHAP ===\n\n";

    const int M = 5; // 5 features
    const char *feature_names[] = {
        "merchant_category", "distance_from_home",
        "device_risk_score", "transaction_velocity",
        "hour_of_day"};

    // Generate background dataset (representative of served distribution)
    std::vector<Eigen::VectorXd> background;
    std::mt19937 rng(42);
    std::normal_distribution<double> N(0, 1);
    for (int i = 0; i < 200; ++i) {
        Eigen::VectorXd b(M);
        for (int j = 0; j < M; ++j) b[j] = N(rng) + j * 0.3;
        background.push_back(b);
    }

    // Create black-box model
    Model model(M);

    // Point to explain
    Eigen::VectorXd x(M);
    x << 1.5, -0.8, 2.3, 0.4, -1.2;

    double model_output = model.score(x);
    std::cout << "1. Point to explain:\n   [";
    for (int j = 0; j < M; ++j)
        std::cout << x[j] << (j < M - 1 ? ", " : "");
    std::cout << "]\n";
    std::cout << "   Model output f(x) = " << model_output << "\n";

    // Run KernelSHAP
    ShapConfig cfg;
    cfg.n_coalitions = 512;
    cfg.background_K = 100;
    cfg.seed = 42;

    std::cout << "\n2. KernelSHAP configuration\n";
    std::cout << "   n_coalitions=" << cfg.n_coalitions
              << " background_K=" << cfg.background_K << "\n";

    ShapValues sv = kernelshap_explain(x, model, background, cfg);

    std::vector<std::string> names;
    for (int j = 0; j < M; ++j) names.push_back(feature_names[j]);

    std::cout << "\n3. SHAP values - waterfall explanation\n";
    print_shap_values(sv, names, model_output);

    // Cohort aggregation demonstration
    std::cout << "\n4. Cohort-level SHAP (simulated)\n";
    std::cout << "   (In production, aggregate SHAP across many instances)\n";
    std::cout << "   Global feature importance ranking:\n";

    // Simulate SHAP values for multiple instances
    std::vector<double> glob_imp(M, 0.0);
    for (int inst = 0; inst < 10; ++inst) {
        Eigen::VectorXd xi(M);
        for (int j = 0; j < M; ++j) xi[j] = N(rng) + j * 0.3;
        auto sv_i = kernelshap_explain(xi, model, background, cfg);
        for (int j = 0; j < M; ++j) glob_imp[j] += std::fabs(sv_i.phi[j]);
    }
    for (int j = 0; j < M; ++j) glob_imp[j] /= 10.0;

    std::vector<std::pair<int, double>> ranked;
    for (int j = 0; j < M; ++j) ranked.push_back({j, glob_imp[j]});
    std::sort(ranked.begin(), ranked.end(),
              [](auto &a, auto &b) { return a.second > b.second; });
    for (auto &[j, imp] : ranked) {
        std::cout << "  " << std::setw(24) << feature_names[j]
                  << ": " << std::fixed << std::setprecision(4) << imp << "\n";
    }

    // Pitfalls
    std::cout << "\n5. KernelSHAP pitfalls:\n";
    std::cout << "   - Feature independence assumption -> correlated features bias values.\n";
    std::cout << "   - Background set choice changes baselines -> document carefully.\n";
    std::cout << "   - Fix: group correlated features; use conditional masking.\n";
    std::cout << "   - Fix: increase n_coalitions for better accuracy at cost of latency.\n";
    std::cout << "   - Fix: batch coalition queries, cache repeated masks.\n";

    std::cout << "\n=== KernelSHAP demo complete ===\n";
    return 0;
}
