/*
 * lime_tabular.cpp
 * Chapter 13: Explainability and Transparency
 *
 * LIME (Local Interpretable Model-agnostic Explanations) explains a single
 * prediction by fitting a sparse linear surrogate model in the neighborhood
 * of the instance.
 *
 * This file covers:
 *   - Perturbation generation (Gaussian noise around standardized point)
 *   - Proximity kernel weighting (Gaussian RBF)
 *   - Weighted ridge regression via Eigen LDLT
 *   - Top-K coefficient extraction
 *   - Local R² for surrogate quality assessment
 *
 * PDF pages: 520-531 (book pp. 520-531)
 */

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// ================================================================
// 1. Configuration (PDF p. 528)
// ================================================================

struct LimeConfig {
    int n_samples = 512;
    int top_k = 8;
    double ridge_lambda = 1e-3;
    double kernel_width = 0.75;
    unsigned seed = 42;
};

// ================================================================
// 2. Explanation output (PDF p. 529)
// ================================================================

struct LimeExplanation {
    std::vector<int> feat_index;
    std::vector<double> feat_weight;
    double intercept = 0.0;
    double local_r2 = 0.0;
};

// ================================================================
// 3. Simple model interface (opaque black box, PDF p. 527)
// ================================================================

struct Model {
    // In production: wraps a TorchScript/ONNX model
    // Here: simulated linear model for demonstration
    Eigen::VectorXd true_weights;

    explicit Model(int d) : true_weights(d) {
        // Random true weights for a simulated black box
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
        for (size_t i = 0; i < X.size(); ++i) {
            out[i] = score(X[i]);
        }
    }
};

// ================================================================
// 4. Feature statistics for standardization
// ================================================================

struct FeatureStats {
    Eigen::VectorXd mean;
    Eigen::VectorXd stddev;
    int d;

    FeatureStats(int dim) : mean(dim), stddev(dim), d(dim) {
        mean.setZero();
        stddev.setOnes();
    }

    void fit(const std::vector<Eigen::VectorXd> &data) {
        if (data.empty()) return;
        mean.setZero();
        // Compute mean
        for (const auto &x : data) mean += x;
        mean /= static_cast<double>(data.size());
        // Compute stddev
        stddev.setZero();
        for (const auto &x : data) {
            Eigen::VectorXd diff = x - mean;
            for (int i = 0; i < d; ++i)
                stddev[i] += diff[i] * diff[i];
        }
        stddev /= static_cast<double>(data.size());
        for (int i = 0; i < d; ++i)
            stddev[i] = std::sqrt(std::max(stddev[i], 1e-8));
    }

    Eigen::VectorXd standardize(const Eigen::VectorXd &x) const {
        return (x - mean).cwiseQuotient(stddev);
    }

    Eigen::VectorXd destandardize(const Eigen::VectorXd &z) const {
        return z.cwiseProduct(stddev) + mean;
    }
};

// ================================================================
// 5. Perturbation generation (PDF p. 529)
//    Sample in standardized space, add Gaussian noise, clamp
// ================================================================

std::vector<Eigen::VectorXd> sample_lime_points(
    const Eigen::VectorXd &x0,
    const FeatureStats &stats,
    int n_samples,
    unsigned seed) {
    const int d = x0.size();
    std::vector<Eigen::VectorXd> Xs;
    Xs.reserve(n_samples);
    std::mt19937 rng(seed);
    std::normal_distribution<double> N01(0.0, 1.0);

    Eigen::VectorXd z0 = stats.standardize(x0);

    for (int i = 0; i < n_samples; ++i) {
        Eigen::VectorXd z = z0;
        for (int j = 0; j < d; ++j) {
            z[j] += 0.2 * N01(rng); // 0.2 std noise
        }
        // Clamp to reasonable range (3 std from center)
        for (int j = 0; j < d; ++j) {
            z[j] = std::max(-3.0, std::min(3.0, z[j]));
        }
        Eigen::VectorXd x = stats.destandardize(z);
        Xs.push_back(x);
    }
    return Xs;
}

// ================================================================
// 6. Weighted ridge regression (PDF pp. 530-531)
// ================================================================

Eigen::VectorXd fit_lime_ridge(
    const std::vector<Eigen::VectorXd> &Xs,
    const std::vector<double> &ys,
    const Eigen::VectorXd &x0,
    const FeatureStats &stats,
    double kernel_width,
    double ridge_lambda) {
    using namespace Eigen;
    const int n = static_cast<int>(Xs.size());
    const int d = x0.size();

    MatrixXd X(n, d + 1);
    VectorXd y(n), w(n);
    VectorXd z0 = stats.standardize(x0);

    for (int i = 0; i < n; ++i) {
        VectorXd zi = stats.standardize(Xs[i]);
        double dist2 = (zi - z0).squaredNorm();
        w[i] = std::exp(-dist2 / (kernel_width * kernel_width));
        X(i, 0) = 1.0; // bias / intercept
        for (int j = 0; j < d; ++j) X(i, j + 1) = Xs[i][j];
        y[i] = ys[i];
    }

    MatrixXd W = w.asDiagonal();
    MatrixXd XtWX = X.transpose() * W * X;

    // Add ridge penalty (skip intercept: j=0)
    for (int j = 1; j <= d; ++j) {
        XtWX(j, j) += ridge_lambda;
    }

    return XtWX.ldlt().solve(X.transpose() * W * y);
}

// ================================================================
// 7. Top-K coefficient extraction (PDF pp. 530-531)
// ================================================================

LimeExplanation build_lime_explanation(const Eigen::VectorXd &beta,
                                       int top_k) {
    struct Item {
        int j;
        double w;
    };
    std::vector<Item> items;
    for (int j = 1; j < beta.size(); ++j) {
        items.push_back({j - 1, beta[j]});
    }

    std::partial_sort(
        items.begin(),
        items.begin() + std::min(top_k, static_cast<int>(items.size())),
        items.end(),
        [](const Item &a, const Item &b) {
            return std::fabs(a.w) > std::fabs(b.w);
        });

    LimeExplanation out;
    out.intercept = beta[0];
    for (int k = 0; k < std::min(top_k, static_cast<int>(items.size())); ++k) {
        out.feat_index.push_back(items[k].j);
        out.feat_weight.push_back(items[k].w);
    }
    return out;
}

// ================================================================
// 8. Local R²: how well the surrogate fits the neighborhood
// ================================================================

double compute_local_r2(const std::vector<Eigen::VectorXd> &Xs,
                        const std::vector<double> &ys,
                        const Eigen::VectorXd &beta,
                        const std::vector<double> &weights) {
    double ss_res = 0.0, ss_tot = 0.0, mean_y = 0.0, sum_w = 0.0;
    for (size_t i = 0; i < ys.size(); ++i) {
        sum_w += weights[i];
        mean_y += weights[i] * ys[i];
    }
    mean_y /= sum_w;

    for (size_t i = 0; i < ys.size(); ++i) {
        double pred = beta[0];
        for (int j = 1; j < beta.size(); ++j) {
            pred += beta[j] * Xs[i][j - 1];
        }
        ss_res += weights[i] * (ys[i] - pred) * (ys[i] - pred);
        ss_tot += weights[i] * (ys[i] - mean_y) * (ys[i] - mean_y);
    }
    return (ss_tot > 0) ? 1.0 - ss_res / ss_tot : 0.0;
}

// ================================================================
// 9. Full LIME pipeline
// ================================================================

LimeExplanation lime_explain(const Eigen::VectorXd &x0,
                             const Model &model,
                             const FeatureStats &stats,
                             const LimeConfig &cfg) {
    // Generate perturbations
    auto Xs = sample_lime_points(x0, stats, cfg.n_samples, cfg.seed);

    // Score in batch
    std::vector<double> ys;
    model.score_batch(Xs, ys);

    // Fit weighted ridge
    Eigen::VectorXd beta = fit_lime_ridge(
        Xs, ys, x0, stats, cfg.kernel_width, cfg.ridge_lambda);

    // Build explanation
    LimeExplanation out = build_lime_explanation(beta, cfg.top_k);

    // Compute local R²
    std::vector<double> weights(Xs.size());
    Eigen::VectorXd z0 = stats.standardize(x0);
    for (size_t i = 0; i < Xs.size(); ++i) {
        Eigen::VectorXd zi = stats.standardize(Xs[i]);
        double dist2 = (zi - z0).squaredNorm();
        weights[i] = std::exp(-dist2 / (cfg.kernel_width * cfg.kernel_width));
    }
    out.local_r2 = compute_local_r2(Xs, ys, beta, weights);

    return out;
}

// ================================================================
// Main demonstration
// ================================================================

int main() {
    std::cout << "=== Chapter 13: LIME Tabular ===\n\n";

    const int d = 6;
    FeatureStats stats(d);

    // Simulate training data for standardization
    std::vector<Eigen::VectorXd> train_data;
    std::mt19937 rng(42);
    std::normal_distribution<double> N(0, 1);
    for (int i = 0; i < 500; ++i) {
        Eigen::VectorXd x(d);
        for (int j = 0; j < d; ++j) x[j] = N(rng) + (j * 0.5);
        train_data.push_back(x);
    }
    stats.fit(train_data);

    // Create a simulated black-box model
    Model model(d);

    // Point to explain
    Eigen::VectorXd x0(d);
    x0 << 1.2, -0.5, 2.1, 0.3, -1.5, 0.8;

    // Run LIME
    LimeConfig cfg;
    cfg.n_samples = 200;
    cfg.top_k = 6;
    cfg.kernel_width = 1.0;
    cfg.seed = 42;

    std::cout << "1. LIME configuration\n";
    std::cout << "   n_samples=" << cfg.n_samples
              << " kernel_width=" << cfg.kernel_width
              << " ridge_lambda=" << cfg.ridge_lambda << "\n";

    LimeExplanation expl = lime_explain(x0, model, stats, cfg);

    // Display results
    std::cout << "\n2. Point to explain (x0):\n   [";
    for (int i = 0; i < d; ++i)
        std::cout << x0[i] << (i < d - 1 ? ", " : "");
    std::cout << "]\n";

    double model_pred = model.score(x0);
    std::cout << "\n3. Model prediction: " << model_pred << "\n";
    std::cout << "   Surrogate intercept: " << expl.intercept << "\n";
    std::cout << "   Local R² (surrogate fit quality): "
              << std::fixed << std::setprecision(3) << expl.local_r2 << "\n";

    std::cout << "\n4. Top-K feature contributions:\n";
    const char *feature_names[] = {
        "debt_to_income", "recent_delinq", "credit_tenure",
        "utilization", "num_inquiries", "oldest_account"};
    for (size_t i = 0; i < expl.feat_index.size(); ++i) {
        int idx = expl.feat_index[i];
        std::cout << "  [" << idx << "] " << std::setw(16) << feature_names[idx]
                  << ": " << std::showpos << std::setw(10)
                  << std::fixed << std::setprecision(4) << expl.feat_weight[i]
                  << std::noshowpos << "\n";
    }

    // Validation: check if surrogate intercept + sum(feature contributions)
    // approximates model output
    std::cout << "\n5. Additivity check:\n";
    double surrogate_pred = expl.intercept;
    for (size_t i = 0; i < expl.feat_index.size(); ++i) {
        surrogate_pred += expl.feat_weight[i] * x0[expl.feat_index[i]];
    }
    // Full fit uses all features (not just top-K), so we use the linear model
    // for comparison
    std::cout << "   Surrogate intercept + Σ w_i * x_i (top-K only): "
              << surrogate_pred << "\n";
    std::cout << "   Model output: " << model_pred << "\n";

    // Pitfalls and guardrails
    std::cout << "\n6. Pitfalls and guardrails:\n";
    std::cout << "   - Kernal width too small -> noisy surrogate.\n";
    std::cout << "   - Kernal width too large -> stops being local.\n";
    std::cout << "   - Correlated features -> importances shift between them.\n";
    std::cout << "   - Fix: cap features shown, document perturbation policy.\n";
    std::cout << "   - Fix: batch all model queries, reuse buffers.\n";
    std::cout << "   - Fix: stable RNG seeds for reproducibility.\n";

    std::cout << "\n=== LIME tabular demo complete ===\n";
    return 0;
}
