/*
 * 07_drift_detection.cpp
 * Chapter 10: Model Deployment and Inference Optimization
 *
 * Drift detection monitors for distribution shifts between the training
 * data and the data seen in production. Three types of drift:
 *
 * 1. Data Drift (Covariate Shift): Input feature distributions change.
 *    Detected via PSI (Population Stability Index) and KS test.
 *
 * 2. Prediction Drift: Model output distributions shift.
 *    Detected via output histogram comparisons and calibration shifts.
 *
 * 3. Concept Drift: The relationship between inputs and labels changes.
 *    Detected via business KPI decline (offline accuracy may stay normal).
 *
 * PSI (Population Stability Index):
 *   PSI = sum_i (P_i - Q_i) * ln(P_i / Q_i)
 *   where P_i is the reference distribution bin proportion,
 *   and Q_i is the current (production) distribution bin proportion.
 *
 *   PSI < 0.1  → no significant drift
 *   0.1 ≤ PSI < 0.25 → moderate drift (investigate)
 *   PSI ≥ 0.25 → significant drift (alert)
 *
 * KS (Kolmogorov-Smirnov) Statistic:
 *   KS = max |CDF_ref(x) - CDF_current(x)|
 *   Measures maximum distance between cumulative distributions.
 *
 *   KS < 0.05 → no significant drift
 *   0.05 ≤ KS < 0.1 → moderate drift
 *   KS ≥ 0.1 → significant drift
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <map>
#include <cassert>
#include <set>

// ----------------------------------------------------------------
// Compute PSI (Population Stability Index) for a continuous variable
//
// Steps:
//   1. Bin the reference data into `num_bins` equal-width bins
//   2. Compute proportion of current data falling into each reference bin
//   3. PSI = sum_i (P_i - Q_i) * ln(P_i / Q_i)
//
//   Small epsilon added to avoid ln(0) issues.
// ----------------------------------------------------------------
double computePSI(
    const std::vector<float> &reference,
    const std::vector<float> &current,
    int num_bins = 10) {
    if (reference.empty() || current.empty()) return 0.0;

    // Determine bin edges from reference
    auto [ref_min, ref_max] = std::minmax_element(reference.begin(), reference.end());
    double range = *ref_max - *ref_min;
    if (range < 1e-8) return 0.0;

    double bin_width = range / num_bins;

    // Count reference proportions
    std::vector<double> ref_props(num_bins, 0.0);
    for (float v : reference) {
        int bin = static_cast<int>((v - *ref_min) / bin_width);
        bin = std::max(0, std::min(num_bins - 1, bin));
        ref_props[bin] += 1.0;
    }
    for (auto &p : ref_props) p /= reference.size();

    // Count current proportions
    std::vector<double> cur_props(num_bins, 0.0);
    for (float v : current) {
        int bin = static_cast<int>((v - *ref_min) / bin_width);
        bin = std::max(0, std::min(num_bins - 1, bin));
        cur_props[bin] += 1.0;
    }
    for (auto &p : cur_props) p /= current.size();

    // Compute PSI
    double psi = 0.0;
    const double eps = 0.0001; // smoothing to avoid ln(0)
    for (int i = 0; i < num_bins; i++) {
        double p = std::max(ref_props[i], eps);
        double q = std::max(cur_props[i], eps);
        psi += (q - p) * std::log(q / p);
    }

    return psi;
}

// ----------------------------------------------------------------
// KS (Kolmogorov-Smirnov) Test Statistic
//
// Compares empirical CDFs of reference and current distributions.
// KS = max |CDF_ref(x) - CDF_current(x)|
// ----------------------------------------------------------------
double computeKS(
    const std::vector<float> &reference,
    const std::vector<float> &current) {
    auto ref_sorted = reference;
    auto cur_sorted = current;
    std::sort(ref_sorted.begin(), ref_sorted.end());
    std::sort(cur_sorted.begin(), cur_sorted.end());

    int i = 0, j = 0;
    double max_diff = 0.0;
    double ref_n = ref_sorted.size();
    double cur_n = cur_sorted.size();

    while (i < ref_n || j < cur_n) {
        double x;
        if (j >= cur_n || (i < ref_n && ref_sorted[i] <= cur_sorted[j])) {
            x = ref_sorted[i];
            i++;
        } else {
            x = cur_sorted[j];
            j++;
        }

        double ref_cdf = static_cast<double>(i) / ref_n;
        double cur_cdf = static_cast<double>(j) / cur_n;
        max_diff = std::max(max_diff, std::abs(ref_cdf - cur_cdf));
    }

    return max_diff;
}

// ----------------------------------------------------------------
// Compute categorical distribution shift (for discrete features)
//
// Uses chi-square-like metric:
//   shift = sum_i (p_i - q_i)^2 / (p_i + q_i)
// ----------------------------------------------------------------
double categoricalShift(
    const std::map<int, double> &ref_dist,
    const std::map<int, double> &cur_dist) {
    // Collect all keys
    std::set<int> all_keys;
    for (auto &[k, _] : ref_dist) all_keys.insert(k);
    for (auto &[k, _] : cur_dist) all_keys.insert(k);

    double shift = 0.0;
    for (int k : all_keys) {
        double p = ref_dist.count(k) ? ref_dist.at(k) : 0.01;
        double q = cur_dist.count(k) ? cur_dist.at(k) : 0.01;
        shift += (q - p) * (q - p) / (p + q);
    }
    return shift;
}

// ----------------------------------------------------------------
// Prediction drift: compare output distribution (logit / softmax)
// between reference and current data
// ----------------------------------------------------------------
double predictionDrift(
    const std::vector<float> &ref_pred_mean,
    const std::vector<float> &cur_pred_mean,
    double threshold = 0.05) {
    // Simple: mean absolute change in prediction means
    double total_change = 0.0;
    for (size_t i = 0; i < ref_pred_mean.size(); i++) {
        total_change += std::abs(cur_pred_mean[i] - ref_pred_mean[i]);
    }
    double avg_change = total_change / ref_pred_mean.size();

    return avg_change;
}

// ----------------------------------------------------------------
// Drift status assessment
// ----------------------------------------------------------------
std::string psiStatus(double psi) {
    if (psi < 0.1) return "OK (no drift)";
    if (psi < 0.25) return "WARN (moderate drift)";
    return "ALERT (significant drift)";
}

std::string ksStatus(double ks) {
    if (ks < 0.05) return "OK (no drift)";
    if (ks < 0.10) return "WARN (moderate drift)";
    return "ALERT (significant drift)";
}

// ----------------------------------------------------------------
// Demo: Reference vs drifted data
// ----------------------------------------------------------------
int main() {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== Drift Detection Demo ===\n\n";

    std::mt19937 rng(42);

    // Generate reference data: N(0, 1)
    std::vector<float> reference(1000);
    std::normal_distribution<float> ref_dist(0.0, 1.0);
    for (auto &v : reference) v = ref_dist(rng);

    // Generate current data (same distribution — should show no drift)
    std::vector<float> current_ok(1000);
    for (auto &v : current_ok) v = ref_dist(rng);

    // Generate drifted data: N(0.3, 1.5) — mean shift + variance increase
    std::vector<float> current_drifted(1000);
    std::normal_distribution<float> drift_dist(0.3, 1.5);
    for (auto &v : current_drifted) v = drift_dist(rng);

    std::cout << "Reference:  mean="
              << std::accumulate(reference.begin(), reference.end(), 0.0) / reference.size()
              << " std="
              << [&]() {
                     double m = std::accumulate(reference.begin(), reference.end(), 0.0) / reference.size();
                     double sq = 0;
                     for (auto v : reference) sq += (v - m) * (v - m);
                     return std::sqrt(sq / reference.size());
                 }()
              << "\n\n";

    // --- No drift case ---
    std::cout << "--- Case 1: No drift (same distribution) ---\n";
    double psi_ok = computePSI(reference, current_ok);
    double ks_ok = computeKS(reference, current_ok);
    std::cout << "  PSI: " << psi_ok << " — " << psiStatus(psi_ok) << "\n";
    std::cout << "  KS:  " << ks_ok << " — " << ksStatus(ks_ok) << "\n\n";

    // --- Drifted case ---
    std::cout << "--- Case 2: Distribution drift (mean shifted, variance increased) ---\n";
    double psi_drifted = computePSI(reference, current_drifted);
    double ks_drifted = computeKS(reference, current_drifted);
    std::cout << "  PSI: " << psi_drifted << " — " << psiStatus(psi_drifted) << "\n";
    std::cout << "  KS:  " << ks_drifted << " — " << ksStatus(ks_drifted) << "\n\n";

    // --- Categorical drift ---
    std::cout << "--- Case 3: Categorical distribution drift ---\n";
    std::map<int, double> ref_cat = {{0, 0.60}, {1, 0.30}, {2, 0.10}};
    std::map<int, double> cur_cat = {{0, 0.40}, {1, 0.25}, {2, 0.20}, {3, 0.15}};
    double cat_shift = categoricalShift(ref_cat, cur_cat);
    std::cout << "  Reference: {0:0.60, 1:0.30, 2:0.10}\n";
    std::cout << "  Current:   {0:0.40, 1:0.25, 2:0.20, 3:0.15}\n";
    std::cout << "  Categorical shift: " << cat_shift;
    std::cout << (cat_shift > 0.15 ? " — ALERT (new category appeared)" : "") << "\n\n";

    // --- Prediction drift ---
    std::cout << "--- Case 4: Prediction drift ---\n";
    std::vector<float> ref_pred = {0.70, 0.15, 0.10, 0.03, 0.02};
    std::vector<float> cur_pred = {0.55, 0.25, 0.12, 0.05, 0.03};
    double pred_drift = predictionDrift(ref_pred, cur_pred);
    std::cout << "  Reference class means: ["
              << ref_pred[0] << ", " << ref_pred[1] << ", "
              << ref_pred[2] << ", " << ref_pred[3] << ", " << ref_pred[4] << "]\n";
    std::cout << "  Current class means:   ["
              << cur_pred[0] << ", " << cur_pred[1] << ", "
              << cur_pred[2] << ", " << cur_pred[3] << ", " << cur_pred[4] << "]\n";
    std::cout << "  Mean prediction change: " << pred_drift;
    std::cout << (pred_drift > 0.05 ? " — WARN" : "") << "\n\n";

    // --- Drift detection dashboard summary ---
    std::cout << "--- Drift Detection Summary ---\n";
    std::cout << "| Metric              | Value      | Status  |\n";
    std::cout << "|---------------------|------------|---------|\n";
    std::cout << "| PSI (no drift)      | " << psi_ok << "  | " << psiStatus(psi_ok) << "\n";
    std::cout << "| PSI (drifted)       | " << psi_drifted << "  | " << psiStatus(psi_drifted) << "\n";
    std::cout << "| KS (no drift)       | " << ks_ok << "  | " << ksStatus(ks_ok) << "\n";
    std::cout << "| KS (drifted)        | " << ks_drifted << "  | " << ksStatus(ks_drifted) << "\n";
    std::cout << "| Categorical shift   | " << cat_shift << "  | "
              << (cat_shift > 0.15 ? "ALERT" : "OK") << "\n";
    std::cout << "| Prediction change   | " << pred_drift << "  | "
              << (pred_drift > 0.05 ? "WARN" : "OK") << "\n\n";

    std::cout << "Recommendation: Run drift notebook weekly.\n";
    std::cout << "  If PSI > 0.25: investigate feature pipeline changes.\n";
    std::cout << "  If KS > 0.10: check for broken data sources.\n";
    std::cout << "  If prediction drift: check calibration, may need retraining.\n";
    std::cout << "  If concept drift suspected: check business KPIs.\n";

    return 0;
}
