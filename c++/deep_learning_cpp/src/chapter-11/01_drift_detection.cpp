/*
 * drift_detection.cpp
 * Chapter 11: Debugging and Retraining Deployed Models
 *
 * Drift detection is the first line of defense for production ML systems.
 * This file implements the core monitoring structures from the PDF:
 *   - Fixed-bin histograms for numerical feature distribution tracking
 *   - PSI (Population Stability Index) for comparing distributions
 *   - KS (Kolmogorov-Smirnov) test for continuous distributions
 *   - Chi-squared test for categorical distributions
 *   - KL divergence and JS divergence for probability distributions
 *   - Welford's algorithm for streaming mean/variance
 *
 * PDF pages: 419-425 (book pp. 420-425)
 * Drift taxonomy: covariate (p(x)), prior/label (p(y)), concept (p(y|x))
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

// ================================================================
// 1. Fixed-bin histogram (PDF pp. 422-424)
// ================================================================

struct Histogram {
    double min_val;
    double max_val;
    std::vector<std::uint64_t> bins;

    Histogram(double min_v, double max_v, std::size_t n_bins) : min_val(min_v), max_val(max_v), bins(n_bins, 0) {
        if (n_bins == 0) {
            throw std::invalid_argument("n_bins must be > 0");
        }
        if (max_val <= min_val) {
            throw std::invalid_argument("max_val must be greater than min_val");
        }
    }
};

std::size_t bin_index(const Histogram &h, double value) {
    if (value <= h.min_val) return 0;
    if (value >= h.max_val) return h.bins.size() - 1;
    const double width =
        (h.max_val - h.min_val) / static_cast<double>(h.bins.size());
    std::size_t idx = static_cast<std::size_t>((value - h.min_val) / width);
    if (idx >= h.bins.size()) {
        idx = h.bins.size() - 1;
    }
    return idx;
}

void update_histogram(Histogram &h, double value) {
    const std::size_t idx = bin_index(h, value);
    h.bins[idx]++;
}

void print_histogram(const Histogram &h) {
    const double width =
        (h.max_val - h.min_val) / static_cast<double>(h.bins.size());
    double bin_start = h.min_val;
    for (std::size_t i = 0; i < h.bins.size(); ++i) {
        std::cout << "  [" << std::fixed << std::setprecision(1)
                  << bin_start << ", " << bin_start + width << "): "
                  << h.bins[i] << "\n";
        bin_start += width;
    }
}

// ================================================================
// 2. Convert histogram counts to proportions (PDF p. 423)
// ================================================================

std::vector<double> to_proportions(const Histogram &h, double epsilon = 1e-12) {
    const std::uint64_t total =
        std::accumulate(h.bins.begin(), h.bins.end(), std::uint64_t{0});
    std::vector<double> proportions(h.bins.size(), 0.0);
    if (total == 0) {
        const double uniform = 1.0 / static_cast<double>(h.bins.size());
        std::fill(proportions.begin(), proportions.end(), uniform);
        return proportions;
    }
    for (std::size_t i = 0; i < h.bins.size(); ++i) {
        proportions[i] =
            std::max(static_cast<double>(h.bins[i]) / static_cast<double>(total),
                     epsilon);
    }
    const double sum =
        std::accumulate(proportions.begin(), proportions.end(), 0.0);
    for (double &p : proportions) {
        p /= sum;
    }
    return proportions;
}

// ================================================================
// 3. PSI (Population Stability Index) (PDF pp. 424-425)
//    Interpretation: < 0.1 little drift, 0.1-0.25 moderate, > 0.25 significant
// ================================================================

double compute_psi(const std::vector<double> &ref,
                   const std::vector<double> &cur) {
    if (ref.size() != cur.size()) {
        throw std::invalid_argument("ref and cur must have the same size");
    }
    double psi = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        psi += (cur[i] - ref[i]) * std::log(cur[i] / ref[i]);
    }
    return psi;
}

double compute_psi(const Histogram &reference,
                   const Histogram &current,
                   double epsilon = 1e-12) {
    if (reference.bins.size() != current.bins.size()) {
        throw std::invalid_argument("Histogram bin counts must match");
    }
    if (reference.min_val != current.min_val || reference.max_val != current.max_val) {
        throw std::invalid_argument("Histogram ranges must match");
    }
    const std::vector<double> ref = to_proportions(reference, epsilon);
    const std::vector<double> cur = to_proportions(current, epsilon);
    return compute_psi(ref, cur);
}

// ================================================================
// 4. KS (Kolmogorov-Smirnov) test (PDF p. 420)
//    Measures max distance between two CDFs: KS = sup|F1(x) - F2(x)|
// ================================================================

double compute_ks(const std::vector<double> &sample_a,
                  const std::vector<double> &sample_b) {
    std::vector<double> a = sample_a;
    std::vector<double> b = sample_b;
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());

    double max_diff = 0.0;
    std::size_t i = 0, j = 0;
    const double n_a = static_cast<double>(a.size());
    const double n_b = static_cast<double>(b.size());

    while (i < a.size() && j < b.size()) {
        double cdf_a = static_cast<double>(i + 1) / n_a;
        double cdf_b = static_cast<double>(j + 1) / n_b;

        if (a[i] <= b[j]) {
            cdf_a = static_cast<double>(i + 1) / n_a;
            cdf_b = static_cast<double>(j) / n_b;
            ++i;
        } else {
            cdf_a = static_cast<double>(i) / n_a;
            cdf_b = static_cast<double>(j + 1) / n_b;
            ++j;
        }
        max_diff = std::max(max_diff, std::abs(cdf_a - cdf_b));
    }
    return max_diff;
}

// ================================================================
// 5. Chi-squared test for categorical distributions (PDF p. 420)
//    chi^2 = sum((O_i - E_i)^2 / E_i)
// ================================================================

double compute_chi_squared(const std::vector<double> &observed,
                           const std::vector<double> &expected) {
    if (observed.size() != expected.size()) {
        throw std::invalid_argument("observed and expected must have the same size");
    }
    double chi2 = 0.0;
    for (std::size_t i = 0; i < observed.size(); ++i) {
        double diff = observed[i] - expected[i];
        if (expected[i] > 0.0) {
            chi2 += (diff * diff) / expected[i];
        }
    }
    return chi2;
}

// ================================================================
// 6. KL divergence (PDF p. 420)
//    D_KL(P||Q) = sum P(i) * log(P(i) / Q(i))
// ================================================================

double compute_kl_divergence(const std::vector<double> &p,
                             const std::vector<double> &q,
                             double epsilon = 1e-12) {
    if (p.size() != q.size()) {
        throw std::invalid_argument("p and q must have the same size");
    }
    double kl = 0.0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        double pi = std::max(p[i], epsilon);
        double qi = std::max(q[i], epsilon);
        kl += pi * std::log(pi / qi);
    }
    return kl;
}

// ================================================================
// 7. JS divergence (PDF p. 420)
//    JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), M = 0.5*(P+Q)
// ================================================================

double compute_js_divergence(const std::vector<double> &p,
                             const std::vector<double> &q) {
    if (p.size() != q.size()) {
        throw std::invalid_argument("p and q must have the same size");
    }
    std::vector<double> m(p.size());
    for (std::size_t i = 0; i < p.size(); ++i) {
        m[i] = 0.5 * (p[i] + q[i]);
    }
    return 0.5 * compute_kl_divergence(p, m) + 0.5 * compute_kl_divergence(q, m);
}

// ================================================================
// 8. Welford's online algorithm (PDF p. 419)
//    Streaming mean and variance in a single pass
// ================================================================

struct WelfordStats {
    int count = 0;
    double mean = 0.0;
    double m2 = 0.0; // sum of squared differences from current mean

    void update(double x) {
        count++;
        double delta = x - mean;
        mean += delta / static_cast<double>(count);
        double delta2 = x - mean;
        m2 += delta * delta2;
    }

    double variance() const {
        return (count > 1) ? m2 / static_cast<double>(count - 1) : 0.0;
    }

    double stddev() const {
        return std::sqrt(variance());
    }
};

// ================================================================
// Main demonstration
// ================================================================

int main() {
    std::cout << "=== Chapter 11: Drift Detection ===\n\n";

    // --- Histogram demo ---
    std::cout << "1. Fixed-bin Histogram (reference distribution)\n";
    Histogram ref_hist(0.0, 100.0, 10);
    std::vector<double> ref_data = {
        5.2, 12.3, 15.7, 18.9, 22.1, 25.4, 28.7, 35.2,
        38.6, 42.1, 45.3, 48.9, 52.0, 55.5, 58.3, 62.7,
        65.1, 68.4, 72.9, 75.0, 78.2, 82.6, 85.1, 88.3, 92.4};
    for (double v : ref_data) update_histogram(ref_hist, v);
    print_histogram(ref_hist);

    // --- Simulate drift: shift values up ---
    std::cout << "\n2. Current histogram (simulated drift: +15 shift)\n";
    Histogram cur_hist(0.0, 100.0, 10);
    std::vector<double> cur_data = {
        18.5, 25.1, 30.2, 33.8, 37.4, 41.0, 44.2, 50.6,
        55.1, 58.3, 62.7, 65.4, 69.8, 73.2, 76.5, 80.1,
        84.3, 88.9, 93.1, 96.4, 28.7, 35.9, 42.3, 48.8, 54.2};
    for (double v : cur_data) update_histogram(cur_hist, v);
    print_histogram(cur_hist);

    // --- PSI ---
    double psi = compute_psi(ref_hist, cur_hist);
    std::cout << "\n3. PSI (Population Stability Index): " << psi << "\n";
    std::cout << "   Interpretation: ";
    if (psi < 0.1)
        std::cout << "little or no drift";
    else if (psi < 0.25)
        std::cout << "moderate drift";
    else
        std::cout << "significant drift";
    std::cout << "\n";

    // --- KS test ---
    std::cout << "\n4. KS test statistic: "
              << compute_ks(ref_data, cur_data) << "\n";
    std::cout << "   (max distance between CDFs; > 0.3 suggests drift)\n";

    // --- Chi-squared ---
    std::cout << "\n5. Chi-squared test\n";
    std::vector<double> ref_props = to_proportions(ref_hist);
    std::vector<double> cur_props = to_proportions(cur_hist);
    // Scale proportions to expected/observed counts
    std::vector<double> observed(cur_props.size());
    std::vector<double> expected(ref_props.size());
    double n_total = 100.0; // hypothetical total count
    for (std::size_t i = 0; i < ref_props.size(); ++i) {
        expected[i] = ref_props[i] * n_total;
        observed[i] = cur_props[i] * n_total;
    }
    double chi2 = compute_chi_squared(observed, expected);
    std::cout << "   chi^2 = " << chi2 << "\n";

    // --- KL and JS divergence ---
    std::cout << "\n6. Divergence measures\n";
    std::cout << "   KL(ref||cur) = " << compute_kl_divergence(ref_props, cur_props) << "\n";
    std::cout << "   JS(ref,cur)  = " << compute_js_divergence(ref_props, cur_props) << "\n";

    // --- Welford streaming stats ---
    std::cout << "\n7. Welford streaming statistics\n";
    WelfordStats welford;
    for (double v : cur_data) welford.update(v);
    std::cout << "   count=" << welford.count
              << " mean=" << welford.mean
              << " stddev=" << welford.stddev() << "\n";

    // --- Alert simulation with hysteresis ---
    std::cout << "\n8. Alert hysteresis simulation\n";
    double threshold = 0.25;
    int alert_countdown = 0;
    const int HYSTERESIS_WINDOW = 3;
    std::vector<double> psi_history = {0.08, 0.12, 0.28, 0.31, 0.29, 0.32, 0.15, 0.10};

    for (double p : psi_history) {
        if (p > threshold) {
            alert_countdown++;
            if (alert_countdown >= HYSTERESIS_WINDOW) {
                std::cout << "  PSI=" << p << " -> ALERT (sustained drift)\n";
            } else {
                std::cout << "  PSI=" << p << " -> accumulating (" << alert_countdown
                          << "/" << HYSTERESIS_WINDOW << ")\n";
            }
        } else {
            alert_countdown = 0;
            std::cout << "  PSI=" << p << " -> clear\n";
        }
    }

    std::cout << "\n=== Drift detection demo complete ===\n";
    return 0;
}
