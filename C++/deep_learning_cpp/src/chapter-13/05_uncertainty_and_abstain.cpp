/*
 * uncertainty_and_abstain.cpp
 * Chapter 13: Explainability and Transparency
 *
 * Responsible explainability requires honest communication of
 * uncertainty. This file covers:
 *   - Calibrated probabilities (temperature scaling proxy)
 *   - Confidence intervals and entropy-based uncertainty
 *   - OOD (out-of-distribution) detection via distance to background
 *   - Abstain gateway: when to say "I don't know"
 *   - Human review routing
 *
 * PDF pages: 543-544 (book pp. 543-544)
 *
 * Key principle: "I don't know" is a feature, not a bug.
 */

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// ================================================================
// 1. Uncertainty quantification
// ================================================================

enum class UncertaintyLevel { LOW,
                              MODERATE,
                              HIGH };

std::string uncertainty_str(UncertaintyLevel u) {
    switch (u) {
    case UncertaintyLevel::LOW: return "low";
    case UncertaintyLevel::MODERATE: return "moderate";
    case UncertaintyLevel::HIGH: return "high";
    default: return "unknown";
    }
}

// ================================================================
// 2. Calibrated probability (temperature scaling proxy)
// ================================================================

double calibrated_score(double raw_logit, double temperature = 1.0) {
    // Temperature scaling: softmax with temperature
    // temperature = 1.0 → no scaling; > 1.0 → smoother (reduces overconfidence)
    return 1.0 / (1.0 + std::exp(-raw_logit / temperature));
}

// ================================================================
// 3. Uncertainty from logits
// ================================================================

struct UncertaintyMetrics {
    double entropy;             // higher = more uncertain
    double top1_top2_margin;    // smaller = less certain
    double predictive_variance; // for regression/ensemble

    UncertaintyLevel classify(double entropy_threshold_med = 0.5,
                              double entropy_threshold_high = 1.0) const {
        if (entropy > entropy_threshold_high) return UncertaintyLevel::HIGH;
        if (entropy > entropy_threshold_med) return UncertaintyLevel::MODERATE;
        return UncertaintyLevel::LOW;
    }
};

UncertaintyMetrics compute_uncertainty(const std::vector<double> &logits) {
    // Softmax
    double max_logit = *std::max_element(logits.begin(), logits.end());
    double sum_exp = 0.0;
    std::vector<double> probs(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum_exp += probs[i];
    }
    for (auto &p : probs) p /= sum_exp;

    // Entropy
    double entropy = 0.0;
    for (double p : probs) {
        if (p > 0) entropy -= p * std::log2(p);
    }

    // Top-2 margin
    double top1 = 0.0, top2 = 0.0;
    for (double p : probs) {
        if (p > top1) {
            top2 = top1;
            top1 = p;
        } else if (p > top2) {
            top2 = p;
        }
    }
    double margin = top1 - top2;

    return {entropy, margin, 0.0};
}

// ================================================================
// 4. OOD detection via distance to background set
// ================================================================

struct OODDetector {
    std::vector<double> bg_distances; // distances from bg set to its centroid
    double threshold;                 // e.g., 95th percentile of bg distances

    void fit(const std::vector<std::vector<double>> &background_feats) {
        if (background_feats.empty()) return;
        size_t d = background_feats[0].size();

        // Compute centroid
        std::vector<double> centroid(d, 0.0);
        for (const auto &f : background_feats) {
            for (size_t j = 0; j < d; ++j) centroid[j] += f[j];
        }
        for (auto &c : centroid) c /= background_feats.size();

        // Compute distances to centroid
        bg_distances.clear();
        for (const auto &f : background_feats) {
            double dist2 = 0.0;
            for (size_t j = 0; j < d; ++j) {
                double diff = f[j] - centroid[j];
                dist2 += diff * diff;
            }
            bg_distances.push_back(std::sqrt(dist2));
        }

        // Threshold: 95th percentile of background distances
        std::sort(bg_distances.begin(), bg_distances.end());
        size_t idx = static_cast<size_t>(0.95 * bg_distances.size());
        if (idx >= bg_distances.size()) idx = bg_distances.size() - 1;
        threshold = bg_distances[idx];
    }

    bool is_ood(const std::vector<double> &features) const {
        if (bg_distances.empty()) return false;
        // Compute centroid (simplified: reuse from fit; here we recompute)
        // For demo, use a simple distance-to-threshold check
        double dist2 = 0.0;
        for (double v : features) dist2 += v * v; // simplified distance
        return std::sqrt(dist2) > threshold * 1.5;
    }
};

// ================================================================
// 5. Abstain gateway
// ================================================================

struct AbstainPolicy {
    double ood_threshold = 0.3;
    double entropy_high = 1.0;
    double margin_low = 0.10;
    double calibrated_score_low = 0.55;
    bool missing_critical_features = false;

    bool should_abstain(const UncertaintyMetrics &um,
                        bool is_ood,
                        double calibrated_prob,
                        bool critical_features_present) const {
        // Always abstain if OOD
        if (is_ood) return true;

        // Always abstain if missing critical features
        if (!critical_features_present) return true;

        // Abstain if uncertainty is HIGH
        if (um.classify(0.5, 1.0) == UncertaintyLevel::HIGH) return true;

        // Abstain if calibrated score is borderline with moderate uncertainty
        if (calibrated_prob < calibrated_score_low && um.classify(0.5, 1.0) == UncertaintyLevel::MODERATE) {
            return true;
        }

        // Abstain if top-2 margin is very small (model is confused)
        if (um.top1_top2_margin < margin_low) return true;

        return false;
    }
};

// ================================================================
// 6. Decision result with uncertainty
// ================================================================

struct DecisionResult {
    std::string decision; // "approve", "deny", "abstain"
    double calibrated_score;
    UncertaintyLevel uncertainty;
    bool is_ood;
    std::string reason;
    std::string audit_ref;
};

void print_decision(const DecisionResult &r) {
    std::cout << "  Decision: " << r.decision << "\n";
    std::cout << "  Score: " << std::fixed << std::setprecision(3)
              << r.calibrated_score
              << " (uncertainty: " << uncertainty_str(r.uncertainty) << ")\n";
    if (r.is_ood) std::cout << "  *** OOD DETECTED ***\n";
    if (r.decision == "abstain") {
        std::cout << "  *** ROUTING TO HUMAN REVIEW ***\n";
    }
    std::cout << "  Reason: " << r.reason << "\n";
    std::cout << "  Audit ref: " << r.audit_ref << "\n";
}

// ================================================================
// Main demonstration
// ================================================================

int main() {
    std::cout << "=== Chapter 13: Uncertainty & Abstain ===\n\n";

    // --- Uncertainty from logits ---
    std::cout << "1. Uncertainty from model logits\n\n";

    struct TestCase {
        std::string label;
        std::vector<double> logits;
    };

    std::vector<TestCase> cases = {
        {"High confidence, low entropy",
         {5.0, 0.5, 0.2, -1.0, -2.0}},
        {"Medium confidence, moderate entropy",
         {2.0, 1.8, 1.5, -0.5, -1.0}},
        {"Low confidence, high entropy (near uniform)",
         {0.5, 0.5, 0.5, 0.5, 0.5}},
        {"Overconfident but wrong (ambiguous)",
         {4.0, 3.95, 0.1, -0.5, -1.0}},
    };

    std::cout << "  " << std::setw(42) << "Case"
              << std::setw(10) << "Entropy"
              << std::setw(10) << "Margin"
              << std::setw(14) << "Uncertainty" << "\n";
    std::cout << "  " << std::string(76, '-') << "\n";
    for (const auto &tc : cases) {
        auto um = compute_uncertainty(tc.logits);
        std::cout << "  " << std::setw(42) << tc.label
                  << std::setw(10) << std::fixed << std::setprecision(4) << um.entropy
                  << std::setw(10) << um.top1_top2_margin
                  << std::setw(14) << uncertainty_str(um.classify(0.5, 1.0)) << "\n";
    }

    // --- Abstain gateway ---
    std::cout << "\n2. Abstain gateway scenarios\n";

    AbstainPolicy policy;
    int audit_counter = 1000;

    auto evaluate_case = [&](const std::string &name,
                             const std::vector<double> &logits,
                             bool is_ood,
                             double calibrated_prob,
                             bool critical_ok) {
        auto um = compute_uncertainty(logits);
        bool abstain = policy.should_abstain(um, is_ood, calibrated_prob, critical_ok);

        DecisionResult r;
        r.calibrated_score = calibrated_prob;
        r.uncertainty = um.classify(0.5, 1.0);
        r.is_ood = is_ood;
        r.audit_ref = "REQ-" + std::to_string(++audit_counter);

        if (abstain) {
            r.decision = "abstain";
            if (is_ood)
                r.reason = "OOD detected; input outside training distribution";
            else if (!critical_ok)
                r.reason = "Missing critical features";
            else if (um.entropy > 1.0)
                r.reason = "High prediction entropy";
            else
                r.reason = "Score near decision boundary with high uncertainty";
        } else {
            r.decision = (calibrated_prob > 0.6) ? "approve" : "deny";
            r.reason = "Within confidence thresholds";
        }

        std::cout << "\n  [" << name << "]\n";
        print_decision(r);
    };

    evaluate_case("Normal loan",
                  {3.0, 0.5, -1.0, -2.0}, false, 0.82, true);
    evaluate_case("Borderline credit",
                  {0.8, 0.7, 0.1, -0.2}, false, 0.58, true);
    evaluate_case("OOD input",
                  {1.0, 0.5, -0.5}, true, 0.65, true);
    evaluate_case("Missing income field",
                  {2.0, 1.0, -0.5}, false, 0.72, false);

    // --- Communicating uncertainty ---
    std::cout << "\n3. Uncertainty communication guidelines\n";
    std::cout << "   - Prefer calibrated probabilities to raw scores.\n";
    std::cout << "   - Provide confidence intervals where applicable.\n";
    std::cout << "   - Mark 'unknown/abstain' explicitly, route to human review.\n";
    std::cout << "   - Use plain language: 'The model estimates 0.73 (±0.08)'\n";
    std::cout << "   - Avoid wording that suggests inevitability.\n";

    // --- Abstain triggers checklist ---
    std::cout << "\n4. Abstain triggers checklist\n";
    std::cout << "   [ ] High predictive entropy (model is confused)\n";
    std::cout << "   [ ] OOD detector above threshold\n";
    std::cout << "   [ ] Missing or invalid critical features\n";
    std::cout << "   [ ] Calibration drift beyond SLO\n";
    std::cout << "   [ ] Champion-challenger strong disagreement\n";
    std::cout << "   [ ] Score near decision boundary with high uncertainty\n";

    std::cout << "\n=== Uncertainty & abstain demo complete ===\n";
    return 0;
}
