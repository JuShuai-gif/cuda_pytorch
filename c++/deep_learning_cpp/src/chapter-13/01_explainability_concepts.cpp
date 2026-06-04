/*
 * explainability_concepts.cpp
 * Chapter 13: Explainability and Transparency in Deep Learning Models
 *
 * Framework for understanding XAI (Explainable AI) in production C++ systems.
 *
 * Three axes of explainability:
 *   1. Local vs Global: one decision vs model-wide behavior
 *   2. Post-hoc vs Intrinsic: analyze trained model vs design for interpretability
 *   3. Model-agnostic vs Model-specific: cross-architecture vs tailored
 *
 * Four stakeholder personas with different needs.
 *
 * PDF pages: 516-519 (book pp. 516-519)
 */

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// ================================================================
// 1. Explainability axes
// ================================================================

enum class Scope { LOCAL,
                   GLOBAL };
enum class Timing { POST_HOC,
                    INTRINSIC };
enum class Genericity { MODEL_AGNOSTIC,
                        MODEL_SPECIFIC };

struct ExplainabilityProfile {
    std::string method;
    Scope scope;
    Timing timing;
    Genericity genericity;
    std::string description;
};

std::string scope_str(Scope s) {
    return (s == Scope::LOCAL) ? "Local" : "Global";
}
std::string timing_str(Timing t) {
    return (t == Timing::POST_HOC) ? "Post-hoc" : "Intrinsic";
}
std::string gen_str(Genericity g) {
    return (g == Genericity::MODEL_AGNOSTIC) ? "Model-agnostic" : "Model-specific";
}

// ================================================================
// 2. Stakeholder definitions
// ================================================================

struct Stakeholder {
    std::string role;
    std::string needs;
    std::string output_format;
    std::string example;
};

// ================================================================
// 3. Explanation result structure
//    What a production explanation API should return
// ================================================================

struct ExplanationResult {
    std::string decision;
    double calibrated_score;
    std::string uncertainty_level; // "low", "medium", "high"
    bool abstain;
    std::vector<std::string> reason_codes;
    std::vector<std::string> top_features;
    std::vector<double> feature_weights;
    std::string audit_ref;
    double local_r2 = -1.0; // surrogate fit quality
};

void print_explanation(const ExplanationResult &r) {
    std::cout << "  Decision: " << r.decision << "\n";
    std::cout << "  Calibrated score: " << std::fixed << std::setprecision(3)
              << r.calibrated_score << " (uncertainty: " << r.uncertainty_level << ")\n";
    if (r.abstain) {
        std::cout << "  *** ABSTAIN - routing to human review ***\n";
    }
    std::cout << "  Reasons: ";
    for (size_t i = 0; i < r.reason_codes.size(); ++i) {
        std::cout << r.reason_codes[i];
        if (i < r.reason_codes.size() - 1) std::cout << ", ";
    }
    std::cout << "\n  Top features:\n";
    for (size_t i = 0; i < r.top_features.size(); ++i) {
        std::cout << "    " << std::setw(20) << r.top_features[i]
                  << ": " << std::showpos << r.feature_weights[i] << "\n";
    }
    if (r.local_r2 >= 0) {
        std::cout << "  Local R² (surrogate fit): " << r.local_r2 << "\n";
    }
    std::cout << "  Audit ref: " << r.audit_ref << "\n";
}

// ================================================================
// Main demonstration
// ================================================================

int main() {
    std::cout << "=== Chapter 13: Explainability Concepts ===\n\n";

    // --- Three axes ---
    std::cout << "1. Three Explainability Axes\n\n";

    std::vector<ExplainabilityProfile> methods = {
        {"LIME", Scope::LOCAL, Timing::POST_HOC, Genericity::MODEL_AGNOSTIC,
         "Local surrogate fits around a single prediction; "
         "works with any model type by treating it as a black box."},
        {"SHAP", Scope::LOCAL, Timing::POST_HOC, Genericity::MODEL_AGNOSTIC,
         "Shapley values from game theory; additive attributions "
         "guarantee local accuracy, consistency, and symmetry."},
        {"Grad-CAM", Scope::LOCAL, Timing::POST_HOC, Genericity::MODEL_SPECIFIC,
         "Class-specific heatmap from CNN convolutional feature maps; "
         "shows where the network focused, not why in feature space."},
        {"Decision Tree", Scope::GLOBAL, Timing::INTRINSIC, Genericity::MODEL_SPECIFIC,
         "Inherently interpretable if depth is bounded; "
         "can serve as a surrogate for a more complex model."},
    };

    std::cout << "  " << std::setw(14) << "Method"
              << std::setw(10) << "Scope"
              << std::setw(12) << "Timing"
              << std::setw(16) << "Genericity" << "\n";
    std::cout << "  " << std::string(52, '-') << "\n";
    for (const auto &m : methods) {
        std::cout << "  " << std::setw(14) << m.method
                  << std::setw(10) << scope_str(m.scope)
                  << std::setw(12) << timing_str(m.timing)
                  << std::setw(16) << gen_str(m.genericity) << "\n";
        std::cout << "    " << m.description << "\n";
    }

    // --- Stakeholder needs ---
    std::cout << "\n2. Stakeholder Needs\n\n";

    std::vector<Stakeholder> stakeholders = {
        {"Clinician", "Case-level, domain features, calibrated risk",
         "Saliency map + confidence interval + OOD flag",
         "Which lung region drove the pneumonia score?"},
        {"Auditor/Compliance", "Model card, fairness, decision logs",
         "Global calibration + cohort breakdown",
         "Is the model unbiased across age groups?"},
        {"End User", "Short, plain language, actionable",
         "Top-3 reasons + improvement suggestions",
         "Why was my loan denied?"},
        {"Engineer/Operator", "Faithful representation, debug info",
         "Attribution patterns + disagreement + drift signals",
         "Which feature is causing production failures?"},
    };

    for (const auto &s : stakeholders) {
        std::cout << "  [" << s.role << "]\n";
        std::cout << "    Needs: " << s.needs << "\n";
        std::cout << "    Format: " << s.output_format << "\n";
        std::cout << "    Example: " << s.example << "\n\n";
    }

    // --- Demo production explanation ---
    std::cout << "3. Production Explanation API Output\n\n";

    ExplanationResult credit_example;
    credit_example.decision = "Deny";
    credit_example.calibrated_score = 0.71;
    credit_example.uncertainty_level = "moderate";
    credit_example.abstain = false;
    credit_example.reason_codes = {"INSUFFICIENT_HISTORY", "HIGH_UTILIZATION"};
    credit_example.top_features = {
        "debt_to_income_ratio", "recent_delinquencies",
        "credit_tenure_years", "utilization_ratio",
        "num_inquiries_6m", "oldest_account_age"};
    credit_example.feature_weights = {
        0.24, 0.18, -0.07, 0.15, 0.09, -0.03};
    credit_example.local_r2 = 0.82;
    credit_example.audit_ref = "REQ-2025-10-05-1234";

    print_explanation(credit_example);

    // --- Abstain scenario ---
    std::cout << "\n4. Abstain Scenario (High Uncertainty)\n\n";

    ExplanationResult abstain_example;
    abstain_example.decision = "abstain";
    abstain_example.calibrated_score = 0.58;
    abstain_example.uncertainty_level = "high";
    abstain_example.abstain = true;
    abstain_example.reason_codes = {"HIGH_UNCERTAINTY", "OOD_DETECTED"};
    abstain_example.top_features = {
        "income_to_debt", "employment_length"};
    abstain_example.feature_weights = {0.12, -0.09};
    abstain_example.audit_ref = "REQ-2025-10-05-1235";

    print_explanation(abstain_example);

    // --- Design principles ---
    std::cout << "\n5. Design Principles\n";
    std::cout << "  1. Explanations must fit latency and privacy budgets.\n";
    std::cout << "     - Compute-heavy perturbation methods need caching or async.\n";
    std::cout << "     - Never leak sensitive fields; redact and aggregate.\n";
    std::cout << "  2. Prefer consistency over cleverness.\n";
    std::cout << "     - Explanations should be stable under small input changes.\n";
    std::cout << "     - If attributions jump wildly, they lose trust/utility.\n";
    std::cout << "  3. Tie explanations to SLOs.\n";
    std::cout << "     - 'Fast mode' for online UI, 'full mode' for offline audit.\n";

    std::cout << "\n=== Explainability concepts demo complete ===\n";
    return 0;
}
