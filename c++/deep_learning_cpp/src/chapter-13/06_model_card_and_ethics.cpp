/*
 * model_card_and_ethics.cpp
 * Chapter 13: Explainability and Transparency
 *
 * Ethical considerations for explainable AI in production:
 *   - Model card (documented limitations, blind spots, operations)
 *   - Audit trail (immutable decision logs)
 *   - Reason codes (human-readable, no PII, no raw weights)
 *   - Human override policy (escalation, review queue, feedback loop)
 *   - Anti-gaming and privacy guardrails
 *
 * PDF pages: 544-547 (book pp. 544-547)
 */

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// ================================================================
// 1. Model card (PDF p. 545)
// ================================================================

struct ModelCard {
    std::string model_name;
    std::string model_version;
    std::string intended_use;
    std::string training_data_vintage;
    std::vector<std::string> known_blind_spots;
    std::vector<std::string> fairness_considerations;
    struct OpLimits {
        int input_height, input_width;
        int max_batch_size;
        double max_latency_ms;
        double max_memory_gb;
    } op_limits;
    struct CalibInfo {
        double ece;
        double brier;
        std::string calibration_date;
    } calibration;
    std::string last_updated;

    void print() const {
        std::cout << "\n  === Model Card: " << model_name
                  << " v" << model_version << " ===\n";
        std::cout << "  Intended use: " << intended_use << "\n";
        std::cout << "  Training data: " << training_data_vintage << "\n";

        std::cout << "  Known blind spots:\n";
        for (const auto &bs : known_blind_spots) {
            std::cout << "    - " << bs << "\n";
        }

        std::cout << "  Fairness considerations:\n";
        for (const auto &f : fairness_considerations) {
            std::cout << "    - " << f << "\n";
        }

        std::cout << "  Operational limits:\n";
        std::cout << "    - Input: " << op_limits.input_height << "x"
                  << op_limits.input_width << "\n";
        std::cout << "    - Max batch: " << op_limits.max_batch_size << "\n";
        std::cout << "    - Max latency: " << op_limits.max_latency_ms << "ms\n";
        std::cout << "    - Max memory: " << op_limits.max_memory_gb << "GB\n";

        std::cout << "  Calibration (ECE=" << calibration.ece
                  << ", Brier=" << calibration.brier
                  << ") as of " << calibration.calibration_date << "\n";
        std::cout << "  Last updated: " << last_updated << "\n";
    }
};

// ================================================================
// 2. Immutable audit log entry (PDF pp. 546)
// ================================================================

struct AuditEntry {
    std::string timestamp;
    std::string request_id;
    std::string model_version;
    std::string preprocessing_version;
    std::string input_hash; // SHA-256 or similar
    double prediction;
    double calibrated_score;
    std::string uncertainty_level;
    std::vector<std::string> reason_codes;
    std::string explanation_hash; // hash of SHAP/LIME/Grad-CAM artifact
    std::string final_action;     // "auto", "human_override", "abstain"
    std::string reviewer_id;      // empty if auto

    void print() const {
        std::cout << "  [" << timestamp << "] req=" << request_id
                  << " model=" << model_version
                  << " pred=" << std::fixed << std::setprecision(3) << prediction
                  << " calib=" << calibrated_score
                  << " uncertainty=" << uncertainty_level
                  << " action=" << final_action << "\n";
        if (!reason_codes.empty()) {
            std::cout << "    reasons: ";
            for (size_t i = 0; i < reason_codes.size(); ++i) {
                std::cout << reason_codes[i];
                if (i < reason_codes.size() - 1) std::cout << ", ";
            }
            std::cout << "\n";
        }
        if (!reviewer_id.empty()) {
            std::cout << "    reviewer: " << reviewer_id << "\n";
        }
    }
};

// ================================================================
// 3. Reason code system (PDF pp. 545-546)
//    Coarse, policy-based reasons — not raw model weights
// ================================================================

struct ReasonCode {
    std::string code;
    std::string description; // plain language
    std::string guidance;    // what user can do
};

std::vector<ReasonCode> standard_reason_codes = {
    {"INSUFFICIENT_HISTORY", "Limited payment history",
     "Continue making on-time payments to build history."},
    {"HIGH_UTILIZATION", "Credit utilization ratio is high",
     "Reducing balances may improve future outcomes."},
    {"RECENT_DELINQUENCY", "Recent missed payment(s)",
     "Consistent payments over time will strengthen your profile."},
    {"SHORT_EMPLOYMENT", "Employment length below typical threshold",
     "Stability in current position may help over time."},
    {"MISSING_INFORMATION", "Required information was not provided",
     "Complete all requested fields for full evaluation."},
};

// ================================================================
// 4. Human override policy (PDF pp. 545-546)
// ================================================================

struct OverridePolicy {
    std::string escalation_path; // queue name or email
    int max_review_time_hours;
    bool require_second_reviewer_for_override;
    std::vector<std::string> auto_escalation_triggers; // OOD, high uncertainty

    void print() const {
        std::cout << "  Escalation path: " << escalation_path << "\n";
        std::cout << "  Max review time: " << max_review_time_hours << " hours\n";
        std::cout << "  Second reviewer required for override: "
                  << (require_second_reviewer_for_override ? "yes" : "no") << "\n";
        std::cout << "  Auto-escalation triggers:\n";
        for (const auto &t : auto_escalation_triggers) {
            std::cout << "    - " << t << "\n";
        }
    }
};

// ================================================================
// 5. Anti-gaming guardrails (PDF pp. 544)
//    Never expose raw model weights or exact thresholds publicly
// ================================================================

struct ExplanationSanitizer {
    // Convert raw feature weights to coarse, policy-based reasons
    std::vector<std::string> sanitize_public_reasons(
        const std::vector<std::string> &feature_names,
        const std::vector<double> &feature_weights,
        double threshold = 0.05) const {
        std::vector<std::string> reasons;
        for (size_t i = 0; i < feature_names.size(); ++i) {
            if (std::fabs(feature_weights[i]) > threshold) {
                // Map raw feature to coarse reason category
                reasons.push_back(map_to_reason(feature_names[i]));
            }
        }
        if (reasons.empty()) {
            reasons.push_back("Based on overall credit profile assessment");
        }
        return reasons;
    }

private:
    std::string map_to_reason(const std::string &feature) const {
        // In production: maintain a approved mapping table
        if (feature.find("payment") != std::string::npos || feature.find("delinquen") != std::string::npos)
            return "payment history";
        if (feature.find("utilization") != std::string::npos || feature.find("balance") != std::string::npos)
            return "account utilization";
        if (feature.find("income") != std::string::npos || feature.find("debt") != std::string::npos)
            return "debt-to-income assessment";
        if (feature.find("tenure") != std::string::npos || feature.find("employment") != std::string::npos)
            return "history length";
        return "overall credit assessment";
    }
};

// ================================================================
// Main demonstration
// ================================================================

int main() {
    std::cout << "=== Chapter 13: Model Card & Ethics ===\n\n";

    // --- Model card ---
    std::cout << "1. Model card (versioned, in source control)\n";
    ModelCard card;
    card.model_name = "CreditRiskEvaluator";
    card.model_version = "3.14";
    card.intended_use = "Evaluate consumer credit applications for personal loans ($1K-$50K)";
    card.training_data_vintage = "2024-Q1 through 2025-Q2, US consumer credit bureau data";
    card.known_blind_spots = {
        "Thin files (less than 6 months of credit history)",
        "Recent immigrants without domestic credit records",
        "Ultra-high-net-worth individuals (tail distribution)",
        "Non-traditional income sources (gig economy, freelancing)"};
    card.fairness_considerations = {
        "Age split: ECE difference < 2% across age bins 18-25, 26-40, 41-60, 60+",
        "Region: performance monitored by 4 census regions; alert if p95 latency differs > 15ms",
        "Gender: not used as a feature; proxy leakage monitored quarterly"};
    card.op_limits = {224, 224, 128, 35.0, 3.0};
    card.calibration = {0.018, 0.067, "2025-09-01"};
    card.last_updated = "2025-09-15";
    card.print();

    // --- Audit trail ---
    std::cout << "\n2. Immutable audit log\n";
    std::vector<AuditEntry> audit_log;

    AuditEntry entry1;
    entry1.timestamp = "2025-10-05T14:23:01Z";
    entry1.request_id = "REQ-2025-10-05-1234";
    entry1.model_version = "3.14";
    entry1.preprocessing_version = "s2.7";
    entry1.input_hash = "a1b2c3...";
    entry1.prediction = 0.71;
    entry1.calibrated_score = 0.68;
    entry1.uncertainty_level = "moderate";
    entry1.reason_codes = {"INSUFFICIENT_HISTORY", "HIGH_UTILIZATION"};
    entry1.final_action = "auto";
    audit_log.push_back(entry1);

    AuditEntry entry2;
    entry2.timestamp = "2025-10-05T15:01:22Z";
    entry2.request_id = "REQ-2025-10-05-1235";
    entry2.model_version = "3.14";
    entry2.preprocessing_version = "s2.7";
    entry2.input_hash = "d4e5f6...";
    entry2.prediction = 0.58;
    entry2.calibrated_score = 0.55;
    entry2.uncertainty_level = "high";
    entry2.reason_codes = {"HIGH_UNCERTAINTY"};
    entry2.final_action = "abstain";
    audit_log.push_back(entry2);

    AuditEntry entry3;
    entry3.timestamp = "2025-10-05T16:45:10Z";
    entry3.request_id = "REQ-2025-10-05-1235";
    entry3.model_version = "3.14";
    entry3.preprocessing_version = "s2.7";
    entry3.input_hash = "d4e5f6...";
    entry3.prediction = 0.58;
    entry3.calibrated_score = 0.55;
    entry3.uncertainty_level = "high";
    entry3.reason_codes = {"HIGH_UNCERTAINTY", "OVERRIDDEN_TO_APPROVE"};
    entry3.final_action = "human_override";
    entry3.reviewer_id = "reviewer-42";
    audit_log.push_back(entry3);

    for (const auto &e : audit_log) e.print();

    // --- Reason codes ---
    std::cout << "\n3. Standard reason codes (user-facing, no raw weights)\n";
    for (const auto &rc : standard_reason_codes) {
        std::cout << "  [" << rc.code << "] " << rc.description << "\n";
        std::cout << "    Guidance: " << rc.guidance << "\n";
    }

    // --- Anti-gaming sanitization ---
    std::cout << "\n4. Explanation sanitization (anti-gaming)\n";

    ExplanationSanitizer sanitizer;
    std::vector<std::string> internal_features = {
        "payment_history_score", "utilization_ratio",
        "debt_to_income", "credit_tenure_months",
        "recent_inquiry_count"};
    std::vector<double> internal_weights = {0.31, 0.22, 0.18, -0.09, 0.07};

    auto public_reasons = sanitizer.sanitize_public_reasons(
        internal_features, internal_weights, 0.05);

    std::cout << "  Internal (SHAP):\n";
    for (size_t i = 0; i < internal_features.size(); ++i) {
        std::cout << "    " << internal_features[i] << ": "
                  << std::showpos << internal_weights[i] << "\n";
    }
    std::cout << "\n  Public (coarse reasons, no raw weights):\n";
    for (const auto &r : public_reasons) {
        std::cout << "    - " << r << "\n";
    }

    // --- Override policy ---
    std::cout << "\n5. Human override policy\n";
    OverridePolicy override_pol;
    override_pol.escalation_path = "review-queue: credit-escalation";
    override_pol.max_review_time_hours = 4;
    override_pol.require_second_reviewer_for_override = true;
    override_pol.auto_escalation_triggers = {
        "OOD detected", "entropy > 1.0", "ECE drift > 5%",
        "missing critical fields", "adverse action flag in policy"};
    override_pol.print();

    // --- Ethics checklist ---
    std::cout << "\n6. Ethics deployment checklist\n";
    std::cout << "  [ ] Model card published and versioned\n";
    std::cout << "  [ ] Data contract documented (units, ranges, schemas)\n";
    std::cout << "  [ ] Known failure modes documented with expected responses\n";
    std::cout << "  [ ] Sanity checks pass (randomized weights destroy attributions)\n";
    std::cout << "  [ ] Abstain gateway active for OOD and high-uncertainty cases\n";
    std::cout << "  [ ] Audit logs are immutable and access-controlled\n";
    std::cout << "  [ ] Public explanations use coarse reason codes (no raw weights)\n";
    std::cout << "  [ ] PII is separated from model features in storage/transport\n";
    std::cout << "  [ ] Human review queue integrated with feedback loop\n";

    std::cout << "\n=== Model card & ethics demo complete ===\n";
    return 0;
}
