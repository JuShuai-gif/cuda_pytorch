/*
 * retraining_workflow.cpp
 * Chapter 11: Debugging and Retraining Deployed Models
 *
 * Retraining should be triggered by evidence, implemented through
 * reproducible pipelines, evaluated under real serving constraints,
 * and promoted gradually through controlled rollouts.
 *
 * This file covers:
 *   - Decision flow: threshold/calibration update vs. full retrain
 *   - Acceptance gate: YAML-like evaluation criteria
 *   - Progressive rollout ladder: shadow -> 5% -> 25% -> 50% -> 100%
 *   - Rollback guardrails with automatic trigger conditions
 *   - Closing the loop after promotion
 *
 * PDF pages: 445-450 (book pp. 445-450)
 */

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// ================================================================
// 1. Retraining decision flow (PDF pp. 445-446)
//    Distinguishes threshold/calibration updates from full retrain
// ================================================================

enum class DriftType {
    NONE,
    PRIOR_SHIFT,      // p(y) changed
    COVARIATE_DRIFT,  // p(x) changed, pipeline issue
    COVARIATE_NO_BUG, // p(x) changed, no pipeline bug
    PREDICTION_DRIFT, // model output distribution changed
    CONCEPT_DRIFT     // p(y|x) changed
};

enum class DriftResponse {
    NO_ACTION,
    ADJUST_THRESHOLD,    // Re-tune decision thresholds
    RECALIBRATE,         // Temperature/Platt scaling
    FIX_PIPELINE,        // Fix normalization/vocab/schema
    RETRAIN,             // Retrain on recent data
    RETRAIN_NEW_FEATURES // Retrain with feature redesign
};

DriftResponse decide_response(DriftType drift,
                              bool pipeline_ok,
                              bool calibration_ok,
                              double psi_value) {
    switch (drift) {
    case DriftType::PRIOR_SHIFT:
        if (calibration_ok)
            return DriftResponse::ADJUST_THRESHOLD;
        else
            return DriftResponse::RECALIBRATE;

    case DriftType::COVARIATE_DRIFT:
        return DriftResponse::FIX_PIPELINE;

    case DriftType::COVARIATE_NO_BUG:
        return DriftResponse::RETRAIN;

    case DriftType::PREDICTION_DRIFT:
        if (calibration_ok)
            return DriftResponse::NO_ACTION; // monitor
        else
            return DriftResponse::RECALIBRATE;

    case DriftType::CONCEPT_DRIFT:
        return (psi_value > 0.3) ? DriftResponse::RETRAIN_NEW_FEATURES : DriftResponse::RETRAIN;

    default:
        return DriftResponse::NO_ACTION;
    }
}

std::string response_name(DriftResponse r) {
    switch (r) {
    case DriftResponse::NO_ACTION: return "NO_ACTION";
    case DriftResponse::ADJUST_THRESHOLD: return "ADJUST_THRESHOLD";
    case DriftResponse::RECALIBRATE: return "RECALIBRATE";
    case DriftResponse::FIX_PIPELINE: return "FIX_PIPELINE";
    case DriftResponse::RETRAIN: return "RETRAIN";
    case DriftResponse::RETRAIN_NEW_FEATURES: return "RETRAIN_NEW_FEATURES";
    default: return "UNKNOWN";
    }
}

// ================================================================
// 2. Acceptance gate (PDF p. 448)
//    Encodes offline quality and systems performance criteria
//    Only candidates that pass both blocks proceed to online testing
// ================================================================

struct AcceptanceGate {
    // Offline quality gates
    double auc_delta_min;   // >= champion - X
    double logloss_pct_max; // <= champion + X%
    double ece_pct_max;     // <= champion + X%

    // Systems performance gates (target hardware: A10G)
    double p95_ms_max;
    double max_mem_gb;
    double load_ms_max;
    bool throughput_must_match; // >= champion QPS

    // Results
    struct Result {
        bool offline_pass = false;
        bool systems_pass = false;
        std::string reject_reason;

        bool pass() const {
            return offline_pass && systems_pass;
        }
    };

    Result evaluate(double candidate_auc, double champion_auc,
                    double candidate_logloss, double champion_logloss,
                    double candidate_ece, double champion_ece,
                    double candidate_p95_ms,
                    double candidate_mem_gb,
                    double candidate_load_ms,
                    double candidate_qps, double champion_qps) const {
        Result r;
        std::ostringstream reasons;

        // Offline checks
        bool auc_ok = (candidate_auc >= champion_auc + auc_delta_min);
        bool logloss_ok = (candidate_logloss <= champion_logloss * (1.0 + logloss_pct_max / 100.0));
        bool ece_ok = (candidate_ece <= champion_ece * (1.0 + ece_pct_max / 100.0));
        r.offline_pass = auc_ok && logloss_ok && ece_ok;

        if (!auc_ok) reasons << "auc_delta=" << (candidate_auc - champion_auc)
                             << " < " << auc_delta_min << "; ";
        if (!logloss_ok) reasons << "logloss regressed; ";
        if (!ece_ok) reasons << "ece regressed; ";

        // Systems checks
        bool p95_ok = (candidate_p95_ms <= p95_ms_max);
        bool mem_ok = (candidate_mem_gb <= max_mem_gb);
        bool load_ok = (candidate_load_ms <= load_ms_max);
        bool qps_ok = (!throughput_must_match) || (candidate_qps >= champion_qps);
        r.systems_pass = p95_ok && mem_ok && load_ok && qps_ok;

        if (!p95_ok) reasons << "p95=" << candidate_p95_ms
                             << "ms > " << p95_ms_max << "ms; ";
        if (!mem_ok) reasons << "mem=" << candidate_mem_gb
                             << "GB > " << max_mem_gb << "GB; ";
        if (!load_ok) reasons << "load=" << candidate_load_ms
                              << "ms > " << load_ms_max << "ms; ";
        if (!qps_ok) reasons << "throughput below champion; ";

        r.reject_reason = reasons.str();
        return r;
    }
};

// ================================================================
// 3. Progressive rollout ladder (PDF pp. 449-450)
//    Shadow -> 5% -> 25% -> 50% -> 100% with guardrails at each gate
// ================================================================

struct Guardrail {
    double p95_latency_ms_max;
    double error_rate_pct_max;
    double kpi_delta_pct_max; // Domain KPI regression tolerance

    bool check(double p95_ms, double err_pct, double kpi_delta_pct) const {
        return p95_ms <= p95_latency_ms_max && err_pct <= error_rate_pct_max && kpi_delta_pct <= kpi_delta_pct_max;
    }
};

enum class RolloutStage {
    SHADOW,
    CANARY_5,
    CANARY_25,
    CANARY_50,
    FULL,
    ROLLED_BACK
};

std::string stage_name(RolloutStage s) {
    switch (s) {
    case RolloutStage::SHADOW: return "SHADOW (0%, mirror)";
    case RolloutStage::CANARY_5: return "CANARY 5%";
    case RolloutStage::CANARY_25: return "CANARY 25%";
    case RolloutStage::CANARY_50: return "CANARY 50%";
    case RolloutStage::FULL: return "FULL (100%)";
    case RolloutStage::ROLLED_BACK: return "ROLLED BACK";
    default: return "UNKNOWN";
    }
}

struct RolloutLadder {
    Guardrail guardrail;
    RolloutStage current_stage = RolloutStage::SHADOW;
    int healthy_windows = 0;
    const int HOLD_WINDOWS = 3; // Must pass N windows before advancing

    // Simulate one monitoring window
    // Returns true if we should advance to the next stage
    bool evaluate_window(double p95_ms, double err_pct, double kpi_delta) {
        bool pass = guardrail.check(p95_ms, err_pct, kpi_delta);

        if (!pass) {
            std::cout << "    !! Guardrail BREACHED: p95=" << p95_ms
                      << "ms err=" << err_pct << "% kpi_delta=" << kpi_delta
                      << "%\n";
            // In production: auto-rollback
            std::cout << "    >> AUTO ROLLBACK triggered\n";
            std::cout << "    >> Switching back to champion (previous model stays warm)\n";
            current_stage = RolloutStage::ROLLED_BACK;
            return false;
        }

        healthy_windows++;
        std::cout << "    Pass (" << healthy_windows << "/" << HOLD_WINDOWS << ")\n";

        if (healthy_windows >= HOLD_WINDOWS) {
            advance_stage();
            healthy_windows = 0;
            return true;
        }
        return false;
    }

    void advance_stage() {
        switch (current_stage) {
        case RolloutStage::SHADOW:
            current_stage = RolloutStage::CANARY_5;
            break;
        case RolloutStage::CANARY_5:
            current_stage = RolloutStage::CANARY_25;
            break;
        case RolloutStage::CANARY_25:
            current_stage = RolloutStage::CANARY_50;
            break;
        case RolloutStage::CANARY_50:
            current_stage = RolloutStage::FULL;
            break;
        default:
            break;
        }
        std::cout << "    >> ADVANCING to " << stage_name(current_stage) << "\n";
    }

    bool is_complete() const {
        return current_stage == RolloutStage::FULL || current_stage == RolloutStage::ROLLED_BACK;
    }
};

// ================================================================
// 4. Close the loop after promotion (PDF p. 450)
// ================================================================

struct PromotionChecklist {
    bool drift_references_updated = false;
    bool lineage_archived = false; // data hash, code, artifacts, report
    bool next_checkpoint_scheduled = false;
    bool drift_triggers_active = false;
    bool human_feedback_loop_active = false;

    bool is_complete() const {
        return drift_references_updated && lineage_archived && next_checkpoint_scheduled && drift_triggers_active && human_feedback_loop_active;
    }

    void print_status() const {
        auto check = [](bool b) -> std::string { return b ? "DONE" : "TODO"; };
        std::cout << "  Update drift/calibration references: " << check(drift_references_updated) << "\n";
        std::cout << "  Archive lineage (data/code/artifacts): " << check(lineage_archived) << "\n";
        std::cout << "  Schedule next checkpoint: " << check(next_checkpoint_scheduled) << "\n";
        std::cout << "  Keep drift triggers active: " << check(drift_triggers_active) << "\n";
        std::cout << "  Feed human-adjudicated labels to next snapshot: " << check(human_feedback_loop_active) << "\n";
    }
};

// ================================================================
// Main demonstration
// ================================================================

int main() {
    std::cout << "=== Chapter 11: Retraining Workflow ===\n\n";

    // --- 1. Decision flow ---
    std::cout << "1. Retraining Decision Flow\n\n";

    struct Scenario {
        std::string name;
        DriftType drift;
        bool pipeline_ok;
        bool calibration_ok;
        double psi;
    };

    std::vector<Scenario> scenarios = {
        {"Holiday gift purchase surge", DriftType::PRIOR_SHIFT, true, false, 0.05},
        {"New OS adds unknown device types", DriftType::COVARIATE_DRIFT, false, true, 0.35},
        {"Prices changed from dolalrs to cents", DriftType::COVARIATE_DRIFT, false, true, 0.42},
        {"Fraud patterns evolved", DriftType::CONCEPT_DRIFT, true, false, 0.18},
        {"Recommendation catalog refresh", DriftType::COVARIATE_NO_BUG, true, true, 0.22},
    };

    for (const auto &s : scenarios) {
        auto resp = decide_response(s.drift, s.pipeline_ok, s.calibration_ok, s.psi);
        std::cout << "  " << s.name << "\n";
        std::cout << "    Drift=" << static_cast<int>(s.drift)
                  << " PSI=" << s.psi
                  << " -> " << response_name(resp) << "\n\n";
    }

    // --- 2. Acceptance gate ---
    std::cout << "2. Acceptance Gate (YAML equivalent)\n\n";

    AcceptanceGate gate{-0.001, 2.0, 10.0, // offline: auc delta, logloss%, ece%
                        35.0, 3.0, 500.0,  // systems: p95_ms, mem_gb, load_ms
                        true};             // throughput must match champion

    struct Candidate {
        std::string name;
        double auc, logloss, ece;
        double p95_ms, mem_gb, load_ms, qps;
    };

    Candidate champion{"champion_v1", 0.85, 0.35, 0.08, 28.0, 2.5, 300.0, 250.0};

    std::vector<Candidate> candidates = {
        {"student_v1 (FP16)", 0.84, 0.36, 0.09, 22.0, 1.8, 200.0, 280.0},
        {"teacher_distilled_INT8", 0.853, 0.34, 0.14, 15.0, 1.2, 150.0, 400.0},
        {"large_ensemble_v2", 0.86, 0.33, 0.07, 55.0, 5.0, 800.0, 120.0},
    };

    for (const auto &c : candidates) {
        auto result = gate.evaluate(
            c.auc, champion.auc, c.logloss, champion.logloss,
            c.ece, champion.ece, c.p95_ms, c.mem_gb, c.load_ms, c.qps, champion.qps);

        std::cout << "  " << c.name << ": "
                  << "OFFLINE=" << (result.offline_pass ? "PASS" : "FAIL")
                  << " SYSTEMS=" << (result.systems_pass ? "PASS" : "FAIL");
        if (!result.reject_reason.empty()) {
            std::cout << "\n    Reject: " << result.reject_reason;
        }
        std::cout << "\n\n";
    }

    // --- 3. Progressive rollout ---
    std::cout << "3. Progressive Rollout Ladder\n\n";

    // Simulate a rollout campaign
    RolloutLadder ladder;
    ladder.guardrail = Guardrail{45.0, 1.0, 0.5}; // p95<=45ms, err<=1%, kpi_delta<=0.5%

    // Simulated monitoring windows with (p95_ms, error_rate%, kpi_delta%)
    std::vector<std::tuple<double, double, double>> windows = {
        // Shadow: mirror traffic, no user impact
        {30.0, 0.3, 0.1},
        {31.0, 0.4, 0.15},
        {32.0, 0.3, 0.1},
        // Canary 5%
        {33.0, 0.5, 0.2},
        {34.0, 0.6, 0.25},
        {33.0, 0.5, 0.2},
        // Canary 25%
        {35.0, 0.6, 0.3},
        {36.0, 0.7, 0.35},
        {35.0, 0.5, 0.3},
        // Canary 50%
        {36.0, 0.7, 0.4},
        {37.0, 0.8, 0.45},
        {36.0, 0.6, 0.4},
        // Full rollout (simulate guardrail breach)
        {48.0, 1.5, 1.2}, // BREACH: auto-rollback
    };

    for (size_t i = 0; i < windows.size(); ++i) {
        auto [p95, err, kpi_delta] = windows[i];
        std::cout << "Window " << (i + 1) << " [" << stage_name(ladder.current_stage) << "]: ";
        bool advanced = ladder.evaluate_window(p95, err, kpi_delta);
        if (ladder.is_complete()) {
            if (ladder.current_stage == RolloutStage::ROLLED_BACK) {
                std::cout << "\n  Rollout terminated -- rollback executed.\n";
                std::cout << "  Previous champion remains warm in memory.\n";
            }
            break;
        }
    }

    // --- 4. Close the loop ---
    std::cout << "\n4. Post-Promotion: Close the Loop\n";
    PromotionChecklist checklist;
    checklist.drift_references_updated = true;
    checklist.lineage_archived = true;
    checklist.next_checkpoint_scheduled = true;
    checklist.drift_triggers_active = true;
    checklist.human_feedback_loop_active = true;

    checklist.print_status();
    std::cout << "\n  Loop closed: " << (checklist.is_complete() ? "YES" : "NO")
              << "\n";

    // --- 5. Concrete scenario ---
    std::cout << "\n5. Concrete Scenario: Fintech Fraud\n";
    std::cout << "  Spring: PSI spike on device_type after OS release.\n";
    std::cout << "    -> Patch feature pipeline (new enum), bump schema version.\n";
    std::cout << "    -> Calibration snaps back without touching the model.\n";
    std::cout << "\n  Holiday: merchant mix and base rates change.\n";
    std::cout << "    -> Recalibrate thresholds, deploy within minutes.\n";
    std::cout << "\n  January: adversaries pivot.\n";
    std::cout << "    -> Shadow shows disagreement on gift card merchants.\n";
    std::cout << "    -> Retrain on 6 weeks of data, export FP16 and INT8.\n";
    std::cout << "    -> Gate on p95 <= 35ms and chargeback rate guardrail.\n";
    std::cout << "    -> Promote via canary, keep champion warm.\n";

    std::cout << "\n=== Retraining workflow demo complete ===\n";
    return 0;
}
