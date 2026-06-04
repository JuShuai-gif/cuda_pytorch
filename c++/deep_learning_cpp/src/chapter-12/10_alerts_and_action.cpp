/*
 * alerts_and_action.cpp
 * Chapter 12: Monitoring Deployed Models
 *
 * Observability without action is just a dashboard. This file covers:
 *   - Alert policy pattern (sustained breach, hysteresis)
 *   - Drift budgets: cumulative ECE and PSI breaches
 *   - ActionRouter: runtime tuning (microbatch delay, concurrency, threshold)
 *   - Shadow → Canary → Promote rollout ladder
 *   - Alert-to-action mapping
 *
 * PDF pages: 502-509 (book pp. 502-509)
 */

#include <atomic>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// ================================================================
// 1. Drift budget (PDF pp. 505-506)
//    Like an error budget, but for distribution change and calibration
// ================================================================

struct DriftBudget {
    double ece_cum = 0.0;           // cumulative daily ECE
    uint32_t psi_breaches = 0;      // number of drift windows above threshold
    uint32_t quality_overspend = 0; // windows where quality SLO was breached

    void accrue_daily(double ece_day, uint32_t psi_windows_over) {
        ece_cum += ece_day;
        psi_breaches += psi_windows_over;
    }

    bool needs_recalibration(double ece_limit = 0.30) const {
        return ece_cum > ece_limit;
    }

    bool needs_retrain(uint32_t psi_limit = 12) const {
        return psi_breaches > psi_limit;
    }

    void print(const std::string &label) const {
        std::cout << "  " << label << ":\n";
        std::cout << "    ECE cumulative: " << std::fixed << std::setprecision(4) << ece_cum;
        std::cout << (needs_recalibration() ? " (NEEDS RECALIBRATION)" : " (ok)") << "\n";
        std::cout << "    PSI breaches: " << psi_breaches;
        std::cout << (needs_retrain() ? " (NEEDS RETRAIN)" : " (ok)") << "\n";
    }
};

// ================================================================
// 2. Alert policy with hysteresis (PDF pp. 502-503)
//    Alerts should fire on sustained breaches, not transient noise
// ================================================================

struct AlertPolicy {
    double threshold;    // e.g., p95 > 120ms or ECE > 0.03
    int sustain_windows; // must breach for N consecutive windows
    int clear_windows;   // must be clear for N windows to reset
    int breach_count = 0;
    int clear_count = 0;
    bool is_firing = false;

    AlertPolicy(double thresh, int sustain, int clear) : threshold(thresh), sustain_windows(sustain), clear_windows(clear) {
    }

    // Evaluate one monitoring window
    // Returns true if alert state changed
    bool evaluate(double current_value) {
        if (current_value > threshold) {
            clear_count = 0;
            breach_count++;
            if (!is_firing && breach_count >= sustain_windows) {
                is_firing = true;
                return true; // Alert just fired
            }
        } else {
            breach_count = 0;
            if (is_firing) {
                clear_count++;
                if (clear_count >= clear_windows) {
                    is_firing = false;
                    return true; // Alert just cleared
                }
            }
        }
        return false;
    }

    std::string status() const {
        if (is_firing) return "FIRING";
        std::string s = "OK (";
        s += std::to_string(breach_count) + "/" + std::to_string(sustain_windows);
        s += ")";
        return s;
    }
};

// ================================================================
// 3. Tuning: runtime service parameters (PDF p. 506)
//    Conservative, reversible changes that don't require redeploy
// ================================================================

struct Tuning {
    std::atomic<int> microbatch_delay_ms{8};
    std::atomic<int> max_concurrency{4};
    std::atomic<float> score_threshold{0.7f};
};

// ================================================================
// 4. ActionRouter: apply runtime changes from alerts (PDF pp. 506-507)
// ================================================================

class ActionRouter {
public:
    explicit ActionRouter(Tuning &t) : t_(t) {
    }

    void apply(const std::string &kind, const std::string &payload) {
        if (kind == "set_threshold") {
            float th = std::stof(payload);
            float old = t_.score_threshold.exchange(th);
            log("threshold", old, th);
        } else if (kind == "set_microbatch_delay") {
            int ms = std::stoi(payload);
            int old = t_.microbatch_delay_ms.exchange(ms);
            log("microbatch_delay_ms", old, ms);
        } else if (kind == "set_concurrency") {
            int k = std::stoi(payload);
            int old = t_.max_concurrency.exchange(k);
            log("max_concurrency", old, k);
        } else if (kind == "enable_debug_mode") {
            debug_mode_ = true;
            std::cerr << "[action] debug_mode=on\n";
        } else if (kind == "disable_debug_mode") {
            debug_mode_ = false;
            std::cerr << "[action] debug_mode=off\n";
        } else {
            std::cerr << "[action] unknown kind: " << kind << "\n";
        }
    }

    bool debug_mode() const {
        return debug_mode_;
    }

private:
    template <typename T>
    void log(const char *key, T oldv, T newv) {
        std::cerr << "[action] " << key << ": " << oldv << " -> " << newv << "\n";
    }

    Tuning &t_;
    bool debug_mode_{false};
};

// ================================================================
// 5. Rollout stage (PDF pp. 503, 508-509)
//    Shadow → Canary 5% → 25% → 50% → Full
// ================================================================

enum class RolloutStage {
    SHADOW,
    CANARY_5,
    CANARY_25,
    CANARY_50,
    FULL,
    ROLLED_BACK
};

struct RolloutGuardrail {
    double p95_ms_max;
    double ece_max;
    double error_rate_pct_max;

    bool check(double p95_ms, double ece, double err_pct) const {
        return p95_ms <= p95_ms_max && ece <= ece_max && err_pct <= error_rate_pct_max;
    }
};

// ================================================================
// 6. Policy test helper (PDF pp. 509)
//    Test policies before production: synthetic drift injection
// ================================================================

struct PolicyTestResult {
    bool latency_alert_fired = false;
    bool calibration_alert_fired = false;
    bool drift_alert_fired = false;
    bool auto_rollback_triggered = false;

    void print() const {
        auto pass = [](bool b) -> std::string { return b ? "FIRED" : "OK"; };
        std::cout << "    latency_alert: " << pass(latency_alert_fired) << "\n";
        std::cout << "    calibration_alert: " << pass(calibration_alert_fired) << "\n";
        std::cout << "    drift_alert: " << pass(drift_alert_fired) << "\n";
        std::cout << "    auto_rollback: " << pass(auto_rollback_triggered) << "\n";
    }
};

// ================================================================
// Main demonstration
// ================================================================

int main() {
    std::cout << "=== Chapter 12: Alerts and Action ===\n\n";

    // --- Alert hysteresis ---
    std::cout << "1. Alert hysteresis (sustained breach detection)\n";

    AlertPolicy ece_alert(0.03, 3, 2);
    // Simulate monitoring windows
    std::vector<double> ece_windows = {
        0.02, 0.025, 0.035, 0.038, 0.042, 0.045, // breaches building
        0.048, 0.039, 0.031, 0.028, 0.022, 0.018 // recovering
    };

    for (size_t i = 0; i < ece_windows.size(); ++i) {
        bool changed = ece_alert.evaluate(ece_windows[i]);
        std::cout << "  Window " << (i + 1) << ": ECE=" << ece_windows[i]
                  << " -> " << ece_alert.status();
        if (changed) std::cout << " (STATE CHANGED!)";
        std::cout << "\n";
    }

    // --- Drift budgets ---
    std::cout << "\n2. Drift budgets (cumulative tracking)\n";

    DriftBudget budget_eu_ios;
    DriftBudget budget_na_android;

    // Simulate 7 days of monitoring
    for (int day = 0; day < 7; ++day) {
        // EU iOS: mostly healthy
        double ece_eu = 0.02 + (std::rand() % 10) / 1000.0; // 0.02-0.03
        uint32_t psi_eu = (std::rand() % 100 < 20) ? 1 : 0; // occasional breach
        budget_eu_ios.accrue_daily(ece_eu, psi_eu);

        // NA Android: degrading
        double ece_na = 0.04 + (std::rand() % 30) / 1000.0; // 0.04-0.07
        uint32_t psi_na = (std::rand() % 100 < 50) ? 1 : 0; // frequent breaches
        budget_na_android.accrue_daily(ece_na, psi_na);
    }

    budget_eu_ios.print("EU iOS");
    budget_na_android.print("NA Android");

    // --- ActionRouter ---
    std::cout << "\n3. ActionRouter: runtime tuning\n";
    Tuning tuning;
    ActionRouter router(tuning);

    std::cout << "  Initial: delay=" << tuning.microbatch_delay_ms.load()
              << "ms concurrency=" << tuning.max_concurrency.load()
              << " threshold=" << tuning.score_threshold.load() << "\n";

    router.apply("set_microbatch_delay", "4"); // reduce batching delay
    router.apply("set_concurrency", "8");      // increase concurrency
    router.apply("set_threshold", "0.65");     // lower score threshold

    std::cout << "  After:  delay=" << tuning.microbatch_delay_ms.load()
              << "ms concurrency=" << tuning.max_concurrency.load()
              << " threshold=" << tuning.score_threshold.load() << "\n";

    // --- Alert to action mapping ---
    std::cout << "\n4. Alert-to-action mapping\n";
    std::cout << "  p95 latency breach with queue-heavy spans:\n";
    std::cout << "    -> Reduce microbatch delay or add capacity (service-side, immediate).\n\n";
    std::cout << "  ECE breach in one cohort:\n";
    std::cout << "    -> Export {score, label} pairs, fit temperature offline,\n";
    std::cout << "       load cohort-tagged calibration artifact, alert clears.\n\n";
    std::cout << "  Feature drift (PSI) breach:\n";
    std::cout << "    -> Route to data pipeline owner first (unit change? schema?)\n";
    std::cout << "       NOT a model problem until pipeline is confirmed clean.\n\n";
    std::cout << "  VRAM < 5%:\n";
    std::cout << "    -> Quantize (FP16/INT8), reduce batch, pooling allocator.\n";

    // --- Rollout guardrails ---
    std::cout << "\n5. Shadow → Canary → Promote (Figure 12.13)\n";
    RolloutGuardrail guardrail{120.0, 0.03, 1.0};

    // Simulated canary evaluation
    struct CanaryStep {
        std::string stage;
        double p95;
        double ece;
        double err_pct;
    };

    std::vector<CanaryStep> steps = {
        {"Shadow", 85.0, 0.015, 0.3},
        {"Canary 5%", 92.0, 0.018, 0.4},
        {"Canary 25%", 105.0, 0.022, 0.5},
        {"Canary 50%", 118.0, 0.028, 0.7},
    };

    for (auto &s : steps) {
        bool pass = guardrail.check(s.p95, s.ece, s.err_pct);
        std::cout << "  " << std::setw(12) << s.stage << ": "
                  << "p95=" << s.p95 << "ms ECE=" << s.ece
                  << " err=" << s.err_pct << "% -> "
                  << (pass ? "PASS" : "FAIL") << "\n";
    }

    // --- Policy testing ---
    std::cout << "\n6. Policy testing: synthetic drift injection\n";
    std::cout << "  Drill 1: Inject feature distribution shift in staging.\n";
    std::cout << "    Expected: drift alert fires, calibration wobbles.\n";
    std::cout << "    No retraining job should start until labels justify it.\n\n";
    std::cout << "  Drill 2: Inject queue delay into serving path.\n";
    std::cout << "    Expected: latency alert fires, control loop lowers\n";
    std::cout << "    microbatch delay, p95 returns to SLO.\n\n";
    std::cout << "  If alerts don't fire when expected, or flap when they\n";
    std::cout << "  shouldn't, fix the policy before deploying broadly.\n";

    std::cout << "\n=== Alerts and action demo complete ===\n";
    return 0;
}
