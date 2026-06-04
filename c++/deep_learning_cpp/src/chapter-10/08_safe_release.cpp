/*
 * 08_safe_release.cpp
 * Chapter 10: Model Deployment and Inference Optimization
 *
 * Safe release strategies for deploying models to production:
 *
 * 1. Shadow Deployment:
 *    - Send a copy of live traffic to the new model
 *    - New model's predictions are logged but NOT returned to users
 *    - Compare predictions and latency against production model
 *    - Zero user impact, full fidelity testing
 *
 * 2. Canary Release:
 *    - Route 1-5% of real traffic to the new model
 *    - Set automatic abort conditions:
 *      - p99 latency > X% above baseline
 *      - Error rate > Y%
 *      - Business KPI degradation
 *      - Safety/toxicity threshold exceeded
 *    - Auto-rollback if any condition violated
 *
 * 3. Blue/Green Deployment:
 *    - Maintain two identical stacks (Blue = current, Green = new)
 *    - Deploy new model to Green, warm up, run health checks
 *    - Switch traffic from Blue to Green in one operation
 *    - Keep Blue hot for immediate rollback
 *    - Decommission Blue after Green passes full business cycle
 *
 * 4. Champion/Challenger Pattern:
 *    - Champion = current production model
 *    - Challenger = candidate model
 *    - Promote: Shadow → Canary(1%) → Canary(5%) → Champion
 *    - Never crown a model based on offline metrics alone
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <functional>
#include <string>
#include <thread>

// ----------------------------------------------------------------
// Simulated model scores for demo
// ----------------------------------------------------------------
struct ModelMetrics {
    std::string name;
    double accuracy = 0.0;
    double p50_ms = 0.0;
    double p99_ms = 0.0;
    double error_rate = 0.0;
    double business_kpi = 0.0; // e.g. conversion rate, engagement
};

// ----------------------------------------------------------------
// Canary Abort Conditions
// ----------------------------------------------------------------
struct CanaryConfig {
    double p99_degradation_threshold = 1.20; // p99 > 120% of baseline → abort
    double error_rate_threshold = 0.05;      // error rate > 5% → abort
    double business_kpi_threshold = 0.95;    // KPI < 95% of baseline → abort
    double canary_traffic_ratio = 0.02;      // 2% traffic initially
    int observation_minutes = 30;
};

// ----------------------------------------------------------------
// Deployment Manager
// ----------------------------------------------------------------
class DeploymentManager {
public:
    DeploymentManager(const CanaryConfig &cfg) : config_(cfg) {
    }

    // ----------------------------------------------------------
    // Shadow deployment: log predictions without affecting users
    // ----------------------------------------------------------
    bool validateShadow(
        const ModelMetrics &champion,
        const ModelMetrics &challenger) {
        std::cout << "--- Shadow Deployment ---\n";
        std::cout << "  Champion:  " << champion.name
                  << " (acc=" << champion.accuracy
                  << " p99=" << champion.p99_ms << "ms)\n";
        std::cout << "  Challenger: " << challenger.name
                  << " (acc=" << challenger.accuracy
                  << " p99=" << challenger.p99_ms << "ms)\n";

        // Check that challenger meets minimum quality bar
        bool quality_ok = true;
        std::string reasons;

        if (challenger.accuracy < champion.accuracy * 0.98) {
            quality_ok = false;
            reasons += " accuracy degraded";
        }

        if (challenger.p99_ms > champion.p99_ms * 1.5) {
            quality_ok = false;
            reasons += " latency degraded";
        }

        std::cout << "  Result: " << (quality_ok ? "PASS" : "FAIL")
                  << reasons << "\n\n";
        return quality_ok;
    }

    // ----------------------------------------------------------
    // Canary release with auto-abort
    // ----------------------------------------------------------
    bool runCanary(
        const ModelMetrics &champion,
        const ModelMetrics &challenger,
        double traffic_ratio) {
        std::cout << "--- Canary Release (" << (traffic_ratio * 100)
                  << "% traffic) ---\n";
        std::cout << "  Monitoring for " << config_.observation_minutes
                  << " minutes...\n\n";

        // Simulate metrics collection
        double challenger_p99 = challenger.p99_ms;
        double champion_p99 = champion.p99_ms;
        double challenger_errors = challenger.error_rate;
        double challenger_kpi = challenger.business_kpi;

        // Abort condition 1: p99 latency
        if (challenger_p99 > champion_p99 * config_.p99_degradation_threshold) {
            std::cout << "  ABORT: p99 latency " << challenger_p99
                      << "ms > " << (champion_p99 * config_.p99_degradation_threshold)
                      << "ms threshold\n";
            return false;
        }

        // Abort condition 2: error rate
        if (challenger_errors > config_.error_rate_threshold) {
            std::cout << "  ABORT: error rate " << challenger_errors
                      << " > " << config_.error_rate_threshold << " threshold\n";
            return false;
        }

        // Abort condition 3: business KPI
        if (challenger_kpi < champion.business_kpi * config_.business_kpi_threshold) {
            std::cout << "  ABORT: business KPI " << challenger_kpi
                      << " < "
                      << (champion.business_kpi * config_.business_kpi_threshold)
                      << " threshold\n";
            return false;
        }

        std::cout << "  PASS: All abort conditions clear.\n";
        std::cout << "    p99: " << challenger_p99 << "ms (vs " << champion_p99 << "ms)\n";
        std::cout << "    Error rate: " << challenger_errors
                  << " (threshold: " << config_.error_rate_threshold << ")\n";
        std::cout << "    Business KPI: " << challenger_kpi
                  << " (vs " << champion.business_kpi << ")\n\n";
        return true;
    }

    // ----------------------------------------------------------
    // Blue/Green switch
    // ----------------------------------------------------------
    void performBlueGreenSwitch(
        const std::string &green_name,
        const std::string &blue_name) {
        std::cout << "--- Blue/Green Switch ---\n";
        std::cout << "  BLUE (current): " << blue_name << "\n";
        std::cout << "  GREEN (new):    " << green_name << "\n";

        // Simulate health check on Green
        std::cout << "  Running Green health checks...\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        std::cout << "  Green health: OK\n";

        // Simulate switching traffic
        std::cout << "  Switching traffic: BLUE → GREEN...\n";
        std::cout << "  GREEN is now LIVE.\n";
        std::cout << "  BLUE kept hot for rollback window (24h).\n\n";
    }

    // ----------------------------------------------------------
    // Rollback procedure
    // ----------------------------------------------------------
    void rollback(const std::string &from, const std::string &to) {
        std::cout << "--- ROLLBACK ---\n";
        std::cout << "  Reverting from " << from << " → " << to << "\n";
        std::cout << "  Traffic switched back.\n";
        std::cout << "  Investigating root cause...\n\n";
    }

    // ----------------------------------------------------------
    // Champion/Challenger promotion pipeline
    // ----------------------------------------------------------
    void promoteModel(
        const ModelMetrics &champion,
        const ModelMetrics &challenger) {
        std::cout << "=== Model Promotion Pipeline ===\n";
        std::cout << "Champion:  " << champion.name
                  << " (acc=" << champion.accuracy << ")\n";
        std::cout << "Challenger: " << challenger.name
                  << " (acc=" << challenger.accuracy << ")\n\n";

        // Stage 1: Offline evaluation (simulated as already done)
        std::cout << "Stage 0: Offline evaluation — PASSED\n";

        // Stage 2: Shadow
        if (!validateShadow(champion, challenger)) {
            std::cout << "Promotion STOPPED at Shadow stage.\n";
            return;
        }

        // Stage 3: Canary 2%
        if (!runCanary(champion, challenger, 0.02)) {
            rollback(challenger.name, champion.name);
            return;
        }

        // Stage 4: Canary 5%
        CanaryConfig expanded_cfg = config_;
        expanded_cfg.canary_traffic_ratio = 0.05;
        if (!runCanary(champion, challenger, 0.05)) {
            rollback(challenger.name, champion.name);
            return;
        }

        // Stage 5: Blue/Green to full traffic
        performBlueGreenSwitch(challenger.name, champion.name);

        std::cout << "=== Promotion Complete ===\n";
        std::cout << "New champion: " << challenger.name << "\n";
        std::cout << "Decommission " << champion.name
                  << " after " << config_.observation_minutes
                  << " min observation.\n";
    }

private:
    CanaryConfig config_;
};

// ----------------------------------------------------------------
// Demo: Simulate a full model promotion pipeline
// ----------------------------------------------------------------
int main() {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "=== Safe Release Demo ===\n\n";

    CanaryConfig cfg;
    cfg.p99_degradation_threshold = 1.20;
    cfg.error_rate_threshold = 0.05;
    cfg.business_kpi_threshold = 0.95;
    cfg.canary_traffic_ratio = 0.02;
    cfg.observation_minutes = 30;

    DeploymentManager dm(cfg);

    ModelMetrics champion;
    champion.name = "ResNet50_v2";
    champion.accuracy = 0.923;
    champion.p50_ms = 8.5;
    champion.p99_ms = 24.0;
    champion.error_rate = 0.001;
    champion.business_kpi = 0.85;

    ModelMetrics challenger;
    challenger.name = "ResNet50_v3";
    challenger.accuracy = 0.931;
    challenger.p50_ms = 7.2;
    challenger.p99_ms = 22.0;
    challenger.error_rate = 0.001;
    challenger.business_kpi = 0.86;

    // Successful promotion
    dm.promoteModel(champion, challenger);

    // ----------------------------------------------------------
    // Simulate a canary that triggers abort (fast-fail scenario)
    // ----------------------------------------------------------
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "=== Simulated Fast-Fail Scenario ===\n";
    std::cout << "========================================\n\n";

    ModelMetrics bad_challenger;
    bad_challenger.name = "ResNet50_broken";
    bad_challenger.accuracy = 0.880;
    bad_challenger.p50_ms = 12.0;
    bad_challenger.p99_ms = 60.0;       // 2.5x worse than champion
    bad_challenger.error_rate = 0.05;   // barely at threshold
    bad_challenger.business_kpi = 0.80; // worse KPI

    // Shadow should fail
    bool shadow_pass = dm.validateShadow(champion, bad_challenger);
    if (!shadow_pass) {
        std::cout << "Shadow failed — model rejected before any user impact.\n";
    }

    // If shadow had passed, canary at 2% would abort on p99
    bool canary_pass = dm.runCanary(champion, bad_challenger, 0.02);
    if (!canary_pass) {
        std::cout << "Canary auto-aborted — traffic reverted to champion.\n";
    }

    std::cout << "\n--- Safe Release Principles ---\n";
    std::cout << "1. Never crown a model on offline metrics alone.\n";
    std::cout << "2. Shadow → Canary(1-2%) → Canary(5%) → Blue/Green.\n";
    std::cout << "3. Set auto-abort on: latency + errors + KPI + safety.\n";
    std::cout << "4. Always keep previous version hot for instant rollback.\n";
    std::cout << "5. Write success criteria, observation window, and rollback\n";
    std::cout << "   rules BEFORE starting the release.\n";

    return 0;
}
