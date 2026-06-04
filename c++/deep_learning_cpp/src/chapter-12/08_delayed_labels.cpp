/*
 * delayed_labels.cpp
 * Chapter 12: Monitoring Deployed Models
 *
 * In production, ground-truth labels often arrive minutes, hours, or
 * days after predictions. This file demonstrates how to reconcile
 * delayed labels without slowing the request path.
 *
 * This file covers:
 *   - CohortKey: compact cohort identification
 *   - PredRecord: lightweight prediction record for later joiner
 *   - RollingQuality: accumulating ECE and Brier from labels
 *   - DelayedLabelJoiner: store predictions, join labels later
 *
 * PDF pages: 492-495 (book pp. 492-495)
 *
 * Architecture: prediction-time latency stays low; exact quality
 * metrics are updated later when labels arrive.
 */

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>

// ================================================================
// 1. Cohort key (PDF p. 492)
//    Stable tags for grouping: region, device, app_version
// ================================================================

struct CohortKey {
    std::string region;
    std::string device;
    std::string app;

    bool operator==(const CohortKey &o) const {
        return region == o.region && device == o.device && app == o.app;
    }
};

struct CohortKeyHash {
    size_t operator()(const CohortKey &k) const noexcept {
        std::hash<std::string> H;
        return H(k.region) ^ (H(k.device) << 1) ^ (H(k.app) << 2);
    }
};

std::string to_string(const CohortKey &k) {
    return k.region + "/" + k.device + "/" + k.app;
}

// ================================================================
// 2. Prediction record (PDF p. 493)
//    Light enough to store at prediction time without slowing path
// ================================================================

struct PredRecord {
    std::string id; // stable request ID (not PII)
    CohortKey cohort;
    float score;   // predicted probability or confidence
    int decision;  // thresholded action (0/1)
    int64_t ts_ms; // event timestamp
};

// ================================================================
// 3. Rolling quality (PDF pp. 493-494)
//    Accumulates ECE and Brier from labeled outcomes
// ================================================================

struct RollingQuality {
    uint64_t n = 0;
    uint64_t n_pos = 0;
    double brier_sum = 0.0;
    // Reliability buckets for ECE (10 bins)
    uint64_t bins[10]{};
    uint64_t bin_pos[10]{};

    void observe(float score, int label) {
        n++;
        if (label) n_pos++;
        double d = static_cast<double>(score) - static_cast<double>(label);
        brier_sum += d * d;
        int b = std::min(9, std::max(0, static_cast<int>(score * 10)));
        bins[b]++;
        if (label) bin_pos[b]++;
    }

    double accuracy() const {
        return (n > 0) ? static_cast<double>(n_pos) / static_cast<double>(n) : 0.0;
    }

    double ece() const {
        if (n == 0) return 0.0;
        double e = 0.0;
        for (int b = 0; b < 10; ++b) {
            if (bins[b] == 0) continue;
            double conf = (b + 0.5) / 10.0;
            double acc = static_cast<double>(bin_pos[b]) / static_cast<double>(bins[b]);
            e += (static_cast<double>(bins[b]) / static_cast<double>(n)) * std::abs(acc - conf);
        }
        return e;
    }

    double brier() const {
        return (n > 0) ? brier_sum / static_cast<double>(n) : 0.0;
    }

    void print(const std::string &label) const {
        std::cout << "  " << label
                  << ": n=" << n
                  << " acc=" << std::fixed << std::setprecision(3) << accuracy()
                  << " ECE=" << ece()
                  << " Brier=" << brier() << "\n";
    }
};

// ================================================================
// 4. Delayed label joiner (PDF pp. 494-495)
//    Stores predictions temporarily, reconciles when labels arrive
// ================================================================

class DelayedLabelJoiner {
public:
    void on_prediction(const PredRecord &pr) {
        std::lock_guard<std::mutex> lk(mu_);
        preds_[pr.id] = pr;
        prune();
    }

    void on_label(const std::string &id, int label) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = preds_.find(id);
        if (it == preds_.end()) return; // too old or missing
        const PredRecord &pr = it->second;
        quality_[pr.cohort].observe(pr.score, label);
        preds_.erase(it);
    }

    RollingQuality get(const CohortKey &k) const {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = quality_.find(k);
        return (it == quality_.end()) ? RollingQuality{} : it->second;
    }

    size_t pending_count() const {
        std::lock_guard<std::mutex> lk(mu_);
        return preds_.size();
    }

    void print_all_quality() const {
        std::lock_guard<std::mutex> lk(mu_);
        for (const auto &pair : quality_) {
            pair.second.print(to_string(pair.first));
        }
    }

private:
    void prune() {
        // Production: drop stale predictions by TTL to bound memory
        // Simple: keep last 10000
        if (preds_.size() <= 10000) return;
        // Drop oldest entries (simplified)
        auto it = preds_.begin();
        // In production, track timestamps and drop by age
        (void)it;
    }

    mutable std::mutex mu_;
    std::unordered_map<std::string, PredRecord> preds_;
    std::unordered_map<CohortKey, RollingQuality, CohortKeyHash> quality_;
};

// ================================================================
// Main demonstration
// ================================================================

int main() {
    std::srand(42);
    std::cout << "=== Chapter 12: Delayed Labels ===\n\n";

    DelayedLabelJoiner joiner;

    // --- Phase 1: Serve predictions (hot path) ---
    std::cout << "1. Phase 1: Serving predictions (hot path, no label knowledge)\n";

    CohortKey eu_ios{"EU", "ios", "4.9"};
    CohortKey na_android{"NA", "android", "4.9"};
    CohortKey eu_android{"EU", "android", "4.8"};

    // Simulate serving 300 predictions across cohorts
    for (int i = 0; i < 100; ++i) {
        float score = 0.1f + static_cast<float>(std::rand() % 90) / 100.0f;
        PredRecord rec{"eu-ios-" + std::to_string(i), eu_ios, score,
                       score > 0.7f ? 1 : 0,
                       static_cast<int64_t>(i * 1000)};
        joiner.on_prediction(rec);
    }
    for (int i = 0; i < 100; ++i) {
        float score = 0.2f + static_cast<float>(std::rand() % 70) / 100.0f;
        PredRecord rec{"na-and-" + std::to_string(i), na_android, score,
                       score > 0.6f ? 1 : 0,
                       static_cast<int64_t>(i * 1000)};
        joiner.on_prediction(rec);
    }
    for (int i = 0; i < 100; ++i) {
        float score = 0.05f + static_cast<float>(std::rand() % 80) / 100.0f;
        PredRecord rec{"eu-and-" + std::to_string(i), eu_android, score,
                       score > 0.65f ? 1 : 0,
                       static_cast<int64_t>(i * 1000)};
        joiner.on_prediction(rec);
    }
    std::cout << "   Served 300 predictions. Pending: " << joiner.pending_count() << "\n";

    // --- Phase 2: Labels arrive later ---
    std::cout << "\n2. Phase 2: Labels arrive (delayed, background joiner)\n";

    // EU iOS: well-calibrated
    for (int i = 0; i < 100; ++i) {
        float score = 0.1f + static_cast<float>(std::rand() % 90) / 100.0f;
        int label = (static_cast<float>(std::rand() % 10000) / 10000.0f < score) ? 1 : 0;
        joiner.on_label("eu-ios-" + std::to_string(i), label);
    }
    // NA Android: somewhat calibrated
    for (int i = 0; i < 100; ++i) {
        float score = 0.2f + static_cast<float>(std::rand() % 70) / 100.0f;
        int label = (static_cast<float>(std::rand() % 10000) / 10000.0f < score * 0.7f) ? 1 : 0;
        joiner.on_label("na-and-" + std::to_string(i), label);
    }
    // EU Android: miscalibrated (overconfident)
    for (int i = 0; i < 100; ++i) {
        float score = 0.05f + static_cast<float>(std::rand() % 80) / 100.0f;
        int label = (static_cast<float>(std::rand() % 10000) / 10000.0f < score * 0.5f) ? 1 : 0;
        joiner.on_label("eu-and-" + std::to_string(i), label);
    }

    std::cout << "   Pending after label join: " << joiner.pending_count() << "\n";

    // --- Phase 3: Read quality metrics ---
    std::cout << "\n3. Phase 3: Read per-cohort quality metrics\n";
    joiner.print_all_quality();

    // --- Quality comparison ---
    std::cout << "\n4. Cohort quality comparison\n";
    auto q_eu_ios = joiner.get(eu_ios);
    auto q_na_and = joiner.get(na_android);
    auto q_eu_and = joiner.get(eu_android);

    if (q_eu_and.ece() > q_eu_ios.ece() * 2.0) {
        std::cout << "  !! EU Android cohort shows high ECE (" << q_eu_and.ece()
                  << " vs " << q_eu_ios.ece()
                  << "). Recalibration recommended.\n";
    }
    if (q_na_and.brier() > q_eu_ios.brier() * 1.5) {
        std::cout << "  !! NA Android cohort Brier score elevated ("
                  << q_na_and.brier() << " vs " << q_eu_ios.brier() << ").\n";
    }

    std::cout << "\n=== Delayed labels demo complete ===\n";
    std::cout << "\nArchitecture note:\n";
    std::cout << "  - Prediction-time latency stays low (record only essentials).\n";
    std::cout << "  - Background joiner reconciles labels when they arrive.\n";
    std::cout << "  - Per-cohort quality surfaces localized regressions.\n";
    return 0;
}
