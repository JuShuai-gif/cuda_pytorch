/*
 * leading_indicators.cpp
 * Chapter 12: Monitoring Deployed Models
 *
 * While labels are delayed, the service needs real-time signals that
 * correlate with quality. Three families of leading indicators:
 *
 *   1. Uncertainty measures (entropy, top-2 margin)
 *   2. Champion-challenger disagreement
 *   3. Abstention/override rates
 *
 * This file covers:
 *   - Softmax entropy: uncertainty proxy (PDF p. 495)
 *   - Top-2 margin: confidence proxy (PDF p. 496)
 *   - Disagreement counter: challenger divergence (PDF p. 496)
 *   - CohortMonitor: EWMA-based leading indicator tracker (PDF p. 498)
 *
 * PDF pages: 495-499 (book pp. 495-499)
 */

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// ================================================================
// 1. Softmax entropy (PDF p. 495)
//    Higher entropy = more uncertainty = potential quality issue
// ================================================================

inline float softmax_entropy(const std::vector<float> &logits) {
    float m = *std::max_element(logits.begin(), logits.end());
    double Z = 0.0;
    std::vector<double> p(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        p[i] = std::exp(static_cast<double>(logits[i] - m));
        Z += p[i];
    }
    double H = 0.0;
    for (double &pi : p) {
        pi /= Z;
        if (pi > 0) H -= pi * std::log(pi + 1e-12);
    }
    return static_cast<float>(H);
}

// ================================================================
// 2. Top-2 margin (PDF p. 496)
//    Smaller margin = model less certain about top prediction
// ================================================================

inline float top2_margin(const std::vector<float> &probs) {
    float a = 0.0f, b = 0.0f;
    for (float v : probs) {
        if (v > a) {
            b = a;
            a = v;
        } else if (v > b) {
            b = v;
        }
    }
    return a - b; // smaller = less certain
}

// ================================================================
// 3. Champion-challenger disagreement (PDF p. 496)
//    Track how often champion and shadow challenger disagree
// ================================================================

struct Disagreement {
    std::atomic<uint64_t> n{0};
    std::atomic<uint64_t> diff{0};

    void observe(int y_hat, int y_hat_shadow) {
        n.fetch_add(1, std::memory_order_relaxed);
        if (y_hat != y_hat_shadow) {
            diff.fetch_add(1, std::memory_order_relaxed);
        }
    }

    double rate() const {
        uint64_t N = n.load(std::memory_order_relaxed);
        uint64_t D = diff.load(std::memory_order_relaxed);
        return (N > 0) ? static_cast<double>(D) / static_cast<double>(N) : 0.0;
    }

    void print(const std::string &label) const {
        auto N = n.load(std::memory_order_relaxed);
        auto D = diff.load(std::memory_order_relaxed);
        std::cout << "  " << label
                  << ": n=" << N
                  << " disagree=" << D
                  << " rate=" << std::fixed << std::setprecision(4) << rate() << "\n";
    }
};

// ================================================================
// 4. Cohort leading indicators with EWMA (PDF pp. 498-499)
//    Smoothing factor alpha = 0.2 gives fast adaptation
// ================================================================

struct CohortLeading {
    double ent_ewma = 0.0;
    double margin_ewma = 0.0;
    double disagree_ewma = 0.0;
    double abstain_ewma = 0.0;
    bool inited = false;

    void update(double ent, double margin, double disagree, double abstain,
                double alpha = 0.2) {
        if (!inited) {
            ent_ewma = ent;
            margin_ewma = margin;
            disagree_ewma = disagree;
            abstain_ewma = abstain;
            inited = true;
            return;
        }
        ent_ewma = (1.0 - alpha) * ent_ewma + alpha * ent;
        margin_ewma = (1.0 - alpha) * margin_ewma + alpha * margin;
        disagree_ewma = (1.0 - alpha) * disagree_ewma + alpha * disagree;
        abstain_ewma = (1.0 - alpha) * abstain_ewma + alpha * abstain;
    }
};

// ================================================================
// 5. Cohort monitor (PDF p. 499)
// ================================================================

// Simple CohortKey definition (inlined to avoid cross-file dependency)
struct CohortKey {
    std::string region, device, app;
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

class CohortMonitor {
public:
    void on_request(const CohortKey &k,
                    double entropy, double margin, bool abstain) {
        std::lock_guard<std::mutex> lk(mu_);
        auto &c = lead_[k];
        c.update(entropy, margin, 0.0, abstain ? 1.0 : 0.0);
    }

    void on_shadow_pair(const CohortKey &k, int y_hat, int y_hat_shadow) {
        std::lock_guard<std::mutex> lk(mu_);
        auto &c = lead_[k];
        double disagree = (y_hat != y_hat_shadow) ? 1.0 : 0.0;
        c.update(c.ent_ewma, c.margin_ewma, disagree, c.abstain_ewma);
    }

    CohortLeading get(const CohortKey &k) const {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = lead_.find(k);
        return (it == lead_.end()) ? CohortLeading{} : it->second;
    }

    void print_all() const {
        std::lock_guard<std::mutex> lk(mu_);
        for (const auto &pair : lead_) {
            const auto &c = pair.second;
            std::cout << "  " << to_string(pair.first) << ": "
                      << "ent=" << std::fixed << std::setprecision(3) << c.ent_ewma
                      << " margin=" << c.margin_ewma
                      << " disagree=" << c.disagree_ewma
                      << " abstain=" << c.abstain_ewma << "\n";
        }
    }

private:
    mutable std::mutex mu_;
    std::unordered_map<CohortKey, CohortLeading, CohortKeyHash> lead_;
};

// ================================================================
// 6. Utility: softmax from logits
// ================================================================

std::vector<float> softmax(const std::vector<float> &logits) {
    float m = *std::max_element(logits.begin(), logits.end());
    std::vector<float> probs(logits.size());
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - m);
        sum += probs[i];
    }
    for (auto &p : probs) p /= static_cast<float>(sum);
    return probs;
}

// ================================================================
// Main demonstration
// ================================================================

int main() {
    std::srand(42);
    std::cout << "=== Chapter 12: Leading Indicators ===\n\n";

    // --- Entropy and Top-2 margin ---
    std::cout << "1. Uncertainty measures (entropy, top-2 margin)\n\n";

    struct SampleLogits {
        std::string desc;
        std::vector<float> logits;
    };

    std::vector<SampleLogits> samples = {
        {"High confidence", {5.0f, 0.5f, 0.2f, -1.0f, -2.0f}},
        {"Medium confidence", {2.0f, 1.8f, 1.5f, -0.5f, -1.0f}},
        {"Low confidence (uniform)", {0.5f, 0.5f, 0.5f, 0.5f, 0.5f}},
        {"Ambiguous top-2", {3.0f, 2.95f, 0.1f, -0.5f, -1.0f}},
    };

    std::cout << "  " << std::setw(22) << "Description"
              << std::setw(12) << "Entropy"
              << std::setw(12) << "Margin"
              << std::setw(12) << "Max(logits)" << "\n";
    std::cout << "  " << std::string(58, '-') << "\n";

    for (const auto &s : samples) {
        float max_logit = *std::max_element(s.logits.begin(), s.logits.end());
        auto probs = softmax(s.logits);
        float ent = softmax_entropy(s.logits);
        float margin = top2_margin(probs);
        std::cout << "  " << std::setw(22) << s.desc
                  << std::setw(12) << std::fixed << std::setprecision(4) << ent
                  << std::setw(12) << margin
                  << std::setw(12) << max_logit << "\n";
    }

    // --- Champion-challenger disagreement ---
    std::cout << "\n2. Champion-challenger disagreement\n";

    Disagreement disagree_healthy, disagree_drifting;
    // Healthy: champion and challenger mostly agree
    for (int i = 0; i < 500; ++i) {
        int champ = std::rand() % 10;
        int chall = (std::rand() % 100 < 95) ? champ : (std::rand() % 10);
        disagree_healthy.observe(champ, chall);
    }
    // Drifting: challenger starts diverging
    for (int i = 0; i < 500; ++i) {
        int champ = std::rand() % 10;
        int chall = (std::rand() % 100 < 70) ? champ : (std::rand() % 10);
        disagree_drifting.observe(champ, chall);
    }
    disagree_healthy.print("Healthy champion-challenger");
    disagree_drifting.print("Drifting champion-challenger");

    // --- Cohort monitor ---
    std::cout << "\n3. Cohort-aware leading indicators\n";

    CohortMonitor monitor;
    CohortKey eu_ios{"EU", "ios", "4.9"};
    CohortKey na_android{"NA", "android", "4.9"};

    // Simulate requests to EU iOS (normal)
    for (int i = 0; i < 50; ++i) {
        double entropy = 0.2 + (std::rand() % 20) / 100.0; // 0.2-0.4
        double margin = 0.5 + (std::rand() % 30) / 100.0;  // 0.5-0.8
        monitor.on_request(eu_ios, entropy, margin, false);
    }
    // Simulate shadow pairs
    for (int i = 0; i < 30; ++i) {
        int champ = std::rand() % 10;
        int chall = (std::rand() % 100 < 92) ? champ : (std::rand() % 10);
        monitor.on_shadow_pair(eu_ios, champ, chall);
    }

    // Simulate requests to NA Android (degrading: high entropy, low margin)
    for (int i = 0; i < 50; ++i) {
        double entropy = 0.5 + (std::rand() % 30) / 100.0; // 0.5-0.8 (high!)
        double margin = 0.1 + (std::rand() % 20) / 100.0;  // 0.1-0.3 (low!)
        bool abstain = (std::rand() % 100 < 15);           // 15% abstain rate
        monitor.on_request(na_android, entropy, margin, abstain);
    }
    for (int i = 0; i < 30; ++i) {
        int champ = std::rand() % 10;
        int chall = (std::rand() % 100 < 65) ? champ : (std::rand() % 10);
        monitor.on_shadow_pair(na_android, champ, chall);
    }

    monitor.print_all();

    // --- Diagnosis ---
    std::cout << "\n4. Diagnosis from leading indicators\n";
    auto eu = monitor.get(eu_ios);
    auto na = monitor.get(na_android);

    if (na.ent_ewma > eu.ent_ewma * 1.5) {
        std::cout << "  !! NA Android: elevated entropy ("
                  << na.ent_ewma << " vs " << eu.ent_ewma
                  << "). Model less confident on this cohort.\n";
    }
    if (na.disagree_ewma > eu.disagree_ewma * 2.0) {
        std::cout << "  !! NA Android: rising disagreement with challenger ("
                  << na.disagree_ewma << " vs " << eu.disagree_ewma
                  << "). Possible drift.\n";
    }
    if (na.abstain_ewma > 0.1) {
        std::cout << "  !! NA Android: high abstention rate ("
                  << na.abstain_ewma << "). Data contract or preprocess issue?\n";
    }

    std::cout << "\n5. Leading indicator rules of thumb\n";
    std::cout << "  Entropy rise → model uncertain (calibration or drift).\n";
    std::cout << "  Margin drop → ambiguous predictions, possible threshold issue.\n";
    std::cout << "  Disagreement rise → champion-challenger divergence (early drift signal).\n";
    std::cout << "  Abstain rate spike → data pipeline or upstream contract issue.\n";
    std::cout << "  ALWAYS break down by cohort -- global averages hide local problems.\n";

    std::cout << "\n=== Leading indicators demo complete ===\n";
    return 0;
}
