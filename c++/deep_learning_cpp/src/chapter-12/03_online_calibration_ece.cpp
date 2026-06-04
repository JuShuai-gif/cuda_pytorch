/*
 * online_calibration_ece.cpp
 * Chapter 12: Monitoring Deployed Models
 *
 * Calibration measures whether predicted probabilities match observed
 * outcomes. A model with high accuracy may still be miscalibrated --
 * claiming 0.8 confidence but only being right 40% of the time.
 *
 * This file covers:
 *   - ECE (Expected Calibration Error): bucketed reliability
 *   - Brier score: mean squared error between prob and outcome
 *   - Streaming ECE with atomic bins for concurrent updates
 *
 * PDF pages: 462-463, 468-471 (book pp. 462-463, 468-471)
 *
 * ECE interpretation: ECE < 3% is good; > 5% needs attention.
 * If ECE rises while accuracy is stable, recalibrate (temperature/
 * Platt scaling) before retraining.
 */

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

// ================================================================
// 1. Calibration bin with atomic counters (PDF p. 463)
//    Supports concurrent updates from multiple worker threads
// ================================================================

struct CalibBin {
    std::atomic<uint64_t> n{0};
    std::atomic<uint64_t> correct{0};
    // C++20 has atomic<double>::fetch_add; for C++17 compatibility
    // use a non-atomic double guarded by the class-level lock pattern.
    // In single-threaded use or with external locking this is safe.
    double sum_conf = 0.0;
};

// ================================================================
// 2. ECE aggregator (PDF pp. 463, 470-471)
//    Bucket predictions by confidence, compare predicted vs observed
// ================================================================

class ECE {
public:
    explicit ECE(int k = 10) : bins_(k) {
    }

    static int bindex(double p, int k) {
        int i = static_cast<int>(p * k);
        return std::min(std::max(i, 0), k - 1);
    }

    void observe(double prob, bool is_correct) {
        int i = bindex(prob, static_cast<int>(bins_.size()));
        bins_[i].n.fetch_add(1, std::memory_order_relaxed);
        bins_[i].sum_conf += prob; // C++17: atomic<double>::fetch_add not available
        if (is_correct) {
            bins_[i].correct.fetch_add(1, std::memory_order_relaxed);
        }
    }

    double value() const {
        uint64_t N = 0;
        for (auto &b : bins_) N += b.n.load(std::memory_order_relaxed);
        if (N == 0) return 0.0;

        double total = 0.0;
        for (auto &b : bins_) {
            auto n = b.n.load(std::memory_order_relaxed);
            if (n == 0) continue;
            double conf = b.sum_conf / static_cast<double>(n);
            double acc = static_cast<double>(b.correct.load(std::memory_order_relaxed)) / static_cast<double>(n);
            total += (static_cast<double>(n) / static_cast<double>(N)) * std::abs(acc - conf);
        }
        return total;
    }

    // Get per-bin details for debugging
    std::vector<std::tuple<int, uint64_t, double, double>> bin_details() const {
        std::vector<std::tuple<int, uint64_t, double, double>> details;
        for (int i = 0; i < static_cast<int>(bins_.size()); ++i) {
            auto n = bins_[i].n.load(std::memory_order_relaxed);
            auto correct = bins_[i].correct.load(std::memory_order_relaxed);
            auto sum_conf = bins_[i].sum_conf;
            double conf = n > 0 ? sum_conf / static_cast<double>(n) : (i + 0.5) / bins_.size();
            double acc = n > 0 ? static_cast<double>(correct) / static_cast<double>(n) : 0.0;
            details.push_back({i, n, conf, acc});
        }
        return details;
    }

private:
    std::vector<CalibBin> bins_;
};

// ================================================================
// 3. Brier score (PDF pp. 469, 471)
//    MSE between predicted probability and binary outcome
//    Lower is better. Random guess (0.5) = 0.25 Brier.
//    Penalizes both poor discrimination AND poor calibration.
// ================================================================

struct Brier {
    uint64_t n = 0;
    double sum = 0.0;

    void observe(double prob, int label01) {
        double err = prob - static_cast<double>(label01);
        sum += err * err;
        n++;
    }

    double value() const {
        return (n > 0) ? sum / static_cast<double>(n) : 0.0;
    }
};

// ================================================================
// 4. Accuracy metric for comparison
//    Accuracy alone doesn't reveal calibration problems
// ================================================================

struct RollingAccuracy {
    uint64_t n = 0;
    uint64_t correct = 0;

    void observe(bool is_correct) {
        n++;
        if (is_correct) correct++;
    }

    double value() const {
        return (n > 0) ? static_cast<double>(correct) / static_cast<double>(n) : 0.0;
    }
};

// ================================================================
// 5. Print reliability diagram (ASCII art)
// ================================================================

void print_reliability_diagram(const ECE &ece, const std::string &title) {
    auto details = ece.bin_details();
    std::cout << "\n  " << title << " Reliability Diagram:\n";
    std::cout << "  " << std::setw(8) << "Bin"
              << std::setw(12) << "N"
              << std::setw(12) << "Conf"
              << std::setw(12) << "Acc"
              << std::setw(12) << "Gap"
              << "  Calibration\n";
    std::cout << "  " << std::string(65, '-') << "\n";

    for (auto &[i, n, conf, acc] : details) {
        if (n == 0) continue;
        double gap = std::abs(acc - conf);
        std::string status = (gap < 0.05) ? "OK" : (gap < 0.10) ? "WARN" :
                                                                  "BAD";
        std::cout << "  [" << i << "] "
                  << std::setw(10) << n
                  << std::setw(12) << std::fixed << std::setprecision(3) << conf
                  << std::setw(12) << std::fixed << std::setprecision(3) << acc
                  << std::setw(12) << std::fixed << std::setprecision(3) << gap
                  << "  " << status << "\n";
    }
}

// ================================================================
// Main demonstration
// ================================================================

int main() {
    std::srand(42);
    std::cout << "=== Chapter 12: Online Calibration & ECE ===\n\n";

    // --- Well-calibrated model ---
    std::cout << "1. Well-calibrated model (predictions match outcomes)\n";
    ECE ece_well;
    Brier brier_well;
    RollingAccuracy acc_well;

    for (int i = 0; i < 1000; ++i) {
        // Generate a probability, then decide outcome based on that probability
        double prob;
        if (i < 200)
            prob = 0.05 + (std::rand() % 10) / 100.0; // ~5-15%
        else if (i < 400)
            prob = 0.20 + (std::rand() % 20) / 100.0; // ~20-40%
        else if (i < 600)
            prob = 0.45 + (std::rand() % 10) / 100.0; // ~45-55%
        else if (i < 800)
            prob = 0.60 + (std::rand() % 20) / 100.0; // ~60-80%
        else
            prob = 0.80 + (std::rand() % 20) / 100.0; // ~80-99%

        // Well-calibrated: outcome matches probability
        bool correct = ((std::rand() % 10000) / 10000.0) < prob;
        ece_well.observe(prob, correct);
        brier_well.observe(prob, correct ? 1 : 0);
        acc_well.observe(correct);
    }
    std::cout << "   ECE=" << ece_well.value() << " Brier=" << brier_well.value()
              << " Accuracy=" << acc_well.value() << "\n";
    print_reliability_diagram(ece_well, "Well-calibrated");

    // --- Overconfident model ---
    std::cout << "\n2. Overconfident model (probabilities too high)\n";
    ECE ece_over;
    Brier brier_over;
    RollingAccuracy acc_over;

    for (int i = 0; i < 1000; ++i) {
        double prob;
        if (i < 200)
            prob = 0.15 + (std::rand() % 15) / 100.0;
        else if (i < 400)
            prob = 0.35 + (std::rand() % 20) / 100.0;
        else if (i < 600)
            prob = 0.60 + (std::rand() % 15) / 100.0;
        else if (i < 800)
            prob = 0.75 + (std::rand() % 15) / 100.0;
        else
            prob = 0.90 + (std::rand() % 10) / 100.0;

        // True underlying probability is lower than stated
        double true_prob = prob * 0.65; // Overconfident by ~35%
        bool correct = ((std::rand() % 10000) / 10000.0) < true_prob;
        ece_over.observe(prob, correct);
        brier_over.observe(prob, correct ? 1 : 0);
        acc_over.observe(correct);
    }
    std::cout << "   ECE=" << ece_over.value() << " Brier=" << brier_over.value()
              << " Accuracy=" << acc_over.value() << "\n";
    print_reliability_diagram(ece_over, "Overconfident");

    // --- Streaming ECE with atomic bins ---
    std::cout << "\n3. Streaming ECE (multi-threaded scenario)\n";
    ECE ece_stream(10);
    // Simulate delayed label arrival
    for (int i = 0; i < 500; ++i) {
        double prob = 0.1 + (std::rand() % 90) / 100.0;
        bool correct = prob > 0.5;
        ece_stream.observe(prob, correct);
    }
    std::cout << "   Streaming ECE after 500 samples = " << ece_stream.value() << "\n";

    // --- Interpretation guide ---
    std::cout << "\n4. Interpretation Guide\n";
    std::cout << "   ECE < 0.03: Good calibration, no action needed.\n";
    std::cout << "   ECE 0.03-0.06: Watch; consider recalibration if persistent.\n";
    std::cout << "   ECE > 0.06: Action needed -- recalibrate or adjust thresholds.\n";
    std::cout << "\n   If ECE rises but accuracy stays flat: RECALIBRATE (temp/Platt scaling).\n";
    std::cout << "   If both ECE and accuracy degrade: RETRAIN may be needed.\n";
    std::cout << "\n   Brier score < 0.25: Better than random. Lower = better.\n";
    std::cout << "   Brier = 0.25: No information (always 0.5). Above 0.25: anti-informative.\n";

    std::cout << "\n=== Calibration ECE demo complete ===\n";
    return 0;
}
