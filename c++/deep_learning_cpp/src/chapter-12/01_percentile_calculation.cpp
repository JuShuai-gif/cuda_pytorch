/*
 * percentile_calculation.cpp
 * Chapter 12: Monitoring Deployed Models
 *
 * Efficient percentile computation is the foundation of tail-latency
 * monitoring. Users feel the tail (p95/p99), not the mean, so
 * lightweight percentile calculation is essential.
 *
 * This file covers:
 *   - p95 using std::nth_element (O(n) average, no full sort)
 *   - p50, p75, p90, p99
 *   - Comparison of mean vs. percentiles (why tail matters)
 *
 * PDF pages: 459-460 (book pp. 459-460)
 *
 * Key insight: Two services with similar means can have very different
 * tail behavior. A dashboard reporting only averages would suggest
 * little difference; percentiles tell the real story.
 */

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

// ================================================================
// 1. p95 using nth_element (O(n) average, PDF p. 459)
// ================================================================

double p95(std::vector<double> &samples) {
    if (samples.empty()) return 0.0;
    const auto k = static_cast<size_t>(std::ceil(0.95 * samples.size())) - 1;
    std::nth_element(samples.begin(), samples.begin() + k, samples.end());
    return samples[k];
}

// ================================================================
// 2. General percentile function
// ================================================================

double percentile(std::vector<double> &samples, double p) {
    if (samples.empty() || p < 0.0 || p > 1.0) return 0.0;
    const auto k = static_cast<size_t>(std::ceil(p * samples.size())) - 1;
    if (k >= samples.size()) return samples.back();
    std::nth_element(samples.begin(), samples.begin() + k, samples.end());
    return samples[k];
}

// ================================================================
// 3. Compute multiple percentiles in one pass (sorts once)
//    Use when you need p50/p95/p99 together
// ================================================================

struct LatencyPercentiles {
    double p50 = 0.0, p75 = 0.0, p90 = 0.0, p95 = 0.0, p99 = 0.0;
    double mean = 0.0;
    double min = 0.0, max = 0.0;
    size_t count = 0;
};

LatencyPercentiles compute_percentiles(std::vector<double> samples) {
    LatencyPercentiles result;
    result.count = samples.size();
    if (samples.empty()) return result;

    std::sort(samples.begin(), samples.end());
    auto at = [&](double p) -> double {
        size_t idx = static_cast<size_t>(std::ceil(p * samples.size())) - 1;
        if (idx >= samples.size()) idx = samples.size() - 1;
        return samples[idx];
    };

    result.p50 = at(0.50);
    result.p75 = at(0.75);
    result.p90 = at(0.90);
    result.p95 = at(0.95);
    result.p99 = at(0.99);
    result.mean = std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(samples.size());
    result.min = samples.front();
    result.max = samples.back();
    return result;
}

// ================================================================
// 4. Demonstrate "why the tail matters" (PDF Figure 12.2)
//    Two services with similar means, different tails
// ================================================================

void demonstrate_tail_importance() {
    // Service A: tight distribution (well-managed tail)
    std::vector<double> service_a;
    for (int i = 0; i < 1000; ++i) {
        // Most requests: 10-30ms
        if (i < 950) {
            service_a.push_back(10.0 + (std::rand() % 21)); // 10-30
        } else {
            service_a.push_back(30.0 + (std::rand() % 21)); // 30-50
        }
    }

    // Service B: long right tail (queuing, GC pauses)
    std::vector<double> service_b;
    for (int i = 0; i < 1000; ++i) {
        if (i < 900) {
            service_b.push_back(8.0 + (std::rand() % 17)); // 8-24 (faster median)
        } else if (i < 970) {
            service_b.push_back(30.0 + (std::rand() % 71)); // 30-100
        } else {
            service_b.push_back(100.0 + (std::rand() % 401)); // 100-500 (long tail)
        }
    }

    auto stats_a = compute_percentiles(service_a);
    auto stats_b = compute_percentiles(service_b);

    std::cout << "\n  Service A (tight) vs Service B (long tail):\n";
    std::cout << "  " << std::setw(10) << "Metric"
              << std::setw(12) << "Service A"
              << std::setw(12) << "Service B"
              << std::setw(12) << "Delta\n";
    std::cout << "  " << std::string(46, '-') << "\n";

    auto print_row = [](const std::string &label, double a, double b) {
        std::cout << "  " << std::setw(10) << label
                  << std::setw(12) << std::fixed << std::setprecision(1) << a
                  << std::setw(12) << b
                  << std::setw(12) << (b - a) << "\n";
    };

    print_row("mean", stats_a.mean, stats_b.mean);
    print_row("p50", stats_a.p50, stats_b.p50);
    print_row("p95", stats_a.p95, stats_b.p95);
    print_row("p99", stats_a.p99, stats_b.p99);
    print_row("max", stats_a.max, stats_b.max);

    std::cout << "\n  >> Both services have similar means ("
              << stats_a.mean << " vs " << stats_b.mean
              << "), but Service B's p99 is " << stats_b.p99
              << "ms -- users feel the tail!\n";
}

// ================================================================
// Main demonstration
// ================================================================

int main() {
    std::srand(42);
    std::cout << "=== Chapter 12: Percentile Calculation ===\n\n";

    // --- Basic p95 ---
    std::cout << "1. Fast p95 (std::nth_element, O(n))\n";
    std::vector<double> latencies = {
        5.2, 8.1, 10.3, 12.5, 15.0, 18.7, 22.1, 25.3, 30.4,
        35.6, 42.1, 50.2, 60.8, 75.3, 95.1, 120.4, 155.0, 200.1};
    std::vector<double> latencies_copy = latencies;
    double p95_val = p95(latencies_copy);
    std::cout << "   p95 = " << p95_val << " ms (n=" << latencies.size() << ")\n";

    // --- Multiple percentiles ---
    std::cout << "\n2. Full percentile breakdown\n";
    auto stats = compute_percentiles(latencies);
    std::cout << "   p50=" << stats.p50 << " p75=" << stats.p75
              << " p90=" << stats.p90 << " p95=" << stats.p95
              << " p99=" << stats.p99 << " mean=" << stats.mean
              << " min=" << stats.min << " max=" << stats.max << "\n";

    // --- Tail importance demo ---
    std::cout << "\n3. Why the tail matters (Figure 12.2)";
    demonstrate_tail_importance();

    // --- SLO interpretation ---
    std::cout << "\n4. SLO Interpretation\n";
    std::cout << "   p95 <= 120ms: Service feels responsive. Exceeded -> check queue spans.\n";
    std::cout << "   TTFB/TTFT: For LLM streaming, first token under 80ms anchors perception.\n";
    std::cout << "   Availability 99.9%: ~43 mins downtime/month (error budget).\n";
    std::cout << "   Little's Law: avg_in_system = arrival_rate * avg_wait.\n";

    std::cout << "\n=== Percentile demo complete ===\n";
    return 0;
}
