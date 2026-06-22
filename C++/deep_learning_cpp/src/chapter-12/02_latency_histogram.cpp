/*
 * latency_histogram.cpp
 * Chapter 12: Monitoring Deployed Models
 *
 * Fixed-bucket histograms are the standard way to track latency
 * distributions in production. They support percentile estimation
 * without sorting and are cheap to update on the request path.
 *
 * This file covers:
 *   - Fixed-bucket latency histogram with atomic buckets
 *   - Percentile approximation from cumulative bucket counts
 *   - Bucket interpretation for debugging (queue vs compute)
 *
 * PDF pages: 461, 475 (book pp. 461, 475)
 *
 * Design rules (PDF p. 481):
 *   1. Thread safety: use atomics, avoid locks in hot path
 *   2. Cardinality control: keep label sets small
 *   3. Fixed-bucket histograms: cheap update, sufficient for p95/p99
 *   4. Single exporter thread: format text periodically, not per-update
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

// ================================================================
// 1. Fixed-bucket latency histogram (PDF pp. 461, 475)
//    Buckets: 0-100ms, 100-200ms, ..., 900-1000ms, >=1s
//    Update cost: O(1), one atomic fetch_add per request
// ================================================================

struct LatencyHistogram {
    static constexpr int kN = 11;
    static constexpr int kWidth = 100000; // 100 ms in microseconds
    std::array<std::atomic<uint64_t>, kN> buckets{};

    void observe(uint64_t us) {
        size_t idx = (us >= static_cast<uint64_t>(kWidth) * (kN - 1)) ? (kN - 1) : (us / kWidth);
        buckets[idx].fetch_add(1, std::memory_order_relaxed);
    }

    // Total count across all buckets
    uint64_t total() const {
        uint64_t sum = 0;
        for (int i = 0; i < kN; ++i) {
            sum += buckets[i].load(std::memory_order_relaxed);
        }
        return sum;
    }

    // Approximate percentile from cumulative bucket counts
    // Returns value in microseconds
    uint64_t percentile(double p) const {
        uint64_t total_count = total();
        if (total_count == 0) return 0;

        uint64_t target = static_cast<uint64_t>(std::ceil(p * total_count));
        uint64_t cumulative = 0;

        for (int i = 0; i < kN; ++i) {
            cumulative += buckets[i].load(std::memory_order_relaxed);
            if (cumulative >= target) {
                if (i == kN - 1) {
                    // Overflow bucket: approximate as lower bound
                    return static_cast<uint64_t>(kWidth) * (kN - 1);
                }
                // Linear interpolation within bucket
                uint64_t bucket_start = static_cast<uint64_t>(kWidth) * i;
                uint64_t bucket_end = bucket_start + kWidth;
                uint64_t in_bucket = buckets[i].load(std::memory_order_relaxed);
                uint64_t prev_cum = cumulative - in_bucket;
                if (in_bucket > 0) {
                    double frac = static_cast<double>(target - prev_cum) / static_cast<double>(in_bucket);
                    return bucket_start + static_cast<uint64_t>(frac * kWidth);
                }
                return bucket_start;
            }
        }
        return static_cast<uint64_t>(kWidth) * (kN - 1);
    }

    uint64_t p50() const {
        return percentile(0.50);
    }
    uint64_t p95() const {
        return percentile(0.95);
    }
    uint64_t p99() const {
        return percentile(0.99);
    }
};

// ================================================================
// 2. Compact fixed-bucket histogram (tighter buckets, PDF p. 475)
//    0-10ms, 10-20ms, ..., 90-100ms, >=100ms
// ================================================================

struct CompactLatHisto {
    static constexpr int kN = 11;
    static constexpr int kStepUs = 10000; // 10ms steps
    std::array<std::atomic<uint64_t>, kN> b{};

    void observe(uint64_t us) {
        size_t i = (us >= static_cast<uint64_t>(kStepUs) * (kN - 1)) ? (kN - 1) : (us / kStepUs);
        b[i].fetch_add(1, std::memory_order_relaxed);
    }

    void print_distribution() const {
        std::cout << "\n  Latency distribution:\n";
        for (int i = 0; i < kN; ++i) {
            uint64_t count = b[i].load(std::memory_order_relaxed);
            uint64_t low = static_cast<uint64_t>(kStepUs) * i / 1000;
            uint64_t high = static_cast<uint64_t>(kStepUs) * (i + 1) / 1000;
            std::string range = (i < kN - 1) ? std::to_string(low) + "-" + std::to_string(high) + "ms" : ">=" + std::to_string(kStepUs / 1000 * (kN - 1)) + "ms";
            std::cout << "    " << std::setw(12) << range
                      << ": " << count << "\n";
        }
    }
};

// ================================================================
// 3. Batch size histogram
//    Batch size distribution reveals utilization patterns:
//    small batches = under-utilized; saturated = near max capacity
// ================================================================

struct BatchSizeHistogram {
    std::array<std::atomic<uint64_t>, 17> buckets{}; // 1-16, >=16

    void observe(size_t batch_size) {
        size_t idx = std::min(batch_size, buckets.size() - 1);
        buckets[idx].fetch_add(1, std::memory_order_relaxed);
    }

    double average() const {
        uint64_t total = 0, sum = 0;
        for (size_t i = 0; i < buckets.size(); ++i) {
            uint64_t c = buckets[i].load(std::memory_order_relaxed);
            total += c;
            sum += i * c;
        }
        return total > 0 ? static_cast<double>(sum) / static_cast<double>(total) : 0.0;
    }

    void print_distribution() const {
        std::cout << "\n  Batch size distribution:\n";
        for (size_t i = 1; i <= 16; ++i) {
            uint64_t count = buckets[i].load(std::memory_order_relaxed);
            if (count > 0) {
                std::cout << "    batch_size=" << i << ": " << count << "\n";
            }
        }
        std::cout << "    avg_batch_size = " << average() << "\n";
    }
};

// ================================================================
// Main demonstration
// ================================================================

int main() {
    std::srand(42);
    std::cout << "=== Chapter 12: Latency Histogram ===\n\n";

    // --- Build latency histogram with simulated traffic ---
    std::cout << "1. Simulating 1000 requests on two services\n";

    LatencyHistogram service_a, service_b;
    CompactLatHisto compact_a, compact_b;
    BatchSizeHistogram batch_a, batch_b;

    for (int i = 0; i < 1000; ++i) {
        // Service A: tight, well-batched
        size_t bs_a = 1 + (std::rand() % 8);                        // batch 1-8
        uint64_t lat_a = 5000 + bs_a * 1000 + (std::rand() % 3000); // 5-19ms
        service_a.observe(lat_a);
        compact_a.observe(lat_a);
        batch_a.observe(bs_a);

        // Service B: long tail (queuing delays)
        size_t bs_b = 1 + (std::rand() % 4); // batch 1-4, smaller
        uint64_t lat_b;
        if (i < 900) {
            lat_b = 4000 + bs_b * 1000 + (std::rand() % 2000); // 4-9ms
        } else {
            lat_b = 20000 + (std::rand() % 80000); // 20-100ms tail
        }
        service_b.observe(lat_b);
        compact_b.observe(lat_b);
        batch_b.observe(bs_b);
    }

    // --- Percentile comparison ---
    std::cout << "\n2. Percentile comparison\n";
    std::cout << "  " << std::setw(10) << "Metric"
              << std::setw(14) << "Service A"
              << std::setw(14) << "Service B"
              << std::setw(14) << "Delta\n";
    std::cout << "  " << std::string(52, '-') << "\n";

    auto print_row = [](const std::string &label, uint64_t a, uint64_t b) {
        std::cout << "  " << std::setw(10) << label
                  << std::setw(12) << (a / 1000.0) << "ms"
                  << std::setw(12) << (b / 1000.0) << "ms"
                  << std::setw(12) << ((b - a) / 1000.0) << "ms\n";
    };

    print_row("p50", service_a.p50(), service_b.p50());
    print_row("p95", service_a.p95(), service_b.p95());
    print_row("p99", service_a.p99(), service_b.p99());

    // --- Compact histogram distribution ---
    compact_a.print_distribution();
    compact_b.print_distribution();

    // --- Batch size analysis ---
    std::cout << "\n3. Batch size analysis\n";
    std::cout << "  Service A: larger batches, better utilization\n";
    batch_a.print_distribution();
    std::cout << "  Service B: smaller batches, under-batching suspected\n";
    batch_b.print_distribution();

    // --- Interpretation ---
    std::cout << "\n4. Diagnostic interpretation\n";
    if (service_b.p99() > service_a.p99() * 2 && batch_b.average() < batch_a.average() * 0.7) {
        std::cout << "  >> Service B: high p99 with small batches suggests ";
        std::cout << "QUEUING or UNDER-BATCHING.\n";
        std::cout << "  >> Fix: enable micro-batching (5-8ms delay) to ";
        std::cout << "increase batch utilization.\n";
    } else {
        std::cout << "  >> Service B p99 is within expected range.\n";
    }

    // --- SLO check ---
    std::cout << "\n5. SLO check (p95 <= 120ms)\n";
    auto check_slo = [](const char *name, uint64_t p95, uint64_t p99) {
        std::cout << "  " << name << ": p95=" << (p95 / 1000.0)
                  << "ms p99=" << (p99 / 1000.0) << "ms -> "
                  << (p95 <= 120000 ? "WITHIN SLO" : "SLO VIOLATED!") << "\n";
    };
    check_slo("Service A", service_a.p95(), service_a.p99());
    check_slo("Service B", service_b.p95(), service_b.p99());

    std::cout << "\n=== Latency histogram demo complete ===\n";
    return 0;
}
