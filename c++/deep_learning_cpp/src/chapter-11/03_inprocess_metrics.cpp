/*
 * inprocess_metrics.cpp
 * Chapter 11: Debugging and Retraining Deployed Models
 *
 * In-process metrics provide cheap, always-on aggregation directly
 * inside the inference process. Using atomics and periodic flushes,
 * we can track request counts, latencies, queue times, and errors
 * with minimal overhead.
 *
 * PDF pages: 433-434 (book pp. 433-434)
 *
 * Usage pattern:
 *   1. Initialize a Metrics struct per process/thread
 *   2. Call observe_request() at the end of each request
 *   3. Call flush_metrics() periodically (every 30-60 seconds)
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

// ================================================================
// 1. Minimal in-process metrics aggregator
//    Uses atomics for thread-safe increment-only counters
// ================================================================

struct Metrics {
    std::atomic<uint64_t> req_total{0};
    std::atomic<uint64_t> err_total{0};
    std::atomic<uint64_t> q_time_us{0};     // queue time (microseconds)
    std::atomic<uint64_t> infer_time_us{0}; // inference time (microseconds)
    std::atomic<uint64_t> bytes_in{0};
    std::atomic<uint64_t> bytes_out{0};

    void observe_request(uint64_t q_us, uint64_t i_us,
                         uint64_t in_b, uint64_t out_b) {
        req_total.fetch_add(1, std::memory_order_relaxed);
        q_time_us.fetch_add(q_us, std::memory_order_relaxed);
        infer_time_us.fetch_add(i_us, std::memory_order_relaxed);
        bytes_in.fetch_add(in_b, std::memory_order_relaxed);
        bytes_out.fetch_add(out_b, std::memory_order_relaxed);
    }

    void observe_error() {
        err_total.fetch_add(1, std::memory_order_relaxed);
    }
};

// ================================================================
// 2. Periodic metrics flush (Prometheus exposition format compatible)
// ================================================================

inline void flush_metrics(const Metrics &m) {
    auto req = m.req_total.load(std::memory_order_relaxed);
    auto err = m.err_total.load(std::memory_order_relaxed);
    auto q_us = m.q_time_us.load(std::memory_order_relaxed);
    auto i_us = m.infer_time_us.load(std::memory_order_relaxed);
    auto b_in = m.bytes_in.load(std::memory_order_relaxed);
    auto b_out = m.bytes_out.load(std::memory_order_relaxed);

    // Prometheus-style exposition (can be scraped directly)
    std::cout << "# HELP inference_requests_total Total inference requests\n";
    std::cout << "# TYPE inference_requests_total counter\n";
    std::cout << "inference_requests_total " << req << "\n";

    std::cout << "# HELP inference_errors_total Total inference errors\n";
    std::cout << "# TYPE inference_errors_total counter\n";
    std::cout << "inference_errors_total " << err << "\n";

    std::cout << "# HELP inference_queue_time_us_total Cumulative queue time (us)\n";
    std::cout << "# TYPE inference_queue_time_us_total counter\n";
    std::cout << "inference_queue_time_us_total " << q_us << "\n";

    std::cout << "# HELP inference_infer_time_us_total Cumulative inference time (us)\n";
    std::cout << "# TYPE inference_infer_time_us_total counter\n";
    std::cout << "inference_infer_time_us_total " << i_us << "\n";

    // Derived: average latencies
    double avg_q = (req > 0) ? static_cast<double>(q_us) / static_cast<double>(req) : 0.0;
    double avg_i = (req > 0) ? static_cast<double>(i_us) / static_cast<double>(req) : 0.0;
    std::cout << "# HELP inference_queue_time_avg_us Average queue time per request (us)\n";
    std::cout << "# TYPE inference_queue_time_avg_us gauge\n";
    std::cout << "inference_queue_time_avg_us " << avg_q << "\n";

    std::cout << "# HELP inference_infer_time_avg_us Average inference time per request (us)\n";
    std::cout << "# TYPE inference_infer_time_avg_us gauge\n";
    std::cout << "inference_infer_time_avg_us " << avg_i << "\n";

    double err_rate = (req > 0) ? static_cast<double>(err) / static_cast<double>(req) * 100.0 : 0.0;
    std::cout << "# HELP inference_error_rate_pct Error rate percentage\n";
    std::cout << "# TYPE inference_error_rate_pct gauge\n";
    std::cout << "inference_error_rate_pct " << err_rate << "\n";
}

// ================================================================
// 3. Latency histogram simulation
//    For measuring p50/p95/p99, maintain an ordered set of recent latencies
// ================================================================

struct LatencyHistogram {
    static constexpr size_t MAX_SAMPLES = 10000;
    std::vector<uint64_t> samples;

    void record(uint64_t latency_us) {
        if (samples.size() < MAX_SAMPLES) {
            samples.push_back(latency_us);
        }
        // Production: use reservoir sampling for unbounded data
    }

    uint64_t percentile(double p) const {
        if (samples.empty()) return 0;
        auto sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        size_t idx = static_cast<size_t>(p * static_cast<double>(sorted.size()));
        if (idx >= sorted.size()) idx = sorted.size() - 1;
        return sorted[idx];
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
// Main: simulate load and flush metrics
// ================================================================

int main() {
    std::cout << "=== Chapter 11: In-Process Metrics ===\n\n";

    Metrics metrics;
    LatencyHistogram latency_hist;

    // Simulate a burst of requests
    std::cout << "Simulating 100 inference requests...\n";
    for (int i = 0; i < 100; ++i) {
        // Random queue time: 50-500 us
        uint64_t q_us = 50 + (std::rand() % 451);
        // Random inference time: 500-5000 us
        uint64_t i_us = 500 + (std::rand() % 4501);
        // Payload sizes
        uint64_t in_bytes = 224 * 224 * 3 * 4; // typical image
        uint64_t out_bytes = 1000 * 4;         // 1000-class logits

        metrics.observe_request(q_us, i_us, in_bytes, out_bytes);
        latency_hist.record(q_us + i_us);

        // Simulate occasional errors (5% error rate)
        if (std::rand() % 100 < 5) {
            metrics.observe_error();
        }
    }

    // Print metrics snapshot
    std::cout << "\n--- Metrics Snapshot (Prometheus format) ---\n";
    flush_metrics(metrics);

    // Print latency percentiles
    std::cout << "\n--- Latency Percentiles ---\n";
    std::cout << "  p50: " << latency_hist.p50() << " us\n";
    std::cout << "  p95: " << latency_hist.p95() << " us\n";
    std::cout << "  p99: " << latency_hist.p99() << " us\n";

    // Show debug questions these metrics can answer
    std::cout << "\n--- Diagnostic Questions ---\n";
    std::cout << "  Did avg batch size fall? Check req_total vs. flush frequency.\n";
    std::cout << "  Did queue time grow while infer time stayed flat? Batching problem.\n";
    std::cout << "  Did bytes_in spike? Possible payload bloat or schema change.\n";

    std::cout << "\n=== Metrics demo complete ===\n";
    return 0;
}
