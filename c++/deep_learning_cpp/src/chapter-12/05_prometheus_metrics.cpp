/*
 * prometheus_metrics.cpp
 * Chapter 12: Monitoring Deployed Models
 *
 * A minimal in-process metrics registry with Prometheus text exposition
 * format. This is the foundation of production monitoring for C++ inference
 * services.
 *
 * This file covers:
 *   - Counter, Gauge, Histogram primitives (PDF pp. 482-483)
 *   - Labels for metric dimensions (PDF p. 484)
 *   - Registry with label lookup (PDF pp. 484-487)
 *   - Prometheus text exposition rendering (PDF pp. 485-486)
 *
 * PDF pages: 480-488 (book pp. 480-488)
 *
 * Design rules:
 *   - Thread safety: atomics for fast-path, lock only at registry lookup
 *   - Cardinality control: bounded labels (never request ID)
 *   - Fixed-bucket histograms: cheap update, sufficient for p95/p99
 *   - Single exporter: format text periodically or on /metrics request
 *
 * NOTE: Metric types use std::unique_ptr storage in registry because
 * std::atomic members make types non-copyable/non-movable.
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// ================================================================
// 1. Metric primitives (PDF pp. 482-483)
// ================================================================

struct Counter {
    std::atomic<uint64_t> value{0};
    void inc(uint64_t n = 1) {
        value.fetch_add(n, std::memory_order_relaxed);
    }
    uint64_t get() const {
        return value.load(std::memory_order_relaxed);
    }
};

struct Gauge {
    std::atomic<long long> value{0};
    void set(long long v) {
        value.store(v, std::memory_order_relaxed);
    }
    long long get() const {
        return value.load(std::memory_order_relaxed);
    }
};

struct Histogram {
    // Fixed buckets in microseconds: 1,2,4,8,16,32,64,128,256, >=256 ms
    static constexpr int K = 10;
    static constexpr std::array<uint64_t, K - 1> bounds_us{
        1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000};
    std::array<std::atomic<uint64_t>, K> buckets{};
    std::atomic<uint64_t> sum_us{0};
    std::atomic<uint64_t> count{0};

    void observe(uint64_t us) {
        size_t i = 0;
        while (i < K - 1 && us > bounds_us[i]) ++i;
        buckets[i].fetch_add(1, std::memory_order_relaxed);
        sum_us.fetch_add(us, std::memory_order_relaxed);
        count.fetch_add(1, std::memory_order_relaxed);
    }
};

// ================================================================
// 2. Labels (PDF p. 484)
// ================================================================

struct Labels {
    std::vector<std::pair<std::string, std::string>> kv;

    Labels() = default;
    Labels(std::initializer_list<std::pair<std::string, std::string>> init) : kv(init) {
    }

    std::string to_text() const {
        if (kv.empty()) return "";
        std::ostringstream os;
        os << "{";
        for (size_t i = 0; i < kv.size(); ++i) {
            if (i) os << ",";
            os << kv[i].first << "=\"" << kv[i].second << "\"";
        }
        os << "}";
        return os.str();
    }
};

// ================================================================
// 3. Registry (PDF pp. 484-487)
//    Uses unique_ptr storage because atomic members prevent copy/move
// ================================================================

class Registry {
public:
    Counter &counter(const std::string &name, const Labels &lbl = {}) {
        return get<Counter>(counters_, name, lbl);
    }
    Gauge &gauge(const std::string &name, const Labels &lbl = {}) {
        return get<Gauge>(gauges_, name, lbl);
    }
    Histogram &histo(const std::string &name, const Labels &lbl = {}) {
        return get<Histogram>(histos_, name, lbl);
    }

    std::string render_metrics_text() const {
        std::ostringstream os;

        // Counters
        for (const auto &it : counters_) {
            os << "# HELP " << it.first << " (counter)\n";
            os << "# TYPE " << it.first << " counter\n";
            for (const auto &row : it.second) {
                os << it.first << row.first.to_text() << " "
                   << row.second->get() << "\n";
            }
        }

        // Gauges
        for (const auto &it : gauges_) {
            os << "# HELP " << it.first << " (gauge)\n";
            os << "# TYPE " << it.first << " gauge\n";
            for (const auto &row : it.second) {
                os << it.first << row.first.to_text() << " "
                   << row.second->get() << "\n";
            }
        }

        // Histograms
        for (const auto &it : histos_) {
            os << "# HELP " << it.first << " (histogram)\n";
            os << "# TYPE " << it.first << " histogram\n";
            for (const auto &row : it.second) {
                const auto &lbl = row.first;
                const auto &h = *row.second;
                uint64_t cum = 0;
                for (size_t i = 0; i < Histogram::K - 1; ++i) {
                    cum += h.buckets[i].load(std::memory_order_relaxed);
                    Labels l = lbl;
                    l.kv.emplace_back(
                        "le", std::to_string(Histogram::bounds_us[i] / 1000.0));
                    os << it.first << "_bucket" << l.to_text() << " "
                       << cum << "\n";
                }
                cum += h.buckets[Histogram::K - 1].load(std::memory_order_relaxed);
                Labels linf = lbl;
                linf.kv.emplace_back("le", "+Inf");
                os << it.first << "_bucket" << linf.to_text() << " "
                   << cum << "\n";
                os << it.first << "_sum" << lbl.to_text() << " "
                   << (h.sum_us.load() / 1000.0) << "\n";
                os << it.first << "_count" << lbl.to_text() << " "
                   << h.count.load() << "\n";
            }
        }

        return os.str();
    }

private:
    template <typename T>
    using Table =
        std::unordered_map<std::string,
                           std::vector<std::pair<Labels, std::unique_ptr<T>>>>;

    template <typename T>
    static T &get(Table<T> &table, const std::string &name, const Labels &lbl) {
        std::lock_guard<std::mutex> lk(mu_);
        auto &vec = table[name];
        for (auto &row : vec) {
            if (row.first.kv == lbl.kv) return *row.second;
        }
        vec.emplace_back(lbl, std::make_unique<T>());
        return *vec.back().second;
    }

    inline static std::mutex mu_;
    Table<Counter> counters_;
    Table<Gauge> gauges_;
    Table<Histogram> histos_;
};

// ================================================================
// 4. /metrics console renderer (PDF pp. 487-488)
// ================================================================

void serve_metrics_console(Registry &reg) {
    std::cout << "\n--- /metrics (Prometheus exposition format) ---\n";
    std::cout << reg.render_metrics_text();
    std::cout << "--- end /metrics ---\n\n";
}

// ================================================================
// Main demonstration
// ================================================================

int main() {
    std::srand(42);
    std::cout << "=== Chapter 12: Prometheus-style Metrics ===\n\n";

    // Create registry and register metrics
    Registry R;

    // Request-level counters (with labels for model, version)
    auto &req_total = R.counter(
        "inference_requests_total",
        {{"model", "resnet50"}, {"version", "1.12.3"}});
    auto &err_total = R.counter(
        "inference_errors_total",
        {{"model", "resnet50"}, {"version", "1.12.3"}});

    // Latency histogram
    auto &lat_histo = R.histo(
        "inference_latency_us",
        {{"model", "resnet50"}, {"version", "1.12.3"}});

    // GPU metrics (gauges)
    auto &gpu_util = R.gauge(
        "gpu_util_percent",
        {{"device", "cuda:0"}});
    auto &vram_free = R.gauge(
        "gpu_vram_free_mb",
        {{"device", "cuda:0"}});

    // Cohort-metric example
    auto &ece_metric = R.gauge(
        "quality_ece",
        {{"region", "EU"}, {"device", "ios"}, {"app", "4.9"}});

    // Simulate request handling
    std::cout << "1. Simulating 100 inference requests...\n";
    for (int i = 0; i < 100; ++i) {
        req_total.inc();

        // Simulate latency: 8-50ms in us
        uint64_t lat_us = 8000 + (std::rand() % 42001);
        lat_histo.observe(lat_us);

        // Occasional errors
        if (std::rand() % 100 < 3) {
            err_total.inc();
        }
    }

    // Update gauges
    gpu_util.set(78);
    vram_free.set(2500);
    ece_metric.set(21); // ECE as milli-percent (0.021 * 1000)

    // Render metrics
    serve_metrics_console(R);

    // Demonstrate label cardinality
    std::cout << "2. Label cardinality demonstration\n";
    std::cout << "   Good:  " << Labels({{"model", "m1"}, {"version", "1.3"}}).to_text() << "\n";
    std::cout << "   Good:  " << Labels({{"region", "EU"}, {"device", "ios"}}).to_text() << "\n";
    std::cout << "   BAD:   labels with request_id cause cardinality explosion\n";

    // Design rules recap
    std::cout << "\n3. Design Rules Recap\n";
    std::cout << "   1. Use atomics for hot-path updates (no locks in request handler).\n";
    std::cout << "   2. Keep label cardinality bounded (model, version, device, region).\n";
    std::cout << "   3. Fixed-bucket histograms: cheap update, support p95/p99.\n";
    std::cout << "   4. Export text periodically (every 30-60s), not on every update.\n";

    std::cout << "\n=== Prometheus metrics demo complete ===\n";
    return 0;
}
