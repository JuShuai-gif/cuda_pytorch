/*
 * distributed_tracing.cpp
 * Chapter 11: Debugging and Retraining Deployed Models
 *
 * Metrics tell you that latency changed; traces tell you where.
 * A trace decomposes end-to-end latency into named spans (parse,
 * preprocess, queue, infer, postprocess) so engineers can pinpoint
 * the bottleneck stage.
 *
 * PDF pages: 434-435 (book pp. 434-435)
 *
 * Key insight: if p99 latency rises by 80ms and the "queue" span grew
 * by ~70ms while "infer" stayed flat, the problem is likely in pacing
 * or batching, not a slower model kernel.
 */

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

// ================================================================
// 1. ISO8601 timestamp helper
// ================================================================

inline std::string iso8601_now() {
    using namespace std::chrono;
    auto t = system_clock::now();
    auto s = time_point_cast<seconds>(t);
    auto subsecs = duration_cast<microseconds>(t - s).count();
    std::time_t tt = system_clock::to_time_t(t);
    std::tm tm = *std::gmtime(&tt);
    std::ostringstream os;
    os << std::put_time(&tm, "%FT%T") << "."
       << std::setw(6) << std::setfill('0') << subsecs << "Z";
    return os.str();
}

// ================================================================
// 2. Span: scoped timer that logs duration on destruction
//    Uses RAII for automatic instrumentation
// ================================================================

struct Span {
    std::string name;
    std::string req_id;
    std::chrono::high_resolution_clock::time_point t0;

    Span(std::string n, std::string r) : name(std::move(n)), req_id(std::move(r)),
                                         t0(std::chrono::high_resolution_clock::now()) {
    }

    ~Span() {
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::high_resolution_clock::now() - t0)
                      .count();
        std::cout << "{\"ts\":\"" << iso8601_now()
                  << "\",\"level\":\"TRACE\",\"msg\":\"span\""
                  << ",\"req\":\"" << req_id
                  << "\",\"span\":\"" << name
                  << "\",\"dur_us\":" << us << "}\n";
    }
};

// ================================================================
// 3. Simulated inference request with instrumented stages
//    Each { } block creates a Span that automatically logs when it exits
// ================================================================

// Simulate work taking a given number of microseconds
void simulate_work(int us, int jitter_us = 0) {
    int actual = us + (jitter_us > 0 ? (std::rand() % (jitter_us * 2) - jitter_us) : 0);
    if (actual < 0) actual = 0;
    std::this_thread::sleep_for(std::chrono::microseconds(actual));
}

void handle_request(const std::string &req_id, int queue_delay_us = 2000) {
    // Top-level span covers the full request
    Span s_all("req_total", req_id);

    {
        Span s_parse("parse", req_id);
        simulate_work(100, 30); // parse ~100 us
    } // Span logs here

    {
        Span s_pre("preprocess", req_id);
        simulate_work(1500, 200); // preprocess ~1.5 ms
    }

    {
        Span s_queue("queue", req_id);
        simulate_work(queue_delay_us, 500); // queue varies by batch config
    }

    {
        Span s_infer("infer", req_id);
        simulate_work(10000, 500); // inference ~10 ms
    }

    {
        Span s_post("postprocess", req_id);
        simulate_work(800, 100); // postprocess ~0.8 ms
    }

    // s_all destructor logs total request duration
}

// ================================================================
// 4. Waterfall trace visualizer
//    Renders a simple ASCII waterfall from span logs
// ================================================================

void print_waterfall_diagnosis(const std::string &req_id, int queue_us) {
    std::cout << "\n--- Waterfall Diagnosis: " << req_id << " ---\n";
    int parse_us = 100, pre_us = 1500, infer_us = 10000, post_us = 800;

    auto bar = [](const std::string &label, int us, int max_us) {
        int width = 60;
        int bar_len = std::max(1, (us * width) / max_us);
        std::cout << "  " << std::left << std::setw(14) << label
                  << "[" << std::string(bar_len, '=') << ">"
                  << std::string(width - bar_len + 1, ' ') << "] "
                  << (us / 1000) << "."
                  << std::setw(3) << std::setfill('0') << (us % 1000)
                  << "ms\n"
                  << std::setfill(' ');
    };

    int max_us = parse_us + pre_us + queue_us + infer_us + post_us;
    bar("parse", parse_us, max_us);
    bar("preprocess", pre_us, max_us);
    bar("queue", queue_us, max_us);
    bar("infer", infer_us, max_us);
    bar("postprocess", post_us, max_us);
    bar("TOTAL", max_us, max_us);

    std::cout << "\n  >> Diagnosis: ";
    if (queue_us > infer_us * 0.3) {
        std::cout << "QUEUE dominates latency. Check batch deadline or pacing.\n";
    } else if (pre_us > infer_us * 0.2) {
        std::cout << "PREPROCESS is expensive. Check normalization/tokenization.\n";
    } else {
        std::cout << "INFERENCE is the bottleneck. Profile model kernels.\n";
    }
}

// ================================================================
// 5. Span aggregator for statistical analysis
//    Track span durations across requests to find systemic slowdowns
// ================================================================

struct SpanStats {
    std::string span_name;
    int count = 0;
    uint64_t total_us = 0;
    uint64_t max_us = 0;
    uint64_t min_us = UINT64_MAX;

    void record(uint64_t us) {
        count++;
        total_us += us;
        if (us > max_us) max_us = us;
        if (us < min_us) min_us = us;
    }

    double avg_us() const {
        return (count > 0) ? static_cast<double>(total_us) / count : 0;
    }

    void print() const {
        std::cout << "  " << std::left << std::setw(14) << span_name
                  << "n=" << count
                  << " avg=" << std::fixed << std::setprecision(1) << avg_us() << "us"
                  << " min=" << min_us << "us"
                  << " max=" << max_us << "us\n";
    }
};

// ================================================================
// Main demonstration
// ================================================================

int main() {
    std::srand(42);
    std::cout << "=== Chapter 11: Distributed Tracing ===\n\n";

    // Scenario 1: Normal request (2ms queue)
    std::cout << "--- Request A: Small batching delay (2ms queue) ---\n";
    handle_request("r-A-normal", 2000);

    // Scenario 2: Degraded request (25ms queue - batching deadline increased)
    std::cout << "\n--- Request B: Large batching delay (25ms queue) ---\n";
    handle_request("r-B-slow", 25000);

    // Waterfall diagnosis for the degraded case
    print_waterfall_diagnosis("r-B-slow", 25000);

    // Span stats summary
    std::cout << "\n--- Span Statistics (simulated across 10 requests) ---\n";
    SpanStats parse_stats{"parse"}, queue_stats{"queue"}, infer_stats{"infer"};

    for (int i = 0; i < 10; ++i) {
        parse_stats.record(100 + (std::rand() % 50));
        int q = (i < 8) ? (2000 + (std::rand() % 1000)) : (22000 + (std::rand() % 5000));
        queue_stats.record(q);
        infer_stats.record(10000 + (std::rand() % 500));
    }

    parse_stats.print();
    queue_stats.print();
    infer_stats.print();

    std::cout << "\n  >> Span stats show queue time dominates in tail (max="
              << queue_stats.max_us << "us), suggesting\n"
              << "     micro-batching deadline is too large for latency-sensitive traffic.\n";

    std::cout << "\n=== Tracing demo complete ===\n";
    return 0;
}
