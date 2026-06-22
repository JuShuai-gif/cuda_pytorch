/*
 * traces_spans.cpp
 * Chapter 12: Monitoring Deployed Models
 *
 * Metrics show that latency moved; traces show WHERE it moved.
 * A span decomposes end-to-end latency into named stages:
 *   parse → preprocess → queue → infer → postprocess → serialize
 *
 * This file covers:
 *   - RAII Span with registry integration
 *   - Full request handler with instrumented stages
 *   - Span statistics aggregation
 *   - Waterfall-style diagnosis
 *
 * PDF pages: 462, 474, 490-491 (book pp. 462, 474, 490-491)
 *
 * Key diagnostic pattern:
 *   queue span grows + infer stable → batching/concurrency issue
 *   infer span grows + queue stable → model/kernel regression
 *   preprocess grows → tokenization or image transform bottleneck
 */

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// ================================================================
// 1. ISO8601 timestamp helper
// ================================================================

inline std::string iso8601_now() {
    using namespace std::chrono;
    auto t = system_clock::now();
    auto s = time_point_cast<seconds>(t);
    auto sub = duration_cast<microseconds>(t - s).count();
    std::time_t tt = system_clock::to_time_t(t);
    std::tm tm = *std::gmtime(&tt);
    std::ostringstream os;
    os << std::put_time(&tm, "%FT%T") << "."
       << std::setw(6) << std::setfill('0') << sub << "Z";
    return os.str();
}

// ================================================================
// 2. Simplified log helper (no registry dependency for this demo)
// ================================================================

void log_span(const std::string &span_name, uint64_t dur_us,
              const std::string &req_id, const std::string &trace_id,
              const std::string &model, const std::string &version,
              const std::string &device) {
    std::cout
        << "{\"ts\":\"" << iso8601_now() << "\""
        << ",\"level\":\"TRACE\""
        << ",\"msg\":\"span\""
        << ",\"req\":\"" << req_id << "\""
        << ",\"trace\":\"" << trace_id << "\""
        << ",\"model\":\"" << model << "\""
        << ",\"ver\":\"" << version << "\""
        << ",\"device\":\"" << device << "\""
        << ",\"span\":\"" << span_name << "\""
        << ",\"dur_us\":" << dur_us << "}\n";
}

// ================================================================
// 3. RAII Span (PDF pp. 462, 474, 490-491)
// ================================================================

struct Span {
    std::string name;
    std::string req_id;
    std::string trace_id;
    std::string model;
    std::string version;
    std::string device;
    std::chrono::high_resolution_clock::time_point t0;

    Span(std::string n, std::string req, std::string trace,
         std::string m, std::string v, std::string dev) : name(std::move(n)), req_id(std::move(req)),
                                                          trace_id(std::move(trace)), model(std::move(m)),
                                                          version(std::move(v)), device(std::move(dev)),
                                                          t0(std::chrono::high_resolution_clock::now()) {
    }

    ~Span() {
        using namespace std::chrono;
        auto us =
            duration_cast<microseconds>(high_resolution_clock::now() - t0).count();
        log_span(name, static_cast<uint64_t>(us), req_id, trace_id,
                 model, version, device);
    }
};

// ================================================================
// 4. Mock work simulation
// ================================================================

void simulate_work(int base_us, int jitter_us = 0) {
    int actual = base_us + (jitter_us > 0 ? (std::rand() % (jitter_us * 2) - jitter_us) : 0);
    if (actual < 0) actual = 0;
    std::this_thread::sleep_for(std::chrono::microseconds(actual));
}

// ================================================================
// 5. Fully instrumented request handler (PDF p. 474)
// ================================================================

void handle_request(const std::string &req_id, const std::string &trace_id,
                    int queue_delay_ms) {
    Span s_all("request_total", req_id, trace_id, "resnet50", "1.12.3", "cuda:0");

    {
        Span s("parse", req_id, trace_id, "resnet50", "1.12.3", "cuda:0");
        simulate_work(4000, 1000); // parse ~4ms
    }
    {
        Span s("preprocess", req_id, trace_id, "resnet50", "1.12.3", "cuda:0");
        simulate_work(1000, 500); // preprocess ~1ms
    }
    {
        Span s("queue", req_id, trace_id, "resnet50", "1.12.3", "cuda:0");
        simulate_work(queue_delay_ms * 1000, 2000); // queue varies
    }
    {
        Span s("infer", req_id, trace_id, "resnet50", "1.12.3", "cuda:0");
        simulate_work(85000, 5000); // infer ~85ms (dominant)
    }
    {
        Span s("postprocess", req_id, trace_id, "resnet50", "1.12.3", "cuda:0");
        simulate_work(10000, 2000); // postprocess ~10ms
    }
    {
        Span s("serialize", req_id, trace_id, "resnet50", "1.12.3", "cuda:0");
        simulate_work(6000, 1000); // serialize ~6ms
    }
    // s_all destructor logs total
}

// ================================================================
// 6. Span statistics for trend analysis
// ================================================================

struct SpanStats {
    std::string name;
    uint64_t count = 0;
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
        auto ms = [](uint64_t us) { return static_cast<double>(us) / 1000.0; };
        std::cout << "  " << std::setw(13) << name
                  << ": n=" << count
                  << " avg=" << std::fixed << std::setprecision(1) << ms(total_us / count) << "ms"
                  << " min=" << ms(min_us) << "ms"
                  << " max=" << ms(max_us) << "ms\n";
    }
};

// ================================================================
// Main demonstration
// ================================================================

int main() {
    std::srand(42);
    std::cout << "=== Chapter 12: Traces and Spans ===\n\n";

    // --- Normal request ---
    std::cout << "1. Normal request (28ms queue, Figure 12.6)\n";
    handle_request("req-normal", "trace-001", 28);

    // --- Degraded request (queue delay increased) ---
    std::cout << "\n2. Degraded request (95ms queue due to batching)\n";
    handle_request("req-degraded", "trace-002", 95);

    // --- Span statistics across multiple requests ---
    std::cout << "\n3. Span statistics (20 requests, mixed queue delays)\n";
    SpanStats stats_parse{"parse"}, stats_queue{"queue"}, stats_infer{"infer"};
    // We collect data from simulated spans (not actual sleep in this demo)
    for (int i = 0; i < 20; ++i) {
        stats_parse.record(3000 + (std::rand() % 2000));
        stats_queue.record((i < 15 ? 25000 : 90000) + (std::rand() % 10000));
        stats_infer.record(84000 + (std::rand() % 10000));
    }
    stats_parse.print();
    stats_queue.print();
    stats_infer.print();

    // --- Diagnosis ---
    std::cout << "\n4. Diagnosis\n";
    if (stats_queue.max_us > stats_infer.avg_us() * 0.5) {
        std::cout << "  Queue span dominates tail: batching or concurrency bottleneck.\n";
        std::cout << "  Fix: shorten micro-batch deadline, add instances, or increase concurrency.\n";
    } else if (stats_infer.max_us > stats_infer.avg_us() * 2) {
        std::cout << "  Infer span shows high variance: possible compute contention.\n";
        std::cout << "  Fix: profile kernels (Nsight), check precision, resize batches.\n";
    } else {
        std::cout << "  Latency is dominated by expected compute. No action needed.\n";
    }

    // --- Span naming conventions ---
    std::cout << "\n5. Standard span names (Table 12.2)\n";
    std::cout << "  parse        - reading and validating raw request\n";
    std::cout << "  preprocess   - tokenizing, resizing, normalizing input\n";
    std::cout << "  queue        - waiting for GPU or worker availability\n";
    std::cout << "  infer        - actual model forward pass\n";
    std::cout << "  postprocess  - decoding, filtering, formatting output\n";
    std::cout << "  serialize    - encoding final response (JSON/protobuf)\n";

    std::cout << "\n=== Traces and spans demo complete ===\n";
    return 0;
}
