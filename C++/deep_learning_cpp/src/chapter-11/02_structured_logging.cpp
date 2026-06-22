/*
 * structured_logging.cpp
 * Chapter 11: Debugging and Retraining Deployed Models
 *
 * Structured logs provide self-describing, machine-readable records
 * for each request in the inference life cycle. Each log line is a JSON
 * object containing identity, model metadata, payload summary, and outcomes.
 *
 * PDF pages: 431-432 (book pp. 431-432)
 */

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

// ================================================================
// 1. ISO8601 timestamp with microsecond precision
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
// 2. Structured log emitter
//    Fields: ts, level, msg, req (request_id), model, ver, device,
//            batch, extra (JSON string for optional fields)
// ================================================================

inline void log_structured(
    const std::string &level,
    const std::string &msg,
    const std::string &req_id,
    const std::string &model,
    const std::string &version,
    const std::string &device,
    int batch,
    const std::string &extra_json = "{}") {
    std::cout
        << "{\"ts\":\"" << iso8601_now()
        << "\",\"level\":\"" << level
        << "\",\"msg\":\"" << msg
        << "\",\"req\":\"" << req_id
        << "\",\"model\":\"" << model
        << "\",\"ver\":\"" << version
        << "\",\"device\":\"" << device
        << "\",\"batch\":" << batch
        << ",\"extra\":" << extra_json << "}\n";
}

// ================================================================
// 3. Request life cycle logging demonstration
// ================================================================

void simulate_inference(const std::string &req_id, int batch_size) {
    // Ingress: log request arrival with shapes and deadline
    log_structured("INFO", "recv", req_id, "resnet50", "1.12.3", "cuda:0",
                   batch_size,
                   R"({"shape":[8,3,224,224],"deadline_ms":200})");

    // Simulate preprocessing
    log_structured("DEBUG", "preprocess_done", req_id, "resnet50", "1.12.3",
                   "cuda:0", batch_size,
                   R"({"prep_ms":2.3,"cache_hit":true})");

    // Simulate inference completion
    log_structured("INFO", "done", req_id, "resnet50", "1.12.3", "cuda:0",
                   batch_size,
                   R"({"p50_ms":12.1,"p99_ms":27.4,"gpu_mem_mb":1420,"cache_hit":true})");
}

void simulate_error(const std::string &req_id, int batch_size) {
    log_structured("ERROR", "NaN_detected", req_id, "resnet50", "1.12.3",
                   "cuda:0", batch_size,
                   R"({"error":"NaN in softmax output","span":"infer","gpu_mem_mb":1380})");
}

// ================================================================
// 4. Cohort-aware logging
//    When debugging by cohort, include region/device/app_version in extra
// ================================================================

void simulate_cohort_request(const std::string &req_id,
                             const std::string &region,
                             const std::string &cohort) {
    log_structured("INFO", "recv", req_id, "ranking_v3", "2.1.0",
                   "cpu", 1,
                   std::string("{\"region\":\"") + region
                       + "\",\"cohort\":\"" + cohort + "\"}");
}

// ================================================================
// Main demonstration
// ================================================================

int main() {
    std::cout << "=== Chapter 11: Structured Logging ===\n\n";

    // Normal request flow
    std::cout << "--- Normal inference request ---\n";
    simulate_inference("r-7f23", 8);

    std::cout << "\n--- Another request ---\n";
    simulate_inference("r-9a41", 16);

    // Error log
    std::cout << "\n--- Error scenario ---\n";
    simulate_error("r-error01", 8);

    // Cohort-based logging
    std::cout << "\n--- Cohort-based requests ---\n";
    simulate_cohort_request("c-us-001", "us-west", "app_v17.2");
    simulate_cohort_request("c-eu-002", "eu-central", "app_v17.1");
    simulate_cohort_request("c-us-003", "us-east", "app_v17.2");

    std::cout << "\n=== Logging demo complete ===\n";
    std::cout << "\nNote: Each line above is a self-describing JSON record.\n";
    std::cout << "Use jq to filter: ./structured_logging | jq 'select(.req==\"r-7f23\")'\n";
    return 0;
}
