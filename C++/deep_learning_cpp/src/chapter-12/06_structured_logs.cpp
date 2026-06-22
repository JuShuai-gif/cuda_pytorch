/*
 * structured_logs.cpp
 * Chapter 12: Monitoring Deployed Models
 *
 * Structured JSON logs complement metrics by preserving request-level
 * evidence. Each log line is a self-describing JSON record containing
 * identity, model metadata, payload summary, timing, and cohort info.
 *
 * PDF pages: 488-489 (book pp. 488-489)
 *
 * What to include (no PII):
 *   - identity: request_id, trace_id, cohort (region, device, app)
 *   - model: model_name, model_version, schema_version
 *   - payload: shapes, data types, batch size, device
 *   - timing: ttfb_ms, p95_ms, per-span durations
 *   - outcomes: status, mem_mb
 */

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

// ================================================================
// 1. ISO8601 timestamp (PDF p. 489)
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
// 2. Structured log emitter with all recommended fields
//    (PDF pp. 465, 489)
// ================================================================

inline void log_json(
    const std::string &level,
    const std::string &msg,
    const std::string &req_id,
    const std::string &trace_id,
    const std::string &model,
    const std::string &version,
    const std::string &device,
    int batch,
    const std::string &extra_json = "{}") {
    std::cout
        << "{\"ts\":\"" << iso8601_now() << "\""
        << ",\"level\":\"" << level << "\""
        << ",\"msg\":\"" << msg << "\""
        << ",\"req\":\"" << req_id << "\""
        << ",\"trace\":\"" << trace_id << "\""
        << ",\"model\":\"" << model << "\""
        << ",\"ver\":\"" << version << "\""
        << ",\"device\":\"" << device << "\""
        << ",\"batch\":" << batch
        << ",\"extra\":" << extra_json << "}\n";
}

// ================================================================
// 3. Cohort-aware structured log (PDF p. 500)
//    Include cohort dimensions for localized debugging
// ================================================================

void log_cohort_diagnostic(
    const std::string &req_id,
    const std::string &region,
    const std::string &device_type,
    const std::string &app_version,
    double score,
    double entropy,
    double margin,
    bool abstain) {
    std::ostringstream extra;
    extra << "{\"cohort\":{"
          << "\"region\":\"" << region << "\","
          << "\"device\":\"" << device_type << "\","
          << "\"app\":\"" << app_version << "\"}"
          << ",\"score\":" << score
          << ",\"entropy\":" << entropy
          << ",\"margin\":" << margin
          << ",\"abstain\":" << (abstain ? "true" : "false") << "}";

    std::cout
        << "{\"ts\":\"" << iso8601_now() << "\""
        << ",\"level\":\"INFO\""
        << ",\"msg\":\"prediction\""
        << ",\"req\":\"" << req_id << "\""
        << ",\"model\":\"clickranker\""
        << ",\"ver\":\"2.1\""
        << ",\"device\":\"" << device_type << "\""
        << ",\"batch\":1"
        << ",\"extra\":" << extra.str() << "}\n";
}

// ================================================================
// 4. Structured request log (compact format, PDF p. 465)
// ================================================================

void log_structured_request(
    const std::string &req_id,
    int batch,
    int B, int C, int H, int W,
    double ttft_ms,
    double p95_ms,
    double vram_free_mb) {
    std::cout
        << "{\"ts\":\"" << iso8601_now() << "\""
        << ",\"req\":\"" << req_id << "\""
        << ",\"model\":\"my_model\",\"ver\":\"1.3.2\""
        << ",\"device\":\"cuda:0\""
        << ",\"batch\":" << batch
        << ",\"shape\":[" << B << "," << C << "," << H << "," << W << "]"
        << ",\"ttft_ms\":" << ttft_ms
        << ",\"p95_ms\":" << p95_ms
        << ",\"vram_free_mb\":" << vram_free_mb << "}\n";
}

// ================================================================
// Main demonstration
// ================================================================

int main() {
    std::cout << "=== Chapter 12: Structured Logs ===\n\n";

    // --- Ingress and completion logs ---
    std::cout << "1. Request life cycle logs\n";
    log_json("INFO", "recv", "r-7f23", "t-9ab1",
             "resnet50", "1.12.3", "cuda:0", 8,
             R"({"shape":[8,3,224,224],"deadline_ms":200})");

    log_json("INFO", "done", "r-7f23", "t-9ab1",
             "resnet50", "1.12.3", "cuda:0", 8,
             R"({"ttfb_ms":38.2,"lat_ms":91.5,"p99_ms":128.0,"mem_mb":1420})");

    // --- Error log ---
    std::cout << "\n2. Error log\n";
    log_json("ERROR", "OOM", "r-error01", "t-err01",
             "resnet50", "1.12.3", "cuda:0", 8,
             R"({"vram_mb":8152,"error":"cudaMalloc failed","span":"infer"})");

    // --- Compact request log ---
    std::cout << "\n3. Compact request log (PDF p. 465 format)\n";
    log_structured_request("r-compact01", 8, 8, 3, 224, 224, 38.2, 91.5, 1420.0);

    // --- Cohort-aware diagnostic logs ---
    std::cout << "\n4. Cohort-aware diagnostic logs\n";
    log_cohort_diagnostic("c-001", "EU", "ios", "4.9", 0.74, 0.39, 0.22, false);
    log_cohort_diagnostic("c-002", "NA", "android", "4.9", 0.82, 0.12, 0.68, false);
    log_cohort_diagnostic("c-003", "EU", "android", "4.8", 0.55, 0.61, 0.04, true);

    // --- What makes logs useful ---
    std::cout << "\n5. Why structured logs?\n";
    std::cout << "  - Each record is self-describing (JSON), machine-parseable.\n";
    std::cout << "  - req_id and trace_id allow joining across services.\n";
    std::cout << "  - Cohort fields (region, device, app) make localized debugging possible.\n";
    std::cout << "  - No PII: shapes and timing, not raw payloads.\n";
    std::cout << "  - Use jq to filter: ./structured_logs | jq 'select(.msg==\"done\")'\n";

    std::cout << "\n=== Structured logs demo complete ===\n";
    return 0;
}
