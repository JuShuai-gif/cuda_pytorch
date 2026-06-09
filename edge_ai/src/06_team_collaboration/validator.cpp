#include "validator.h"

int64_t ContractValidator::percentile(const std::vector<int64_t> &sorted, double p) {
    if (sorted.empty()) return 0;
    int idx = static_cast<int>(sorted.size() * p);
    if (idx >= static_cast<int>(sorted.size())) idx = static_cast<int>(sorted.size()) - 1;
    if (idx < 0) idx = 0;
    return sorted[idx];
}

void ContractValidator::compute_stats(const std::vector<int64_t> &latencies,
                                      int64_t &p50, int64_t &p99,
                                      int64_t &max_val, int64_t &jitter) {
    if (latencies.empty()) {
        p50 = p99 = max_val = jitter = 0;
        return;
    }
    std::vector<int64_t> sorted = latencies;
    std::sort(sorted.begin(), sorted.end());
    p50 = percentile(sorted, 0.50);
    p99 = percentile(sorted, 0.99);
    max_val = sorted.back();
    jitter = max_val - sorted.front();
}

std::vector<Violation> ContractValidator::validate(
    const PerformanceContract &contract,
    const MeasurementBatch &batch) {
    std::vector<Violation> violations;

    if (batch.samples.empty()) {
        violations.push_back({contract.module_name, "samples", 1, 0, "count", false});
        return violations;
    }

    std::vector<int64_t> latencies;
    for (const auto &s : batch.samples) {
        latencies.push_back(s.latency_us);
    }
    std::sort(latencies.begin(), latencies.end());

    int64_t p50 = percentile(latencies, 0.50);
    int64_t p99 = percentile(latencies, 0.99);
    int64_t max_val = latencies.back();
    int64_t min_val = latencies.front();
    int64_t jitter = max_val - min_val;

    violations.push_back({contract.module_name, "latency_p50_us",
                          contract.latency_p50_us, p50, "us",
                          contract.validate_latency_p50(p50)});

    violations.push_back({contract.module_name, "latency_p99_us",
                          contract.latency_p99_us, p99, "us",
                          contract.validate_latency_p99(p99)});

    violations.push_back({contract.module_name, "max_latency_us",
                          contract.latency_max_us, max_val, "us",
                          contract.validate_latency_max(max_val)});

    violations.push_back({contract.module_name, "jitter_us",
                          contract.jitter_max_us, jitter, "us",
                          contract.validate_jitter(jitter)});

    violations.push_back({contract.module_name, "throughput_fps",
                          static_cast<int64_t>(contract.min_fps * 1000),
                          static_cast<int64_t>(batch.measured_fps * 1000),
                          "mFPS",
                          contract.validate_throughput(batch.measured_fps)});

    violations.push_back({contract.module_name, "missed_detections_per_1000",
                          contract.missed_detections_per_1000,
                          batch.missed_detections,
                          "count",
                          batch.missed_detections <= contract.missed_detections_per_1000});

    violations.push_back({contract.module_name, "planning_timeout_count",
                          contract.planning_timeout_count,
                          batch.planning_timeouts,
                          "count",
                          batch.planning_timeouts <= contract.planning_timeout_count});

    return violations;
}
