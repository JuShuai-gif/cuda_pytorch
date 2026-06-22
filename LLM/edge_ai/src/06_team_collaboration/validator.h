#pragma once

#include "interface.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

class ContractValidator {
public:
    static int64_t percentile(const std::vector<int64_t> &sorted, double p);

    static std::vector<Violation> validate(
        const PerformanceContract &contract,
        const MeasurementBatch &batch);

    // 从延迟数据计算 P50、P99、最大值、抖动
    static void compute_stats(const std::vector<int64_t> &latencies,
                              int64_t &p50, int64_t &p99,
                              int64_t &max_val, int64_t &jitter);
};
