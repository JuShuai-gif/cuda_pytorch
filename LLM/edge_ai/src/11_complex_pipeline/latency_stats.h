#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <string>
#include <vector>

class LatencyStats {
public:
    void record_stage(const std::string &stage, int64_t latency_us) {
        std::lock_guard<std::mutex> lock(mutex_);
        data_[stage].push_back(latency_us);
    }

    void record_e2e(int64_t latency_us) {
        std::lock_guard<std::mutex> lock(mutex_);
        e2e_data_.push_back(latency_us);
    }

    void print_summary() const;
    void write_json_report(const std::string &filepath,
                           const std::string &pipeline_name,
                           int total_frames) const;

private:
    void print_stage_stats(const std::string &stage,
                           const std::vector<int64_t> &samples) const;

    static std::vector<int> compute_histogram(const std::vector<int64_t> &v,
                                              int num_bins = 20);

    static double compute_mean(const std::vector<int64_t> &v) {
        if (v.empty()) return 0.0;
        return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    }

    static double compute_stddev(const std::vector<int64_t> &v, double mean) {
        if (v.size() < 2) return 0.0;
        double sq_sum = 0.0;
        for (auto val : v) {
            double diff = val - mean;
            sq_sum += diff * diff;
        }
        return std::sqrt(sq_sum / (v.size() - 1));
    }

    static int64_t percentile(std::vector<int64_t> sorted, int pct) {
        if (sorted.empty()) return 0;
        size_t idx =
            static_cast<size_t>(std::ceil(pct / 100.0 * sorted.size())) - 1;
        if (idx >= sorted.size()) idx = sorted.size() - 1;
        return sorted[idx];
    }

    mutable std::mutex mutex_;
    std::map<std::string, std::vector<int64_t>> data_;
    std::vector<int64_t> e2e_data_;
};
