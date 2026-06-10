#include "latency_stats.h"

#include <cstdio>
#include <algorithm>
#include <cmath>
#include <numeric>

void LatencyStats::record(const std::string &stage, int64_t latency_ns) {
    std::lock_guard<std::mutex> lock(mutex_);
    data_[stage].push_back(latency_ns);
}

void LatencyStats::write_json(const std::string &filepath,
                              const std::string &mode_label,
                              int pipeline_depth,
                              int num_frames) const {
    std::lock_guard<std::mutex> lock(mutex_);

    FILE *f = std::fopen(filepath.c_str(), "w");
    if (!f) return;

    std::fprintf(f, "{\n");
    std::fprintf(f, "  \"pipeline_name\": \"perception_planning_control\",\n");
    std::fprintf(f, "  \"mode\": \"%s\",\n", mode_label.c_str());
    std::fprintf(f, "  \"pipeline_depth\": %d,\n", pipeline_depth);
    std::fprintf(f, "  \"total_frames\": %d,\n", num_frames);

    // 聚合统计
    std::fprintf(f, "  \"aggregate_stats\": {\n");
    bool first_stage = true;
    for (const auto &[stage, samples] : data_) {
        if (samples.empty()) continue;
        if (!first_stage) std::fprintf(f, ",\n");
        first_stage = false;

        auto sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        double mean = compute_mean(samples);
        double stddev = compute_stddev(samples, mean);
        int64_t p50 = percentile(sorted, 50);
        int64_t p99 = percentile(sorted, 99);
        int64_t max_val = sorted.back();

        std::fprintf(f, "    \"%s\": {\n", stage.c_str());
        std::fprintf(f, "      \"mean_ns\": %.1f,\n", mean);  // 均值：系统"一般"多快。容易被极端值拉偏，需配合其他指标看
        std::fprintf(f, "      \"std_ns\": %.1f,\n", stddev); // 标准差：延迟波动幅度。值越大系统越不稳定，抖动(jitter)越严重
        std::fprintf(f, "      \"p50_ns\": %ld,\n", p50);     // 中位数：一半请求比它快。比均值更能代表"典型体验"，不受极端值影响
        std::fprintf(f, "      \"p99_ns\": %ld,\n", p99);     // 99分位：99% 的请求延迟≤此值。暴露尾延迟问题，是自动驾驶安全的关键指标
        std::fprintf(f, "      \"max_ns\": %ld\n", max_val);  // 最大值：最坏情况。用于判断是否击穿实时性安全底线（如 L4 <100ms）
        std::fprintf(f, "    }");
    }
    std::fprintf(f, "\n  }");

    // 吞吐量
    if (data_.count("end_to_end") && !data_.at("end_to_end").empty()) {
        double mean_e2e_ns = compute_mean(data_.at("end_to_end"));
        double fps = 1e9 / mean_e2e_ns;
        std::fprintf(f, ",\n  \"throughput_fps\": %.2f", fps);
    }

    std::fprintf(f, "\n}\n");
    std::fclose(f);
}

const std::map<std::string, std::vector<int64_t>> &LatencyStats::data() const {
    return data_;
}

double LatencyStats::compute_mean(const std::vector<int64_t> &v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

double LatencyStats::compute_stddev(const std::vector<int64_t> &v, double mean) {
    if (v.size() < 2) return 0.0;
    double sq_sum = 0.0;
    for (auto val : v) {
        double diff = static_cast<double>(val) - mean;
        sq_sum += diff * diff;
    }
    return std::sqrt(sq_sum / (v.size() - 1));
}

int64_t LatencyStats::percentile(const std::vector<int64_t> &sorted, int pct) {
    size_t idx = static_cast<size_t>(
                     std::ceil(pct / 100.0 * sorted.size()))
                 - 1;
    if (idx >= sorted.size()) idx = sorted.size() - 1;
    return sorted[idx];
}
