#pragma once

#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <cstdint>

// ============================================================================
// 用于延迟分析的统计收集器。线程安全。
// ============================================================================
class LatencyStats {
public:
    void record(const std::string &stage, int64_t latency_ns);

    // 将结构化 JSON 指标写入文件
    void write_json(const std::string &filepath,
                    const std::string &mode_label,
                    int pipeline_depth,
                    int num_frames) const;

    // 访问原始数据供外部使用
    const std::map<std::string, std::vector<int64_t>> &data() const;

private:
    static double compute_mean(const std::vector<int64_t> &v);
    static double compute_stddev(const std::vector<int64_t> &v, double mean);
    static int64_t percentile(const std::vector<int64_t> &sorted, int pct);

    mutable std::mutex mutex_;
    std::map<std::string, std::vector<int64_t>> data_;
};
