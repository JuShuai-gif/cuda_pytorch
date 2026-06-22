#include "latency_stats.h"

#include <sstream>

// ============================================================================
// JSON 辅助函数：简单的手动 JSON 构建器（无需外部库）
// ============================================================================
static void write_json_key_value(std::ostream &os, const std::string &key,
                                 double value, int precision = 2,
                                 bool last = false) {
    os << "    \"" << key << "\": " << std::fixed << std::setprecision(precision)
       << value;
    if (!last) os << ",";
    os << "\n";
}

static void write_json_key_int(std::ostream &os, const std::string &key,
                               int64_t value, bool last = false) {
    os << "    \"" << key << "\": " << value;
    if (!last) os << ",";
    os << "\n";
}

static void write_json_key_str(std::ostream &os, const std::string &key,
                               const std::string &value, bool last = false) {
    os << "    \"" << key << "\": \"" << value << "\"";
    if (!last) os << ",";
    os << "\n";
}

// ============================================================================
// 直方图计算
// ============================================================================
std::vector<int> LatencyStats::compute_histogram(const std::vector<int64_t> &v,
                                                 int num_bins) {
    std::vector<int> bins(num_bins, 0);
    if (v.empty()) return bins;
    auto [mn, mx] = std::minmax_element(v.begin(), v.end());
    int64_t min_val = *mn, max_val = *mx;
    if (min_val == max_val) {
        bins[0] = static_cast<int>(v.size());
        return bins;
    }
    double bin_width =
        static_cast<double>(max_val - min_val) / static_cast<double>(num_bins);
    for (auto val : v) {
        int idx = static_cast<int>(
            static_cast<double>(val - min_val) / bin_width);
        if (idx >= num_bins) idx = num_bins - 1;
        if (idx < 0) idx = 0;
        bins[idx]++;
    }
    return bins;
}

// ============================================================================
// 控制台摘要
// ============================================================================
void LatencyStats::print_summary() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::cout << "\n"
              << std::string(80, '=') << "\n";
    std::cout << "  流水线延迟统计摘要\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << std::left << std::setw(18) << "阶段" << std::setw(10)
              << "样本数" << std::setw(12) << "平均(us)" << std::setw(12)
              << "标准差(us)" << std::setw(12) << "P50(us)" << std::setw(12)
              << "P99(us)" << std::setw(12) << "最大(us)"
              << "\n";
    std::cout << std::string(88, '-') << "\n";

    std::vector<std::pair<std::string, std::vector<int64_t>>> sorted;
    for (const auto &kv : data_) {
        sorted.emplace_back(kv);
    }
    std::sort(sorted.begin(), sorted.end(),
              [](const auto &a, const auto &b) {
                  return compute_mean(a.second) > compute_mean(b.second);
              });

    for (const auto &[stage, samples] : sorted) {
        if (samples.empty()) continue;
        print_stage_stats(stage, samples);
    }

    if (!e2e_data_.empty()) {
        std::cout << std::string(88, '-') << "\n";
        print_stage_stats("端到端", e2e_data_);
        double e2e_mean = compute_mean(e2e_data_);
        std::cout << "\n吞吐量：" << std::fixed << std::setprecision(2)
                  << (1e6 / e2e_mean) << " FPS\n";
    }

    if (!sorted.empty()) {
        const auto &bottleneck = sorted.front();
        std::cout << "\n瓶颈阶段：" << bottleneck.first << "（"
                  << std::fixed << std::setprecision(1)
                  << compute_mean(bottleneck.second) << " us 平均）\n";
    }
}

void LatencyStats::print_stage_stats(
    const std::string &stage, const std::vector<int64_t> &samples) const {
    auto sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    double mean = compute_mean(samples);
    double stddev = compute_stddev(samples, mean);
    int64_t p50 = percentile(sorted, 50);
    int64_t p99 = percentile(sorted, 99);
    int64_t max_val = sorted.back();

    std::cout << std::left << std::setw(18) << stage << std::setw(10)
              << sorted.size() << std::setw(12) << std::fixed
              << std::setprecision(1) << mean << std::setw(12) << std::fixed
              << std::setprecision(1) << stddev << std::setw(12) << p50
              << std::setw(12) << p99 << std::setw(12) << max_val << "\n";
}

// ============================================================================
// JSON 报告输出
// ============================================================================
void LatencyStats::write_json_report(const std::string &filepath,
                                     const std::string &pipeline_name,
                                     int total_frames) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ofstream f(filepath);
    if (!f.is_open()) {
        std::cerr << "错误：无法打开 " << filepath << " 进行写入\n";
        return;
    }

    f << "{\n";
    write_json_key_str(f, "pipeline", pipeline_name);
    write_json_key_int(f, "total_frames", total_frames);

    // 阶段数据部分
    f << "  \"stages\": {\n";

    // 按平均值降序收集阶段名称
    std::vector<std::pair<std::string, std::vector<int64_t>>> sorted;
    for (const auto &kv : data_) {
        sorted.emplace_back(kv);
    }
    std::sort(sorted.begin(), sorted.end(),
              [](const auto &a, const auto &b) {
                  return compute_mean(a.second) > compute_mean(b.second);
              });

    size_t stage_idx = 0;
    for (const auto &[stage, samples] : sorted) {
        if (samples.empty()) continue;
        auto s = samples;
        std::sort(s.begin(), s.end());
        double mean = compute_mean(samples);
        int64_t p50 = percentile(s, 50);
        int64_t p99 = percentile(s, 99);
        int64_t max_val = s.back();
        int64_t min_val = s.front();
        double stddev = compute_stddev(samples, mean);
        auto hist = compute_histogram(samples);

        f << "    \"" << stage << "\": {\n";
        write_json_key_value(f, "mean_us", mean, 2);
        write_json_key_value(f, "p50_us", static_cast<double>(p50), 2);
        write_json_key_value(f, "p99_us", static_cast<double>(p99), 2);
        write_json_key_value(f, "max_us", static_cast<double>(max_val), 2);
        write_json_key_value(f, "min_us", static_cast<double>(min_val), 2);
        write_json_key_value(f, "stddev_us", stddev, 2);
        write_json_key_int(f, "count", static_cast<int64_t>(s.size()));

        // 直方图
        f << "    \"histogram\": [";
        for (size_t i = 0; i < hist.size(); i++) {
            if (i > 0) f << ", ";
            f << hist[i];
        }
        bool is_last_stage = (stage_idx == sorted.size() - 1);
        f << "]\n";
        f << "    }";
        if (!is_last_stage) f << ",";
        f << "\n";
        stage_idx++;
    }
    f << "  },\n";

    // 端到端数据部分
    f << "  \"e2e\": ";
    if (!e2e_data_.empty()) {
        auto s = e2e_data_;
        std::sort(s.begin(), s.end());
        double mean = compute_mean(s);
        int64_t p50 = percentile(s, 50);
        int64_t p99 = percentile(s, 99);
        int64_t max_val = s.back();
        int64_t min_val = s.front();
        double stddev = compute_stddev(s, mean);
        auto hist = compute_histogram(s);

        f << "{\n";
        write_json_key_value(f, "mean_us", mean, 2);
        write_json_key_value(f, "p50_us", static_cast<double>(p50), 2);
        write_json_key_value(f, "p99_us", static_cast<double>(p99), 2);
        write_json_key_value(f, "max_us", static_cast<double>(max_val), 2);
        write_json_key_value(f, "min_us", static_cast<double>(min_val), 2);
        write_json_key_value(f, "stddev_us", stddev, 2);

        f << "    \"histogram\": [";
        for (size_t i = 0; i < hist.size(); i++) {
            if (i > 0) f << ", ";
            f << hist[i];
        }
        f << "]\n";
        f << "  },\n";
    } else {
        f << "{},\n";
    }

    // 瓶颈识别
    std::string bottleneck = "unknown";
    double bottleneck_pct = 0.0;
    if (!sorted.empty()) {
        bottleneck = sorted.front().first;
        double bn_mean = compute_mean(sorted.front().second);
        double total_mean = 0.0;
        for (const auto &[_, smp] : sorted) {
            total_mean += compute_mean(smp);
        }
        if (total_mean > 0) {
            bottleneck_pct = bn_mean / total_mean * 100.0;
        }
    }
    write_json_key_str(f, "bottleneck", bottleneck);
    write_json_key_value(f, "bottleneck_pct", bottleneck_pct, 1, true);

    f << "}\n";
    f.close();

    std::cout << "遥测数据已写入：" << filepath << "\n";
}
