#pragma once

#include <string>
#include <vector>
#include <cstdint>

// ============================================================================
// PerformanceContract: 为机器人系统模块定义延迟/抖动/吞吐量需求
// ============================================================================
struct PerformanceContract {
    std::string module_name;
    std::string team_owner;

    // 延迟需求（微秒）
    int64_t latency_p50_us;
    int64_t latency_p99_us;
    int64_t latency_max_us;

    // 抖动需求（微秒）
    int64_t jitter_max_us;

    // 吞吐量需求
    double min_fps;

    // 机器人专属指标
    int64_t missed_detections_per_1000; // 最大允许的漏检数
    int64_t planning_timeout_count;     // 最大允许的规划超时次数

    // 资源需求
    double max_cpu_percent;
    int64_t max_memory_mb;

    bool validate_latency_p50(int64_t measured_us) const {
        return measured_us <= latency_p50_us;
    }
    bool validate_latency_p99(int64_t measured_us) const {
        return measured_us <= latency_p99_us;
    }
    bool validate_latency_max(int64_t measured_us) const {
        return measured_us <= latency_max_us;
    }
    bool validate_jitter(int64_t measured_us) const {
        return measured_us <= jitter_max_us;
    }
    bool validate_throughput(double measured_fps) const {
        return measured_fps >= min_fps;
    }
};

// ============================================================================
// MeasurementSample: 单帧计时数据
// ============================================================================
struct MeasurementSample {
    int64_t latency_us;
    int64_t frame_id;
    int64_t timestamp_us;
};

// ============================================================================
// MeasurementBatch: 用于验证的样本集合
// ============================================================================
struct MeasurementBatch {
    std::string module_name;
    std::vector<MeasurementSample> samples;
    double measured_fps;
    double measured_cpu_percent;
    int64_t measured_memory_mb;
    int64_t missed_detections;
    int64_t planning_timeouts;
};

// ============================================================================
// 违规记录
// ============================================================================
struct Violation {
    std::string module_name;
    std::string metric_name;
    int64_t required_value;
    int64_t measured_value;
    std::string unit;
    bool is_pass;

    double severity() const {
        if (is_pass) return 0.0;
        if (required_value == 0) return 1.0;
        return static_cast<double>(measured_value) / static_cast<double>(required_value);
    }
};

// ============================================================================
// 模块接口（跨团队 API 合约）
// ============================================================================

// 感知团队: 提供目标检测结果
struct PerceptionInput {
    int frame_id;
    int image_width;
    int image_height;
};

struct PerceptionOutput {
    int frame_id;
    int num_detections;
    int64_t latency_us;
    int64_t timestamp_us;
};

class PerceptionModule {
public:
    PerceptionModule();
    PerceptionOutput process(const PerceptionInput &input);
    PerformanceContract get_contract() const;
};

// 规划团队: 提供轨迹规划
struct PlanningInput {
    int frame_id;
    int num_obstacles;
};

struct PlanningOutput {
    int frame_id;
    std::vector<std::pair<int, int>> trajectory;
    int64_t latency_us;
    int64_t timestamp_us;
    bool timed_out;
};

class PlanningModule {
public:
    PlanningModule();
    PlanningOutput process(const PlanningInput &input);
    PerformanceContract get_contract() const;
    static constexpr int GRID_SIZE = 200;
};

// 控制团队: 提供执行器指令
struct ControlInput {
    int frame_id;
    double target_speed;
    double target_angle;
};

struct ControlOutput {
    int frame_id;
    double throttle;
    double steering;
    int64_t latency_us;
    int64_t timestamp_us;
};

class ControlModule {
public:
    ControlModule();
    ControlOutput process(const ControlInput &input);
    PerformanceContract get_contract() const;
};
