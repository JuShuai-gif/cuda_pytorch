#pragma once

#include "pipeline_config.h"
#include "stage_types.h"

#include <chrono>
#include <cstdint>
#include <random>

// 高分辨率墙上时钟计时器
class Timer {
public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    int64_t elapsed_us() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(
                   end - start_)
            .count();
    }
    int64_t elapsed_ns() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_)
            .count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

// 自 epoch 起当前时间的纳秒值
int64_t now_ns();

// 真实流水线阶段实现（均执行实际数据处理）
SensorFrame run_sensor(int32_t frame_id, const PipelineConfig &cfg,
                       std::mt19937 &rng);
PreprocessedData run_preprocess(const SensorFrame &input,
                                const PipelineConfig &cfg,
                                std::mt19937 &rng);
DetectionResult run_detection(const PreprocessedData &input,
                              const PipelineConfig &cfg, std::mt19937 &rng);
TrackingResult run_tracking(const DetectionResult &input,
                            const PipelineConfig &cfg, std::mt19937 &rng);
PredictionResult run_prediction(const TrackingResult &input,
                                const PipelineConfig &cfg, std::mt19937 &rng);
PlanningResult run_planning(const PredictionResult &input,
                            const PipelineConfig &cfg, std::mt19937 &rng);
ControlCommand run_control(const PlanningResult &input,
                           const PipelineConfig &cfg);
