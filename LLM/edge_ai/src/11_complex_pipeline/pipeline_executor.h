#pragma once

#include "pipeline_config.h"
#include "latency_stats.h"
#include "stage_queue.h"
#include "stage_ops.h"

#include <atomic>
#include <random>
#include <thread>

// 顺序执行：一次处理一帧，逐阶段执行
class SequentialExecutor {
public:
    SequentialExecutor(const PipelineConfig &cfg, LatencyStats &stats) : cfg_(cfg), stats_(stats), rng_(cfg.seed) {
    }

    void run();

private:
    const PipelineConfig &cfg_;
    LatencyStats &stats_;
    std::mt19937 rng_;
};

// 流水线执行：七阶段流水线，支持并发处理多帧。
// 每个阶段从其输入队列读取，并写入下一个阶段的队列。
class PipelinedExecutor {
public:
    PipelinedExecutor(const PipelineConfig &cfg, LatencyStats &stats) : cfg_(cfg), stats_(stats), stop_flag_(false) {
    }

    void run();

private:
    void sensor_worker();
    void preprocess_worker();
    void detection_worker();
    void tracking_worker();
    void prediction_worker();
    void planning_worker();
    void control_worker();

    const PipelineConfig &cfg_;
    LatencyStats &stats_;
    std::atomic<bool> stop_flag_{false};

    // 队列：输入类型 = 接收阶段所消费的数据类型
    StageQueue<SensorFrame> sensor_in_q_;        // 主线程 -> 传感器工作线程
    StageQueue<SensorFrame> prep_in_q_;          // 传感器工作线程 -> 预处理工作线程
    StageQueue<PreprocessedData> det_in_q_;      // 预处理工作线程 -> 检测工作线程
    StageQueue<DetectionResult> tracking_in_q_;  // 检测工作线程 -> 跟踪工作线程
    StageQueue<TrackingResult> pred_in_q_;       // 跟踪工作线程 -> 预测工作线程
    StageQueue<PredictionResult> planning_in_q_; // 预测工作线程 -> 规划工作线程
    StageQueue<PlanningResult> control_in_q_;    // 规划工作线程 -> 控制工作线程
};
