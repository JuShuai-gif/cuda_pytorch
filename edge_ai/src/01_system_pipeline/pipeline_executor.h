#pragma once

#include "pipeline_config.h"
#include "latency_stats.h"

#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>

// ============================================================================
// 顺序执行: 完整处理完一帧后再处理下一帧
// ============================================================================
void run_sequential(const PipelineConfig &cfg, LatencyStats &stats);

// ============================================================================
// 流水线执行: 多个帧在阶段工作线程中同时运行
// ============================================================================
class PipelinedExecutor {
public:
    PipelinedExecutor(const PipelineConfig &cfg, LatencyStats &stats);
    void run();

private:
    void perception_worker();
    void planning_worker();
    void control_worker();

    const PipelineConfig &cfg_;
    LatencyStats &stats_;

    std::queue<PipelineSensorData> input_queue_;
    std::mutex in_mutex_;
    std::condition_variable in_cv_;

    std::queue<PipelinePerceptionOut> perception_queue_;
    std::mutex pq_mutex_;
    std::condition_variable pq_cv_;

    std::queue<PipelinePlanningOut> planning_queue_;
    std::mutex lq_mutex_;
    std::condition_variable lq_cv_;

    std::atomic<bool> stop_{false};
};
