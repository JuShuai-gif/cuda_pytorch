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

    // 流水线三阶段数据流：
    //   传感器数据(input) → 感知(perception) → 规划(planning) → 控制(control)
    //
    // 每对 queue + mutex + condition_variable 构成一个"生产者-消费者"通道：
    //   - queue:    存放中间数据（上一阶段产出，下一阶段消费）
    //   - mutex:    保护 queue 的并发访问（生产者写入 / 消费者取出）
    //   - cv:       条件变量，解决"队列空时消费者傻等（忙轮询）"的问题
    //              消费者在队列空时 cv.wait() 让出 CPU 进入休眠
    //              生产者 push 后 cv.notify_one() 唤醒一个等待的消费者
    //              既避免空转浪费 CPU，又保证数据一到立即处理

    // 输入通道：main 线程生产原始传感器帧 → perception_worker 消费
    std::queue<PipelineSensorData> input_queue_;
    std::mutex in_mutex_;
    std::condition_variable in_cv_;

    // 中间通道：perception_worker 生产检测结果 → planning_worker 消费
    std::queue<PipelinePerceptionOut> perception_queue_;
    std::mutex pq_mutex_;
    std::condition_variable pq_cv_;

    // 输出通道：planning_worker 生产轨迹 → control_worker 消费
    std::queue<PipelinePlanningOut> planning_queue_;
    std::mutex lq_mutex_;
    std::condition_variable lq_cv_;

    std::atomic<bool> stop_{false};
};
