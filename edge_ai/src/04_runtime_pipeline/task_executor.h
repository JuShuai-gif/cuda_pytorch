#pragma once

#include "task_node.h"

#include <vector>
#include <chrono>
#include <cstdio>
#include <string>

// ============================================================================
// 高精度计时器，用于执行性能分析
// ============================================================================
class Timer {
public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    double elapsed_us() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(end - start_).count();
    }
    double elapsed_ms() const {
        return elapsed_us() / 1000.0;
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

// ============================================================================
// TaskGraphExecutor: 线程池 DAG 任务调度器
// ============================================================================
class TaskGraphExecutor {
public:
    explicit TaskGraphExecutor(int num_threads);

    void add_node(TaskNode node);
    void build_graph();
    void execute();

    void write_profile_json(const std::string &filepath,
                            int num_threads,
                            double wall_time_us) const;

    const std::vector<TaskNode> &nodes() const;

    // 执行后访问流水线上下文
    PipelineContext &context();
    const PipelineContext &context() const;

private:
    int num_threads_;
    std::vector<TaskNode> nodes_;
    PipelineContext ctx_;
};

// ============================================================================
// 机器人流水线节点工厂函数
// ============================================================================

// 节点：生成合成相机 + LiDAR 传感器数据
void node_sensor_capture(PipelineContext &ctx);

// 节点：将 1920x1080 RGB 缩放为 640x480 灰度图，并归一化
void node_image_preprocess(PipelineContext &ctx);

// 节点：点云距离滤波 + 体素降采样
void node_lidar_preprocess(PipelineContext &ctx);

// 节点：图像滑动窗口边缘检测，点云聚类
void node_detection(PipelineContext &ctx);

// 节点：跟踪目标的卡尔曼滤波状态更新
void node_tracking(PipelineContext &ctx);

// 节点：A* 网格搜索路径规划
void node_planning(PipelineContext &ctx);

// 节点：根据轨迹计算 PID 控制指令
void node_control(PipelineContext &ctx);
