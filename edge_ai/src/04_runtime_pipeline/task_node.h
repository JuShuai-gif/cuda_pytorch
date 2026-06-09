#pragma once

#include <string>
#include <vector>
#include <functional>
#include <cstdint>

// ============================================================================
// 机器人任务节点之间共享的数据结构
// ============================================================================

struct TaskCameraImage {
    int width = 0;
    int height = 0;
    int channels = 0;
    int64_t timestamp_ns = 0;
    std::vector<uint8_t> data;
};

struct TaskPoint3D {
    float x = 0.0f, y = 0.0f, z = 0.0f;
    float intensity = 0.0f;
    int ring = 0;
};

struct TaskPointCloud {
    int64_t timestamp_ns = 0;
    std::vector<TaskPoint3D> points;
};

struct TaskPreprocessedImage {
    int width = 0;
    int height = 0;
    std::vector<float> data; // 归一化 [0, 1]
};

struct TaskDetectionBox {
    int class_id = 0;
    float confidence = 0.0f;
    float x = 0.0f, y = 0.0f, z = 0.0f;
    float w = 0.0f, h = 0.0f, d = 0.0f;
};

struct TaskDetections {
    std::vector<TaskDetectionBox> boxes;
};

struct TaskKalmanTrack {
    float x = 0.0f, y = 0.0f, vx = 0.0f, vy = 0.0f;
    float P[16] = {};
};

struct TaskTrackingResult {
    std::vector<TaskKalmanTrack> tracks;
};

struct TaskWaypoint {
    float x = 0.0f, y = 0.0f;
};

struct TaskTrajectory {
    std::vector<TaskWaypoint> path;
};

struct TaskControlCommand {
    float throttle = 0.0f;
    float brake = 0.0f;
    float steering = 0.0f;
};

// ============================================================================
// 共享流水线上下文（任务节点之间传递的数据）
// ============================================================================

struct PipelineContext {
    TaskCameraImage camera_image;
    TaskPointCloud point_cloud;
    TaskPreprocessedImage preprocessed_image;
    TaskPointCloud filtered_cloud;
    TaskDetections detections;
    TaskTrackingResult tracking_result;
    TaskTrajectory trajectory;
    TaskControlCommand control_cmd;
    std::vector<std::vector<int>> occupancy_grid;
};

// ============================================================================
// TaskNode: DAG 任务图中的工作单元
// ============================================================================
struct TaskNode {
    std::string name;
    std::function<void()> work;
    std::vector<int> dependencies;
    std::vector<int> dependents;
    int indegree = 0;
    double elapsed_us = 0.0;
    bool completed = false;

    TaskNode() = default;
    TaskNode(std::string n, std::vector<int> deps, std::function<void()> w) : name(std::move(n)), work(std::move(w)), dependencies(std::move(deps)) {
    }
};
