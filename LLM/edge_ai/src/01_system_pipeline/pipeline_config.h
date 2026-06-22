#pragma once

#include <vector>
#include <cstdint>
#include <string>

// ============================================================================
// 带纳秒时间戳的真实机器人数据结构
// ============================================================================

struct CameraImage {
    int width = 0;
    int height = 0;
    int channels = 0;
    int64_t timestamp_ns = 0;
    std::vector<uint8_t> data;
    std::string encoding; // "rgb8" 或 "bayer_rggb"
};

struct Point3D {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float intensity = 0.0f;
    int ring = 0;
};

struct PointCloud {
    int64_t timestamp_ns = 0;
    std::vector<Point3D> points;
};

struct DetectionBox {
    int class_id = 0; // 0=汽车, 1=行人, 2=骑行者, 3=卡车
    float confidence = 0.0f;
    float x = 0.0f, y = 0.0f, z = 0.0f;
    float w = 0.0f, h = 0.0f, d = 0.0f;
    float vx = 0.0f, vy = 0.0f;
};

struct Detections {
    int64_t timestamp_ns = 0;
    int frame_id = 0;
    std::vector<DetectionBox> boxes;
};

struct Waypoint {
    float x = 0.0f;
    float y = 0.0f;
    float t = 0.0f; // 沿样条线的参数化位置 [0, 1]
    float v = 0.0f; // 该航点目标速度 (m/s)
};

struct Trajectory {
    int64_t timestamp_ns = 0;
    std::vector<Waypoint> waypoints;
};

struct ControlCommand {
    float throttle = 0.0f; // 0..1
    float brake = 0.0f;    // 0..1
    float steering = 0.0f; // -1..1 (负=左转, 正=右转)
    int64_t timestamp_ns = 0;
};

// ============================================================================
// 用于目标跟踪的卡尔曼滤波状态 (恒定速度模型)
// ============================================================================

struct KalmanTrack {
    int track_id = 0;
    int class_id = 0;
    float confidence = 0.0f;
    // 状态: [x, y, vx, vy]
    float x = 0.0f, y = 0.0f, vx = 0.0f, vy = 0.0f;
    // 协方差矩阵 (4x4, 行主序)
    float P[16] = {};
    int age = 0;
    int missed = 0;
};

// ============================================================================
// 流水线数据流类型 (由执行器队列使用)
// ============================================================================

struct PipelineSensorData {
    int frame_id = 0;
    int64_t timestamp_ns = 0;
    CameraImage camera_image;
    PointCloud point_cloud;
};

struct PipelinePerceptionOut {
    int frame_id = 0;
    int64_t timestamp_ns = 0;
    int64_t e2e_start_ns = 0;
    Detections detections;
    int64_t perception_time_ns = 0;
};

struct PipelinePlanningOut {
    int frame_id = 0;
    int64_t timestamp_ns = 0;
    int64_t e2e_start_ns = 0;
    Trajectory trajectory;
    std::vector<KalmanTrack> tracks;
    int64_t planning_time_ns = 0;
};

struct PipelineControlOut {
    int frame_id = 0;
    int64_t timestamp_ns = 0;
    int64_t e2e_start_ns = 0;
    ControlCommand command;
    int64_t control_time_ns = 0;
};

// ============================================================================
// 流水线配置
// ============================================================================

struct PipelineConfig {
    int pipeline_depth = 3;
    int num_frames = 100;
    bool verbose = false;
    int stats_interval_frames = 20;
};
