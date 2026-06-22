#pragma once

#include <cstdint>
#include <vector>
#include <array>
#include <map>

// ============================================================================
// 传感器数据结构（逼真的机器人传感器数据）
// ============================================================================

struct Point3D {
    float x, y, z;
    float intensity;
};

struct CameraImage {
    static constexpr int WIDTH = 1920;
    static constexpr int HEIGHT = 1080;
    static constexpr int CHANNELS = 3;
    std::vector<uint8_t> data; // WIDTH * HEIGHT * CHANNELS，行主序
};

struct IMUReading {
    float ax, ay, az; // 加速度（m/s^2）
    float gx, gy, gz; // 角速度（rad/s）
};

struct GPSReading {
    double lat, lon, alt; // 纬度, 经度, 海拔高度
};

// ============================================================================
// 流水线各阶段输出
// ============================================================================

// 阶段 1：原始传感器帧
struct SensorFrame {
    int32_t frame_id = 0;
    int64_t timestamp_ns = 0;
    CameraImage camera_image;
    std::vector<Point3D> lidar_points;
    IMUReading imu_accel;
    IMUReading imu_gyro;
    GPSReading gps;
};

// 阶段 2：预处理后数据
struct PreprocessedData {
    int32_t frame_id = 0;
    int64_t timestamp_ns = 0;

    // 图像：640x480x3 float32，归一化到 [0,1]
    static constexpr int IMG_W = 640;
    static constexpr int IMG_H = 480;
    static constexpr int IMG_C = 3;
    std::vector<float> image_tensor; // IMG_H * IMG_W * IMG_C

    // 激光雷达：体素网格，将体素索引映射到点索引列表（相对于
    // 存储在一旁的下采样主数组）
    struct VoxelKey {
        int32_t ix, iy, iz;
        bool operator<(const VoxelKey &o) const {
            if (ix != o.ix) return ix < o.ix;
            if (iy != o.iy) return iy < o.iy;
            return iz < o.iz;
        }
    };
    std::map<VoxelKey, std::vector<Point3D>> lidar_voxels;
    // 地面平面：ax + by + cz + d = 0
    std::array<float, 4> ground_plane = {0, 0, 1, 0};
};

// 阶段 3：检测结果
struct DetectionResult {
    int32_t frame_id = 0;
    int64_t timestamp_ns = 0;
    int32_t num_detections = 0;

    struct Box3D {
        int32_t class_id = 0; // 0=汽车, 1=行人, 2=骑行者, 3=卡车
        float confidence = 0.0f;
        float x = 0, y = 0, z = 0;              // 中心点
        float width = 0, height = 0, depth = 0; // 尺寸
        float vx = 0, vy = 0;                   // 估计速度
    };
    std::vector<Box3D> boxes;
};

// 阶段 4：跟踪结果（卡尔曼滤波跟踪）
struct TrackingResult {
    int32_t frame_id = 0;
    int64_t timestamp_ns = 0;
    int32_t num_tracks = 0;

    struct Track {
        int32_t track_id = 0;
        int32_t age = 0; // 自创建以来的帧数
        // 状态：[x, y, z, vx, vy, vz, ax, ay, az]
        std::array<float, 9> state = {};
        // 协方差：9x9 矩阵，行主序存储
        std::array<float, 81> covariance = {};
    };
    std::vector<Track> tracks;
};

// 阶段 5：预测结果
struct PredictionResult {
    int32_t frame_id = 0;
    int64_t timestamp_ns = 0;

    struct Waypoint {
        float x, y;
        float t; // 从当前时刻起的秒数
    };

    struct PredictedTrajectory {
        int32_t track_id = 0;
        std::vector<Waypoint> waypoints; // 5 秒时域内的 100 个时间步
    };
    std::vector<PredictedTrajectory> trajectories;
};

// 阶段 6：规划结果
struct PlanningResult {
    int32_t frame_id = 0;
    int64_t timestamp_ns = 0;

    struct EgoWaypoint {
        float x, y, theta; // 位置 + 航向角
        float v;           // 目标速度
        float t;           // 从当前时刻起的秒数
    };
    std::vector<EgoWaypoint> ego_trajectory;
    std::vector<float> cost_map; // 100x100 网格，行主序
};

// 阶段 7：控制指令
struct ControlCommand {
    int32_t frame_id = 0;
    int64_t timestamp_ns = 0;
    float throttle = 0.0f;        // [0, 1]
    float brake = 0.0f;           // [0, 1]
    float steering = 0.0f;        // [-1, 1]
    int32_t gear = 1;             // 1=前进, 0=停车, -1=倒车
    float target_velocity = 0.0f; // m/s
};
