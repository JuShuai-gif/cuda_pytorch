#pragma once

#include "pipeline_config.h"

#include <cstdint>
#include <vector>

// ============================================================================
// 纳秒级高精度时间戳
// ============================================================================
int64_t now_ns();

// ============================================================================
// 传感器数据生成
// ============================================================================

CameraImage generate_camera_image(int width, int height, int channels);
PointCloud generate_lidar_point_cloud(int num_points, int num_rings);

// ============================================================================
// 感知阶段: 处理原始传感器数据并生成检测结果
// ============================================================================

// RGB -> 灰度转换
std::vector<float> rgb_to_grayscale(const CameraImage &image);

// 双线性缩放，从 src_w x src_h 缩放到 dst_w x dst_h
std::vector<float> bilinear_resize(const std::vector<float> &src,
                                   int src_w, int src_h,
                                   int dst_w, int dst_h);

// 归一化像素值到 [0, 1]
void normalize_to_unit(std::vector<float> &data);

// 按距离范围 [min_range, max_range] 过滤点云
PointCloud filter_by_range(const PointCloud &cloud,
                           float min_range, float max_range);

// 使用给定体素大小 (米) 进行体素网格降采样
PointCloud voxel_downsample(const PointCloud &cloud, float voxel_size);

// 完整感知流水线: 相机 + 激光雷达处理，输出 Detections
Detections run_perception(const PipelineSensorData &sensor,
                          int64_t *out_preprocess_ns = nullptr,
                          int64_t *out_lidar_ns = nullptr,
                          int64_t *out_detection_ns = nullptr);

// ============================================================================
// 规划阶段: 基于检测结果的跟踪与轨迹生成
// ============================================================================

// 从检测结果初始化卡尔曼跟踪
std::vector<KalmanTrack> init_kalman_tracks(const Detections &detections);

// 卡尔曼滤波预测步 (恒定速度模型)
void kalman_predict(std::vector<KalmanTrack> &tracks, float dt_s);

// 卡尔曼滤波测量更新步
void kalman_update(KalmanTrack &track, float mx, float my);

// 计算每个跟踪目标的避碰时间 (秒)
std::vector<float> compute_ttc(const std::vector<KalmanTrack> &tracks,
                               float ego_speed);

// 通过航点生成三次样条轨迹
Trajectory generate_cubic_spline(const std::vector<Waypoint> &control_points,
                                 int num_samples);

// 完整规划流水线
PipelinePlanningOut run_planning(const PipelinePerceptionOut &perception);

// ============================================================================
// 控制阶段: 基于 PID 的轨迹跟踪
// ============================================================================

// 计算横向误差 (点到线段的符号距离)
float cross_track_error(float px, float py,
                        float ax, float ay,
                        float bx, float by);

// PID 控制器状态
struct PIDState {
    float kp = 1.0f;
    float ki = 0.0f;
    float kd = 0.0f;
    float integral = 0.0f;
    float prev_error = 0.0f;
    float integral_limit = 1.0f;
};

// 单步 PID 计算
float pid_step(PIDState &pid, float error, float dt_s);

// 完整控制流水线
PipelineControlOut run_control(const PipelinePlanningOut &planning);
