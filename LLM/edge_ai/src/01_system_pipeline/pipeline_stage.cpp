#include "pipeline_stage.h"

#include <cmath>
#include <cstdlib>
#include <random>
#include <algorithm>
#include <unordered_map>
#include <functional>
#include <vector>
#include <chrono>

// ============================================================================
// 时间戳
// ============================================================================

static std::mt19937 &rng() {
    static std::mt19937 instance(std::random_device{}());
    return instance;
}

int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
}

// ============================================================================
// 传感器数据生成
// ============================================================================

CameraImage generate_camera_image(int width, int height, int channels) {
    CameraImage img;
    img.width = width;
    img.height = height;
    img.channels = channels;
    img.timestamp_ns = now_ns();
    img.encoding = "rgb8";
    img.data.resize(width * height * channels);

    std::uniform_int_distribution<int> dist(0, 255);
    for (size_t i = 0; i < img.data.size(); ++i) {
        img.data[i] = static_cast<uint8_t>(dist(rng()));
    }
    return img;
}

PointCloud generate_lidar_point_cloud(int num_points, int num_rings) {
    PointCloud cloud;
    cloud.timestamp_ns = now_ns();

    const float fov_up = 15.0f * static_cast<float>(M_PI) / 180.0f;
    const float fov_down = -25.0f * static_cast<float>(M_PI) / 180.0f;
    const int points_per_ring = num_points / num_rings;

    std::uniform_real_distribution<float> range_dist(0.5f, 120.0f);
    std::uniform_real_distribution<float> intensity_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> noise_dist(-0.02f, 0.02f);

    for (int ring = 0; ring < num_rings; ++ring) {
        float v_angle = fov_down + (fov_up - fov_down) * static_cast<float>(ring) / static_cast<float>(num_rings - 1);
        for (int i = 0; i < points_per_ring; ++i) {
            float h_angle = 2.0f * static_cast<float>(M_PI) * static_cast<float>(i) / static_cast<float>(points_per_ring);
            float range = range_dist(rng());
            float x = range * std::cos(v_angle) * std::cos(h_angle) + noise_dist(rng());
            float y = range * std::cos(v_angle) * std::sin(h_angle) + noise_dist(rng());
            float z = range * std::sin(v_angle) + noise_dist(rng());
            float intensity = intensity_dist(rng());
            cloud.points.push_back({x, y, z, intensity, ring});
        }
    }
    return cloud;
}

// ============================================================================
// 图像预处理: RGB -> 灰度
// ============================================================================

std::vector<float> rgb_to_grayscale(const CameraImage &image) {
    int total_pixels = image.width * image.height;
    std::vector<float> gray(total_pixels);

    // ITU-R BT.601 亮度系数
    for (int i = 0; i < total_pixels; ++i) {
        float r = static_cast<float>(image.data[i * 3 + 0]);
        float g = static_cast<float>(image.data[i * 3 + 1]);
        float b = static_cast<float>(image.data[i * 3 + 2]);
        gray[i] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
    return gray;
}

// ============================================================================
// 图像预处理: 双线性缩放
// ============================================================================

std::vector<float> bilinear_resize(const std::vector<float> &src,
                                   int src_w, int src_h,
                                   int dst_w, int dst_h) {
    std::vector<float> dst(dst_w * dst_h, 0.0f);
    float scale_x = static_cast<float>(src_w) / static_cast<float>(dst_w);
    float scale_y = static_cast<float>(src_h) / static_cast<float>(dst_h);

    for (int dy = 0; dy < dst_h; ++dy) {
        float sy = static_cast<float>(dy) * scale_y;
        int y0 = static_cast<int>(sy);
        int y1 = std::min(y0 + 1, src_h - 1);
        float y_frac = sy - static_cast<float>(y0);

        for (int dx = 0; dx < dst_w; ++dx) {
            float sx = static_cast<float>(dx) * scale_x;
            int x0 = static_cast<int>(sx);
            int x1 = std::min(x0 + 1, src_w - 1);
            float x_frac = sx - static_cast<float>(x0);

            float top = src[y0 * src_w + x0] * (1.0f - x_frac) + src[y0 * src_w + x1] * x_frac;
            float bot = src[y1 * src_w + x0] * (1.0f - x_frac) + src[y1 * src_w + x1] * x_frac;
            dst[dy * dst_w + dx] = top * (1.0f - y_frac) + bot * y_frac;
        }
    }
    return dst;
}

// ============================================================================
// 图像预处理: 归一化到 [0, 1]
// ============================================================================

void normalize_to_unit(std::vector<float> &data) {
    float factor = 1.0f / 255.0f;
    for (float &v : data) {
        v *= factor;
    }
}

// ============================================================================
// 点云: 距离过滤
// ============================================================================

PointCloud filter_by_range(const PointCloud &cloud,
                           float min_range, float max_range) {
    PointCloud filtered;
    filtered.timestamp_ns = cloud.timestamp_ns;
    for (const auto &p : cloud.points) {
        float dist = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
        if (dist >= min_range && dist <= max_range) {
            filtered.points.push_back(p);
        }
    }
    return filtered;
}

// ============================================================================
// 点云: 体素网格降采样
// ============================================================================

struct VoxelKey {
    int ix, iy, iz;
    bool operator==(const VoxelKey &o) const {
        return ix == o.ix && iy == o.iy && iz == o.iz;
    }
};

struct VoxelKeyHash {
    size_t operator()(const VoxelKey &k) const {
        return (static_cast<size_t>(k.ix) * 73856093) ^ (static_cast<size_t>(k.iy) * 19349663) ^ (static_cast<size_t>(k.iz) * 83492791);
    }
};

PointCloud voxel_downsample(const PointCloud &cloud, float voxel_size) {
    PointCloud downsampled;
    downsampled.timestamp_ns = cloud.timestamp_ns;
    float inv_vs = 1.0f / voxel_size;

    std::unordered_map<VoxelKey, Point3D, VoxelKeyHash> voxel_map;
    for (const auto &p : cloud.points) {
        VoxelKey key{
            static_cast<int>(std::floor(p.x * inv_vs)),
            static_cast<int>(std::floor(p.y * inv_vs)),
            static_cast<int>(std::floor(p.z * inv_vs))};
        if (voxel_map.find(key) == voxel_map.end()) {
            voxel_map[key] = p;
        }
    }

    downsampled.points.reserve(voxel_map.size());
    for (const auto &kv : voxel_map) {
        downsampled.points.push_back(kv.second);
    }
    return downsampled;
}

// ============================================================================
// 基于预处理数据的检测生成
// ============================================================================

static Detections generate_detections(int frame_id, int num_objects) {
    Detections det;
    det.timestamp_ns = now_ns();
    det.frame_id = frame_id;

    std::uniform_real_distribution<float> pos_dist(-50.0f, 50.0f);
    std::uniform_real_distribution<float> size_dist(1.0f, 5.0f);
    std::uniform_real_distribution<float> conf_dist(0.5f, 1.0f);
    std::uniform_real_distribution<float> vel_dist(-15.0f, 15.0f);
    std::uniform_int_distribution<int> class_dist(0, 3);

    for (int i = 0; i < num_objects; ++i) {
        DetectionBox box;
        box.class_id = class_dist(rng());
        box.confidence = conf_dist(rng());
        box.x = pos_dist(rng());
        box.y = pos_dist(rng());
        box.z = 0.0f;
        box.w = size_dist(rng());
        box.h = size_dist(rng());
        box.d = size_dist(rng());
        box.vx = vel_dist(rng());
        box.vy = vel_dist(rng());
        det.boxes.push_back(box);
    }
    return det;
}

// ============================================================================
// 完整感知流水线
// ============================================================================

Detections run_perception(const PipelineSensorData &sensor,
                          int64_t *out_preprocess_ns,
                          int64_t *out_lidar_ns,
                          int64_t *out_detection_ns) {
    int64_t t0 = now_ns();

    // 相机预处理: 灰度化 + 缩放 + 归一化
    int64_t t1 = now_ns();
    std::vector<float> gray = rgb_to_grayscale(sensor.camera_image);
    std::vector<float> resized = bilinear_resize(gray,
                                                 sensor.camera_image.width,
                                                 sensor.camera_image.height,
                                                 640, 480);
    normalize_to_unit(resized);
    int64_t t2 = now_ns();
    if (out_preprocess_ns) *out_preprocess_ns = t2 - t1;

    // 激光雷达预处理: 距离过滤 + 体素降采样
    int64_t t3 = now_ns();
    PointCloud filtered = filter_by_range(sensor.point_cloud, 0.5f, 100.0f);
    PointCloud downsampled = voxel_downsample(filtered, 0.1f);
    int64_t t4 = now_ns();
    if (out_lidar_ns) *out_lidar_ns = t4 - t3;

    // 检测 (合成，基于处理后数据密度)
    int64_t t5 = now_ns();
    int num_objects = 3 + static_cast<int>(downsampled.points.size()) / 500;
    Detections detections = generate_detections(sensor.frame_id,
                                                std::max(1, std::min(num_objects, 20)));
    int64_t t6 = now_ns();
    if (out_detection_ns) *out_detection_ns = t6 - t5;

    (void)t0;
    return detections;
}

// ============================================================================
// 卡尔曼滤波实现 (恒定速度模型)
// ============================================================================

std::vector<KalmanTrack> init_kalman_tracks(const Detections &detections) {
    std::vector<KalmanTrack> tracks;
    for (size_t i = 0; i < detections.boxes.size(); ++i) {
        const auto &box = detections.boxes[i];
        KalmanTrack track;
        track.track_id = static_cast<int>(i);
        track.class_id = box.class_id;
        track.confidence = box.confidence;
        track.x = box.x;
        track.y = box.y;
        track.vx = box.vx;
        track.vy = box.vy;
        // 初始化协方差为单位矩阵 * 0.1
        for (int j = 0; j < 16; ++j) track.P[j] = 0.0f;
        track.P[0] = 0.1f;
        track.P[5] = 0.1f;
        track.P[10] = 0.05f;
        track.P[15] = 0.05f;
        track.age = 1;
        track.missed = 0;
        tracks.push_back(track);
    }
    return tracks;
}

void kalman_predict(std::vector<KalmanTrack> &tracks, float dt_s) {
    // 状态转移矩阵 F (恒定速度)
    //   x' = x + vx*dt
    //   y' = y + vy*dt
    //   vx' = vx
    //   vy' = vy
    float F[16] = {
        1.0f, 0.0f, dt_s, 0.0f,
        0.0f, 1.0f, 0.0f, dt_s,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f};

    // 过程噪声 Q = 小对角矩阵
    float q_pos = 0.01f;
    float q_vel = 0.005f;
    float Q[16] = {
        q_pos, 0.0f, 0.0f, 0.0f,
        0.0f, q_pos, 0.0f, 0.0f,
        0.0f, 0.0f, q_vel, 0.0f,
        0.0f, 0.0f, 0.0f, q_vel};

    for (auto &track : tracks) {
        // 预测状态: x = F * x
        float nx = F[0] * track.x + F[1] * track.y + F[2] * track.vx + F[3] * track.vy;
        float ny = F[4] * track.x + F[5] * track.y + F[6] * track.vx + F[7] * track.vy;
        float nvx = F[8] * track.x + F[9] * track.y + F[10] * track.vx + F[11] * track.vy;
        float nvy = F[12] * track.x + F[13] * track.y + F[14] * track.vx + F[15] * track.vy;
        track.x = nx;
        track.y = ny;
        track.vx = nvx;
        track.vy = nvy;

        // 预测协方差: P = F * P * F^T
        float FP[16] = {};
        for (int i = 0; i < 4; ++i)
            for (int k = 0; k < 4; ++k)
                for (int j = 0; j < 4; ++j)
                    FP[i * 4 + j] += F[i * 4 + k] * track.P[k * 4 + j];

        float FPFt[16] = {};
        for (int i = 0; i < 4; ++i)
            for (int k = 0; k < 4; ++k)
                for (int j = 0; j < 4; ++j)
                    FPFt[i * 4 + j] += FP[i * 4 + k] * F[j * 4 + k]; // F[j*4+k] 即 F^T[k,j]

        // P = FPFt + Q
        for (int i = 0; i < 16; ++i) {
            track.P[i] = FPFt[i] + Q[i];
        }

        track.age++;
    }
}

static bool invert_2x2(float a, float b, float c, float d, float out[4]) {
    float det = a * d - b * c;
    if (std::fabs(det) < 1e-10f) return false;
    float inv_det = 1.0f / det;
    out[0] = d * inv_det;
    out[1] = -b * inv_det;
    out[2] = -c * inv_det;
    out[3] = a * inv_det;
    return true;
}

void kalman_update(KalmanTrack &track, float mx, float my) {
    // 测量矩阵 H = [[1,0,0,0], [0,1,0,0]]
    // 新息 y = z - H*x
    float y_innov[2] = {mx - track.x, my - track.y};

    // S = H * P * H^T + R
    // H*P*H^T 即 P 的左上 2x2 子矩阵
    float meas_noise = 0.05f;
    float S[4] = {
        track.P[0] + meas_noise, track.P[1],
        track.P[4], track.P[5] + meas_noise};

    float S_inv[4];
    if (!invert_2x2(S[0], S[1], S[2], S[3], S_inv)) return;

    // K = P * H^T * S_inv
    // P * H^T 即 P 的第 0 列和第 1 列 (前两列)
    float K[8] = {};
    for (int i = 0; i < 4; ++i) {
        K[i * 2 + 0] = track.P[i * 4 + 0] * S_inv[0] + track.P[i * 4 + 1] * S_inv[2];
        K[i * 2 + 1] = track.P[i * 4 + 0] * S_inv[1] + track.P[i * 4 + 1] * S_inv[3];
    }

    // x = x + K * y
    track.x += K[0] * y_innov[0] + K[1] * y_innov[1];
    track.y += K[2] * y_innov[0] + K[3] * y_innov[1];
    track.vx += K[4] * y_innov[0] + K[5] * y_innov[1];
    track.vy += K[6] * y_innov[0] + K[7] * y_innov[1];

    // P = (I - K*H) * P
    float I_KH[16] = {
        1.0f - K[0], -K[1], 0.0f, 0.0f,
        -K[2], 1.0f - K[3], 0.0f, 0.0f,
        -K[4], -K[5], 1.0f, 0.0f,
        -K[6], -K[7], 0.0f, 1.0f};

    float newP[16] = {};
    for (int i = 0; i < 4; ++i)
        for (int k = 0; k < 4; ++k)
            for (int j = 0; j < 4; ++j)
                newP[i * 4 + j] += I_KH[i * 4 + k] * track.P[k * 4 + j];

    for (int i = 0; i < 16; ++i) track.P[i] = newP[i];
}

// ============================================================================
// 避碰时间计算
// ============================================================================

std::vector<float> compute_ttc(const std::vector<KalmanTrack> &tracks,
                               float ego_speed) {
    std::vector<float> ttc_values;
    for (const auto &t : tracks) {
        float dist = std::sqrt(t.x * t.x + t.y * t.y);
        float rel_speed = std::max(0.1f,
                                   ego_speed - std::sqrt(t.vx * t.vx + t.vy * t.vy));
        ttc_values.push_back(dist / rel_speed);
    }
    return ttc_values;
}

// ============================================================================
// 三次样条轨迹生成
// ============================================================================

Trajectory generate_cubic_spline(const std::vector<Waypoint> &control_points,
                                 int num_samples) {
    Trajectory traj;
    traj.timestamp_ns = now_ns();

    if (control_points.size() < 2) {
        traj.waypoints = control_points;
        return traj;
    }

    int n = static_cast<int>(control_points.size());
    // 自然三次样条: 求解三对角方程组以得到二阶导数
    std::vector<float> h(n - 1);
    for (int i = 0; i < n - 1; ++i) {
        h[i] = control_points[i + 1].t - control_points[i].t;
    }

    std::vector<float> alpha(n, 0.0f);
    for (int i = 1; i < n - 1; ++i) {
        float dx_a = control_points[i].x - control_points[i - 1].x;
        float dx_b = control_points[i + 1].x - control_points[i].x;
        float dy_a = control_points[i].y - control_points[i - 1].y;
        float dy_b = control_points[i + 1].y - control_points[i].y;
        // 使用欧几里得弧长进行参数化
        float arc_a = std::sqrt(dx_a * dx_a + dy_a * dy_a);
        float arc_b = std::sqrt(dx_b * dx_b + dy_b * dy_b);
        alpha[i] = 6.0f * (arc_b / h[i] - arc_a / h[i - 1]) / (h[i] + h[i - 1]);
    }

    // 追赶法求解三对角方程组
    std::vector<float> l(n, 1.0f);
    std::vector<float> mu(n, 0.0f);
    std::vector<float> z(n, 0.0f);
    std::vector<float> m_x(n, 0.0f);
    std::vector<float> m_y(n, 0.0f);

    // x 的前向消元
    {
        std::vector<float> alpha_x(alpha);
        for (int i = 1; i < n - 1; ++i) {
            l[i] = 2.0f * (control_points[i + 1].t - control_points[i - 1].t) - h[i - 1] * mu[i - 1];
            mu[i] = h[i] / l[i];
            z[i] = (alpha_x[i] - h[i - 1] * z[i - 1]) / l[i];
        }
        // 回代
        for (int i = n - 2; i >= 0; --i) {
            m_x[i] = z[i] - mu[i] * m_x[i + 1];
        }
    }

    // y 的前向消元 (复用 x 消元中的 l、mu)
    {
        std::vector<float> alpha_y(n, 0.0f);
        for (int i = 1; i < n - 1; ++i) {
            alpha_y[i] = 6.0f * ((control_points[i + 1].y - control_points[i].y) / h[i] - (control_points[i].y - control_points[i - 1].y) / h[i - 1]) / (h[i] + h[i - 1]);
        }
        std::vector<float> ly(n, 1.0f);
        std::vector<float> muy(n, 0.0f);
        std::vector<float> zy(n, 0.0f);
        for (int i = 1; i < n - 1; ++i) {
            ly[i] = 2.0f * (control_points[i + 1].t - control_points[i - 1].t) - h[i - 1] * muy[i - 1];
            muy[i] = h[i] / ly[i];
            zy[i] = (alpha_y[i] - h[i - 1] * zy[i - 1]) / ly[i];
        }
        for (int i = n - 2; i >= 0; --i) {
            m_y[i] = zy[i] - muy[i] * m_y[i + 1];
        }
    }

    // 对样条曲线进行采样
    traj.waypoints.reserve(num_samples);
    for (int s = 0; s < num_samples; ++s) {
        float t = control_points[0].t + (control_points[n - 1].t - control_points[0].t) * static_cast<float>(s) / static_cast<float>(num_samples - 1);

        // 查找所在的段
        int seg = 0;
        for (int i = 0; i < n - 1; ++i) {
            if (t <= control_points[i + 1].t + 1e-6f) {
                seg = i;
                break;
            }
        }
        if (seg >= n - 1) seg = n - 2;

        float h_seg = h[seg];
        float a = t - control_points[seg].t;
        float b = control_points[seg + 1].t - t;

        float x_val = (m_x[seg] * b * b * b + m_x[seg + 1] * a * a * a) / (6.0f * h_seg) + (control_points[seg].x / h_seg - m_x[seg] * h_seg / 6.0f) * b + (control_points[seg + 1].x / h_seg - m_x[seg + 1] * h_seg / 6.0f) * a;

        float y_val = (m_y[seg] * b * b * b + m_y[seg + 1] * a * a * a) / (6.0f * h_seg) + (control_points[seg].y / h_seg - m_y[seg] * h_seg / 6.0f) * b + (control_points[seg + 1].y / h_seg - m_y[seg + 1] * h_seg / 6.0f) * a;

        Waypoint wp;
        wp.x = x_val;
        wp.y = y_val;
        wp.t = t;
        wp.v = 10.0f; // 目标速度 10 m/s
        traj.waypoints.push_back(wp);
    }
    return traj;
}

// ============================================================================
// 完整规划流水线
// ============================================================================

PipelinePlanningOut run_planning(const PipelinePerceptionOut &perception) {
    PipelinePlanningOut result;
    result.frame_id = perception.frame_id;
    result.e2e_start_ns = perception.e2e_start_ns;

    int64_t t0 = now_ns();

    // 初始化/运行卡尔曼滤波器
    result.tracks = init_kalman_tracks(perception.detections);
    kalman_predict(result.tracks, 0.1f); // 100ms 预测步长
    for (auto &track : result.tracks) {
        kalman_update(track, track.x + 0.01f, track.y + 0.01f); // 微小人工更新
    }

    // 计算 TTC
    compute_ttc(result.tracks, 12.0f);

    // 构建样条控制点
    std::vector<Waypoint> control_points;
    for (int i = 0; i < 10; ++i) {
        Waypoint wp;
        wp.x = static_cast<float>(i) * 8.0f;
        wp.y = 3.0f * std::sin(static_cast<float>(i) * 0.3f);
        wp.t = static_cast<float>(i);
        wp.v = 10.0f + 2.0f * std::cos(static_cast<float>(i) * 0.5f);
        control_points.push_back(wp);
    }
    result.trajectory = generate_cubic_spline(control_points, 50);

    int64_t t1 = now_ns();
    result.planning_time_ns = t1 - t0;
    result.timestamp_ns = t1;
    return result;
}

// ============================================================================
// 控制: 横向误差
// ============================================================================

float cross_track_error(float px, float py,
                        float ax, float ay,
                        float bx, float by) {
    float dx = bx - ax;
    float dy = by - ay;
    float len_sq = dx * dx + dy * dy;
    if (len_sq < 1e-10f) {
        float ex = px - ax;
        float ey = py - ay;
        return std::sqrt(ex * ex + ey * ey);
    }
    float t = ((px - ax) * dx + (py - ay) * dy) / len_sq;
    t = std::max(0.0f, std::min(1.0f, t));
    float proj_x = ax + t * dx;
    float proj_y = ay + t * dy;
    float ex = px - proj_x;
    float ey = py - proj_y;
    return std::sqrt(ex * ex + ey * ey);
}

// ============================================================================
// PID 控制器单步计算
// ============================================================================

float pid_step(PIDState &pid, float error, float dt_s) {
    pid.integral += error * dt_s;
    pid.integral = std::max(-pid.integral_limit,
                            std::min(pid.integral_limit, pid.integral));
    float derivative = (error - pid.prev_error) / dt_s;
    pid.prev_error = error;
    return pid.kp * error + pid.ki * pid.integral + pid.kd * derivative;
}

// ============================================================================
// 完整控制流水线
// ============================================================================

PipelineControlOut run_control(const PipelinePlanningOut &planning) {
    PipelineControlOut result;
    result.frame_id = planning.frame_id;
    result.e2e_start_ns = planning.e2e_start_ns;

    int64_t t0 = now_ns();

    // 当前自车位姿 (模拟为略微偏离目标)
    float ego_x = 2.0f;
    float ego_y = 0.5f;
    float ego_speed = 11.0f;

    // 查找最近的航点
    float min_dist = 1e9f;
    size_t closest_idx = 0;
    for (size_t i = 0; i < planning.trajectory.waypoints.size(); ++i) {
        float dx = ego_x - planning.trajectory.waypoints[i].x;
        float dy = ego_y - planning.trajectory.waypoints[i].y;
        float d = dx * dx + dy * dy;
        if (d < min_dist) {
            min_dist = d;
            closest_idx = i;
        }
    }

    // 计算到下一段轨迹的横向误差
    float cte = 0.0f;
    if (closest_idx + 1 < planning.trajectory.waypoints.size()) {
        cte = cross_track_error(ego_x, ego_y,
                                planning.trajectory.waypoints[closest_idx].x,
                                planning.trajectory.waypoints[closest_idx].y,
                                planning.trajectory.waypoints[closest_idx + 1].x,
                                planning.trajectory.waypoints[closest_idx + 1].y);
    }

    // 横向 PID (根据横向误差计算转向)
    static PIDState lateral_pid;
    lateral_pid.kp = 0.05f;
    lateral_pid.ki = 0.001f;
    lateral_pid.kd = 0.01f;
    lateral_pid.integral_limit = 0.5f;
    float steering_raw = pid_step(lateral_pid, cte, 0.05f);

    // 纵向 PID (速度控制)
    float target_speed = 10.0f;
    if (closest_idx < planning.trajectory.waypoints.size()) {
        target_speed = planning.trajectory.waypoints[closest_idx].v;
    }
    static PIDState longitudinal_pid;
    longitudinal_pid.kp = 0.3f;
    longitudinal_pid.ki = 0.01f;
    longitudinal_pid.kd = 0.05f;
    longitudinal_pid.integral_limit = 0.3f;
    float speed_error = target_speed - ego_speed;
    float accel_raw = pid_step(longitudinal_pid, speed_error, 0.05f);

    // 转换为车辆控制命令
    result.command.throttle = std::max(0.0f, std::min(1.0f, accel_raw));
    result.command.brake = std::max(0.0f, std::min(1.0f, -accel_raw));
    result.command.steering = std::max(-1.0f, std::min(1.0f, steering_raw));
    result.command.timestamp_ns = now_ns();

    int64_t t1 = now_ns();
    result.control_time_ns = t1 - t0;
    result.timestamp_ns = t1;
    return result;
}
