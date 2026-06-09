#include "stage_ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <numeric>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>

// ============================================================================
// 工具函数：时间戳
// ============================================================================
int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
}

// ============================================================================
// 工具函数：生成 [lo, hi] 范围内的随机浮点数
// ============================================================================
static float rand_float(std::mt19937 &rng, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    return dist(rng);
}

// ============================================================================
// 阶段 1：传感器数据生成
// ============================================================================
SensorFrame run_sensor(int32_t frame_id, const PipelineConfig &cfg,
                       std::mt19937 &rng) {
    (void)cfg;
    SensorFrame frame;
    frame.frame_id = frame_id;
    frame.timestamp_ns = now_ns();

    // 相机：生成随机的 1920x1080x3 uint8 图像
    const int cam_size = CameraImage::WIDTH * CameraImage::HEIGHT * CameraImage::CHANNELS;
    frame.camera_image.data.resize(cam_size);
    std::uniform_int_distribution<uint8_t> pixel_dist(0, 255);
    for (int i = 0; i < cam_size; i++) {
        frame.camera_image.data[i] = pixel_dist(rng);
    }

    // 激光雷达：10 万个点，具有逼真的 360 度分布模式
    const int num_points = 100000;
    frame.lidar_points.reserve(num_points);
    std::normal_distribution<float> elev_dist(0.0f, 0.05f);
    std::exponential_distribution<float> range_dist(1.0f / 30.0f);
    std::uniform_real_distribution<float> azimuth_dist(0.0f, 2.0f * M_PI);
    std::uniform_real_distribution<float> intensity_dist(0.0f, 1.0f);

    for (int i = 0; i < num_points; i++) {
        Point3D pt;
        float az = azimuth_dist(rng);
        float el = elev_dist(rng);
        float dist = range_dist(rng);
        if (dist > 140.0f) dist = 140.0f;
        if (dist < 0.1f) dist = 0.1f;

        pt.x = dist * std::cos(el) * std::cos(az);
        pt.y = dist * std::cos(el) * std::sin(az);
        pt.z = dist * std::sin(el);
        pt.intensity = intensity_dist(rng);
        frame.lidar_points.push_back(pt);
    }

    // IMU：含小偏置的逼真数值
    std::normal_distribution<float> accel_dist(0.0f, 0.1f);
    std::normal_distribution<float> gyro_dist(0.0f, 0.01f);
    frame.imu_accel.ax = accel_dist(rng);
    frame.imu_accel.ay = accel_dist(rng);
    frame.imu_accel.az = 9.81f + accel_dist(rng); // 重力
    frame.imu_gyro.gx = gyro_dist(rng);
    frame.imu_gyro.gy = gyro_dist(rng);
    frame.imu_gyro.gz = gyro_dist(rng);

    // GPS
    frame.gps.lat = 37.7749 + rand_float(rng, -0.001, 0.001);
    frame.gps.lon = -122.4194 + rand_float(rng, -0.001, 0.001);
    frame.gps.alt = 10.0f + rand_float(rng, -1.0f, 1.0f);

    return frame;
}

// ============================================================================
// 阶段 2：预处理
// ============================================================================

// RGB 到 YUV 转换矩阵（BT.601）
static const float RGB2YUV[3][3] = {
    {0.299f, 0.587f, 0.114f},
    {-0.14713f, -0.28886f, 0.436f},
    {0.615f, -0.51499f, -0.10001f},
};

// 通过块平均实现简单的类双线性图像缩放
static void resize_image_block_avg(const uint8_t *src, int sw, int sh, int sc,
                                   float *dst, int dw, int dh) {
    int sx_ratio = sw / dw; // 1920 / 640 = 3
    int sy_ratio = sh / dh; // 1080 / 480 = 2

    for (int dy = 0; dy < dh; dy++) {
        for (int dx = 0; dx < dw; dx++) {
            // 对源图中对应的块求平均
            for (int c = 0; c < sc; c++) {
                float sum = 0.0f;
                int count = 0;
                for (int sy = dy * sy_ratio; sy < (dy + 1) * sy_ratio && sy < sh; sy++) {
                    for (int sx = dx * sx_ratio; sx < (dx + 1) * sx_ratio && sx < sw; sx++) {
                        int src_idx = (sy * sw + sx) * sc + c;
                        sum += static_cast<float>(src[src_idx]);
                        count++;
                    }
                }
                int dst_idx = (dy * dw + dx) * sc + c;
                dst[dst_idx] = (sum / static_cast<float>(count)) / 255.0f;
            }
        }
    }

    // 原地应用 RGB->YUV 转换
    for (int i = 0; i < dw * dh; i++) {
        float r = dst[i * sc + 0];
        float g = dst[i * sc + 1];
        float b = dst[i * sc + 2];

        float y = RGB2YUV[0][0] * r + RGB2YUV[0][1] * g + RGB2YUV[0][2] * b;
        float u = RGB2YUV[1][0] * r + RGB2YUV[1][1] * g + RGB2YUV[1][2] * b + 0.5f;
        float v = RGB2YUV[2][0] * r + RGB2YUV[2][1] * g + RGB2YUV[2][2] * b + 0.5f;

        dst[i * sc + 0] = std::max(0.0f, std::min(1.0f, y));
        dst[i * sc + 1] = std::max(0.0f, std::min(1.0f, u));
        dst[i * sc + 2] = std::max(0.0f, std::min(1.0f, v));
    }
}

// 简化的 RANSAC 地面平面拟合
// 返回平面 ax+by+cz+d=0 的 {a, b, c, d}
static std::array<float, 4> fit_ground_plane(const std::vector<Point3D> &points,
                                             std::mt19937 &rng) {
    const int n = static_cast<int>(points.size());
    if (n < 3) return {0, 0, 1, 0};

    const float dist_thresh = 0.2f;
    const int ransac_iters = 50;
    int best_inliers = 0;
    std::array<float, 4> best_plane = {0, 0, 1, 0};

    std::uniform_int_distribution<int> idx_dist(0, n - 1);

    for (int iter = 0; iter < ransac_iters; iter++) {
        int i1 = idx_dist(rng);
        int i2 = idx_dist(rng);
        int i3 = idx_dist(rng);
        if (i1 == i2 || i2 == i3 || i1 == i3) continue;

        const auto &p1 = points[i1];
        const auto &p2 = points[i2];
        const auto &p3 = points[i3];

        // 计算法向量 = (p2-p1) x (p3-p1)
        float ux = p2.x - p1.x, uy = p2.y - p1.y, uz = p2.z - p1.z;
        float vx = p3.x - p1.x, vy = p3.y - p1.y, vz = p3.z - p1.z;
        float nx = uy * vz - uz * vy;
        float ny = uz * vx - ux * vz;
        float nz = ux * vy - uy * vx;
        float len = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (len < 1e-6f) continue;
        nx /= len;
        ny /= len;
        nz /= len;
        // 确保法向量朝上
        if (nz < 0) {
            nx = -nx;
            ny = -ny;
            nz = -nz;
        }
        float d = -(nx * p1.x + ny * p1.y + nz * p1.z);

        // 统计内点
        int inliers = 0;
        for (int j = 0; j < n; j++) {
            float dist = std::abs(nx * points[j].x + ny * points[j].y + nz * points[j].z + d);
            if (dist < dist_thresh) inliers++;
        }
        if (inliers > best_inliers) {
            best_inliers = inliers;
            best_plane = {nx, ny, nz, d};
        }
    }
    return best_plane;
}

PreprocessedData run_preprocess(const SensorFrame &input,
                                const PipelineConfig &cfg,
                                std::mt19937 &rng) {
    (void)cfg;
    PreprocessedData result;
    result.frame_id = input.frame_id;
    result.timestamp_ns = now_ns();

    // --- 图像预处理 ---
    int out_size = PreprocessedData::IMG_W * PreprocessedData::IMG_H * PreprocessedData::IMG_C;
    result.image_tensor.resize(out_size, 0.0f);
    resize_image_block_avg(
        input.camera_image.data.data(), CameraImage::WIDTH, CameraImage::HEIGHT,
        CameraImage::CHANNELS, result.image_tensor.data(),
        PreprocessedData::IMG_W, PreprocessedData::IMG_H);

    // --- 激光雷达预处理 ---
    // 1. 距离滤波
    std::vector<Point3D> filtered;
    filtered.reserve(input.lidar_points.size());
    for (const auto &pt : input.lidar_points) {
        float dist = std::sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
        if (dist >= 0.5f && dist <= 120.0f) {
            filtered.push_back(pt);
        }
    }

    // 2. 通过 RANSAC 进行地面移除
    result.ground_plane = fit_ground_plane(filtered, rng);
    float a = result.ground_plane[0], b = result.ground_plane[1];
    float c = result.ground_plane[2], dval = result.ground_plane[3];

    std::vector<Point3D> non_ground;
    non_ground.reserve(filtered.size());
    for (const auto &pt : filtered) {
        float dist = std::abs(a * pt.x + b * pt.y + c * pt.z + dval);
        if (dist >= 0.2f) {
            non_ground.push_back(pt);
        }
    }

    // 3. 体素下采样（0.1m 网格）
    const float voxel_size = 0.1f;
    std::map<PreprocessedData::VoxelKey, std::vector<Point3D>> voxel_map;
    for (const auto &pt : non_ground) {
        PreprocessedData::VoxelKey key;
        key.ix = static_cast<int32_t>(std::floor(pt.x / voxel_size));
        key.iy = static_cast<int32_t>(std::floor(pt.y / voxel_size));
        key.iz = static_cast<int32_t>(std::floor(pt.z / voxel_size));
        voxel_map[key].push_back(pt);
    }

    // 计算每个体素的质心
    result.lidar_voxels.clear();
    for (auto &[key, pts] : voxel_map) {
        float sum_x = 0, sum_y = 0, sum_z = 0, sum_i = 0;
        for (const auto &p : pts) {
            sum_x += p.x;
            sum_y += p.y;
            sum_z += p.z;
            sum_i += p.intensity;
        }
        float n = static_cast<float>(pts.size());
        Point3D centroid;
        centroid.x = sum_x / n;
        centroid.y = sum_y / n;
        centroid.z = sum_z / n;
        centroid.intensity = sum_i / n;
        // 为每个体素存储单个质心
        result.lidar_voxels[key] = {centroid};
    }

    // IMU 偏置校正：减去平均偏置（简化版）
    // 在实际系统中，偏置会通过标定获得。此处仅做中心化处理。
    // 对于合成数据而言是无操作。

    return result;
}

// ============================================================================
// 阶段 3：目标检测
// ============================================================================

// 对图像块应用 3x3 卷积核
static float conv3x3(const float *patch, int pw, int ph, int pc,
                     const float *kernel, int kx, int ky) {
    (void)ph;
    float sum = 0;
    for (int c = 0; c < pc; c++) {
        for (int dy = 0; dy < 3; dy++) {
            for (int dx = 0; dx < 3; dx++) {
                int pi = (ky + dy) * pw + (kx + dx);
                sum += patch[pi * pc + c] * kernel[dy * 3 + dx];
            }
        }
    }
    return sum;
}

// 对体素质心进行欧氏距离聚类
static std::vector<std::vector<int>> euclidean_cluster(
    const std::vector<Point3D> &points, float dist_thresh, int min_pts) {
    int n = static_cast<int>(points.size());
    if (n < min_pts) return {};

    // 构建空间网格
    float cell_size = dist_thresh;
    float min_x = points[0].x, max_x = points[0].x;
    float min_y = points[0].y, max_y = points[0].y;
    float min_z = points[0].z, max_z = points[0].z;
    for (const auto &p : points) {
        if (p.x < min_x) min_x = p.x;
        if (p.x > max_x) max_x = p.x;
        if (p.y < min_y) min_y = p.y;
        if (p.y > max_y) max_y = p.y;
        if (p.z < min_z) min_z = p.z;
        if (p.z > max_z) max_z = p.z;
    }
    min_x -= cell_size;
    min_y -= cell_size;
    min_z -= cell_size;

    // 将点映射到网格单元
    struct CellKey {
        int ix, iy, iz;
        bool operator==(const CellKey &o) const {
            return ix == o.ix && iy == o.iy && iz == o.iz;
        }
    };
    struct CellKeyHash {
        size_t operator()(const CellKey &k) const {
            return (static_cast<size_t>(k.ix) * 73856093) ^ (static_cast<size_t>(k.iy) * 19349663) ^ (static_cast<size_t>(k.iz) * 83492791);
        }
    };
    std::unordered_map<CellKey, std::vector<int>, CellKeyHash> grid;
    std::vector<CellKey> point_cell(n);

    for (int i = 0; i < n; i++) {
        CellKey key;
        key.ix = static_cast<int>(std::floor((points[i].x - min_x) / cell_size));
        key.iy = static_cast<int>(std::floor((points[i].y - min_y) / cell_size));
        key.iz = static_cast<int>(std::floor((points[i].z - min_z) / cell_size));
        grid[key].push_back(i);
        point_cell[i] = key;
    }

    // BFS 聚类
    std::vector<bool> visited(n, false);
    std::vector<std::vector<int>> clusters;

    float dist_thresh_sq = dist_thresh * dist_thresh;

    for (int i = 0; i < n; i++) {
        if (visited[i]) continue;
        visited[i] = true;

        std::vector<int> cluster;
        std::vector<int> queue;
        cluster.push_back(i);
        queue.push_back(i);

        while (!queue.empty()) {
            int cur = queue.back();
            queue.pop_back();
            const auto &ck = point_cell[cur];

            // 检查 27 个相邻单元
            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dz = -1; dz <= 1; dz++) {
                        CellKey nk{ck.ix + dx, ck.iy + dy, ck.iz + dz};
                        auto it = grid.find(nk);
                        if (it == grid.end()) continue;
                        for (int nb : it->second) {
                            if (visited[nb]) continue;
                            float dx_v = points[cur].x - points[nb].x;
                            float dy_v = points[cur].y - points[nb].y;
                            float dz_v = points[cur].z - points[nb].z;
                            float dsq = dx_v * dx_v + dy_v * dy_v + dz_v * dz_v;
                            if (dsq < dist_thresh_sq) {
                                visited[nb] = true;
                                cluster.push_back(nb);
                                queue.push_back(nb);
                            }
                        }
                    }
                }
            }
        }

        if (static_cast<int>(cluster.size()) >= min_pts) {
            clusters.push_back(std::move(cluster));
        }
    }

    return clusters;
}

DetectionResult run_detection(const PreprocessedData &input,
                              const PipelineConfig &cfg,
                              std::mt19937 &rng) {
    DetectionResult result;
    result.frame_id = input.frame_id;
    result.timestamp_ns = now_ns();
    (void)cfg;

    // --- 对图像进行 CNN 滑动窗口检测 ---
    const int img_w = PreprocessedData::IMG_W; // 640
    const int img_h = PreprocessedData::IMG_H; // 480
    const int img_c = PreprocessedData::IMG_C; // 3

    // 生成随机 3x3 卷积核
    float kernel[3][3];
    std::normal_distribution<float> kernel_dist(0.0f, 0.5f);
    for (int dy = 0; dy < 3; dy++)
        for (int dx = 0; dx < 3; dx++)
            kernel[dy][dx] = kernel_dist(rng);

    const int window = 224;
    const int stride = 112;
    std::uniform_int_distribution<int> class_dist(0, 3);

    // 在图像上滑动窗口，对随机 3x3 子块应用卷积
    std::vector<DetectionResult::Box3D> img_boxes;
    for (int y = 0; y <= img_h - window; y += stride) {
        for (int x = 0; x <= img_w - window; x += stride) {
            // 在窗口内随机选取 3x3 位置
            int cy = std::uniform_int_distribution<int>(0, window - 3)(rng);
            int cx = std::uniform_int_distribution<int>(0, window - 3)(rng);

            // 应用卷积
            float response = std::abs(
                conv3x3(&input.image_tensor[((y + cy) * img_w + (x + cx)) * img_c],
                        img_w, img_h, img_c, &kernel[0][0], x + cx, y + cy));

            if (response > 1.0f) {
                DetectionResult::Box3D box;
                box.x = static_cast<float>(x) + window / 2.0f;
                box.y = static_cast<float>(y) + window / 2.0f;
                box.z = 0.0f;
                box.width = static_cast<float>(window);
                box.height = static_cast<float>(window);
                box.depth = 2.0f;
                box.confidence = std::min(response / 5.0f, 0.99f);
                box.class_id = class_dist(rng);
                box.vx = 0;
                box.vy = 0;
                img_boxes.push_back(box);
            }
        }
    }

    // --- 激光雷达欧氏距离聚类 ---
    // 将所有体素质心收集到一个平坦数组中
    std::vector<Point3D> centroids;
    centroids.reserve(input.lidar_voxels.size());
    for (const auto &[key, pts] : input.lidar_voxels) {
        if (!pts.empty()) centroids.push_back(pts[0]);
    }

    const float cluster_dist = 1.5f;
    const int min_cluster_pts = 5;
    auto clusters = euclidean_cluster(centroids, cluster_dist, min_cluster_pts);

    std::vector<DetectionResult::Box3D> lidar_boxes;
    for (const auto &cluster : clusters) {
        // 计算包围盒
        float cx = centroids[cluster[0]].x, cy = centroids[cluster[0]].y, cz = centroids[cluster[0]].z;
        float minx = cx, miny = cy, minz = cz;
        float maxx = cx, maxy = cy, maxz = cz;
        float sumx = 0, sumy = 0, sumz = 0;
        for (int idx : cluster) {
            const auto &p = centroids[idx];
            sumx += p.x;
            sumy += p.y;
            sumz += p.z;
            if (p.x < minx) minx = p.x;
            if (p.y < miny) miny = p.y;
            if (p.z < minz) minz = p.z;
            if (p.x > maxx) maxx = p.x;
            if (p.y > maxy) maxy = p.y;
            if (p.z > maxz) maxz = p.z;
        }
        float nf = static_cast<float>(cluster.size());
        DetectionResult::Box3D box;
        box.x = sumx / nf;
        box.y = sumy / nf;
        box.z = sumz / nf;
        box.width = maxx - minx;
        box.height = maxy - miny;
        box.depth = maxz - minz;
        box.confidence = std::min(static_cast<float>(cluster.size()) / 200.0f, 0.95f);
        box.class_id = class_dist(rng);
        box.vx = 0;
        box.vy = 0;
        lidar_boxes.push_back(box);
    }

    // 合并图像和激光雷达检测结果
    result.boxes.reserve(img_boxes.size() + lidar_boxes.size());
    for (auto &b : img_boxes) result.boxes.push_back(std::move(b));
    for (auto &b : lidar_boxes) result.boxes.push_back(std::move(b));
    result.num_detections = static_cast<int32_t>(result.boxes.size());

    return result;
}

// ============================================================================
// 阶段 4：多目标跟踪（卡尔曼滤波）
// ============================================================================

// 九状态卡尔曼滤波器：[x,y,z, vx,vy,vz, ax,ay,az]
// 测量值：[x, y, z]
static constexpr float KF_DT = 0.05f; // 50ms 帧间隔

// 状态转移矩阵 F（9x9）
static void kf_predict(std::array<float, 9> &state,
                       std::array<float, 81> &cov,
                       float process_noise = 0.1f) {
    // x += vx*dt + 0.5*ax*dt^2，以此类推
    state[0] += state[3] * KF_DT + 0.5f * state[6] * KF_DT * KF_DT;
    state[1] += state[4] * KF_DT + 0.5f * state[7] * KF_DT * KF_DT;
    state[2] += state[5] * KF_DT + 0.5f * state[8] * KF_DT * KF_DT;
    state[3] += state[6] * KF_DT;
    state[4] += state[7] * KF_DT;
    state[5] += state[8] * KF_DT;
    // 加速度：随机游走（无变化，噪声添加到协方差中）

    // 协方差更新：P' = F*P*F^T + Q（简化版）
    // 显式构建 F 矩阵
    float dt = KF_DT;
    float dt2_2 = 0.5f * dt * dt;

    // F * P（9x9 * 9x9）
    float F[9][9] = {};
    for (int i = 0; i < 9; i++) F[i][i] = 1.0f;
    F[0][3] = dt;
    F[0][6] = dt2_2;
    F[1][4] = dt;
    F[1][7] = dt2_2;
    F[2][5] = dt;
    F[2][8] = dt2_2;
    F[3][6] = dt;
    F[4][7] = dt;
    F[5][8] = dt;

    // FP = F * P（行主序）
    float FP[9][9] = {};
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            for (int k = 0; k < 9; k++) {
                FP[i][j] += F[i][k] * cov[k * 9 + j];
            }
        }
    }

    // P' = FP * F^T + Q
    float Q_diag = process_noise * process_noise;
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            float sum = 0;
            for (int k = 0; k < 9; k++) {
                sum += FP[i][k] * F[j][k];
            }
            cov[i * 9 + j] = sum;
            if (i == j) cov[i * 9 + j] += Q_diag;
        }
    }
}

// 使用测量值 [mx, my, mz] 进行卡尔曼更新
static void kf_update(std::array<float, 9> &state,
                      std::array<float, 81> &cov,
                      float mx, float my, float mz,
                      float meas_noise = 0.5f) {
    // H = [I_3x3 | 0_3x6]
    // y = z - H*x（新息）
    float y_x = mx - state[0];
    float y_y = my - state[1];
    float y_z = mz - state[2];

    // S = H*P*H^T + R（3x3 新息协方差）
    float R_diag = meas_noise * meas_noise;
    float S[3][3] = {
        {cov[0 * 9 + 0] + R_diag, cov[0 * 9 + 1], cov[0 * 9 + 2]},
        {cov[1 * 9 + 0], cov[1 * 9 + 1] + R_diag, cov[1 * 9 + 2]},
        {cov[2 * 9 + 0], cov[2 * 9 + 1], cov[2 * 9 + 2] + R_diag},
    };

    // 对 3x3 S 矩阵求逆
    float det = S[0][0] * (S[1][1] * S[2][2] - S[1][2] * S[2][1]) - S[0][1] * (S[1][0] * S[2][2] - S[1][2] * S[2][0]) + S[0][2] * (S[1][0] * S[2][1] - S[1][1] * S[2][0]);
    if (std::abs(det) < 1e-9f) return;

    float inv_det = 1.0f / det;
    float Sinv[3][3] = {
        {(S[1][1] * S[2][2] - S[1][2] * S[2][1]) * inv_det,
         (S[0][2] * S[2][1] - S[0][1] * S[2][2]) * inv_det,
         (S[0][1] * S[1][2] - S[0][2] * S[1][1]) * inv_det},
        {(S[1][2] * S[2][0] - S[1][0] * S[2][2]) * inv_det,
         (S[0][0] * S[2][2] - S[0][2] * S[2][0]) * inv_det,
         (S[0][2] * S[1][0] - S[0][0] * S[1][2]) * inv_det},
        {(S[1][0] * S[2][1] - S[1][1] * S[2][0]) * inv_det,
         (S[0][1] * S[2][0] - S[0][0] * S[2][1]) * inv_det,
         (S[0][0] * S[1][1] - S[0][1] * S[1][0]) * inv_det},
    };

    // K = P*H^T * S^-1（9x3 = 9x9 * 9x3 * 3x3）
    // P*H^T 是 P 的前 3 列
    float K[9][3] = {};
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                K[i][j] += cov[i * 9 + k] * Sinv[k][j];
            }
        }
    }

    // 状态更新：x = x + K*y
    state[0] += K[0][0] * y_x + K[0][1] * y_y + K[0][2] * y_z;
    state[1] += K[1][0] * y_x + K[1][1] * y_y + K[1][2] * y_z;
    state[2] += K[2][0] * y_x + K[2][1] * y_y + K[2][2] * y_z;
    state[3] += K[3][0] * y_x + K[3][1] * y_y + K[3][2] * y_z;
    state[4] += K[4][0] * y_x + K[4][1] * y_y + K[4][2] * y_z;
    state[5] += K[5][0] * y_x + K[5][1] * y_y + K[5][2] * y_z;
    state[6] += K[6][0] * y_x + K[6][1] * y_y + K[6][2] * y_z;
    state[7] += K[7][0] * y_x + K[7][1] * y_y + K[7][2] * y_z;
    state[8] += K[8][0] * y_x + K[8][1] * y_y + K[8][2] * y_z;

    // 协方差更新：P = (I - K*H)*P
    // I - K*H 是 9x9 矩阵，前 3 列为单位阵减去 K
    // 计算 KH = K * H：由于 H 选取前 3 列，KH(i,j) = K(i, j)（j<3），否则为 0
    float IKH[9][9] = {};
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            IKH[i][j] = (i == j) ? 1.0f : 0.0f;
            if (j < 3) IKH[i][j] -= K[i][j];
        }
    }

    // P' = IKH * P
    float new_cov[81] = {};
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            float sum = 0;
            for (int k = 0; k < 9; k++) {
                sum += IKH[i][k] * cov[k * 9 + j];
            }
            new_cov[i * 9 + j] = sum;
        }
    }
    for (int i = 0; i < 81; i++) cov[i] = new_cov[i];
}

TrackingResult run_tracking(const DetectionResult &input,
                            const PipelineConfig &cfg,
                            std::mt19937 &rng) {
    TrackingResult result;
    result.frame_id = input.frame_id;
    result.timestamp_ns = now_ns();
    (void)cfg;

    // 持久化跟踪存储，带线程安全访问
    static std::vector<TrackingResult::Track> active_tracks;
    static int32_t next_track_id = 0;
    static std::mutex track_mutex;
    std::lock_guard<std::mutex> guard(track_mutex);

    // 对所有现有跟踪进行前向预测
    for (auto &track : active_tracks) {
        kf_predict(track.state, track.covariance);
        track.age++;
    }

    // 简单贪心关联：为每个检测结果找到最近的跟踪
    std::vector<bool> det_used(input.num_detections, false);
    const float max_assoc_dist = 5.0f; // 米

    for (auto &track : active_tracks) {
        int best_det = -1;
        float best_dist = max_assoc_dist;
        for (int d = 0; d < input.num_detections; d++) {
            if (det_used[d]) continue;
            float dx = track.state[0] - input.boxes[d].x;
            float dy = track.state[1] - input.boxes[d].y;
            float dz = track.state[2] - input.boxes[d].z;
            float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (dist < best_dist) {
                best_dist = dist;
                best_det = d;
            }
        }
        if (best_det >= 0) {
            det_used[best_det] = true;
            kf_update(track.state, track.covariance,
                      input.boxes[best_det].x,
                      input.boxes[best_det].y,
                      input.boxes[best_det].z);
        }
    }

    // 为未关联的检测结果创建新跟踪
    for (int d = 0; d < input.num_detections; d++) {
        if (det_used[d]) continue;
        TrackingResult::Track new_track;
        new_track.track_id = next_track_id++;
        new_track.age = 1;
        new_track.state = {};
        new_track.state[0] = input.boxes[d].x;
        new_track.state[1] = input.boxes[d].y;
        new_track.state[2] = input.boxes[d].z;
        new_track.state[3] = input.boxes[d].vx;
        new_track.state[4] = input.boxes[d].vy;
        // 将协方差初始化为 10 * 单位矩阵
        for (int i = 0; i < 9; i++) {
            new_track.covariance[i * 9 + i] = 10.0f;
        }
        active_tracks.push_back(new_track);
    }

    // 移除旧跟踪（超过 100 帧未更新的跟踪... 简化处理）
    // 为简单起见，我们保留所有跟踪

    result.tracks = active_tracks;
    result.num_tracks = static_cast<int32_t>(result.tracks.size());
    (void)rng;
    return result;
}

// ============================================================================
// 阶段 5：轨迹预测
// ============================================================================

PredictionResult run_prediction(const TrackingResult &input,
                                const PipelineConfig &cfg,
                                std::mt19937 &rng) {
    PredictionResult result;
    result.frame_id = input.frame_id;
    result.timestamp_ns = now_ns();
    (void)cfg;
    (void)rng;

    // 匀速模型外推，5 秒预测时域
    const float horizon = 5.0f;                       // 秒
    const float dt = 0.05f;                           // 20 Hz
    const int steps = static_cast<int>(horizon / dt); // 100 步

    result.trajectories.reserve(input.num_tracks);
    for (const auto &track : input.tracks) {
        PredictionResult::PredictedTrajectory traj;
        traj.track_id = track.track_id;
        traj.waypoints.reserve(steps);

        float x = track.state[0];
        float y = track.state[1];
        float vx = track.state[3];
        float vy = track.state[4];
        float ax = track.state[6];
        float ay = track.state[7];

        for (int i = 0; i < steps; i++) {
            float t = (i + 1) * dt;
            // 匀加速模型
            float px = x + vx * t + 0.5f * ax * t * t;
            float py = y + vy * t + 0.5f * ay * t * t;
            traj.waypoints.push_back({px, py, t});
        }
        result.trajectories.push_back(std::move(traj));
    }
    return result;
}

// ============================================================================
// 阶段 6：路径规划（100x100 网格上的 A* + 三次样条平滑）
// ============================================================================

struct GridCell {
    int x, y;
    float g, h;
    GridCell *parent = nullptr;

    float f() const {
        return g + h;
    }
};

struct CompareCell {
    bool operator()(const GridCell *a, const GridCell *b) const {
        return a->f() > b->f();
    }
};

static float heuristic(int x1, int y1, int x2, int y2) {
    return std::sqrt(static_cast<float>((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)));
}

// 对路径进行三次样条平滑
static void smooth_path_cubic(const std::vector<std::pair<float, float>> &raw,
                              std::vector<PlanningResult::EgoWaypoint> &smoothed,
                              float total_time) {
    int n = static_cast<int>(raw.size());
    if (n < 2) return;

    smoothed.reserve(n);
    for (int i = 0; i < n; i++) {
        PlanningResult::EgoWaypoint wp;
        wp.x = raw[i].first;
        wp.y = raw[i].second;
        wp.theta = 0.0f;
        wp.v = 10.0f; // 恒定速度
        wp.t = total_time * static_cast<float>(i) / static_cast<float>(n - 1);

        // 从相邻点计算航向角
        if (i < n - 1) {
            wp.theta = std::atan2(raw[i + 1].second - raw[i].second,
                                  raw[i + 1].first - raw[i].first);
        } else if (i > 0) {
            wp.theta = std::atan2(raw[i].second - raw[i - 1].second,
                                  raw[i].first - raw[i - 1].first);
        }
        smoothed.push_back(wp);
    }

    // 对航向角和速度应用移动平均平滑
    for (int i = 1; i < n - 1; i++) {
        smoothed[i].theta = std::atan2(
            smoothed[i + 1].y - smoothed[i - 1].y,
            smoothed[i + 1].x - smoothed[i - 1].x);
    }
}

PlanningResult run_planning(const PredictionResult &input,
                            const PipelineConfig &cfg,
                            std::mt19937 &rng) {
    (void)cfg;
    PlanningResult result;
    result.frame_id = input.frame_id;
    result.timestamp_ns = now_ns();
    (void)input;

    const int grid_w = 100;
    const int grid_h = 100;

    // 生成随机障碍物地图
    std::vector<bool> obstacles(grid_w * grid_h, false);
    std::bernoulli_distribution obs_dist(0.15); // 15% 障碍物密度
    for (int i = 0; i < grid_w * grid_h; i++) {
        obstacles[i] = obs_dist(rng);
    }
    // 清除起点和终点
    obstacles[0] = false;
    obstacles[grid_w * grid_h - 1] = false;

    // 存储代价地图
    result.cost_map.resize(grid_w * grid_h, 0.0f);
    for (int i = 0; i < grid_w * grid_h; i++) {
        result.cost_map[i] = obstacles[i] ? 1.0f : 0.0f;
    }

    // A* 搜索：从 (0,0) 到 (99,99)
    int start_x = 0, start_y = 0;
    int goal_x = grid_w - 1, goal_y = grid_h - 1;

    std::vector<std::vector<GridCell>> cells(
        grid_h, std::vector<GridCell>(grid_w));
    for (int y = 0; y < grid_h; y++)
        for (int x = 0; x < grid_w; x++)
            cells[y][x] = {x, y, 0, 0, nullptr};

    std::priority_queue<GridCell *, std::vector<GridCell *>, CompareCell> open;
    std::vector<std::vector<bool>> closed(grid_h,
                                          std::vector<bool>(grid_w, false));

    cells[start_y][start_x].g = 0;
    cells[start_y][start_x].h = heuristic(start_x, start_y, goal_x, goal_y);
    open.push(&cells[start_y][start_x]);

    bool found = false;
    const int dx8[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const int dy8[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

    while (!open.empty()) {
        GridCell *cur = open.top();
        open.pop();

        if (cur->x == goal_x && cur->y == goal_y) {
            found = true;
            break;
        }

        if (closed[cur->y][cur->x]) continue;
        closed[cur->y][cur->x] = true;

        for (int d = 0; d < 8; d++) {
            int nx = cur->x + dx8[d];
            int ny = cur->y + dy8[d];
            if (nx < 0 || nx >= grid_w || ny < 0 || ny >= grid_h) continue;
            if (obstacles[ny * grid_w + nx]) continue;
            if (closed[ny][nx]) continue;

            float step_cost = (dx8[d] != 0 && dy8[d] != 0) ? 1.414f : 1.0f;
            float new_g = cur->g + step_cost;

            if (new_g < cells[ny][nx].g || cells[ny][nx].parent == nullptr) {
                cells[ny][nx].g = new_g;
                cells[ny][nx].h = heuristic(nx, ny, goal_x, goal_y);
                cells[ny][nx].parent = cur;
                open.push(&cells[ny][nx]);
            }
        }
    }

    // 提取路径
    std::vector<std::pair<float, float>> raw_path;
    if (found) {
        GridCell *cur = &cells[goal_y][goal_x];
        while (cur) {
            raw_path.emplace_back(static_cast<float>(cur->x) * 0.5f - 25.0f,
                                  static_cast<float>(cur->y) * 0.5f - 25.0f);
            cur = cur->parent;
        }
        std::reverse(raw_path.begin(), raw_path.end());
    } else {
        // 回退方案：直线路径
        for (int i = 0; i <= 50; i++) {
            float t = static_cast<float>(i) / 50.0f;
            raw_path.emplace_back(t * 50.0f - 25.0f, 0.0f);
        }
    }

    // 使用三次样条近似进行平滑
    smooth_path_cubic(raw_path, result.ego_trajectory, 5.0f);

    return result;
}

// ============================================================================
// 阶段 7：车辆控制（Stanley + PID）
// ============================================================================

ControlCommand run_control(const PlanningResult &input,
                           const PipelineConfig &cfg) {
    ControlCommand cmd;
    cmd.frame_id = input.frame_id;
    cmd.timestamp_ns = now_ns();
    cmd.gear = 1;
    (void)cfg;

    if (input.ego_trajectory.empty()) {
        cmd.throttle = 0;
        cmd.brake = 1.0f;
        cmd.steering = 0;
        cmd.target_velocity = 0;
        return cmd;
    }

    // 前视点：第一个航点
    const auto &target = input.ego_trajectory.front();
    float target_v = target.v;

    // 简单的自车状态（假设在原点，朝正前方）
    float ego_x = 0, ego_y = 0, ego_theta = 0;

    // --- Stanley 控制器（横向） ---
    // 横向偏差
    float dx = target.x - ego_x;
    float dy = target.y - ego_y;
    float cross_track = -dx * std::sin(ego_theta) + dy * std::cos(ego_theta);

    // 航向误差
    float heading_error = target.theta - ego_theta;
    // 归一化到 [-pi, pi]
    while (heading_error > M_PI) heading_error -= 2 * M_PI;
    while (heading_error < -M_PI) heading_error += 2 * M_PI;

    // Stanley 控制律
    float k_stanley = 0.5f;
    float k_soft = 1.0f;
    float current_v = 10.0f; // 假设的当前速度
    float vel_clamped = std::max(current_v, 1.0f);
    float steering = heading_error + std::atan2(k_stanley * cross_track, vel_clamped + k_soft);

    // --- PID 控制器（纵向） ---
    static float prev_error = 0;
    static float integral = 0;
    float kp = 0.5f, ki = 0.05f, kd = 0.1f;

    float vel_error = target_v - current_v;
    integral += vel_error * KF_DT;
    float derivative = (vel_error - prev_error) / KF_DT;
    prev_error = vel_error;

    float control_output = kp * vel_error + ki * integral + kd * derivative;

    // 将控制输出映射到油门/刹车
    if (control_output > 0) {
        cmd.throttle = std::min(control_output, 1.0f);
        cmd.brake = 0;
    } else {
        cmd.throttle = 0;
        cmd.brake = std::min(-control_output, 1.0f);
    }

    cmd.steering = std::max(-1.0f, std::min(1.0f, steering / static_cast<float>(M_PI_4)));
    cmd.target_velocity = target_v;

    return cmd;
}
