#include "task_executor.h"

#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_map>
#include <functional>
#include <unordered_set>

// ============================================================================
// TaskGraphExecutor: 带线程池的 DAG 调度器
// ============================================================================

TaskGraphExecutor::TaskGraphExecutor(int num_threads) : num_threads_(num_threads) {
}

void TaskGraphExecutor::add_node(TaskNode node) {
    nodes_.push_back(std::move(node));
}

void TaskGraphExecutor::build_graph() {
    int n = static_cast<int>(nodes_.size());
    for (int i = 0; i < n; ++i) {
        for (int dep : nodes_[i].dependencies) {
            nodes_[dep].dependents.push_back(i);
        }
        nodes_[i].indegree = static_cast<int>(nodes_[i].dependencies.size());
    }
}

void TaskGraphExecutor::execute() {
    std::mutex mtx;
    std::condition_variable cv;
    std::queue<int> ready_queue;
    int pending = static_cast<int>(nodes_.size());

    for (int i = 0; i < static_cast<int>(nodes_.size()); ++i) {
        if (nodes_[i].indegree == 0) {
            ready_queue.push(i);
        }
    }

    auto worker = [&]() {
        while (true) {
            int node_idx = -1;
            {
                std::unique_lock<std::mutex> lk(mtx);
                cv.wait(lk, [&]() { return !ready_queue.empty() || pending == 0; });
                if (pending == 0) return;
                node_idx = ready_queue.front();
                ready_queue.pop();
            }

            Timer t;
            t.start();
            nodes_[node_idx].work();
            nodes_[node_idx].elapsed_us = t.elapsed_us();

            {
                std::lock_guard<std::mutex> lk(mtx);
                for (int dep : nodes_[node_idx].dependents) {
                    nodes_[dep].indegree--;
                    if (nodes_[dep].indegree == 0) {
                        ready_queue.push(dep);
                    }
                }
                nodes_[node_idx].completed = true;
                pending--;
            }
            cv.notify_all();
        }
    };

    std::vector<std::thread> threads;
    int actual_threads = std::min(num_threads_, static_cast<int>(nodes_.size()));
    for (int t = 0; t < actual_threads; ++t) {
        threads.emplace_back(worker);
    }
    for (auto &th : threads) th.join();
}

void TaskGraphExecutor::write_profile_json(const std::string &filepath,
                                           int num_threads,
                                           double wall_time_us) const {
    FILE *f = std::fopen(filepath.c_str(), "w");
    if (!f) return;

    std::fprintf(f, "{\n");
    std::fprintf(f, "  \"workload\": \"robot_processing_pipeline\",\n");
    std::fprintf(f, "  \"num_nodes\": %zu,\n", nodes_.size());
    std::fprintf(f, "  \"num_threads\": %d,\n", num_threads);
    std::fprintf(f, "  \"nodes\": [\n");

    for (size_t i = 0; i < nodes_.size(); ++i) {
        const auto &node = nodes_[i];
        std::fprintf(f, "    {\n");
        std::fprintf(f, "      \"name\": \"%s\",\n", node.name.c_str());
        std::fprintf(f, "      \"dependencies\": [");
        for (size_t j = 0; j < node.dependencies.size(); ++j) {
            if (j > 0) std::fprintf(f, ", ");
            std::fprintf(f, "\"%s\"", nodes_[node.dependencies[j]].name.c_str());
        }
        std::fprintf(f, "],\n");
        std::fprintf(f, "      \"elapsed_us\": %.2f,\n", node.elapsed_us);
        std::fprintf(f, "      \"elapsed_ms\": %.3f\n", node.elapsed_us / 1000.0);
        std::fprintf(f, "    }%s\n", (i < nodes_.size() - 1) ? "," : "");
    }

    std::fprintf(f, "  ],\n");

    // 通过拓扑排序进行关键路径分析
    std::vector<double> earliest_finish(nodes_.size(), 0.0);
    std::vector<int> topo_order;
    {
        std::vector<int> indeg(nodes_.size());
        for (size_t i = 0; i < nodes_.size(); ++i)
            indeg[i] = nodes_[i].indegree;
        std::queue<int> q;
        for (size_t i = 0; i < nodes_.size(); ++i)
            if (indeg[i] == 0) q.push(static_cast<int>(i));
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            topo_order.push_back(u);
            for (int v : nodes_[u].dependents) {
                indeg[v]--;
                if (indeg[v] == 0) q.push(v);
            }
        }
    }

    std::vector<int> prev(nodes_.size(), -1);
    for (int u : topo_order) {
        for (int v : nodes_[u].dependents) {
            double candidate = earliest_finish[u] + nodes_[u].elapsed_us;
            if (candidate > earliest_finish[v]) {
                earliest_finish[v] = candidate;
                prev[v] = u;
            }
        }
    }

    // 找到关键路径终点节点
    int crit_end = 0;
    double crit_time = 0.0;
    for (size_t i = 0; i < nodes_.size(); ++i) {
        double finish = earliest_finish[i] + nodes_[i].elapsed_us;
        if (finish > crit_time) {
            crit_time = finish;
            crit_end = static_cast<int>(i);
        }
    }

    std::vector<std::string> crit_path;
    for (int u = crit_end; u >= 0; u = prev[u]) {
        crit_path.push_back(nodes_[u].name);
    }
    std::reverse(crit_path.begin(), crit_path.end());

    std::fprintf(f, "  \"critical_path\": [");
    for (size_t i = 0; i < crit_path.size(); ++i) {
        if (i > 0) std::fprintf(f, ", ");
        std::fprintf(f, "\"%s\"", crit_path[i].c_str());
    }
    std::fprintf(f, "],\n");
    std::fprintf(f, "  \"critical_path_us\": %.2f,\n", crit_time);

    double total_seq_us = 0.0;
    for (const auto &node : nodes_) total_seq_us += node.elapsed_us;
    std::fprintf(f, "  \"total_sequential_us\": %.2f,\n", total_seq_us);
    std::fprintf(f, "  \"parallel_wall_us\": %.2f,\n", wall_time_us);

    double speedup = (wall_time_us > 0.0) ? (total_seq_us / wall_time_us) : 0.0;
    double efficiency = (num_threads > 0) ? (speedup / num_threads * 100.0) : 0.0;
    std::fprintf(f, "  \"speedup\": %.2f,\n", speedup);
    std::fprintf(f, "  \"parallelism_efficiency_pct\": %.2f\n", efficiency);

    std::fprintf(f, "}\n");
    std::fclose(f);
}

const std::vector<TaskNode> &TaskGraphExecutor::nodes() const {
    return nodes_;
}

PipelineContext &TaskGraphExecutor::context() {
    return ctx_;
}

const PipelineContext &TaskGraphExecutor::context() const {
    return ctx_;
}

// ============================================================================
// 随机数生成器辅助函数
// ============================================================================

static std::mt19937 &rng() {
    static std::mt19937 instance(std::random_device{}());
    return instance;
}

// ============================================================================
// 节点 1：传感器采集
//    生成合成相机图像（1920x1080x3）和 LiDAR 点云
//    （10 万点，64 线），包含真实模式。
// ============================================================================

void node_sensor_capture(PipelineContext &ctx) {
    // 生成相机图像（1920x1080x3 随机 RGB 数据）
    ctx.camera_image.width = 1920;
    ctx.camera_image.height = 1080;
    ctx.camera_image.channels = 3;
    int total_pixels = 1920 * 1080 * 3;
    ctx.camera_image.data.resize(total_pixels);
    std::uniform_int_distribution<int> pixel_dist(0, 255);
    for (int i = 0; i < total_pixels; ++i) {
        ctx.camera_image.data[i] = static_cast<uint8_t>(pixel_dist(rng()));
    }

    // 生成 LiDAR 点云（10 万点，64 线）
    const float fov_up = 15.0f * static_cast<float>(M_PI) / 180.0f;
    const float fov_down = -25.0f * static_cast<float>(M_PI) / 180.0f;
    const int num_rings = 64;
    const int pts_per_ring = 100000 / num_rings;
    ctx.point_cloud.points.reserve(100000);

    std::uniform_real_distribution<float> range_dist(0.5f, 120.0f);
    std::uniform_real_distribution<float> intensity_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> noise_dist(-0.02f, 0.02f);

    for (int ring = 0; ring < num_rings; ++ring) {
        float v_angle = fov_down + (fov_up - fov_down) * static_cast<float>(ring) / static_cast<float>(num_rings - 1);
        for (int i = 0; i < pts_per_ring; ++i) {
            float h_angle = 2.0f * static_cast<float>(M_PI) * static_cast<float>(i) / static_cast<float>(pts_per_ring);
            float range = range_dist(rng());
            TaskPoint3D p;
            p.x = range * std::cos(v_angle) * std::cos(h_angle) + noise_dist(rng());
            p.y = range * std::cos(v_angle) * std::sin(h_angle) + noise_dist(rng());
            p.z = range * std::sin(v_angle) + noise_dist(rng());
            p.intensity = intensity_dist(rng());
            p.ring = ring;
            ctx.point_cloud.points.push_back(p);
        }
    }
}

// ============================================================================
// 节点 2：图像预处理
//    RGB -> 灰度（ITU-R BT.601），双线性缩放 1920x1080 -> 640x480，
//    归一化到 [0, 1]。
// ============================================================================

void node_image_preprocess(PipelineContext &ctx) {
    int src_w = ctx.camera_image.width;
    int src_h = ctx.camera_image.height;

    // RGB -> 灰度
    int total_pixels = src_w * src_h;
    std::vector<float> gray(total_pixels);
    for (int i = 0; i < total_pixels; ++i) {
        float r = static_cast<float>(ctx.camera_image.data[i * 3 + 0]);
        float g = static_cast<float>(ctx.camera_image.data[i * 3 + 1]);
        float b = static_cast<float>(ctx.camera_image.data[i * 3 + 2]);
        gray[i] = 0.299f * r + 0.587f * g + 0.114f * b;
    }

    // 双线性缩放至 640x480
    int dst_w = 640;
    int dst_h = 480;
    std::vector<float> resized(dst_w * dst_h);
    float scale_x = static_cast<float>(src_w) / static_cast<float>(dst_w);
    float scale_y = static_cast<float>(src_h) / static_cast<float>(dst_h);

    for (int dy = 0; dy < dst_h; ++dy) {
        float sy = static_cast<float>(dy) * scale_y;
        int y0 = static_cast<int>(sy);
        int y1 = std::min(y0 + 1, src_h - 1);
        float y_frac = sy - static_cast<float>(y0);
        int y0_row = y0 * src_w;
        int y1_row = y1 * src_w;

        for (int dx = 0; dx < dst_w; ++dx) {
            float sx = static_cast<float>(dx) * scale_x;
            int x0 = static_cast<int>(sx);
            int x1 = std::min(x0 + 1, src_w - 1);
            float x_frac = sx - static_cast<float>(x0);

            float top = gray[y0_row + x0] * (1.0f - x_frac) + gray[y0_row + x1] * x_frac;
            float bot = gray[y1_row + x0] * (1.0f - x_frac) + gray[y1_row + x1] * x_frac;
            resized[dy * dst_w + dx] = top * (1.0f - y_frac) + bot * y_frac;
        }
    }

    // 归一化到 [0, 1]
    float inv_255 = 1.0f / 255.0f;
    for (float &v : resized) v *= inv_255;

    ctx.preprocessed_image.width = dst_w;
    ctx.preprocessed_image.height = dst_h;
    ctx.preprocessed_image.data = std::move(resized);
}

// ============================================================================
// 节点 3：LiDAR 预处理
//    距离滤波（0.5m - 100m），体素网格降采样（10cm）。
//    与 image_preprocess 并行运行。
// ============================================================================

struct VoxKey {
    int ix, iy, iz;
    bool operator==(const VoxKey &o) const {
        return ix == o.ix && iy == o.iy && iz == o.iz;
    }
};

struct VoxKeyHash {
    size_t operator()(const VoxKey &k) const {
        return (static_cast<size_t>(k.ix) * 73856093) ^ (static_cast<size_t>(k.iy) * 19349663) ^ (static_cast<size_t>(k.iz) * 83492791);
    }
};

void node_lidar_preprocess(PipelineContext &ctx) {
    // 距离滤波
    ctx.filtered_cloud.points.reserve(ctx.point_cloud.points.size());
    for (const auto &p : ctx.point_cloud.points) {
        float dist = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
        if (dist >= 0.5f && dist <= 100.0f) {
            ctx.filtered_cloud.points.push_back(p);
        }
    }

    // 体素网格降采样（10cm）
    float voxel_size = 0.1f;
    float inv_vs = 1.0f / voxel_size;
    std::unordered_map<VoxKey, TaskPoint3D, VoxKeyHash> voxel_map;

    for (const auto &p : ctx.filtered_cloud.points) {
        VoxKey key{
            static_cast<int>(std::floor(p.x * inv_vs)),
            static_cast<int>(std::floor(p.y * inv_vs)),
            static_cast<int>(std::floor(p.z * inv_vs))};
        if (voxel_map.find(key) == voxel_map.end()) {
            voxel_map[key] = p;
        }
    }

    ctx.filtered_cloud.points.clear();
    ctx.filtered_cloud.points.reserve(voxel_map.size());
    for (const auto &kv : voxel_map) {
        ctx.filtered_cloud.points.push_back(kv.second);
    }
}

// ============================================================================
// 节点 4：检测
//    预处理图像上的滑动窗口 3x3 Sobel 边缘检测，
//    点云欧几里得聚类。依赖 image_preprocess 和
//    lidar_preprocess。
// ============================================================================

void node_detection(PipelineContext &ctx) {
    int w = ctx.preprocessed_image.width;
    int h = ctx.preprocessed_image.height;
    const auto &img = ctx.preprocessed_image.data;

    // 3x3 Sobel 边缘检测（滑动窗口卷积）
    std::vector<float> edge_map(w * h, 0.0f);
    // 水平 Sobel 卷积核：[-1,0,1; -2,0,2; -1,0,1]
    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            float gx = -1.0f * img[(y - 1) * w + (x - 1)] + 0.0f * img[(y - 1) * w + x] + 1.0f * img[(y - 1) * w + (x + 1)] + -2.0f * img[y * w + (x - 1)] + 0.0f * img[y * w + x] + 2.0f * img[y * w + (x + 1)] + -1.0f * img[(y + 1) * w + (x - 1)] + 0.0f * img[(y + 1) * w + x] + 1.0f * img[(y + 1) * w + (x + 1)];

            float gy = -1.0f * img[(y - 1) * w + (x - 1)] + -2.0f * img[(y - 1) * w + x] + -1.0f * img[(y - 1) * w + (x + 1)] + 0.0f * img[y * w + (x - 1)] + 0.0f * img[y * w + x] + 0.0f * img[y * w + (x + 1)] + 1.0f * img[(y + 1) * w + (x - 1)] + 2.0f * img[(y + 1) * w + x] + 1.0f * img[(y + 1) * w + (x + 1)];

            edge_map[y * w + x] = std::sqrt(gx * gx + gy * gy);
        }
    }

    // 寻找局部最大值作为候选检测（简单 NMS）
    std::uniform_int_distribution<int> class_dist(0, 3);
    std::uniform_real_distribution<float> conf_dist(0.5f, 0.99f);

    int det_count = 0;
    for (int y = 2; y < h - 2; y += 16) {
        for (int x = 2; x < w - 2; x += 16) {
            float val = edge_map[y * w + x];
            bool is_max = true;
            for (int dy = -1; dy <= 1 && is_max; ++dy)
                for (int dx = -1; dx <= 1 && is_max; ++dx)
                    if (edge_map[(y + dy) * w + (x + dx)] > val)
                        is_max = false;
            if (is_max && val > 0.1f && det_count < 15) {
                TaskDetectionBox box;
                box.class_id = class_dist(rng());
                box.confidence = conf_dist(rng());
                box.x = static_cast<float>(x) / static_cast<float>(w) * 50.0f - 25.0f;
                box.y = static_cast<float>(y) / static_cast<float>(h) * 50.0f - 25.0f;
                box.z = 0.0f;
                box.w = 2.0f + conf_dist(rng());
                box.h = 2.0f + conf_dist(rng());
                box.d = 2.0f + conf_dist(rng());
                ctx.detections.boxes.push_back(box);
                ++det_count;
            }
        }
    }

    // 点云欧几里得聚类（简单区域生长）
    const auto &pts = ctx.filtered_cloud.points;
    int n = static_cast<int>(pts.size());
    if (n < 3) return;

    std::vector<bool> visited(n, false);
    float cluster_thresh = 0.5f; // 50cm 距离阈值

    for (int i = 0; i < n && ctx.detections.boxes.size() < 25; ++i) {
        if (visited[i]) continue;
        std::vector<int> cluster;
        std::queue<int> q;
        q.push(i);
        visited[i] = true;

        while (!q.empty()) {
            int idx = q.front();
            q.pop();
            cluster.push_back(idx);
            // 在空间哈希中搜索邻居（简化：线性扫描子集）
            int search_start = std::max(0, idx - 2000);
            int search_end = std::min(n, idx + 2000);
            for (int j = search_start; j < search_end; ++j) {
                if (visited[j]) continue;
                float dx = pts[idx].x - pts[j].x;
                float dy = pts[idx].y - pts[j].y;
                float dz = pts[idx].z - pts[j].z;
                float dist = dx * dx + dy * dy + dz * dz;
                if (dist < cluster_thresh * cluster_thresh) {
                    visited[j] = true;
                    q.push(j);
                }
            }
        }

        if (cluster.size() >= 10) {
            // 计算包围盒
            float min_x = pts[cluster[0]].x, max_x = min_x;
            float min_y = pts[cluster[0]].y, max_y = min_y;
            float min_z = pts[cluster[0]].z, max_z = min_z;
            for (int ci : cluster) {
                min_x = std::min(min_x, pts[ci].x);
                max_x = std::max(max_x, pts[ci].x);
                min_y = std::min(min_y, pts[ci].y);
                max_y = std::max(max_y, pts[ci].y);
                min_z = std::min(min_z, pts[ci].z);
                max_z = std::max(max_z, pts[ci].z);
            }
            TaskDetectionBox box;
            box.class_id = class_dist(rng());
            box.confidence = conf_dist(rng());
            box.x = (min_x + max_x) * 0.5f;
            box.y = (min_y + max_y) * 0.5f;
            box.z = (min_z + max_z) * 0.5f;
            box.w = max_x - min_x;
            box.h = max_y - min_y;
            box.d = max_z - min_z;
            ctx.detections.boxes.push_back(box);
        }
    }
}

// ============================================================================
// 节点 5：跟踪
//    每个检测目标的卡尔曼滤波器状态预测与更新。
//    依赖 detection。
// ============================================================================

static void kalman_predict_track(TaskKalmanTrack &track, float dt_s) {
    float F[16] = {
        1.0f, 0.0f, dt_s, 0.0f,
        0.0f, 1.0f, 0.0f, dt_s,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f};

    float nx = F[0] * track.x + F[1] * track.y + F[2] * track.vx + F[3] * track.vy;
    float ny = F[4] * track.x + F[5] * track.y + F[6] * track.vx + F[7] * track.vy;
    float nvx = F[8] * track.x + F[9] * track.y + F[10] * track.vx + F[11] * track.vy;
    float nvy = F[12] * track.x + F[13] * track.y + F[14] * track.vx + F[15] * track.vy;
    track.x = nx;
    track.y = ny;
    track.vx = nvx;
    track.vy = nvy;

    // P = F*P*F^T + Q
    float FP[16] = {};
    for (int i = 0; i < 4; ++i)
        for (int k = 0; k < 4; ++k)
            for (int j = 0; j < 4; ++j)
                FP[i * 4 + j] += F[i * 4 + k] * track.P[k * 4 + j];

    float FPFt[16] = {};
    for (int i = 0; i < 4; ++i)
        for (int k = 0; k < 4; ++k)
            for (int j = 0; j < 4; ++j)
                FPFt[i * 4 + j] += FP[i * 4 + k] * F[j * 4 + k];

    float q_pos = 0.01f, q_vel = 0.005f;
    for (int i = 0; i < 4; ++i) track.P[i * 4 + i] += (i < 2 ? q_pos : q_vel);

    for (int i = 0; i < 16; ++i) track.P[i] = FPFt[i] + (i % 5 == 0 ? (i < 10 ? q_pos : q_vel) : 0.0f);
}

static void kalman_update_track(TaskKalmanTrack &track, float mx, float my) {
    float y_innov[2] = {mx - track.x, my - track.y};
    float meas_noise = 0.05f;
    float S00 = track.P[0] + meas_noise;
    float S01 = track.P[1];
    float S10 = track.P[4];
    float S11 = track.P[5] + meas_noise;
    float det = S00 * S11 - S01 * S10;
    if (std::fabs(det) < 1e-10f) return;
    float inv_det = 1.0f / det;
    float Si00 = S11 * inv_det;
    float Si01 = -S01 * inv_det;
    float Si10 = -S10 * inv_det;
    float Si11 = S00 * inv_det;

    float K[8] = {};
    for (int i = 0; i < 4; ++i) {
        K[i * 2 + 0] = track.P[i * 4 + 0] * Si00 + track.P[i * 4 + 1] * Si10;
        K[i * 2 + 1] = track.P[i * 4 + 0] * Si01 + track.P[i * 4 + 1] * Si11;
    }

    track.x += K[0] * y_innov[0] + K[1] * y_innov[1];
    track.y += K[2] * y_innov[0] + K[3] * y_innov[1];
    track.vx += K[4] * y_innov[0] + K[5] * y_innov[1];
    track.vy += K[6] * y_innov[0] + K[7] * y_innov[1];

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

void node_tracking(PipelineContext &ctx) {
    ctx.tracking_result.tracks.reserve(ctx.detections.boxes.size());
    for (size_t i = 0; i < ctx.detections.boxes.size(); ++i) {
        const auto &box = ctx.detections.boxes[i];
        TaskKalmanTrack track;
        track.x = box.x;
        track.y = box.y;
        track.vx = 0.0f;
        track.vy = 0.0f;
        for (int j = 0; j < 16; ++j) track.P[j] = 0.0f;
        track.P[0] = track.P[5] = 0.1f;
        track.P[10] = track.P[15] = 0.05f;

        kalman_predict_track(track, 0.1f);
        kalman_update_track(track, box.x + 0.01f, box.y + 0.01f);

        ctx.tracking_result.tracks.push_back(track);
    }
}

// ============================================================================
// 节点 6：规划
//    生成随机 100x100 占用网格并运行 A* 搜索。
//    依赖 tracking。
// ============================================================================

struct AStarNode {
    int x, y;
    float g, h;
    bool operator>(const AStarNode &o) const {
        return (g + h) > (o.g + o.h);
    }
};

static float heuristic(int x1, int y1, int x2, int y2) {
    float dx = static_cast<float>(x1 - x2);
    float dy = static_cast<float>(y1 - y2);
    return std::sqrt(dx * dx + dy * dy);
}

void node_planning(PipelineContext &ctx) {
    const int grid_sz = 100;
    // 生成占用网格：30% 障碍物
    ctx.occupancy_grid.assign(grid_sz, std::vector<int>(grid_sz, 0));
    std::uniform_real_distribution<float> obs_dist(0.0f, 1.0f);
    for (int y = 0; y < grid_sz; ++y) {
        for (int x = 0; x < grid_sz; ++x) {
            if (obs_dist(rng()) < 0.3f) {
                ctx.occupancy_grid[y][x] = 1;
            }
        }
    }
    // 清除起点和目标点
    ctx.occupancy_grid[5][5] = 0;
    ctx.occupancy_grid[grid_sz - 10][grid_sz - 10] = 0;

    // 从 (5,5) 到 (90,90) 的 A* 搜索
    int start_x = 5, start_y = 5;
    int goal_x = grid_sz - 10, goal_y = grid_sz - 10;

    std::priority_queue<AStarNode, std::vector<AStarNode>, std::greater<AStarNode>> open_set;
    std::vector<std::vector<float>> g_score(grid_sz, std::vector<float>(grid_sz, 1e9f));
    std::vector<std::vector<std::pair<int, int>>> parent(
        grid_sz, std::vector<std::pair<int, int>>(grid_sz, {-1, -1}));

    open_set.push({start_x, start_y, 0.0f, heuristic(start_x, start_y, goal_x, goal_y)});
    g_score[start_y][start_x] = 0.0f;

    const int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const int dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

    bool found = false;
    while (!open_set.empty()) {
        AStarNode cur = open_set.top();
        open_set.pop();

        if (cur.x == goal_x && cur.y == goal_y) {
            found = true;
            break;
        }

        for (int d = 0; d < 8; ++d) {
            int nx = cur.x + dx[d];
            int ny = cur.y + dy[d];
            if (nx < 0 || nx >= grid_sz || ny < 0 || ny >= grid_sz) continue;
            if (ctx.occupancy_grid[ny][nx] == 1) continue;

            float move_cost = (dx[d] != 0 && dy[d] != 0) ? 1.414f : 1.0f;
            float tentative_g = cur.g + move_cost;
            if (tentative_g < g_score[ny][nx]) {
                g_score[ny][nx] = tentative_g;
                parent[ny][nx] = {cur.x, cur.y};
                open_set.push({nx, ny, tentative_g, heuristic(nx, ny, goal_x, goal_y)});
            }
        }
    }

    // 重建路径
    if (found) {
        int cx = goal_x, cy = goal_y;
        std::vector<std::pair<int, int>> raw_path;
        while (cx != -1) {
            raw_path.push_back({cx, cy});
            auto p = parent[cy][cx];
            cx = p.first;
            cy = p.second;
        }
        std::reverse(raw_path.begin(), raw_path.end());

        ctx.trajectory.path.reserve(raw_path.size());
        for (const auto &pt : raw_path) {
            TaskWaypoint wp;
            wp.x = static_cast<float>(pt.first) * 0.5f;
            wp.y = static_cast<float>(pt.second) * 0.5f;
            ctx.trajectory.path.push_back(wp);
        }
    } else {
        // 回退：直线路径
        ctx.trajectory.path.push_back({0.0f, 0.0f});
        ctx.trajectory.path.push_back({50.0f, 50.0f});
    }
}

// ============================================================================
// 节点 7：控制
//    从轨迹计算油门/刹车/转向的 PID 控制器。
//    依赖 planning。
// ============================================================================

void node_control(PipelineContext &ctx) {
    if (ctx.trajectory.path.empty()) {
        ctx.control_cmd = {0.0f, 0.0f, 0.0f};
        return;
    }

    // 当前自车位置（模拟）
    float ego_x = 2.0f, ego_y = 2.0f, ego_speed = 10.0f;

    // 找到最近的航点
    float min_dist = 1e9f;
    size_t closest = 0;
    for (size_t i = 0; i < ctx.trajectory.path.size(); ++i) {
        float dx = ego_x - ctx.trajectory.path[i].x;
        float dy = ego_y - ctx.trajectory.path[i].y;
        float d = dx * dx + dy * dy;
        if (d < min_dist) {
            min_dist = d;
            closest = i;
        }
    }

    // 前瞻点
    size_t lookahead = std::min(closest + 5, ctx.trajectory.path.size() - 1);
    float target_x = ctx.trajectory.path[lookahead].x;
    float target_y = ctx.trajectory.path[lookahead].y;

    // 横向跟踪误差
    float dx = target_x - ctx.trajectory.path[closest].x;
    float dy = target_y - ctx.trajectory.path[closest].y;
    float len = std::sqrt(dx * dx + dy * dy);
    float ex = ego_x - ctx.trajectory.path[closest].x;
    float ey = ego_y - ctx.trajectory.path[closest].y;
    float cte = (len > 1e-6f) ? (ex * dy - ey * dx) / len : 0.0f;

    // 横向 PID（转向）
    static float lateral_integral = 0.0f;
    static float lateral_prev_error = 0.0f;
    float kp_lat = 0.1f, ki_lat = 0.005f, kd_lat = 0.02f;
    lateral_integral += cte * 0.05f;
    lateral_integral = std::max(-0.5f, std::min(0.5f, lateral_integral));
    float lateral_deriv = (cte - lateral_prev_error) / 0.05f;
    lateral_prev_error = cte;
    float steering = kp_lat * cte + ki_lat * lateral_integral + kd_lat * lateral_deriv;

    // 纵向 PID（速度）
    float target_speed = 10.0f;
    static float long_integral = 0.0f;
    static float long_prev_error = 0.0f;
    float kp_long = 0.3f, ki_long = 0.01f, kd_long = 0.05f;
    float speed_err = target_speed - ego_speed;
    long_integral += speed_err * 0.05f;
    long_integral = std::max(-0.3f, std::min(0.3f, long_integral));
    float long_deriv = (speed_err - long_prev_error) / 0.05f;
    long_prev_error = speed_err;
    float accel = kp_long * speed_err + ki_long * long_integral + kd_long * long_deriv;

    ctx.control_cmd.throttle = std::max(0.0f, std::min(1.0f, accel));
    ctx.control_cmd.brake = std::max(0.0f, std::min(1.0f, -accel));
    ctx.control_cmd.steering = std::max(-1.0f, std::min(1.0f, steering));
}
