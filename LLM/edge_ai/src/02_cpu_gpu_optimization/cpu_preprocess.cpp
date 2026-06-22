#include "cpu_preprocess.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstring>
#include <unordered_map>
#include <vector>

// ============================================================================
// 三维体素坐标的简单哈希
// ============================================================================
struct VoxelKey {
    int ix, iy, iz;
    bool operator==(const VoxelKey &o) const {
        return ix == o.ix && iy == o.iy && iz == o.iz;
    }
};

struct VoxelKeyHash {
    size_t operator()(const VoxelKey &k) const {
        return static_cast<size_t>(k.ix * 73856093
                                   ^ k.iy * 19349663
                                   ^ k.iz * 83492791);
    }
};

// ============================================================================
// 生成一张模拟机器人相机视角场景的合成图像
// ============================================================================
void load_synthetic_image(uint8_t *img, int width, int height, int channels,
                          int frame_id, int64_t *out_timestamp_us) {
    auto t0 = std::chrono::high_resolution_clock::now();

    const int horizon_y = height * 2 / 5; // 天空/地面分界线
    const int lane_y = horizon_y + height / 8;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * channels;
            int r, g, b;

            if (y < horizon_y) {
                // 天空渐变: 顶部为蓝色，靠近地平线变亮
                float t = static_cast<float>(y) / horizon_y;
                r = static_cast<int>(100 + 60 * t + (frame_id % 30));
                g = static_cast<int>(140 + 70 * t);
                b = static_cast<int>(200 + 55 * (1.0f - t));
            } else {
                // 地面: 绿灰色，带道路条纹
                float ground_t = static_cast<float>(y - horizon_y) / (height - horizon_y);
                r = static_cast<int>(60 + 40 * ground_t);
                g = static_cast<int>(100 + 30 * ground_t);
                b = static_cast<int>(50 + 20 * ground_t);

                // 道路标线: 水平车道线
                int lane_phase = ((y - lane_y) / 40 + frame_id) % 20;
                if (y >= lane_y && y < lane_y + 5 && lane_phase < 15) {
                    // 白色虚线车道
                    int stripe = (x / 80 + frame_id / 5) % 6;
                    if (stripe < 4) {
                        r = 230;
                        g = 230;
                        b = 220;
                    }
                }

                // 中心线
                if (x > width / 2 - 2 && x < width / 2 + 2) {
                    int stripe2 = ((y - lane_y) / 60 + frame_id / 5) % 4;
                    if (stripe2 < 3) {
                        r = 240;
                        g = 240;
                        b = 60;
                    }
                }
            }

            // 添加合成物体: 圆形表示 "汽车/障碍物"
            float fx = static_cast<float>(x);
            float fy = static_cast<float>(y);
            bool on_object = false;

            // 前方移动 "车辆"
            float obj_cx = static_cast<float>(width / 2 + static_cast<int>(std::sin(frame_id * 0.05f) * 80.0f));
            float obj_cy = static_cast<float>(lane_y + 30);
            float obj_dx = fx - obj_cx;
            float obj_dy = fy - obj_cy;
            if (obj_dx * obj_dx + obj_dy * obj_dy < 900.0f) {
                r = 200;
                g = 50;
                b = 50;
                on_object = true;
            }

            // 右侧行人
            float ped_cx = static_cast<float>(width - 200 - frame_id % 50);
            float ped_cy = static_cast<float>(lane_y + 80);
            float ped_dx = fx - ped_cx;
            float ped_dy = fy - ped_cy;
            if (ped_dx * ped_dx + ped_dy * ped_dy < 400.0f) {
                r = 50;
                g = 50;
                b = 200;
                on_object = true;
            }

            // 左侧交通标志
            float sign_cx = 180.0f;
            float sign_cy = static_cast<float>(horizon_y + 20);
            float sign_dx = fx - sign_cx;
            float sign_dy = fy - sign_cy;
            if (sign_dx * sign_dx + sign_dy * sign_dy < 250.0f) {
                r = 200;
                g = 180;
                b = 20;
                on_object = true;
            }

            // 添加传感器噪声 (类 Box-Muller 方法近似高斯分布)
            int noise = (static_cast<int>(fx * 7.0f + fy * 13.0f + frame_id * 31.0f) % 21) - 10;
            if (!on_object) {
                r = std::max(0, std::min(255, r + noise));
                g = std::max(0, std::min(255, g + noise));
                b = std::max(0, std::min(255, b + noise));
            }

            img[idx + 0] = static_cast<uint8_t>(r);
            img[idx + 1] = static_cast<uint8_t>(g);
            img[idx + 2] = static_cast<uint8_t>(b);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    *out_timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
                            t0.time_since_epoch())
                            .count();
}

// ============================================================================
// 双线性缩放 + 归一化到 float32 [0, 1]
// ============================================================================
int64_t cpu_resize_normalize(const uint8_t *src, int src_w, int src_h, int c,
                             float *dst, int dst_w, int dst_h,
                             int64_t *out_timestamp_us) {
    auto t0 = std::chrono::high_resolution_clock::now();

    const float scale_x = static_cast<float>(src_w) / dst_w;
    const float scale_y = static_cast<float>(src_h) / dst_h;

    for (int dy = 0; dy < dst_h; ++dy) {
        float sy = static_cast<float>(dy) * scale_y;
        int sy0 = static_cast<int>(sy);
        int sy1 = std::min(sy0 + 1, src_h - 1);
        float fy = sy - static_cast<float>(sy0);

        for (int dx = 0; dx < dst_w; ++dx) {
            float sx = static_cast<float>(dx) * scale_x;
            int sx0 = static_cast<int>(sx);
            int sx1 = std::min(sx0 + 1, src_w - 1);
            float fx = sx - static_cast<float>(sx0);

            for (int ch = 0; ch < c; ++ch) {
                float v00 = static_cast<float>(src[(sy0 * src_w + sx0) * c + ch]);
                float v10 = static_cast<float>(src[(sy0 * src_w + sx1) * c + ch]);
                float v01 = static_cast<float>(src[(sy1 * src_w + sx0) * c + ch]);
                float v11 = static_cast<float>(src[(sy1 * src_w + sx1) * c + ch]);

                float top = v00 * (1.0f - fx) + v10 * fx;
                float bot = v01 * (1.0f - fx) + v11 * fx;
                float val = top * (1.0f - fy) + bot * fy;

                // 归一化到 [0, 1] 并按通道优先布局存储: [C][H][W]
                dst[ch * dst_h * dst_w + dy * dst_w + dx] = val / 255.0f;
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    if (out_timestamp_us) {
        *out_timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                t0.time_since_epoch())
                                .count();
    }

    return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

// ============================================================================
// 体素网格降采样: 每个被占据体素的质心
// ============================================================================
size_t cpu_lidar_voxelize(const float *points, size_t num_points,
                          float voxel_size,
                          float *out_points, size_t max_out,
                          int64_t *out_timestamp_us) {
    auto t0 = std::chrono::high_resolution_clock::now();

    struct VoxelAccum {
        float sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f;
        int count = 0;
    };

    std::unordered_map<VoxelKey, VoxelAccum, VoxelKeyHash> voxels;

    float inv_vs = 1.0f / voxel_size;
    for (size_t i = 0; i < num_points; ++i) {
        int ix = static_cast<int>(std::floor(points[i * 3 + 0] * inv_vs));
        int iy = static_cast<int>(std::floor(points[i * 3 + 1] * inv_vs));
        int iz = static_cast<int>(std::floor(points[i * 3 + 2] * inv_vs));
        VoxelKey key{ix, iy, iz};
        auto &acc = voxels[key];
        acc.sum_x += points[i * 3 + 0];
        acc.sum_y += points[i * 3 + 1];
        acc.sum_z += points[i * 3 + 2];
        acc.count++;
    }

    size_t out_count = 0;
    for (const auto &kv : voxels) {
        if (out_count >= max_out) break;
        const auto &acc = kv.second;
        float inv = 1.0f / static_cast<float>(acc.count);
        out_points[out_count * 3 + 0] = acc.sum_x * inv;
        out_points[out_count * 3 + 1] = acc.sum_y * inv;
        out_points[out_count * 3 + 2] = acc.sum_z * inv;
        ++out_count;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    if (out_timestamp_us) {
        *out_timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                t0.time_since_epoch())
                                .count();
    }
    return out_count;
}

// ============================================================================
// 贪心非极大值抑制
// ============================================================================
static float box_iou(const DetectionBox &a, const DetectionBox &b) {
    float ax1 = a.x;
    float ay1 = a.y;
    float ax2 = a.x + a.w;
    float ay2 = a.y + a.h;
    float bx1 = b.x;
    float by1 = b.y;
    float bx2 = b.x + b.w;
    float by2 = b.y + b.h;

    float inter_x1 = std::max(ax1, bx1);
    float inter_y1 = std::max(ay1, by1);
    float inter_x2 = std::min(ax2, bx2);
    float inter_y2 = std::min(ay2, by2);

    float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;

    float area_a = a.w * a.h;
    float area_b = b.w * b.h;
    float union_area = area_a + area_b - inter_area;

    if (union_area <= 0.0f) return 0.0f;
    return inter_area / union_area;
}

size_t cpu_nms(const DetectionBox *boxes, size_t num_boxes,
               float iou_threshold,
               DetectionBox *out_boxes, size_t max_out,
               int64_t *out_timestamp_us) {
    auto t0 = std::chrono::high_resolution_clock::now();

    if (num_boxes == 0) {
        auto t1 = std::chrono::high_resolution_clock::now();
        if (out_timestamp_us) {
            *out_timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                    t0.time_since_epoch())
                                    .count();
        }
        return 0;
    }

    // 复制并按置信度降序排序
    std::vector<DetectionBox> sorted(boxes, boxes + num_boxes);
    std::sort(sorted.begin(), sorted.end(),
              [](const DetectionBox &a, const DetectionBox &b) {
                  return a.confidence > b.confidence;
              });

    std::vector<bool> suppressed(num_boxes, false);
    size_t kept = 0;

    for (size_t i = 0; i < num_boxes && kept < max_out; ++i) {
        if (suppressed[i]) continue;
        out_boxes[kept++] = sorted[i];

        for (size_t j = i + 1; j < num_boxes; ++j) {
            if (suppressed[j]) continue;
            if (box_iou(sorted[i], sorted[j]) > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    if (out_timestamp_us) {
        *out_timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                t0.time_since_epoch())
                                .count();
    }
    return kept;
}
