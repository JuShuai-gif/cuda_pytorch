#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

// ============================================================================
// NMS 中使用的检测框
// ============================================================================
struct DetectionBox {
    float x, y, w, h;
    float confidence;
    int class_id;
};

// ============================================================================
// 生成一张 1920x1080x3 uint8 合成图像，模拟真实相机帧。
// 用场景填充缓冲区: 渐变天空、地平面以及几何图形
// (圆形、矩形) 作为模拟物体。
// ============================================================================
void load_synthetic_image(uint8_t *img, int width, int height, int channels,
                          int frame_id, int64_t *out_timestamp_us);

// ============================================================================
// 使用双线性插值将 WxHxC uint8 图像缩放到 new_W x new_H x C float32。
// 输出值归一化到 [0, 1]。
// 返回消耗时间，单位微秒。
// ============================================================================
int64_t cpu_resize_normalize(const uint8_t *src, int src_w, int src_h, int c,
                             float *dst, int dst_w, int dst_h,
                             int64_t *out_timestamp_us);

// ============================================================================
// 点云的体素网格降采样。
// 输入: Nx3 float 数组 (x, y, z，单位米)。
// 输出: 降采样后的点 (每个被占据的体素一个点，使用质心)。
// 返回输出点数量。
// ============================================================================
size_t cpu_lidar_voxelize(const float *points, size_t num_points,
                          float voxel_size,
                          float *out_points, size_t max_out,
                          int64_t *out_timestamp_us);

// ============================================================================
// 检测框的贪心非极大值抑制。
// 按置信度降序排序，抑制 IoU > 阈值的框。
// 返回保留的框数量。
// ============================================================================
size_t cpu_nms(const DetectionBox *boxes, size_t num_boxes,
               float iou_threshold,
               DetectionBox *out_boxes, size_t max_out,
               int64_t *out_timestamp_us);
