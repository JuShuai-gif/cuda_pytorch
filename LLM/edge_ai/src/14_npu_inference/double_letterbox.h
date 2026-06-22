#pragma once

#include <vector>
#include <cstdint>
#include <string>

// ============================================================================
// 双 Letterbox 问题演示
//
// 场景：预处理代码硬编码 model_input_width=640，但实际模型输入为 512x512
// SDK 检测到不匹配后自动执行 CPU resize (640→512)，造成双 Letterbox 浪费
//
// 相机帧: 1920×1080 → RGA resize 640×640 (22ms) → SDK CPU resize 512×512 (22ms)
// 实际需要: 1920×1080 → RGA resize 512×512 (一次即可，0.3ms)
// ============================================================================

// 生成模拟的 BGR 图像数据 (1920×1080×3)
std::vector<uint8_t> generate_synthetic_image(int width, int height, int channels);

// ============================================================================
// 双线性插值缩放 (模拟 RGA/CPU resize)
// 纯 CPU 实现，使用真实浮点运算，模拟 RGA 硬件的精度
// ============================================================================
std::vector<uint8_t> resize_bilinear_cpu(
    const uint8_t *src, int srcW, int srcH, int channels,
    int dstW, int dstH);

// ============================================================================
// 模拟错误配置导致的 "双 Letterbox" 路径：
//   Camera(1920×1080) → RGA resize 640×640 → SDK检测640≠512 → CPU resize 512×512
// 返回：总体预处理耗时 (ms)
// ============================================================================
double resize_to_target_wrong(const uint8_t *src, int srcW, int srcH, int channels,
                              int wrongTarget, int actualTarget,
                              std::vector<uint8_t> &out);

// ============================================================================
// 模拟正确配置路径：
//   Camera(1920×1080) → RGA resize 直接到 512×512
// 返回：总体预处理耗时 (ms)
// ============================================================================
double resize_to_target_correct(const uint8_t *src, int srcW, int srcH, int channels,
                                int target,
                                std::vector<uint8_t> &out);

// ============================================================================
// BGR→RGB 快速字节交换 (模拟 NEON vswp 指令)
// 每 3 字节一组交换 B 和 R 通道
// ============================================================================
void bgr_to_rgb_fast(std::vector<uint8_t> &image, int width, int height, int channels);

// ============================================================================
// 检测尺寸不匹配并触发 CPU resize (模拟 SDK 内部行为)
// 返回 true 表示发生了 double letterbox
// ============================================================================
bool detect_mismatch_and_fix(int providedW, int providedH,
                             int modelW, int modelH,
                             const uint8_t *src,
                             std::vector<uint8_t> &out);
