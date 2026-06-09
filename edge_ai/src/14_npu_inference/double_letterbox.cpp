#include "double_letterbox.h"

#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <algorithm>

// 简单的 Timer 类
class ScopedTimer {
public:
    ScopedTimer(double &result_ms) : result_(result_ms), started_(true) {
        start_ = std::chrono::high_resolution_clock::now();
    }
    ~ScopedTimer() {
        if (started_) {
            auto end = std::chrono::high_resolution_clock::now();
            result_ = std::chrono::duration<double, std::milli>(end - start_).count();
        }
    }
    void stop() {
        if (started_) {
            auto end = std::chrono::high_resolution_clock::now();
            result_ = std::chrono::duration<double, std::milli>(end - start_).count();
            started_ = false;
        }
    }

private:
    double &result_;
    bool started_;
    std::chrono::high_resolution_clock::time_point start_;
};

// ============================================================================
// 生成模拟的 BGR 图像数据
// 填充真实模式的像素数据以模拟实际相机输出
// ============================================================================
std::vector<uint8_t> generate_synthetic_image(int width, int height, int channels) {
    size_t totalPixels = static_cast<size_t>(width) * height * channels;
    std::vector<uint8_t> img(totalPixels);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int base = (y * width + x) * channels;
            // 创建渐变色以模拟真实场景
            img[base + 0] = static_cast<uint8_t>((x * 255) / width);                  // B
            img[base + 1] = static_cast<uint8_t>((y * 255) / height);                 // G
            img[base + 2] = static_cast<uint8_t>(((x + y) * 255) / (width + height)); // R
        }
    }

    return img;
}

// ============================================================================
// 双线性插值缩放
// 完整的 CPU 双线性插值实现，模拟真实的 RGA/CPU resize
//
// 算法: 对每个目标像素 (x', y') 反向映射到源图像坐标 (x, y)
//       在源图像上做 4 邻域双线性插值
// ============================================================================
std::vector<uint8_t> resize_bilinear_cpu(
    const uint8_t *src, int srcW, int srcH, int channels,
    int dstW, int dstH) {
    std::vector<uint8_t> dst(static_cast<size_t>(dstW) * dstH * channels);

    float scaleX = static_cast<float>(srcW) / dstW;
    float scaleY = static_cast<float>(srcH) / dstH;

    for (int dy = 0; dy < dstH; ++dy) {
        float sy = (dy + 0.5f) * scaleY - 0.5f;
        int y0 = static_cast<int>(std::floor(sy));
        int y1 = std::min(y0 + 1, srcH - 1);
        y0 = std::max(y0, 0);
        float vy = sy - y0;

        for (int dx = 0; dx < dstW; ++dx) {
            float sx = (dx + 0.5f) * scaleX - 0.5f;
            int x0 = static_cast<int>(std::floor(sx));
            int x1 = std::min(x0 + 1, srcW - 1);
            x0 = std::max(x0, 0);
            float vx = sx - x0;

            int dstBase = (dy * dstW + dx) * channels;
            int src00 = (y0 * srcW + x0) * channels;
            int src01 = (y0 * srcW + x1) * channels;
            int src10 = (y1 * srcW + x0) * channels;
            int src11 = (y1 * srcW + x1) * channels;

            for (int c = 0; c < channels; ++c) {
                float val = (1 - vx) * (1 - vy) * src[src00 + c] + vx * (1 - vy) * src[src01 + c] + (1 - vx) * vy * src[src10 + c] + vx * vy * src[src11 + c];
                dst[dstBase + c] = static_cast<uint8_t>(std::clamp(val, 0.0f, 255.0f));
            }
        }
    }

    return dst;
}

// ============================================================================
// 错误配置路径: 双 Letterbox
//
// 步骤:
//   1. 从 1920×1080 resize 到 640×640 (RGA 硬件快速)
//   2. SDK 检测 640≠512，自动触发 CPU resize (640→512) ← 双倍浪费
// ============================================================================
double resize_to_target_wrong(const uint8_t *src, int srcW, int srcH, int channels,
                              int wrongTarget, int actualTarget,
                              std::vector<uint8_t> &out) {
    double total_ms = 0;
    double step1_ms = 0, step2_ms = 0;

    // 步骤 1: RGA resize 到错误尺寸 (640×640)
    // 实际 RGA 硬件只需 ~0.3ms，但在 CPU 模拟上需要更多时间
    {
        ScopedTimer t(step1_ms);
        auto intermediate = resize_bilinear_cpu(src, srcW, srcH, channels,
                                                wrongTarget, wrongTarget);

        // 步骤 2: SDK 检测尺寸不匹配，CPU resize 到正确尺寸 (640→512)
        // 这 22ms 的开销是完全浪费的
        {
            ScopedTimer t2(step2_ms);
            out = resize_bilinear_cpu(intermediate.data(), wrongTarget, wrongTarget,
                                      channels, actualTarget, actualTarget);
        }
    }

    total_ms = step1_ms + step2_ms;

    std::cout << "    [错误配置] " << srcW << "×" << srcH << " → "
              << wrongTarget << "×" << wrongTarget << ": "
              << step1_ms << " ms\n";
    std::cout << "    [SDK 检测不匹配] " << wrongTarget << "≠" << actualTarget
              << " → CPU resize " << actualTarget << "×" << actualTarget << ": "
              << step2_ms << " ms\n";
    std::cout << "    总预处理耗时: " << total_ms << " ms (双 Letterbox 浪费)\n";

    return total_ms;
}

// ============================================================================
// 正确配置路径: 直接 resize 到模型输入尺寸
//
// 步骤:
//   1. 从 1920×1080 直接 resize 到 512×512
//   2. 完成
// ============================================================================
double resize_to_target_correct(const uint8_t *src, int srcW, int srcH, int channels,
                                int target,
                                std::vector<uint8_t> &out) {
    double ms = 0;
    {
        ScopedTimer t(ms);
        out = resize_bilinear_cpu(src, srcW, srcH, channels, target, target);
    }

    std::cout << "    [正确配置] " << srcW << "×" << srcH << " → "
              << target << "×" << target << ": "
              << ms << " ms\n";

    return ms;
}

// ============================================================================
// BGR→RGB 字节交换
// 每 3 字节一组，交换 B 和 R 通道 (0 和 2)
// 模拟 NEON vswp 指令的批量交换
// ============================================================================
void bgr_to_rgb_fast(std::vector<uint8_t> &image, int width, int height, int channels) {
    if (channels != 3) return;

    size_t totalPixels = static_cast<size_t>(width) * height;

    // 16 像素一组 (模拟 NEON 128-bit: 指令一次处理 16 字节 = 5.33 像素)
    size_t i = 0;
    for (; i + 16 <= totalPixels; i += 16) {
        for (size_t j = 0; j < 16; ++j) {
            size_t base = (i + j) * 3;
            std::swap(image[base], image[base + 2]);
        }
    }

    // 尾数处理
    for (; i < totalPixels; ++i) {
        size_t base = i * 3;
        std::swap(image[base], image[base + 2]);
    }

    std::cout << "    BGR→RGB 交换完成: " << totalPixels << " 像素\n";
}

// ============================================================================
// 模拟 SDK 内部的尺寸检测和修复
// 相当于 SDK 的隐式 resize 行为
// ============================================================================
bool detect_mismatch_and_fix(int providedW, int providedH,
                             int modelW, int modelH,
                             const uint8_t *src,
                             std::vector<uint8_t> &out) {
    if (providedW != modelW || providedH != modelH) {
        std::cout << "    ⚠ 尺寸不匹配! 提供: " << providedW << "×" << providedH
                  << ", 模型需要: " << modelW << "×" << modelH << "\n";
        std::cout << "    SDK 自动触发 CPU resize... (本次 resize 完全浪费)\n";

        out = resize_bilinear_cpu(src, providedW, providedH, 3, modelW, modelH);
        return true;
    }
    return false;
}
