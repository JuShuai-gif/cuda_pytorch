#include "pipeline_sim.h"
#include "double_letterbox.h"
#include "io_persistence.h"

#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <numeric>
#include <fstream>

// ============================================================================
// 阶段 1: 模拟 Camera 采集
// ============================================================================
std::vector<uint8_t> simulate_camera_capture(int width, int height, int channels) {
    return generate_synthetic_image(width, height, channels);
}

// ============================================================================
// 阶段 2: 模拟 MPP 解码
// 真实 MPP 硬件解码约 3ms (1080p MJPEG)
// 这里做真实的数据搬运 + 格式转换以模拟解码开销
// ============================================================================
std::vector<uint8_t> simulate_mpp_decode(const std::vector<uint8_t> &raw, int width, int height, int channels) {
    size_t totalSize = static_cast<size_t>(width) * height * channels;
    std::vector<uint8_t> decoded(totalSize);

    // 模拟 MJPEG 解码: YUV → BGR 转换 + 数据重排
    // 实际解码涉及 IDCT + 反量化，这里做等效的整数运算
    for (size_t i = 0; i < totalSize; i += 3) {
        // 模拟 YUV→BGR 矩阵乘法 (真实 MPP 用硬件矩阵单元)
        int y = raw[i];      // 模拟 Y 分量
        int cb = raw[i + 1]; // 模拟 Cb 分量
        int cr = raw[i + 2]; // 模拟 Cr 分量

        int r = y + 1.402f * (cr - 128);
        int g = y - 0.344136f * (cb - 128) - 0.714136f * (cr - 128);
        int b = y + 1.772f * (cb - 128);

        decoded[i] = static_cast<uint8_t>(std::clamp(b, 0, 255));
        decoded[i + 1] = static_cast<uint8_t>(std::clamp(g, 0, 255));
        decoded[i + 2] = static_cast<uint8_t>(std::clamp(r, 0, 255));
    }

    return decoded;
}

// ============================================================================
// 阶段 3: RGA CSC+Resize
// ============================================================================
std::vector<uint8_t> simulate_rga_resize(const uint8_t *src, int srcW, int srcH,
                                         int channels, int dstW, int dstH) {
    return resize_bilinear_cpu(src, srcW, srcH, channels, dstW, dstH);
}

// ============================================================================
// 阶段 4: NPU 推理模拟
// 使用真实的浮点运算来模拟 NPU 推理的计算量
//
// 模拟 YOLO 风格推理:
//   输入: 512×512×3 (786,432 像素)
//   操作: 3×3 卷积模拟 (真实模型有多层, 这里模拟等效计算量)
//   目标: 产生 ~11ms 的计算时间 (取决于 CPU)
// ============================================================================
std::vector<float> simulate_npu_inference(const uint8_t *input, int width, int height, int /*channels*/) {
    int featureDim = 80; // 输出特征图维度 (YOLO 检测头输出)
    int numAnchors = 3;
    int numClasses = 10;

    // 输出: [anchors × (5 + classes) × featureH × featureW]
    int outputSize = numAnchors * (5 + numClasses) * featureDim * featureDim;
    std::vector<float> output(outputSize);

    // 模拟卷积: 3×3 卷积核在 512×512 滑动
    // 真实 NPU 用 MAC 阵列做，这里用浮点模拟等效计算量
    int stride = 8; // 下采样步长
    int outH = height / stride;
    int outW = width / stride;

    // 每个输出位置的 3×3 卷积计算
    const float kernel[3][3][3] = {
        {{0.1f, 0.2f, 0.1f}, {0.0f, 0.5f, 0.0f}, {-0.1f, 0.2f, -0.1f}},
        {{0.3f, 0.0f, 0.3f}, {0.1f, 0.4f, 0.1f}, {0.0f, 0.3f, 0.0f}},
        {{0.2f, 0.1f, 0.2f}, {0.3f, 0.0f, 0.3f}, {0.1f, 0.2f, 0.1f}}};

    // 为每个输出通道计算
    for (int oc = 0; oc < 12; ++oc) { // 12 个输出通道
        float bias = (oc - 6) * 0.1f;

        for (int oy = 1; oy < outH - 1; ++oy) {
            for (int ox = 1; ox < outW - 1; ++ox) {
                float sum = bias;

                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        for (int c = 0; c < 3; ++c) {
                            int sy = oy * stride + ky + 1;
                            int sx = ox * stride + kx + 1;
                            int srcIdx = (sy * width + sx) * 3 + c;
                            sum += input[srcIdx] * kernel[ky + 1][kx + 1][c];
                        }
                    }
                }

                float relu = std::max(0.0f, sum);

                int outIdx = (oc * outH + oy) * outW + ox;
                if (outIdx < outputSize) {
                    output[outIdx] = relu;
                }
            }
        }
    }

    return output;
}

// ============================================================================
// 阶段 5: 后处理
// 模拟: FP16→FP32 转换 + NMS + 检测框解码
// ============================================================================
void simulate_postprocess(const std::vector<float> &npuOutput) {
    size_t n = npuOutput.size();

    // 模拟 FP16→FP32 转换 (在 IO persistence 演示中已覆盖)
    // 这里做 NMS 模拟 (非极大值抑制)
    float threshold = 0.5f;
    size_t numBoxes = n / 10; // 模拟检测框数量
    if (numBoxes == 0) return;

    // 模拟 NMS: 对检测框做 score 过滤
    size_t kept = 0;
    for (size_t i = 0; i < numBoxes && i < 1000; ++i) {
        // 模拟 score (每框的置信度)
        float score = 0;
        for (size_t j = 0; j < 5 && (i * 5 + j) < n; ++j) {
            score += npuOutput[i * 5 + j];
        }
        if (score > threshold) {
            ++kept;
        }
    }

    // 保持 volatile 防止被优化掉
    volatile size_t keptVolatile = kept;
    (void)keptVolatile;
}

// ============================================================================
// 运行完整管线
// ============================================================================
PipelineResult run_pipeline_simulation(const PipelineConfig &config, int numFrames) {
    PipelineResult result;
    result.configLabel = config.useCorrectConfig ? "正确配置 (512×512 直接 resize)" :
                                                   "错误配置 (640→512 双 Letterbox)";

    std::vector<double> allFrameTimes; // 记录每帧耗时用于统计
    StageTimings accumulatedTimings;

    std::cout << "\n  运行管线: " << result.configLabel << "\n";
    std::cout << "  帧数: " << numFrames << "\n";

    for (int frame = 0; frame < numFrames; ++frame) {
        auto frame_start = std::chrono::high_resolution_clock::now();

        // ── 阶段 1: Camera 采集 ──
        auto t1 = std::chrono::high_resolution_clock::now();
        auto rawFrame = simulate_camera_capture(config.cameraWidth, config.cameraHeight, config.channels);
        auto t2 = std::chrono::high_resolution_clock::now();
        accumulatedTimings.capture_us +=
            std::chrono::duration<double, std::micro>(t2 - t1).count();

        // ── 阶段 2: MPP 解码 ──
        auto t3 = std::chrono::high_resolution_clock::now();
        auto decoded = simulate_mpp_decode(rawFrame, config.cameraWidth, config.cameraHeight, config.channels);
        auto t4 = std::chrono::high_resolution_clock::now();
        accumulatedTimings.decode_us +=
            std::chrono::duration<double, std::micro>(t4 - t3).count();

        // ── 阶段 3: RGA Resize ──
        std::vector<uint8_t> resized;
        auto t5 = std::chrono::high_resolution_clock::now();

        if (config.useCorrectConfig) {
            // 正确配置: 直接 resize 到模型尺寸
            resized = simulate_rga_resize(decoded.data(),
                                          config.cameraWidth, config.cameraHeight,
                                          config.channels,
                                          config.modelInputW, config.modelInputH);
        } else {
            // 错误配置: 先 resize 到错误尺寸, SDK 再修复
            // 模拟第一次 resize 到 640×640
            auto intermediate = simulate_rga_resize(decoded.data(),
                                                    config.cameraWidth, config.cameraHeight,
                                                    config.channels,
                                                    config.wrongLetterbox, config.wrongLetterbox);

            // 模拟 SDK 检测到不匹配, CPU resize → 512×512
            auto t5b = std::chrono::high_resolution_clock::now();
            resized = simulate_rga_resize(intermediate.data(),
                                          config.wrongLetterbox, config.wrongLetterbox,
                                          config.channels,
                                          config.modelInputW, config.modelInputH);
            auto t5c = std::chrono::high_resolution_clock::now();
            accumulatedTimings.doubleResize_us +=
                std::chrono::duration<double, std::micro>(t5c - t5b).count();
        }

        auto t6 = std::chrono::high_resolution_clock::now();
        accumulatedTimings.resize_us +=
            std::chrono::duration<double, std::micro>(t6 - t5).count();

        // ── 阶段 4: NPU 推理 ──
        auto t7 = std::chrono::high_resolution_clock::now();
        auto npuOut = simulate_npu_inference(resized.data(),
                                             config.modelInputW, config.modelInputH,
                                             config.channels);
        auto t8 = std::chrono::high_resolution_clock::now();
        accumulatedTimings.npu_us +=
            std::chrono::duration<double, std::micro>(t8 - t7).count();

        // ── 阶段 5: 后处理 ──
        auto t9 = std::chrono::high_resolution_clock::now();
        simulate_postprocess(npuOut);
        auto t10 = std::chrono::high_resolution_clock::now();
        accumulatedTimings.post_us +=
            std::chrono::duration<double, std::micro>(t10 - t9).count();

        auto frame_end = std::chrono::high_resolution_clock::now();
        double frame_ms = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
        allFrameTimes.push_back(frame_ms);

        // 进度提示 (每 10 帧)
        if ((frame + 1) % 10 == 0 || frame == numFrames - 1) {
            std::cout << "\r    帧 " << (frame + 1) << "/" << numFrames
                      << " | 当前帧: " << std::fixed << std::setprecision(1)
                      << frame_ms << " ms" << std::flush;
        }
    }
    std::cout << "\n";

    // 计算统计数据
    int n = numFrames;
    accumulatedTimings.capture_us /= n;
    accumulatedTimings.decode_us /= n;
    accumulatedTimings.resize_us /= n;
    accumulatedTimings.npu_us /= n;
    accumulatedTimings.post_us /= n;
    accumulatedTimings.doubleResize_us /= n;

    result.avgTimings = accumulatedTimings;

    // 计算平均帧时间 (去掉第一帧预热)
    double avgFrameMs = 0;
    if (numFrames > 1) {
        avgFrameMs = std::accumulate(allFrameTimes.begin() + 1, allFrameTimes.end(), 0.0)
                     / (numFrames - 1);
    } else {
        avgFrameMs = allFrameTimes[0];
    }

    result.totalTime_ms = avgFrameMs;
    result.effectiveFPS = 1000.0 / avgFrameMs;

    // NPU 利用率 = NPU 推理时间 / 总帧时间
    result.npuUtilization = (std::isnan(avgFrameMs) || avgFrameMs == 0) ? 0 :
                                                                          (accumulatedTimings.npu_us / 1000.0 / avgFrameMs * 100.0);

    // 打印汇总
    std::cout << "\n  ┌─────────────────────────────────────────────────────┐\n";
    std::cout << "  │ 管线阶段耗时汇总 (平均)                             │\n";
    std::cout << "  ├─────────────────────────────────────────────────────┤\n";
    std::cout << "  │ Camera 采集:     " << std::setw(8) << std::fixed << std::setprecision(2)
              << (accumulatedTimings.capture_us / 1000) << " ms           │\n";
    std::cout << "  │ MPP 解码:        " << std::setw(8) << std::fixed << std::setprecision(2)
              << (accumulatedTimings.decode_us / 1000) << " ms           │\n";
    std::cout << "  │ RGA Resize:      " << std::setw(8) << std::fixed << std::setprecision(2)
              << (accumulatedTimings.resize_us / 1000) << " ms           │\n";
    if (accumulatedTimings.doubleResize_us > 0) {
        std::cout << "  │   └─ 双Letterbox: " << std::setw(8) << std::fixed << std::setprecision(2)
                  << (accumulatedTimings.doubleResize_us / 1000) << " ms (浪费)    │\n";
    }
    std::cout << "  │ NPU 推理:        " << std::setw(8) << std::fixed << std::setprecision(2)
              << (accumulatedTimings.npu_us / 1000) << " ms           │\n";
    std::cout << "  │ 后处理:          " << std::setw(8) << std::fixed << std::setprecision(2)
              << (accumulatedTimings.post_us / 1000) << " ms           │\n";
    std::cout << "  ├─────────────────────────────────────────────────────┤\n";
    std::cout << "  │ 总帧时间 (平均): " << std::setw(8) << std::fixed << std::setprecision(2)
              << result.totalTime_ms << " ms           │\n";
    std::cout << "  │ 有效 FPS:         " << std::setw(8) << std::fixed << std::setprecision(1)
              << result.effectiveFPS << "             │\n";
    std::cout << "  │ NPU 利用率:      " << std::setw(7) << std::fixed << std::setprecision(1)
              << result.npuUtilization << "%           │\n";
    std::cout << "  └─────────────────────────────────────────────────────┘\n";

    return result;
}

// ============================================================================
// 对比错误配置 vs 正确配置
// ============================================================================
void compare_configs(int numFrames) {
    std::cout << "\n"
              << std::string(72, '=') << "\n"
              << "  管线配置对比: 错误配置 vs 正确配置\n"
              << std::string(72, '=') << "\n";

    // 错误配置
    PipelineConfig wrongCfg;
    wrongCfg.useCorrectConfig = false;
    wrongCfg.wrongLetterbox = 640;
    auto wrongResult = run_pipeline_simulation(wrongCfg, numFrames);

    // 正确配置
    PipelineConfig correctCfg;
    correctCfg.useCorrectConfig = true;
    auto correctResult = run_pipeline_simulation(correctCfg, numFrames);

    // 对比
    std::cout << "\n"
              << std::string(72, '=') << "\n"
              << "  最终对比\n"
              << std::string(72, '=') << "\n";

    std::cout << "  ┌──────────────────┬──────────────┬──────────────┬──────────┐\n";
    std::cout << "  │ 指标             │ 错误配置    │ 正确配置    │ 改善     │\n";
    std::cout << "  ├──────────────────┼──────────────┼──────────────┼──────────┤\n";
    std::cout << "  │ Resize 耗时 (ms) │ " << std::setw(10) << std::fixed << std::setprecision(1)
              << (wrongResult.avgTimings.resize_us / 1000)
              << "  │ " << std::setw(10)
              << (correctResult.avgTimings.resize_us / 1000)
              << "  │ " << std::setw(6)
              << ((1 - correctResult.avgTimings.resize_us / std::max(wrongResult.avgTimings.resize_us, 1.0)) * 100)
              << "%  │\n";
    std::cout << "  │ 总帧时间 (ms)   │ " << std::setw(10) << std::fixed << std::setprecision(1)
              << wrongResult.totalTime_ms
              << "  │ " << std::setw(10)
              << correctResult.totalTime_ms
              << "  │ " << std::setw(6)
              << ((1 - correctResult.totalTime_ms / std::max(wrongResult.totalTime_ms, 0.001)) * 100)
              << "%  │\n";
    std::cout << "  │ FPS             │ " << std::setw(10) << std::fixed << std::setprecision(1)
              << wrongResult.effectiveFPS
              << "  │ " << std::setw(10)
              << correctResult.effectiveFPS
              << "  │ +" << std::setw(5)
              << (correctResult.effectiveFPS - wrongResult.effectiveFPS)
              << "  │\n";
    std::cout << "  │ NPU 利用率 (%)  │ " << std::setw(10) << std::fixed << std::setprecision(1)
              << wrongResult.npuUtilization
              << "  │ " << std::setw(10)
              << correctResult.npuUtilization
              << "  │ " << std::setw(6)
              << (correctResult.npuUtilization - wrongResult.npuUtilization)
              << "%  │\n";
    std::cout << "  └──────────────────┴──────────────┴──────────────┴──────────┘\n";
}
