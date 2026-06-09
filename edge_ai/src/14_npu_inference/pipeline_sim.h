#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <functional>

// ============================================================================
// NPU 推理管线仿真
//
// 完整管线:
//   Camera → MPP Decode → RGA CSC+Resize → NPU Inference → Post-process
//
// 两种运行模式:
//   wrong_config: 错误配置 (640×640 letterbox → SDK CPU resize 到 512×512)
//   correct_config: 正确配置 (直接 resize 到 512×512)
// ============================================================================

// 管线配置
struct PipelineConfig {
    int cameraWidth = 1920;
    int cameraHeight = 1080;
    int channels = 3;
    int modelInputW = 512;    // 模型实际输入宽度
    int modelInputH = 512;    // 模型实际输入高度
    int wrongLetterbox = 640; // 错误的硬编码 letterbox 尺寸
    bool useCorrectConfig = true;
};

// 单帧处理的阶段耗时 (微秒)
struct StageTimings {
    double capture_us = 0;      // Camera 采集 (模拟)
    double decode_us = 0;       // MPP 解码
    double resize_us = 0;       // RGA CSC+Resize
    double npu_us = 0;          // NPU 推理
    double post_us = 0;         // 后处理
    double doubleResize_us = 0; // 双 letterbox 额外开销 (仅 wrong config)
};

// 管线运行结果
struct PipelineResult {
    double totalTime_ms;     // 全管线单帧总耗时
    double effectiveFPS;     // 有效 FPS
    StageTimings avgTimings; // 各阶段平均耗时
    double npuUtilization;   // NPU 利用率 (%)
    std::string configLabel; // 配置描述
};

// ============================================================================
// 阶段 1: 模拟 Camera 采集 (生成 1920×1080 BGR 帧)
// ============================================================================
std::vector<uint8_t> simulate_camera_capture(int width, int height, int channels);

// ============================================================================
// 阶段 2: 模拟 MPP 解码 (JPEG/MJPEG 解码，实际 3ms)
// 这里做真实的字节拷贝 + 少量计算以模拟解码开销
// ============================================================================
std::vector<uint8_t> simulate_mpp_decode(const std::vector<uint8_t> &raw, int width, int height, int channels);

// ============================================================================
// 阶段 3: RGA CSC+Resize (颜色空间转换 + 缩放)
// 使用真实双线性插值，模拟 RGA 硬件行为
// ============================================================================
std::vector<uint8_t> simulate_rga_resize(const uint8_t *src, int srcW, int srcH,
                                         int channels, int dstW, int dstH);

// ============================================================================
// 阶段 4: NPU 推理模拟
// 对 512×512×3 图像做真实浮点矩阵运算 (卷积模拟)，耗时约 11ms
// ============================================================================
std::vector<float> simulate_npu_inference(const uint8_t *input, int width, int height, int channels);

// ============================================================================
// 阶段 5: 后处理 (FP16→FP32 转换 + NMS 模拟)
// ============================================================================
void simulate_postprocess(const std::vector<float> &npuOutput);

// ============================================================================
// 运行完整 pipeline 多帧，返回统计结果
// ============================================================================
PipelineResult run_pipeline_simulation(const PipelineConfig &config, int numFrames);

// ============================================================================
// 对比错误配置 vs 正确配置
// ============================================================================
void compare_configs(int numFrames);
