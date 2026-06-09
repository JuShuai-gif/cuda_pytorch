#include "io_persistence.h"
#include "double_letterbox.h"
#include "pipeline_sim.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <ctime>
#include <cstring>
#include <cmath>

// ============================================================================
// 打印分隔符
// ============================================================================
static void print_header(const std::string &title) {
    std::cout << "\n"
              << std::string(72, '=')
              << "\n  " << title
              << "\n"
              << std::string(72, '=') << "\n";
}

// ============================================================================
// 演示 1: IO 持久化
// ============================================================================
static void demo_io_persistence() {
    print_header("演示 1: IO 持久化 (预注册 DMA-BUF vs 每次分配)");

    // 模拟模型输入缓冲区大小: 512×512×3 = 786,432 字节 ≈ 768 KB
    const size_t modelInputSize = 512 * 512 * 3; // ~768 KB
    const int iterations = 100;

    std::cout << "\n  模型输入缓冲区: " << modelInputSize / 1024 << " KB\n";
    std::cout << "  模拟迭代轮数: " << iterations << "\n";

    double avg_input_set = simulate_rknn_input_set(modelInputSize, iterations);
    double avg_persistent = simulate_io_persistent(modelInputSize, iterations);

    std::cout << "\n  ┌──────────────────────────────────────────────┐\n";
    std::cout << "  │ IO 持久化对比                                │\n";
    std::cout << "  ├──────────────────────────────────────────────┤\n";
    std::cout << "  │ rknn_input_set (每次分配):   " << std::setw(7) << std::fixed
              << std::setprecision(2) << avg_input_set << " ms    │\n";
    std::cout << "  │ rknn_set_io_mem (持久化):    " << std::setw(7) << std::fixed
              << std::setprecision(2) << avg_persistent << " ms    │\n";
    std::cout << "  │ 加速比:                      " << std::setw(6) << std::fixed
              << std::setprecision(1)
              << (avg_input_set / std::max(avg_persistent, 0.0001))
              << " x     │\n";
    std::cout << "  └──────────────────────────────────────────────┘\n";

    // 模拟 FP16 转换对比
    print_header("演示 1b: NEON LUT FP16→FP32 转换加速");

    const size_t outputElements = 512 * 512 * 3; // 模型输出元素数
    std::vector<uint8_t> inputData(outputElements);
    std::vector<float> outputData(outputElements);

    // 填充模拟数据
    for (size_t i = 0; i < outputElements; ++i) {
        inputData[i] = static_cast<uint8_t>(i % 256);
    }

    std::cout << "\n  输出元素数: " << outputElements << " (≈ 3.1 MB FP32)\n";

    // 预热并计时
    double avg_naive = convert_fp16_naive(inputData.data(), outputData.data(), outputElements, 10);
    double avg_lut = convert_fp16_lut(inputData.data(), outputData.data(), outputElements, 10);

    std::cout << "\n  ┌──────────────────────────────────────────────┐\n";
    std::cout << "  │ FP16 转换对比                                │\n";
    std::cout << "  ├──────────────────────────────────────────────┤\n";
    std::cout << "  │ 逐元素 C 标量转换:          " << std::setw(7) << std::fixed
              << std::setprecision(2) << avg_naive << " ms    │\n";
    std::cout << "  │ NEON LUT 查表法:            " << std::setw(7) << std::fixed
              << std::setprecision(2) << avg_lut << " ms    │\n";
    std::cout << "  │ 加速比:                     " << std::setw(6) << std::fixed
              << std::setprecision(1)
              << (avg_naive / std::max(avg_lut, 0.0001))
              << " x     │\n";
    std::cout << "  └──────────────────────────────────────────────┘\n";
}

// ============================================================================
// 演示 2: 双 Letterbox 问题
// ============================================================================
static void demo_double_letterbox() {
    print_header("演示 2: 双 Letterbox 问题 (配置与模型不匹配)");

    const int cameraW = 1920;
    const int cameraH = 1080;
    const int channels = 3;
    const int wrongTarget = 640;  // 硬编码的错误 letterbox 尺寸
    const int actualTarget = 512; // 模型实际输入尺寸

    // 生成合成相机图像
    std::cout << "\n  生成合成相机图像: " << cameraW << "×" << cameraH
              << " (" << (cameraW * cameraH * channels / 1024) << " KB)\n";
    auto cameraFrame = generate_synthetic_image(cameraW, cameraH, channels);

    std::vector<uint8_t> wrongOut, correctOut;

    // 错误路径: 双 Letterbox
    std::cout << "\n  ── 错误配置路径 ──\n";
    std::cout << "  (预处理代码写死了 640×640，但模型实际是 512×512)\n";
    double wrongTime = resize_to_target_wrong(
        cameraFrame.data(), cameraW, cameraH, channels,
        wrongTarget, actualTarget, wrongOut);

    // 正确路径: 直接 resize
    std::cout << "\n  ── 正确配置路径 ──\n";
    std::cout << "  (通过命令行参数指定正确的模型输入尺寸 512×512)\n";
    double correctTime = resize_to_target_correct(
        cameraFrame.data(), cameraW, cameraH, channels,
        actualTarget, correctOut);

    // BGR→RGB 转换
    std::cout << "\n  ── BGR→RGB 快速转换 ──\n";
    bgr_to_rgb_fast(correctOut, actualTarget, actualTarget, channels);

    // 对比
    std::cout << "\n  ┌──────────────────────────────────────────────┐\n";
    std::cout << "  │ 双 Letterbox 对比                            │\n";
    std::cout << "  ├──────────────────────────────────────────────┤\n";
    std::cout << "  │ 错误路径 (640→512 双 resize): " << std::setw(7) << std::fixed
              << std::setprecision(2) << wrongTime << " ms    │\n";
    std::cout << "  │ 正确路径 (512 直接 resize):   " << std::setw(7) << std::fixed
              << std::setprecision(2) << correctTime << " ms    │\n";
    std::cout << "  │ 节省:                          " << std::setw(7) << std::fixed
              << std::setprecision(2) << (wrongTime - correctTime) << " ms    │\n";
    std::cout << "  │ 加速比:                        " << std::setw(6) << std::fixed
              << std::setprecision(1)
              << (wrongTime / std::max(correctTime, 0.001))
              << " x     │\n";
    std::cout << "  └──────────────────────────────────────────────┘\n";

    // 展示 SDK 检测
    std::cout << "\n  ── SDK 内部尺寸检测模拟 ──\n";
    std::vector<uint8_t> sdkOut;
    bool mismatch = detect_mismatch_and_fix(
        wrongTarget, wrongTarget, actualTarget, actualTarget,
        cameraFrame.data(), sdkOut);
    if (mismatch) {
        std::cout << "  结论: SDK 的自动修复导致了 22ms 的额外 CPU resize 开销\n";
        std::cout << "  教训: 始终确保预处理尺寸与模型输入尺寸一致!\n";
    }
}

// ============================================================================
// 演示 3: 完整管线模拟
// ============================================================================
static void demo_full_pipeline() {
    print_header("演示 3: 完整管线模拟 (错误 vs 正确配置)");

    std::cout << "\n  模拟管线:\n";
    std::cout << "    Camera (1920×1080) → MPP Decode → RGA Resize → NPU → Post\n";
    std::cout << "    错误配置: RGA resize 640×640 → SDK CPU resize 512×512\n";
    std::cout << "    正确配置: RGA resize 512×512 (一步到位)\n";

    int numFrames = 20; // 模拟 20 帧
    std::cout << "  模拟帧数: " << numFrames << "\n";

    compare_configs(numFrames);

    // 关键教训
    std::cout << "\n  ┌─────────────────────────────────────────────────────────────┐\n";
    std::cout << "  │ 关键教训                                                    │\n";
    std::cout << "  ├─────────────────────────────────────────────────────────────┤\n";
    std::cout << "  │ 1. 「外围比核心慢」: 管线开销 (IO+预处理+后处理) 远大于     │\n";
    std::cout << "  │    NPU 推理本身，优化重点是管线而非模型。                    │\n";
    std::cout << "  │ 2. 「配置匹配 = 零成本优化」: 一行参数纠正换来数倍提升。    │\n";
    std::cout << "  │ 3. 「默认 ≠ 最优」: SDK 的自动行为往往是性能陷阱。          │\n";
    std::cout << "  │ 4. 「先测再做」: 用 perf 数据定位瓶颈，不凭直觉优化。       │\n";
    std::cout << "  │ 5. 「IO 持久化 = 零拷贝基础」: 一次分配，持续复用。         │\n";
    std::cout << "  │ 6. 「NEON/SIMD 表驱动」: 查表法对数据格式转换提升显著。     │\n";
    std::cout << "  │ 7. 「利用率是诊断指标」: 低 NPU 利用率 = 有阻塞，排查管线。 │\n";
    std::cout << "  └─────────────────────────────────────────────────────────────┘\n";
}

// ============================================================================
// 写入结果 JSON
// ============================================================================
static void write_metrics_json() {
    std::ofstream of("npu_inference_metrics.json");

    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);

    of << "{\n";
    of << "  \"title\": \"NPU 推理优化案例研究 - RK3588\",\n";
    of << "  \"timestamp\": \"" << std::ctime(&time_t_now);
    of.seekp(-1, std::ios_base::cur);
    of << "\",\n";
    of << "  \"scenario\": {\n";
    of << "    \"hardware\": \"RK3588, 3-core NPU, 6 TOPS\",\n";
    of << "    \"pipeline\": \"Camera 1080p MJPEG → MPP Decode → RGA CSC+Resize → NPU Inference → Output\",\n";
    of << "    \"baseline_fps\": 13,\n";
    of << "    \"optimized_fps\": 29\n";
    of << "  },\n";
    of << "  \"optimizations\": [\n";
    of << "    {\n";
    of << "      \"name\": \"IO 持久化\",\n";
    of << "      \"method\": \"rknn_set_io_mem 预注册 DMA-BUF\",\n";
    of << "      \"before_ms\": 27,\n";
    of << "      \"after_ms\": 0.1,\n";
    of << "      \"improvement\": \"-99.6%\"\n";
    of << "    },\n";
    of << "    {\n";
    of << "      \"name\": \"双 Letterbox 修复\",\n";
    of << "      \"method\": \"匹配模型输入尺寸 (640→512)\",\n";
    of << "      \"before_ms\": 22,\n";
    of << "      \"after_ms\": 0.3,\n";
    of << "      \"improvement\": \"-98.6%\"\n";
    of << "    },\n";
    of << "    {\n";
    of << "      \"name\": \"NEON LUT FP16 转换\",\n";
    of << "      \"method\": \"256 元素查找表 + vst1q_f16\",\n";
    of << "      \"before_ms_range\": \"36-55\",\n";
    of << "      \"after_ms_range\": \"0.3-3.6\",\n";
    of << "      \"improvement\": \"-90~99%\"\n";
    of << "    }\n";
    of << "  ],\n";
    of << "  \"key_metrics_changes\": {\n";
    of << "    \"fps\": { \"before\": 13, \"after\": 29, \"improvement\": \"+123%\" },\n";
    of << "    \"npu_utilization\": { \"before\": \"27%\", \"after\": \"85%\", \"improvement\": \"+215%\" },\n";
    of << "    \"total_frame_time_ms\": { \"before\": \"~100\", \"after\": \"~15\", \"improvement\": \"-85%\" }\n";
    of << "  },\n";
    of << "  \"key_lessons\": [\n";
    of << "    \"外围比核心慢: 管线开销 82ms vs NPU 推理 11ms\",\n";
    of << "    \"默认 SDK 行为 ≠ 最优, 必须测量验证\",\n";
    of << "    \"配置匹配是性价比最高的优化\",\n";
    of << "    \"IO 持久化是零拷贝通信的基础\",\n";
    of << "    \"NEON/SIMD 表驱动对数据转换有数量级提升\",\n";
    of << "    \"利用率是逆向诊断指标: 低利用率 = 有阻塞\"\n";
    of << "  ]\n";
    of << "}\n";
    of.close();

    std::cout << "\n  指标已写入 npu_inference_metrics.json\n";
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  NPU 推理优化实战: RK3588 从 13fps 到 29fps                    ║\n";
    std::cout << "║  案例研究: 外围管线优化 > 模型本身                              ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";

    // 演示 1: IO 持久化
    demo_io_persistence();

    // 演示 2: 双 Letterbox
    demo_double_letterbox();

    // 演示 3: 完整管线
    demo_full_pipeline();

    // 写入指标 JSON
    write_metrics_json();

    std::cout << "\n  所有演示完成。\n";
    return 0;
}
