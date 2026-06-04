/*
 * 05_cnn_terminology.cpp - 第 6 章：卷积神经网络
 * CNN 关键术语演示：池化、步长、特征图、感受野（对应原书第 203-206 页）
 *
 * 演示内容：
 *   1. 池化 (Pooling)        - MaxPool2d vs AvgPool2d 对比
 *   2. 步长 (Stride)         - stride=1 与 stride=2 空间分辨率影响
 *   3. 特征图 (Feature Maps) - 4 个滤波器产生 4 张特征图
 *   4. 感受野 (Receptive Field) - 3 层 3×3 conv 的感受野逐层计算
 */

#include <torch/torch.h>
#include <iostream>
#include <cmath>

/* =========================== 1. 池化演示 =========================== */

/* 对比最大池化与平均池化的形状与语义差异
 * @param device: 计算设备 (CPU/GPU)
 */
void demo_pooling(torch::Device device) {
    std::cout << "\n"
              << "【1. 池化 (Pooling) 演示】" << "\n"
              << "═══════════════════════════════════════════════════════" << std::endl;

    // 创建随机特征图 [batch_size=1, channels=1, height=28, width=28]
    auto feature_map = torch::rand({1, 1, 28, 28}).to(device);

    std::cout << "  原始特征图形状: " << feature_map.sizes() << "  (batch×channels×H×W)" << std::endl;

    /* 最大池化：2×2 窗口，取窗口内最大值 */
    auto max_pooled = torch::max_pool2d(feature_map, 2);
    std::cout << "  MaxPool2d(kernel=2×2)   形状: " << max_pooled.sizes()
              << "   缩小比: 1/4 (28→14)" << std::endl;

    /* 平均池化：2×2 窗口，取窗口内均值 */
    auto avg_pooled = torch::avg_pool2d(feature_map, 2);
    std::cout << "  AvgPool2d(kernel=2×2)   形状: " << avg_pooled.sizes()
              << "   缩小比: 1/4 (28→14)" << std::endl;

    // 对比两种池化在相同区域的取值
    auto region = feature_map.index({0, 0,
                                     torch::indexing::Slice(0, 2),
                                     torch::indexing::Slice(0, 2)});
    auto max_val = region.max();
    auto avg_val = region.mean();

    std::cout << "\n  对比左上角 2×2 区域:\n"
              << "    区域值: " << region.sizes() << "\n"
              << "    MaxPool 取值: " << max_val.item<float>() << "\n"
              << "    AvgPool 取值: " << avg_val.item<float>() << "\n"
              << std::endl;

    // 池化说明（中文）
    std::cout << "  ┌─ 池化层的作用 ─────────────────────────────────┐\n"
              << "  │ MaxPool：提取最显著的特征，保留纹理和边缘信息 │\n"
              << "  │          对微小位移具有平移不变性               │\n"
              << "  │ AvgPool：平滑特征图，保留整体背景信息           │\n"
              << "  │          常用于网络末端的全局平均池化           │\n"
              << "  └────────────────────────────────────────────────┘\n"
              << std::endl;
}

/* =========================== 2. 步长演示 =========================== */

/* 对比不同 stride 对输出空间分辨率的影响
 * 公式: out = floor((in - kernel + 2*pad) / stride) + 1
 * @param device: 计算设备 (CPU/GPU)
 */
void demo_stride(torch::Device device) {
    std::cout << "\n"
              << "【2. 步长 (Stride) 演示】" << "\n"
              << "═══════════════════════════════════════════════════════" << std::endl;

    // 输入：单通道 28×28 图像
    auto input = torch::rand({1, 1, 28, 28}).to(device);

    /* 步长 = 1 的卷积层 */
    auto conv_s1 = torch::nn::Conv2d(
        torch::nn::Conv2dOptions(1, 1, 3).stride(1).padding(0));
    conv_s1->to(device);

    /* 步长 = 2 的卷积层 */
    auto conv_s2 = torch::nn::Conv2d(
        torch::nn::Conv2dOptions(1, 1, 3).stride(2).padding(0));
    conv_s2->to(device);

    // 前向传播
    auto out_s1 = conv_s1->forward(input);
    auto out_s2 = conv_s2->forward(input);

    // 理论输出尺寸
    int in_size = 28, kernel = 3, pad = 0;
    int out_s1_theory = (in_size - kernel + 2 * pad) / 1 + 1;
    int out_s2_theory = (in_size - kernel + 2 * pad) / 2 + 1;

    std::cout << "  输入形状: " << input.sizes() << "  (batch=1, channels=1, 28×28)"
              << "\n  卷积核: 3×3, padding=0" << std::endl;

    std::cout << "\n  stride=1  Conv2d:\n"
              << "    实际输出形状: " << out_s1.sizes()
              << "  |  理论尺寸: " << out_s1_theory << "×" << out_s1_theory
              << "\n    公式: (28 - 3 + 2×0) / 1 + 1 = " << out_s1_theory << std::endl;

    std::cout << "\n  stride=2  Conv2d:\n"
              << "    实际输出形状: " << out_s2.sizes()
              << "  |  理论尺寸: " << out_s2_theory << "×" << out_s2_theory
              << "\n    公式: (28 - 3 + 2×0) / 2 + 1 = " << out_s2_theory
              << "\n    空间分辨率减半（28x28 → 13x13），减少计算量\n"
              << std::endl;

    std::cout << "  ┌─ Stride 的作用 ─────────────────────────────────┐\n"
              << "  │ stride=1: 密集采样，保留更多空间细节         │\n"
              << "  │ stride=2: 下采样，空间分辨率减半              │\n"
              << "  │          可替代池化层实现降维                 │\n"
              << "  │          减少后续层计算量 → 加速训练          │\n"
              << "  └────────────────────────────────────────────────┘\n"
              << std::endl;
}

/* =========================== 3. 特征图演示 =========================== */

/* 展示 Conv2d 中多个滤波器产生多张特征图的过程
 * 说明：实际可视化需 matplotlib，此处只打印形状并解释概念
 * @param device: 计算设备 (CPU/GPU)
 */
void demo_feature_maps(torch::Device device) {
    std::cout << "\n"
              << "【3. 特征图 (Feature Maps) 演示】" << "\n"
              << "═══════════════════════════════════════════════════════" << std::endl;

    // 输入：单通道 28×28（如 MNIST 灰度图）
    auto input = torch::rand({1, 1, 28, 28}).to(device);

    // 1 输入通道 → 4 输出通道（4 个 3×3 卷积核）
    auto conv = torch::nn::Conv2d(
        torch::nn::Conv2dOptions(1, 4, 3).stride(1).padding(0));
    conv->to(device);

    auto output = conv->forward(input);

    std::cout << "  输入形状: " << input.sizes()
              << "  (1 张 28×28 灰度图)" << std::endl;
    std::cout << "  卷积层: Conv2d(in_ch=1, out_ch=4, kernel=3×3)" << std::endl;
    std::cout << "  输出形状: " << output.sizes()
              << "  (4 张 26×26 特征图)" << std::endl;

    // 分别展示每张特征图的统计信息
    std::cout << "\n  各特征图统计:" << std::endl;
    for (int i = 0; i < 4; ++i) {
        auto feat_i = output.index({0, i}); // [H, W]
        auto mean_val = feat_i.mean().item<float>();
        auto std_val = feat_i.std().item<float>();
        auto min_val = feat_i.min().item<float>();
        auto max_val = feat_i.max().item<float>();
        std::cout << "    特征图 #" << (i + 1) << " [" << feat_i.size(0)
                  << "×" << feat_i.size(1) << "]:"
                  << "  μ=" << mean_val << ", σ=" << std_val
                  << ", min=" << min_val << ", max=" << max_val << std::endl;
    }

    std::cout << "\n  ┌─ 特征图的概念 ─────────────────────────────────┐\n"
              << "  │ 每个卷积核学习到一种特定的特征检测模式        │\n"
              << "  │ 如边缘检测器、角点检测器、纹理检测器等         │\n"
              << "  │ 浅层特征图: 检测边缘、颜色、纹理等低级特征    │\n"
              << "  │ 深层特征图: 检测形状、物体部件等高级语义      │\n"
              << "  │ 特征图可视化需借助 matplotlib / OpenCV          │\n"
              << "  └────────────────────────────────────────────────┘\n"
              << std::endl;
}

/* =========================== 4. 感受野计算 =========================== */

/* 计算并打印 3 层 3×3 卷积的感受野变化
 * 递推公式: RF_{k} = RF_{k-1} + (kernel_size - 1) × cumulative_stride
 * 当 stride=1 时: RF_{k} = RF_{k-1} + (k - 1)
 * @param device: 计算设备 (CPU/GPU)
 */
void demo_receptive_field(torch::Device /* unused */) {
    std::cout << "\n"
              << "【4. 感受野 (Receptive Field) 计算】" << "\n"
              << "═══════════════════════════════════════════════════════"
              << std::endl;

    std::cout << "  条件: 3 层连续 3×3 卷积，stride=1, padding=0" << std::endl;
    std::cout << "  递推公式: RF(k) = RF(k-1) + (kernel - 1) × 累积步长积\n"
              << std::endl;

    // 逐层计算感受野
    int rf = 1; // 输入层每个像素的感受野为自身
    std::cout << "  Layer 0 (输入层):        RF = " << rf << "×" << rf << std::endl;

    for (int layer = 1; layer <= 3; ++layer) {
        rf += (3 - 1); // kernel=3, stride=1 → 每次增加 2
        std::cout << "  Layer " << layer << " (3×3 Conv):        RF = "
                  << rf << "×" << rf
                  << "  (+2 = 3-1)" << std::endl;
    }

    int feature_size = 28;
    std::cout << "\n  ┌─ 感受野的含义 ─────────────────────────────────┐\n"
              << "  │ 感受野 = 输出特征图上某个像素\"看到\"的        │\n"
              << "  │         输入图像区域大小                       │\n"
              << "  │                                                │\n"
              << "  │ 第 1 层 3×3 Conv: RF = 3                       │\n"
              << "  │   输出像素由输入 3×3 区域计算得出             │\n"
              << "  │                                                │\n"
              << "  │ 第 2 层 3×3 Conv: RF = 5                       │\n"
              << "  │   第 1 层的 3 个输出 → 映射回输入 5 个像素    │\n"
              << "  │                                                │\n"
              << "  │ 第 3 层 3×3 Conv: RF = 7                       │\n"
              << "  │   深层像素整合了输入 7×7 区域的信息           │\n"
              << "  │                                                │\n"
              << "  │ 对 " << feature_size << "×" << feature_size << " 输入: 3 层 3×3 conv 后特征图尺寸 = "
              << (feature_size - 2 * 3) << "×" << (feature_size - 2 * 3) << "                       │\n"
              << "  │                                                │\n"
              << "  │ 设计启示:                                     │\n"
              << "  │  · 堆叠小卷积核 (3×3) 可替代大卷积核 (7×7)    │\n"
              << "  │  · 同等感受野下参数更少、非线性更强           │\n"
              << "  │  · 这就是 VGG 网络的核心设计理念               │\n"
              << "  └────────────────────────────────────────────────┘\n"
              << std::endl;
}

/* ================================ main =================================== */

int main() {
    std::cout << "\n"
              << "╔══════════════════════════════════════════════════════════╗\n"
              << "║       第 6 章：卷积神经网络 — CNN 关键术语演示           ║\n"
              << "╚══════════════════════════════════════════════════════════╝\n";

    // 自动设备选择
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "\n  使用设备: " << (device == torch::kCUDA ? "CUDA (GPU)" : "CPU")
              << std::endl;

    /* 依次执行 4 个演示模块 */
    demo_pooling(device);
    demo_stride(device);
    demo_feature_maps(device);
    demo_receptive_field(device);

    /* ------- 总结 ------- */
    std::cout << "════════════════════════════════════════════════════════\n"
              << "  CNN 核心概念总结:\n"
              << "  · 池化 (Pooling): 降维 + 平移不变性 + 控制过拟合\n"
              << "  · 步长 (Stride): 控制输出分辨率与计算量\n"
              << "  · 特征图 (Feature Maps): 多滤波器并行学习不同模式\n"
              << "  · 感受野 (Receptive Field): 深层像素整合更大输入区域\n"
              << "════════════════════════════════════════════════════════\n"
              << std::endl;

    return 0;
}
