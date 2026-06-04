/*
 * 03_vgg16_mnist.cpp - 第 6 章：卷积神经网络
 * 简化 VGG 风格网络用于 MNIST 手写数字识别（对应原书第 183-186 页）
 *
 * 演示内容：
 *   1. 使用 Sequential 构建特征提取器（卷积块堆叠）
 *   2. VGG 风格：小卷积核(3×3) + ReLU + MaxPool2d 的重复模块
 *   3. features → flatten → classifier 的三段式前向传播
 *   4. 统计并打印模型可训练参数总数
 *
 *   注：本文件仅为架构演示，完整 MNIST 训练需搭配 DataLoader 使用
 */

#include <torch/torch.h>
#include <iostream>
#include <iomanip>

/* ======================= VGG 风格网络定义 ======================== */

/*
 * VGGStyle: 面向 MNIST 的简化 VGG 网络
 *
 * 特征提取通道数变化: 1(灰度图) → 8 → 16 → 32
 * 空间尺寸变化:    28×28 → 14×14 → 7×7 → 3×3
 * 分类器:          288(32×3×3) → 64 → 10
 */
struct VGGStyle : torch::nn::Module {
    // 特征提取部分（卷积 + 池化）
    torch::nn::Sequential features{nullptr};

    // 分类器部分（全连接 + Dropout）
    torch::nn::Sequential classifier{nullptr};

    /*
     * 构造函数
     * MNIST 为单通道 28×28 灰度图，因此输入通道固定为 1
     */
    VGGStyle() {
        // 构建特征提取器 —— 三个 VGG 风格卷积块
        // Block 1: 1×28×28 → 8×28×28(conv) → 8×14×14(pool)
        // Block 2: 8×14×14 → 16×14×14(conv) → 16×7×7(pool)
        // Block 3: 16×7×7  → 32×7×7(conv)   → 32×3×3(pool)
        features = register_module("features", torch::nn::Sequential(
                                                   /* ===== 卷积块 1 ===== */
                                                   // Conv2d: 1 通道 → 8 通道, 3×3 卷积, padding=1 (尺寸保持不变)
                                                   torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 8, 3).padding(1)),
                                                   torch::nn::ReLU(),                                    // ReLU 激活
                                                   torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)), // 28×28 → 14×14

                                                   /* ===== 卷积块 2 ===== */
                                                   torch::nn::Conv2d(torch::nn::Conv2dOptions(8, 16, 3).padding(1)),
                                                   torch::nn::ReLU(),
                                                   torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)), // 14×14 → 7×7

                                                   /* ===== 卷积块 3 ===== */
                                                   torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 3).padding(1)),
                                                   torch::nn::ReLU(),
                                                   torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)) // 7×7 → 3×3
                                                   ));

        // 构建分类器 —— 两个全连接层
        // 展平后大小: 32 通道 × 3 × 3 = 288
        classifier = register_module("classifier", torch::nn::Sequential(
                                                       torch::nn::Linear(32 * 3 * 3, 64), // 288 → 64
                                                       torch::nn::ReLU(),                 // 隐藏层激活
                                                       torch::nn::Dropout(0.5),           // 50% Dropout 防过拟合
                                                       torch::nn::Linear(64, 10)          // 64 → 10 (数字 0-9)
                                                       ));
    }

    /*
     * 前向传播：三段式流水线
     *   1. features:   卷积特征提取
     *   2. flatten:    展平为 1D 向量
     *   3. classifier: 全连接分类
     */
    torch::Tensor forward(torch::Tensor x) {
        x = features->forward(x);    // 卷积特征提取
        x = x.view({x.size(0), -1}); // 展平: [N, C, H, W] → [N, C×H×W]
        x = classifier->forward(x);  // 全连接分类
        return x;
    }
};

/* ================================ main ================================= */

int main() {
    std::cout << std::fixed << std::setprecision(2);

    /* ------ 1. 设备选择 ------ */
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "第 6 章：卷积神经网络 - VGG 风格 MNIST 网络演示\n"
              << std::endl;
    std::cout << "使用设备: " << (device == torch::cuda::is_available() ? "CUDA (GPU)" : "CPU")
              << "\n"
              << std::endl;

    /* ------ 2. 创建模型并移至设备 ------ */
    VGGStyle model;
    model->to(device);

    /* ------ 3. 打印模型架构总览 ------ */
    std::cout << "/* ============ VGG 风格 MNIST 网络架构 ============ */\n"
              << std::endl;
    std::cout << model << std::endl;

    /* ------ 4. 前向传播演示（随机 MNIST 风格输入） ------ */
    // 模拟一批 8 张单通道 28×28 的灰度手写数字图像
    auto input = torch::randn({8, 1, 28, 28},
                              torch::TensorOptions().dtype(torch::kFloat32));
    input = input.to(device);

    std::cout << "\n/* ============ 前向传播形状变化 ============ */\n"
              << std::endl;
    std::cout << "  输入形状 (N, C, H, W):  " << input.sizes() << std::endl;

    // 观察经过特征提取后的形状
    {
        torch::NoGradGuard no_grad; // 演示用，无需梯度
        auto feat = model->features->forward(input);
        std::cout << "  特征提取后 (features):   " << feat.sizes()
                  << "  [32 通道 × 3 × 3]" << std::endl;

        // 观察展平后的形状
        auto flat = feat.view({feat.size(0), -1});
        std::cout << "  展平后 (flatten):        " << flat.sizes()
                  << "  [" << feat.size(1) * feat.size(2) * feat.size(3)
                  << " 维向量]" << std::endl;

        // 最终分类输出
        auto out = model->classifier->forward(flat);
        std::cout << "  分类器输出 (classifier):  " << out.sizes()
                  << "  [10 类 logits]" << std::endl;
    }

    // 完整的端到端前向传播
    std::cout << "\n  端到端前向:  " << input.sizes() << " → "
              << model->forward(input).sizes() << std::endl;

    /* ------ 5. 模型尺寸逐阶段演变 ------ */
    std::cout << "\n/* ============ 空间尺寸演变 ============ */\n"
              << std::endl;
    std::cout << "  [1×28×28] 输入原始灰度图像\n";
    std::cout << "      ↓  Conv2d(1→8, k3p1) + ReLU\n";
    std::cout << "  [8×28×28]\n";
    std::cout << "      ↓  MaxPool2d(2)\n";
    std::cout << "  [8×14×14]  尺寸减半\n";
    std::cout << "      ↓  Conv2d(8→16, k3p1) + ReLU\n";
    std::cout << "  [16×14×14]\n";
    std::cout << "      ↓  MaxPool2d(2)\n";
    std::cout << "  [16×7×7]   尺寸再减半\n";
    std::cout << "      ↓  Conv2d(16→32, k3p1) + ReLU\n";
    std::cout << "  [32×7×7]\n";
    std::cout << "      ↓  MaxPool2d(2)\n";
    std::cout << "  [32×3×3]   第三次减半（下取整）\n";
    std::cout << "      ↓  Flatten\n";
    std::cout << "  [288]\n";
    std::cout << "      ↓  Linear(288→64) + ReLU + Dropout(0.5)\n";
    std::cout << "  [64]\n";
    std::cout << "      ↓  Linear(64→10)\n";
    std::cout << "  [10]       输出 logits (数字 0-9)\n"
              << std::endl;

    /* ------ 6. 统计可训练参数 ------ */
    int64_t total_params = 0;
    for (const auto &param : model->parameters()) {
        total_params += param.numel(); // numel() 返回张量中元素的总数
    }

    std::cout << "/* ============ 可训练参数统计 ============ */\n"
              << std::endl;
    std::cout << "  可训练参数总数: " << total_params << std::endl;

    // 逐层参数明细
    int layer_idx = 0;
    for (const auto &pair : model->named_parameters()) {
        std::cout << "    " << std::setw(30) << std::left << pair.key()
                  << "  →  " << std::setw(8) << pair.value().sizes()
                  << "  (" << pair.value().numel() << " 个参数)" << std::endl;
        layer_idx++;
    }

    std::cout << "\n  注：本文件仅为 VGG 风格架构演示，完整 MNIST 训练需搭配 DataLoader\n"
              << std::endl;

    return 0;
}
