/*
 * 02_convolution_libtorch.cpp - 第 6 章：卷积神经网络
 * LibTorch 卷积层基础演示（对应原书第 182-183 页）
 *
 * 演示内容：
 *   1. 使用 torch::nn::Module 自定义卷积网络
 *   2. torch::nn::Conv2d + ReLU 的基本组合
 *   3. padding=0 与 padding=1 对输出尺寸的影响
 *   4. 输出尺寸计算公式
 */

#include <torch/torch.h>
#include <iostream>
#include <iomanip>

/* ========================= 卷积网络模型定义 ========================== */

/* ConvNetImpl: 单层卷积 + ReLU 激活的最简结构 */
struct ConvNetImpl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}; // 二维卷积层
    torch::nn::ReLU relu{nullptr};    // ReLU 激活函数

    /*
     * 构造函数
     * @param in_c   输入通道数 (e.g., 3 for RGB)
     * @param out_c  输出通道数 (卷积核数量)
     * @param ks     卷积核尺寸 (kernel_size)
     */
    ConvNetImpl(int64_t in_c, int64_t out_c, int64_t ks) : conv1(torch::nn::Conv2dOptions(in_c, out_c, ks).padding(0)) // 默认无填充
    {
        // 将子模块注册到 Module 中，使其参数可被 optimizer 管理
        relu = register_module("relu", torch::nn::ReLU());
        conv1 = register_module("conv1", conv1);
    }

    /* 前向传播 */
    torch::Tensor forward(torch::Tensor x) {
        x = conv1(x); // 卷积运算
        x = relu(x);  // ReLU 非线性激活
        return x;
    }
};

// 生成与结构体同名的句柄类型：ConvNet
TORCH_MODULE(ConvNet);

/* ========================= 带填充的卷积网络 ========================== */

/* ConvNetPad: 与 ConvNet 相同结构，但在构造函数中设置 padding=1 */
struct ConvNetPadImpl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::ReLU relu{nullptr};

    ConvNetPadImpl(int64_t in_c, int64_t out_c, int64_t ks) : conv1(torch::nn::Conv2dOptions(in_c, out_c, ks).padding(1)) // 使用 1 像素填充
    {
        relu = register_module("relu", torch::nn::ReLU());
        conv1 = register_module("conv1", conv1);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = conv1(x);
        x = relu(x);
        return x;
    }
};

TORCH_MODULE(ConvNetPad);

/* ================================ main ================================= */

int main() {
    std::cout << std::fixed << std::setprecision(2);

    /* ------ 1. 设备选择 ------ */
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "第 6 章：卷积神经网络 - LibTorch 卷积层演示\n"
              << std::endl;
    std::cout << "使用设备: " << (device == torch::cuda::is_available() ? "CUDA (GPU)" : "CPU")
              << "\n"
              << std::endl;

    /* ------ 2. 创建随机输入张量 ------ */
    // 模拟一批 3 通道 28×28 的彩色图像（batch_size = 4）
    auto input = torch::randn({4, 3, 28, 28},
                              torch::TensorOptions().dtype(torch::kFloat32));
    input = input.to(device);
    std::cout << "输入张量形状: " << input.sizes() << "  [N, C, H, W]" << std::endl;

    /* ------ 3. 无填充卷积 (padding=0) ------ */
    std::cout << "\n/* ===== 示例 1: padding=0 的卷积层 ===== */\n"
              << std::endl;

    ConvNet model_no_pad(3, 16, 3); // 3 输入通道, 16 输出通道, 3×3 卷积核
    model_no_pad->to(device);

    // 前向传播并观察形状变化
    auto output_no_pad = model_no_pad->forward(input);
    std::cout << "  ConvNet(3 → 16, kernel=3, padding=" << 0 << ")\n";
    std::cout << "    输入形状: " << input.sizes() << "\n";
    std::cout << "    输出形状: " << output_no_pad.sizes() << std::endl;

    // 解释：H' = W' = 28 - 3 + 2×0 + 1 = 26
    int H_in = 28, W_in = 28;
    int kernel = 3, pad = 0, stride = 1;
    int H_out = (H_in - kernel + 2 * pad) / stride + 1;
    int W_out = (W_in - kernel + 2 * pad) / stride + 1;
    std::cout << "    手工计算: (" << H_in << " - " << kernel
              << " + 2×" << pad << ") / " << stride << " + 1 = "
              << H_out << "×" << W_out << std::endl;

    /* ------ 4. 带填充卷积 (padding=1) ------ */
    std::cout << "\n/* ===== 示例 2: padding=1 保持空间尺寸不变 ===== */\n"
              << std::endl;

    ConvNetPad model_pad(3, 16, 3); // 同样的 3×3 卷积，但 padding=1
    model_pad->to(device);

    auto output_pad = model_pad->forward(input);
    std::cout << "  ConvNetPad(3 → 16, kernel=3, padding=" << 1 << ")\n";
    std::cout << "    输入形状: " << input.sizes() << "\n";
    std::cout << "    输出形状: " << output_pad.sizes() << std::endl;

    pad = 1;
    H_out = (H_in - kernel + 2 * pad) / stride + 1;
    W_out = (W_in - kernel + 2 * pad) / stride + 1;
    std::cout << "    手工计算: (" << H_in << " - " << kernel
              << " + 2×" << pad << ") / " << stride << " + 1 = "
              << H_out << "×" << W_out << "  ← 尺寸保持不变！" << std::endl;

    /* ------ 5. 维度演变规律总结 ------ */
    std::cout << "\n/* ============ 输出尺寸计算公式 ============ */\n"
              << std::endl;
    std::cout << "  公式: output_size = (input_size - kernel_size + 2×padding) / stride + 1\n"
              << std::endl;
    std::cout << "  常用设置:\n";
    std::cout << "    - kernel=3, padding=0, stride=1 → 每层缩小 2 像素\n";
    std::cout << "    - kernel=3, padding=1, stride=1 → 尺寸不变（\"same\" 卷积）\n";
    std::cout << "    - kernel=3, padding=1, stride=2 → 尺寸减半（池化替代）\n";
    std::cout << "    - MaxPool2d(2)                  → 尺寸减半\n"
              << std::endl;

    return 0;
}
