/*
 * 02_neuron_demo.cpp - 第 4 章：构建基础神经网络
 * LibTorch 单神经元 + 激活函数演示（对应原书第 116-118 页）
 *
 * 演示内容：
 *   1. 自动设备选择（CUDA / CPU）
 *   2. 随机输入批次的创建
 *   3. Linear 全连接层的定义与 Kaiming 初始化
 *   4. 前向传播：Linear → ReLU / Sigmoid / Tanh
 *   5. 输出形状与数值检查
 */

#include <torch/torch.h>
#include <iostream>

int main() {
    /* =========================== 设备选择 =========================== */
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    // 打印当前使用的计算设备
    std::cout << "使用设备: " << device << std::endl;

    /* =========================== 构造输入张量 ======================= */
    int64_t batch_size = 4;
    int64_t in_features = 5;
    // 创建服从标准正态分布的随机输入批次 [4, 5]
    auto x = torch::randn({batch_size, in_features}).to(device);
    std::cout << "输入形状: " << x.sizes() << std::endl;

    /* ========================== 定义全连接层 ========================= */
    int64_t out_features = 3;
    auto fc = torch::nn::Linear(in_features, out_features);
    // 使用 Kaiming 均匀分布初始化权重，偏置置零
    torch::nn::init::kaiming_uniform_(fc->weight);
    torch::nn::init::zeros_(fc->bias);
    // 将层移动到目标设备
    fc->to(device);

    /* =================== 前向传播 + ReLU 激活 ======================== */
    auto z = fc->forward(x); // 线性变换: z = Wx + b
    auto y = torch::relu(z); // ReLU 激活: y = max(0, z)

    /* ======================= 输出形状与采样 =========================== */
    std::cout << "logits 形状: " << z.sizes() << std::endl;
    std::cout << "ReLU 输出形状: " << y.sizes() << std::endl;
    // 打印前两行输出以作检查
    std::cout << "ReLU 输出（前 2 行）:\n"
              << y.slice(/*dim=*/0, /*start=*/0, /*end=*/2) << std::endl;

    /* ==================== 对比其他激活函数 =========================== */
    auto y_sigmoid = torch::sigmoid(z); // Sigmoid: σ(z) = 1/(1+e^{-z})
    auto y_tanh = torch::tanh(z);       // Tanh:    tanh(z)

    std::cout << "Sigmoid 输出（前 2 行）:\n"
              << y_sigmoid.slice(0, 0, 2) << std::endl;
    std::cout << "Tanh 输出（前 2 行）:\n"
              << y_tanh.slice(0, 0, 2) << std::endl;

    return 0;
}
