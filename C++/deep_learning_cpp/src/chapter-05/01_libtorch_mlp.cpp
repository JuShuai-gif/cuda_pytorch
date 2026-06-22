/*
 * 01_libtorch_mlp.cpp - 第 5 章：C++ 与 LibTorch 实战
 * 基于 LibTorch 实现多层感知机解决 XOR 异或问题（对应原书第 139-141 页）
 *
 * 演示内容：
 *   1. 使用 torch::nn::Module 自定义 MLP 模型
 *   2. 自动设备选择（CUDA / CPU）
 *   3. SGD 优化器 + MSE 损失函数训练循环
 *   4. torch::NoGradGuard 推理模式演示
 */

#include <torch/torch.h>
#include <iostream>
#include <iomanip>

/* ============================== MLP 模型定义 ============================== */

/* 三层感知机结构：2 输入 → 4 隐藏 → 3 隐藏 → 1 输出 */
struct MLP : torch::nn::Module {
    torch::nn::Linear layer1{nullptr}, layer2{nullptr}, layer3{nullptr};

    MLP() {
        // 输入层 → 第一隐藏层: 2 → 4
        layer1 = register_module("layer1", torch::nn::Linear(2, 4));
        // 第一隐藏层 → 第二隐藏层: 4 → 3
        layer2 = register_module("layer2", torch::nn::Linear(4, 3));
        // 第二隐藏层 → 输出层: 3 → 1
        layer3 = register_module("layer3", torch::nn::Linear(3, 1));
    }

    /* 前向传播：ReLU 激活（隐藏层）+ 线性输出 */
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(layer1->forward(x)); // 第一隐藏层 + ReLU
        x = torch::relu(layer2->forward(x)); // 第二隐藏层 + ReLU
        x = layer3->forward(x);              // 输出层（无激活，回归任务）
        return x;
    }
};

/* ================================= main =================================== */

int main() {
    /* ------- 1. 设备选择 ------- */
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "使用设备: " << (device == torch::kCUDA ? "CUDA (GPU)" : "CPU")
              << "\n"
              << std::endl;

    /* ------- 2. XOR 异或数据集 ------- */
    // X: 4 个样本，每个 2 维特征
    auto X = torch::tensor({{0.f, 0.f},
                            {0.f, 1.f},
                            {1.f, 0.f},
                            {1.f, 1.f}},
                           torch::TensorOptions().dtype(torch::kFloat32));
    // y: 4 个目标值（异或真值表）
    auto y = torch::tensor({{0.f},
                            {1.f},
                            {1.f},
                            {0.f}},
                           torch::TensorOptions().dtype(torch::kFloat32));

    // 将数据移动到目标设备
    X = X.to(device);
    y = y.to(device);

    /* ------- 3. 创建模型并移至设备 ------- */
    MLP model;
    model.to(device);

    /* ------- 4. 定义优化器（SGD，学习率 0.1） ------- */
    torch::optim::SGD optimizer(
        model.parameters(),
        torch::optim::SGDOptions(0.1));

    /* ------- 5. 训练循环 ------- */
    const int epochs = 1000;
    const int report_interval = 100; // 每 100 轮打印一次损失

    std::cout << "开始训练 XOR 问题（" << epochs << " 轮）...\n"
              << std::endl;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        optimizer.zero_grad();                  // 梯度清零
        auto output = model.forward(X);         // 前向传播
        auto loss = torch::mse_loss(output, y); // 计算均方误差损失
        loss.backward();                        // 反向传播
        optimizer.step();                       // 更新参数

        // 每 report_interval 轮打印损失
        if (epoch % report_interval == 0 || epoch == 1) {
            std::cout << "  epoch " << std::setw(5) << epoch
                      << "  |  MSE = " << std::fixed
                      << std::setprecision(6) << loss.item<float>()
                      << std::endl;
        }
    }

    /* ------- 6. 训练结果评估（推理模式） ------- */
    std::cout << "\n训练完成，测试结果:\n"
              << std::endl;

    {
        // NoGradGuard 禁止梯度计算，减少内存占用并加速推理
        torch::NoGradGuard no_grad;

        auto predictions = model.forward(X); // 对全部样本推理

        // 打印每个样本的预测值与目标值对比
        auto pred_acc = predictions.accessor<float, 2>();
        auto y_acc = y.accessor<float, 2>();

        for (int i = 0; i < 4; ++i) {
            float x1 = (i >> 1) & 1; // 高位为 x₁
            float x2 = i & 1;        // 低位为 x₂
            std::cout << "  (" << x1 << ", " << x2 << ")"
                      << "  →  预测 " << pred_acc[i][0]
                      << "  (目标 " << y_acc[i][0] << ")" << std::endl;
        }
    }

    std::cout << std::endl;
    return 0;
}
