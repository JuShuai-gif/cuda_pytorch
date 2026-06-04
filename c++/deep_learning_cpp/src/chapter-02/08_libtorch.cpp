/*
 * libtorch.cpp
 * 第2章：C++中的数据准备与预处理
 *
 * 自编码器（AE）是一种无监督神经网络，通过重构输入来学习
 * 压缩表示。它由以下部分组成：
 *   - 编码器：将输入 d 压缩 -> 隐藏层 h（降维）
 *   - 解码器：将隐藏层 h 重构 -> 输出 d
 *
 * 自编码器作为 PCA 的非线性降维替代方案。
 * 与 PCA（线性投影）不同，自编码器可以学习复杂的非线性流形，
 * 使其适用于主成分不足以表达的数据。
 *
 * 变体：
 *   - 去噪自编码器：输入被损坏；模型学习重构干净数据
 *   - 稀疏自编码器：在隐藏层激活上添加稀疏性约束
 *   - 变分自编码器（VAE）：概率潜在空间，能够生成数据
 *
 * 用例：特征提取、异常检测（高重构误差 = 异常）、
 * 数据压缩、深度网络的预训练。
 *
 * 何时选择自编码器而非 PCA：
 *   - PCA：快速、确定性、可解释的分量。适用于线性相关性、
 *     小数据集。
 *   - AE：学习非线性流形，更适合复杂数据（图像、文本）。
 *     需要更多数据和调参。
 */

#include <torch/torch.h>
#include <iostream>
#include <iomanip>

// ----------------------------------------------------------------
// 简单自编码器：编码器 (d -> h) + 解码器 (h -> d)
// 使用 ReLU 激活以实现非线性。
// ----------------------------------------------------------------
struct Autoencoder : torch::nn::Module {
    torch::nn::Linear encoder{nullptr}, decoder{nullptr};

    Autoencoder(int inputDim, int hiddenDim) {
        encoder = register_module("encoder",
                                  torch::nn::Linear(inputDim, hiddenDim));
        decoder = register_module("decoder",
                                  torch::nn::Linear(hiddenDim, inputDim));
    }

    // 完整前向传播：编码 -> 解码
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(encoder->forward(x));
        return decoder->forward(x);
    }

    // 提取压缩表示（编码器输出）
    torch::Tensor encode(torch::Tensor x) {
        return torch::relu(encoder->forward(x));
    }
};

// ----------------------------------------------------------------
// 训练自编码器并返回压缩表示。
// 损失函数：输入与重构之间的均方误差。
// 优化器：Adam（自适应学习率，良好的默认选择）。
// ----------------------------------------------------------------
torch::Tensor autoencode(torch::Tensor data, int hiddenDim,
                         int epochs = 100, double lr = 0.01) {
    int inputDim = data.size(1);
    Autoencoder model(inputDim, hiddenDim);

    // 将模型移动到与输入匹配的设备和数据类型
    model.to(data.device(), data.scalar_type());

    auto optimizer = torch::optim::Adam(
        model.parameters(),
        torch::optim::AdamOptions(lr));

    for (int epoch = 0; epoch < epochs; ++epoch) {
        optimizer.zero_grad();
        auto output = model.forward(data);
        auto loss = torch::mse_loss(output, data);
        loss.backward();
        optimizer.step();

        if (epoch % 25 == 0 || epoch == epochs - 1) {
            std::cout << "  Epoch " << std::setw(3) << epoch
                      << " | loss: " << std::fixed
                      << std::setprecision(6) << loss.item<float>()
                      << "\n";
        }
    }

    // 返回编码（压缩）后的表示
    return model.encode(data);
}

int main() {
    std::cout << "=== Autoencoder Dimensionality Reduction (LibTorch) ===\n\n";

    // 创建合成数据：50个样本 x 10个特征
    // 模拟具有结构的数据（某些特征是相关的）
    int numSamples = 50;
    int inputDim = 10;
    int hiddenDim = 3; // 从 10维 压缩到 3维

    torch::manual_seed(42);
    auto X = torch::randn({numSamples, inputDim});

    std::cout << "Data: " << numSamples << " samples x "
              << inputDim << " features\n";
    std::cout << "Hidden dim: " << hiddenDim
              << " (compression ratio: " << inputDim
              << " -> " << hiddenDim << ")\n\n";

    std::cout << "Training autoencoder...\n";
    auto encoded = autoencode(X, hiddenDim, 100, 0.01);

    // 显示压缩表示
    std::cout << "\nCompressed representation (first 8 samples):\n";
    auto enc_acc = encoded.accessor<float, 2>();
    for (int i = 0; i < std::min(8, numSamples); ++i) {
        std::cout << "  sample " << i << ": [";
        for (int j = 0; j < hiddenDim; ++j) {
            std::cout << std::fixed << std::setprecision(4) << enc_acc[i][j];
            if (j + 1 < hiddenDim) std::cout << ", ";
        }
        std::cout << "]\n";
    }
    std::cout << "  ...\n\n";

    // 与 PCA 的比较（概念层面）
    std::cout << "Autoencoder vs PCA:\n";
    std::cout << "  PCA:     Linear projection, fast, deterministic, interpretable\n";
    std::cout << "  AE:      Nonlinear manifold, learns complex patterns\n";
    std::cout << "  When to use AE: Data has nonlinear structure, images,\n";
    std::cout << "                  anomaly detection (reconstruction error)\n";

    return 0;
}
