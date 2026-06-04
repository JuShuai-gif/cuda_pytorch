/*
 * Dimensionality_reduction.cpp
 * 第 2 章：C++ 中的数据准备与预处理
 *
 * 高维特征可能导致"维度灾难"，增加计算成本和
 * 过拟合风险。降维将数据投影到低维空间，
 * 同时保留重要结构。
 *
 * 涵盖的技术（来自 PDF"降维"一节）：
 *   - PCA（主成分分析）：线性投影到最大方差方向。
 *     需要 Eigen 3.4+ 进行矩阵运算。
 *     快速、可解释；适合移除相关/冗余特征。
 *   - t-SNE（t 分布随机邻域嵌入）：用于可视化的非线性技术
 *     （通常 2D/3D）。保留局部邻域结构。
 *     需要外部库（mlpack/TSNE-Cpp）。计算开销高。
 *   - 自编码器（AE）：神经网络，学习压缩表示（编码器）
 *     并重建输入（解码器）。PCA 的非线性替代方案。
 *     此处使用 LibTorch 实现。
 *
 * 注意：对 t-SNE，推荐使用 mlpack。本文件展示使用模式；
 * 安装 mlpack 并取消注释 t-SNE 部分即可运行。
 */

#include <iostream>
#include <vector>
#include <iomanip>

// --- PCA with Eigen ---
#include <Eigen/Dense>

// 执行 PCA，将数据从 d 维降至 numComponents 维。
// 步骤：1) 中心化数据，2) 计算协方差，3) 特征分解，
//       4) 投影到顶部特征向量。
Eigen::MatrixXd performPCA(const Eigen::MatrixXd &data, int numComponents) {
    // 中心化：减去列均值
    Eigen::RowVectorXd mean = data.colwise().mean();
    Eigen::MatrixXd centered = data.rowwise() - mean;

    // 协方差矩阵：(X^T * X) / (n-1)
    Eigen::MatrixXd cov = (centered.adjoint() * centered) / double(data.rows() - 1);

    // 对称协方差矩阵的特征分解
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(cov);
    // 特征向量按特征值升序排列；取最后 numComponents 个
    Eigen::MatrixXd eigenvectors =
        solver.eigenvectors().rightCols(numComponents);

    // 将中心化数据投影到选定的特征向量上
    return centered * eigenvectors;
}

// --- 使用 LibTorch 的自编码器 ---
// 非线性降维：编码器压缩 d -> h，解码器重建 h -> d。
// 训练最小化重建误差。编码器输出用作降维后的表示。

#include <torch/torch.h>

struct AE : torch::nn::Module {
    torch::nn::Linear enc{nullptr}, dec{nullptr};

    AE(int d, int h) {
        enc = register_module("enc", torch::nn::Linear(d, h));
        dec = register_module("dec", torch::nn::Linear(h, d));
    }

    torch::Tensor forward(torch::Tensor x) {
        return dec(torch::relu(enc(x)));
    }

    // 提取压缩（隐藏层）表示
    torch::Tensor encode(torch::Tensor x) {
        return torch::relu(enc(x));
    }
};

// 训练自编码器并返回降维后的表示
torch::Tensor autoencoderReduce(torch::Tensor data, int hiddenDim,
                                int epochs = 200) {
    int inputDim = data.size(1);
    AE model(inputDim, hiddenDim);

    // 将模型参数移到与数据相同的设备和数据类型
    model.to(data.device(), data.scalar_type());

    auto optimizer = torch::optim::Adam(
        model.parameters(), torch::optim::AdamOptions(0.01));

    for (int epoch = 0; epoch < epochs; ++epoch) {
        optimizer.zero_grad();
        auto output = model.forward(data);
        auto loss = torch::mse_loss(output, data);
        loss.backward();
        optimizer.step();

        if (epoch % 50 == 0 || epoch == epochs - 1) {
            std::cout << "  AE epoch " << epoch
                      << " loss: " << loss.item<float>() << "\n";
        }
    }

    // 返回压缩表示
    return model.encode(data);
}

// --- 辅助函数 ---
void printMatrix(const std::string &label, const Eigen::MatrixXd &mat,
                 int maxRows = 6) {
    std::cout << label << ":\n";
    int rows = std::min((int)mat.rows(), maxRows);
    for (int i = 0; i < rows; ++i) {
        std::cout << "  [";
        for (int j = 0; j < mat.cols(); ++j) {
            std::cout << std::fixed << std::setprecision(3) << mat(i, j);
            if (j + 1 < mat.cols()) std::cout << ", ";
        }
        std::cout << "]\n";
    }
    if (mat.rows() > maxRows)
        std::cout << "  ... (" << mat.rows() - maxRows << " 行更多)\n";
}

int main() {
    std::cout << "=== Dimensionality Reduction Demos ===\n\n";

    // ===========================================
    // 1. PCA 演示（使用 Eigen）
    // ===========================================
    std::cout << "[PCA] 线性投影到最大方差方向。\n";
    std::cout << "  用例：移除相关特征，加速训练，可视化。\n";

    // 创建 10 个样本 x 5 个特征
    Eigen::MatrixXd data(10, 5);
    data << 2.5, 2.4, 0.5, 0.7, 1.0,
        0.5, 0.7, 1.2, 0.3, 0.8,
        2.2, 2.9, 0.3, 0.9, 1.1,
        1.9, 2.2, 0.8, 0.6, 0.9,
        3.1, 3.0, 0.2, 0.8, 1.2,
        2.3, 2.7, 0.4, 0.5, 1.0,
        2.0, 1.6, 1.0, 0.4, 0.7,
        1.0, 1.1, 1.5, 0.2, 0.5,
        1.5, 1.6, 1.1, 0.5, 0.8,
        1.1, 0.9, 1.4, 0.3, 0.6;

    printMatrix("原始数据 (10x5)", data);

    // 降至 2 个成分
    Eigen::MatrixXd reduced = performPCA(data, 2);
    printMatrix("PCA 降维后 (10x2)", reduced);
    std::cout << std::endl;

    // ===========================================
    // 2. 自编码器演示（使用 LibTorch）
    // ===========================================
    std::cout << "[自编码器] 基于 NN 的非线性压缩。\n";
    std::cout << "  用例：学习复杂的非线性流形。\n";

    // 将 Eigen 矩阵转换为 LibTorch 张量
    torch::Tensor t = torch::from_blob(
                          const_cast<double *>(data.data()),
                          {data.rows(), data.cols()}, torch::kFloat64)
                          .clone();

    // 将 5 个输入维度降至 2 个隐藏维度
    torch::Tensor encoded = autoencoderReduce(t, 2, 200);

    std::cout << "\n自编码器编码后 (10x2):\n";
    auto encoded_acc = encoded.accessor<double, 2>();
    for (int i = 0; i < std::min((int)encoded.size(0), 6); ++i) {
        std::cout << "  [";
        for (int j = 0; j < encoded.size(1); ++j) {
            std::cout << std::fixed << std::setprecision(3) << encoded_acc[i][j];
            if (j + 1 < encoded.size(1)) std::cout << ", ";
        }
        std::cout << "]\n";
    }
    std::cout << "  ...\n\n";

    // ===========================================
    // 3. t-SNE（占位）
    // ===========================================
    std::cout << "[t-SNE] 非线性可视化技术 (2D/3D)。\n";
    std::cout << "  保留局部邻域结构。\n";
    std::cout << "  用例：可视化高维聚类。\n";
    std::cout << "  注意：需要外部库（mlpack 或 TSNE-Cpp）。\n";
    std::cout << "  安装 mlpack 并参考 mlpack t-SNE API。\n";

    return 0;
}
