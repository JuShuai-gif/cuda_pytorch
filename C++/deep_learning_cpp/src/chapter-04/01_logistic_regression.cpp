/*
 * 01_logistic_regression.cpp
 * 第 4 章：构建基础神经网络
 *
 * 从零实现二元逻辑回归，不使用任何外部库。
 * 使用两个高斯簇构造合成 2D 数据集进行二分类。
 *
 * 涵盖的概念（来自 PDF "构建基础神经网络" 一节）：
 *   - Sigmoid 激活函数将线性输出映射到 [0,1] 概率
 *   - 二值交叉熵（BCE）损失作为分类目标
 *   - 批量梯度下降：对整个数据集取平均梯度
 *   - 0.5 阈值决策边界
 *   - 准确率评估
 */

#include <iostream>
#include <array>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <iomanip>

// ----------------------------------------------------------------
// Sigmoid 激活函数：将任意实数映射到 (0, 1) 区间
// ----------------------------------------------------------------
static inline double sigmoid(double z) {
    return 1.0 / (1.0 + std::exp(-z));
}

// ----------------------------------------------------------------
// 二值交叉熵损失：BCE = -[y*log(p) + (1-y)*log(1-p)]
// 为防止 log(0)，对 p 做数值截断处理
// ----------------------------------------------------------------
static inline double bce_loss(double p, int y) {
    const double eps = 1e-12;
    p = std::clamp(p, eps, 1.0 - eps);
    return -(y * std::log(p) + (1 - y) * std::log(1.0 - p));
}

int main() {
    std::cout << "=== 逻辑回归 from Scratch ===\n\n";

    // --- 生成合成 2D 数据集：两个高斯簇 ---
    const int n_per_class = 200;
    const int n = 2 * n_per_class;

    // 类别 0：均值 (-2, -2)，标准差 1.0
    // 类别 1：均值 ( 2,  2)，标准差 1.0
    const std::array<double, 2> mu0 = {-2.0, -2.0};
    const std::array<double, 2> mu1 = {2.0, 2.0};
    const double sigma = 1.0;

    std::mt19937 rng(123);
    std::normal_distribution<double> gauss(0.0, sigma);

    std::vector<double> x1(n), x2(n);
    std::vector<int> labels(n);

    for (int i = 0; i < n_per_class; ++i) {
        // 类别 0
        x1[i] = mu0[0] + gauss(rng);
        x2[i] = mu0[1] + gauss(rng);
        labels[i] = 0;

        // 类别 1
        int idx = n_per_class + i;
        x1[idx] = mu1[0] + gauss(rng);
        x2[idx] = mu1[1] + gauss(rng);
        labels[idx] = 1;
    }
    std::cout << "已生成 " << n << " 个数据点（每类 " << n_per_class << " 个）。\n";
    std::cout << "  类别 0：N(μ=(" << mu0[0] << "," << mu0[1]
              << "), σ=" << sigma << ")\n";
    std::cout << "  类别 1：N(μ=(" << mu1[0] << "," << mu1[1]
              << "), σ=" << sigma << ")\n\n";

    // --- 初始化参数 ---
    double w1 = 0.0;
    double w2 = 0.0;
    double b = 0.0;
    const double lr = 0.1;
    const int epochs = 3000;

    std::cout << "初始参数：w1 = " << w1 << " , w2 = " << w2 << " , b = " << b << "\n";
    std::cout << "学习率 = " << lr << " , 训练轮数 = " << epochs << "\n";
    std::cout << "开始训练...\n\n";

    // --- 训练循环（批量梯度下降） ---
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double grad_w1 = 0.0;
        double grad_w2 = 0.0;
        double grad_b = 0.0;
        double loss_sum = 0.0;

        for (int i = 0; i < n; ++i) {
            // 前向传播
            double z = w1 * x1[i] + w2 * x2[i] + b;
            double p = sigmoid(z);

            // 累积 BCE 损失
            loss_sum += bce_loss(p, labels[i]);

            // 累积梯度
            // ∂BCE/∂w = (p - y) * x , ∂BCE/∂b = (p - y)
            double diff = p - labels[i];
            grad_w1 += diff * x1[i];
            grad_w2 += diff * x2[i];
            grad_b += diff;
        }

        // 对整个数据集取平均
        grad_w1 /= n;
        grad_w2 /= n;
        grad_b /= n;
        double avg_loss = loss_sum / n;

        // 梯度下降更新
        w1 -= lr * grad_w1;
        w2 -= lr * grad_w2;
        b -= lr * grad_b;

        // 每 500 个 epoch 打印损失
        if ((epoch + 1) % 500 == 0) {
            std::cout << "Epoch " << std::setw(4) << (epoch + 1)
                      << " | BCE Loss = " << std::fixed << std::setprecision(6) << avg_loss
                      << " | w1 = " << std::setprecision(4) << w1
                      << " , w2 = " << w2
                      << " , b = " << b << "\n";
        }
    }

    // --- 训练后评估准确率 ---
    int correct = 0;
    for (int i = 0; i < n; ++i) {
        double z = w1 * x1[i] + w2 * x2[i] + b;
        double p = sigmoid(z);
        int pred = (p >= 0.5) ? 1 : 0;
        if (pred == labels[i]) {
            ++correct;
        }
    }
    double accuracy = 100.0 * correct / n;

    std::cout << "\n========== 训练完成 ==========\n";
    std::cout << "学习参数：w1 = " << std::fixed << std::setprecision(4) << w1
              << " , w2 = " << w2
              << " , b = " << b << "\n";
    std::cout << "准确率：" << std::setprecision(2) << accuracy
              << "% (" << correct << "/" << n << ")\n";

    // 决策边界：w1*x1 + w2*x2 + b = 0  ->  x2 = -(w1/w2)*x1 - b/w2
    std::cout << "\n决策边界公式：\n";
    std::cout << "  " << w1 << "*x1 + " << w2 << "*x2 + " << b << " = 0\n";
    if (std::abs(w2) > 1e-12) {
        std::cout << "  即 x2 = " << (-w1 / w2) << "*x1 + " << (-b / w2) << "\n";
    }

    return 0;
}
