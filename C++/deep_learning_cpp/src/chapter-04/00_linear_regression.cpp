/*
 * 00_linear_regression.cpp
 * 第 4 章：构建基础神经网络
 *
 * 从零实现线性回归，不使用任何外部库。
 * 使用合成数据 y = 2.5*x + 0.7 + noise 来验证学到的参数。
 *
 * 涵盖的概念（来自 PDF "构建基础神经网络" 一节）：
 *   - 通过 SGD 进行参数学习：逐个数据点更新 w 和 b
 *   - MSE 损失作为优化目标
 *   - 训练过程中损失曲线的监控
 *   - 学到的参数与真实值的比较
 */

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>

int main() {
    std::cout << "=== 线性回归 from Scratch ===\n\n";

    // --- 生成合成数据：y = 2.5*x + 0.7 + noise ---
    const int n = 200;
    const double true_w = 2.5;
    const double true_b = 0.7;
    const double noise_std = 0.5;

    std::mt19937 rng(42);
    std::normal_distribution<double> noise_dist(0.0, noise_std);
    std::uniform_real_distribution<double> x_dist(-3.0, 3.0);

    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = x_dist(rng);
        y[i] = true_w * x[i] + true_b + noise_dist(rng);
    }
    std::cout << "已生成 " << n << " 个合成数据点。" << std::endl;
    std::cout << "  真实参数：w = " << true_w << " , b = " << true_b << "\n";
    std::cout << "  噪声标准差 = " << noise_std << "\n\n";

    // --- 初始化参数 ---
    double w = 0.0;
    double b = 0.0;
    const double lr = 0.01; // SGD 需要比批量梯度下降更小的学习率
    const int epochs = 1000;

    std::cout << "初始参数：w = " << w << " , b = " << b << "\n";
    std::cout << "学习率 = " << lr << " , 训练轮数 = " << epochs << "\n";
    std::cout << "开始训练...\n\n";

    // --- 训练循环（随机梯度下降，逐点更新） ---
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // 随机打乱数据顺序，避免周期性模式
        for (int i = 0; i < n; ++i) {
            double y_hat = w * x[i] + b; // 前向传播：预测值
            double error = y_hat - y[i]; // 残差

            // 梯度计算（MSE 损失对每个数据点的梯度）
            // ∂MSE/∂w = 2*(ŷ-y)*x , ∂MSE/∂b = 2*(ŷ-y)
            double grad_w = 2.0 * error * x[i];
            double grad_b = 2.0 * error;

            // SGD 更新
            w -= lr * grad_w;
            b -= lr * grad_b;
        }

        // 每 200 个 epoch 计算并打印整体 MSE 损失
        if ((epoch + 1) % 200 == 0) {
            double total_loss = 0.0;
            for (int i = 0; i < n; ++i) {
                double y_hat = w * x[i] + b;
                double diff = y_hat - y[i];
                total_loss += diff * diff;
            }
            double mse = total_loss / n;
            std::cout << "Epoch " << std::setw(4) << (epoch + 1)
                      << " | MSE = " << std::fixed << std::setprecision(6) << mse
                      << " | w = " << std::setprecision(4) << w
                      << " , b = " << b << "\n";
        }
    }

    // --- 打印最终结果 ---
    std::cout << "\n========== 训练完成 ==========\n";
    std::cout << "真实参数： w = " << true_w << " , b = " << true_b << "\n";
    std::cout << "学习参数： w = " << std::fixed << std::setprecision(4) << w
              << " , b = " << b << "\n";
    std::cout << "w 误差：" << std::abs(w - true_w)
              << " , b 误差：" << std::abs(b - true_b) << "\n";

    return 0;
}
