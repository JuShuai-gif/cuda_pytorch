/*
 * 03_mlp_eigen.cpp - 第 4 章：构建基础神经网络
 * 基于 Eigen 从零实现多层感知机（对应原书第 118-121 页）
 *
 * 演示内容：
 *   1. NeuralNetwork 类的完整实现（含前向传播与反向传播）
 *   2. Sigmoid 激活函数及其导数
 *   3. 随机梯度下降（SGD）训练循环
 *   4. XOR 异或问题作为玩具数据集进行验证
 */

#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <vector>
#include <random>

/* ============================ NeuralNetwork 类 ======================= */
class NeuralNetwork {
private:
    // 权重矩阵: hidden × input, output × hidden
    Eigen::MatrixXd weights_input_hidden;
    Eigen::MatrixXd weights_hidden_output;
    // 偏置向量
    Eigen::VectorXd bias_hidden;
    Eigen::VectorXd bias_output;
    // 学习率
    double learning_rate;

    // 前向传播中间值（反向传播时复用）
    Eigen::VectorXd hidden_z; // 隐藏层线性输出
    Eigen::VectorXd hidden_a; // 隐藏层激活输出
    Eigen::VectorXd output_z; // 输出层线性输出

    /* ------- Sigmoid 激活函数 ------- */
    Eigen::VectorXd sigmoid(const Eigen::VectorXd &z) const {
        // σ(z) = 1 / (1 + e^{-z})
        return 1.0 / (1.0 + (-z.array()).exp());
    }

    /* ------- Sigmoid 导数（已知激活值 a = σ(z)） ------- */
    Eigen::VectorXd sigmoid_derivative(const Eigen::VectorXd &a) const {
        // σ'(z) = σ(z) * (1 - σ(z)) = a * (1 - a)
        return a.array() * (1.0 - a.array());
    }

public:
    /* ------- 构造函数：随机初始化权重与偏置 ------- */
    NeuralNetwork(int input_size, int hidden_size, int output_size, double lr) : learning_rate(lr) {
        // 使用 [-1, 1] 均匀分布随机初始化
        weights_input_hidden = Eigen::MatrixXd::Random(hidden_size, input_size);
        weights_hidden_output = Eigen::MatrixXd::Random(output_size, hidden_size);
        bias_hidden = Eigen::VectorXd::Random(hidden_size);
        bias_output = Eigen::VectorXd::Random(output_size);
    }

    /* ------- 前向传播 ------- */
    Eigen::VectorXd feedforward(const Eigen::VectorXd &input) {
        // 隐藏层: z₁ = W_ih · x + b_h  →  a₁ = σ(z₁)
        hidden_z = weights_input_hidden * input + bias_hidden;
        hidden_a = sigmoid(hidden_z);

        // 输出层: z₂ = W_ho · a₁ + b_o  →  ŷ = σ(z₂)
        output_z = weights_hidden_output * hidden_a + bias_output;
        return sigmoid(output_z);
    }

    /* ------- 训练（单样本 SGD 更新） ------- */
    void train(const Eigen::VectorXd &input, const Eigen::VectorXd &target) {
        // 1. 前向传播
        Eigen::VectorXd output = feedforward(input);

        // 2. 输出层误差与 delta
        Eigen::VectorXd output_error = target - output;
        Eigen::VectorXd output_delta =
            output_error.array() * sigmoid_derivative(output).array();

        // 3. 隐藏层误差与 delta
        Eigen::VectorXd hidden_error =
            weights_hidden_output.transpose() * output_delta;
        Eigen::VectorXd hidden_delta =
            hidden_error.array() * sigmoid_derivative(hidden_a).array();

        // 4. SGD 参数更新（+= 累加梯度修正）
        weights_hidden_output +=
            learning_rate * output_delta * hidden_a.transpose();
        bias_output += learning_rate * output_delta;

        weights_input_hidden +=
            learning_rate * hidden_delta * input.transpose();
        bias_hidden += learning_rate * hidden_delta;
    }
};

/* ============================== main =================================== */
int main() {
    // 构造网络: 2 输入 → 3 隐藏 → 1 输出，学习率 0.1
    NeuralNetwork nn(2, 3, 1, 0.1);

    // XOR 异或玩具数据集: {x₁, x₂} → {y}
    struct Sample {
        double x1, x2, y;
    };
    std::vector<Sample> dataset = {
        {0.0, 0.0, 0.0},
        {0.0, 1.0, 1.0},
        {1.0, 0.0, 1.0},
        {1.0, 1.0, 0.0}};

    const int epochs = 5000;
    std::cout << "开始训练 XOR 问题（" << epochs << " 轮）...\n"
              << std::endl;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double epoch_loss = 0.0;

        for (const auto &s : dataset) {
            Eigen::VectorXd input(2);
            input << s.x1, s.x2;
            Eigen::VectorXd target(1);
            target << s.y;

            nn.train(input, target);

            // 累积每个样本的均方误差
            Eigen::VectorXd pred = nn.feedforward(input);
            double error = target(0) - pred(0);
            epoch_loss += error * error;
        }

        epoch_loss /= dataset.size(); // 平均 MSE

        // 每 1000 轮打印一次损失
        if (epoch % 1000 == 0) {
            std::cout << "epoch " << epoch << "  |  MSE = "
                      << epoch_loss << std::endl;
        }
    }

    /* ------- 训练结束后测试所有样本 ------- */
    std::cout << "\n训练完成，测试结果:\n"
              << std::endl;
    for (const auto &s : dataset) {
        Eigen::VectorXd input(2);
        input << s.x1, s.x2;
        Eigen::VectorXd pred = nn.feedforward(input);

        std::cout << "  (" << s.x1 << ", " << s.x2 << ")"
                  << "  →  预测 " << pred(0)
                  << "  (目标 " << s.y << ")" << std::endl;
    }

    return 0;
}
