/*
 * 00_eigen_mlp_xor.cpp - 第 5 章：用 Eigen 从零构建多层感知机
 * 基于矩阵运算的 Layer 抽象 + 反向传播（对应原书第 130-135 页）
 *
 * 演示内容：
 *   1. Layer 类：可组合的矩阵化前向/反向传播（ReLU 激活）
 *   2. MultilayerPerceptron 类：层链式调用与 Mini-batch 训练
 *   3. XOR 异或问题验证网络非线性拟合能力
 *   4. 批量计算：输入输出均为 MatrixXd（每列一个样本）
 */

#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <vector>

/* =========================== Layer 单层 =============================== */
class Layer {
private:
    // 可训练参数
    Eigen::MatrixXd weights; // output_size × input_size
    Eigen::VectorXd biases;  // output_size

    // 前向传播缓存（反向传播时复用）
    Eigen::MatrixXd activation; // 线性输出 z = W·x + b
    Eigen::MatrixXd input;      // 该层输入 x
    Eigen::MatrixXd output;     // 激活后输出 a = ReLU(z)

public:
    /* ------- 构造函数：随机初始化权重，偏置置零 ------- */
    Layer(int input_size, int output_size) {
        // 小随机数打破对称性
        weights = Eigen::MatrixXd::Random(output_size, input_size) * 0.1;
        biases = Eigen::VectorXd::Zero(output_size);
    }

    /* ------- ReLU 激活函数 ------- */
    static Eigen::MatrixXd relu(const Eigen::MatrixXd &x) {
        return x.cwiseMax(0.0);
    }

    /* ------- ReLU 导数（对线性输出 z 求导） ------- */
    static Eigen::MatrixXd relu_derivative(const Eigen::MatrixXd &x) {
        return (x.array() > 0.0).cast<double>();
    }

    /* ------- 前向传播：x 为 input_size × batch_size ------- */
    Eigen::MatrixXd forward(const Eigen::MatrixXd &x) {
        input = x; // 缓存输入，反向传播计算梯度时使用
        // z = W·x + b（通过 replicate 广播偏置到每一列）
        activation = weights * input
                     + biases.replicate(1, input.cols());
        output = relu(activation); // a = ReLU(z)
        return output;
    }

    /* ------- 反向传播：从上游梯度计算参数梯度并更新 ------- */
    Eigen::MatrixXd backward(const Eigen::MatrixXd &grad_output,
                             double learning_rate) {
        // δ = ∂L/∂z = ∂L/∂a ◦ ReLU'(z)
        Eigen::MatrixXd grad_activation =
            grad_output.array() * relu_derivative(activation).array();

        // ∂L/∂W = δ · x^T
        Eigen::MatrixXd grad_weights =
            grad_activation * input.transpose();

        // ∂L/∂b = Σδ（每行求和）
        Eigen::VectorXd grad_biases =
            grad_activation.rowwise().sum();

        // ∂L/∂x = W^T · δ（向下层传递）
        Eigen::MatrixXd grad_input =
            weights.transpose() * grad_activation;

        // SGD 参数更新
        weights -= learning_rate * grad_weights;
        biases -= learning_rate * grad_biases;

        return grad_input;
    }
};

/* ======================= MultilayerPerceptron 多层感知机 =============== */
class MultilayerPerceptron {
private:
    std::vector<Layer> layers;
    double learning_rate;

public:
    /* ------- 构造函数：根据每层神经元数量构建 Layer 栈 ------- */
    MultilayerPerceptron(const std::vector<int> &layer_sizes, double lr) : learning_rate(lr) {
        for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
            layers.emplace_back(layer_sizes[i], layer_sizes[i + 1]);
        }
    }

    /* ------- 前向传播：逐层传递 ------- */
    Eigen::MatrixXd forward(const Eigen::MatrixXd &input) {
        Eigen::MatrixXd current = input;
        for (auto &layer : layers) {
            current = layer.forward(current);
        }
        return current;
    }

    /* ------- 训练一个 epoch（全量 Mini-batch） ------- */
    void train(const Eigen::MatrixXd &X, const Eigen::MatrixXd &y,
               int epochs) {
        // X: input_dim × N,  y: output_dim × N
        const int batch_size = static_cast<int>(y.cols());

        for (int epoch = 0; epoch < epochs; ++epoch) {
            // 1. 前向传播
            Eigen::MatrixXd pred = forward(X);

            // 2. 均方误差损失 MSE = (1/N) Σ (ŷ - y)²
            double loss = (pred - y).squaredNorm() / batch_size;

            // 3. 输出层梯度 ∂MSE/∂ŷ = 2(ŷ - y) / N
            Eigen::MatrixXd grad = 2.0 * (pred - y) / batch_size;

            // 4. 反向传播（逆序遍历所有层）
            for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i) {
                grad = layers[i].backward(grad, learning_rate);
            }

            // 5. 每 100 轮打印一次损失
            if (epoch % 100 == 0) {
                std::cout << "epoch " << epoch
                          << "  |  MSE = " << loss << std::endl;
            }
        }
    }

    /* ------- 预测：仅前向传播 ------- */
    Eigen::MatrixXd predict(const Eigen::MatrixXd &X) {
        return forward(X);
    }
};

/* ================================ main ================================= */
int main() {
    // XOR 数据集：每列一个样本，输入 2×4，标签 1×4
    Eigen::MatrixXd X(2, 4);
    // {{0,0}, {0,1}, {1,0}, {1,1}} → {0, 1, 1, 0}
    X << 0.0, 0.0, 1.0, 1.0,
        0.0, 1.0, 0.0, 1.0;

    Eigen::MatrixXd y(1, 4);
    y << 0.0, 1.0, 1.0, 0.0;

    // 构建多层感知机: 2→4→3→1，学习率 0.01
    std::vector<int> layer_sizes = {2, 4, 3, 1};
    MultilayerPerceptron mlp(layer_sizes, 0.01);

    const int epochs = 2000;
    std::cout << "开始训练 XOR 问题（" << epochs << " 轮）...\n"
              << std::endl;
    mlp.train(X, y, epochs);

    // 训练完成后测试所有样本
    Eigen::MatrixXd predictions = mlp.predict(X);
    std::cout << "\n训练完成，测试结果:\n"
              << std::endl;
    for (int i = 0; i < 4; ++i) {
        std::cout << "  (" << X(0, i) << ", " << X(1, i) << ")"
                  << "  →  预测 " << predictions(0, i)
                  << "  (目标 " << y(0, i) << ")" << std::endl;
    }

    return 0;
}
