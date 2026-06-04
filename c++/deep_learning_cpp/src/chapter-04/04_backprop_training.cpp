/*
 * 04_backprop_training.cpp - 第 4 章：构建基础神经网络
 * 反向传播深入剖析 + SGD 变体对比演示（对应原书第 121-125 页）
 *
 * 演示内容：
 *   1. 手写反向传播的完整推导与实现（无框架依赖）
 *   2. Sigmoid 激活函数与导数
 *   3. 均方误差（MSE）损失函数
 *   4. 三种训练模式的实现与对比：
 *      (a) 批量梯度下降（Batch GD）—— 每轮对全部样本平均梯度后更新
 *      (b) 随机梯度下降（SGD）—— 逐样本更新
 *      (c) 小批量随机梯度下降（Mini-batch SGD）—— 每 2 个样本更新一次
 */

#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <cmath>
#include <iomanip>
#include <algorithm>

/* =========================== 激活函数工具 =========================== */

// Sigmoid: σ(z) = 1 / (1 + e^{-z})
Eigen::VectorXd sigmoid(const Eigen::VectorXd &z) {
    return 1.0 / (1.0 + (-z.array()).exp());
}

// Sigmoid 导数（已知激活值 a = σ(z)）: σ'(z) = a * (1 - a)
Eigen::VectorXd sigmoid_deriv_from_output(const Eigen::VectorXd &a) {
    return a.array() * (1.0 - a.array());
}

/* =========================== 单次训练 + MSE ========================== */

// 对单个样本执行完整的前向传播 → 反向传播 → SGD 参数更新，返回 MSE
double train_one_mse(
    const Eigen::VectorXd &x,
    const Eigen::VectorXd &t,
    Eigen::MatrixXd &Wih, // [3×2] 输入 → 隐藏权重
    Eigen::VectorXd &bh,  // [3]   隐藏层偏置
    Eigen::MatrixXd &Who, // [1×3] 隐藏 → 输出权重
    Eigen::VectorXd &bo,  // [1]   输出层偏置
    double lr) {
    /* ------ 1. 前向传播 ------ */
    Eigen::VectorXd z1 = Wih * x + bh;  // 隐藏层线性输入
    Eigen::VectorXd a1 = sigmoid(z1);   // 隐藏层激活
    Eigen::VectorXd z2 = Who * a1 + bo; // 输出层线性输入
    Eigen::VectorXd y = sigmoid(z2);    // 最终预测

    /* ------ 2. 计算 MSE ------ */
    double mse = (t - y).squaredNorm() / t.size();

    /* ------ 3. 反向传播 ------ */
    // 输出层 delta: δ_o = σ'(y) ⊙ (t - y)
    Eigen::VectorXd delta_out =
        sigmoid_deriv_from_output(y).array() * (t - y).array();

    // 隐藏层 delta: δ_h = (W_ho^T · δ_o) ⊙ σ'(a₁)
    Eigen::VectorXd delta_hid =
        (Who.transpose() * delta_out).array()
        * sigmoid_deriv_from_output(a1).array();

    /* ------ 4. 梯度计算 ------ */
    Eigen::MatrixXd dWho = delta_out * a1.transpose(); // [1×3]
    Eigen::VectorXd dbo = delta_out;                   // [1]
    Eigen::MatrixXd dWih = delta_hid * x.transpose();  // [3×2]
    Eigen::VectorXd dbh = delta_hid;                   // [3]

    /* ------ 5. SGD 参数更新（+=） ------ */
    Who += lr * dWho;
    bo += lr * dbo;
    Wih += lr * dWih;
    bh += lr * dbh;

    return mse;
}

/* ========================= 参数随机初始化 ========================== */

void init_random(Eigen::MatrixXd &W, Eigen::VectorXd &b) {
    W = Eigen::MatrixXd::Random(W.rows(), W.cols());
    b = Eigen::VectorXd::Random(b.rows());
}

/* ======================== 训练模式对比 ============================== */

int main() {
    // XOR 异或数据集
    Eigen::MatrixXd X(4, 2); // 4 个样本，每个 2 维
    Eigen::MatrixXd T(4, 1); // 4 个目标值
    X << 0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0;
    T << 0.0,
        1.0,
        1.0,
        0.0;

    // 将数据和目标转为样本列表便于操作
    std::vector<Eigen::VectorXd> x_samples(4), t_samples(4);
    for (int i = 0; i < 4; ++i) {
        x_samples[i] = X.row(i).transpose();
        t_samples[i] = T.row(i).transpose();
    }

    const int epochs = 2000;
    const double lr = 0.1;
    const int input = 2;
    const int hidden = 3;
    const int output = 1;
    const int report_interval = 200; // 每 200 轮报告一次

    /* ================================================================
     * (a) 批量梯度下降（Batch GD）
     *     累积全部样本的梯度后一次性更新参数
     * ================================================================ */
    {
        // 初始化参数
        Eigen::MatrixXd Wih(hidden, input);
        Eigen::VectorXd bh(hidden);
        Eigen::MatrixXd Who(output, hidden);
        Eigen::VectorXd bo(output);
        init_random(Wih, bh);
        init_random(Who, bo);

        std::cout << "══════ (a) 批量梯度下降 (Batch GD) ══════\n"
                  << std::endl;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            // 累积梯度
            Eigen::MatrixXd acc_dWho = Eigen::MatrixXd::Zero(output, hidden);
            Eigen::VectorXd acc_dbo = Eigen::VectorXd::Zero(output);
            Eigen::MatrixXd acc_dWih = Eigen::MatrixXd::Zero(hidden, input);
            Eigen::VectorXd acc_dbh = Eigen::VectorXd::Zero(hidden);
            double epoch_loss = 0.0;

            for (int i = 0; i < 4; ++i) {
                // 前向传播
                Eigen::VectorXd z1 = Wih * x_samples[i] + bh;
                Eigen::VectorXd a1 = sigmoid(z1);
                Eigen::VectorXd z2 = Who * a1 + bo;
                Eigen::VectorXd y = sigmoid(z2);

                epoch_loss += (t_samples[i] - y).squaredNorm() / output;

                // 反向传播（仅计算梯度，不更新）
                Eigen::VectorXd delta_out =
                    sigmoid_deriv_from_output(y).array()
                    * (t_samples[i] - y).array();
                Eigen::VectorXd delta_hid =
                    (Who.transpose() * delta_out).array()
                    * sigmoid_deriv_from_output(a1).array();

                acc_dWho += delta_out * a1.transpose();
                acc_dbo += delta_out;
                acc_dWih += delta_hid * x_samples[i].transpose();
                acc_dbh += delta_hid;
            }

            // 平均梯度后一次性更新
            Who += lr * acc_dWho / 4.0;
            bo += lr * acc_dbo / 4.0;
            Wih += lr * acc_dWih / 4.0;
            bh += lr * acc_dbh / 4.0;

            epoch_loss /= 4.0;

            if (epoch % report_interval == 0) {
                std::cout << "  epoch " << std::setw(5) << epoch
                          << "  |  MSE = " << epoch_loss << std::endl;
            }
        }
    }

    /* ================================================================
     * (b) 随机梯度下降（SGD）
     *     每个样本单独计算梯度并立即更新，每轮内随机打乱样本顺序
     * ================================================================ */
    {
        Eigen::MatrixXd Wih(hidden, input);
        Eigen::VectorXd bh(hidden);
        Eigen::MatrixXd Who(output, hidden);
        Eigen::VectorXd bo(output);
        init_random(Wih, bh);
        init_random(Who, bo);

        // 随机数生成器用于洗牌
        std::random_device rd;
        std::mt19937 rng(rd());

        std::cout << "\n══════ (b) 随机梯度下降 (SGD) ══════\n"
                  << std::endl;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            // 每轮随机打乱样本顺序（SGD 的典型做法）
            std::vector<int> indices = {0, 1, 2, 3};
            std::shuffle(indices.begin(), indices.end(), rng);

            double epoch_loss = 0.0;
            for (int idx : indices) {
                // 逐样本训练并更新参数
                epoch_loss += train_one_mse(
                    x_samples[idx], t_samples[idx], Wih, bh, Who, bo, lr);
            }
            epoch_loss /= 4.0;

            if (epoch % report_interval == 0) {
                std::cout << "  epoch " << std::setw(5) << epoch
                          << "  |  MSE = " << epoch_loss << std::endl;
            }
        }
    }

    /* ================================================================
     * (c) 小批量随机梯度下降（Mini-batch SGD）
     *     每 2 个样本累积梯度后更新一次
     * ================================================================ */
    {
        Eigen::MatrixXd Wih(hidden, input);
        Eigen::VectorXd bh(hidden);
        Eigen::MatrixXd Who(output, hidden);
        Eigen::VectorXd bo(output);
        init_random(Wih, bh);
        init_random(Who, bo);

        // 随机数生成器
        std::random_device rd;
        std::mt19937 rng(rd());

        const int batch_size = 2;
        std::cout << "\n══════ (c) 小批量随机梯度下降 (Mini-batch SGD, size="
                  << batch_size << ") ══════\n"
                  << std::endl;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            // 打乱样本顺序
            std::vector<int> indices = {0, 1, 2, 3};
            std::shuffle(indices.begin(), indices.end(), rng);

            double epoch_loss = 0.0;

            for (int start = 0; start < 4; start += batch_size) {
                // 累积小批量内的梯度
                Eigen::MatrixXd acc_dWho = Eigen::MatrixXd::Zero(output, hidden);
                Eigen::VectorXd acc_dbo = Eigen::VectorXd::Zero(output);
                Eigen::MatrixXd acc_dWih = Eigen::MatrixXd::Zero(hidden, input);
                Eigen::VectorXd acc_dbh = Eigen::VectorXd::Zero(hidden);
                double batch_loss = 0.0;

                int end = std::min(start + batch_size, 4);
                for (int i = start; i < end; ++i) {
                    int idx = indices[i];

                    Eigen::VectorXd z1 = Wih * x_samples[idx] + bh;
                    Eigen::VectorXd a1 = sigmoid(z1);
                    Eigen::VectorXd z2 = Who * a1 + bo;
                    Eigen::VectorXd y = sigmoid(z2);

                    batch_loss += (t_samples[idx] - y).squaredNorm() / output;

                    Eigen::VectorXd delta_out =
                        sigmoid_deriv_from_output(y).array()
                        * (t_samples[idx] - y).array();
                    Eigen::VectorXd delta_hid =
                        (Who.transpose() * delta_out).array()
                        * sigmoid_deriv_from_output(a1).array();

                    acc_dWho += delta_out * a1.transpose();
                    acc_dbo += delta_out;
                    acc_dWih += delta_hid * x_samples[idx].transpose();
                    acc_dbh += delta_hid;
                }

                int actual_batch = end - start;
                Who += lr * acc_dWho / actual_batch;
                bo += lr * acc_dbo / actual_batch;
                Wih += lr * acc_dWih / actual_batch;
                bh += lr * acc_dbh / actual_batch;

                epoch_loss += batch_loss;
            }

            epoch_loss /= 4.0;

            if (epoch % report_interval == 0) {
                std::cout << "  epoch " << std::setw(5) << epoch
                          << "  |  MSE = " << epoch_loss << std::endl;
            }
        }
    }

    std::cout << "\n训练完成 — 三种优化模式对比结束。\n"
              << std::endl;
    return 0;
}
