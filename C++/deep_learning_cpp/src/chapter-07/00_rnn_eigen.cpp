/*
 * 00_rnn_eigen.cpp - 第 7 章：循环神经网络与 LSTM
 * 基于 Eigen 从零实现 RNNCell 循环神经单元（对应原书第 214-216 页）
 *
 * 演示内容：
 *   1. RNNCell 类：W_xh, W_hh, W_hy 三组权重矩阵的 Xavier 初始化
 *   2. tanh 隐藏层激活 + sigmoid 输出层激活
 *   3. 循环结构：每个时间步复用同一组权重处理新输入
 *   4. 处理简单二元序列 [1,0], [0,1], [1,1] 展示隐状态传递
 *
 *   公式：
 *     h_t = tanh(W_xh · x_t + W_hh · h_{t-1} + b_h)
 *     y_t = sigmoid(W_hy · h_t + b_y)
 */

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <functional>
#include <cmath>
#include <random>
#include <iomanip>

/* ============================ RNNCell 循环神经单元 ====================== */
class RNNCell {
private:
    int input_size;
    int hidden_size;
    int output_size;

    // 权重矩阵: W_xh(输入→隐藏), W_hh(隐藏→隐藏), W_hy(隐藏→输出)
    Eigen::MatrixXf W_xh;
    Eigen::MatrixXf W_hh;
    Eigen::MatrixXf W_hy;

    // 偏置向量
    Eigen::VectorXf b_h;
    Eigen::VectorXf b_y;

    // 隐状态（在时间步之间传递）
    Eigen::VectorXf hidden_state;

    // 激活函数与导数
    std::function<float(float)> hidden_activation; // 默认 tanh
    std::function<float(float)> output_activation; // 默认 sigmoid

public:
    /* ------- Xavier 均匀初始化 ------- */
    static Eigen::MatrixXf xavier_init(int rows, int cols, unsigned &seed) {
        std::mt19937 gen(seed++);
        // Xavier: 方差 = 2 / (fan_in + fan_out), 均匀分布边界 = √(6/(fan_in + fan_out))
        float limit = std::sqrt(6.0f / (rows + cols));
        std::uniform_real_distribution<float> dist(-limit, limit);
        Eigen::MatrixXf mat(rows, cols);
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                mat(r, c) = dist(gen);
        return mat;
    }

    /* ------- 构造函数 ------- */
    RNNCell(int input_size, int hidden_size, int output_size) : input_size(input_size), hidden_size(hidden_size),
                                                                output_size(output_size) {
        unsigned seed = 42;

        // Xavier 初始化所有权重矩阵
        W_xh = xavier_init(hidden_size, input_size, seed);
        W_hh = xavier_init(hidden_size, hidden_size, seed);
        W_hy = xavier_init(output_size, hidden_size, seed);

        // 偏置初始化为零
        b_h = Eigen::VectorXf::Zero(hidden_size);
        b_y = Eigen::VectorXf::Zero(output_size);

        // 隐状态初始化为零向量（t=0 时不存在历史信息）
        hidden_state = Eigen::VectorXf::Zero(hidden_size);

        // 默认激活函数: tanh 用于隐藏层
        hidden_activation = [](float x) {
            return std::tanh(x);
        };

        // sigmoid 用于输出层，将输出映射到 (0, 1)
        output_activation = [](float x) {
            return 1.0f / (1.0f + std::exp(-x));
        };
    }

    /* ------- 前向传播（单个时间步） ------- */
    /*
     * 参数 input: 当前时间步的输入向量 x_t，尺寸 input_size × 1
     * 返回值: pair<hidden_state, output>，即 h_t 与 y_t
     *
     * 核心公式:
     *   h_t = tanh(W_xh · x_t + W_hh · h_{t-1} + b_h)
     *   y_t = sigmoid(W_hy · h_t + b_y)
     */
    std::pair<Eigen::VectorXf, Eigen::VectorXf> forward(
        const Eigen::VectorXf &input) {
        // 步骤1: 计算隐藏层线性组合 z_h = W_xh·x_t + W_hh·h_{t-1} + b_h
        Eigen::VectorXf z_h = W_xh * input + W_hh * hidden_state + b_h;

        // 步骤2: 应用 tanh 激活 → h_t
        hidden_state = z_h.unaryExpr(hidden_activation);

        // 步骤3: 计算输出层线性组合 z_y = W_hy·h_t + b_y
        Eigen::VectorXf z_y = W_hy * hidden_state + b_y;

        // 步骤4: 应用 sigmoid 激活 → y_t
        Eigen::VectorXf output = z_y.unaryExpr(output_activation);

        return {hidden_state, output};
    }

    /* ------- 重置隐状态（开始新序列时调用） ------- */
    void reset_hidden_state() {
        hidden_state = Eigen::VectorXf::Zero(hidden_size);
    }

    /* ------- 获取当前隐状态（调试用） ------- */
    Eigen::VectorXf get_hidden_state() const {
        return hidden_state;
    }

    /* ------- 打印权重信息 ------- */
    void print_weights() const {
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "══════ RNNCell 权重信息 ══════\n";
        std::cout << "结构: " << input_size << " 输入 → "
                  << hidden_size << " 隐藏 → "
                  << output_size << " 输出\n\n";
        std::cout << "--- W_xh (" << hidden_size << "×" << input_size
                  << "), 输入→隐藏 ---\n"
                  << W_xh << "\n\n";
        std::cout << "--- W_hh (" << hidden_size << "×" << hidden_size
                  << "), 隐藏→隐藏(循环连接) ---\n"
                  << W_hh << "\n\n";
        std::cout << "--- W_hy (" << output_size << "×" << hidden_size
                  << "), 隐藏→输出 ---\n"
                  << W_hy << "\n\n";
        std::cout << "--- b_h 隐藏偏置 ---\n"
                  << b_h.transpose() << "\n\n";
        std::cout << "--- b_y 输出偏置 ---\n"
                  << b_y.transpose() << "\n";
        std::cout << "══════════════════════════════\n";
    }
};

/* ============================== main =================================== */
int main() {
    std::cout << std::fixed << std::setprecision(4);

    // 构造 RNN: 2 输入 → 4 隐藏 → 1 输出，用于简单二元序列任务
    RNNCell rnn(2, 4, 1);

    rnn.print_weights();

    // === 演示序列处理 ===
    // 输入序列: [1,0] → [0,1] → [1,1]（每步两个值，如二元特征）
    std::vector<Eigen::VectorXf> sequence;
    {
        Eigen::VectorXf x1(2);
        x1 << 1.0f, 0.0f;
        sequence.push_back(x1);
        Eigen::VectorXf x2(2);
        x2 << 0.0f, 1.0f;
        sequence.push_back(x2);
        Eigen::VectorXf x3(2);
        x3 << 1.0f, 1.0f;
        sequence.push_back(x3);
    }

    std::cout << "\n══════ 逐时间步处理序列 ══════\n";
    std::cout << "（权重在每个时间步复用，h_t 跨步传递）\n\n";

    rnn.reset_hidden_state(); // 新序列开始，清零隐状态

    for (size_t t = 0; t < sequence.size(); ++t) {
        // 前向传播：输入 x_t，得到 h_t 和 y_t
        auto [hidden, output] = rnn.forward(sequence[t]);

        std::cout << "=== 时间步 t=" << t << " ===\n";
        std::cout << "  输入 x" << t << "  = ["
                  << sequence[t](0) << ", "
                  << sequence[t](1) << "]\n";
        std::cout << "  W_xh * x" << t << " 加上 W_hh * h_{" << (t > 0 ? std::to_string(t - 1) : "0=0")
                  << "}\n";
        std::cout << "  隐状态 h" << t << " = [";
        for (int i = 0; i < hidden.size(); ++i)
            std::cout << (i > 0 ? ", " : "") << hidden(i);
        std::cout << "]\n";
        std::cout << "  输出   y" << t << " = " << output(0) << "\n\n";
    }

    // === RNN 循环特性说明 ===
    std::cout << "══════ RNN 循环特性说明 ══════\n";
    std::cout << "RNN 的核心是「权重复用」：\n";
    std::cout << "  - 同一个 W_xh 作用于所有时间步的输入 x_t\n";
    std::cout << "  - 同一个 W_hh 作用于所有时间步的隐状态 h_{t-1}\n";
    std::cout << "  - 共享权重使 RNN 能够处理任意长度的序列\n";
    std::cout << "  - h_t 作为「记忆」将历史信息传递到未来\n";
    std::cout << "══════════════════════════════\n";

    return 0;
}
