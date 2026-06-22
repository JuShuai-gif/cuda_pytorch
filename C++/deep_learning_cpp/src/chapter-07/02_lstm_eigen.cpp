/*
 * 02_lstm_eigen.cpp - 第 7 章：循环神经网络与 LSTM
 * 基于 Eigen 的 LSTM 单元与前向网络实现（对应原书第 227-230 页）
 *
 * 演示内容：
 *   1. EigenLSTMCell：将遗忘门/输入门/候选门/输出门的计算合并为单次矩阵乘法
 *   2. "梯度高速公路"：遗忘门偏置初始化为 1.0，保证梯度沿时间轴反向传播不衰减
 *   3. LSTMNetwork：单层 LSTM + 线性输出投影
 *   4. 时序 XOR 模式演示：同一输入在不同位置因上下文不同而产生不同输出
 */

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>

using Eigen::MatrixXf;
using Eigen::VectorXf;

/* ============================= 工具函数 ================================== */

// Sigmoid 激活函数（带数值稳定钳位）
inline float sigmoid(float x) {
    // 钳位避免 exp 溢出
    if (x > 50.0f) return 1.0f;
    if (x < -50.0f) return 0.0f;
    return 1.0f / (1.0f + std::exp(-x));
}

// 将 sigmoid 逐元素应用于向量
inline VectorXf sigmoid_vec(const VectorXf &x) {
    VectorXf result(x.size());
    for (int i = 0; i < x.size(); ++i) {
        result(i) = sigmoid(x(i));
    }
    return result;
}

// 将 tanh 逐元素应用于向量（带数值稳定钳位）
inline VectorXf tanh_vec(const VectorXf &x) {
    VectorXf result(x.size());
    for (int i = 0; i < x.size(); ++i) {
        // 钳位：tanh(±50) ≈ ±1.0 已饱和
        float clamped = std::max(-50.0f, std::min(50.0f, x(i)));
        result(i) = std::tanh(clamped);
    }
    return result;
}

/* ======================== Xavier 初始化 ================================== */
/*
 * Xavier / Glorot uniform 初始化
 * 方差 = 2.0 / (fan_in + fan_out)，均匀分布区间 [-limit, +limit]
 */
void xavier_init(MatrixXf &mat, std::mt19937 &rng) {
    int fan_in = static_cast<int>(mat.cols());
    int fan_out = static_cast<int>(mat.rows());
    float limit = std::sqrt(6.0f / (fan_in + fan_out));
    std::uniform_real_distribution<float> dist(-limit, limit);
    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            mat(i, j) = dist(rng);
        }
    }
}

/* ======================= EigenLSTMCell =================================== */
/*
 * 单个 LSTM 单元的核心实现。
 *
 * 关键设计：所有四个门（遗忘门、输入门、候选门、输出门）的线性变换
 * 通过 W_combined 和 U_combined 一次性完成，仅需一次矩阵乘法。
 *
 * 门的排列顺序：
 *   [0:hidden)            → 遗忘门 (forget gate)
 *   [hidden:2*hidden)     → 输入门 (input gate)
 *   [2*hidden:3*hidden)   → 候选状态 (candidate / \tilde{C})
 *   [3*hidden:4*hidden)   → 输出门 (output gate)
 *
 * 梯度高速公路原理：
 *   C_t = forget_gate ⊙ C_{t-1} + input_gate ⊙ \tilde{C}_t
 *   当 forget_gate ≈ 1 时，∂C_t/∂C_{t-1} = forget_gate ≈ 1，
 *   梯度可以沿时间轴几乎无损地回传——这就是 LSTM 解决长期依赖的核心。
 */
class EigenLSTMCell {
private:
    int input_size;  // 每个时刻输入的特征数
    int hidden_size; // 隐藏状态 / 细胞状态的维度

    // 合并权重矩阵：
    //   W_combined ∈ R^{4*hidden_size × input_size}
    //   U_combined ∈ R^{4*hidden_size × hidden_size}
    //   b_combined ∈ R^{4*hidden_size}
    MatrixXf W_combined; // 输入 → 门（四个门的权重垂直堆叠）
    MatrixXf U_combined; // 隐藏状态 → 门（四个门的权重垂直堆叠）
    VectorXf b_combined; // 偏置（四个门的偏置垂直堆叠）

public:
    /* ------- 构造函数：Xavier 初始化 + 遗忘门偏置设 1.0 ----------- */
    EigenLSTMCell(int in_size, int h_size,
                  std::mt19937 &rng) : input_size(in_size), hidden_size(h_size),
                                       W_combined(4 * h_size, in_size),
                                       U_combined(4 * h_size, h_size),
                                       b_combined(VectorXf::Zero(4 * h_size)) {
        // Xavier 初始化 W 和 U
        xavier_init(W_combined, rng);
        xavier_init(U_combined, rng);

        // 关键技巧：遗忘门偏置初始化为 1.0
        // 这使 forget_gate 初始 ≈ σ(1.0) ≈ 0.73，
        // 让网络在训练初期倾向于"记住"，而非"忘记"
        // 这是 LSTM 论文推荐的实践（Jozefowicz et al., 2015）
        b_combined.segment(0, hidden_size).setOnes();
    }

    /* ------- 前向传播：单时间步 -------------------------------- */
    /*
     * 输入：
     *   input      - 当前时刻的输入向量 (input_size)
     *   prev_hidden - 上一时刻的隐藏状态 (hidden_size)
     *   prev_cell   - 上一时刻的细胞状态 (hidden_size)
     *
     * 返回：
     *   {new_hidden, new_cell} - 当前时刻的隐藏状态和细胞状态
     */
    struct ForwardResult {
        VectorXf hidden;
        VectorXf cell;
    };

    ForwardResult forward(const VectorXf &input,
                          const VectorXf &prev_hidden,
                          const VectorXf &prev_cell) {
        // ① 一次性计算所有四个门的线性部分（单次大矩阵乘法！）
        // gates = W_combined * input + U_combined * prev_hidden + b_combined
        VectorXf gates = W_combined * input
                         + U_combined * prev_hidden
                         + b_combined;

        // ② 从合并的 gates 向量中拆解出四个门
        VectorXf forget_gate = gates.segment(0, hidden_size);                  // [0:h)
        VectorXf input_gate = gates.segment(hidden_size, hidden_size);         // [h:2h)
        VectorXf candidate_gate = gates.segment(2 * hidden_size, hidden_size); // [2h:3h)
        VectorXf output_gate = gates.segment(3 * hidden_size, hidden_size);    // [3h:4h)

        // ③ 施加激活函数
        // 遗忘门、输入门、输出门：sigmoid（输出值 ∈ (0,1)，起门控作用）
        forget_gate = sigmoid_vec(forget_gate);
        input_gate = sigmoid_vec(input_gate);
        output_gate = sigmoid_vec(output_gate);

        // 候选状态：tanh（输出值 ∈ (-1,1)，代表候选信息）
        candidate_gate = tanh_vec(candidate_gate);

        // ④ 更新细胞状态（核心记忆机制）
        // C_t = f_t ⊙ C_{t-1} + i_t ⊙ \tilde{C}_t
        //   - f_t ⊙ C_{t-1}：遗忘门决定保留多少旧记忆 → "梯度高速公路"
        //   - i_t ⊙ \tilde{C}_t：输入门决定吸收多少新信息 → "信息滤网"
        VectorXf new_cell = forget_gate.array() * prev_cell.array()
                            + input_gate.array() * candidate_gate.array();

        // ⑤ 计算新的隐藏状态（输出门过滤细胞状态）
        // h_t = o_t ⊙ tanh(C_t)
        //   - o_t：输出门决定将细胞状态的哪些部分暴露给下一层/下一时刻
        //   - tanh(C_t)：将细胞状态压缩到 (-1,1)，避免数值爆炸
        VectorXf new_hidden = output_gate.array() * tanh_vec(new_cell).array();

        return {new_hidden, new_cell};
    }

    // 访问器（供 LSTMNetwork 控制）
    int getInputSize() const {
        return input_size;
    }
    int getHiddenSize() const {
        return hidden_size;
    }
};

/* ======================= LSTMNetwork ===================================== */
/*
 * 基于 EigenLSTMCell 的完整 LSTM 网络：
 *   输入序列 → LSTM 逐时间步更新隐藏/细胞状态 → 输出投影
 */
class LSTMNetwork {
private:
    EigenLSTMCell lstm_cell;
    MatrixXf W_output; // 输出投影权重 [1, hidden_size]
    VectorXf b_output; // 输出偏置 [1]
    std::mt19937 rng;

public:
    /* ------- 构造函数 ----------- */
    LSTMNetwork(int input_size, int hidden_size) : lstm_cell(input_size, hidden_size, rng),
                                                   W_output(1, hidden_size),
                                                   b_output(VectorXf::Zero(1)),
                                                   rng(std::mt19937(42)) {
        // Xavier 初始化输出层
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (int i = 0; i < hidden_size; ++i) {
            W_output(0, i) = dist(rng) * 0.1f;
        }
    }

    /* ------- 前向传播：处理整个时序 ----------------------------- */
    /*
     * 逐时间步迭代：读取输入序列，更新隐藏 / 细胞状态，
     * 最后将隐藏状态线性投影为标量输出。
     */
    float forward(const std::vector<VectorXf> &sequence) {
        int seq_len = static_cast<int>(sequence.size());
        int hidden_size = lstm_cell.getHiddenSize();

        // 初始状态设零
        VectorXf hidden = VectorXf::Zero(hidden_size);
        VectorXf cell = VectorXf::Zero(hidden_size);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "\n--- 逐时间步 LSTM 状态演化 ---\n";

        for (int t = 0; t < seq_len; ++t) {
            auto [new_hidden, new_cell] = lstm_cell.forward(
                sequence[t], hidden, cell);

            hidden = new_hidden;
            cell = new_cell;

            std::cout << "t=" << t << " | input=[" << sequence[t].transpose()
                      << "] | hidden=[" << hidden.transpose()
                      << "] | cell=[" << cell.transpose() << "]\n";
        }

        // 输出投影：将最终隐藏状态映射为标量
        float output = (W_output * hidden)(0) + b_output(0);

        std::cout << "  最终隐藏状态: " << hidden.transpose() << "\n";
        std::cout << "  最终输出值: " << output << "\n";

        return output;
    }
};

/* =========================== main ======================================== */
/*
 * 演示：时序上下文依赖
 *
 * 使用场景：输入序列为 [[1,0], [0,1], [1,0]]，
 * 虽然 t=0 和 t=2 的输入相同都是 [1,0]，
 * 但 LSTM 的隐藏状态已经累积了 t=1 的信息，
 * 因此会对相同输入产生不同的输出——这正是 RNN/LSTM
 * 区别于前馈网络的核心能力：理解 "上下文"。
 *
 * 四个门的作用速查：
 *   - 遗忘门 (forget gate)：   控制从细胞状态中丢弃什么信息
 *   - 输入门 (input gate)：    控制将哪些新信息写入细胞状态
 *   - 候选门 (candidate gate)：生成候选的新信息内容
 *   - 输出门 (output gate)：   控制从细胞状态输出什么给下一层
 *
 * 梯度高速公路 = 细胞状态 (cell state)
 *   回忆反向传播链式法则：
 *     ∂L/∂C_{t-1} = ∂L/∂C_t · forget_gate_t
 *   如果 forget_gate ≈ 1.0，梯度几乎无衰减地穿越时间，
 *   这正是 LSTM 能处理 100+ 时间步依赖的原因。
 */
int main() {
    std::cout << "================================================================\n";
    std::cout << "  第 7 章：循环神经网络与 LSTM - Eigen 实现\n";
    std::cout << "================================================================\n\n";

    // 创建网络：输入维度 = 2，隐藏维度 = 4
    std::cout << "创建 LSTMNetwork(输入维度=2, 隐藏维度=4)...\n";
    LSTMNetwork model(2, 4);

    // 构造时序输入序列（XOR 模式）
    // t=0: [1,0] → 期望较低值
    // t=1: [0,1] → 期望中间值
    // t=2: [1,0] → 期望较高值（因为 cell 记住了 [0,1] 的信息）
    std::vector<VectorXf> sequence;
    {
        VectorXf x0(2);
        x0 << 1.0f, 0.0f;
        VectorXf x1(2);
        x1 << 0.0f, 1.0f;
        VectorXf x2(2);
        x2 << 1.0f, 0.0f; // 与 x0 相同！
        sequence = {x0, x1, x2};
    }

    std::cout << "输入序列: [[1,0], [0,1], [1,0]]（注意 t=0 和 t=2 输入相同）\n";
    float result = model.forward(sequence);

    std::cout << "\n================================\n";
    std::cout << "  关键洞察\n";
    std::cout << "================================\n";
    std::cout << "1. 遗忘门偏置初始化为 1.0 → σ(1.0)≈0.73，历史信息被保留\n";
    std::cout << "2. 同一输入 [1,0] 在 t=0 和 t=2 处产生的隐藏状态不同\n";
    std::cout << "   → 因为中间经过 t=1 的 [0,1] 修改了细胞状态\n";
    std::cout << "3. 梯度高速公路：forget_gate≈1 确保梯度沿时间反向传播不衰减\n";

    return 0;
}
