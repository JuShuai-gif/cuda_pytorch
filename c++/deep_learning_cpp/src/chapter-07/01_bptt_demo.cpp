/*
 * 01_bptt_demo.cpp - 第 7 章：循环神经网络与 LSTM
 * BPTT 反向传播与梯度裁剪演示（对应原书第 216-221 页）
 *
 * 演示内容：
 *   1. 沿时间反向传播 (BPTT): 从 t=seq_len-1 向下到 t=0 累积梯度
 *   2. 梯度裁剪 (Gradient Clipping): L2 范数超过阈值时整体缩放
 *   3. 截断 BPTT 概念: 长序列只反向传播最后 K 个时间步
 *   4. 使用人造大梯度展示裁剪前后的范数对比
 *
 *   BPTT 核心公式:
 *     δ_t = (h_t - target_t + dh_{t+1} · W_hh^T) ◦ tanh'(z_t)
 *     dW_xh += δ_t · x_t^T
 *     dW_hh += δ_t · h_{t-1}^T
 *     dh_{t-1} = W_hh^T · δ_t
 */

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <random>

using Matrix = Eigen::MatrixXf;
using Vector = Eigen::VectorXf;

/* ========================= 前向传播状态缓存 ============================== */
// 保存前向传播过程中每个时间步的中间值，供 BPTT 反向传播使用
struct RNNForwardState {
    std::vector<Vector> inputs;          // 每个时间步的输入 x_0 .. x_{T-1}
    std::vector<Vector> hidden_states;   // 每个时间步的隐状态 h_0 .. h_{T-1}
    std::vector<Vector> pre_activations; // 每个时间步的线性输出 z_t = W_xh·x_t + W_hh·h_{t-1} + b_h
    std::vector<Vector> outputs;         // 每个时间步的输出 y_t

    Vector initial_hidden; // h_{-1}，即初始隐状态（通常为零向量）
};

/* ========================= tanh 及其导数 ================================ */
inline float tanh_derivative(float tanh_output) {
    // tanh'(x) = 1 - tanh²(x)，传入的是 tanh(x) 的结果
    return 1.0f - tanh_output * tanh_output;
}

/* ======================= 梯度裁剪 ======================================= */
/*
 * 计算所有梯度矩阵/向量的 L2 范数（弗罗贝尼乌斯范数），
 * 若总范数 > threshold，则将所有梯度等比例缩放到 threshold。
 *
 * 参数:
 *   grad_dict: 包含所有梯度矩阵/向量的 vector
 *   threshold: 裁剪阈值（默认 5.0，原书推荐 5~10 之间）
 *
 * 返回: float 裁剪前的范数与裁剪后的范数
 */
std::pair<float, float> clip_gradients(
    std::vector<Eigen::MatrixXf> &grad_dict,
    float threshold = 5.0f) {
    // 步骤1: 计算所有梯度的总 L2 范数
    float total_norm_sq = 0.0f;
    for (const auto &grad : grad_dict) {
        total_norm_sq += grad.squaredNorm();
    }
    float total_norm = std::sqrt(total_norm_sq);

    float new_norm = total_norm;

    // 步骤2: 若超过阈值，等比例缩放所有梯度
    if (total_norm > threshold) {
        float scale = threshold / total_norm;
        for (auto &grad : grad_dict) {
            grad *= scale;
        }
        new_norm = threshold; // 缩放后范数等于阈值
    }

    return {total_norm, new_norm};
}

/* ======================= BPTT 反向传播 ================================== */
/*
 * 沿时间反向传播 (Backpropagation Through Time)
 *
 * 参数:
 *   state:      前向传播缓存（含所有时间步的 x, h, z）
 *   targets:    目标序列 y_target_0 .. y_target_{T-1}
 *   W_xh:       输入→隐藏权重矩阵 [input, hidden]
 *   W_hh:       隐藏→隐藏权重矩阵 [hidden, hidden]
 *
 * 输出参数:
 *   dW_xh:   W_xh 梯度累加
 *   dW_hh:   W_hh 梯度累加
 *   db_h:    隐藏偏置梯度累加
 */
void backward_pass(
    const RNNForwardState &state,
    const Matrix &W_xh,
    const Matrix &W_hh,
    const std::vector<Vector> &targets,
    Matrix &dW_xh,
    Matrix &dW_hh,
    Vector &db_h) {
    int seq_len = static_cast<int>(state.inputs.size());
    int hidden_size = static_cast<int>(W_hh.rows());

    // dh_next: 从未来时间步传递回来的梯度（t+1 传递到 t）
    Vector dh_next = Vector::Zero(hidden_size);

    // 从最后一个时间步向前逐步累积梯度
    for (int t = seq_len - 1; t >= 0; --t) {
        // 步骤1: 输出误差 = 当前隐状态 - 目标值（简化 MSE 导数）
        //         加上从未来传递的梯度 dh_next
        Vector output_error = state.hidden_states[t] - targets[t] + dh_next;

        // 步骤2: 通过 tanh 导数传播到预激活层
        //   δ_t = error ◦ tanh'(z_t)   其中 tanh'(z) = 1 - tanh²(z) = 1 - h_t²
        Vector pre_grad(output_error.size());
        for (int i = 0; i < output_error.size(); ++i) {
            pre_grad(i) = output_error(i) * tanh_derivative(state.hidden_states[t](i));
        }

        // 步骤3: 累加权重梯度（BPTT: 沿时间步求和）
        //   dW_xh += δ_t · x_t^T
        dW_xh += pre_grad * state.inputs[t].transpose();

        //   dW_hh += δ_t · h_{t-1}^T  （t=0 时使用 init_hidden）
        Vector h_prev = (t > 0) ? state.hidden_states[t - 1] : state.initial_hidden;
        dW_hh += pre_grad * h_prev.transpose();

        //   偏置梯度: db_h += δ_t
        db_h += pre_grad;

        // 步骤4: 将梯度传递给 t-1 时刻
        //   dh_{t-1} = W_hh^T · δ_t
        dh_next = W_hh.transpose() * pre_grad;
    }
}

// 实际输出用,简单的一步：输出 = h_t（即用隐状态作为输出预测）
inline Vector compute_output(const Vector &hidden_state) {
    return hidden_state; // 简化：输出即隐状态
}

/* ======================= 前向传播 ====================================== */
RNNForwardState forward_pass(
    const std::vector<Vector> &inputs,
    const Matrix &W_xh,
    const Matrix &W_hh,
    const Vector &b_h,
    const Vector &initial_hidden) {
    RNNForwardState state;
    state.inputs = inputs;
    state.initial_hidden = initial_hidden;

    int seq_len = static_cast<int>(inputs.size());
    int hidden_size = static_cast<int>(b_h.size());

    Vector h_prev = initial_hidden;

    for (int t = 0; t < seq_len; ++t) {
        // z_t = W_xh·x_t + W_hh·h_{t-1} + b_h
        Vector z_t = W_xh * inputs[t] + W_hh * h_prev + b_h;
        state.pre_activations.push_back(z_t);

        // h_t = tanh(z_t)
        Vector h_t = z_t.unaryExpr([](float v) { return std::tanh(v); });
        state.hidden_states.push_back(h_t);

        // y_t = h_t（简化输出 = 隐状态）
        state.outputs.push_back(h_t);

        h_prev = h_t;
    }

    return state;
}

/* ============================== main =================================== */
int main() {
    std::cout << std::fixed << std::setprecision(4);

    const int input_size = 2;
    const int hidden_size = 3;
    const int seq_len = 5;

    // === 构造随机权重的小型 RNN ===
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    Matrix W_xh(hidden_size, input_size);
    Matrix W_hh(hidden_size, hidden_size);
    Vector b_h(hidden_size);

    for (int r = 0; r < hidden_size; ++r) {
        for (int c = 0; c < input_size; ++c)
            W_xh(r, c) = dist(gen);
        for (int c = 0; c < hidden_size; ++c)
            W_hh(r, c) = dist(gen);
        b_h(r) = dist(gen);
    }

    std::cout << "══════ 小规模 RNN 权重信息 ══════\n";
    std::cout << "结构: " << input_size << " 输入 → "
              << hidden_size << " 隐藏，序列长度 " << seq_len << "\n";
    std::cout << "W_xh:\n"
              << W_xh << "\n";
    std::cout << "W_hh:\n"
              << W_hh << "\n\n";

    // === 生成输入序列和目标序列 ===
    std::vector<Vector> inputs;
    std::vector<Vector> targets;
    for (int t = 0; t < seq_len; ++t) {
        Vector x(input_size);
        x << dist(gen), dist(gen);
        inputs.push_back(x);

        // 随机目标（简化：隐状态应逼近的向量）
        Vector y(hidden_size);
        for (int i = 0; i < hidden_size; ++i)
            y(i) = dist(gen);
        targets.push_back(y);
    }

    Vector initial_hidden = Vector::Zero(hidden_size);

    // === 步骤1: 前向传播 ===
    RNNForwardState state = forward_pass(
        inputs, W_xh, W_hh, b_h, initial_hidden);

    std::cout << "══════ 前向传播完成 ══════\n";
    for (int t = 0; t < seq_len; ++t) {
        std::cout << "t=" << t << "  输入=("
                  << inputs[t](0) << ", " << inputs[t](1) << ")  "
                  << "h_t=[";
        for (int i = 0; i < hidden_size; ++i)
            std::cout << (i > 0 ? ", " : "") << state.hidden_states[t](i);
        std::cout << "]\n";
    }

    // === 步骤2: BPTT 反向传播（正常梯度） ===
    Matrix dW_xh_norm = Matrix::Zero(hidden_size, input_size);
    Matrix dW_hh_norm = Matrix::Zero(hidden_size, hidden_size);
    Vector db_h_norm = Vector::Zero(hidden_size);

    backward_pass(state, W_xh, W_hh, targets, dW_xh_norm, dW_hh_norm, db_h_norm);

    std::cout << "\n══════ 正常 BPTT 反向传播梯度 ══════\n";
    std::cout << "dW_xh:\n"
              << dW_xh_norm << "\n";
    std::cout << "dW_hh:\n"
              << dW_hh_norm << "\n";
    std::cout << "db_h: " << db_h_norm.transpose() << "\n";

    // === 步骤3: 人为构造大梯度演示裁剪效果 ===
    std::cout << "\n══════ 梯度裁剪演示 ══════\n";

    // 制造一个人为偏大的梯度（模拟训练中可能爆炸的梯度）
    Matrix dW_xh_big = dW_xh_norm * 50.0f;
    Matrix dW_hh_big = dW_hh_norm * 50.0f;
    Vector db_h_big = db_h_norm * 50.0f;

    std::vector<Eigen::MatrixXf> grad_list;
    grad_list.push_back(dW_xh_big);
    grad_list.push_back(dW_hh_big);

    // 将 db_h 也加入列表（作为单列矩阵）
    Eigen::MatrixXf db_mat = db_h_big;
    grad_list.push_back(db_mat);

    float threshold = 5.0f;

    // 裁剪前的总范数
    float norm_before = 0.0f;
    for (const auto &g : grad_list) norm_before += g.squaredNorm();
    norm_before = std::sqrt(norm_before);

    std::cout << "裁剪前梯度总 L2 范数: " << norm_before << "\n";

    // 执行裁剪
    auto [before, after] = clip_gradients(grad_list, threshold);

    std::cout << "\n";
    std::cout << "  → 无裁剪时: grad_norm = " << before << "\n";
    std::cout << "  → 有裁剪时: grad_norm = " << after
              << " (阈值 " << threshold << ")\n";
    std::cout << "  → 缩放因子: " << threshold / before << "\n";

    // === 步骤4: 截断 BPTT 概念说明 ===
    std::cout << "\n══════ 截断 BPTT 概念说明 ══════\n";
    std::cout << "对于极长序列（如全文文本），完整 BPTT 存在两个问题:\n";
    std::cout << "  1. 计算量: T 时间步的反向传播复杂度 O(T²)\n";
    std::cout << "  2. 梯度问题: 过多连乘导致梯度消失/爆炸\n\n";
    std::cout << "截断策略: 只反向传播最后 K 个时间步（K=10~50）\n";
    std::cout << "  - 优点: 计算量降低到 O(K · seq_len / K)\n";
    std::cout << "  - 典型实现: 每 K 步截断一次梯度流，保留隐状态继续\n";
    std::cout << "  - 结合梯度裁剪: 进一步防止梯度爆炸\n";
    std::cout << "══════════════════════════════════\n";

    return 0;
}
