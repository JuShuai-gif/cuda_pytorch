/*
 * 03_lstm_libtorch.cpp - 第 7 章：循环神经网络与 LSTM
 * 基于 LibTorch 的生产级 LSTM 实现（对应原书第 231-233 页）
 *
 * 演示内容：
 *   1. torch::nn::LSTM：多层 LSTM + Dropout + batch_first
 *   2. 输出投影层 + Dropout 正则化
 *   3. 混合精度训练概念（autocast FP16 注释示例）
 *   4. 模型参数统计与 state_dict 序列化 / 反序列化
 */

#include <torch/torch.h>
#include <iostream>
#include <iomanip>

/* =================== ProductionLSTM ============================= */
/*
 * 生产级 LSTM 分类/回归模型。
 *
 * 结构：
 *   输入 → 双层 Biased LSTM（带 dropout）→ Dropout → 线性输出投影
 *
 * 混合精度训练（Mixed Precision / AMP）概述：
 *   前向传播使用 FP16（torch::autocast），反向传播保持 FP32。
 *   FP16 tensor cores 吞吐量约为 FP32 的 2~8 倍（NVIDIA A100），
 *   同时显存占用减半。梯度缩放（GradScaler）防止小梯度下溢。
 *   用法示例（注释）：
 *     auto guard = torch::autocast(torch::kCUDA);
 *     auto output = model->forward(x);
 */
class ProductionLSTM : public torch::nn::Module {
public:
    /* ------- 构造函数 ----------- */
    ProductionLSTM(int input_size,  // 词表大小 / 输入特征数
                   int hidden_size, // LSTM 隐藏维度
                   int output_size = 1) {
        // 双层 LSTM：第一层学习局部模式，第二层学习序列级别抽象
        // num_layers=2 → 堆叠两层，每一层的输出作为下一层的输入
        // dropout=0.3  → 层间 Dropout（仅多层时有意义）
        // batch_first=true → 输入形状为 (batch, seq_len, input_size)
        lstm = register_module("lstm",
                               torch::nn::LSTM(
                                   torch::nn::LSTMOptions(input_size, hidden_size)
                                       .num_layers(2)        // 双层 LSTM
                                       .dropout(0.3)         // 层间正则化
                                       .batch_first(true))); // (B, T, D) 格式

        // Dropout 层：进一步正则化，防止过拟合
        dropout = register_module("dropout", torch::nn::Dropout(0.3));

        // 输出投影：将 LSTM 隐藏状态映射到目标维度
        output_projection = register_module("output_projection",
                                            torch::nn::Linear(hidden_size, output_size));
    }

    /* ------- 前向传播 ----------- */
    /*
     * 输入 x: (batch_size, seq_len, input_size)
     * 返回: (batch_size, seq_len, output_size) - 每个时间步的预测
     *
     * 若只需最后一个时间步输出，可在调用后 x.slice(1,-1) 截取。
     */
    torch::Tensor forward(torch::Tensor x) {
        // ① 通过双层 LSTM
        //    lstm(x) 返回 std::tuple<output, (h_n, c_n)>
        //      output: (batch, seq_len, hidden_size * num_directions)
        //      h_n:    (num_layers * num_directions, batch, hidden_size)
        //      c_n:    (num_layers * num_directions, batch, hidden_size)
        auto lstm_output = lstm->forward(x);
        auto out = std::get<0>(lstm_output); // 提取输出序列

        // ② Dropout 正则化（训练时生效，eval 模式下自动关闭）
        out = dropout->forward(out);

        // ③ 线性输出投影
        //    将每个时间步的隐藏向量映射到输出维度
        out = output_projection->forward(out);

        return out; // (batch, seq_len, output_size)
    }

private:
    torch::nn::LSTM lstm{nullptr};                // 核心 LSTM 模块
    torch::nn::Linear output_projection{nullptr}; // 输出投影
    torch::nn::Dropout dropout{nullptr};          // 正则化 dropout
};

/* =========================== 工具函数 ================================== */

/*
 * 统计模型中的可训练参数总数。
 * torch::nn::Module::parameters() 返回所有注册参数的列表。
 */
int64_t count_parameters(const torch::nn::Module &model) {
    int64_t total = 0;
    for (const auto &param : model.parameters()) {
        total += param.numel(); // number of elements
    }
    return total;
}

/*
 * 打印参数的名称和形状。
 */
void print_parameters(const torch::nn::Module &model) {
    std::cout << "\n--- 可训练参数列表 ---\n";
    for (const auto &pair : model.named_parameters()) {
        std::cout << "  " << std::setw(40) << std::left << pair.key()
                  << " | shape: " << pair.value().sizes() << "\n";
    }
}

/* =========================== main ======================================== */
/*
 * 演示流程：
 *   1. 设备选择（CUDA / CPU）
 *   2. 创建 ProductionLSTM(10, 64) 模型
 *   3. 生成随机输入 → 前向传播 → 打印输入 / 输出形状
 *   4. 统计并打印可训练参数
 *   5. 序列化：torch::save → 反序列化：torch::load
 */
int main() {
    std::cout << "================================================================\n";
    std::cout << "  第 7 章：循环神经网络与 LSTM - LibTorch 生产级实现\n";
    std::cout << "================================================================\n\n";

    // ① 设备选择
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        std::cout << "设备: CUDA (GPU)\n\n";
    } else {
        std::cout << "设备: CPU\n\n";
    }

    // ② 创建模型
    const int vocab_size = 10;  // 词表大小 / 输入特征数
    const int hidden_size = 64; // LSTM 隐藏维度
    const int batch_size = 2;   // 批次大小
    const int seq_len = 5;      // 序列长度

    ProductionLSTM model(vocab_size, hidden_size);
    model->to(device); // 将模型参数移动到目标设备

    std::cout << "模型结构:\n"
              << "  输入维度 : " << vocab_size << "\n"
              << "  隐藏维度 : " << hidden_size << "\n"
              << "  LSTM 层数: 2（带层间 dropout=0.3）\n"
              << "  输出投影 : Linear(" << hidden_size << " → 1)\n\n";

    // ③ 生成随机序列输入
    //    形状: (batch_size, seq_len, input_size)
    //    模拟 2 个句子，每个句子 5 个词，词向量维度 10
    auto input = torch::randn({batch_size, seq_len, vocab_size}).to(device);

    std::cout << "输入张量形状: " << input.sizes() << "\n";

    // 前向传播
    model->eval();              // 切换到评估模式（关闭 dropout）
    torch::NoGradGuard no_grad; // 禁用梯度计算

    auto output = model->forward(input);

    std::cout << "输出张量形状: " << output.sizes() << "\n";
    std::cout << "  含义: (batch=" << batch_size
              << ", seq_len=" << seq_len
              << ", output_size=1) — 每个时间步的标量预测\n";

    // ④ 参数统计
    int64_t total_params = count_parameters(model);
    std::cout << "\n可训练参数总数: " << total_params << "\n";
    print_parameters(model);

    // ⑤ 序列化与反序列化
    std::cout << "\n--- 模型序列化 ---\n";

    // 保存模型（state_dict 方式）
    torch::save(model, "lstm_model.pt");
    std::cout << "模型已保存至: lstm_model.pt\n";

    // 完整序列化也可用 torch::jit::script::save(model, "model.pt")
    // 此处使用 state_dict 方式，便于跨版本加载

    // 加载模型
    ProductionLSTM loaded_model(vocab_size, hidden_size);
    torch::load(loaded_model, "lstm_model.pt");
    loaded_model->to(device);
    loaded_model->eval();
    std::cout << "模型已从 lstm_model.pt 加载\n";

    // 验证加载后的前向传播
    auto loaded_output = loaded_model.forward(input);
    bool close = torch::allclose(output, loaded_output);
    std::cout << "加载后输出一致性检查: "
              << (close ? "通过 ✓" : "失败 ✗") << "\n";

    std::cout << "\n================================\n";
    std::cout << "  关键概念回顾\n";
    std::cout << "================================\n";
    std::cout << "1. 双层 LSTM: 第一层提取局部特征，第二层建模序列级依赖\n";
    std::cout << "2. 层间 Dropout(0.3): 仅在多层时生效，防止层间过拟合\n";
    std::cout << "3. batch_first=True: 输入形状 (B,T,D)，与 Transformer 一致\n";
    std::cout << "4. 混合精度(AMP): FP16 前向/反向 ≈2× 速度，显存减半\n";
    std::cout << "5. state_dict 序列化: torch::save/load 保存/恢复参数\n";

    return 0;
}
