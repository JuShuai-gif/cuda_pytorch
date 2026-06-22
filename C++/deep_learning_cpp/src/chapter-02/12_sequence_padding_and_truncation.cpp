/*
 * sequence_padding_and_truncation.cpp
 * 第2章：C++ 中的数据准备与预处理
 *
 * 神经网络期望固定大小的输入，但现实世界中的序列
 *（文本、时间序列、音频）具有可变长度。填充和截断
 * 用于在批次内标准化序列长度。
 *
 * 技术：
 *   - 填充（Padding）：在较短序列末尾追加特殊 token（通常为 0），
 *     以匹配批次的最大长度或预定义的 max_len。
 *   - 截断（Truncation）：将超过 max_len 的序列截断。选择策略：
 *       * 头部截断（Head truncation）：保留前 max_len 个 token
 *       * 尾部截断（Tail truncation）：保留后 max_len 个 token（文本常用）
 *   - 注意力掩码（Attention mask）：一个二值掩码（1 表示真实 token，0 表示填充），
 *     告诉模型忽略填充位置。对于 Transformer 至关重要，
 *     以防止对填充位置进行注意力计算。
 *
 * 何时使用：
 *   - Transformer 模型（BERT、GPT）：max_len 为 512/1024/2048 token
 *   - RNN/LSTM：使用 pack_padded_sequence 提高效率（在计算中跳过填充）
 *   - 时间序列：对齐不同长度的序列以进行批次处理
 */

#include <iostream>
#include <vector>
#include <algorithm>

// ----------------------------------------------------------------
// 将序列填充或截断到固定长度。
// 如果序列长度超过 max_len：截断保留前 max_len 个 token。
// 如果序列较短：用 pad_value 填充以达到 max_len。
//
// 对于文本，pad_value 通常为 [PAD] token ID（0）。
// 对于时间序列，pad_value 通常为 0.0 或 NaN（配合掩码使用）。
// ----------------------------------------------------------------
template <class T>
std::vector<T> padOrTruncate(
    const std::vector<T> &ids, size_t max_len, T pad) {
    if (ids.size() >= max_len)
        return {ids.begin(), ids.begin() + max_len};
    auto out = ids;
    out.resize(max_len, pad);
    return out;
}

// ----------------------------------------------------------------
// 创建注意力掩码：1 表示真实 token，0 表示填充位置。
// 这告诉 Transformer 模型应该对哪些位置进行注意力计算。
// 没有掩码的话，模型会将填充 token 视为有意义的 token。
// ----------------------------------------------------------------
std::vector<int64_t> makeMask(size_t len, size_t max_len) {
    std::vector<int64_t> m(std::min(len, max_len), 1);
    m.resize(max_len, 0);
    return m;
}

// ----------------------------------------------------------------
// 批次整理：将批次中所有序列填充到相同长度。
// 返回填充后的序列和对应的掩码。
// 生产环境中，使用动态填充（填充到批次最大长度而非固定的 max_len）
// 以获得更好的效率。
// ----------------------------------------------------------------
template <class T>
void collateBatch(
    const std::vector<std::vector<T>> &batch,
    std::vector<std::vector<T>> &padded,
    std::vector<std::vector<int64_t>> &masks,
    size_t max_len, T pad_val) {
    padded.resize(batch.size());
    masks.resize(batch.size());
    for (size_t i = 0; i < batch.size(); ++i) {
        padded[i] = padOrTruncate(batch[i], max_len, pad_val);
        masks[i] = makeMask(batch[i].size(), max_len);
    }
}

// 辅助函数
template <class T>
void printSeq(const std::string &label, const std::vector<T> &seq,
              const std::vector<int64_t> &mask,
              const std::string &padStr = "PAD") {
    std::cout << label << " [";
    for (size_t i = 0; i < seq.size(); ++i) {
        if (mask.empty() || mask[i] == 1)
            std::cout << seq[i];
        else
            std::cout << padStr;
        if (i + 1 < seq.size()) std::cout << ", ";
    }
    std::cout << "]";
    if (!mask.empty()) {
        std::cout << "  mask=[";
        for (size_t i = 0; i < mask.size(); ++i) {
            std::cout << mask[i];
            if (i + 1 < mask.size()) std::cout << ", ";
        }
        std::cout << "]";
    }
    std::cout << "\n";
}

int main() {
    std::cout << "=== Sequence Padding & Truncation Demo ===\n\n";

    const size_t MAX_LEN = 6;
    const int PAD_ID = 0;

    // 可变长度的 token 序列（例如，分词后的句子）
    std::vector<std::vector<int>> sequences = {
        {101, 2054, 2003, 1037, 3231, 1021, 102}, // 长度=7 -> 截断
        {101, 1996, 4823, 102},                   // 长度=4 -> 填充
        {1, 2, 3, 4, 5, 6},                       // 长度=6 -> 恰好
        {7, 8},                                   // 长度=2 -> 填充
    };

    std::cout << "最大长度：" << MAX_LEN
              << "，填充 token ID：" << PAD_ID << "\n\n";

    // 对每个序列进行填充和掩码
    std::vector<std::vector<int>> padded;
    std::vector<std::vector<int64_t>> masks;
    collateBatch(sequences, padded, masks, MAX_LEN, PAD_ID);

    for (size_t i = 0; i < sequences.size(); ++i) {
        printSeq("Seq " + std::to_string(i), padded[i], masks[i]);
    }

    std::cout << "\n--- 说明 ---\n";
    std::cout << "注意力掩码：1=真实 token（参与注意力计算），0=PAD（忽略）。\n";
    std::cout << "截断：对于文本，从头部保留（保留前几个 token）\n";
    std::cout << "  用于分类任务；从尾部保留用于生成任务。\n";
    std::cout << "填充侧：'post'（右侧）是标准做法；'pre'（左侧）用于\n";
    std::cout << "  某些自回归模型。\n";

    return 0;
}
