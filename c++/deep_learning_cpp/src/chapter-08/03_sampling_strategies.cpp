/*
 * 第 8 章：生成网络、自编码器与大语言模型
 * 第 290-298 页：文本生成的采样策略 (Sampling Strategies)
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <iomanip>
#include <string>

/* ============================================================================
 * 第 8 章：生成网络、自编码器与大语言模型
 * 简单词汇表 + 特殊令牌
 * ============================================================================
 */
static const std::vector<std::string> VOCABULARY = {
    /* 0 */ "，", /* 1 */ "。",
    /* 2 */ "<EOS>", /* 3 */ "<UNK>",
    /* 4 */ "猫", /* 5 */ "狗",
    /* 6 */ "坐", /* 7 */ "跑",
    /* 8 */ "在", /* 9 */ "上",
    /* 10 */ "垫子", /* 11 */ "地板",
    /* 12 */ "桌子", /* 13 */ "椅子",
    /* 14 */ "大", /* 15 */ "小",
    /* 16 */ "漂亮", /* 17 */ "柔软",
    /* 18 */ "棕色", /* 19 */ "白色",
    /* 20 */ "黑色", /* 21 */ "红色"};

// 特殊令牌索引
const int EOS_TOKEN_ID = 2;
const int UNK_TOKEN_ID = 3;

/* ============================================================================
 * 第 8 章：生成网络、自编码器与大语言模型
 * 辅助函数：Softmax —— exp(x_i) / Σ exp(x_j)
 * ============================================================================
 */
std::vector<double> softmax(const std::vector<double> &logits) {
    std::vector<double> probs(logits.size());

    // 数值稳定性：减去最大值，防止 exp 溢出
    double max_val = *std::max_element(logits.begin(), logits.end());
    double sum_exp = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_val);
        sum_exp += probs[i];
    }

    for (size_t i = 0; i < probs.size(); ++i) {
        probs[i] /= sum_exp;
    }
    return probs;
}

/* ============================================================================
 * 第 8 章：生成网络、自编码器与大语言模型
 * 温度缩放：logits / temperature，再 softmax
 * 返回值：缩放后的概率分布
 * temp → 0：趋近贪婪（概率集中）
 * temp = 1：原始分布
 * temp → ∞：趋近均匀分布
 * ============================================================================
 */
std::vector<double> apply_temperature(const std::vector<double> &logits, double temp) {
    std::vector<double> scaled(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        scaled[i] = logits[i] / temp;
    }
    return softmax(scaled);
}

/* ============================================================================
 * 第 8 章：生成网络、自编码器与大语言模型
 * 策略 1：贪婪解码 (Greedy Decoding)
 * 每步始终选择概率最高的令牌，输出完全确定
 * 示例："猫坐 在 ___" → 始终预测 "垫子"
 * ============================================================================
 */
int greedy_decode(const std::vector<double> &probs) {
    return std::distance(probs.begin(),
                         std::max_element(probs.begin(), probs.end()));
}

/* ============================================================================
 * 第 8 章：生成网络、自编码器与大语言模型
 * 策略 2：Top-K 采样 (Top-K Sampling)
 * 只保留概率最高的 k 个令牌，重新归一化后按多项式分布采样
 * 较大的 k 增加多样性，较小的 k 增加确定性
 * ============================================================================
 */
std::vector<int> top_k_candidates(const std::vector<double> &probs, int k) {
    // 创建 (索引, 概率) 对，按概率降序排序
    std::vector<std::pair<int, double>> indexed;
    for (size_t i = 0; i < probs.size(); ++i) {
        indexed.emplace_back(static_cast<int>(i), probs[i]);
    }
    std::sort(indexed.begin(), indexed.end(),
              [](const auto &a, const auto &b) { return a.second > b.second; });

    // 取前 k 个
    std::vector<int> result;
    for (int i = 0; i < k && i < static_cast<int>(indexed.size()); ++i) {
        result.push_back(indexed[i].first);
    }
    return result;
}

std::vector<std::pair<int, double>> top_k_distribution(
    const std::vector<double> &probs, int k) {
    std::vector<std::pair<int, double>> indexed;
    for (size_t i = 0; i < probs.size(); ++i) {
        indexed.emplace_back(static_cast<int>(i), probs[i]);
    }
    std::sort(indexed.begin(), indexed.end(),
              [](const auto &a, const auto &b) { return a.second > b.second; });

    // 取前 k 个，重新归一化
    std::vector<std::pair<int, double>> result;
    double sum = 0.0;
    for (int i = 0; i < k && i < static_cast<int>(indexed.size()); ++i) {
        result.push_back(indexed[i]);
        sum += indexed[i].second;
    }
    for (auto &p : result) {
        p.second /= sum;
    }
    return result;
}

int top_k_sample(const std::vector<double> &probs, int k,
                 std::mt19937 &rng) {
    auto dist = top_k_distribution(probs, k);

    // 多项式采样
    std::vector<double> weights;
    for (auto &p : dist) weights.push_back(p.second);
    std::discrete_distribution<int> multinomial(weights.begin(), weights.end());
    int sampled_idx = multinomial(rng);
    return dist[sampled_idx].first;
}

/* ============================================================================
 * 第 8 章：生成网络、自编码器与大语言模型
 * 策略 3：Top-P / 核采样 (Nucleus Sampling)
 * 从高到低累积概率，保留累积概率超过 p 的最小令牌集合
 * p = 0.9 表示"我们只考虑能覆盖 90% 概率质量的那些令牌"
 * 然后重新归一化 → 采样
 * ============================================================================
 */
std::vector<std::pair<int, double>> top_p_distribution(
    const std::vector<double> &probs, double p) {
    std::vector<std::pair<int, double>> indexed;
    for (size_t i = 0; i < probs.size(); ++i) {
        indexed.emplace_back(static_cast<int>(i), probs[i]);
    }
    // 按概率降序排序
    std::sort(indexed.begin(), indexed.end(),
              [](const auto &a, const auto &b) { return a.second > b.second; });

    std::vector<std::pair<int, double>> result;
    double cumsum = 0.0;
    for (auto &item : indexed) {
        cumsum += item.second;
        result.push_back(item);
        if (cumsum >= p) break; // 达到累积阈值，停止
    }

    // 重新归一化
    double sum = 0.0;
    for (auto &item : result) sum += item.second;
    for (auto &item : result) item.second /= sum;

    return result;
}

std::vector<int> top_p_candidates(const std::vector<double> &probs, double p) {
    auto dist = top_p_distribution(probs, p);
    std::vector<int> result;
    for (auto &item : dist) result.push_back(item.first);
    return result;
}

int top_p_sample(const std::vector<double> &probs, double p,
                 std::mt19937 &rng) {
    auto dist = top_p_distribution(probs, p);
    std::vector<double> weights;
    for (auto &item : dist) weights.push_back(item.second);
    std::discrete_distribution<int> multinomial(weights.begin(), weights.end());
    int sampled_idx = multinomial(rng);
    return dist[sampled_idx].first;
}

/* ============================================================================
 * 第 8 章：生成网络、自编码器与大语言模型
 * 策略 4：束搜索 (Beam Search) —— 概念级说明
 * 束搜索维护 beam_width 个最可能的候选序列，而非只保留一个
 * 在每个时间步，对每个候选的所有可能扩展计算分数，再次只保留 beam_width 个最优序列
 *
 * 算法伪代码：
 *   beam_width = 3
 *   sequences = [([], 0.0)]  // (令牌序列, 对数概率和)
 *   for each time_step:
 *       candidates = []
 *       for each (seq, score) in sequences:
 *           if seq 最后是 EOS: candidates.append((seq, score))  // 已完成
 *           else:
 *               for each token t with prob > 0:
 *                   new_score = score + log(prob[t])
 *                   candidates.append((seq + [t], new_score))
 *       // 保留得分最高的 beam_width 个候选
 *       sequences = top_k(candidates, beam_width)
 *
 * 注意：完整的束搜索需要模型来提供每一步的 logits，
 * 因此在纯 STL 环境中无法完整实现。以上为算法说明。
 * ============================================================================
 */

/* ============================================================================
 * 第 8 章：生成网络、自编码器与大语言模型
 * 终止条件说明
 * ============================================================================
 */
void print_termination_conditions() {
    std::cout << "\n╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║  终止条件 (Termination Conditions)                    ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n";
    std::cout << "  ① EOS 令牌：模型预测出 <EOS>，自然停止生成\n";
    std::cout << "  ② max_length：达到预设的最大序列长度，强制截断\n";
    std::cout << "  ③ max_new_tokens：新生成的令牌数达到上限\n";
    std::cout << "  ④ 早停 (Early Stopping)：生成质量不再提升时提前终止\n";
}

/* ============================================================================
 * 第 8 章：生成网络、自编码器与大语言模型
 * 打印表格行
 * ============================================================================
 */
void print_table_row(double temp,
                     const std::vector<double> &probs,
                     int greedy_choice,
                     const std::vector<int> &topk_list,
                     const std::vector<int> &topp_list) {
    // 温度列
    std::cout << std::setw(6) << std::fixed << std::setprecision(1)
              << temp << " | ";

    // 贪婪列：概率最高的那个令牌 p(token|context)
    double greedy_prob = probs[greedy_choice];
    std::cout << std::setw(8) << VOCABULARY[greedy_choice]
              << "(" << std::fixed << std::setprecision(2) << greedy_prob << ") | ";

    // Top-K 列：列出前 5 个候选
    for (size_t i = 0; i < topk_list.size() && i < 5; ++i) {
        if (i > 0) std::cout << " ";
        std::cout << VOCABULARY[topk_list[i]];
    }
    for (size_t i = topk_list.size(); i < 5; ++i) {
        std::cout << "     "; // 填充空白
    }
    std::cout << " | ";

    // Top-P 列：落入核内的令牌
    for (size_t i = 0; i < topp_list.size(); ++i) {
        if (i > 0) std::cout << " ";
        std::cout << VOCABULARY[topp_list[i]];
    }
    std::cout << "\n";
}

/* ============================================================================
 * 第 8 章：生成网络、自编码器与大语言模型
 * 主函数
 * ============================================================================
 */
int main() {
    std::random_device rd;
    std::mt19937 rng(rd());

    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║  第 8 章：文本生成采样策略演示                        ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n";

    // -----------------------------------------------------------------
    // 构造示例 logits：“猫 坐 在 ___” 的下一步预测分布
    // 词汇表索引映射见顶部 VOCABULARY
    // 我们希望模型倾向于预测 "垫子"(10), "地板"(11), "桌子"(12), "椅子"(13) 等
    // -----------------------------------------------------------------
    std::cout << "\n上下文：「猫 坐 在 ___」\n\n";

    std::vector<double> logits(VOCABULARY.size(), 0.01);
    logits[10] = 3.5;  // "垫子" — 最高概率
    logits[11] = 2.8;  // "地板"
    logits[12] = 2.0;  // "桌子"
    logits[13] = 1.5;  // "椅子"
    logits[9] = 1.0;   // "上"
    logits[4] = 0.8;   // "猫"
    logits[5] = 0.6;   // "狗"
    logits[14] = 0.4;  // "大"
    logits[15] = 0.3;  // "小"
    logits[17] = 0.2;  // "柔软"
    logits[18] = 0.15; // "棕色"
    logits[2] = 0.1;   // "<EOS>"
    logits[0] = 0.05;  // "，"
    logits[1] = 0.05;  // "。"

    // === 演示 softmax ===
    std::cout << "【Softmax 转换示例】\n";
    auto base_probs = softmax(logits);
    std::cout << "  logits[\"垫子\"] = " << logits[10]
              << " → softmax 概率 = " << std::fixed << std::setprecision(4)
              << base_probs[10] << "\n";
    std::cout << "  logits[\"地板\"] = " << logits[11]
              << " → softmax 概率 = " << std::fixed << std::setprecision(4)
              << base_probs[11] << "\n\n";

    // === 打印表格标题 ===
    std::cout << "┌──────┬────────────────────┬─────────────────────────────┬────────────────────────────┐\n";
    std::cout << "│ 温度 │ 贪婪(Greedy)        │  Top-K 候选 (k=5)           │  Top-P 候选 (p=0.9)         │\n";
    std::cout << "├──────┼────────────────────┼─────────────────────────────┼────────────────────────────┤\n";

    // === 对不同温度进行实验 ===
    std::vector<double> temperatures = {0.1, 0.5, 1.0, 2.0};
    for (double temp : temperatures) {
        auto temp_probs = apply_temperature(logits, temp);

        // 贪婪
        int greedy = greedy_decode(temp_probs);

        // Top-K
        auto topk = top_k_candidates(temp_probs, 5);

        // Top-P
        auto topp = top_p_candidates(temp_probs, 0.9);

        print_table_row(temp, temp_probs, greedy, topk, topp);
    }
    std::cout << "└──────┴────────────────────┴─────────────────────────────┴────────────────────────────┘\n";

    // === 温度影响说明 ===
    std::cout << "\n【温度分析】\n";
    std::cout << "  temp=0.1：概率高度集中，≈ 贪婪 → 输出确定、可能重复\n";
    std::cout << "  temp=0.5：适度集中 → 质量与多样性的折中\n";
    std::cout << "  temp=1.0：原始分布 → 自然多样性\n";
    std::cout << "  temp=2.0：分布趋于均匀 → 高多样性、可能产生乱码\n";

    // === 策略对比采样示例 ===
    std::cout << "\n【多次采样对比】（温度=1.0，运行 5 次）\n";
    std::cout << "  贪婪:      ";
    for (int run = 0; run < 5; ++run) {
        std::cout << VOCABULARY[greedy_decode(base_probs)] << " ";
    }
    std::cout << " ← 始终相同（确定性）\n";

    std::cout << "  Top-K(5):  ";
    for (int run = 0; run < 5; ++run) {
        std::cout << VOCABULARY[top_k_sample(base_probs, 5, rng)] << " ";
    }
    std::cout << " ← k=5 有适度变化\n";

    std::cout << "  Top-P(.9): ";
    for (int run = 0; run < 5; ++run) {
        std::cout << VOCABULARY[top_p_sample(base_probs, 0.9, rng)] << " ";
    }
    std::cout << " ← 核内动态采样\n";

    // === 束搜索概念说明 ===
    std::cout << "\n【束搜索 (Beam Search) 概念】\n";
    std::cout << "  beam_width=3 时，每个时间步维护 3 条最佳候选路径。\n";
    std::cout << "  相比贪婪解码，束搜索能探索更多可能性，\n";
    std::cout << "  例如「猫 坐 在 垫子」vs「猫 坐 在 地板」二选一不会仅凭首步决定。\n";
    std::cout << "  代价：计算量 = 贪婪的 beam_width 倍（需要模型支持）。\n";

    // === 终止条件 ===
    print_termination_conditions();

    std::cout << "\n";
    return 0;
}
