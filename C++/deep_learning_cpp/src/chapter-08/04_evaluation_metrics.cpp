/*
 * 第 8 章：生成网络、自编码器与大语言模型
 * 第 300-306 页：文本评估指标与分析 (Evaluation Metrics)
 */

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <unordered_map>
#include <set>
#include <cmath>
#include <iomanip>
#include <algorithm>

/* ============================================================================
 * 第 8 章：生成网络、自编码器与大语言模型
 * 辅助工具：将句子切分为单词（简单按空格分词）
 * ============================================================================
 */
std::vector<std::string> tokenize(const std::string &text) {
    std::vector<std::string> tokens;
    std::istringstream iss(text);
    std::string word;
    while (iss >> word) {
        // 去除简单标点
        std::string clean;
        for (char c : word) {
            if (std::isalpha(static_cast<unsigned char>(c)) || std::isdigit(static_cast<unsigned char>(c))) {
                clean += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            }
        }
        if (!clean.empty()) tokens.push_back(clean);
    }
    return tokens;
}

/* ============================================================================
 * 第 8 章：生成网络、自编码器与大语言模型
 * 构造 n-gram
 * ============================================================================
 */
std::vector<std::string> make_ngrams(const std::vector<std::string> &tokens, int n) {
    std::vector<std::string> ngrams;
    if (n <= 0 || tokens.size() < static_cast<size_t>(n)) return ngrams;
    for (size_t i = 0; i <= tokens.size() - n; ++i) {
        std::string gram;
        for (int j = 0; j < n; ++j) {
            if (j > 0) gram += "_";
            gram += tokens[i + j];
        }
        ngrams.push_back(gram);
    }
    return ngrams;
}

/* ============================================================================
 * 第 8 章：生成网络、自编码器与大语言模型
 * 第 1 节：TF-IDF 演示
 * TF(词项, 文档) = 词项在文档中的出现次数 / 文档总词数
 * IDF(词项) = log(总文档数 / 包含该词项的文档数)
 * TF-IDF = TF × IDF
 * ============================================================================
 */
void section1_tfidf() {
    std::cout << "\n┌─────────────────────────────────────────────────────┐\n";
    std::cout << "│  第 1 节：TF-IDF 演示                                 │\n";
    std::cout << "└─────────────────────────────────────────────────────┘\n\n";

    // 4 个小型文档
    std::vector<std::string> documents = {
        "猫 坐 在 垫子 上",
        "狗 跑 在 地板 上",
        "猫 和 狗 都 是 动物",
        "垫子 非常 柔软 猫 喜欢 坐 在 上面"};

    // 分词
    std::vector<std::vector<std::string>> tokenized_docs;
    for (const auto &doc : documents) {
        tokenized_docs.push_back(tokenize(doc));
    }

    // 打印文档
    std::cout << "文档集合：\n";
    for (size_t i = 0; i < documents.size(); ++i) {
        std::cout << "  Doc" << (i + 1) << ": \"" << documents[i] << "\"\n";
    }

    // 收集所有唯一词项
    std::set<std::string> all_terms;
    for (const auto &tokens : tokenized_docs) {
        for (const auto &t : tokens) all_terms.insert(t);
    }

    size_t total_docs = tokenized_docs.size();

    // 计算 TF、IDF、TF-IDF 并打印表格
    std::cout << "\n┌──────────┬─────────────────────────────────────────────────────┐\n";
    std::cout << "│ 词项     │  Doc1        Doc2        Doc3        Doc4        IDF   │\n";
    std::cout << "│          │  TF  TF-IDF  TF  TF-IDF  TF  TF-IDF  TF  TF-IDF       │\n";
    std::cout << "├──────────┼─────────────────────────────────────────────────────┤\n";

    for (const auto &term : all_terms) {
        // 计算 IDF
        int docs_containing = 0;
        for (const auto &tokens : tokenized_docs) {
            if (std::find(tokens.begin(), tokens.end(), term) != tokens.end()) {
                ++docs_containing;
            }
        }
        double idf = std::log(static_cast<double>(total_docs) / docs_containing);

        // 打印该行
        // 中文字符显示宽度问题，用固定宽度尽量对齐
        std::ostringstream line;
        line << "│ " << std::left << std::setw(8) << term << "│";

        for (size_t d = 0; d < tokenized_docs.size(); ++d) {
            const auto &tokens = tokenized_docs[d];
            double doc_len = static_cast<double>(tokens.size());
            int term_count = static_cast<int>(
                std::count(tokens.begin(), tokens.end(), term));
            double tf = (doc_len > 0) ? term_count / doc_len : 0.0;
            double tfidf = tf * idf;

            line << " " << std::fixed << std::setprecision(2) << std::setw(4) << tf
                 << " " << std::setw(6) << tfidf;
        }

        line << " " << std::fixed << std::setprecision(2) << std::setw(5) << idf;
        line << " │";
        std::cout << line.str() << "\n";
    }
    std::cout << "└──────────┴─────────────────────────────────────────────────────┘\n";

    std::cout << "\n解释：\n";
    std::cout << "  TF 高 + IDF 高 = 该词对该文档很重要，在全局又少见\n";
    std::cout << "  TF 高 + IDF 低 = 该词是常见词（如\"在\"），区分度不强\n";
}

/* ============================================================================
 * 第 8 章：生成网络、自编码器与大语言模型
 * 第 2 节：N-gram 演示
 * 一元语法 (unigram) —— 单个词
 * 二元语法 (bigram)  —— 连续两个词
 * 三元语法 (trigram) —— 连续三个词
 * ============================================================================
 */
void section2_ngrams() {
    std::cout << "\n┌─────────────────────────────────────────────────────┐\n";
    std::cout << "│  第 2 节：N-gram 演示                                  │\n";
    std::cout << "└─────────────────────────────────────────────────────┘\n\n";

    std::string sentence = "The wind is blowing";
    std::cout << "原句: \"" << sentence << "\"\n\n";

    auto tokens = tokenize(sentence);

    // 打印分词结果
    std::cout << "分词: ";
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) std::cout << " | ";
        std::cout << tokens[i];
    }
    std::cout << "\n\n";

    // 一元语法
    auto unigrams = make_ngrams(tokens, 1);
    std::cout << "一元语法 (Unigrams):  ";
    for (size_t i = 0; i < unigrams.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << "\"" << unigrams[i] << "\"";
    }
    std::cout << "  共 " << unigrams.size() << " 个\n";

    // 二元语法
    auto bigrams = make_ngrams(tokens, 2);
    std::cout << "二元语法 (Bigrams):   ";
    for (size_t i = 0; i < bigrams.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << "\"" << bigrams[i] << "\"";
    }
    std::cout << "  共 " << bigrams.size() << " 个\n";

    // 三元语法
    auto trigrams = make_ngrams(tokens, 3);
    std::cout << "三元语法 (Trigrams):  ";
    for (size_t i = 0; i < trigrams.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << "\"" << trigrams[i] << "\"";
    }
    std::cout << "  共 " << trigrams.size() << " 个\n";

    std::cout << "\nN-gram 用于：\n";
    std::cout << "  • 语言模型评估（统计 n-gram 频率）\n";
    std::cout << "  • BLEU/ROUGE 自动评估指标的基础\n";
    std::cout << "  • 文本特征工程\n";
}

/* ============================================================================
 * 第 8 章：生成网络、自编码器与大语言模型
 * 第 3 节：困惑度 (Perplexity) —— 概念演示
 * PPL = exp( - (1/N) × Σ log P(w_i | w_{1:i-1}) )
 * 其中 N 是总令牌数，P(w_i|...) 是模型赋予正确令牌的概率
 *
 * 困惑度 = 1    → 模型完全确定（完美预测）
 * 困惑度 = 词表大小 → 模型完全随机猜测
 * 困惑度 = 15   → 高度不确定（相当于每个位置有 15 个等可能选项）
 * ============================================================================
 */
void section3_perplexity() {
    std::cout << "\n┌─────────────────────────────────────────────────────┐\n";
    std::cout << "│  第 3 节：困惑度 (Perplexity) —— 概念演示               │\n";
    std::cout << "└─────────────────────────────────────────────────────┘\n\n";

    std::cout << "公式：PPL = exp( -1/N × Σ_{i=1}^{N} log P(w_i | w_{1:i-1}) )\n\n";

    // 构造一个示例：假设序列长度为 5，模型给出的条件概率
    std::vector<double> cond_probs = {0.9, 0.8, 0.7, 0.85, 0.6};
    double sum_log = 0.0;
    for (double p : cond_probs) {
        sum_log += std::log(p);
    }
    double N = static_cast<double>(cond_probs.size());
    double ppl = std::exp(-sum_log / N);

    std::cout << "示例计算：\n";
    std::cout << "  序列 \"猫 坐 在 垫子 上\" 的每步条件概率：\n";
    std::cout << "    P(猫) = 0.9, P(坐|猫) = 0.8, P(在|猫坐) = 0.7,\n";
    std::cout << "    P(垫子|猫坐在) = 0.85, P(上|猫坐在垫子) = 0.6\n";
    std::cout << "  Σ log P = ";
    for (size_t i = 0; i < cond_probs.size(); ++i) {
        if (i > 0) std::cout << " + ";
        std::cout << "log(" << cond_probs[i] << ")";
    }
    std::cout << " = " << std::fixed << std::setprecision(4) << sum_log << "\n";
    std::cout << "  困惑度 = exp( -(" << sum_log << ") / " << static_cast<int>(N) << " )"
              << " = " << std::fixed << std::setprecision(4) << ppl << "\n\n";

    std::cout << "====================\n";
    std::cout << "困惑度 = 1 → 表示完美预测，模型完全确定下一词\n";
    std::cout << "困惑度 = 15 → 表示高度不确定，模型对预测没有把握\n";
    std::cout << "困惑度越低，模型对测试数据的拟合越好\n";
    std::cout << "注意：在解码时人的偏好更关注生成质量而非原始困惑度\n";
}

/* ============================================================================
 * 第 8 章：生成网络、自编码器与大语言模型
 * 第 4 节：BLEU 分数 —— 概念 + 简化版 n-gram 精度
 * 参考译文 reference = "the cat is on the mat"
 * 候选译文 candidate  = "the cat sat on the mat"
 *
 * 步骤：
 *   1. 计算 unigram 精度：匹配的 unigram 数 / 候选 unigram 总数
 *   2. 简短惩罚 (Brevity Penalty, BP)：
 *      如果候选长度 < 参考长度 → BP = exp(1 - ref_len / cand_len)
 *      否则 → BP = 1
 *   3. 简化 BLEU = BP × 精度
 * ============================================================================
 */
void section4_bleu() {
    std::cout << "\n┌─────────────────────────────────────────────────────┐\n";
    std::cout << "│  第 4 节：BLEU 分数 —— 概念 + 简化版演示                 │\n";
    std::cout << "└─────────────────────────────────────────────────────┘\n\n";

    std::string reference = "the cat is on the mat";
    std::string candidate = "the cat sat on the mat";

    auto ref_tokens = tokenize(reference);
    auto cand_tokens = tokenize(candidate);

    std::cout << "参考: \"" << reference << "\"\n";
    std::cout << "候选: \"" << candidate << "\"\n\n";

    // ---- Unigram 精度 ----
    // 计算每个参考 unigram 最多能被匹配的次数（防止多匹配）
    std::unordered_map<std::string, int> ref_unigram_counts;
    for (const auto &t : ref_tokens) ref_unigram_counts[t]++;

    std::unordered_map<std::string, int> cand_unigram_counts;
    for (const auto &t : cand_tokens) cand_unigram_counts[t]++;

    int matched = 0;
    for (const auto &kv : cand_unigram_counts) {
        auto it = ref_unigram_counts.find(kv.first);
        if (it != ref_unigram_counts.end()) {
            matched += std::min(kv.second, it->second); // 截断计数
        }
    }

    double total_cand = static_cast<double>(cand_tokens.size());
    double precision = (total_cand > 0) ? matched / total_cand : 0.0;

    std::cout << "匹配的 unigram 数: " << matched
              << " / 候选 unigram 数: " << static_cast<int>(total_cand) << "\n";
    std::cout << "Unigram 精度 = " << std::fixed << std::setprecision(4)
              << precision << "\n";

    // ---- 简短惩罚 ----
    double ref_len = static_cast<double>(ref_tokens.size());
    double cand_len = static_cast<double>(cand_tokens.size());
    double bp = (cand_len < ref_len) ? std::exp(1.0 - ref_len / cand_len) : 1.0;

    std::cout << "参考长度 = " << static_cast<int>(ref_len)
              << ", 候选长度 = " << static_cast<int>(cand_len)
              << " → BP = " << std::fixed << std::setprecision(4) << bp << "\n";

    // ---- 简化 BLEU ----
    double bleu = bp * precision;
    std::cout << "简化 BLEU = BP × 精度 = " << std::fixed << std::setprecision(4)
              << bleu << "\n";

    std::cout << "\n完整 BLEU 会使用 1-gram 到 4-gram 的加权几何平均，这里仅演示原理。\n";
}

/* ============================================================================
 * 第 8 章：生成网络、自编码器与大语言模型
 * 第 5 节：ROUGE-1 召回率 —— 概念
 * Recall = 匹配的 unigram 数 / 参考 unigram 总数
 * F1     = 2 × P × R / (P + R)
 * ============================================================================
 */
void section5_rouge() {
    std::cout << "\n┌─────────────────────────────────────────────────────┐\n";
    std::cout << "│  第 5 节：ROUGE-1 召回率 —— 概念演示                     │\n";
    std::cout << "└─────────────────────────────────────────────────────┘\n\n";

    std::string reference = "the cat is on the mat";
    std::string candidate = "the cat sat on the mat";

    auto ref_tokens = tokenize(reference);
    auto cand_tokens = tokenize(candidate);

    std::cout << "参考: \"" << reference << "\"\n";
    std::cout << "候选: \"" << candidate << "\"\n\n";

    // 使用集合计算
    std::set<std::string> ref_set(ref_tokens.begin(), ref_tokens.end());
    std::set<std::string> cand_set(cand_tokens.begin(), cand_tokens.end());

    // 求交集（简化：按 set 计算）
    std::vector<std::string> intersection;
    std::set_intersection(ref_set.begin(), ref_set.end(),
                          cand_set.begin(), cand_set.end(),
                          std::back_inserter(intersection));
    int intersect_size = static_cast<int>(intersection.size());

    double P = static_cast<double>(intersect_size) / cand_tokens.size();
    double R = static_cast<double>(intersect_size) / ref_tokens.size();

    std::cout << "匹配 unigram 数（按集合交集）: " << intersect_size << "\n";
    std::cout << "精度 (Precision) = " << intersect_size << " / "
              << cand_tokens.size() << " = " << std::fixed
              << std::setprecision(4) << P << "\n";
    std::cout << "召回 (Recall)   = " << intersect_size << " / "
              << ref_tokens.size() << " = " << std::fixed
              << std::setprecision(4) << R << "\n";

    double f1 = (P + R > 0) ? 2.0 * P * R / (P + R) : 0.0;
    std::cout << "F1 分数 = 2 × P × R / (P + R) = " << std::fixed
              << std::setprecision(4) << f1 << "\n";

    std::cout << "\nROUGE 一般用于评估摘要/翻译生成质量\n";
    std::cout << "ROUGE-N 看 n-gram 重叠 → 信息覆盖\n";
    std::cout << "ROUGE-L 看最长公共子序列 → 语序保持\n";
}

/* ============================================================================
 * 第 8 章：生成网络、自编码器与大语言模型
 * 评估维度一览表
 * ============================================================================
 */
void print_evaluation_dimensions() {
    std::cout << "\n┌─────────────────────────────────────────────────────┐\n";
    std::cout << "│  LLM 输出评估维度 (Evaluation Dimensions)              │\n";
    std::cout << "├─────────────────────────────────────────────────────┤\n";
    std::cout << "│  ① 准确性 (Accuracy)   │ 生成内容的事实正确性          │\n";
    std::cout << "│  ② 相关性 (Relevance)  │ 输出与用户意图/查询的匹配度    │\n";
    std::cout << "│  ③ 帮助性 (Helpfulness)│ 对用户目标的有用程度与完整度    │\n";
    std::cout << "│  ④ 安全性 (Safety)     │ 拒绝有害请求、不生成不当内容    │\n";
    std::cout << "│  ⑤ 效率 (Efficiency)   │ 推理速度、内存占用、吞吐量       │\n";
    std::cout << "│  ⑥ 鲁棒性 (Robustness) │ 对噪声输入、对抗样本的稳定性    │\n";
    std::cout << "├─────────────────────────────────────────────────────┤\n";
    std::cout << "│  自动指标：BLEU, ROUGE, METEOR, BERTScore, Perplexity│\n";
    std::cout << "│  人工评估：A/B 测试、Likert 量表、ELO 排名              │\n";
    std::cout << "└─────────────────────────────────────────────────────┘\n";
}

/* ============================================================================
 * 第 8 章：生成网络、自编码器与大语言模型
 * 主函数
 * ============================================================================
 */
int main() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║  第 8 章：文本评估指标与分析演示                        ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n";

    section1_tfidf();
    section2_ngrams();
    section3_perplexity();
    section4_bleu();
    section5_rouge();
    print_evaluation_dimensions();

    std::cout << "\n";
    return 0;
}
