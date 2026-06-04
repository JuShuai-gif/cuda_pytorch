/*
 * 04_text_processing.cpp - 第 7 章：循环神经网络与 LSTM
 * 文本处理流水线演示（对应原书第 237-247 页）
 *
 * 演示内容：
 *   1. 大小写统一（lowercase）
 *   2. 标点符号去除（punctuation removal）
 *   3. 选择性字符保留（selective character removal）
 *   4. 句子切分（sentence splitting）
 *   5. 分词类型对比：词级 / 字符级 / 子词级
 *   6. BPE（Byte Pair Encoding）概念说明
 *   7. 停用词去除演示与注意事项
 *
 * 纯 STL 实现，无第三方依赖。
 */

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <regex>
#include <unordered_set>
#include <sstream>
#include <iomanip>

/* ======================== 1. 大小写转换 ======================== */

/*
 * 将输入文本统一转换为小写。
 * 使用 std::transform 搭配 ::tolower 实现。
 */
std::string to_lowercase(const std::string &text) {
    std::string result = text;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}

/* ======================== 2. 标点符号去除 ======================== */

/*
 * 去除文本中的标点符号：
 *   - 仅保留字母数字字符（isalnum）和空格（isspace）；
 *   - 使用正则将连续多个空格折叠为单个空格；
 *   - 去除首尾空白字符。
 */
std::string remove_punctuation(const std::string &text) {
    std::string cleaned;
    cleaned.reserve(text.size());

    // 保留字母、数字和空格，其余字符丢弃
    for (char c : text) {
        if (std::isalnum(static_cast<unsigned char>(c)) || std::isspace(static_cast<unsigned char>(c))) {
            cleaned += c;
        } else {
            cleaned += ' '; // 用空格替代标点，防止单词粘连
        }
    }

    // 将多个连续空格折叠为单个空格
    std::regex multi_space("\\s+");
    std::string collapsed = std::regex_replace(cleaned, multi_space, " ");

    // 去除首尾空白
    size_t start = collapsed.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    size_t end = collapsed.find_last_not_of(" \t\n\r");
    return collapsed.substr(start, end - start + 1);
}

/* ======================== 3. 选择性字符去除 ======================== */

/*
 * 根据用户指定的保留字符集去除标点。
 * keep_chars 中的字符会被额外保留（不会替换为空格）。
 */
std::string remove_selective_punctuation(
    const std::string &text,
    const std::unordered_set<char> &keep_chars) {
    std::string result;
    result.reserve(text.size());

    for (char c : text) {
        if (std::isalnum(static_cast<unsigned char>(c)) || std::isspace(static_cast<unsigned char>(c)) || keep_chars.find(c) != keep_chars.end()) {
            result += c;
        } else {
            result += ' ';
        }
    }

    // 折叠多余空格并去除首尾空白
    std::regex multi_space("\\s+");
    std::string collapsed = std::regex_replace(result, multi_space, " ");
    size_t start = collapsed.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    size_t end = collapsed.find_last_not_of(" \t\n\r");
    return collapsed.substr(start, end - start + 1);
}

/* ======================== 4. 句子切分 ======================== */

/*
 * 按句末标点（. ! ?）将文本切分为句子列表。
 */
std::vector<std::string> split_sentences(const std::string &text) {
    std::vector<std::string> sentences;
    std::string current;

    for (char c : text) {
        current += c;
        if (c == '.' || c == '!' || c == '?') {
            // 去除首尾空白
            size_t s = current.find_first_not_of(" \t\n\r");
            if (s != std::string::npos) {
                size_t e = current.find_last_not_of(" \t\n\r");
                sentences.push_back(current.substr(s, e - s + 1));
            }
            current.clear();
        }
    }

    // 处理最后一段（无终止标点时）
    if (!current.empty()) {
        size_t s = current.find_first_not_of(" \t\n\r");
        if (s != std::string::npos) {
            size_t e = current.find_last_not_of(" \t\n\r");
            sentences.push_back(current.substr(s, e - s + 1));
        }
    }

    return sentences;
}

/* ======================== 5. 分词类型演示 ======================== */

// 词语级分词：按空格切分
std::vector<std::string> word_tokenize(const std::string &text) {
    std::vector<std::string> tokens;
    std::istringstream iss(text);
    std::string word;
    while (iss >> word) {
        tokens.push_back(word);
    }
    return tokens;
}

// 字符级分词：将每个字符作为独立 token
std::vector<std::string> char_tokenize(const std::string &text) {
    std::vector<std::string> tokens;
    for (char c : text) {
        tokens.push_back(std::string(1, c));
    }
    return tokens;
}

// 辅助打印：向量元素用逗号分隔
void print_tokens(const std::string &label,
                  const std::vector<std::string> &tokens) {
    std::cout << "  " << label << " [";
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::cout << "\"" << tokens[i] << "\"";
        if (i + 1 < tokens.size()) std::cout << ",";
    }
    std::cout << "]" << std::endl;
}

/* ======================== 6. BPE 概念说明（注释） ======================== */

/*
 * BPE（Byte Pair Encoding，字节对编码）子词分词算法：
 *
 *   1. 初始化：将所有字符（或字节）作为基础词汇表。
 *   2. 迭代：在训练语料中统计所有相邻符号对的共现频率，
 *      找出出现频率最高的符号对。
 *   3. 合并：将最高频符号对合并为一个新的子词单元，
 *      加入词汇表。
 *   4. 重复步骤 2-3，直到达到预设的词汇表大小或最大迭代次数。
 *   5. 推理：对未见过的词，从字符开始逐步应用已学到的合并规则，
 *      将其拆分为已知的子词单元。
 *
 *   BPE 的优势：
 *     - 解决了 OOV（Out-Of-Vocabulary）问题：任何生僻词都可以
 *       拆解为子词组合；
 *     - 比字符级 token 更短的序列长度，训练更快；
 *     - 自动发现语素模式（如 "ing"、"ed" 等后缀）。
 *
 *   应用：GPT 系列、BERT 等主流语言模型均使用 BPE 或其变体。
 */

/* ======================== 7. 停用词去除演示 ======================== */

// 常见英语停用词集合
const std::unordered_set<std::string> STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "can", "shall",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "this", "that", "these", "those",
    "so", "if", "then", "than", "too", "very", "just", "about",
    "into", "up", "out", "over", "under", "again", "further", "once"};

/*
 * 注意：情感分析任务中，以下停用词应予以保留，
 * 因为它们携带关键的情感/否定信息：
 *   - 否定词：not, never, no, neither, nor
 *   - 程度词：very, extremely, quite, rather
 */
const std::unordered_set<std::string> SENTIMENT_KEEP = {
    "not", "never", "no", "very", "too"};

std::vector<std::string> remove_stopwords(
    const std::vector<std::string> &tokens,
    bool sentiment_mode = false) {
    std::vector<std::string> filtered;
    for (const auto &token : tokens) {
        // 在情感分析模式下，保留情感相关的关键词
        if (sentiment_mode && SENTIMENT_KEEP.find(token) != SENTIMENT_KEEP.end()) {
            filtered.push_back(token);
            continue;
        }
        // 过滤掉停用词
        if (STOP_WORDS.find(token) != STOP_WORDS.end()) {
            continue;
        }
        filtered.push_back(token);
    }
    return filtered;
}

/* ============================== main ================================= */

int main() {
    std::cout << "第 7 章：文本处理流水线演示 (纯 STL 实现)"
              << "\n"
              << std::endl;

    /* ---------- 1. 大小写转换 ---------- */
    std::cout << "【1. 大小写转换】" << std::endl;
    {
        std::string text = "Machine Learning with C++ is Powerful!";
        std::cout << "  原文: " << text << std::endl;
        std::cout << "  小写: " << to_lowercase(text) << "\n"
                  << std::endl;
    }

    /* ---------- 2. 标点符号去除 ---------- */
    std::cout << "【2. 标点符号去除】" << std::endl;
    {
        std::string text = "Hello, World! How are you? I'm fine -- thank you.";
        std::cout << "  原文 : " << text << std::endl;
        std::cout << "  清洗: " << remove_punctuation(text) << "\n"
                  << std::endl;
    }

    /* ---------- 3. 选择性字符去除 ---------- */
    std::cout << "【3. 选择性字符去除】" << std::endl;
    {
        std::string text = "C++17 features: auto, lambda, constexpr...";
        std::unordered_set<char> keep = {'+', '-'}; // 保留 C++ 的加号
        std::cout << "  原文   : " << text << std::endl;
        std::cout << "  保留 +-: " << remove_selective_punctuation(text, keep)
                  << "\n"
                  << std::endl;
    }

    /* ---------- 4. 句子切分 ---------- */
    std::cout << "【4. 句子切分】" << std::endl;
    {
        std::string text =
            "Deep learning is transforming AI. "
            "RNNs handle sequences well! "
            "Can machines understand language?";
        auto sentences = split_sentences(text);
        std::cout << "  原文: " << text << std::endl;
        std::cout << "  切分为 " << sentences.size() << " 个句子:" << std::endl;
        for (size_t i = 0; i < sentences.size(); ++i) {
            std::cout << "    [" << (i + 1) << "] " << sentences[i] << std::endl;
        }
        std::cout << std::endl;
    }

    /* ---------- 5. 分词类型对比 ---------- */
    std::cout << "【5. 分词类型对比】" << std::endl;
    {
        std::string sentence =
            "Machine learning algorithms can revolutionize "
            "healthcare diagnostics.";
        std::string lower = to_lowercase(sentence);
        std::string cleaned = remove_punctuation(lower);

        std::cout << "  例句: \"" << lower << "\"\n"
                  << std::endl;

        // 词级分词
        auto word_tokens = word_tokenize(cleaned);
        print_tokens("词级 tokens    (" + std::to_string(word_tokens.size()) + " tokens)",
                     word_tokens);

        // 字符级分词
        auto char_tokens = char_tokenize(lower);
        std::cout << "  字符级 tokens  (" << char_tokens.size()
                  << " tokens): [";
        for (size_t i = 0; i < char_tokens.size(); ++i) {
            std::cout << char_tokens[i];
            if (i + 1 < char_tokens.size()) std::cout << ",";
        }
        std::cout << "]" << std::endl;

        // 子词级（人工演示 BPE 效果）
        std::vector<std::string> subword_tokens = {
            "Mach", "ine", "learn", "ing", "algor", "ithms",
            "can", "revol", "ution", "ize", "health", "care",
            "diagn", "ostics", "."};
        print_tokens("子词级 tokens   (" + std::to_string(subword_tokens.size()) + " tokens)",
                     subword_tokens);

        // 对比表格
        std::cout << "\n  ┌────────────────────┬───────────────────────────────┐" << std::endl;
        std::cout << "  │ 分词方式            │ 优势与权衡                      │" << std::endl;
        std::cout << "  ├────────────────────┼───────────────────────────────┤" << std::endl;
        std::cout << "  │ 词级     (8 token)  │ 语义完整，词汇量大，OOV 问题    │" << std::endl;
        std::cout << "  │ 字符级  (68 token)  │ 无 OOV，序列长，训练慢           │" << std::endl;
        std::cout << "  │ 子词级  (15 token)  │ 平衡方案，自适应语素，主流选择    │" << std::endl;
        std::cout << "  └────────────────────┴───────────────────────────────┘\n"
                  << std::endl;
    }

    /* ---------- 6. BPE 概念 ---------- */
    std::cout << "【6. BPE（Byte Pair Encoding）概念】" << std::endl;
    std::cout << "  详见源码中的详细注释说明。\n"
              << std::endl;

    /* ---------- 7. 停用词去除 ---------- */
    std::cout << "【7. 停用词去除】" << std::endl;
    {
        std::string raw = "The machine learning algorithms are very "
                          "powerful tools for analyzing large datasets.";
        std::string cleaned = remove_punctuation(to_lowercase(raw));
        auto all_tokens = word_tokenize(cleaned);

        std::cout << "  原文     : " << raw << std::endl;
        std::cout << "  全部词   : ";
        for (size_t i = 0; i < all_tokens.size(); ++i)
            std::cout << all_tokens[i] << (i + 1 < all_tokens.size() ? " " : "");
        std::cout << " (" << all_tokens.size() << " 词)" << std::endl;

        auto filtered = remove_stopwords(all_tokens);
        std::cout << "  去停用词: ";
        for (size_t i = 0; i < filtered.size(); ++i)
            std::cout << filtered[i] << (i + 1 < filtered.size() ? " " : "");
        std::cout << " (" << filtered.size() << " 词)" << std::endl;

        double reduction = (1.0 - static_cast<double>(filtered.size()) / all_tokens.size()) * 100.0;
        std::cout << "  压缩率   : " << all_tokens.size() << " → "
                  << filtered.size() << " 词 ("
                  << std::fixed << std::setprecision(0)
                  << reduction << "% 减少)" << std::endl;

        // 情感分析提醒
        std::cout << "\n  ⚠ 注意：情感分析任务中不应移除 \"not\"\"never\"\"very\""
                  << " 等关键词，因为它们携带情感信息。" << std::endl;

        // 演示情感模式
        std::string sent_text = "This movie is not very good.";
        std::string sent_clean = remove_punctuation(to_lowercase(sent_text));
        auto sent_tokens = word_tokenize(sent_clean);
        auto sent_filtered = remove_stopwords(sent_tokens, true); // 情感模式

        std::cout << "\n  情感分析示例:" << std::endl;
        std::cout << "    原文: " << sent_text << std::endl;
        std::cout << "    去停用词（情感模式保留 not/very）: ";
        for (size_t i = 0; i < sent_filtered.size(); ++i)
            std::cout << sent_filtered[i] << (i + 1 < sent_filtered.size() ? " " : "");
        std::cout << std::endl;
    }

    std::cout << "\n文本处理流水线演示完成。" << std::endl;
    return 0;
}
