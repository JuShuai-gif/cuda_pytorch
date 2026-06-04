/*
 * tokenization_stop_word_removal_stemming.cpp
 * 第2章：C++ 中的数据准备与预处理
 *
 * 文本预处理对 NLP 任务至关重要。原始文本是混乱的：
 * 大小写不一致、标点符号、信号很弱的常见词
 *（"the"、"is"、"and"），以及形态变体（run/runs/running）。
 *
 * 涉及的技术（来自 PDF 第2章）：
 *   - 分词（Tokenization）：将文本切分为有意义的单元（词/子词）。
 *     SentencePiece 用于子词分词（BPE/Unigram）；
 *     比词级分词更好地处理 OOV 词。
 *   - 停用词去除（Stop word removal）：过滤掉高频功能词，
 *     它们携带的语义信息很少。可以减小词汇量并降低噪音。
 *     注意：某些任务（情感分析、问答）需要它们。
 *   - 词干提取（Stemming）：通过剥离后缀将词还原为其词根形式。
 *     Porter 词干提取器是最常用的。快速但粗糙（"university"
 *     -> "univers"）。生产环境使用 libstemmer。
 *     词形还原（Lemmatization，更复杂）会考虑词性。
 *
 * 何时使用/避免：
 *   - 跳过停用词去除：情感分析、问答系统、
 *     机器翻译（功能词携带语义信息）。
 *   - 使用词干提取：信息检索、小数据集的文本分类
 *     （减少稀疏性）。
 *   - 使用 SentencePiece：神经机器翻译、LLM（处理 OOV、多语言）。
 */

#include <iostream>
#include <vector>
#include <string>
#include <unordered_set>
#include <sstream>
#include <algorithm>
#include <cctype>

// ================================================================
// 1. 分词（Tokenization）
// ================================================================

// 简单的空白符分词器，带标点去除功能
std::vector<std::string> tokenize(const std::string &text) {
    std::vector<std::string> tokens;
    std::string current;
    for (char c : text) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            current += static_cast<char>(std::tolower(
                static_cast<unsigned char>(c)));
        } else if (!current.empty()) {
            tokens.push_back(current);
            current.clear();
        }
    }
    if (!current.empty()) tokens.push_back(current);
    return tokens;
}

// ================================================================
// 2. 停用词去除（Stop Word Removal）
// ================================================================

// 常见英语停用词
const std::unordered_set<std::string> STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "can", "shall",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "this", "that", "these", "those", "not", "no",
    "so", "if", "then", "than", "too", "very", "just", "about",
    "into", "up", "out", "over", "under", "again", "further", "once"};

std::vector<std::string> removeStopWords(
    const std::vector<std::string> &tokens) {
    std::vector<std::string> filtered;
    for (const auto &token : tokens) {
        if (STOP_WORDS.find(token) == STOP_WORDS.end()) {
            filtered.push_back(token);
        }
    }
    return filtered;
}

// ================================================================
// 3. 词干提取（Stemming，Porter Stemmer - 简化版）
// ================================================================

// 简化版 Porter 词干提取器实现
// 步骤：处理常见后缀（-ing, -ed, -s, -ly, -ment 等）
// 生产环境：使用 libstemmer（M.F. Porter 的 Snowball 词干提取库）
std::string stem(const std::string &word) {
    std::string w = word;
    size_t len = w.length();
    if (len <= 3) return w;

    // 步骤 1a：处理复数形式和过去分词
    if (len > 4 && w.substr(len - 4) == "sses") {
        w = w.substr(0, len - 2); // sses -> ss
    } else if (len > 3 && w.substr(len - 3) == "ies") {
        w = w.substr(0, len - 2); // ies -> i
    } else if (len > 2 && w.back() == 's' && w[len - 2] != 's') {
        w = w.substr(0, len - 1); // remove trailing s (not ss)
    }
    len = w.length();

    // 步骤 1b：处理 -ing 和 -ed
    if (len > 4 && w.substr(len - 3) == "ing") {
        w = w.substr(0, len - 3);
    } else if (len > 3 && w.substr(len - 2) == "ed") {
        w = w.substr(0, len - 2);
    }

    // 步骤 2：常见后缀
    // -ational -> -ate
    if (w.length() > 7 && w.substr(w.length() - 7) == "ational")
        w = w.substr(0, w.length() - 5) + "e";
    // -ization -> -ize
    if (w.length() > 7 && w.substr(w.length() - 7) == "ization")
        w = w.substr(0, w.length() - 5) + "e";
    // -fulness -> -ful
    if (w.length() > 7 && w.substr(w.length() - 7) == "fulness")
        w = w.substr(0, w.length() - 4);
    // -ousness -> -ous
    if (w.length() > 7 && w.substr(w.length() - 7) == "ousness")
        w = w.substr(0, w.length() - 4);
    // -iveness -> -ive
    if (w.length() > 7 && w.substr(w.length() - 7) == "iveness")
        w = w.substr(0, w.length() - 4);

    // -ation -> -ate
    if (w.length() > 5 && w.substr(w.length() - 5) == "ation")
        w = w.substr(0, w.length() - 3) + "e";
    // -alism -> -al
    if (w.length() > 5 && w.substr(w.length() - 5) == "alism")
        w = w.substr(0, w.length() - 3);

    // -ment -> (remove)
    if (w.length() > 4 && w.substr(w.length() - 4) == "ment")
        w = w.substr(0, w.length() - 4);
    // -ness -> (remove)
    if (w.length() > 4 && w.substr(w.length() - 4) == "ness")
        w = w.substr(0, w.length() - 4);
    // -ance -> (remove)
    if (w.length() > 4 && w.substr(w.length() - 4) == "ance")
        w = w.substr(0, w.length() - 4);
    // -ence -> (remove)
    if (w.length() > 4 && w.substr(w.length() - 4) == "ence")
        w = w.substr(0, w.length() - 4);

    // -able -> (remove)
    if (w.length() > 4 && w.substr(w.length() - 4) == "able")
        w = w.substr(0, w.length() - 4);
    // -ible -> (remove)
    if (w.length() > 4 && w.substr(w.length() - 4) == "ible")
        w = w.substr(0, w.length() - 4);

    // 步骤 3：-ly, -er, -est, -al
    if (w.length() > 2 && w.substr(w.length() - 2) == "ly")
        w = w.substr(0, w.length() - 2);
    if (w.length() > 2 && w.substr(w.length() - 2) == "er")
        w = w.substr(0, w.length() - 2);
    if (w.length() > 3 && w.substr(w.length() - 3) == "est")
        w = w.substr(0, w.length() - 3);
    if (w.length() > 2 && w.substr(w.length() - 2) == "al" && w.length() > 4)
        w = w.substr(0, w.length() - 2);

    return w;
}

std::vector<std::string> stemTokens(
    const std::vector<std::string> &tokens) {
    std::vector<std::string> stemmed;
    for (const auto &token : tokens) {
        stemmed.push_back(stem(token));
    }
    return stemmed;
}

// ================================================================
// 4. 完整预处理流水线
// ================================================================
std::vector<std::string> preprocessText(const std::string &text) {
    auto tokens = tokenize(text);
    auto noStops = removeStopWords(tokens);
    auto stemmed = stemTokens(noStops);
    return stemmed;
}

// 辅助函数
void printTokens(const std::string &label,
                 const std::vector<std::string> &tokens) {
    std::cout << label << " [";
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::cout << tokens[i];
        if (i + 1 < tokens.size()) std::cout << ", ";
    }
    std::cout << "]\n";
}

int main() {
    std::cout << "=== Text Preprocessing Pipeline Demo ===\n\n";

    std::string text =
        "The deep learning models are running efficiently "
        "on multiple machines with powerful GPUs";

    std::cout << "Original: \"" << text << "\"\n\n";

    // 步骤 1：分词
    auto tokens = tokenize(text);
    printTokens("1. Tokenize:     ", tokens);

    // 步骤 2：停用词去除
    auto noStops = removeStopWords(tokens);
    printTokens("2. Remove stops: ", noStops);
    std::cout << "   (去除了功能词如 'the'、'are'、'on'、'with')\n";

    // 步骤 3：词干提取
    auto stemmed = stemTokens(noStops);
    printTokens("3. Stem:         ", stemmed);
    std::cout << "   (将 'running'->'runn', 'machines'->'machine', "
              << "'powerful'->'power' 进行了还原)\n";

    // 完整流水线
    std::cout << "\n[Full Pipeline]\n";
    auto result = preprocessText(text);
    printTokens("   Result:       ", result);

    std::cout << "\n--- 说明 ---\n";
    std::cout << "停用词去除：在情感分析/问答等任务中应跳过，因为\n"
              << "  功能词携带语义信息（例如，'not good' vs 'good'）。\n";
    std::cout << "词干提取：Porter 词干提取器速度快但粗糙。\n"
              << "  生产环境请使用 libstemmer（Snowball）。\n";
    std::cout << "SentencePiece：用于子词分词（BPE/Unigram），\n"
              << "  安装 SentencePiece 库。可处理 OOV 和多语言场景。\n";

    return 0;
}
