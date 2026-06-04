/*
 * Encoding_categorical_features.cpp
 * 第 2 章：C++ 中的数据准备与预处理
 *
 * 类别特征编码将离散标签转换为神经网络可以接受的数值表示。
 * 不同的编码策略适用于不同的模型族和数据基数。
 *
 * 涵盖的技术（来自 PDF“编码类别特征”一节）：
 *   - One-hot 编码：N 个类别 -> N 个二元列。
 *     适用于线性/NN 模型，但在高基数时会爆炸。
 *     线性模型的共线性修复：删除一列作为参考。
 *   - 频率编码：将每个类别映射到其出现次数。
 *     保留频率信息，不会维度爆炸。
 *     每折单独计算频率以避免数据泄露。
 *   - 序数编码：将有序类别映射到整数（XS < S < M < L < XL）。
 *     仅当存在真实顺序时使用；否则注入虚假的顺序关系。
 *   - 二进制编码：整数 ID -> 跨越 ceil(log2 N) 位的二进制表示。
 *     相比 one-hot 大幅减少维度；注意：引入了伪序数结构。
 *   - 嵌入编码：在训练期间学习稠密向量，捕获语义相似性。
 *     对未见过的类别使用 <UNK>/OOV 标记。
 */

#include <vector>
#include <map>
#include <string>
#include <bitset>
#include <cmath>
#include <iostream>
#include <iomanip>

// ----------------------------------------------------------------
// One-hot 编码：每个类别变成一个长度为（唯一类别数）的二元向量。
// 最适合低基数特征与线性/NN 模型。
// ----------------------------------------------------------------
std::vector<std::vector<int>> oneHotEncode(
    const std::vector<std::string> &categories) {
    std::map<std::string, int> categoryMap;
    int index = 0;
    for (const auto &cat : categories) {
        if (categoryMap.find(cat) == categoryMap.end()) {
            categoryMap[cat] = index++;
        }
    }
    std::vector<std::vector<int>> encoded(
        categories.size(),
        std::vector<int>(categoryMap.size(), 0));
    for (size_t i = 0; i < categories.size(); ++i) {
        encoded[i][categoryMap[categories[i]]] = 1;
    }
    return encoded;
}

// ----------------------------------------------------------------
// 频率编码：将每个类别映射到其在数据中的出现次数。
// 单个数值特征，保留频率信息，不会维度爆炸。
// 对重尾类别可考虑对数频率或截断处理。
// ----------------------------------------------------------------
std::vector<int> frequencyEncode(
    const std::vector<std::string> &categories) {
    std::map<std::string, int> freqMap;
    for (const auto &cat : categories) {
        freqMap[cat]++;
    }
    std::vector<int> encoded;
    for (const auto &cat : categories) {
        encoded.push_back(freqMap[cat]);
    }
    return encoded;
}

// ----------------------------------------------------------------
// 序数编码：为有序类别分配连续的整数。
// 仅当存在真实顺序时使用（如教育水平、尺寸标签）。
// 在训练集上拟合映射；将未见过的类别路由到 <UNK>。
// ----------------------------------------------------------------
std::vector<int> ordinalEncode(
    const std::vector<std::string> &categories) {
    std::map<std::string, int> categoryMap;
    int index = 0;
    for (const auto &cat : categories) {
        if (categoryMap.find(cat) == categoryMap.end()) {
            categoryMap[cat] = index++;
        }
    }
    std::vector<int> encoded;
    for (const auto &cat : categories) {
        encoded.push_back(categoryMap[cat]);
    }
    return encoded;
}

// ----------------------------------------------------------------
// 二进制编码：将整数 ID 映射到二进制字符串，使用 ceil(log2 N) 位。
// 对于高基数，比 one-hot 小得多。
// 注意：引入了伪序数邻近关系。
// ----------------------------------------------------------------
std::vector<std::string> binaryEncode(
    const std::vector<int> &categories) {
    std::vector<std::string> encoded;
    for (const auto &cat : categories) {
        encoded.push_back(std::bitset<8>(cat).to_string());
    }
    return encoded;
}

// ----------------------------------------------------------------
// 嵌入编码（玩具演示）：将每个唯一字符串类别映射到一个
// 确定性的正弦稠密向量，大小为 embeddingSize。
// 生产中，嵌入与模型联合学习。
// ----------------------------------------------------------------
std::vector<std::vector<double>> embeddingEncode(
    const std::vector<std::string> &categories, int embeddingSize) {
    std::map<std::string, int> categoryMap;
    int index = 0;
    for (const auto &cat : categories) {
        if (categoryMap.find(cat) == categoryMap.end()) {
            categoryMap[cat] = index++;
        }
    }
    std::vector<std::vector<double>> embeddings(
        categoryMap.size(),
        std::vector<double>(embeddingSize, 0));
    for (size_t i = 0; i < embeddings.size(); ++i) {
        for (size_t j = 0; j < (size_t)embeddingSize; ++j) {
            embeddings[i][j] = std::sin((double)(i + j));
        }
    }
    std::vector<std::vector<double>> encoded;
    for (const auto &cat : categories) {
        encoded.push_back(embeddings[categoryMap[cat]]);
    }
    return encoded;
}

// ----------------------------------------------------------------
// 辅助函数：打印二维向量
// ----------------------------------------------------------------
template <typename T>
void print2D(const std::vector<std::vector<T>> &mat,
             const std::string &label) {
    std::cout << label << ":\n";
    for (const auto &row : mat) {
        std::cout << "  [";
        for (size_t j = 0; j < row.size(); ++j) {
            if constexpr (std::is_floating_point_v<T>)
                std::cout << std::fixed << std::setprecision(2);
            std::cout << row[j];
            if (j + 1 < row.size()) std::cout << ", ";
        }
        std::cout << "]\n";
    }
}

int main() {
    // 样本数据：8 个类别观测值
    std::vector<std::string> categories = {
        "cat", "dog", "cat", "bird", "dog", "cat", "fish", "bird"};

    std::cout << "=== Categorical Feature Encoding Demos ===\n\n";
    std::cout << "Input categories: ";
    for (size_t i = 0; i < categories.size(); ++i) {
        std::cout << categories[i];
        if (i + 1 < categories.size()) std::cout << ", ";
    }
    std::cout << "\n\n";

    // 1. One-hot
    auto oh = oneHotEncode(categories);
    print2D(oh, "One-hot encoding");

    // 2. Frequency
    auto freq = frequencyEncode(categories);
    std::cout << "\nFrequency encoding: [";
    for (size_t i = 0; i < freq.size(); ++i) {
        std::cout << freq[i];
        if (i + 1 < freq.size()) std::cout << ", ";
    }
    std::cout << "]\n";

    // 3. Ordinal
    auto ord = ordinalEncode(categories);
    std::cout << "\nOrdinal encoding: [";
    for (size_t i = 0; i < ord.size(); ++i) {
        std::cout << ord[i];
        if (i + 1 < ord.size()) std::cout << ", ";
    }
    std::cout << "]\n";

    // 4. Binary (use ordinal indices as integer IDs)
    auto bin = binaryEncode(ord);
    std::cout << "\nBinary encoding: [";
    for (size_t i = 0; i < bin.size(); ++i) {
        std::cout << bin[i];
        if (i + 1 < bin.size()) std::cout << ", ";
    }
    std::cout << "]\n";

    // 5. Embedding (embeddingSize=4)
    auto emb = embeddingEncode(categories, 4);
    print2D(emb, "\nEmbedding encoding (dim=4)");

    return 0;
}
