/*
 * embedding_based_vectorization.cpp
 * 第2章：C++ 中的数据准备与预处理
 *
 * 基于嵌入的向量化将文本 token 转换为密集的数值向量，
 * 这些向量能够捕获语义信息。与独热编码（稀疏、无语义）不同，
 * 嵌入将相似词在向量空间中放置得彼此靠近，
 * 使模型能够在相关概念之间进行泛化。
 *
 * 涉及的技术：
 *   - TF-IDF（词频-逆文档频率）：根据词在文档中出现的频率
 *     与在整个语料库中出现的频率来赋予权重。
 *     在单个文档中常见但在整个语料库中罕见的词得分较高。
 *     简单、可解释的基线方法；但没有语义相似性。
 *   - Word2Vec 风格的嵌入：每个词被映射到一个密集向量，
 *     其中相似的词具有相似的向量。两种主要架构：
 *       * CBOW：根据上下文预测目标词
 *       * Skip-gram：根据目标词预测上下文
 *     捕获语义关系（例如，king - man + woman ≈ queen）。
 *   - 句子/文档嵌入：通过平均或其他方式组合词向量
 *     来表示较长的文本片段。
 *
 * 何时使用 TF-IDF vs 嵌入：
 *   - TF-IDF：快速基线、可解释的特征重要性、小数据集
 *   - 嵌入：需要语义理解、迁移学习、大词汇量
 */

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <algorithm>

// ----------------------------------------------------------------
// 简单分词器：按空白符切分文本
// ----------------------------------------------------------------
std::vector<std::string> tokenize(const std::string &text) {
    std::vector<std::string> tokens;
    std::istringstream iss(text);
    std::string token;
    while (iss >> token) {
        // 转换为小写以实现大小写不敏感匹配
        std::transform(token.begin(), token.end(), token.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        tokens.push_back(token);
    }
    return tokens;
}

// ----------------------------------------------------------------
// TF-IDF 向量化
// TF(t,d) = 词 t 在文档 d 中的出现次数 / 文档 d 的总词数
// IDF(t)  = log(N / df(t))
// TF-IDF(t,d) = TF(t,d) * IDF(t)
//
// 适用场景：文本分类基线、搜索排序、
//           具有可解释权重的简单文档相似度计算。
// 不适用场景：捕获同义词之间的语义相似性。
// ----------------------------------------------------------------
struct TFIDFVectorizer {
    std::map<std::string, int> wordToIdx;
    std::map<std::string, double> idf;
    int vocabSize = 0;
    int totalDocs = 0;

    // 在语料库上拟合：计算每个词的 IDF
    void fit(const std::vector<std::vector<std::string>> &corpus) {
        totalDocs = (int)corpus.size();
        std::map<std::string, int> docFreq;
        for (const auto &doc : corpus) {
            std::map<std::string, bool> seen;
            for (const auto &word : doc) {
                if (!seen[word]) {
                    docFreq[word]++;
                    seen[word] = true;
                }
                if (wordToIdx.find(word) == wordToIdx.end()) {
                    wordToIdx[word] = vocabSize++;
                }
            }
        }
        for (const auto &[word, df] : docFreq) {
            idf[word] = std::log((double)(totalDocs + 1) / (df + 1)) + 1.0;
        }
    }

    // 将单个文档转换为 TF-IDF 向量
    std::vector<double> transform(const std::vector<std::string> &doc) {
        std::vector<double> vec(vocabSize, 0.0);
        std::map<std::string, int> termFreq;
        for (const auto &word : doc) termFreq[word]++;

        double totalTerms = (double)doc.size();
        for (const auto &[word, tf] : termFreq) {
            auto it = wordToIdx.find(word);
            if (it != wordToIdx.end()) {
                double tfNorm = tf / totalTerms;
                double idfVal = idf.count(word) ? idf[word] : 1.0;
                vec[it->second] = tfNorm * idfVal;
            }
        }
        return vec;
    }
};

// ----------------------------------------------------------------
// 玩具级 Word2Vec 风格嵌入（确定性，仅用于演示）
// 生产环境中，使用预训练嵌入（GloVe、fastText）或
// 在模型训练期间通过 LibTorch 的 nn::Embedding 层进行训练。
//
// 本演示创建一个带有正弦值的简单查找表。
// 索引相邻的词获得相似的向量（基于正弦位置编码）。
// ----------------------------------------------------------------
struct SimpleEmbedding {
    int dim;
    std::map<std::string, std::vector<double>> table;

    SimpleEmbedding(int embeddingDim) : dim(embeddingDim) {
    }

    // 为词汇表构建嵌入表
    void buildTable(const std::vector<std::string> &vocab) {
        for (size_t i = 0; i < vocab.size(); ++i) {
            std::vector<double> vec(dim);
            for (int j = 0; j < dim; ++j) {
                // 基于词索引和维度的正弦编码
                double angle = (double)i / std::pow(10000.0, 2.0 * j / dim);
                vec[j] = (j % 2 == 0) ? std::sin(angle) : std::cos(angle);
            }
            table[vocab[i]] = vec;
        }
    }

    // 查找某个词的嵌入向量
    std::vector<double> get(const std::string &word,
                            const std::vector<double> &unkVec = {}) {
        auto it = table.find(word);
        if (it != table.end()) return it->second;
        // 返回 UNK 向量或零向量
        return unkVec.empty() ? std::vector<double>(dim, 0.0) : unkVec;
    }

    // 对词嵌入求平均以获得句子嵌入
    std::vector<double> sentenceEmbed(const std::vector<std::string> &tokens) {
        std::vector<double> avg(dim, 0.0);
        int count = 0;
        for (const auto &tok : tokens) {
            auto vec = get(tok);
            for (int j = 0; j < dim; ++j) avg[j] += vec[j];
            count++;
        }
        if (count > 0) {
            for (int j = 0; j < dim; ++j) avg[j] /= count;
        }
        return avg;
    }
};

// 辅助函数：打印密集向量
void printDense(const std::string &label, const std::vector<double> &v,
                int maxN = 8) {
    std::cout << label << " [";
    size_t n = std::min(v.size(), (size_t)maxN);
    for (size_t i = 0; i < n; ++i) {
        std::cout << std::fixed << std::setprecision(3) << v[i];
        if (i + 1 < v.size()) std::cout << ", ";
    }
    if (v.size() > (size_t)maxN) std::cout << "...";
    std::cout << "]\n";
}

int main() {
    std::cout << "=== Embedding-Based Vectorization Demos ===\n\n";

    // ===========================================
    // 1. TF-IDF 演示
    // ===========================================
    std::cout << "[TF-IDF] 按文档频率为词赋予权重。\n";
    std::cout << "  高分：词在本文档中出现频繁，"
                 "在其他文档中罕见。\n\n";

    std::vector<std::string> docs = {
        "deep learning with neural networks",
        "machine learning is powerful",
        "deep neural networks and deep learning",
        "c plus plus programming language"};

    std::vector<std::vector<std::string>> corpus;
    for (const auto &doc : docs) corpus.push_back(tokenize(doc));

    TFIDFVectorizer tfidf;
    tfidf.fit(corpus);

    for (size_t i = 0; i < docs.size(); ++i) {
        auto vec = tfidf.transform(corpus[i]);
        std::cout << "Doc" << i << ": \"" << docs[i] << "\"\n";
        // 显示非零条目
        for (const auto &[word, idx] : tfidf.wordToIdx) {
            if (vec[idx] > 0.001)
                std::cout << "  " << word << ": " << std::setprecision(3) << vec[idx] << "\n";
        }
        std::cout << "\n";
    }

    // ===========================================
    // 2. Word2Vec 风格嵌入演示
    // ===========================================
    std::cout << "[词嵌入] 捕获语义信息的密集向量。\n";
    std::cout << "  相似的词具有相似的向量（余弦相似度）。\n\n";

    std::vector<std::string> vocab = {
        "deep", "learning", "network", "model",
        "neural", "machine", "training", "data"};

    SimpleEmbedding emb(4); // 4 维嵌入
    emb.buildTable(vocab);

    std::cout << "词向量（4 维）：\n";
    for (const auto &word : vocab) {
        auto vec = emb.get(word);
        printDense("  " + word, vec);
    }

    // 通过平均获得句子嵌入
    std::cout << "\n[句子嵌入] 对词向量取平均。\n";
    auto sentence = tokenize("deep neural network");
    auto sentVec = emb.sentenceEmbed(sentence);
    printDense("  'deep neural network'", sentVec);
    std::cout << "  取平均是一种简单但有效的基线方法。\n"
              << "  要获得更好的效果，请使用预训练模型或加权池化。\n";

    // ===========================================
    // 3. 预训练嵌入（GloVe + Eigen）
    //    PDF 第 60-62 页引用
    // ===========================================
    std::cout << "\n[预训练嵌入] 加载 GloVe 向量（PDF 第 60-62 页）。\n";
    std::cout << "  GloVe（Global Vectors）提供预训练的词向量，\n";
    std::cout << "  在数十亿 token 上训练得到。\n";
    std::cout << "  下载地址：https://nlp.stanford.edu/projects/glove/\n\n";

    std::cout << "  典型的 C++ 加载模式（伪代码）：\n";
    std::cout << "    std::ifstream file(\"glove.6B.100d.txt\");\n";
    std::cout << "    std::unordered_map<std::string, Eigen::VectorXf> embeddings;\n";
    std::cout << "    std::string line, word;\n";
    std::cout << "    while (getline(file, line)) {\n";
    std::cout << "      std::istringstream iss(line);\n";
    std::cout << "      iss >> word;\n";
    std::cout << "      Eigen::VectorXf vec(100);\n";
    std::cout << "      for (int d = 0; d < 100; ++d) iss >> vec(d);\n";
    std::cout << "      embeddings[word] = vec;\n";
    std::cout << "    }\n";
    std::cout << "  // 然后使用 'embeddings[word]' 进行查找。\n";
    std::cout << "  // 常见维度：50, 100, 200, 300。\n";
    std::cout << "  // GloVe 可以捕获线性类比关系：king - man + woman ≈ queen。\n";

    return 0;
}
