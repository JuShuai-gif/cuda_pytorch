/*
 * 05_word2vec_libtorch.cpp - 第 7 章：循环神经网络与 LSTM
 * Skip-gram Word2Vec 基于 LibTorch 实现（对应原书第 248-253 页）
 *
 * 演示内容：
 *   1. WordEmbeddings 类：构建词汇表、生成 Skip-gram 训练对
 *   2. 使用 torch::nn::Embedding + Linear 层训练词向量
 *   3. SGD 优化器 + 交叉熵损失训练循环
 *   4. 训练后展示词嵌入向量与语义关系
 *   5. 概念说明：king - man + woman ≈ queen 的向量类比
 *
 * 依赖：LibTorch（C++ 发行版）
 * 编译时需链接 torch 库。
 */

#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include <random>
#include <iomanip>

/* ======================== WordEmbeddings 类 ======================== */

class WordEmbeddings {
public:
    /*
     * 构造函数：初始化词嵌入层
     *   embed_dim - 词向量维度
     */
    explicit WordEmbeddings(int embed_dim) : embed_dim_(embed_dim), vocab_size_(0),
                                             device_(torch::kCPU) {
        // 预分配特殊 token
        id_to_word_.push_back("<PAD>"); // 填充 token (ID=0)
        id_to_word_.push_back("<UNK>"); // 未知词 token (ID=1)
        word_to_id_["<PAD>"] = 0;
        word_to_id_["<UNK>"] = 1;
    }

    /* ===================== 构建词汇表 ===================== */

    /*
     * 从文本文件构建词汇表：
     *   1. 读取文件，转小写，去除非字母数字字符；
     *   2. 统计词频；
     *   3. 过滤低频词（min_freq=2）；
     *   4. 添加 <UNK>/<PAD> 特殊 token。
     */
    void buildVocabulary(const std::string &filename, int min_freq = 2) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "警告: 无法打开文件 " << filename
                      << "，使用内置语料库。" << std::endl;
            buildVocabularyFromString(builtInCorpus(), min_freq);
            return;
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        buildVocabularyFromString(buffer.str(), min_freq);
    }

    // 从字符串直接构建词汇表（用于内置语料）
    void buildVocabularyFromString(const std::string &text, int min_freq = 2) {
        // 统计词频
        std::unordered_map<std::string, int> freq;
        std::string word;
        for (char c : text) {
            if (std::isalnum(static_cast<unsigned char>(c))) {
                word += static_cast<char>(
                    std::tolower(static_cast<unsigned char>(c)));
            } else {
                if (!word.empty()) {
                    freq[word]++;
                    word.clear();
                }
            }
        }
        if (!word.empty()) freq[word]++;

        // 按词频排序（降序）
        std::vector<std::pair<std::string, int>> sorted_freq(
            freq.begin(), freq.end());
        std::sort(sorted_freq.begin(), sorted_freq.end(),
                  [](const auto &a, const auto &b) {
                      return a.second > b.second;
                  });

        // 添加高频词到词汇表
        for (const auto &[w, count] : sorted_freq) {
            if (count >= min_freq) {
                word_to_id_[w] = static_cast<int>(id_to_word_.size());
                id_to_word_.push_back(w);
            }
        }

        vocab_size_ = static_cast<int>(id_to_word_.size());
        std::cout << "词汇表大小: " << vocab_size_ << " (含 <PAD>/<UNK>)"
                  << "，最低频率阈值: " << min_freq << std::endl;
    }

    /* ===================== Skip-gram 训练对 ===================== */

    /*
     * 从文本生成 Skip-gram 训练对：
     *   以每个词为中心词，窗口范围内的词为上下文词，
     *   生成 (center_id, context_id) 对。
     */
    std::vector<std::pair<int, int>> generateSkipGramPairs(
        const std::string &text, int window_size = 2) {
        // 先将文本转换为词 ID 序列
        std::vector<int> ids;
        std::string word;
        for (char c : text) {
            if (std::isalnum(static_cast<unsigned char>(c))) {
                word += static_cast<char>(
                    std::tolower(static_cast<unsigned char>(c)));
            } else {
                if (!word.empty()) {
                    ids.push_back(wordToId(word));
                    word.clear();
                }
            }
        }
        if (!word.empty()) ids.push_back(wordToId(word));

        // 生成 (center, context) 训练对
        std::vector<std::pair<int, int>> pairs;
        int n = static_cast<int>(ids.size());
        for (int i = 0; i < n; ++i) {
            int center = ids[i];
            if (center == 0) continue; // 跳过 <PAD>

            for (int w = -window_size; w <= window_size; ++w) {
                if (w == 0) continue; // 自身不算上下文
                int pos = i + w;
                if (pos < 0 || pos >= n) continue;
                int context = ids[pos];
                if (context == 0) continue; // 跳过 <PAD>
                pairs.emplace_back(center, context);
            }
        }

        return pairs;
    }

    /* ===================== Skip-gram 训练 ===================== */

    /*
     * Skip-gram 训练流程：
     *   1. 生成训练对；
     *   2. 构建输出层 (embed_dim → vocab_size)；
     *   3. SGD 优化器优化 embedding + output_layer；
     *   4. 遍历训练对：center→embed→logits→交叉熵损失→反向传播→参数更新；
     *   5. 每 10 轮打印损失。
     */
    void trainSkipGram(const std::string &corpus, int epochs = 100,
                       double lr = 0.01) {
        auto pairs = generateSkipGramPairs(corpus);
        if (pairs.empty()) {
            std::cerr << "错误: 训练对为空，无法训练。" << std::endl;
            return;
        }

        std::cout << "\nSkip-gram 训练对数量: " << pairs.size()
                  << " | 轮数: " << epochs
                  << " | 学习率: " << lr << std::endl;

        // 输出层：将词向量映射回词汇表概率
        auto output_layer = torch::nn::Linear(
            torch::nn::LinearOptions(embed_dim_, vocab_size_)
                .bias(false)); // Word2Vec 训练中输出层通常无偏置
        output_layer->to(device_);

        // 优化器：同时优化 embedding 层和输出层
        std::vector<torch::Tensor> all_params;
        for (auto &param : embedding_->parameters()) {
            all_params.push_back(param);
        }
        for (auto &param : output_layer->parameters()) {
            all_params.push_back(param);
        }

        torch::optim::SGD optimizer(all_params,
                                    torch::optim::SGDOptions(lr));

        // 打乱训练对
        std::random_device rd;
        std::mt19937 rng(rd());

        for (int epoch = 1; epoch <= epochs; ++epoch) {
            std::shuffle(pairs.begin(), pairs.end(), rng);

            double total_loss = 0.0;
            int batch_count = 0;
            const int batch_size = 32;

            for (size_t i = 0; i < pairs.size(); i += batch_size) {
                optimizer.zero_grad();

                // 准备批次数据
                std::vector<int64_t> centers, contexts;
                size_t end = std::min(i + batch_size, pairs.size());
                for (size_t j = i; j < end; ++j) {
                    centers.push_back(pairs[j].first);
                    contexts.push_back(pairs[j].second);
                }

                int current_batch = static_cast<int>(centers.size());

                // 中心词索引转为 tensor
                auto center_tensor = torch::tensor(
                    centers, torch::TensorOptions()
                                 .dtype(torch::kLong)
                                 .device(device_));
                // 上下文词索引转为 tensor (目标)
                auto context_tensor = torch::tensor(
                    contexts, torch::TensorOptions()
                                  .dtype(torch::kLong)
                                  .device(device_));

                // 前向传播: center → embedding → logits
                auto emb = embedding_->forward(center_tensor);
                auto logits = output_layer->forward(emb);

                // 交叉熵损失
                auto loss = torch::nn::functional::cross_entropy(
                    logits, context_tensor,
                    torch::nn::functional::CrossEntropyFuncOptions()
                        .reduction(torch::kMean));

                // 反向传播与参数更新
                loss.backward();
                optimizer.step();

                total_loss += loss.item<double>();
                batch_count++;
            }

            // 每 10 轮打印损失
            if (epoch % 10 == 0 || epoch == 1) {
                double avg_loss = total_loss / batch_count;
                std::cout << "  轮次 " << std::setw(3) << epoch << "/" << epochs
                          << " | 损失: " << std::fixed
                          << std::setprecision(4) << avg_loss << std::endl;
            }
        }

        std::cout << "训练完成。" << std::endl;
    }

    /* ===================== 辅助方法 ===================== */

    // 查询词对应的 ID（未知词返回 <UNK> 的 ID=1）
    int wordToId(const std::string &word) const {
        auto it = word_to_id_.find(word);
        return (it != word_to_id_.end()) ? it->second : 1;
    }

    // 查询 ID 对应的词
    std::string idToWord(int id) const {
        if (id >= 0 && id < static_cast<int>(id_to_word_.size()))
            return id_to_word_[id];
        return "<UNK>";
    }

    // 获取词的嵌入向量（前 4 维用于打印）
    std::vector<float> getEmbedding(const std::string &word) const {
        int id = wordToId(word);

        // 读取 embedding 权重矩阵
        auto weight = embedding_->weight; // [vocab_size, embed_dim]
        auto row = weight[id];            // [embed_dim]

        std::vector<float> vec(embed_dim_);
        for (int i = 0; i < embed_dim_; ++i) {
            vec[i] = row[i].item<float>();
        }
        return vec;
    }

    // 词汇表大小
    int vocabSize() const {
        return vocab_size_;
    }
    int embedDim() const {
        return embed_dim_;
    }

    /*
     * 创建 embedding 层（需在 vocab 构建后调用）。
     * 必须在 buildVocabulary 之后、trainSkipGram 之前调用。
     */
    void initializeEmbedding() {
        embedding_ = torch::nn::Embedding(vocab_size_, embed_dim_);
        embedding_->to(device_);
        std::cout << "Embedding 层初始化: " << vocab_size_ << " × "
                  << embed_dim_ << std::endl;
    }

private:
    /* ===================== 内置语料 ===================== */
    std::string builtInCorpus() {
        return R"(
            Deep learning is a subset of machine learning in artificial
            intelligence that has networks capable of learning unsupervised
            from data that is unstructured or unlabeled. Also known as deep
            neural learning or deep neural network. Deep learning models
            can learn from vast amounts of data. Researchers use deep
            learning for image recognition and natural language processing.
            Neural networks process data through layers of nodes. Each node
            applies a function to the input data. The output of one layer
            becomes the input to the next layer. Deep learning requires
            large datasets and powerful computing resources. Machine
            learning algorithms improve with experience. Deep neural
            networks have revolutionized computer vision and speech
            recognition. Natural language processing uses deep learning
            to understand human language. Image recognition is one of
            the most successful applications of deep learning. Speech
            recognition allows computers to understand spoken words.
            The king is a man who rules a kingdom. The queen is a woman
            who rules a kingdom. Men and women have different roles in
            society. A king is to a man as a queen is to a woman.
        )";
    }

    // 成员变量
    torch::nn::Embedding embedding_{nullptr};
    int embed_dim_;
    int vocab_size_;
    torch::Device device_{torch::kCPU};
    std::unordered_map<std::string, int> word_to_id_;
    std::vector<std::string> id_to_word_;
};

/* ============================== main ================================= */

int main() {
    std::cout << "第 7 章：Skip-gram Word2Vec 基于 LibTorch"
              << "\n"
              << std::endl;

    /*
     * 概念说明：向量空间的语义类比
     *
     * Word2Vec 将词映射到稠密向量空间，使得语义相近的词向量相近。
     * 一个著名的类比：
     *   king - man + woman ≈ queen
     * 即「国王」减去「男性」加上「女性」在向量空间中接近「女王」。
     * 这揭示了词向量能捕捉语义关系（性别、国家-首都等）。
     */

    /* ---------- 1. 创建 WordEmbeddings ---------- */
    std::cout << "【1. 创建 WordEmbeddings (embed_dim=8)】" << std::endl;
    WordEmbeddings w2v(8);
    w2v.initializeEmbedding();

    /* ---------- 2. 构建词汇表 ---------- */
    std::cout << "\n【2. 构建词汇表（使用内置语料）】" << std::endl;
    w2v.buildVocabulary("nonexistent_corpus.txt");

    /* ---------- 3. 训练 Skip-gram ---------- */
    std::cout << "\n【3. 训练 Skip-gram 模型】" << std::endl;
    // 使用内置语料进行训练
    std::string corpus = R"(
        deep learning is machine learning with deep neural networks.
        natural language processing uses deep learning for understanding.
        machine learning algorithms learn from vast amounts of data.
        deep neural networks process data through layers of nodes.
        the king is a man who rules a kingdom.
        the queen is a woman who rules a kingdom.
        a king is to a man as a queen is to a woman.
        learning from data is the key to artificial intelligence.
    )";
    w2v.trainSkipGram(corpus, 50);

    /* ---------- 4. 展示训练后的词嵌入 ---------- */
    std::cout << "\n【4. 训练后的词嵌入向量（前 4 维）】" << std::endl;

    std::vector<std::string> demo_words = {
        "king", "queen", "man", "woman",
        "learning", "deep", "data", "neural"};

    // 表头
    std::cout << "  " << std::left << std::setw(12) << "词";
    for (int d = 0; d < std::min(4, w2v.embedDim()); ++d) {
        std::cout << std::setw(10) << ("dim[" + std::to_string(d) + "]");
    }
    std::cout << std::endl;

    // 分隔线
    std::cout << "  ";
    for (int i = 0; i < 12 + 4 * 10; ++i) std::cout << "-";
    std::cout << std::endl;

    // 打印每个词的嵌入
    for (const auto &word : demo_words) {
        auto vec = w2v.getEmbedding(word);
        std::cout << "  " << std::left << std::setw(12) << word;
        for (int d = 0; d < std::min(4, w2v.embedDim()); ++d) {
            std::cout << std::setw(10) << std::fixed
                      << std::setprecision(4) << vec[d];
        }
        std::cout << std::endl;
    }

    /* ---------- 5. 向量类比演示 ---------- */
    std::cout << "\n【5. 向量类比：king - man + woman ≈ queen】" << std::endl;
    {
        auto king_vec = w2v.getEmbedding("king");
        auto man_vec = w2v.getEmbedding("man");
        auto woman_vec = w2v.getEmbedding("woman");
        auto queen_vec = w2v.getEmbedding("queen");

        std::cout << "  king  - man  + woman ≈ queen" << std::endl;
        std::cout << "  ";
        for (int d = 0; d < std::min(4, w2v.embedDim()); ++d) {
            double approx = king_vec[d] - man_vec[d] + woman_vec[d];
            std::cout << std::setw(10) << std::fixed
                      << std::setprecision(4) << approx;
        }
        std::cout << std::endl;

        std::cout << "  实际 queen 向量: ";
        for (int d = 0; d < std::min(4, w2v.embedDim()); ++d) {
            std::cout << std::setw(10) << std::fixed
                      << std::setprecision(4) << queen_vec[d];
        }
        std::cout << std::endl;

        // 计算余弦相似度（向量夹角余弦，值越接近 1 越相似）
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (int d = 0; d < w2v.embedDim(); ++d) {
            double approx = king_vec[d] - man_vec[d] + woman_vec[d];
            dot += approx * queen_vec[d];
            norm_a += approx * approx;
            norm_b += queen_vec[d] * queen_vec[d];
        }
        double cosine_sim = dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-8);

        std::cout << "\n  (king-man+woman) 与 queen 的余弦相似度: "
                  << std::fixed << std::setprecision(4) << cosine_sim
                  << std::endl;
        std::cout << "  ⚠ 注：小语料 + 低维向量，结果仅供参考。"
                  << "大规模训练后此类比会非常显著。" << std::endl;
    }

    std::cout << "\nWord2Vec Skip-gram 演示完成。" << std::endl;
    return 0;
}
