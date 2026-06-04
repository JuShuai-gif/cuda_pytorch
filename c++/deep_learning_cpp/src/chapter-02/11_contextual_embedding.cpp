/*
 * contextual_embedding.cpp
 * 第2章：C++ 中的数据准备与预处理
 *
 * 上下文嵌入（如 BERT）生成的 token 表示依赖于周围的上下文，
 * 不像静态嵌入（Word2Vec）那样每个词只有一个固定的向量。
 * 这使得同一个词在不同的句子中可以有不同的向量表示
 *（例如，"bank" 在 "river bank" 和 "bank account" 中不同）。
 *
 * 本示例通过 ONNX Runtime 运行 BERT 模型来提取 [CLS]
 * token 嵌入（句子级表示）或完整的隐状态序列。
 *
 * 依赖项：
 *   - ONNX Runtime（从 onnxruntime GitHub releases 预编译获取）
 *   - 导出为 ONNX 格式的 BERT 模型
 *
 * 注意：本示例默认使用模拟设置，因为 BERT ONNX 模型
 * 体积较大（约 400MB+）。将 USE_REAL_MODEL 设为 true 并提供模型路径
 * 即可使用实际模型运行。
 *
 * 使用场景：使用预训练语言模型进行下游任务的特征提取
 *（分类、相似度搜索、聚类）。
 */

#include <iostream>
#include <vector>
#include <array>
#include <string>

// 当你有可用的 BERT ONNX 模型时，设为 true
#define USE_REAL_MODEL 0

#if USE_REAL_MODEL
#include <onnxruntime_cxx_api.h>

// ----------------------------------------------------------------
// 通过 ONNX Runtime 运行 BERT 推理以提取 [CLS] token 嵌入。
// 输入：token ID 和注意力掩码（来自分词器）
// 输出：[CLS] token（第一个位置）的浮点向量，维度 = hiddenSize
//
// 典型的 BERT-base 模型：
//   - 输入："input_ids" [1, seq_len]、"attention_mask" [1, seq_len]
//   - 输出："last_hidden_state" [1, seq_len, 768]
//   - CLS 嵌入 = last_hidden_state[0, 0, :]
// ----------------------------------------------------------------
std::vector<float> runBertCLS(
    const std::vector<int64_t> &ids,
    const std::vector<int64_t> &mask,
    const char *model_path) {
    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "bert");
    Ort::Session session(env, model_path, Ort::SessionOptions{});

    std::array<int64_t, 2> shape{1, (int64_t)ids.size()};
    auto mem = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    // 创建输入张量
    auto t_ids = Ort::Value::CreateTensor<int64_t>(
        mem,
        const_cast<int64_t *>(ids.data()),
        ids.size(),
        shape.data(),
        2);

    auto t_mask = Ort::Value::CreateTensor<int64_t>(
        mem,
        const_cast<int64_t *>(mask.data()),
        mask.size(),
        shape.data(),
        2);

    // 输入/输出名称取决于导出的模型
    const char *in_names[] = {"input_ids", "attention_mask"};
    const char *out_names[] = {"last_hidden_state"};

    Ort::Value in_vals[] = {
        std::move(t_ids),
        std::move(t_mask)};

    auto outs = session.Run(
        Ort::RunOptions{},
        in_names,
        in_vals,
        2,
        out_names,
        1);

    // 提取 [CLS] token 嵌入（第一个位置）
    // last_hidden_state 形状：[1, seq_len, hidden_size]
    float *p = outs[0].GetTensorMutableData<float>();
    int hiddenSize = 768;
    return std::vector<float>(p, p + hiddenSize);
}
#endif

int main() {
    std::cout << "=== Contextual Embedding Demo (BERT via ONNX Runtime) ===\n\n";

    std::cout << "上下文嵌入为同一个词产生不同的向量，\n";
    std::cout << "取决于其周围的上下文：\n";
    std::cout << "  'bank' 在 'river bank' vs 'bank account' 中\n";
    std::cout << "  静态嵌入（Word2Vec）：两者使用相同的向量。\n";
    std::cout << "  上下文嵌入（BERT）：不同的向量。\n\n";

#if USE_REAL_MODEL
    // 示例：使用 BERT 分词器（WordPiece）对句子 "deep learning with c plus plus" 进行分词，
    // 然后运行推理。
    // 生产环境中，使用适当的分词器（例如，通过 HuggingFace
    // tokenizers C++ API 或 SentencePiece）。

    // 模拟的 token ID 和注意力掩码（应由分词器生成）
    std::vector<int64_t> ids = {101, 2784, 4083, 2007, 3124, 2207, 2207, 102};
    std::vector<int64_t> mask(ids.size(), 1);

    const char *modelPath = "path/to/bert-base-uncased.onnx";

    try {
        auto embedding = runBertCLS(ids, mask, modelPath);
        std::cout << "BERT [CLS] embedding (first 8 dims): [";
        for (int i = 0; i < 8; ++i)
            std::cout << embedding[i] << (i < 7 ? ", " : "");
        std::cout << "...]\n";
        std::cout << "维度：" << embedding.size()
                  << " （BERT-base 的 hidden_size）\n";
    } catch (const std::exception &e) {
        std::cerr << "ONNX Runtime error: " << e.what() << "\n";
        std::cerr << "Make sure the BERT ONNX model exists at: "
                  << modelPath << "\n";
    }
#else
    std::cout << "[模拟模式] 将 USE_REAL_MODEL 设为 1 并提供 BERT\n";
    std::cout << "ONNX 模型即可运行实际推理。\n\n";

    std::cout << "实际推理的设置步骤：\n";
    std::cout << "  1. 将 BERT 导出为 ONNX（通过 HuggingFace optimum-cli）：\n";
    std::cout << "     python -m optimum.exporters.onnx --model bert-base-uncased bert_onnx/\n";
    std::cout << "  2. 或从 HuggingFace 模型中心下载（ONNX 格式）\n";
    std::cout << "  3. 设置模型路径并将 USE_REAL_MODEL=1\n";
    std::cout << "  4. 还需要一个分词器（WordPiece/BPE）将文本转换为 token ID\n\n";

    std::cout << "使用虚拟数据的模拟推理：\n";
    std::string text = "deep learning with c plus plus";
    std::cout << "  输入文本：\"" << text << "\"\n";
    std::cout << "  Token ID：[101, 2784, 4083, ...]（来自 WordPiece 分词器）\n";
    std::cout << "  输出：[CLS] 嵌入向量（BERT-base 为 768 维）\n";
    std::cout << "  [CLS] token 聚合了句子级信息。\n\n";

    std::cout << "使用场景：\n";
    std::cout << "  - 句子相似度（[CLS] 向量之间的余弦相似度）\n";
    std::cout << "  - 文本分类（将 [CLS] 输入到分类头）\n";
    std::cout << "  - 语义搜索（嵌入空间中的最近邻搜索）\n";
#endif

    return 0;
}
