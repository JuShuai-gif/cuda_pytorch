# Lecture 01: 课程概览与 Tokenization

## 本讲核心问题

1. 为什么 CS336 这门课的核心公式是 `accuracy = efficiency x resources`，而不是 `accuracy = model_size x data`？
2. Tokenization 到底在做什么？为什么 BPE 是 LLM 的事实标准？
3. 为什么说 tokenizer 选择错误会直接摧毁训练效率？
4. Tokenizer-free 模型（如 ByT5、BLT）能否取代 subword tokenizer？
5. 当 context length 达到 1M tokens 时，tokenization 会引发哪些新问题？

## 通俗解释

### accuracy = efficiency x resources

想象你要搬家，家里有 100 箱东西。你有两个选择：一是租一辆大卡车一趟拉完（堆资源），二是优化打包方式，用小货车跑三趟也能搬完（提效率）。大模型训练同理——你可以买更多 GPU（堆 resources），也可以设计更好的算法和系统（提 efficiency）。最终效果 = 效率 x 资源。

CS336 跟大多数深度学习课程不同的地方就在这里：它不教你设计新架构发 paper，而是教你**在给定资源下，怎么榨干每一滴算力**。这才是工业界真正需要的技能。

### BPE ≈ 把常见词组合"打包"

英文单词 "unfortunately" 如果按字母输入，需要 12 步。BPE（Byte Pair Encoding）的做法是先统计语料库中最常出现的字符对（比如 "un"），将它们合并成一个新 token。反复执行这个过程，最终 "unfortunately" 可能被拆成 `["un", "fortunate", "ly"]` 三个 token。

这就好比快递打包——不是把每个零件单独寄，而是把经常一起出现的东西装进一个箱子。常见词组自动变成一个大 token，罕见词则拆得更细。结果就是：**vocab 大小固定（比如 32K）、覆盖率极高、几乎没有 OOV（Out-of-Vocabulary）问题**。

### 为什么 tokenizer 影响训练效率

假设你的 tokenizer 平均每个 token 对应 3.5 个字符。一段英文可能 1000 个 token 就表达完了。但如果 tokenizer 设计得不好，每个 token 只对应 1.5 个字符，同样的信息需要 2300 个 token 才能表达。

在 Transformer 中，attention 的计算复杂度是 O(N^2)（N 是序列长度）。这意味着：**如果 tokenizer 让 N 变成了 2 倍，attention 的计算量就变成了 4 倍**。而且 KV cache 的大小也直接正比于 token 数量。所以在推理阶段，tokenizer 的效率直接转化为延迟和吞吐。

## 数学公式 + 工程意义

### BPE 合并规则

BPE 的核心操作是统计训练语料中所有相邻 token pair 的频率，选择频率最高的 pair 合并：

```
new_token = argmax_{a,b} count(a, b)
```

这不是为了"数学美感"——它的工程意义是：**在固定 vocab size 约束下，最大化语言信息的压缩率**。vocab size 不能无限增大（embedding 矩阵大小 = vocab_size x d_model，占大量显存），所以需要在压缩率和显存之间做 trade-off。

### 为什么选择 vocab size = 32K

| vocab size | embedding 参数量 (d_model=4096) | 覆盖率 | 推理效率 |
|-----------|-------------------------------|--------|---------|
| 16K       | 65M                           | ~98%   | 低（too many tokens） |
| 32K       | 131M                          | ~99.5% | 平衡 |
| 64K       | 262M                          | ~99.8% | 渐降 |
| 128K      | 524M                          | ~99.9% | 显存压力大 |

Llama 系列使用 32K vocab，GPT-4 使用 ~100K vocab。DeepSeek-V3 也使用了 128K vocab——这是因为它有更大的 d_model 和 MoE 架构，embedding 占比相对较小。

### Context length 与 tokenizer 的关系

如果上下文长度从 4K 扩展到 128K（32x），attention 计算量从 O(4K^2) 变成 O(128K^2)——

```
attention_flops ≈ 2 * batch * heads * N^2 * head_dim
```

这意味着同样的推理请求，延迟可能增加 1000x 以上。这也是 FlashAttention 和 sparse attention 等优化的出发点。

## 工业界真实实现

### SentencePiece 与 Llama 系列

Llama 和 Llama 2 使用的是 SentencePiece 实现的 BPE tokenizer，vocab size = 32,000。Llama 3 将其扩展到 128,000。这个选择的背后逻辑：

```
model_params = vocab_size * d_model  (embedding) 
             + n_layers * (4*d_model^2 + 8*d_model^2)  (attention + MLP)
```

对于 Llama 2-7B（d_model=4096, n_layers=32），embedding 参数 = 32K x 4096 ≈ 131M，占总参数 ~2%。vocab 从 32K 增到 128K 会让 embedding 变成 524M，占比升到 ~7%——对于小模型来说这个比重已经不容忽视。

### DeepSeek-V3 的 tokenizer 选择

DeepSeek-V3 使用 128K vocab 的 BPE tokenizer，并针对中文做了优化。一个关键决策是加入了大量中文常用字符和词组，使得中文文本的 token 效率（每 token 表达的信息量）显著高于英文 tokenizer 直接用到中文的效果。

### GPT-4 的 tokenizer

GPT-4 使用 ~100K vocab 的 tokenizer（基于 cl100k_base）。OpenAI 发现较大的 vocab 在长文本场景下能减少 token 数量，从而降低 API 成本。这也是 `tiktoken` 库直接统计 token 数来计费的原因。

### Tokenizer-free 模型

ByT5 直接使用 UTF-8 bytes 作为 token（vocab size = 256），完全消除了 tokenizer。但问题也很明显：同样的文本，token 数量膨胀约 4x，训练和推理都慢很多。

Meta 最新的 BLT（Byte Latent Transformer）采用折中方案：用轻量级 local encoder 将 bytes 编码为 patches，大模型处理 patches，再用 local decoder 解码回 bytes。这是一种动态 tokenization——信息密度高的区域（如英文单词）被紧凑编码，信息密度低的区域（如空格重复）被高效跳过。

HNet（Hierarchical Network）则使用多层 tokenization，不同粒度在不同层处理，类似人类阅读时同时关注字、词、句三个层次。

## CUDA/GPU 视角

### Tokenization 在 CPU 上的 bottleneck

实际推理流程中，tokenization 发生在 CPU 端，然后将 token ids 传输到 GPU。对于长文本输入（比如 100K tokens 的文档），CPU 上的 BPE tokenization 本身就需要数秒，成为端到端延迟的组成部分。

更重要的问题是：**tokenization 是严格串行的**——每个 token 依赖于前一个 token 的合并决策。BPE 编码过程无法在 GPU 上并行，这是 tokenizer-free 模型的一个重要动机。

### Embedding lookup 的 GPU 操作

从 GPU 角度看，token ids -> embeddings 是一个 gather 操作：

```python
# This is a gather (memory-bound), not a compute operation
embeddings = embedding_table[token_ids]  # [batch, seq_len, d_model]
```

这个操作是 **memory-bound** 的：每次 forward pass 都需要从 HBM 读取整个 embedding table（vocab_size x d_model 个 float），但实际只用了 batch_size x seq_len 个 embedding。对于大 vocab（100K+），embedding 的 HBM 读取量可能超过 attention 的计算量。

### KV cache 与 tokenization

推理时，每个新 token 都需要更新 KV cache。KV cache 的大小 = n_layers x 2 x batch x seq_len x d_model。这是扩展 context length 最大的 memory bottleneck。tokenizer 如果让 seq_len 翻倍，KV cache 的显存占用也翻倍。

## 本讲与整个 LLM 系统的关系

```
Tokenizer -> Embedding -> Attention -> MLP -> Loss -> Optimizer -> Distributed -> Inference
    ^                                                                                     |
    |_____________________________ 本讲（入口） _____________________________________________|
```

Tokenization 是整个 LLM pipeline 的入口。它的决策影响：

- **Embedding 层**：vocab size 决定 embedding 矩阵的显存占用
- **Attention 层**：token 数量决定 O(N^2) 的计算量
- **推理**：token 数量直接决定首 token 延迟和 KV cache 大小
- **训练**：token 效率影响每个 step 能处理的有效信息量

如果把 LLM 系统想象成一条高速公路，tokenization 就是收费站——收费站效率直接决定了整条路的吞吐上限。

## 面试问题

**Q1: BPE 训练过程中，如何高效统计相邻 token pair 的频率？内存不够怎么办？**

A: 标准做法是维护一个 pair -> count 的哈希表。第一遍扫描建立所有相邻 pair 的计数。每次合并后只需要更新受影响的 pair（被合并的两个 token 的前后相邻 pair）。对于大规模语料（TB 级），需要分块处理——每块独立统计，最后合并计数。SentencePiece 使用 memory-efficient 的算法，只维护 top-K 频繁的 pair。

**Q2: 为什么 Llama 3 的 vocab size 从 32K 扩大到 128K？**

A: 三个原因：(1) Llama 3 支持多语言，需要覆盖更多字符；(2) 模型更大（8B 起步），embedding 占比降低；(3) 减少 token 数量能直接降低推理成本。但代价是 embedding 参数量从 131M 增加到 524M，训练时 embedding 的 HBM 读取压力也更大。

**Q3: 如果让你从零设计一个中文 LLM 的 tokenizer，你会怎么决策？**

A: 考虑四点：(1) 语料——中文需要覆盖 GB2312/GBK 字符集 + 常用词组，vocab 至少 32K；(2) 压缩率——中文字符信息密度高，tokenizer 应该在常用词级别合并，避免每个汉字一个 token；(3) 训练数据——如果训练语料主要是代码（含大量英文），需要兼顾英文 token 效率；(4) 推理场景——如果面向长文档分析，需要优先考虑 token 效率而非 embedding 显存。会从 32K vocab、BPE + byte fallback（类似 GPT-4 的 cl100k_base）的配置开始实验，根据 token efficiency（压缩率）和 downstream perplexity 做 ablation。

**Q4: tokenizer-free 模型（如 ByT5）为什么不流行？**

A: 核心问题是效率——同样的文本，token 数量膨胀 4x 意味着 attention 计算量增加 16x。虽然省去了 tokenizer 工程，但训练和推理成本飙升。BLT 等动态 patching 方法可能改变这个局面，但目前 BPE 仍是 Pareto-optimal 的选择。
