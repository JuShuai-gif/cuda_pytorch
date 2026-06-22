# 第 9 章：Transformer 与大语言模型微调（C++ 实现）

基于 *Deep Learning with C++*（Packt，ISBN 9781835880036）第 9 章，第 342–404 页。

---

## 目录

1. [章节概述](#章节概述)
2. [文件索引](#文件索引)
3. [编译与运行](#编译与运行)
4. [技术速查](#技术速查)
5. [PDF 完整内容对照](#pdf-完整内容对照)
6. [注意事项](#注意事项)

---

## 章节概述

第 9 章深入讲解 Transformer 架构的核心原理、完整实现，以及大语言模型的微调与压缩技术。从 Self-Attention 的数学基础出发，逐步构建 Multi-Head Attention、位置编码、完整的 Encoder-Decoder 架构，最后介绍分布式训练（DDP/FSDP）和模型压缩（知识蒸馏、剪枝、量化）三大实用技术。

### 六大核心主题

| 主题 | 说明 |
|------|------|
| Attention 机制 | 从 Scaled Dot-Product 到 Multi-Head Attention 的完整推导与实现 |
| 位置编码 | 可学习编码 → 正弦编码 → RoPE → ALiBi 的演进路线 |
| Transformer 架构 | Encoder（双向自注意力）与 Decoder（因果掩码 + 交叉注意力）的完整构建 |
| 预训练范式 | BERT（双向编码器）vs GPT（自回归解码器）的架构差异与适用场景 |
| 分布式训练 | DDP（数据并行）与 FSDP（全分片数据并行）的选择策略与实现 |
| 模型压缩 | 知识蒸馏、结构化/非结构化剪枝、PTQ/QAT 量化的完整流水线 |

### 五大挑战

| 挑战 | 说明 |
|------|------|
| 计算复杂度 | Self-Attention 的 O(n²) 复杂度对长序列构成瓶颈 |
| 位置信息注入 | Transformer 位置无关性要求精心设计位置编码方案 |
| 显存限制 | 大模型（>10B 参数）需多 GPU 甚至多节点部署 |
| 推理延迟 | 自回归生成逐 token 解码，需模型压缩加速 |
| 训练稳定性 | 深层 Transformer 需残差连接 + LayerNorm 缓解梯度问题 |

---

## 文件索引

### 一、Attention 机制 — PDF 第 342–349 页

| 文件 | PDF 页 | 涵盖知识点 | 依赖 |
|------|--------|-----------|------|
| `01_self_attention.cpp` | 342–345 | Scaled Dot-Product Attention（Q/K/V 计算、√d_k 缩放、Softmax 归一化） | LibTorch |
| `02_multi_head_attention.cpp` | 346–349 | Multi-Head Attention（头拆分、并行注意力、拼接投影），因果掩码 | LibTorch |

### 二、位置编码 — PDF 第 349–357 页

| 文件 | PDF 页 | 涵盖知识点 | 依赖 |
|------|--------|-----------|------|
| `03_positional_encoding.cpp` | 349–357 | 正弦位置编码、RoPE（旋转位置编码）、ALiBi（线性偏置）、可学习位置嵌入 | LibTorch |

### 三、Transformer 架构 — PDF 第 358–367 页

| 文件 | PDF 页 | 涵盖知识点 | 依赖 |
|------|--------|-----------|------|
| `04_transformer_encoder.cpp` | 358–361 | Encoder 层（Self-Attention→Add&Norm→FFN→Add&Norm），完整编码器堆叠 | LibTorch |
| `05_transformer_decoder.cpp` | 361–367 | Decoder 层（Masked Self-Attn→Cross-Attn→FFN），训练 vs 推理（自回归生成） | LibTorch |

### 四、模型压缩 — PDF 第 371–404 页

| 文件 | PDF 页 | 涵盖知识点 | 依赖 |
|------|--------|-----------|------|
| `06_knowledge_distillation.cpp` | 382–385 | 教师-学生训练，暗知识，温度缩放，KL 散度 + 交叉熵联合损失 | LibTorch |
| `07_pruning.cpp` | 385–389 | 非结构化剪枝（权重幅值），结构化剪枝（注意力头级），迭代剪枝 + 再训练 | LibTorch |
| `08_quantization.cpp` | 389–404 | 对称/非对称量化，Min-Max/百分位数校准，伪量化（QAT），每通道量化 | LibTorch |

### 五、分布式训练 — PDF 第 369–381 页（代码讲解，非独立运行）

| 技术 | 说明 |
|------|------|
| DDP（Distributed Data Parallel） | 模型完整复制到各 GPU，通过 NCCL All-Reduce 同步梯度。在 `06_knowledge_distillation.cpp` 的注释中提供了 DDP 伪代码 |
| FSDP（Fully Sharded Data Parallel） | 参数/梯度/优化器状态分片到所有 GPU，训练时 All-Gather 重建。适用于单 GPU 放不下的大模型 |

---

## 编译与运行

### 环境要求

```bash
# 必需
C++17 编译器（GCC 11+ / Clang 14+）
CMake 3.22+
LibTorch → $HOME/Downloads/libtorch

# 可选（分布式训练示例需要，代码中以注释形式提供）
NCCL 2.x → apt install libnccl-dev
```

### 编译

```bash
cd c++/deep_learning_cpp/build
cmake ..
cmake --build . --target <target_name> -j$(nproc)
```

### 运行示例

```bash
# Attention 机制
./build/chapter09/self_attention
./build/chapter09/multi_head_attention

# 位置编码
./build/chapter09/positional_encoding

# Transformer 架构
./build/chapter09/transformer_encoder
./build/chapter09/transformer_decoder

# 模型压缩
./build/chapter09/knowledge_distillation
./build/chapter09/pruning
./build/chapter09/quantization
```

---

## 技术速查

### Attention 机制

| 概念 | 公式 / 说明 | 关键点 |
|------|-------------|--------|
| Scaled Dot-Product | `softmax(QK^T / √d_k) V` | √d_k 防止点积过大导致 Softmax 梯度消失 |
| Self-Attention | Q=K=V 来自同一输入 | 序列内每个词关注所有其他词 |
| Cross-Attention | Q 来自 Decoder，K=V 来自 Encoder | 解码时"查阅"编码器的输出 |
| Causal Mask | 上三角置 -∞ | 保证自回归生成不泄露未来信息 |

### Multi-Head Attention

| 参数 | 说明 |
|------|------|
| `num_heads` | 并行的注意力头数（典型：8、12、16） |
| `d_k = d_model / num_heads` | 每头维度，总计算量与单头相同 |
| 头间分工 | 不同头学习不同模式：语法、语义、长距离依赖、局部交互 |

### 位置编码对比

| 方法 | 类型 | 优势 | 劣势 |
|------|------|------|------|
| 正弦编码 | 绝对、固定 | 无需学习参数，可泛化到训练外长度 | 高维空间运动混乱，模型易过拟合训练位置 |
| 可学习嵌入 | 绝对、可学习 | 简单直接 | 无法泛化到未见长度 |
| RoPE | 相对、旋转 | 天然相对位置，长距离衰减，无额外参数 | 实现稍复杂 |
| ALiBi | 相对、线性偏置 | 极简，外推能力强 | 单向偏置（仅用于解码器） |
| T5 偏置 | 相对、可学习 | 灵活建模相对距离 | 需学习额外参数 |

### Transformer 架构

| 组件 | Encoder | Decoder |
|------|---------|---------|
| Self-Attention | 双向（无掩码） | 因果掩码（只看左侧） |
| Cross-Attention | 无 | Q←Decoder, K,V←Encoder |
| 用途 | 理解任务（分类、NER、QA） | 生成任务（翻译、摘要、对话） |
| 代表模型 | BERT | GPT |

### BERT vs GPT

| 维度 | BERT（编码器） | GPT（解码器） |
|------|---------------|---------------|
| 信息流 | 双向 | 单向（左到右） |
| 擅长 | 理解任务 | 生成任务 |
| 注意力 | 双向自注意力 | 因果掩码注意力 |
| 预训练目标 | MLM（掩码语言模型）+ NSP | Next Token Prediction |
| 规模演进 | BERT-base(110M)→BERT-large(340M) | GPT-1(117M)→GPT-4(~1.8T) |

### 分布式训练

| 策略 | 适用场景 | 通信需求 | 内存效率 |
|------|----------|----------|----------|
| DDP | 模型适合单 GPU，数据多 | All-Reduce 梯度（每步） | 低（每 GPU 完整副本） |
| FSDP | 模型 > 单 GPU 显存 | All-Gather + Reduce-Scatter | 高（分片存储） |

### 模型压缩

| 方法 | 压缩率 | 精度损失 | 适用阶段 | 核心操作 |
|------|--------|----------|----------|----------|
| 知识蒸馏 | 模型大小任意缩小 | 中等（可恢复） | 训练后 | 教师软标签 + 学生联合损失 |
| 非结构化剪枝 | 50–95% 稀疏 | 低–中 | 训练后 + 微调 | 权重幅值排序置零 |
| 结构化剪枝 | 20–50% FLOPs | 中 | 训练后 + 微调 | 移除注意力头/层/神经元 |
| PTQ 量化 | 4× 压缩 | 低–中 | 训练后 | Min-Max/百分位数校准 |
| QAT 量化 | 4× 压缩 | 极低 | 训练中 | 伪量化模拟量化噪声 |

---

## PDF 完整内容对照

以下是 PDF 第 342–404 页的完整纲要，标注了各节对应的实现文件：

| PDF 页 | 内容 | 实现文件 |
|--------|------|---------|
| 342–343 | 章节概述、技术要求 | `note.md` |
| 342–345 | Scaled Dot-Product Attention 数学推导与 C++ 实现 | `01_self_attention.cpp` |
| 346–349 | Multi-Head Attention 原理与实现 | `02_multi_head_attention.cpp` |
| 349–352 | 可学习位置嵌入、正弦位置编码 | `03_positional_encoding.cpp` |
| 352–354 | RoPE（旋转位置编码）数学推导 | `03_positional_encoding.cpp` |
| 354–357 | T5 相对偏置、ALiBi 线性偏置 | `03_positional_encoding.cpp` |
| 358–361 | Transformer Encoder 完整实现（层堆叠、FFN、LayerNorm） | `04_transformer_encoder.cpp` |
| 361–364 | Transformer Decoder（因果掩码、交叉注意力、训练模式） | `05_transformer_decoder.cpp` |
| 364–367 | 自回归推理生成、训练 vs 推理对比 | `05_transformer_decoder.cpp` |
| 367–369 | BERT 架构（双向编码器、MLM+NSP） | `note.md`（对比表格） |
| 369–371 | GPT 架构（自回归解码器、GPT-1→GPT-5 演进） | `note.md`（对比表格） |
| 371–376 | DDP 分布式数据并行（DistributedSampler、梯度 All-Reduce） | 各文件注释中 |
| 376–381 | FSDP 全分片数据并行（All-Gather、Reduce-Scatter、参数分片） | 各文件注释中 |
| 382–385 | 知识蒸馏（教师模型、高温软目标、联合损失） | `06_knowledge_distillation.cpp` |
| 385–387 | 非结构化剪枝（幅值 + 梯度 + Wanda） | `07_pruning.cpp` |
| 387–389 | 结构化剪枝（层级、头级、神经元级） | `07_pruning.cpp` |
| 389–392 | 量化基础（对称/非对称、范围选择） | `08_quantization.cpp` |
| 392–398 | PTQ 训练后量化（校准、每通道量化） | `08_quantization.cpp` |
| 398–404 | QAT 量化感知训练（伪量化、量化梯度、宽最小值） | `08_quantization.cpp` |
| 404 | 章节问题、拓展阅读（27 条参考资源） | — |

---

## 注意事项

### 外部库依赖

| 文件 | 需要的外部库 | 说明 |
|------|-------------|------|
| 全部文件 | LibTorch（C++17） | 位于 `$HOME/Downloads/libtorch` |
| DDP/FSDP（注释） | NCCL 2.x | 分布式训练需 NCCL，本章以注释/伪代码形式提供 |

### PDF 中提及但未独立实现的用法

| 知识点 | PDF 页 | 说明 |
|--------|--------|------|
| DDP 完整训练循环 | 371–376 | 需要多 GPU + NCCL 环境，在 `06_knowledge_distillation.cpp` 中以伪代码展示 |
| FSDP All-Gather/Reduce-Scatter | 376–381 | 需多 GPU + PyTorch FSDP API，在注释中提供了算法步骤 |
| BERT/GPT 完整预训练 | 367–371 | 需要大规模语料 + 多 GPU，仅在 `note.md` 中做了架构对比分析 |
| WordPiece/Byte-Pair Encoding 分词 | — | 预训练模型依赖特定分词器，使用 ONNX Runtime 加载（参见第 2 章 `11_contextual_embedding.cpp`） |
| T5 相对位置偏置 | 354 | 在 `03_positional_encoding.cpp` 中以注释形式说明了 T5 偏置的设计思路 |
| Wanda 剪枝准则 | 387 | 在 `07_pruning.cpp` 中以注释形式说明 Wanda 的激活感知评分公式 |
| 每通道量化 | 396–398 | 在 `08_quantization.cpp` 中以函数实现 |

### 其他注意事项

- 所有 LibTorch 示例使用 C++17，LibTorch 路径为 `$HOME/Downloads/libtorch`。
- Attention 计算涉及大量 `batch matrix multiply (bmm)` 和 `transpose` 操作，确保 PyTorch 版本 ≥ 2.0。
- Transformer Encoder/Decoder 的层数是可配置参数，PDF 原始论文使用 6 层。
- 量化示例中的校准数据为随机生成，实际使用时应使用真实校准数据集。
- 知识蒸馏的温度参数 T 通常取 2–20，更高的 T 产生更软的分布。
