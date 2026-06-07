# Lab 01: Tokenization 与训练基础

## 任务目标

通过本实验，你将：

1. 理解并实现 Byte-Pair Encoding (BPE) tokenization 算法
2. 理解 cross-entropy loss 在语言模型训练中的数学含义
3. 实现 perplexity 计算，并理解它如何反映模型质量
4. 完成一个简单的 tokenizer 训练流程

## 实验任务

### Task 1: 实现 BPE Tokenizer (50%)

在 `starter.py` 的骨架代码基础上完成以下实现：

1. 统计训练语料中所有相邻 token pair 的出现频率
2. 找到频率最高的 pair 并进行合并 (merge)
3. 重复上述步骤直到达到目标 vocab size
4. 实现 `encode()` 将文本转换为 token IDs
5. 实现 `decode()` 将 token IDs 转换回文本

### Task 2: 计算 Cross-Entropy Loss (25%)

在 `starter.py` 中实现：

1. 手动实现 cross-entropy loss 的计算公式
2. 与 PyTorch 的 `F.cross_entropy()` 进行对比验证
3. 理解 label smoothing 对 loss 的影响

### Task 3: 计算 Perplexity (25%)

在 `starter.py` 中实现：

1. 从 cross-entropy loss 推导并计算 perplexity
2. 理解：perplexity = exp(cross-entropy)
3. 在实际模型输出上验证 perplexity 的计算

## 验收标准

- [ ] BPE tokenizer 的 `encode()`/`decode()` 能够正确往返 (round-trip)
- [ ] merge 规则与 HuggingFace `tokenizers` 库结果一致
- [ ] cross-entropy loss 实现与 `F.cross_entropy()` 误差 < 1e-6
- [ ] perplexity 计算正确 (p=exp(loss))
- [ ] 所有单元测试通过 (`python test.py`)

## 参考资料

- [BPE 原始论文 (Sennrich et al., 2016)](https://arxiv.org/abs/1508.07909)
- [GPT-2 tokenizer 分析](https://huggingface.co/learn/nlp-course/chapter6/5)
- [tiktoken 源码](https://github.com/openai/tiktoken)

## 时间估计

约 2-3 小时
