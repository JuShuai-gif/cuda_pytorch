# Lecture 12: 评估 Evaluation

## 本讲核心问题

大语言模型的评估是 AI 领域最具争议的问题之一。我们如何客观衡量一个 LLM 的"能力"？Perplexity 能告诉我们什么、不能告诉我们什么？为什么 Chatbot Arena 的 Elo 排名和 MMLU 分数经常不一致？本讲回答三个核心问题：(1) 评估指标的数学基础（Perplexity）及其局限；(2) 主流 benchmark 的分类和适用场景；(3) 评估中的陷阱：data contamination、benchmark hacking 和 Goodhart's Law。

---

## 通俗解释

**Perplexity**（困惑度）的核心直觉：
想象一个学生在读一本教科书。如果这本书的内容和他已知的知识非常匹配，他读到每一段都不会太惊讶——"嗯，果然如此"（低 Perplexity）。反之，如果他每读一页都不断惊呼"什么？！原来是这样？！那也太奇怪了！"——说明这本书的内容对他来说是陌生的（高 Perplexity）。

Perplexity 衡量的是：**模型对下一个 token 的平均惊讶程度**。数学上是交叉熵损失的指数：

```
Perplexity = exp(cross_entropy_loss) = exp(-(1/N) Σ log P(token_i | context))
```

如果一个模型对语料库的 Perplexity 是 10，意味着平均来说，模型对每个位置的预测就像在 10 个等可能的选项中选一个——相当于瞎猜的难度。

**为什么仅靠 Perplexity 不够？**
Perplexity 告诉你模型对文本的"熟悉程度"，但不告诉你模型能否：
- 解决一个数学证明题（MMLU/ARC 测）
- 理解不可靠叙述者的弦外之音（复杂阅读理解测）
- 写一段用户满意的对话（Chatbot Arena 测）
- 调用 API 并在出错时重试（SWE-bench 测）

正如一个学生能把课本倒背如流（低 Perplexity），但不一定能在考试中灵活运用知识（高 Benchmark）。

**Data Contamination（数据污染）**：
假设你偷偷拿到了考试答案，在考试前背下来。考试时你当然能得 100 分——但这 100 分不代表你真正"理解"了知识。同样，如果 LLM 的训练数据恰好包含了 benchmark 的题目和答案，它的高分可能是"背答案"而非"真理解"。这就是数据污染的假象。

**Goodhart's Law**：当一个指标成为目标，它就不再是一个好指标。Benchmark hacking 就是 Goodhart's Law 在 AI 评估中的体现：当研发团队针对 MMLU 优化模型时，MMLU 分数上升了，但模型的真实能力可能并没有相应提升。

---

## 数学公式 + 工程意义

### 1. Perplexity 的完整定义

对于 token 序列 w₁, w₂, ..., w_N 和语言模型 P：

```
Cross Entropy Loss:
L = -(1/N) Σ_{i=1}^{N} log P(w_i | w_{<i})

Perplexity:
PPL = exp(L) = exp(-(1/N) Σ log P(w_i | w_{<i}))
```

**Perplexity 的直观含义**：
- PPL = 1：完美预测——模型 100% 确定下一个 token（不可能）
- PPL = V（词表大小，如 128K）：随机均匀猜测（最差情况）
- PPL = 5：模型在 5 个选项中犹豫不决——相当好的模型

**现代 LLM 的典型 Perplexity**：

| 模型 | WikiText-103 PPL | 上下文长度 | 备注 |
|------|-----------------|-----------|------|
| GPT-2 (1.5B) | 16.38 | 1024 | 2019 基线 |
| GPT-3 (175B) | 10.69 | 2048 | — |
| Llama 1 (65B) | 5.87 | 2048 | 开源模型突破 |
| Llama 2 (70B) | 5.01 | 4096 | — |
| Llama 3 (70B) | ~4.5 | 8192 | — |
| 人类 | ~2-3 | — | 估计值 |

注意：不同 tokenizer 的 Perplexity 不可直接比较——tokenizer 的词汇量和切分方式会影响 PPL 数值。这就是为什么 Megatron-LM 论文中强调："不同 tokenizer 的 perplexity 要小心比较。"

### 2. Benchmark 分类体系

| 类别 | 代表数据集 | 评估方式 | 衡量能力 |
|------|-----------|----------|----------|
| **语言建模** | WikiText-103, Penn Treebank, C4 | Perplexity | 基础语言能力 |
| **知识/考试** | MMLU, ARC, HellaSwag | 多选题准确率 | 知识广度、推理 |
| **阅读理解** | LAMBADA, RACE, SQuAD | 准确率/F1 | 文本理解 |
| **数学推理** | GSM8K, MATH, AIME | 准确率 | 数学能力 |
| **代码能力** | HumanEval, MBPP, SWE-bench | pass@k, resolve rate | 编程能力 |
| **对话/指令** | MT-Bench, AlpacaEval, Chatbot Arena | 胜率/评分/Elo | 对话质量 |
| **通用能力** | BIG-bench, HELM | 多维评分 | 综合能力 |

### 3. Chatbot Arena 的 Elo 评分系统

Chatbot Arena 使用 Bradley-Terry 模型计算 Elo 评分：

```
P(model_A beats model_B) = 1 / (1 + 10^{(R_B - R_A) / 400})
```

其中 R_A 和 R_B 是模型的 Elo 评分。每次用户盲测投票后，更新 Elo：

```
R_A_new = R_A_old + K × (outcome - expected)
```

K 是更新系数（如 K=32），outcome 为 1（A 赢）、0.5（平）、0（A 输）。

**Chatbot Arena 的优势**：直接衡量用户偏好，包含"不可量化的质量维度"（如对话风格、幽默感、共情能力）。

**局限**：样本偏差（投票者通常是 AI 从业者，不代表普通用户）、评估成本高（需要大量人工投票）。

### 4. SWE-bench 的评估方法

SWE-bench 评估 LLM 解决真实 GitHub issue 的能力。流程：

1. 给模型一个 GitHub issue + 对应的代码库
2. 模型生成 patch 修复 issue
3. 运行原 issue 对应的单元测试（这些测试在 issue 修复前是失败的）
4. **Resolve rate** = 通过的 issue 数 / 总 issue 数

SWE-bench Verified（严格筛选后的可靠子集）上：

| 模型 | Resolve Rate | 日期 |
|------|-------------|------|
| GPT-4 | 1.7% | 2024.01 |
| GPT-4 + SWE-agent | 12.5% | 2024.03 |
| Claude 3.5 Sonnet + Agent | 49% | 2024.10 |
| OpenAI o3 | 71.7% | 2025.02 |

**关键洞察**：从 GPT-4 到 o3，半年内从 1.7% 跳到 71.7%——这不是模型聪明了 40 倍，而是 agentic 框架（让模型有工具、有循环）解锁了能力。这也说明**评估不仅要测模型，还要考虑模型的使用方式**。

---

## 工业界真实实现

### Llama 3 的评估策略

Meta 在 Llama 3 报告中使用了多层评估体系：
1. **预训练阶段**：WikiText-103 PPL、C4 PPL、验证集 loss 曲线
2. **后训练/对齐阶段**：自动化 benchmarks（MMLU, GSM8K, HumanEval）+ 人工评测（human evaluation）
3. **安全评估**：CyberSecEval 测代码安全、红队测试测有害输出
4. **与 Llama 2 的对比**：每个 benchmark 上标注相对于 Llama 2 的提升

Llama 3 报告特别强调：他们在训练过程中**隔离了 benchmark 数据**，使用 n-gram overlap detection 和 embedding similarity 来检测并移除训练数据中可能泄露的 benchmark 样本。

### DeepSeek-V3 的评估方法

DeepSeek-V3 的评估亮点：
1. **多语言**：中文 benchmark（C-Eval, CMMLU）+ 英文 benchmark + 代码 benchmark
2. **长上下文**：在 128K 上下文下测试"大海捞针"（Needle-in-a-Haystack）
3. **推理能力**：AIME 2024 数学竞赛题、GPQA（研究生级物理/化学/生物）

DeepSeek-V3 报告中的一个细节：他们注意到 MMLU 数据可能存在于 CommonCrawl 训练数据中，因此对每条 benchmark 数据做了**去污染**处理（decontamination），去除了高 n-gram overlap 的样本。

### 评估基础设施：lm-evaluation-harness

EleutherAI 的 **lm-evaluation-harness** 是工业界的标准评估框架，支持 200+ benchmarks：

```python
# 使用 lm-eval 评估 Llama 3 8B 在 MMLU 上
lm_eval --model hf \
    --model_args pretrained=meta-llama/Meta-Llama-3-8B \
    --tasks mmlu \
    --device cuda:0 \
    --batch_size auto
```

核心设计：**标准化**——确保所有模型在相同的 prompt template、相同的 few-shot 示例、相同的评分逻辑下进行比较。这解决了早期评估中各论文"自选 prompt 格式"导致的不可比性问题。

---

## CUDA/GPU 视角

评估与 GPU 的关系相对间接，但有一个重要方面：**高效评估需要批量推理**。和训练不同，评估是纯推理，decode 是 memory-bound。评估 pipeline 的优化包括：

- **KV Cache 复用**：在多选题评估中，同一份 prompt + 不同选项常共享 prefix
- **tensor parallelism**：大模型评估需要跨多卡，Megatron-LM 的评估模式使用 TP
- **量化评估**：INT8/FP8 量化后模型在 benchmark 上的分数下降通常 < 0.5%，可以用量化加速评估

大规模评估（如 HELM，需评估 30+ 模型 × 40+ benchmarks）的总 GPU 小时可达数万。Google 的 HELM 评估使用 TPU v4 pod，利用 TPU 的高带宽内存优势加速批量评估。

---

## 本讲与整个 LLM 系统的关系

评估不是 LLM 开发的"附庸"，而是核心驱动力：

- **训练方向**：Mid-training checkpoint 的 benchmark 分数决定是否继续训练、是否加大数据量
- **对齐质量**：RLHF 后的人类偏好评估（勝率）直接衡量 SFT/RLHF 是否有效
- **模型选择**：在选择 base model 进行 fine-tuning 时，评估数据指导模型选择
- **Scaffolding 设计**：Agentic 框架的性能以 SWE-bench 等 agentic benchmark 的分数来衡量

**最大的教训**：不要迷信单一指标。LLaMA 1 论文在 MMLU 上不如 GPT-3.5，但在真实用户评测（Chatbot Arena）上胜率很高。Perplexity 低不意味着能做数学题，MMLU 高不意味着对话质量好。**评估体系需要和实际使用场景对齐。**

---

## 面试问题

1. **Perplexity 的数学含义是什么？为什么不同 tokenizer 的 Perplexity 不能直接比较？**

2. **MMLU 和 Chatbot Arena 分别测什么？为什么它们的排名可能不一致？**

3. **Data Contamination 是什么？如何检测和防止？** 讨论 n-gram overlap、embedding similarity、decontamination 技术。

4. **SWE-bench 如何评估编程能力？为什么 Agentic 框架能极大提升 SWE-bench 分数？**

5. **Goodhart's Law 在 LLM 评估中如何体现？** 举出 benchmark hacking 的具体例子。

6. **如何设计一个"好的" benchmark？** 讨论覆盖面、难度梯度、避免天花板效应、隔离训练数据等原则。

7. **Pass@k 指标的含义？为什么 HumanEval 使用 pass@1 和 pass@10 而不是简单的准确率？**

8. **评估中的 prompt sensitivity 问题？** 为什么同一个模型在不同 prompt template 下的 MMLU 分数可以差 5-10%？
