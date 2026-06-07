# Paper 15: StreamingLLM — Efficient Streaming Language Models with Attention Sinks (Xiao et al., ICLR 2024)

> 论文全称：**Efficient Streaming Language Models with Attention Sinks**
> 发表会议：ICLR 2024
> 作者：Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis（MIT / Meta FAIR / CMU）

---

## 1. 论文解决什么问题

LLM 在流式场景（如多轮对话、实时翻译、代码补全）下需要处理无限长的输入序列。然而，标准 LLM 的 KV Cache 随序列长度线性增长，内存很快耗尽。已有的滑动窗口方案（如只保留最近 $L$ 个 token）会导致**严重的性能崩溃**——当初始 token 被驱逐后，困惑度急剧上升。本文提出 StreamingLLM，不需要任何微调就能让 LLM 稳定处理**无限长度**的流式输入，速度提升最高 **22.2 倍**。

---

## 2. 核心方法

### 注意力沉没现象（Attention Sink）

作者发现了一个此前未被注意的现象：
- 在自回归推理中，**最初的几个 token（通常第 0-3 个）会吸收大量注意力权重**，无论它们的内容是什么
- 即使这些初始 token 在语义上不重要，所有后续 token 的注意力仍会集中到它们身上——它们起着类似"注意力回收站（sink）"的作用
- 当滑动窗口丢弃这些 token 时，注意力分布会剧烈偏移，导致模型输出崩溃

### StreamingLLM 的核心策略

保持两种 token 在 KV Cache 中：
1. **Attention Sink tokens（4 个初始 token）**：始终保留
2. **最近窗口内的 token**：保留最近 $L$ 个 token 的 KV
3. 其余中间 token 的 KV 被逐出

### 为什么 Attention Sink Token 重要？

标准 softmax 注意力要求注意力权重对每个 query 求和为 1：

$$\sum_j \text{softmax}(QK^T)_{i,j} = 1$$

当所有"有意义"的 KV 都被丢弃后，模型被迫将注意力分配到不相关的 token 上，破坏了数值稳定性。保留 sink token 提供了"稳定的注意力锚点"，使 softmax 始终有合理的去处。

### 实现细节
- 只需在推理时修改 KV Cache 管理逻辑，**不需任何训练或微调**
- 可与任何基于 Transformer 的 LLM（LLaMA、Mistral、Falcon 等）兼容
- 推荐使用 4 个 attention sink token + 最近的 2000 个 token

---

## 3. 关键公式

### 自回归生成中的 KV Cache 管理

设序列长度为 $N$，KV Cache 容量为 $C$，保留最近的 $L$ 个 token 和 $S$ 个 sink token，其中 $S + L = C$：

$$\text{KV Cache}_t = \{\mathbf{k}_0, \dots, \mathbf{k}_{S-1}\} \cup \{\mathbf{k}_{t-L+1}, \dots, \mathbf{k}_t\} \quad \text{当 } t > C$$

### 注意力计算（仅对保留的 token）

$$\text{Attention}(\mathbf{q}_t, \text{KV Cache}_t) = \text{softmax}\left(\frac{\mathbf{q}_t \mathbf{K}_{\text{cache}}^T}{\sqrt{d_k}}\right) \mathbf{V}_{\text{cache}}$$

其中 $\mathbf{K}_{\text{cache}}, \mathbf{V}_{\text{cache}}$ 仅包含 sink token 和最近 window token。

### 滑动窗口崩溃时的困惑度

设 $\mathcal{P}(x)$ 为模型的概率输出，当 KV Cache 仅保留最近 $L$ 个 token 时：

$$\lim_{t \to \infty} PPL(t) \to \infty \quad \text{（发散）}$$

而使用 StreamingLLM 后：

$$\lim_{t \to \infty} PPL(t) \to PPL_{\text{stable}} \quad \text{（保持稳定）}$$

### 加速比

$$Speedup = \frac{\text{Full attention time}}{\text{StreamingLLM time}} \approx \frac{O(N^2)}{O(C \cdot N)}$$

当输入长度为 4M token、窗口为 4 个 sink + 2000 个最近 token 时，加速比可达 **22.2×**。

---

## 4. 实验结论

| 模型 | 方法 | 序列长度 4M PPL | 解码速度 (tokens/s) |
|------|------|------------------|----------------------|
| LLaMA-2-7B | Full Attention | OOM | — |
| LLaMA-2-7B | Sliding Window (2k) | 发散 (>10³) | ~50 |
| LLaMA-2-7B | StreamingLLM (4+2k) | **7.88** | ~1100 |
| LLaMA-2-13B | StreamingLLM (4+2k) | **6.12** | ~620 |
| Falcon-7B | StreamingLLM (4+2k) | **8.05** | ~1080 |

- 滑动窗口法在 token 数超过窗口大小后，PPL 急剧上升（从 ~8 飙升至 >1000），模型输出变成乱码
- StreamingLLM 在 **4M token** 长度下 PPL 保持稳定，与全注意力（在 OOM 前）的短序列 PPL 持平
- 4 个 sink token 是 sweet spot：少于 4 个时 PPL 略有上升，多于 4 个时没有额外收益
- StreamingLLM 还可与 **位置编码外推**（如 PI、YaRN）结合使用，进一步改善长序列性能
- 在 LongChat（多轮对话）、Passage Retrieval（文档检索）等真实流式任务上表现与全注意力一致

---

## 5. 工业价值

- **零成本部署**：无需微调，修改几行推理代码即可支持无限长流式输入
- **实时应用**：已被集成到多个 LLM 推理框架（如 llama.cpp、vLLM、text-generation-inference）中
- **降低硬件门槛**：让 7B 模型在消费级 GPU（如 RTX 4090）上也能处理百万级 token 的输入流
- **AI Agent 场景**：长期运行的 Agent（如 AutoGPT、ChatDev）需要持续的多轮对话能力，StreamingLLM 是关键基础设施

---

## 6. 与课程 Lecture 的关系

- **Lecture 15（Long-Context LLM）**：StreamingLLM 是长上下文推理的核心技术之一，解决了如何在有限显存下高效处理无限长序列的实际问题
- **Lecture 1（Efficiency Metrics）**：本文关注的是**内存效率**（KV Cache 大小）和**推理延迟**（prefill + decode 阶段的 FLOPs），是效率指标在生成式模型中的重要体现
- **Lecture 13（LLM Deployment）**：KV Cache 管理是 LLM 部署的关键瓶颈，StreamingLLM 提供了系统层的最优策略

---

## 7. 我应该如何复现

1. **环境准备**：PyTorch 2.0+，HuggingFace Transformers
2. **加载模型**：使用 `meta-llama/Llama-2-7b-hf` 或 `mistralai/Mistral-7B-v0.1`
3. **修改 KV Cache 逻辑**：
   - 在每层自注意力模块中，找到 `past_key_values` 的管理代码
   - 实现 `_trim_kv_cache(past_key_values, num_sink=4, window_size=2000)`
   - 每次追加新 token 的 KV 后，检查总长度是否超过 `num_sink + window_size`
   - 若超过，保留前 4 个 + 后 2000 个，删除中间所有 token 的 KV
4. **测试指标**：
   - 在 PG19 或 arXiv 长文档上测试 PPL（使用前 4000 token 做 prefilling，继续推理至 100k token）
   - 度量解码速度（tokens/s）和 GPU 峰值显存
5. **对比实验**：
   - Baseline 1: 全注意力（能跑多长跑多长）
   - Baseline 2: 纯滑动窗口（window=2000，无 sink）
   - StreamingLLM: 4 sink + 2000 window
6. **关键注意事项**：
   - HuggingFace 的 `past_key_values` 已支持动态长度，需手动实现 trim
   - Sink token 数量设为 4 是经验最优值，不同模型可能需要微调
   - 结合 FlashAttention 使用时要注意 trim 操作的兼容性
