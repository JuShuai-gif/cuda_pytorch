# Lecture 17: 多模态模型

## 1. 本讲核心问题

LLM 在文本上取得了巨大成功，但世界不只是由文字构成的——图像、视频、音频、代码、3D 模型等都是重要的信息载体。**本讲的核心问题是：如何让 LLM 理解和生成非文本模态的内容？** 从 CLIP 的对比学习到 LLaVA 的视觉-语言连接，再到 Chameleon 的 omni-model 愿景，多模态模型正在成为 LLM 进化的下一个关键方向。更深层的问题是：非文本信息应该用连续表示（continuous tokens）还是离散表示（discrete tokens）？

---

## 2. 通俗解释

### CLIP ≈ 同时学看图说话和看话说图

想象你在教一个 AI 学生同时学两件事：

- **任务 A（看图说话）：** 给一张猫的照片，学生在 32768 个文字选项中选出"这是一只猫"而不是"这是一只狗"
- **任务 B（看话说图）：** 给一句话"这是一只猫"，学生在 32768 个图片选项中选出猫的照片

**CLIP 的关键 trick：** 让学生同时做 A 和 B，并且 A 和 B 必须互相验证！如果图片和文字真的匹配，那么 image embedding 和 text embedding 应该非常接近（cosine similarity 高）。通过这种"互相验证"的方式，CLIP 学习到了一种**共同表示空间**——在这个空间里，猫的图片和"猫"这个词距离很近。

**核心类比：**
- **单模态学习** ≈ 只看图或者只读文字（两个孤立的世界）
- **CLIP 的多模态学习** ≈ 建立图文之间的"翻译词典"，让视觉世界和语言世界通了

### Vision Transformer (ViT) ≈ 把图片当成一篇文章

普通 Transformer 处理文本：把一句话切成 tokens → 每个 token 是一个嵌入向量 → 用 Self-Attention 理解 token 之间的关系。

Vision Transformer (ViT) 处理图片：把图片切成 N 个小方块（patches，比如 16×16 像素）→ 每个 patch 拉平成一个向量 → 把这些向量当作"visual tokens" → 用 Self-Attention 理解 patch 之间的关系。

**类比：** 就像你读文字是一行一行地看，ViT 看图片是一块一块地看——每一块就像是一个"视觉单词"。

### LLaVA ≈ 给 LLM 装上眼睛

LLaVA 的设计思路非常简单：

```
图片 ──▶ Vision Encoder (CLIP) ──▶ 图片特征向量
                                        │
                                        ▼
                               [投影层：把视觉特征对齐到文字空间]
                                        │
                                        ▼
                               LLM (Vicuna/Llama) ──▶ 文字回答
```

**类比：** 给一个从未见过世界的聪明人（LLM）装上电子眼（Vision Encoder）。电子眼把看到的东西翻译成聪明人能理解的语言（投影层），然后聪明人就能回答"那是什么？""发生了什么？"这类问题了。

---

## 3. 数学公式 + 工程意义

### 3.1 CLIP: Contrastive Language-Image Pre-training

CLIP 使用 **contrastive loss**（对比损失），在一个 batch 内进行图文匹配：

**输入：** 一个 batch 包含 N 对 (image, text)
**编码：**
- $I_i = f_{image}(x_i^{img})$ — 图片编码器（ViT 或 ResNet）
- $T_i = f_{text}(x_i^{text})$ — 文字编码器（Transformer）

**Loss 函数：**
$$
\mathcal{L}_{CLIP} = -\frac{1}{2N} \left[ \sum_{i=1}^{N} \log \frac{\exp(I_i \cdot T_i / \tau)}{\sum_{j=1}^{N} \exp(I_i \cdot T_j / \tau)} + \sum_{i=1}^{N} \log \frac{\exp(T_i \cdot I_i / \tau)}{\sum_{j=1}^{N} \exp(T_i \cdot I_j / \tau)} \right]
$$

其中 $\tau$ 是可学习的温度参数（temperature），典型值初始化在 0.07 左右。

**工程直觉：** 损失函数的前半部分是"给定图片找文字"（image→text），后半部分是"给定文字找图片"（text→image）。两者同时优化，迫使两个编码器产生对齐的表示。

**训练规模：** OpenAI CLIP 的训练数据为 400M (image, text) 对，从互联网上自动抓取，不需要人工标注。

### 3.2 SigLIP: Sigmoid Loss for Language-Image Pre-training

SigLIP 把 CLIP 的多分类 softmax 损失替换为二分类 sigmoid 损失：

$$
\mathcal{L}_{SigLIP} = -\frac{1}{|B|} \sum_{i=1}^{|B|} \sum_{j=1}^{|B|} \left[ y_{ij} \cdot \log \sigma(I_i \cdot T_j / \tau + b) + (1 - y_{ij}) \cdot \log(1 - \sigma(I_i \cdot T_j / \tau + b)) \right]
$$

其中 $y_{ij} = 1$ 当 $i=j$（正例），否则 $y_{ij} = -1$（负例）。$b$ 是额外的 bias 参数。

**SigLIP 的优势：**

| 特性 | CLIP (Softmax) | SigLIP |
|------|---------------|--------|
| Loss 类型 | Global（依赖整个 batch） | Local（独立于 batch） |
| Batch size 敏感性 | 高（需要大 batch） | 低（小 batch 也可以用） |
| 计算复杂度 | $O(N^2)$ | $O(N^2)$，但更易并行 |
| 负样本利用 | 隐含在整个 batch 中 | 显式对所有负样本操作 |
| 可扩展性 | 受 batch size 限制 | 更灵活 |

### 3.3 Vision Transformer (ViT) 的数学原理

将一张图片 $X \in \mathbb{R}^{H \times W \times C}$ 切分成 $P \times P$ 的 patches：

$$
\text{Patches} = \left\{ X_{patch}^{(k)} \in \mathbb{R}^{P \times P \times C} \mid k = 1, \ldots, N \right\}
$$
其中 $N = \frac{HW}{P^2}$。

每个 patch 被 flatten 并通过线性投影映射到 d 维空间：

$$
\mathbf{z}_k = \mathbf{E} \cdot \text{flatten}(X_{patch}^{(k)}) + \mathbf{b}, \quad \mathbf{z}_k \in \mathbb{R}^d
$$

然后加上位置编码并输入 Transformer：

$$
\mathbf{z}_0 = [\mathbf{z}_{cls}; \mathbf{z}_{1}; \ldots; \mathbf{z}_{N}] + \mathbf{E}_{pos}
$$

**工程参数（ViT-L/14 为例）：**
- Patch size: $14 \times 14$
- 输入分辨率: $224 \times 224$ → $16 \times 16 = 256$ patches
- Embedding dimension: 1024
- 层数: 24
- 参数量: ~307M

### 3.4 LLaVA 的投影层设计

LLaVA 在 vision encoder 和 LLM 之间插入了简单的投影（projection）：

$$
\mathbf{h}_{text} = \mathbf{W}_{proj} \cdot \mathbf{h}_{vision} + \mathbf{b}_{proj}
$$

其中 $\mathbf{W}_{proj} \in \mathbb{R}^{d_{text} \times d_{vision}}$ 是一个可学习的线性投影矩阵。

**工程意义：** LLaVA 的投影层非常简单（线性层或 2 层 MLP），证明了**不需要复杂的跨模态融合，一个简单的投影就足以把视觉信息对齐到语言空间。**

---

## 4. 工业界真实实现

### 4.1 OpenAI CLIP (2021) — 多模态的基础设施

| 组件 | 配置 |
|------|------|
| Image Encoder | ViT-L/14 (307M params) 或 ResNet-50x64 |
| Text Encoder | Transformer (12 layers, 512 dim) |
| 训练数据 | 400M (image, text) 对 (WIT dataset) |
| 训练时间 | ~12 days (ViT-L)、~18 days (ResNet) |
| 训练 GPU | 256 V100 / 592 V100 |

**CLIP 的深远影响：** 几乎所有后续的多模态模型都使用 CLIP 作为 vision encoder（或使用 CLIP-style training），包括 DALL-E 2/3、Stable Diffusion、GPT-4V、LLaVA 等。

### 4.2 LLaVA 系列 — 给开源 LLM 装上眼睛

| 版本 | 时间 | Vision Encoder | LLM | 特点 |
|------|------|---------------|-----|------|
| LLaVA-1.0 | 2023.04 | CLIP ViT-L/14 | Vicuna-7B | 首个开源 multi-modal conversation |
| LLaVA-1.5 | 2023.10 | CLIP ViT-L/14 (336px) | Vicuna-7B/13B | 全连接投影层 + 更好的数据 |
| LLaVA-1.6 | 2024.01 | CLIP ViT-L/14 | Mistral-7B / Nous-Hermes-Yi-34B | 高分辨率 + 多图支持 |
| LLaVA-OneVision | 2024.08 | SigLIP | Qwen2 等 | 统一单图/多图/视频理解 |

**LLaVA-1.5 的训练流程（两阶段）：**

```
阶段 1: Pre-training for Feature Alignment
- 冻结 Vision Encoder + LLM
- 只训练投影层
- 数据：~558K 图片-描述对
- Loss：标准语言模型 loss（描述文字）
- GPU：8x A100, ~4 hours

阶段 2: Visual Instruction Tuning
- 解冻 LLM（或部分 LLM 层）
- 训练投影层 + LLM
- 数据：~665K 视觉指令数据（多轮对话、复杂推理）
- GPU：8x A100, ~10 hours
```

### 4.3 Qwen-VL 系列（阿里巴巴）

| 版本 | 时间 | 核心架构 | 规模 |
|------|------|---------|------|
| Qwen-VL | 2023.08 | ViT-G + Qwen-7B | 9.6B total |
| Qwen2-VL | 2024.08 | ViT (dynamic resolution) + Qwen2 | 2B/7B/72B |
| Qwen3-VL | 2025 | 改进 ViT + Qwen3 | TBD |

**Qwen2-VL 的创新：**
- **Dynamic Resolution:** 不固定图片输入大小，根据原始比例动态调整
- **Naive Dynamic Resolution (NDR):** 将图片重新排列为(任意, 28pix × 28pix)的子图网格
- **Multilingual:** 支持 29 种以上的语言

### 4.4 GPT-4V (OpenAI, 2023) — 封闭源的代表

虽然 OpenAI 没有公开 GPT-4V 的架构细节，但社区推测：
- Vision encoder 可能是在大量多模态数据上训练的定制 CLIP 变体
- 支持高分辨率图片（可能到 1024px 或更高）
- 文本和图片共享同一个 transformer decoder（early fusion）
- 训练过程可能使用了 interleaved text-image 数据

### 4.5 Gemini (Google, 2023/2024) — 原生多模态

Google 的 Gemini 系列采取了更激进的设计——**原生多模态**：

- 从训练一开始就使用 interleaved text-image-audio-video 数据
- 使用 multi-modal tokenizer 统一处理不同模态
- Gemini 1.5 Pro 支持 1M token 的上下文窗口，可以处理整个长视频

### 4.6 Chameleon (Meta, 2024) — 走向 Omni-Model

Chameleon 的目标是实现**完全的 token-based 统一多模态模型**：

| 模态 | Token 化方式 | 特点 |
|------|-------------|------|
| 文本 | BPE tokenizer | 标准文本 tokenization |
| 图片 | VQ-VAE + VQGAN | 离散化图像 token（~1024 tokens/图） |
| 训练 | 混合 next-token prediction | 统一 loss，不做特殊模态处理 |

**核心设计哲学：** "所有模态都是 tokens。" 不像 LLaVA 那样用连续表示，Chameleon 把图片也变成离散 tokens，让 transformer 用完全相同的机制处理文本和图片。

---

## 5. CUDA/GPU 视角

### 5.1 CLIP 训练的计算需求

```
CLIP 训练 = 图片编码器（ViT）+ 文字编码器（Transformer）
         + 对比损失（矩阵乘法）

计算瓶颈：
- ViT forward: ~1.5 × N_patches × d^2 × layers FLOPs
- 对比损失（N×N 矩阵）：~N^2 × d FLOPs

对于 batch_size = 32768, d = 1024:
- 对比损失矩阵：32768^2 × 1024 ≈ 1.1 × 10^12 ≈ 1.1T FLOPs
```

**关键 GPU 优化：**
- 使用 gradient checkpointing 减少显存（ViT 部分特别重要）
- 对比损失的矩阵乘法可以用 NVIDIA 的 cuBLAS 高效实现
- 大 batch size（32K+）需要多 GPU 数据并行 + 梯度累积

### 5.2 LLaVA 训练的 GPU 需求

| 阶段 | 需要的 GPU 显存（7B LLM） | 典型配置 |
|------|--------------------------|---------|
| Visual Encoder 推理 | ~4 GB (ViT-L/14) | 可共享显存 |
| LLM 训练 (LoRA) | ~24 GB | 1-2x A100 |
| LLM 训练 (Full) | ~80 GB | 4-8x A100 |
| 投影层训练 | ~30 GB | 2x A100 |

### 5.3 多模态推理的延迟分析

多模态推理比纯文本推理慢的主要原因：

```
纯文本推理延迟 = T_decoding × (t_A + t_FFN + t_KV)
多模态推理延迟 = T_decoding × (t_A + t_FFN + t_KV) + T_vision_encode

其中 T_vision_encode 是一次性的（只需编码一次图片）
- ViT-L/14: ~10-50ms (A100, 224×224)
- 高分辨率（如 1344×1344）: ~100-500ms
```

**优化策略：**
- 视觉编码可以 cache（同一张图片多次查询时）
- 使用更小的 vision encoder（如 SigLIP 的 SoViT-400m）
- Token 压缩：减少输入到 LLM 的视觉 token 数量

```python
# LLaVA-style inference pseudocode
def llava_inference(image, question, llm, vision_encoder, projector):
    # Step 1: Encode image (one-time cost, can be cached)
    vision_features = vision_encoder(image)  # [N_patches, d_vision]
    
    # Step 2: Project to text space
    vision_tokens = projector(vision_features)  # [N_patches, d_text]
    
    # Step 3: Construct prompt with vision tokens as prefix
    prompt = format_prompt(vision_tokens, question)
    
    # Step 4: Standard LLM decoding
    response = llm.generate(prompt)
    return response
```

---

## 6. 本讲与整个 LLM 系统的关系

```
┌────────────────────────────────────────────────────────────┐
│                LLM 系统的多模态扩展                          │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  纯文本 LLM ──────────────────────────────────────────▶     │
│  (GPT-3, Llama, DeepSeek)                                   │
│                                                             │
│  + Vision Encoder ────────────────────────────────────▶     │
│  (LLaVA, GPT-4V, Qwen-VL)  →  "看图说话" 能力               │
│                                                             │
│  + Vision Generation ─────────────────────────────────▶     │
│  (DALL-E, Stable Diffusion) →  "按描述生成图" 能力           │
│                                                             │
│  + Audio Encoder/Decoder ─────────────────────────────▶     │
│  (Whisper, AudioPaLM) →  "听说" 能力                        │
│                                                             │
│  + Omni-Model ────────────────────────────────────────▶     │
│  (Chameleon, Gemini) →  任何模态到任何模态                  │
└────────────────────────────────────────────────────────────┘
```

### 连续 Token vs 离散 Token 的争论

这是多模态模型设计中最核心的技术选择：

| 维度 | 连续 Token (LLaVA) | 离散 Token (Chameleon) |
|------|-------------------|----------------------|
| **表示方式** | 向量（直接来自 encoder） | 离散 ID（来自 VQ-VAE） |
| **信息损失** | 低（连续表示） | 有（量化损失） |
| **Token 数** | 多（256+ tokens/图） | 相对少（可用压缩） |
| **与 LLM 兼容性** | 需要投影层 | 天然兼容（都是 token） |
| **生成能力** | 只能理解，不能生成 | 可以理解 + 生成 |
| **灵活性** | 高（可以调整分辨率等） | 受 tokenizer 限制 |
| **代表工作** | LLaVA, GPT-4V, Qwen-VL | Chameleon, DALL-E, Emu |

**趋势：** 连续 token 在纯理解任务上更有优势（更少信息损失），离散 token 在生成任务上更有优势（天然可生成）。未来的 omni-model 可能需要结合两者。

### 多模态 Scaling

多模态模型的 scaling law 比纯文本更复杂：

1. **数据配比：** 图片和文字的最佳比例是什么？目前没有定论
2. **分辨率 scaling：** 更高的分辨率带来更好的细节理解，但计算量平方增长
3. **多任务 trade-off：** 强化视觉能力可能削弱纯文本能力

---

## 7. 面试问题

1. **CLIP 的核心训练方式是什么？它的对比损失函数是如何定义的？为什么 CLIP 能实现 zero-shot 分类？**

   *参考答案：CLIP 使用对比学习同时训练 image encoder 和 text encoder，使匹配的图文对在嵌入空间中距离近，不匹配的距离远。损失函数使用 batch 内的交叉熵（见公式部分）。CLIP 实现 zero-shot 分类的方法是：把分类任务变成图文匹配——把 N 个类别名变成 N 句描述文字（如 "a photo of a cat"），然后找与图片最匹配的文字，不需要任何该任务的训练数据。*

2. **LLaVA 如何把视觉信息和 LLM 结合起来？为什么只需要一个简单的投影层？**

   *参考答案：LLaVA 使用 CLIP 的 vision encoder 提取图片特征，然后通过一个线性投影层（或 2 层 MLP）把视觉特征映射到 LLM 的 token embedding 空间，作为 LLM 的输入前缀。只需要简单投影层的原因是：CLIP 已经通过对比学习学到了与语言空间部分对齐的视觉表示，只需要一个轻量的转换就能让 LLM 理解。如果 CLIP 特征和 LLM 的表示空间完全不兼容，需要更复杂的对齐网络。*

3. **SigLIP 和 CLIP 的区别是什么？为什么 SigLIP 对 batch size 不那么敏感？**

   *参考答案：CLIP 使用 softmax 对比损失，损失函数依赖于整个 batch 内的负样本分布——小 batch 意味着更少的负样本，对比学习信号更弱。SigLIP 使用 sigmoid 二分类损失，每个 (image, text) 对独立判断是否匹配——batch size 只影响一个 step 中计算的对数，但不改变每个对的损失形式。这使得 SigLIP 可以在小 batch 下也能有效训练，更适合资源受限的场景。*

4. **什么是 Vision Transformer (ViT)？和 CNN 相比有什么优势和劣势？**

   *参考答案：ViT 将图片切成 patches，每个 patch 是 transformer 的一个 token，用自注意力机制建模 patch 之间的关系。优势：（1）全局感受野（每层都看全图），CNN 需要多层才能扩大感受野；（2）更简单的架构（不需要设计卷积核）；（3）容易 scale（扩大参数量效果稳定提升）；（4）与文本 transformer 架构统一。劣势：（1）需要更多数据和/或更强的数据增强才能超越 CNN；（2）在小数据集上不如 CNN；（3）对高分辨率图片效率较低（复杂度 $O(N^2)$，N 是 patch 数）。*

5. **多模态模型中，连续 token 和离散 token 各有什么优劣？什么任务更适合哪种？**

   *参考答案：见上表。连续 token 适合纯理解任务（VQA、captioning），因为信息损失小；离散 token 适合需要生成输出的任务（图像生成），因为文本 LLM 已经擅长从离散 token 采样。很多商业应用（如 GPT-4V）选择连续 token，因为理解 + 文字回答是主要场景，不需要生成图片。但 omni-model 的长期趋势可能倾向于离散 token，因为它提供了模态统一的可能。*

6. **如果你要训练一个多模态 LLM，但有有限的 GPU 预算，你会怎么设计训练策略？**

   *参考答案：（1）复用已训练的 vision encoder（如 CLIP ViT-L/14）和 LLM（如 Llama 或 Qwen），不用从头训练；（2）使用 LLaVA 的两阶段策略：先只训练投影层（小成本），再微调 LLM 或用 LoRA；（3）视觉 encoder 保持冻结，只更新投影层和 LLM；（4）优先使用公开的多模态指令数据集（如 LLaVA-Instruct、ShareGPT-4V）；（5）如果做中文，使用 Qwen-VL 或 InternVL 作为基础模型。整个流程在 4-8 块 A100 上可以在 1-2 周内完成。*

---

## 参考：多模态模型发展年表

| 时间 | 模型/工作 | 模态 | 核心贡献 |
|------|----------|------|---------|
| 2021.01 | CLIP (OpenAI) | Image+Text | 对比学习实现多模态对齐 |
| 2021.10 | ViT (Google) | Image | Transformer 处理图片的范式转变 |
| 2022.03 | Flamingo (DeepMind) | Image+Text+Video | 早期视觉-语言对话 |
| 2023.03 | GPT-4 (OpenAI) | Image+Text | 首个公开 demo 的多模态 GPT |
| 2023.04 | LLaVA-1.0 | Image+Text | 开源多模态对话模型 |
| 2023.05 | BLIP-2 (Salesforce) | Image+Text | Q-Former 轻量连接 |
| 2023.09 | Qwen-VL (Alibaba) | Image+Text | 强大的中文多模态 |
| 2023.12 | Gemini (Google) | Text+Image+Audio+Video | 原生多模态 |
| 2024.06 | Chameleon (Meta) | Text+Image | 统一 token 的 omni-model |
| 2024.08 | Qwen2-VL (Alibaba) | Image+Text+Video | Dynamic Resolution |
| 2024.08 | LLaVA-OneVision | Image+Multi-image+Video | 统一视觉理解 |
| 2025 | Qwen3-VL | Image+Text+Video | 新一代多模态 |

> **关键 Takeaways:**
> 1. CLIP 是多模态 AI 的基础设施——几乎所有后续工作都依赖它
> 2. LLaVA 证明了连接 vision encoder 和 LLM 可以非常简单（只需线性投影）
> 3. 多模态模型的趋势是从"单模态理解"到"全模态理解 + 生成"
> 4. 连续 token 和离散 token 各有优势，没有绝对的最优方案
> 5. 视觉 token 的"压缩"是关键——如何用最少的 token 传达最多的视觉信息
> 6. 原生多模态（Gemini/Chameleon）是未来方向，但目前大部分实践仍使用连接式架构（LLaVA-style）
