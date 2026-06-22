# Lecture 13: 数据 Data I - 数据源

## 1. 本讲核心问题

**数据是训练大语言模型（LLM）最重要的因素，没有之一。** 本讲的核心问题是：训练一个 LLM 需要什么数据？从哪里来？不同模型如何选择数据策略？数据涉及哪些法律和版权问题？为什么各家大公司对自己的数据配方守口如瓶？

---

## 2. 通俗解释

### 数据 ≈ 教材质量决定学生水平

想象你在培养一个超级学霸：

- **Common Crawl** ≈ 互联网上所有乱七八糟的帖子、广告、论坛、微博——量大但质量参差不齐
- **Wikipedia** ≈ 百科全书——质量高、结构好，但覆盖面有限
- **GitHub** ≈ 编程习题册——专门训练编程能力
- **ArXiv** ≈ 学术期刊——训练学术写作和推理能力
- **Books** ≈ 经典名著——训练文学素养和长文本理解
- **StackExchange** ≈ 知乎高质量问答——训练问答和解释能力

一个学生（模型）如果只学 Wikipedia（像 BERT），会变得知识面窄但精确；如果什么都学（像 GPT-3），会知识面广但可能学到很多垃圾信息。关键在于**怎么搭配**和**怎么筛选**。

**核心类比：** 差教材 + 最好的老师 = 教不出好学生。数据质量直接决定模型能力的上限，再好的训练算法也救不了垃圾数据。

---

## 3. 数学公式 + 工程意义

### 3.1 数据 token 量计算

训练 LLM 的第一个工程问题是：**我们有多少数据？**

```
Total Tokens ≈ 数据集大小(GB) × Token per Byte 比率
               ≈ 数据集大小(GB) × 0.25 ~ 0.3 (for English text)
```

例如：Common Crawl 原始数据约 200-400 TB（原始 HTML），经过清洗后约 10-20 TB 纯文本，对应 **约 3-6 万亿 tokens**。

### 3.2 数据多样性与 loss 的关系

从信息论角度，训练数据的 diversity 直接影响模型的泛化能力：

$$
\mathcal{L}(D_{train}) \approx \mathcal{L}(D_{test}) + \underbrace{\frac{d}{N}}_{\text{模型容量}} + \underbrace{\text{KL}(P_{test} \| P_{train})}_{\text{数据分布差异}}
$$

**工程意义：** 如果训练数据和实际使用场景数据分布差异太大（KL divergence 高），模型 loss 会很高。这就是为什么要让训练数据覆盖尽可能多的领域和场景。

### 3.3 Scaling Laws 中的数据角色

根据 Chinchilla scaling law：

$$
N_{opt} \propto C^{0.5}, \quad D_{opt} \propto C^{0.5}
$$

其中 $C$ 是总计算量（FLOPs），$N$ 是模型参数量，$D$ 是训练 token 数。**工程意义：** 如果想要 compute-optimal 训练，数据和模型参数需要同步增长。更大的模型需要更多高质量数据，这是所有大模型公司面临的"数据墙"困境。

---

## 4. 工业界真实实现

### 4.1 BERT (2018) — "精选教材"策略

| 数据源 | 大小 |
|-------|------|
| BooksCorpus | 800M words |
| English Wikipedia | 2,500M words |
| **总计** | **~3.3B tokens** |

BERT 只用了两个高质量数据源，总共约 33 亿 tokens。这在当时是合适的——BERT 是编码器模型，主要用于理解任务，不需要生成能力。

### 4.2 GPT-3 (2020) — "什么都学"策略

| 数据源 | 权重 | Token 数 |
|-------|------|---------|
| Common Crawl (filtered) | 60% | ~300B |
| WebText2 | 22% | ~110B |
| Books1 + Books2 | 16% | ~80B |
| Wikipedia | 3% | ~15B |
| **总计** | | **~500B tokens** |

GPT-3 的数据混合策略体现了"质量加权"思想：虽然 Common Crawl 质量低，但量大，需要给更多权重；高质量数据如 Wikipedia 给较少权重也可以起到"锚定"质量的作用。

**关键工程决策：** GPT-3 在训练时**不在 epoch 边界停止**，而是按 token 数均匀采样——这意味着高质量小数据集（如 Wikipedia）会被重复训练更多次。

### 4.3 Llama 系列 — "多样化高质量"策略

| 数据源 | Llama 1 | Llama 2 | Llama 3 |
|-------|---------|---------|---------|
| CommonCrawl | ✓ | ✓ | ✓ |
| C4 | ✓ | ✓ | ✓ |
| GitHub | ✓ | ✓ | ✓ |
| Wikipedia | ✓ | ✓ | ✓ |
| Books | ✓ | ✓ | ✓ |
| ArXiv | ✓ | ✓ | ✓ |
| StackExchange | ✓ | ✓ | ✓ |
| **总 tokens** | **~1.4T** | **~2T** | **~15T** |

Llama 3 使用 15T tokens 训练 405B 参数模型，约 $D/N \approx 37$，远高于 Chinchilla 的 20 倍建议比例——这是一种**刻意 overtrain** 的策略，目的是在推理时用小模型获得大模型的能力，降低推理成本。

### 4.4 DeepSeek V3/V4 — "数据为核心竞争力"

DeepSeek 使用约 **32T tokens** 训练，这在 2024 年是非常大的数据规模。DeepSeek 对其数据处理 pipeline 高度保密——这本身就是一种竞争壁垒。

### 4.5 为什么公司对数据保密？

1. **竞争壁垒：** 数据处理方法是核心 IP，决定模型质量
2. **版权风险：** 详细披露数据来源可能引发法律诉讼
3. **数据污染：** 竞争对手可能利用数据信息进行模型逆向或攻击

---

## 5. CUDA/GPU 视角

### 5.1 数据加载是 GPU 饥饿的根源

在训练大规模 LLM 时，GPU 的利用率往往受限于数据加载速度：

```
GPU Utilization = min(1, Data Throughput / GPU Processing Speed)
```

**典型问题：**
- 从 HDD 读取 15T 文本数据 → 顺序读取速度 ~200MB/s → 需要 ~20 小时
- 从 NVMe SSD 读取 → ~3GB/s → 需要 ~1.4 小时
- 从网络存储（如 AWS S3）读取 → 带宽受限，需要数据预取和缓存

**工业实践：**
- NVIDIA Megatron-LM 使用数据分片，每个 GPU 读取不同数据分片
- 数据预处理（tokenization）通常在 CPU 上离线完成，避免浪费 GPU 算力
- 使用 `torch.utils.data.DataLoader` 的 `num_workers` 和 `prefetch_factor` 参数优化数据 pipeline

```python
# Example: Megatron-LM style data loading
dataloader = DataLoader(
    dataset,
    batch_size=micro_batch_size,
    num_workers=8,          # 8 CPU workers
    prefetch_factor=4,      # prefetch 4 batches per worker
    pin_memory=True,        # pin to GPU memory for faster transfer
)
```

### 5.2 数据预处理的计算成本

15T tokens 的数据预处理（HTML 清洗、去重、质量过滤）通常需要 **数百到数千 CPU 小时**，这是一笔不可忽视的计算开销。DCLM 和 Dolma 等项目公开了他们的预处理 pipeline，帮助社区降低这一成本。

---

## 6. 本讲与整个 LLM 系统的关系

```
┌─────────────────────────────────────────────────────────┐
│                      LLM 训练全流程                       │
├───────────┬───────────┬───────────┬─────────────────────┤
│  数据收集  │  数据处理  │  预训练    │  对齐 & 部署         │
│ (Lecture13)│(Lecture14)│(Lecture12)│  (Lecture15)        │
├───────────┴───────────┴───────────┴─────────────────────┤
│  数据质量 ────────决定────────▶ 模型能力上限               │
│  数据规模 ────────决定────────▶ 是否需要更多参数           │
│  数据多样性 ──────决定────────▶ 泛化能力                  │
└─────────────────────────────────────────────────────────┘
```

**核心洞见：** 数据是 LLM 训练的"第一性原理"。模型架构、训练算法、硬件优化——这些都可以被视为"在给定数据下最大化利用率的工程手段"。但数据本身的质量和规模从根本上决定了模型能力的天花板。

**Pre-training → Mid-training → Post-training 的数据变化：**

| 阶段 | 数据特征 | 示例 |
|------|---------|------|
| Pre-training | 大规模、广覆盖、原始文本 | 15T tokens 通用文本 |
| Mid-training | 长文本、特定领域、高质量 | 长上下文数据、代码 |
| Post-training | 指令格式、对话格式、人类偏好 | SFT 数据、RLHF 数据 |

---

## 7. 面试问题

1. **为什么说数据是训练 LLM 最重要的因素？如果给你无限的 GPU 算力但只有 Wikipedia 数据，你能训练出 GPT-4 级别的模型吗？为什么？**
   
   *参考答案：不能。数据决定了模型的知识广度和语言能力上限。Wikipedia 覆盖领域有限（缺少代码、对话、创意写作等），且总量不足。模型会遇到"数据墙"——再多算力也无法创造不存在于训练数据中的知识。*

2. **GPT-3 的数据混合中 Common Crawl 占 60%，但为什么还要加入 Wikipedia（3%）？为什么不直接用 Wikipedia 就够了？**

   *参考答案：Wikipedia 质量高但覆盖面窄。Common Crawl 提供广覆盖（包括各种领域、风格、语言现象），但需要 Wikipedia 这样的高质量数据作为"锚点"来引导训练方向。少量高质量数据 + 大量广覆盖数据是目前的最佳实践。*

3. **Llama 3 使用 D/N ≈ 37（远大于 Chinchilla 建议的 20），为什么要"overtrain"？**

   *参考答案：推理成本考虑。overtrained 的小模型在推理时比 undertrained 的大模型更经济。用更多数据训练的相对较小模型，在推理时可以接近大模型的表现但推理成本更低——这对于需要部署的公司尤为重要。*

4. **如果你发现训练数据中包含了测试基准（如 MMLU）的题目，你应该怎么做？为什么这是严重问题？**

   *参考答案：必须重新清洗数据。这叫"数据污染"（data contamination），会导致 benchmark 分数虚高，无法反映模型真实能力。这是学术不端，会导致研究结果无效，损害声誉。*

5. **为什么大型 AI 公司都不公开他们的训练数据细节？列举至少三个原因。**

   *参考答案：（1）竞争壁垒——数据处理方法是核心 IP；（2）法律风险——详细披露数据来源可能引发版权诉讼；（3）安全考虑——攻击者可以利用数据信息构建对抗样本或进行数据投毒；（4）商业合作——有些数据是购买的专有数据，受合同限制。*

---

## 参考数据集详情

### The Pile (2020, EleutherAI)
- **规模：** 825 GB 文本
- **组成：** 22 个子集，包括 PubMed, ArXiv, GitHub, StackExchange, Books3 等
- **特点：** 第一个高质量、公开可复现的 LLM 训练数据集

### C4 (Colossal Clean Crawled Corpus, 2020)
- **规模：** 750 GB (~175B tokens)
- **来源：** Common Crawl 经过严格清洗
- **特点：** T5 模型发布的数据集，成为后续很多模型的基准

### Dolma (2023, AI2)
- **规模：** 3T tokens
- **组成：** Common Crawl + C4 + Reddit + StackExchange + Books + 学术论文
- **特点：** 完全开放，包括处理工具链和文档；OLMo 模型的训练数据

### DCLM (2024, 多家合作)
- **规模：** ~2T tokens (DCLM-Baseline)
- **特点：** 强调可复现性；证明了好的数据过滤可以用更少的数据达到相同效果

### FineWeb (2024, Hugging Face)
- **规模：** 15T tokens
- **特点：** 从 Common Crawl 中提取的高质量 web 数据；使用 fastText 和 KenLM 进行质量过滤

---

> **关键 Takeaways:**
> 1. 数据的质和量共同决定模型能力上限
> 2. 不同模型的数据策略差异巨大，但都遵循"广覆盖 + 高质量锚点"原则
> 3. 数据是真正的竞争壁垒——比模型架构更难以复制
> 4. Overtraining 趋势下，对数据规模的需求只会越来越大
