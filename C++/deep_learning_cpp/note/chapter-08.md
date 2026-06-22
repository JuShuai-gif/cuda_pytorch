# 第 8 章：生成网络、自编码器与大语言模型

基于 *Deep Learning with C++*（Packt，ISBN 9781835880036）第 8 章，第 266–308 页。本章覆盖三大生成式 AI 范式——自编码器（压缩重建）、VAE（概率隐空间）、GAN（对抗博弈）——以及 LLM 的自回归生成、采样策略（温度/TopK/TopP/束搜索）和评估指标（BLEU/ROUGE/困惑度/BERTScore）。

---

## 目录

1. [章节概述](#章节概述)
2. [三大生成范式](#三大生成范式)
3. [文件索引](#文件索引)
4. [编译与运行](#编译与运行)
5. [技术速查](#技术速查)
6. [PDF 完整内容对照](#pdf-完整内容对照)

---

## 章节概述

生成模型学习训练数据的**底层分布**，而非简单的分类边界。判别模型（P(y|x)，前面的 CNN/RNN）区分类别，生成模型（P(x) 或 P(x,y)）可以合成全新的数据点。

本章从三个经典生成范式开始，然后过渡到现代 LLM 的生成机制：

- **自编码器**：编码器压缩→潜在空间→解码器重建，MSE 损失，去噪变体
- **VAE**：编码为概率分布（μ, σ²），重参数化技巧保证梯度流通，ELBO = BCE + KL
- **GAN**：生成器 vs 鉴别器 minimax 博弈，五步训练，模式坍塌挑战
- **LLM 生成**：自回归因果语言建模，温度/贪婪/TopK/TopP/束搜索，终止条件
- **评估**：困惑度、BLEU、ROUGE、METEOR、BERTScore

### 判别模型 vs 生成模型

| 特性       | 判别模型                      | 生成模型                       |
| ---------- | ----------------------------- | ------------------------------ |
| 建模目标   | P(y\|x) —— 分类边界           | P(x) 或 P(x,y) —— 数据分布     |
| 任务       | 分类、分割、标签预测          | 合成新样本、去噪、数据增强     |
| 输出       | 类别标签                      | 新数据（图像/文本/音频）       |
| 例子       | CNN 分类、RNN 情感分析         | VAE、GAN、LLM（GPT）        |
| 指标       | Accuracy、F1                  | BLEU、ROUGE、Perplexity        |

---

## 三大生成范式

### 自编码器：压缩-重建

```
输入(28×28) → Encoder → Bottleneck(compact) → Decoder → 输出(28×28)
              Conv+Pool     latent space       ConvT+Upsample
损失: MSE(输出, 输入)
```

- 编码器：3 个 Conv2d + MaxPool2d → 28×28→3×3×128
- 瓶颈层：Conv2d(128→64)
- 解码器：3 个 ConvTranspose2d + Upsample → 3×3→28×28
- 无监督学习：不需要标签，输入本身就是目标

### VAE：概率隐空间

```cpp
// 核心创新：重参数化技巧
torch::Tensor reparameterize(mu, logvar) {
    auto std = torch::exp(0.5 * logvar);   // σ = e^(0.5*log(σ²))
    auto eps = torch::randn_like(std);     // ε ~ N(0,1) ← 固定分布
    return mu + eps * std;                 // z = μ + σ⊙ε ← 可微分！
}

// 损失 = BCE重建 + β·KL散度
loss = BCE(recon, x) + β * (-0.5 * Σ(1 + logvar - μ² - e^logvar))
//     重建质量         拉向标准正态分布(regularization)
```

**为什么需要重参数化技巧？** 直接从 N(μ,σ²) 采样会截断梯度流（随机操作不可微）。将随机性 σ 剥离为独立的 ε~N(0,1)，使 μ 和 σ 的梯度可直接计算。

### GAN：对抗博弈

```
min_G max_D  V(D,G) = E[log D(x)] + E[log(1 - D(G(z)))]
```

五步训练法：
1. 加载真实数据
2. G 从 z~N(0,1) 生成假样本
3. 配对真(label=1)+假(label=0)
4. **训练 D**（冻结 G）：最大化 D(real)→1, D(fake)→0
5. **训练 G**（冻结 D）：最大化 D(G(z))→1（欺骗 D）

关键实现细节：
- `fake_images.detach()` —— 训练 D 时阻止梯度流向 G
- 训练 G 时用 `real_labels` 作为目标——G 想骗过 D
- Adam 参数特殊配置：lr=0.0002, β₁=0.5（低于默认的 0.9）——稳定训练

### 采样策略

| 策略       | 机制                                | 多线程          | 适用场景           |
| ---------- | ----------------------------------- | --------------- | ------------------ |
| 贪婪解码   | 每步选 argmax                       | 确定→单一输出   | 数据提取、模板     |
| 束搜索     | 保持 k 条候选路径                   | 确定→全局优化   | 翻译、摘要         |
| Top-K      | 只从概率最高的 k 个中采样            | 随机→受控       | 对话、内容生成     |
| Top-P      | 累积概率 ≥ p 的最小 token 集合选择  | 随机→自适应     | 对话、创意写作     |

### 温度缩放

```
logits_scaled = logits / T
```

- `T < 1`：分布更尖锐→更确定→适合问答/代码
- `T = 1`：原始概率分布
- `T > 1`：分布更平坦→更随机→适合创意写作

---

## 文件索引

### 一、自编码器与 VAE — PDF 第 267–279 页

| 文件                      | PDF 页    | 涵盖知识点                                                       | 依赖     |
| ------------------------- | --------- | ---------------------------------------------------------------- | -------- |
| `00_autoencoder.cpp`      | 268–271   | CNN Encoder-Decoder、MaxPool 压缩、ConvTranspose 扩张、MSE 重建   | LibTorch |
| `01_vae.cpp`              | 274–278   | 重参数化技巧、β-VAE 损失（BCE+KL）、ELBO、概率隐空间生成       | LibTorch |

### 二、GAN — PDF 第 279–287 页

| 文件           | PDF 页    | 涵盖知识点                                                     | 依赖     |
| -------------- | --------- | -------------------------------------------------------------- | -------- |
| `02_gan.cpp`   | 280–287   | Generator/Discriminator、minimax 博弈、五步训练、detach 控制梯度 | LibTorch |

### 三、采样策略 — PDF 第 290–299 页

| 文件                          | PDF 页    | 涵盖知识点                                                       | 依赖 |
| ----------------------------- | --------- | ---------------------------------------------------------------- | ---- |
| `03_sampling_strategies.cpp`  | 290–299   | 温度缩放、贪婪解码、Top-K、Top-P、束搜索概念、终止条件（EOS/max_len） | STL  |

### 四、评估指标与文本分析 — PDF 第 300–306 页

| 文件                          | PDF 页    | 涵盖知识点                                                       | 依赖 |
| ----------------------------- | --------- | ---------------------------------------------------------------- | ---- |
| `04_evaluation_metrics.cpp`   | 300–306   | TF-IDF、N-gram、困惑度、BLEU/ROUGE 简化计算、评估维度总览         | STL  |

---

## 编译与运行

```bash
# 环境：C++17 + LibTorch (00-02) / 纯 STL (03-04)
cd build && cmake ..
cmake --build . --target autoencoder -j$(nproc)
```

```bash
./build/chapter08/autoencoder          # CNN 自编码器训练+重建
./build/chapter08/vae                  # VAE 重参数化+生成
./build/chapter08/gan                  # GAN 五步对抗训练
./build/chapter08/sampling_strategies   # 4 种采样策略+温度对比(无外部依赖)
./build/chapter08/evaluation_metrics    # TF-IDF/N-gram/BLEU/ROUGE(无外部依赖)
```

---

## 技术速查

### 重参数化技巧 (Reparameterization Trick)

| 步骤         | 代码                                           | 可微？ |
| ------------ | ---------------------------------------------- | ------ |
| 原始采样     | `z = sample(N(μ, σ²))`                         | ✗      |
| 重参数化     | `z = μ + σ * ε` where `ε ~ N(0,1)`            | ✓      |

### GAN 训练关键点

| 关注项           | 建议值/做法                                  |
| ---------------- | ------------------------------------------- |
| optimizer        | Adam(lr=0.0002, β₁=0.5, β₂=0.999)           |
| detach控制       | `fake_images.detach()` when training D       |
| 训练比例         | 通常 D:G = 1:1 或 k:1                         |
| 预训练D          | 先用纯真实数据训练几个 iteration——提高稳定性 |
| 模式坍塌         | G 反复生成相同样本 → 增大 batch、加噪声       |

### LLM 采样策略决策树

```
需要确定性输出？
  Y → Greedy(最快) 或 Beam Search(更优，翻译/摘要)
  N → 需要控制多样性？
      → Top-K(k=40~50) 简单固定候选数
      → Top-P(p=0.9) 自适应候选数（现代默认）
      温度: 问答0.2/创意写作1.0/随机探索2.0
```

---

## PDF 完整内容对照

| 书本页   | 内容                                                          | 实现文件                       |
| -------- | ------------------------------------------------------------- | ------------------------------ |
| 266–267  | 判别 vs 生成模型、生成 AI 发展脉络                            | --                             |
| 267–270  | CNN 自编码器架构（Encoder/Bottleneck/Decoder）、MSE 训练       | `00_autoencoder.cpp`           |
| 270–271  | 去噪自编码器概念                                               | --                             |
| 271–273  | VAE 原理、重参数化技巧推导（∂z/∂μ=1, ∂z/∂σ=ε）               | `01_vae.cpp`                   |
| 274–278  | VAE LibTorch 实现（编码器/bottleneck/解码器+损失）            | `01_vae.cpp`                   |
| 278–279  | Seq2Seq 自编码器（RNN/LSTM 变体）                             | --                             |
| 279–283  | GAN 架构（G/D 对称）、minimax 数学、五步训练                   | `02_gan.cpp`                   |
| 284–287  | GAN LibTorch 实现（Generator/Discriminator/训练循环）         | `02_gan.cpp`                   |
| 287–289  | 自回归生成、因果语言建模                                       | `03_sampling_strategies.cpp`   |
| 290–291  | 温度缩放、Softmax 概率变换                                     | `03_sampling_strategies.cpp`   |
| 292–298  | 贪婪解码→Beam Search→Top-K→Top-P、终止条件（EOS/max_len）     | `03_sampling_strategies.cpp`   |
| 300–301  | TF-IDF、N-gram（unigram/bigram/trigram）                      | `04_evaluation_metrics.cpp`    |
| 302–304  | 困惑度、BLEU(精度)、ROUGE(召回)、METEOR(综合)                | `04_evaluation_metrics.cpp`    |
| 304–306  | 模型级指标 BERTScore、评估维度总览                             | `04_evaluation_metrics.cpp`    |
| 306–308  | 章节小结 + 拓展阅读（VAE/GAN/WGAN/BERT 论文）                | --                             |

---

## 注意事项

- **03/04 为纯 STL 实现**，无需任何外部依赖，始终可编译运行
- **00-02 需要 LibTorch**，CMake 找不到时自动跳过
- GAN 训练极不稳定——实际应用需要 DCGAN/WGAN 等改进版架构（本章仅演示基础版本）
- 束搜索完整实现需要模型推理（本章概念演示用伪代码）
- BLEU/ROUGE 的简化计算仅作教学，生产环境使用专用库（sacreBLEU、rouge-score）
- `β-VAE`：β>1 时增强解耦，使各隐变量独立编码不同语义因子
