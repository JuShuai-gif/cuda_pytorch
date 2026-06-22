# 背景知识：Scaling Laws — Kaplan vs Chinchilla

## 1. 什么是 Scaling Laws？

Scaling Laws（缩放定律）描述了模型性能（loss）如何随着计算量 $C$、参数量 $N$、数据量 $D$ 的变化而变化：

$$L(N, D) = f(N, D)$$

目的：预测在给定 compute budget 下，如何最优分配 $N$ 和 $D$。

---

## 2. Kaplan Scaling Laws (2020)

### 2.1 核心结论

OpenAI 的 Kaplan 团队训练了一系列不同规模的模型，发现：

| 关系                 | 公式                               | 指数   |
| -------------------- | ---------------------------------- | ------ |
| Loss vs Params       | $L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}$ | $\alpha_N \approx 0.076$ |
| Loss vs Data         | $L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}$ | $\alpha_D \approx 0.095$ |
| Loss vs Compute      | $L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}$ | $\alpha_C \approx 0.050$ |

**Kaplan 的关键建议：**

$$N_{opt} \propto C^{0.73}, \quad D_{opt} \propto C^{0.27}$$

> 结论：增大模型比增大数据更有效！

### 2.2 实验设定

- Model family: decoder-only Transformer
- 规模范围: 768 到 1.5B 参数
- 每个模型训练到 convergence
- 使用学习率 schedule 来决定何时停止

### 2.3 存在的缺陷

Kaplan 的结论后来被发现存在以下问题：

1. **未控制 learning rate schedule**：不同规模的模型使用了不同的 LR schedule，小模型可能未被充分训练
2. **未控制 token 数**：大模型训练的 token 数比小模型多，导致 unfair comparison
3. **固定模型 shape 的假设**：实际中可以通过调整 depth/width 来优化

---

## 3. Chinchilla Scaling Laws (2022)

### 3.1 实验设计改进

DeepMind 的 Chinchilla 团队做了更严格的实验：

- 在固定 compute budget $C$ 下，系统性地扫描不同的 $(N, D)$ 组合
- 对于每个 $(N, D)$ 组合，使用 cosine LR schedule 且**训练到结束**
- 保证了 fair comparison

### 3.2 核心公式

**Approach 1 (Fixed model sizes):**

$$\hat{L}(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}$$

拟合结果：
- $E = 1.69$ (irreducible loss，即数据本身的 entropy)
- $A = 406.4$, $\alpha \approx 0.34$
- $B = 410.7$, $\beta \approx 0.28$

**Approach 2: Compute-optimal allocation**

$$\text{given } C = 6ND, \quad N_{opt}(C) \propto C^a, \quad D_{opt}(C) \propto C^b$$

结果：

$$N_{opt} \propto C^{0.50}, \quad D_{opt} \propto C^{0.50}$$

> 关键修正：**参数量和训练数据应该等比例增长！**

### 3.3 对比表

| 指标                     | Kaplan               | Chinchilla           |
| ------------------------ | -------------------- | -------------------- |
| $N_{opt}$ 指数           | $C^{0.73}$           | $C^{0.50}$           |
| $D_{opt}$ 指数           | $C^{0.27}$           | $C^{0.50}$           |
| 训练 70B 所需 tokens     | ~300B                | ~1.4T                |
| 方法论                   | 固定 N，变 D         | 固定 C，扫 (N, D)    |

### 3.4 Chinchilla 的实践影响

以 70B 参数模型为例：
- **Kaplan 建议**：训练 ~300B tokens → 训练不足！
- **Chinchilla 建议**：训练 ~1.4T tokens → 更充分的训练

这直接影响了 LLaMA 的设计：LLaMA-7B 训练了 1T tokens。

---

## 4. IsoFLOP Curves

### 4.1 概念

IsoFLOP curve 是在**固定 FLOP budget** 下，loss 随 $(N, D)$ 变化的等高线：

```
     D (tokens)
     ^
     │   ┌─────────────────────────┐
     │   │  IsoFLOP = 1e19         │
     │   │     ╲                   │
     │   │       ╲  ★ optimal     │
     │   │         ╲               │
     │   │  IsoFLOP = 1e18        │
     │   │     ╲                   │
     │   │       ╲  ★             │
     │   └─────────────────────────┘
     └──────────────────────────────→ N (params)
```

### 4.2 绘制方法

1. 运行多组 $(N_i, D_i)$ 训练，记录 loss $L_i$
2. 对于每组，计算 FLOPs $C_i \approx 6 N_i D_i$
3. 拟合函数 $L(N, D) = E + AN^{-\alpha} + BD^{-\beta}$
4. 从拟合函数生成 dense grid，绘制 contour

### 4.3 Compute-optimal 分配

Compute-optimal 配置在每条 IsoFLOP curve 上对应 loss 最小的点：

$$\frac{\partial L}{\partial N}\bigg|_{C=6ND} = 0$$

推导得到：

$$\frac{N_{opt}}{D_{opt}} = \left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha+\beta}} \cdot D^{\frac{\beta-\alpha}{\alpha+\beta}}$$

当 $\alpha \approx \beta$ 时（Chinchilla 的情况），$N_{opt} \propto D_{opt}$。

---

## 5. 超越 Chinchilla

### 5.1 DeepSeek 的发现 (2024)

DeepSeek 发现对于 MoE 模型，scaling law 有所不同：

- MoE 模型可以用更少的 compute 达到同等性能
- 但 optimal $N$ vs $D$ 的 ratio 可能因为 expert 数量而变化

### 5.2 数据重复的影响

Chinchilla 假设数据是 unique 的。但实际中：
- 多 epoch 训练可以从 "Chinchilla-optimal" 继续降低 loss
- 数据重复 4 次以内 loss 仍在下降（Muennighoff et al., 2023）
- 所以 "Chinchilla-optimal" 是一个建议，非硬性限制

### 5.3 小模型的 Scaling

对于 < 1B 的模型，Chinchilla 的预测可能不准确：

- 小模型的 $N_{opt}$ 可能需要更少的 tokens
- 实践中很多小模型（如 MobileLLM）采用更大 $D/N$ 比例

---

## 6. 核心公式速查

| 公式                                  | 含义                            |
| ------------------------------------- | ------------------------------- |
| $L(N, D) = E + AN^{-\alpha} + BD^{-\beta}$ | Chinchilla 参数化             |
| $C \approx 6ND$                       | Compute FLOPs (Kaplan approx)   |
| $N_{opt} \propto C^a$                 | Compute-optimal 参数            |
| $D_{opt} \propto C^b$                 | Compute-optimal 数据            |
| $N_{opt} \propto D_{opt}$ (Chinchilla)| 参数和数据等比例增长            |
