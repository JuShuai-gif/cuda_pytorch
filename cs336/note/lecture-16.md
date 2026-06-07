# Lecture 16: Scaling Laws

## 1. 本讲核心问题

当你有一个固定的算力预算（比如 1000 块 H100 跑 3 个月），你应该选择多大的模型？用多少数据训练？**本讲的核心问题是：模型参数量 N、训练数据量 D、计算量 C 之间到底存在怎样的数学关系？** 更进一步：这些 scaling law 如何指导实际的模型设计和资源分配？为什么"predictability"比"optimality"更重要？

---

## 2. 通俗解释

### Scaling Law ≈ 烘焙配方

把训练 LLM 比作烤一个多层蛋糕：

- **面粉量 = 数据量 D**（原料，越多越好但需要比例合适）
- **烤箱容量 = 模型参数量 N**（容器，太大浪费空间，太小不够装）
- **烤制时间 = 计算量 C**（总资源预算，包括烤箱功率 × 时间）

**核心发现（Chinchilla）：** 如果你想做"最优"的蛋糕，面粉和烤箱容量需要大约 20:1 的比例（$D \approx 20N$）。用 10 倍的面粉（数据）配 1 倍的烤箱（模型），不如用 10 倍的烤箱配 10 倍的面粉——**两者需要同步增长。**

**但如果面粉很便宜，烤箱很贵（推理成本高呢）？**
那就多放面粉（$D \gg 20N$），烤出一个"小而精"的蛋糕——在吃的时候（推理）省钱。这就是 Llama 3 的 **overtraining** 策略。

### Kaplan 发现 vs Chinchilla 修正

- **Kaplan（2020）：** "多放烤箱少放面粉也行"（偏好大模型，数据不用太多）
- **Chinchilla（2022）：** "错！面粉和烤箱要同步增长"（$D \approx 20N$ 是最优的）

这就像两个烘焙师对配方有不同理解——Chinchilla 用更严谨的实验设计证明了 Kaplan 的结论有偏差。

---

## 3. 数学公式 + 工程意义

### 3.1 Kaplan Scaling Laws (OpenAI, 2020)

Kaplan 等人提出的三个核心关系：

**Law 1: 模型参数量 N 与 Loss 的关系**
$$
L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N} + L_{\infty}
$$
其中 $\alpha_N \approx 0.076$，$N_c \approx 8.8 \times 10^{13}$（常数）。

**Law 2: 数据量 D 与 Loss 的关系**
$$
L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D} + L_{\infty}
$$
其中 $\alpha_D \approx 0.095$。

**Law 3: 计算量 C 与 Loss 的关系**
$$
L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C} + L_{\infty}
$$
其中 $\alpha_C \approx 0.050$。

**工程意义：** 给定计算预算 $C$，Kaplan 建议"优先增大模型而非数据"：
$$
N_{opt} \propto C^{0.73}, \quad D_{opt} \propto C^{0.27}
$$

### 3.2 Chinchilla Scaling Laws (DeepMind, 2022)

Chinchilla 使用**三种不同方法**验证了 scaling law：

#### Approach 1: 固定模型大小 N，变化数据量 D

对于每个固定 N，训练不同 D 直到收敛，拟合：
$$
L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}
$$

**结果：** $\alpha \approx 0.34$，$\beta \approx 0.28$

#### Approach 2: IsoFLOP Curves（等计算量曲线）

固定计算预算，寻找最优 (N, D) 组合：

```
C (FLOPs)       N_opt (参数)     D_opt (tokens)    D_opt / N_opt
10^18           73M              2.38B              ~32
10^19           305M             9.1B               ~30
10^20           1.26B            33B                ~26
10^21           5.15B            116B               ~23
10^22           20.6B            403B               ~20
```

**曲线特征：** 在 IsoFLOP curve 上，loss 先快速下降（增加参数），然后缓慢下降（数据不够），最后到达最优点。最优点满足 $D_{opt} / N_{opt} \approx 20$。

#### Approach 3: Parametric Fit（参数化拟合）

把所有实验数据放在一起，拟合参数化公式：

$$
\hat{L}(N, D) \triangleq E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}
$$

得到最终拟合参数：
- $E = 1.69$（不可约减 loss，受限于数据本身）
- $A = 406.4$，$B = 410.7$
- $\alpha = 0.34$，$\beta = 0.28$

**推导出 Compute-Optimal 关系：**

$$
N_{opt}(C) \propto C^{\frac{\beta}{\alpha+\beta}}, \quad D_{opt}(C) \propto C^{\frac{\alpha}{\alpha+\beta}}
$$

代入数值：$N_{opt} \propto C^{0.46}$，$D_{opt} \propto C^{0.54}$

**与 Kaplan 对比：**

| | Kaplan | Chinchilla |
|---|--------|------------|
| $N_{opt} \propto$ | $C^{0.73}$ | $C^{0.46}$ |
| $D_{opt} \propto$ | $C^{0.27}$ | $C^{0.54}$ |
| $D/N$ ratio | ~1 | ~20 |
| 偏好 | 大模型、少数据 | 模型和数据同步增长 |
| **对于 10^24 FLOPs** | $N_{opt} \approx$ 500B | $N_{opt} \approx$ 70B |

**工程结论：** Chinchilla（70B 参数，1.4T tokens）用与 Gopher（280B，300B tokens）相同的计算量但更优的分配，达到了更低的 loss。

### 3.3 Overtrained Scaling Laws

Chinchilla 的结论在 **推理成本** 也纳入考虑后需要修正。假设总成本：

$$
\text{Total Cost} = \underbrace{C_{train}(N, D)}_{\text{训练成本}} + \underbrace{k \cdot C_{inference}(N)}_{\text{推理成本 × k次调用}}
$$

当 $k$ 很大时（模型被大量部署），最优的 $D/N$ 比例会大于 20——这就是 **overtraining**。

**公式修正：**
$$
\frac{D_{opt}}{N_{opt}} \approx 20 \times f(k)
$$
其中 $f(k) > 1$ 当 $k$ 足够大时。

### 3.4 muP (Maximal Update Parameterization) — µTransfer

**核心问题：** 调参在小模型上找到最优 learning rate，能否直接在大模型上用？

**muP 的答案：可以！** 通过特定的参数化方式（$\mu$P），使最优超参数在不同宽度之间保持一致：

| 超参数 | $\mu$P | Standard Param. (SP) |
|--------|--------|---------------------|
| Learning rate | 稳定转移 | 需要重新调 |
| Initialization std | 稳定转移 | 不稳定 |
| Embedding multiplier | $\propto 1/\sqrt{d}$ | 常数 |

**工程意义：** 先在小模型（如 10M 参数）做大量调参实验，找到最优超参数，然后直接 transfer 到大模型（如 10B 参数）——节省大量 GPU 成本。

---

## 4. 工业界真实实现

### 4.1 Chinchilla (DeepMind, 2022) — 改变行业的实验

**实验设计：**
- 训练了 400+ 个不同配置的模型，从 70M 到 16B 参数
- 每个配置使用 3 种以上的数据量
- 计算总预算超过 10^23 FLOPs（仅用于研究 scaling law）

**核心发现：** Gopher（280B 参数，300B tokens）如果改为 Chinchilla（70B 参数，1.4T tokens），用同样的计算量但更好的分配，loss 降低了 ~0.1（在语言建模中是巨大的改进）。

**影响：**
- 全行业重新审视模型规模规划
- "数据不够，增加参数来凑"的策略被证明是低效的
- 数据质量和数量变得比以往任何时候都重要

### 4.2 Llama 3 的 Overtraining 决策（Meta, 2024）

| 模型 | 参数 N | 训练 tokens D | D/N | 备注 |
|------|--------|--------------|-----|------|
| Chinchilla | 70B | 1.4T | 20 | 严格 compute-optimal |
| Llama 2-70B | 70B | 2T | 29 | 轻微 overtrained |
| Llama 3-8B | 8B | 15T | 1875 | 极度 overtrained |
| Llama 3-70B | 70B | 15T | 214 | overtrained |
| Llama 3-405B | 405B | 15T | 37 | overtrained |

**Meta 的理由：** "我们选择 overtrain 较小的模型，因为它们在推理时更便宜。15T tokens 对于 405B 参数来说已经远超 Chinchilla 最优，但我们相信 training longer 的好处超过了额外的训练成本。"

### 4.3 DeepSeek V3/V4 的数据规模选择

DeepSeek 使用约 **32T tokens** 训练其 MoE（Mixture of Experts）模型。由于 MoE 模型的 active parameters 远小于 total parameters，按 active parameters 计算的 D/N 可能远超 20。

### 4.4 GPT-4 的 Scaling Strategy（猜测）

虽然没有公开数据，但社区推测：
- GPT-4 是一个约 1.8T 参数的 MoE 模型（8 个 expert，每次激活 ~280B）
- 训练数据量可能在 10-15T tokens 之间
- 按 active params 计算的 D/N ≈ 35-50——这是一个有意 overtrained 的设计

---

## 5. CUDA/GPU 视角

### 5.1 Scaling Laws 指导 GPU 资源分配

```
总 GPU 资源 = GPU 数 × 训练时间

重点：Scaling law 告诉我们如何在 N 和 D 之间分配
- 增大 N：需要更多 GPU 内存（参数 + 优化器状态）
- 增大 D：需要更多训练时间（更多 steps）
```

**具体例子：计算 10^24 FLOPs 的预算怎么分配？**

| 方案 | Model (N) | Data (D) | GPU 需求 | 训练时间 |
|------|----------|----------|---------|---------|
| Kaplan-style | 500B | 500B tokens | 64x H100 (需要 TP+PP) | 较长 |
| Chinchilla | 70B | 3.5T tokens | 8x H100 (单机即可) | 较短 |

虽然 Chinchilla 方案模型小（70B vs 500B），但 D 大很多，总时间取决于硬件配置。**工程上的关键 insight：** 更小的模型更容易部署和调试。

### 5.2 IsoFLOP 曲线的工程实现

```python
def isoFLOP_sweep(flop_budget, param_range, data_range):
    """
    Run an isoFLOP sweep to find optimal (N, D) for given FLOP budget.
    
    Compute for transformers: C ≈ 6 × N × D (forward + backward)
    """
    results = []
    for N in param_range:
        # Given fixed FLOPS budget, D is determined
        D = flop_budget / (6 * N)
        # Filter: skip if D is too small or too large
        if D < min_data or D > max_data:
            continue
        # Train model with (N, D) and record loss
        loss = train_model(N, int(D))
        results.append((N, D, loss))
    
    # Find (N, D) with minimum loss
    best = min(results, key=lambda x: x[2])
    return best
```

### 5.3 muP 的实际 GPU 节省

**传统方法：** 为大模型独立调参需要 ~10-50 次试运行（每次花费数万美元 GPU 时间）。

**muP 方法：** 在 10M 参数小模型上调参（每次 ~$10 GPU），然后 transfer 到 10B 模型。**节省 ~99% 调参 GPU 成本。**

```python
# muTransfer pseudo-code
for lr in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]:
    small_model = make_model(width=128, param="muP")  # ~10M params
    train(small_model, lr=lr)  # cheap: ~$10
    
best_lr = find_best_lr()  # 3e-4

# Transfer to large model without re-tuning
large_model = make_model(width=4096, param="muP")  # ~10B params  
train(large_model, lr=best_lr)  # expensive: ~$100K, but only run once
```

---

## 6. 本讲与整个 LLM 系统的关系

```
┌──────────────────────────────────────────────────────────────┐
│   Scaling Laws 在整个 LLM 生命周期中的作用                     │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  [Scaling Law] ──▶ 决定 (N, D) 分配 ──▶ 数据收集 + 模型设计   │
│                                      │                        │
│                                      ▼                        │
│                               [训练] (Lecture 12)              │
│                                      │                        │
│                                      ▼                        │
│                          [对齐] (Lecture 15) ──▶ [部署]        │
│                                                               │
│  [推理成本] 反馈 ──▶ 修正 overtrained scaling                    │
└──────────────────────────────────────────────────────────────┘
```

### Predictability ≥ Optimality

在工业实践中，**知道 scaling law 的预测范围比恰好找到最优点更重要：**

1. **资源规划：** 知道 loss 如何随 N 和 D 变化，可以准确预估训练成本和时间
2. **商业决策：** "用 1000 万 GPU 小时能训练出什么水平的模型？"——这个问题必须用 scaling law 回答
3. **资源配置：** 如果有 15T tokens 的数据，知道模型至少需要多大才能"消化"这些数据

**核心洞察（来自 Chinchilla 论文）：** "Loss is predictable" 是 scaling law 最重要的贡献——你可以提前知道你的模型大概能有多好。

### 为什么 Scaling Law 需要 Careful Construction

1. **实验设计偏差：** 如果只用小模型拟合 → 外推到大模型时误差很大（suboptimal fitting）
2. **学习率衰减不充分：** 如果每个实验的学习率衰减不彻底 → 低估了模型的最终能力
3. **C ≈ 6ND 的假设：** 这个公式假设是 dense transformer，对 MoE、encoder-decoder 等架构需要修正
4. **Batch size 效应：** 过小的 batch size 可能导致噪音大，过大的 batch size 可能导致欠拟合

---

## 7. 面试问题

1. **Kaplan 和 Chinchilla 的 scaling laws 的核心区别是什么？为什么 Chinchilla 的结论被认为更可靠？**

   *参考答案：Kaplan 主张 $N_{opt} \propto C^{0.73}$（大模型少数据），Chinchilla 主张 $N_{opt} \propto C^{0.46}$（模型和数据同步增长，$D \approx 20N$）。Chinchilla 更可靠的原因：（1）使用了更大的模型范围（70M-16B vs 更窄的范围）；（2）每种配置训练更多数据量；（3）使用 IsoFLOP curves 作为独立验证方法；（4）仔细控制了学习率衰减等实验细节。Kaplan 的实验设计低估了小模型 + 多数据的组合效果。*

2. **什么是 IsoFLOP curve？为什么它是验证 scaling law 的好方法？**

   *参考答案：IsoFLOP curve 是固定总计算量 C 时，不同模型大小 N 对应的 loss。它提供了直接验证 $N_{opt}$ 的方法——通过非参数化的方式找到给定计算量下的最优 N。这避免了参数化拟合可能带来的偏差。*

3. **为什么 Llama 3 选择 overtrain（D/N >> 20）？这个决策的 trade-off 是什么？**

   *参考答案：推理成本考虑。overtrained 的小模型在推理时更便宜（更少参数 → 更少 GPU 内存 → 更低延迟 → 更低成本），且性能接近更大但 undertrained 的模型。Trade-off：训练成本更高（更多 token × 更多训练时间），但推理成本更低。当模型被大量部署时，额外的训练成本被推理节省所抵消。*

4. **muP（µTransfer）解决了什么问题？它的核心 idea 是什么？**

   *参考答案：muP 解决了"小模型上找到的最优超参数能否 transfer 到大模型"的问题。核心 idea：通过特定的参数化和初始化方式（学习率 $\propto 1/\text{width}$，初始化方差 $\propto 1/\text{fan\_in}$），使网络在不同宽度下的训练动态保持一致。这样最优学习率、batch size 等超参数在不同规模模型间可以稳定转移，减少大模型调参的 GPU 成本。*

5. **给定 $C = 10^{24}$ FLOPs 的预算，使用 Chinchilla 公式计算 $N_{opt}$ 和 $D_{opt}$。如果改用 Kaplan 公式呢？**

   *参考答案：*
   - Chinchilla: $N_{opt} \approx 70B$，$D_{opt} \approx 1.4T$ tokens（$D/N \approx 20$）
   - Kaplan: $N_{opt} \approx 500B$，$D_{opt} \approx 300B$ tokens（$D/N \approx 0.6$）
   
   两个方案的模型大小相差 7 倍。

6. **如果一个公司有 50T tokens 的 clean data 但只有训练 100B 模型的 GPU 预算，scaling law 如何指导决策？**

   *参考答案：这个场景是"数据过剩 + 算力受限"。Chinchilla 说应该用 50T tokens 几乎全部都用来训练 100B 模型（$D/N = 500$，极度 overtrained）。虽然可能不完全 compute-optimal，但从推理成本角度可能是最优的——一个 100B 的模型大量 overtrained 可能比 400B 的 compute-optimal 模型在推理时更经济。如果推理成本是关键约束，overtraining 是更好的策略。*

---

## 参考：Scaling Laws 的演进年表

| 年份 | 工作 | 核心贡献 |
|------|------|---------|
| 2017 | Hestness et al. | 最早提出 deep learning 的 power-law scaling |
| 2020 | Kaplan et al. (OpenAI) | 系统的 LLM scaling law，N/D/C 关系 |
| 2022 | Chinchilla (DeepMind) | 修正 Kaplan，提出 $D \approx 20N$ |
| 2022 | Hoffmann et al. | 详细分析 scaling law 的实验设计方法论 |
| 2023 | LLaMA (Meta) | 实际验证 overtrained 模型的优势 |
| 2024 | muP / µTransfer | 小模型调参 → 大模型复用 |
| 2024 | Llama 3 (Meta) | 工业级 overtrained：D/N >> 20 |
| 2025 | DeepSeek V3 | 32T tokens 的超大规模数据训练 |

> **关键 Takeaways:**
> 1. Scaling law 是 LLM 研究的"牛顿定律"——对模型的性能和行为提供可量化的预测
> 2. Chinchilla 修正了 Kaplan：数据和参数需要同步增长 ($D \approx 20N$)
> 3. 工业实践越来越倾向于 overtrained：推理成本比训练成本更重要
> 4. IsoFLOP curves 是最可靠的 scaling law 验证方法
> 5. Predictability > Optimality：能预测比能精确最优更重要
> 6. muP 让超大模型调参从"烧钱"变成"科学"
