# 02｜SmoothQuant / AWQ / GPTQ：工业量化方案的动机与实测

## 本模块解决的问题

上一章（量化基础）发现：activation 的 outlier 是量化的主要敌人。本章研究三个工业方案如何各显神通地对付它，以及现代硬件的 FP8/FP4：

```text
SmoothQuant：把 activation 的 outlier "迁移"到 weight
AWQ：       量化 scale 的选择用 activation 的重要性加权（低精度下才有效）
GPTQ：      用二阶信息逐层重建权重，最小化量化误差
FP8/FP4：   现代硬件的原生低精度格式
```

配套代码：`src/compression/quantization/smoothquant.py`、`awq.py`。

---

## 1. SmoothQuant：迁移 activation outlier

### 动机

LLM 的 activation 有 **outlier channel**：少数 channel 的幅度比其他大 ~100x。per-tensor 量化时，scale = `max(|x|)/127` 被 outlier 主导，导致所有正常 channel 量化粗到不可用。

### 原理

线性层的乘积可以被"重分配"，而不改变结果：

```text
Y = X @ W = (X @ diag(s)^{-1}) @ (diag(s) @ W) = X_hat @ W_hat

s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)   （通常 alpha = 0.5）
```

outlier channel 的 `max(|X_j|)` 大 → `s_j` 大 → `X_hat = X/s` 缩小、`W_hat = W*s` 放大。**outlier 从 activation 迁移到了 weight**，两者都变平滑，per-tensor int8 重新可用。

### 本机实测

```text
direct（不迁移，per-tensor int8）: max_abs_err = 0.0733
SmoothQuant（迁移后 per-tensor int8）: max_abs_err = 0.0143   （5.1x 减少）
activation range ratio after = 0.213   （outlier 迁移后 activation 范围缩到 21%）
```

关键：`X @ W = X_hat @ W_hat` **数学上恒等**（测试验证），迁移本身无损，只是把量化的难度从 activation 转移到两个更均匀的量上。

---

## 2. AWQ：activation-aware 的 scale 搜索

### 动机

per-channel 量化用 `scale = max(|W_j|)/127`。这个 max-based scale 对**均匀高斯**权重已经接近最优，但对**含 outlier 元素**的权重 channel，max 被 outlier 拉大，整个 channel 量化粗。

### 原理

AWQ 的核心：**scale 的选择应该用 activation 的重要性（saliency）加权，并且允许 clip 少量 outlier 来换取大部分值的精度**。它搜索一个 per-channel 乘子，最小化 saliency 加权的量化误差。

### 关键实测发现：AWQ 只在低精度下有效

本实验最意外的结果是这个——**AWQ 在 int8 下没有收益，在 int4/int3 下才有**：

```text
int8（127 级）：naive max-scale 已接近最优，AWQ 无收益（reduction 1.0x）
int4（15 级） ：AWQ 1.5x 误差减少
int3（7 级）  ：AWQ 1.96~6.7x 误差减少
```

原因：int8 有 127 个量化级别，max-based scale 的 rounding 误差已经足够小；int4/int3 级别少，scale 的 rounding 误差大，此时"clip 掉少数 outlier、让大多数值用更细的 scale"收益巨大。

**这正是 AWQ 论文的真实定位**：AWQ 主要是为 **int4/int3** 权重量化设计的（int8 用 per-channel max 就够）。

### 本机实测（int4，含 outlier 权重）

```text
naive max-based：weighted error = 46.3
AWQ（clip 搜索）：weighted error = 28.0   （1.66x 减少）
mean multiplier  = 0.717   （AWQ 选的最优 scale 是 max 的 71.7%，即主动 clip）
```

`mean_multiplier = 0.717 < 1` 说明 AWQ 确实选择了"clip 少量 outlier、让 scale 缩小"的策略。

---

## 3. GPTQ：二阶信息逐层重建

### 原理

GPTQ 不满足于"每层独立量化"，而是用 **Hessian 信息**做逐层重建：

```text
1. 逐层处理：量化当前层的权重
2. 用二阶信息（Hessian H = 2X^T X）补偿量化误差：
   量化权重 w_q 后，对剩余权重做更新，最小化该层的输出误差
3. 递归处理所有权重列（Hessian 的 Cholesky 分解做增量更新）
```

核心洞察：量化误差可以在**同层剩余权重**里被补偿，而不是独立地每个权重都量化。这让 GPTQ 在 int3/int4 下比 AWQ 更精确，但实现复杂（需要 Cholesky 分解 + 逐列更新）。

**本模块不实现 GPTQ 的完整算法**（需要 Hessian 计算和增量 Cholesky，属于独立的研究级实现），只讲原理。工业上直接用 `AutoGPTQ` / `llm-awq` 等库。

### 三个方案的定位对比

| 方案 | 粒度 | 精度（int4） | 实现复杂度 | 特点 |
|---|---|---|---|---|
| per-channel max | channel | 基线 | 极低 | 不需要数据 |
| SmoothQuant | channel | 中（需迁移） | 低 | 解决 activation outlier |
| AWQ | channel | 中高 | 中 | 需要 calibration 数据 |
| GPTQ | channel | 高 | 高 | 二阶重建，最精确 |

---

## 4. FP8 / FP4：现代硬件低精度

### FP8

FP8 有两种格式（IEEE/OCP 标准）：

```text
E4M3：4 bit 指数 + 3 bit 尾数（精度高，范围小）—— 前向 weight/activation
E5M2：5 bit 指数 + 2 bit 尾数（范围大，精度低）—— 梯度（训练用）
```

FP8 相比 INT8 的优势：**不需要 calibration**（浮点格式有自适应指数，天然覆盖大动态范围），量化误差更可控。Hopper（H100）和 Blackwell（含本机 Thor）的 Tensor Core 原生支持 FP8。

### FP4

更激进的 4-bit 浮点（如 E2M1），用于超低精度推理。Blackwell 的 Tensor Core 开始支持 FP4。精度损失大，通常需要 GPTQ 类方法补偿。

### 本机实测状态（诚实记录）

```text
torch.float8_e4m3fn / float8_e5m2：dtype 定义存在
fp8 原生 matmul：Not Implemented（addmm_cuda 不支持 Float8_e4m3fn）
```

本机 torch 2.11 的原生 fp8 matmul **未实现**（需要 torchao 或专用 kernel），所以 FP8 的实测**标记 Not Validated**，只讲格式理论。这是"禁止伪造"原则的又一次应用：dtype 存在不等于可用。

---

## 5. 回答：为什么这些方案都聚焦 weight 而非 activation

回顾三个方案，SmoothQuant 管 activation、AWQ/GPTQ 管 weight。共同逻辑：

1. **activation 是运行时数据，无法离线优化**——weight 可以离线量化、搜索、重建（AWQ/GPTQ 都是离线对 weight 做文章）。
2. **activation 的 outlier 是动态的**，只能靠"迁移"（SmoothQuant）或"动态 scale"（per-token/动态量化）处理。
3. **weight 的量化成本可以摊到离线**，所以 weight 可以用更贵的算法（GPTQ 的 Hessian）。

所以工业上的常见组合：**SmoothQuant 迁移 outlier + AWQ/GPTQ 量化 weight + 动态 per-token 量化 activation**。

---

## 6. 本模块闭环小结

```text
问题：activation outlier 和权重非均匀分布破坏量化精度
      ↓
SmoothQuant：迁移 outlier（5.1x 误差减少，数学无损）
AWQ：        saliency 加权 scale 搜索（int4 下 1.66x，int8 无收益）
GPTQ：       二阶逐层重建（理论，最精确最复杂）
FP8/FP4：    原生浮点低精度（本机 torch 未实现，Not Validated）
      ↓
下一步：Stage 9 模型剪枝（unstructured/structured/channel/head + 块稀疏）
```

要继续就说「继续」。
