# Lecture 04: 训练基础 (Training Fundamentals)

## 本讲核心问题

1. Cross-entropy loss 在 LLM 中的实际计算是怎样的？为什么不用 MSE？
2. AdamW 为什么把 weight decay 从梯度更新中"解耦"出来？这解决了什么实际问题？
3. Cosine vs WSD (Warmup-Stable-Decay) learning rate schedule：为什么 WSD 越来越流行？
4. Batch size 的"临界点"现象：超过 critical batch size 后为什么收益递减？
5. muP (Maximal Update Parameterization) 如何让我们用小模型做超参搜索，直接迁移到大模型？

## 通俗解释

### Cross-entropy Loss ≈ 猜词游戏评分

语言模型训练就是玩一个猜词游戏：给定前面所有的词，猜下一个词是什么。如果模型给正确答案 99% 的概率，loss 接近 0；如果只给 1%，loss 很高。

Cross-entropy 就是衡量"模型的概率分布"和"真实分布"之间的差距。对于语言模型，真实分布是 one-hot（正确答案概率 100%，其他 0%）：

```
loss = -log(P_model(correct_token))
```

为什么不用 MSE？因为 MSE 对概率输出的惩罚是二次的——当模型给正确答案 1% 概率时，MSE 惩罚的是平方差，cross-entropy 惩罚的是负对数。Cross-entropy 与 softmax 搭配，梯度形式优雅 (`p - y`)，且是最大似然估计的直接实现。MSE 会导致训练初期梯度很小（saturation problem）。

### AdamW ≈ 下山时不仅看当前坡度，还看历史惯性，同时不跑偏

想象你在雾气弥漫的山上往下走：

- **SGD**：只看脚下的坡度，每次走一小步。容易在缓坡上走太慢，在陡坡上冲过头
- **SGD+Momentum**：除了看脚下，还保持之前的运动方向（惯性），像滚雪球
- **Adam**：除了惯性（m = 一阶动量），还根据"这个方向坡度的变化历史"调整步子大小（v = 二阶动量）。陡坡自动减小步长，缓坡自动增大步长
- **AdamW**：在 Adam 的基础上，weight decay 单独施加在原始参数上，不混在梯度更新里

为什么解耦 weight decay？因为在 Adam 中，weight decay 和梯度都被 adaptive learning rate 除过——这导致 L2 正则化效果不稳定，大梯度方向的正则化被削弱。

### Learning Rate Schedule "WSD" 为什么流行

- **Cosine**：训练全程按余弦曲线衰减，前期快速下降，后期缓慢衰减。问题：如果训练到一半发现数据不够，重新训练会打乱 schedule
- **WSD (Warmup-Stable-Decay)**：warmup 阶段快速上升（前 1-5% steps），stable 阶段保持恒定（主体 90-95%），decay 阶段快速下降（最后 1-5%）。好处是 stable 阶段可以随时停止和恢复，方便数据配比调整和 multi-phase 训练

就像煮饭：大火爆炒（warmup），转中火慢炖（stable 稳定期），最后小火收汁（decay）。中火阶段越长，炖得越烂（数据处理越充分）。

## 数学公式 + 工程意义

### Cross-entropy Loss 的梯度

```
L = -1/N * sum(log(softmax(logits)[y_i]))
  = -1/N * sum(logits[y_i] - log(sum(exp(logits))))

dL/d(logits_j) = softmax(logits)[j] - (1 if j==y_i else 0)
                = p_j - y_j  # Beautifully simple!
```

梯度是预测概率和真实标签的差——这意味梯度天然在 (0, 1) 之间，不会爆炸。工程上，`log_softmax` 有数值稳定版本（减去 max(logits) 防止 exp 溢出）。

### AdamW 更新规则

```
# Standard Adam
m_t    = beta1 * m_{t-1} + (1 - beta1) * g_t        # 一阶动量（惯性）
v_t    = beta2 * v_{t-1} + (1 - beta2) * g_t^2      # 二阶动量（方差估计）
m_hat  = m_t / (1 - beta1^t)                         # 偏差修正
v_hat  = v_t / (1 - beta2^t)
theta  = theta - lr * m_hat / (sqrt(v_hat) + eps)

# AdamW: Decouple weight decay
theta  = theta - lr * m_hat / (sqrt(v_hat) + eps) - lr * wd * theta
#          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^
#                    gradient step                         weight decay (decoupled)
```

解耦的数学意义：weight decay 等价于每一步把参数乘 (1 - lr*wd)，这是严格的 L2 正则化。在原始 Adam 中，weight decay 被除以 sqrt(v_hat)，导致 L2 惩罚强度因参数而异。

### Learning Rate Schedule 公式

**Cosine Schedule**：
```
lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T))
```

**WSD Schedule**：
```
lr(t) = lr_min  + (lr_max - lr_min) * t/T_warmup                     for t < T_warmup
        lr_max                                                       for T_warmup <= t < T_stable
        lr_min  + (lr_max - lr_min) * (T_total - t)/(T_total - T_stable)  for t >= T_stable
```

工程意义上，WSD 的 stable 阶段让模型在接近最优 LR 下持续学习——这在 cosine 中是不可能的（cosine 一直在衰减）。实验表明 stable 阶段占 90% 训练时间的 WSD 比 cosine 最终 perplexity 更低。

### Gradient Clipping

```
if norm(grad) > max_norm:
    grad = grad * max_norm / norm(grad)
```

为什么需要？训练大模型时偶尔会出现单个 batch 的梯度极大（gradient spike），一次更新就能把参数推出稳定区域。Gradient clipping 限制了单步更新的最大幅度。Llama 使用 `max_norm=1.0`。

### Weight Initialization 与 muP

**Xavier Initialization (Glorot)**：
```
W ~ U(-sqrt(6/(n_in + n_out)), sqrt(6/(n_in + n_out)))
```
保持前向和反向传播的方差不变。

**Kaiming Initialization (He)**：
```
W ~ N(0, sqrt(2/n_in))
```
针对 ReLU 激活的优化（一半神经元被激活时方差减半，所以乘以 2）。

**muP (Maximal Update Parameterization)**：
核心发现——如果我们希望小模型和大模型的训练动态一致（相同的 optimal LR、相同的 loss curve），参数的初始化和 LR 需要按如下方式缩放：

| Parameter | Fan-in dependence |
|-----------|------------------|
| Input embedding | LR ~ 1, init ~ 1 |
| Hidden weights  | LR ~ 1/fan_in, init ~ 1/sqrt(fan_in) |
| Output head     | LR ~ 1/fan_in, init ~ 1/sqrt(fan_in) |

muP 让研究员可以用千万参数的小模型搜索超参，找到的最优 LR 直接用在千亿参数的大模型上——节省了巨大的试错成本。

### Critical Batch Size

训练 loss 的减少率随 batch size 增加而提升——但有上限：

```
当 batch_size < B_crit:
    增加 batch_size -> loss 下降更快（线性加速）

当 batch_size > B_crit:
    增加 batch_size -> 几乎没收益（效率递减）
```

B_crit 取决于模型大小和数据难度，通常在大模型中约为 2M-4M tokens。这就是为什么 Llama 2 使用 global batch size = 4M tokens——刚好在 critical batch size 附近。

## 工业界真实实现

### Llama 的训练配置

Llama 2-70B 的超参配置：

```yaml
optimizer: AdamW
learning_rate: 1.5e-4
weight_decay: 0.1
beta1: 0.9
beta2: 0.95
eps: 1e-5

schedule: cosine
warmup: 2000 steps
total_steps: ~2T tokens / 4M tokens_per_step = ~500,000 steps

batch_size:
  global: 4M tokens (e.g. 1024 sequences * 4096 tokens)
  micro: depends on GPU memory

gradient_clipping: 1.0

inductive_bias:
  - use_cache: True (during training, reuse KV cache within sequence)
  - mixed_precision: bf16
```

注意 beta2=0.95 而非默认的 0.999——这是 LLM 训练的常见调整。beta2 大意味着用更长的历史来估计梯度方差，对于长训练（>100K steps）是有益的；但太大会延迟对新数据模式的适应。0.95 是经验平衡点。

### DeepSeek-V3 的训练技巧

DeepSeek-V3 使用 fp8 混合精度训练（行业首创的大规模 fp8 训练）：

- Forward: fp8（大幅节省显存和通信）
- Gradients: bf16
- Optimizer states: fp32
- 使用 block-wise quantization 减少精度损失

他们还使用了 **multi-token prediction**：每个位置不仅预测下一个 token，还预测下下个 token。这增加了训练信号密度，在不增加显存的情况下提升了模型效果。

### nanoGPT 的学习价值

nanoGPT（Andrej Karpathy）是约 500 行代码的最小可训练 GPT 实现。虽然不是工业级，但它展示了训练流程的完整结构：

```python
# Training loop essence (from nanoGPT)
for iter_num in range(max_iters):
    # Sample a batch
    x, y = get_batch('train')

    # Forward pass
    logits, loss = model(x, y)

    # Backward pass
    optimizer.zero_grad(set_to_none=True)  # set_to_none saves memory
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Optimizer step
    optimizer.step()

    # Learning rate decay
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```

`set_to_none=True` 而不是 `zero_grad()`——这告诉 PyTorch 直接释放梯度 tensor 而不是写 0，节省了 HBM 带宽。

### Megatron-LM 的分布式训练中的优化器

Megatron-LM 在 tensor parallelism + pipeline parallelism + data parallelism 三者叠加时，optimizer step 发生在每个 data parallel rank 上。所有 micro-batch 的梯度先 reduce（all-reduce across DP ranks），然后在每个 rank 上独立执行 AdamW step。

## CUDA/GPU 视角

### Optimizer step 为什么是 memory bound？

AdamW 的一次 update 流程是 elementwise 操作：

```python
# For each parameter p:
m = beta1 * m + (1 - beta1) * p.grad       # Read m, Read grad, Write m
v = beta2 * v + (1 - beta2) * p.grad^2     # Read v, Read grad, Write v
m_hat = m / (1 - beta1^t)                   # Read m, Write m_hat
v_hat = v / (1 - beta2^t)                   # Read v, Write v_hat
p = p - lr * m_hat / (sqrt(v_hat) + eps)   # Read p, Write p
p = p - lr * wd * p                         # Read p, Write p (AdamW decoupled)
```

这些都是 elementwise 操作，**算术强度极低**（~10-20 FLOPs/byte）。在 H100 上，optimizer step 每秒处理的参数量受 HBM bandwidth（3.35 TB/s）限制。70B 参数（bf16 = 140 GB）的 optimizer step 在最理想情况下也需要 `140 GB / 3350 GB/s = 41ms`。

实际中 optimizer step 需要更多时间，因为有多步读写。Fused Adam 内核（如 `apex.optimizers.FusedAdam`）将多步融合成一次 HBM pass，能快 2-3x。

### set_to_none=True vs zero_grad

```python
optimizer.zero_grad()           # WRITES zero to all gradient tensors (memory-bound!)
optimizer.zero_grad(set_to_none=True)  # Just sets .grad = None (O(1) operation)
```

大模型中 `zero_grad()` 需要把每个参数的梯度 tensor 写 0——对 70B 模型就是 140 GB 的 HBM 写入。`set_to_none=True` 用 Python 的垃圾回收绕过了这一步。这是最简单的 GPU 优化之一，但很多代码库忽略了。

### Gradient all-reduce 的通信开销

在 8 卡数据并行中，每次 optimizer step 之前需要 all-reduce 梯度。8 卡 H100 的 NVLink 带宽是 900 GB/s（双向），但 70B 模型的梯度有 140 GB。All-reduce 的理论延迟：

```
time = 2 * (n-1)/n * data_size / bandwidth
     = 2 * 7/8 * 140 GB / 900 GB/s = 0.27 seconds
```

这就是为什么在大规模训练中，通信成为主要瓶颈之一——forward+backward 可能只要 1 秒，但 all-reduce 就要 0.27 秒。ZeRO 和 FSDP 通过分片减小了 all-reduce 的数据量。

## 本讲与整个 LLM 系统的关系

```
Tokenizer -> Embedding -> Attention -> MLP -> Loss -> Optimizer -> Distributed -> Inference
                                                     ^^^^^^^^^^^^^^
                                                      本讲核心  |
```

Training 阶段是整个 LLM 生命周期中计算量最大、成本最高的部分。训练成本通常占模型总成本的 60-90%（取决于是否开源复用）。理解 loss 的计算、optimizer 的选择、LR schedule 的调优，不仅影响模型质量，还直接决定训练能否在预算内完成。

这部分的优化思路通常是：用系统技巧（activation checkpointing, gradient accumulation, ZeRO）压榨显存，为更大的 batch size 腾空间；用数学技巧（AdamW 解耦, muP, WSD schedule）让训练更稳定、更高效。

## 面试问题

**Q1: Adam 和 AdamW 的核心区别是什么？为什么 LLM 都用 AdamW？**

A: 核心区别是 weight decay 的解耦。在标准 Adam 中，weight decay 是通过在梯度上加 `wd * theta` 实现的，这个值会被二阶动量 v_hat 缩放——大梯度方向的 weight decay 被衰减。AdamW 直接在参数更新时减去 `lr * wd * theta`，使得 weight decay 在所有方向均匀施加。LLM 使用 AdamW 是因为：(1) 解耦后 L2 正则化效果更强且更一致；(2) SGD 中 weight decay 和 L2 等价，但在 Adam 中不等价——AdamW 恢复了这种等价性；(3) 实验显示 AdamW 在大模型上泛化更好。

**Q2: 为什么 Transformer 训练使用 warmup？**

A: 训练初期，参数是随机初始化的，梯度的方向和大小变化很大。Adam 的二阶动量估计 v 在前几步是不准确的——warmup 用小学习率"预热"，让 v 积累足够信息，然后再加速。不用 warmup 可能导致前几个 step 直接跳过最优区域。warmup 步数通常占总训练的 1-5%，大模型需要更多 warmup（Llama 用 2000 steps）。

**Q3: Critical batch size 的物理含义是什么？如何估计？**

A: Critical batch size 是训练效率的拐点——小于它，增加 batch size 线性加速；大于它，收益递减。物理含义：当 batch size 足够大时，每一步的梯度估计已经足够精确，再增大 batch 无法减少梯度噪声。估计方法：跑多个 batch size 的实验，画出 `loss_decrease_per_step * batch_size` 随 batch_size 变化的曲线——拐点就是 critical batch size。大模型的 B_crit 通常在 1M-8M tokens。工程上，当你发现 `2x batch size -> 1.1x speedup` 时，已经超过了 B_crit。

**Q4: muP 为什么能让小模型的超参直接迁移到大模型？**

A: muP 通过控制初始化方差和 LR 对模型宽度的依赖关系，使得不同宽度模型的训练动态保持一致。具体说，如果输入维度是 n，隐藏权重 W 用 sigma/sqrt(n) 初始化，LR 用 eta/n，那么在 n -> infinity 的极限下，每一层的激活和梯度的分布与 n 无关。小模型和大模型的训练动态都在这个极限的附近，所以最佳超参趋同。这可以用随机矩阵的 spectral norm 理论严格证明——本质是 Gaussian 随机矩阵的最大奇异值随宽度收敛。
