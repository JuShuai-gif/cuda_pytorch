# 实验 4 常见错误

## 1. 伪量化实现错误

### 错误：在错误维度上进行分组
**问题描述：** 将权重沿 out_features 分组而非 in_features 分组。
**正确做法：** Per-group 量化应沿 in_features 维度分组，每组独立计算量化参数。
```python
def pseudo_quantize_weight(weight, bits=4, group_size=128):
    # weight: (out_features, in_features)
    out_feat, in_feat = weight.shape
    result = torch.zeros_like(weight)
    
    qmax = 2 ** (bits - 1) - 1
    
    for oc in range(out_feat):
        # 沿 in_features 维度分组
        for g_start in range(0, in_feat, group_size):
            g_end = min(g_start + group_size, in_feat)
            group = weight[oc, g_start:g_end]
            
            max_val = group.abs().max()
            scale = max_val / qmax if max_val > 0 else 1.0
            
            q_val = torch.clamp(torch.round(group / scale), -qmax, qmax)
            result[oc, g_start:g_end] = q_val * scale
    
    return result
```

### 错误：用 FP32 存储量化值
**问题描述：** 量化后用 `torch.float32` 存储量化后的整数。
**正确做法：** 在实际部署中，量化值应使用 `torch.int8` 或 `torch.int4` 存储。模拟阶段用 FP32/FP16 是正确的。

---

## 2. 显著性通道识别错误

### 错误：用权重幅度而非激活幅度判断显著性
**问题描述：** 根据权重的绝对值大小来判断显著性。
**正确做法：** AWQ 的"显著性"是基于激活值而非权重的。应该看该通道的激活值幅度有多大。
```python
# 错误：使用权重大小
salient_channels = torch.argsort(weight.abs().sum(dim=1))[-k:]

# 正确：使用激活幅度
salient_channels = torch.argsort(activation_magnitudes)[-k:]
```

### 错误：在错误层收集激活统计
**问题描述：** 只在 Embedding 层收集激活统计。
**正确做法：** 应该对所有 Linear 层（Q、K、V、O、gate、up、down）收集激活统计。

---

## 3. 缩放操作错误

### 错误：缩放因子方向错误
**问题描述：** 对权重缩放后，激活缩放因子的计算方向反了。
**正确做法：**
```python
# 对权重乘以 scale_factors
scaled_weight = weight * scale_factors.view(-1, 1)

# 对前一层激活除以 scale_factors（注意这是前一层输出通道的缩放）
# 激活缩放因子 = 1 / scale_factors
activation_inv_scale = 1.0 / scale_factors
```

### 错误：忘记缩放等效性的前提条件
**问题描述：** 在有非线性激活函数的层之间使用缩放。
**正确做法：** 缩放等效性只在纯线性层之间成立。如果在两个 Linear 层之间有 ReLU 等非线性激活，需要额外处理。
```python
# 对于 Linear -> ReLU -> Linear 的情况
# 不能直接使用缩放等效性，因为 ReLU 不是线性齐次的
# ReLU(α * x) != α * ReLU(x)
```

---

## 4. 自动缩放搜索错误

### 错误：搜索范围过大或过小
**问题描述：** alpha_range 设置不合理。
**正确做法：** 论文中典型的搜索范围是 (0.5, 2.0)，步长约 0.1。
```python
candidates = torch.linspace(0.5, 2.0, 20)  # 0.5, 0.58, 0.66, ..., 2.0
```

### 错误：搜索粒度过细导致耗时过长
**问题描述：** 对 LLM 的每个通道都用 100+ 个候选值搜索。
**正确做法：** 论文中使用 20 个候选值，且只对显著通道进行搜索。

### 错误：在搜索时破坏原始权重
**问题描述：** 搜索过程中直接修改了权重张量。
**正确做法：** 使用 `weight.clone()` 或在搜索后恢复。
```python
for ch in salient_channels:
    best_error = float("inf")
    best_scale = 1.0
    original_row = weight[ch].clone()
    
    for alpha in candidates:
        scaled_row = original_row * alpha
        q_row = pseudo_quantize_weight(scaled_row.unsqueeze(0), bits).squeeze(0)
        error = F.mse_loss(q_row / alpha, original_row).item()
        if error < best_error:
            best_error = error
            best_scale = alpha
    
    best_scales[ch] = best_scale
```

---

## 5. 评估错误

### 错误：困惑度计算中的维度处理
**问题描述：** 在计算交叉熵时，logits 和 targets 的 reshape 操作错误。
**正确做法：**
```python
# 自回归语言模型的标准困惑度计算
logits = model(input_ids[:, :-1])  # (B, T-1, vocab_size)
targets = input_ids[:, 1:]         # (B, T-1)

loss = F.cross_entropy(
    logits.reshape(-1, logits.size(-1)),  # (B*(T-1), vocab_size)
    targets.reshape(-1),                   # (B*(T-1),)
)
perplexity = torch.exp(loss)
```

### 错误：使用相同的校准数据和评估数据
**问题描述：** 校准数据和困惑度评估使用相同的数据。
**正确做法：** 应使用不同的数据：校准数据用于收集统计和搜索，评估数据用于测量最终的困惑度。

---

## 6. 概念性错误

### 错误：混淆 AWQ 和 GPTQ
**问题描述：** 认为 AWQ 和 GPTQ 是一样的。
**正确做法：**
- **GPTQ**: 基于 OBQ（Optimal Brain Quantization），逐列量化并使用 Hessian 信息校正剩余权重的误差
- **AWQ**: 基于激活感知的缩放，通过缩小显著通道的量化范围来减少误差
- 两者可互补使用，不是替代关系
