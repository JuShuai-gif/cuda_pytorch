# 实验 1 常见错误

## 1. 剪枝实现错误

### 错误：对偏置（bias）进行了剪枝
**问题描述：** 学生错误地对 `bias` 参数也进行了剪枝。
**正确做法：** 只对 `weight` 参数进行剪枝。偏置参数量很少，剪枝偏置几乎没有收益，反而可能严重影响模型性能。
```python
# 错误做法
for name, param in model.named_parameters():
    if 'weight' in name or 'bias' in name:  # 错误：包含了 bias
        magnitude_prune(param, sparsity)

# 正确做法
for module in model.modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        magnitude_prune(module.weight.data, sparsity)
```

### 错误：未正确计算阈值
**问题描述：** 使用 `sort` 或 `topk` 时维度处理错误。
**正确做法：** 将权重展平后再计算阈值。
```python
# 正确做法
flat_weight = weight.abs().flatten()
k = int(sparsity * flat_weight.numel())
threshold = torch.kthvalue(flat_weight, k).values
mask = (weight.abs() >= threshold).float()
return weight * mask
```

### 错误：将剪枝后的权重当作已删除
**问题描述：** 认为将权重置零就减少了参数量。
**正确做法：** 非结构化剪枝只是将权重置零，总参数量不变。但可以通过稀疏矩阵存储格式（如 CSR）在实际中减少存储和计算量。

---

## 2. 敏感性扫描错误

### 错误：修改了原模型但未恢复
**问题描述：** 在敏感性扫描中修改了模型权重，但下一轮测试时没有恢复。
**正确做法：** 每次测试前保存原始权重副本，测试后恢复。
```python
# 正确做法
original_weight = module.weight.data.clone()
# ... 执行剪枝和评估 ...
module.weight.data.copy_(original_weight)
```

### 错误：对所有层同时剪枝
**问题描述：** 同时对所有层进行相同比例的剪枝来测量敏感性。
**正确做法：** 敏感性扫描应该逐层独立进行。每次只剪枝一层，保持其他层不变，这样才能准确测量每层的敏感性。

---

## 3. 微调错误

### 错误：微调时未固定剪枝掩码
**问题描述：** 在微调过程中，被剪枝的权重又被更新了。
**正确做法：** 微调时需要保持剪枝掩码，确保被置零的权重始终保持为零。
```python
# 在每次 optimizer.step() 之后
for module in model.modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        module.weight.data *= mask  # 重新应用掩码
```

### 错误：学习率设置不当
**问题描述：** 使用与训练时相同的学习率进行微调。
**正确做法：** 微调时应使用较小的学习率（通常是原学习率的 1/10 或 1/100），避免破坏已学到的特征。

---

## 4. 评估错误

### 错误：在训练模式下评估
**问题描述：** 忘记调用 `model.eval()`，导致 BatchNorm 等层行为异常。
**正确做法：** 所有评估操作前调用 `model.eval()`，评估后恢复 `model.train()`。

### 错误：延迟测量不准确
**问题描述：** 单次推理测量延迟，或未进行 GPU 同步。
**正确做法：** 多次测量取平均，且 GPU 上需要调用 `torch.cuda.synchronize()`。
```python
# 正确做法
if torch.cuda.is_available():
    torch.cuda.synchronize()
start = time.time()
# ... 推理 ...
if torch.cuda.is_available():
    torch.cuda.synchronize()
end = time.time()
```

---

## 5. 概念性错误

### 错误：混淆稀疏度含义
**问题描述：** 混淆了"保留比例"和"剪枝比例"。
**正确做法：**
- **稀疏度 (Sparsity)** = 零值权重占比 = 被剪枝的权重 / 总权重
- **密度 (Density)** = 非零权重占比 = 1 - 稀疏度

### 错误：期望高稀疏度仍有高精度
**问题描述：** 认为 90% 以上稀疏度也能保持接近原始的精度。
**正确做法：** 非结构化剪枝在 50%~70% 稀疏度时通常能保持较好精度，超过 90% 精度会显著下降。这一限制也是结构化剪枝和知识蒸馏等方法的研究动机。
