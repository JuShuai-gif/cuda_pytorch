# 实验 2 常见错误

## 1. 量化公式实现错误

### 错误：量化零点的计算方向错误
**问题描述：** 零点的计算公式理解错误。
**正确做法：**
```python
# 对称量化（推荐用于权重）
qmin = -(1 << (bits - 1))       # 例如 int8: -128
qmax = (1 << (bits - 1)) - 1    # 例如 int8: 127
scale = max(abs(tensor.min()), abs(tensor.max())) / qmax
zero_point = 0  # 对称量化零点为 0

# 量化
q = torch.clamp(torch.round(tensor / scale), qmin, qmax).to(torch.int32)
# 反量化
x_deq = q.float() * scale
```

### 错误：scale 为零导致除零
**问题描述：** 当张量的 min == max 时，scale 计算为 0。
**正确做法：**
```python
range_val = tensor.max() - tensor.min()
if range_val < 1e-8:
    scale = 1.0
else:
    scale = range_val / (qmax - qmin)
```

### 错误：忘记 clamp 操作
**问题描述：** 量化时未 clamp，导致量化值超出范围。
**正确做法：** 始终使用 `torch.clamp` 确保量化值在 [qmin, qmax] 内。

---

## 2. K-means 量化错误

### 错误：初始化策略不当
**问题描述：** 使用全零或全随机初始化聚类中心，导致收敛缓慢。
**正确做法：** 使用均匀采样初始化或 kmeans++ 初始化。
```python
# 均匀采样初始化
indices = torch.linspace(0, n - 1, k).long()
centroids = sorted_flat[indices].clone()

# 或者随机选择 k 个点
indices = torch.randperm(n)[:k]
centroids = flat[indices].clone()
```

### 错误：迭代次数不足
**问题描述：** 只迭代 1-2 次，聚类未收敛。
**正确做法：** 至少迭代 10-20 次，或直到聚类中心变化小于阈值。
```python
for it in range(num_iters):
    distances = torch.abs(flat.unsqueeze(1) - centroids.unsqueeze(0))
    assignments = distances.argmin(dim=1)
    
    new_centroids = torch.zeros_like(centroids)
    for i in range(k):
        mask = (assignments == i)
        if mask.sum() > 0:
            new_centroids[i] = flat[mask].mean()
    
    # 处理空聚类
    empty_clusters = (new_centroids == 0).all(dim=0) if new_centroids.dim() > 0 else ...
    
    shift = (new_centroids - centroids).abs().max()
    centroids = new_centroids
    if shift < 1e-6:
        break
```

---

## 3. 激活值校准错误

### 错误：只用一个批次校准
**问题描述：** 只用 1 个批次的数据进行激活值校准。
**正确做法：** 用多个批次（建议 100-1000 个样本）进行校准，以获得更准确的激活范围。

### 错误：使用训练数据进行校准
**问题描述：** 在训练集上校准激活值范围。
**正确做法：** 应该使用一个独立的小型校准集（通常从训练集中取一小部分，但不在其上训练），以避免过拟合校准数据。

### 错误：钩子函数实现不当
**问题描述：** 前向钩子中未正确收集统计信息。
**正确做法：**
```python
activation_stats = {}

def hook_fn(name):
    def fn(module, input, output):
        if name not in activation_stats:
            activation_stats[name] = {"min": float("inf"), "max": float("-inf")}
        activation_stats[name]["min"] = min(activation_stats[name]["min"], output.min().item())
        activation_stats[name]["max"] = max(activation_stats[name]["max"], output.max().item())
    return fn

# 注册钩子
hooks = []
for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU)):
        hooks.append(module.register_forward_hook(hook_fn(name)))
```

---

## 4. 量化推理模块错误

### 错误：量化权重时未 detach
**问题描述：** 量化操作影响了梯度计算图。
**正确做法：** 量化只在推理时使用，确保使用 `torch.no_grad()` 或 `.detach()`。
```python
with torch.no_grad():
    q_weight, w_scale, w_zp = linear_quantize(self.conv.weight.data, self.weight_bits)
    w_deq = linear_dequantize(q_weight, w_scale, w_zp)
```

### 错误：混淆了推理精度和训练精度
**问题描述：** 在量化推理模块中仍然启用梯度计算。
**正确做法：** 量化推理模块应在 `eval()` 模式下运行，不参与训练。

---

## 5. 概念性错误

### 错误：认为量化后直接减小了模型大小
**问题描述：** 用 FP32 张量存储量化值，然后声称模型被压缩了。
**正确做法：** 真正的模型压缩需要：
1. 用 `torch.int8` 存储量化权重
2. 在实际部署中使用整数运算
3. 代码中的量化模拟只是精度验证，实际压缩需要专门的序列化格式

### 错误：忽略 BatchNorm 的融合
**问题描述：** 量化时未考虑 Conv-BN 融合。
**正确做法：** 在量化前应该将 Conv2d 和 BatchNorm2d 融合（fold），因为量化部署中通常只有一个量化卷积层。
```python
def fuse_conv_bn(conv, bn):
    fused = nn.Conv2d(conv.in_channels, conv.out_channels,
                       conv.kernel_size, conv.stride, conv.padding)
    # 融合权重和偏置
    w_conv = conv.weight.data
    bn_std = (bn.running_var + bn.eps).sqrt()
    fused.weight.data = w_conv * (bn.weight / bn_std).view(-1, 1, 1, 1)
    fused.bias.data = (conv.bias - bn.running_mean) * (bn.weight / bn_std) + bn.bias
    return fused
```
