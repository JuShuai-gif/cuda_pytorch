# 实验 3 常见错误

## 1. 搜索空间定义错误

### 错误：搜索空间过大导致无法搜索
**问题描述：** 定义的搜索空间过于庞大（如每个参数有几十种选择），导致穷举或搜索需要极长时间。
**正确做法：** 本实验使用简化的搜索空间，总共约 4 × 4 × 4 × 2 = 128 种组合。实际 NAS 论文中通常使用数十万到数百万的搜索空间，需要配合权重共享、代理模型等方法。

### 错误：未正确处理通道数递增
**问题描述：** 在构建 CNN 时，所有层使用相同的通道数，或者通道数在某层变化后未正确传递到下一层。
**正确做法：** 通常通道数随深度递增，且需要确保前一层输出通道数等于后一层输入通道数。
```python
in_channels = 3
for i in range(num_layers):
    out_channels = channels[i]
    self.convs.append(nn.Conv2d(in_channels, out_channels, ...))
    in_channels = out_channels  # 更新为下一层的输入通道数
```

---

## 2. CNNBuilder 构建错误

### 错误：特征图尺寸计算错误
**问题描述：** 未正确追踪每层后的特征图尺寸，导致全连接层输入维度计算错误。
**正确做法：** 逐层追踪特征图尺寸变化，考虑 padding、stride 和 pooling 的影响。
```python
size = image_size
for conv in self.convs:
    size = (size + 2 * conv.padding[0] - conv.kernel_size[0]) // conv.stride[0] + 1
    if pooling:
        size = size // 2
# 全连接层输入维度 = channels[-1] * size * size
```

### 错误：跳跃连接中维度不匹配
**问题描述：** 在实现残差连接时，跳跃连接的输入和输出维度不匹配导致相加失败。
**正确做法：** 使用 1x1 卷积进行维度匹配。
```python
def forward(self, x):
    residual = x
    out = self.conv(x)
    if out.shape != residual.shape:
        residual = self.skip_conv(residual)  # 1x1 conv 调整维度
    return F.relu(out + residual)
```

---

## 3. 进化搜索错误

### 错误：只保留最佳个体
**问题描述：** 每代只保留最佳个体，丢弃了所有其他个体，导致种群多样性丧失。
**正确做法：** 保留种群中的多个个体，通常种群大小至少 10-20，以维持基因多样性。
```python
# 错误做法
population = [best_child]  # 只保留最佳

# 正确做法：锦标赛选择 + 精英保留
def tournament_select(population, fitnesses, k=3):
    indices = random.sample(range(len(population)), k)
    best_idx = indices[np.argmax([fitnesses[i] for i in indices])]
    return population[best_idx]
```

### 错误：变异破坏了架构有效性
**问题描述：** 变异后通道数列表长度与层数不匹配。
**正确做法：** 在变异中维护一致性。
```python
def mutate(arch, ...):
    kernel_sizes = list(arch.kernel_sizes)
    channels = list(arch.channels)
    num_layers = arch.num_layers
    
    if random.random() < mutation_prob:
        idx = random.randint(0, num_layers - 1)
        kernel_sizes[idx] = random.choice(SEARCH_SPACE["kernel_sizes"])
    
    if random.random() < mutation_prob:
        idx = random.randint(0, num_layers - 1)
        channels[idx] = random.choice(SEARCH_SPACE["channels"])
    
    return Architecture(kernel_sizes, channels, num_layers, arch.use_skip)
```

---

## 4. 精度预测器错误

### 错误：输入特征编码不当
**问题描述：** 对架构的编码方式不能充分表达架构的特征差异。
**正确做法：** 编码应包含足够的信息来区分不同架构，建议使用统计特征（均值、最大值、最小值、标准差等）而非简单的整数编码。
```python
def encode_architecture(arch):
    features = [
        np.mean(arch.kernel_sizes),
        np.max(arch.kernel_sizes),
        np.min(arch.kernel_sizes),
        np.mean(arch.channels),
        np.max(arch.channels),
        np.min(arch.channels),
        arch.num_layers,
        int(arch.use_skip),
    ]
    return torch.tensor(features, dtype=torch.float32)
```

### 错误：训练/测试数据泄露
**问题描述：** 用训练精度预测器的数据来评估预测器的性能。
**正确做法：** 将收集的架构-精度对划分为训练集和测试集，或者使用交叉验证。

---

## 5. Pareto 前沿提取错误

### 错误：支配关系定义错误
**问题描述：** 混淆了"支配"和"非支配"的定义。
**正确做法：**
```python
def is_dominated(a, b, objectives):
    """
    判断 a 是否被 b 支配
    对于精度（越大越好）和 MACs（越小越好），需要分别处理方向
    """
    # accuracy: 越大越好 -> a 被 b 支配需要 b.acc >= a.acc
    # macs: 越小越好 -> a 被 b 支配需要 b.macs <= a.macs
    worse_or_equal = True
    strictly_worse = False
    
    # accuracy
    if b[0] < a[0]:
        worse_or_equal = False
    if b[0] > a[0]:
        strictly_worse = True
    
    # macs (取负，越小越好)
    if b[1] > a[1]:
        worse_or_equal = False
    if b[1] < a[1]:
        strictly_worse = True
    
    return worse_or_equal and strictly_worse
```

### 错误：Pareto 前沿点太少
**问题描述：** 只找到了 1-2 个 Pareto 点。
**正确做法：** Pareto 前沿的质量取决于搜索的充分程度。可以通过增加评估架构数量或使用更高效的搜索策略来获得更丰富的 Pareto 前沿。

---

## 6. 评估和度量错误

### 错误：MACs 计算忽略了全连接层
**问题描述：** 只计算了卷积层的 MACs。
**正确做法：** MACs 应包含卷积层和全连接层。
```python
# 全连接层 MACs = in_features * out_features
total_macs += in_features * out_features  # 乘法
total_macs += out_features  # 加法（通常与乘法合并为 MAC）
```
