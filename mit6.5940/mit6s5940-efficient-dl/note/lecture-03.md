# Lecture 03-04: 剪枝 (Pruning) - 把神经网络的"脂肪"减掉

## 1. 本讲核心问题

> 神经网络中大量权重接近于0，它们真的有用吗？能不能直接删掉？删了会怎样？删多少？怎么删？

## 2. 通俗解释

**生活类比 — 减肥 vs 截肢**:

- **非结构化剪枝**（fine-grained）: 就像抽脂——在全身各处的脂肪细胞中，挑出那些"小的"抽掉。效果很猛（能瘦很多），但身体结构变得稀疏不规则，做运动时反而更累（硬件不友好）。
- **结构化剪枝**（channel pruning）: 就像截掉一整条腿——虽然残忍但身体结构保持完整，行动依然协调（硬件加速效果好）。

这解释了为什么：同样减掉50%的"体重"，结构化剪枝实际加速效果远超非结构化剪枝。

## 3. 关键公式

### 3.1 剪枝的数学定义

$$\min_W \mathcal{L}(W) \quad \text{s.t.} \quad \|W\|_0 \le k$$

- $\|W\|_0$ = 非零权重的个数（L0范数）
- $k$ = 我们希望保留的非零参数量上限
- 我们想让 $\mathcal{L}$ 尽可能小，同时 $W$ 中大部分变成零

### 3.2 幅度剪枝 (Magnitude-based Pruning)

这是最基础、最常用的方法：

$$\text{重要性}(w_i) = |w_i|$$

把所有权重按绝对值排序，最小的那些 → 变成0。

**为什么有效？** 直觉上，如果 $|w_i| \approx 0$，那它乘以任何输入值也接近0，对最终输出贡献极小，删掉影响不大。

### 3.3 敏感度分析

某些层对剪枝特别敏感。我们需要逐层分析：

$$\text{Sensitivity}(l) = \frac{\text{Acc}_{\text{baseline}} - \text{Acc}_{\text{pruned}}(l, s)}{\text{Acc}_{\text{baseline}}}$$

- 剪枝率 $s$ （比如50%）
- 只剪第 $l$ 层，测量准确率下降
- 敏感度高的层 → 少剪一点；不敏感的层 → 多剪一点

### 3.4 结构化剪枝 — 通道剪枝

对卷积层，按通道的重要性排序：

$$\text{Importance}(c_i) = \|W_{:,i,:,:}\|_F = \sqrt{\sum W_{:,i,:,:}^2}$$

即计算每个输入通道的权重矩阵的 Frobenius 范数（L2范数），范数越大的通道 → 越重要。

### 3.5 剪枝后的计算量减少

通道剪枝移除30%通道 → 计算量减少约50%：

$$(1 - 0.3)^2 \approx 0.49 \implies \text{约减少50%}$$

因为卷积的计算量与输入通道数 × 输出通道数成正比。

## 4. 公式背后的直觉

### 为什么稀疏不一定快？

非结构化剪枝产生的是"瑞士奶酪"般的稀疏矩阵：

```
[0.5  0   0   0.3]     ← 零散分布在矩阵中
[0    0.7 0   0  ]
[0.2  0   0   0.8]
[0    0   0.9 0  ]
```

GPU 的 Tensor Core 喜欢**稠密矩阵**。稀疏矩阵需要特殊编码（如 CSR 格式），额外的索引计算和内存跳转反而拖慢速度。

NVIDIA A100 支持 **2:4 结构化稀疏**：每4个连续值中正好有2个是0。这是一种硬件和算法的折中。

### 为什么通道剪枝能真正加速？

```
通道剪枝前: Conv(C_in=64, C_out=128, K=3)
通道剪枝后: Conv(C_in=44, C_out=89, K=3)    ← 移除30%通道
```

卷积核变小了 → 计算量平方级减少 → GEMM 调用直接变快。不需要特殊的稀疏格式。

## 5. 工业界用途

| 应用场景 | 剪枝类型 | 典型效果 |
|----------|----------|----------|
| 手机端人脸检测 | 通道剪枝 | 2x加速, 0.3%精度损失 |
| 自动驾驶模型 | 结构化剪枝+N:M sparsity | 用TensorRT-A100, 2x加速 |
| 服务器推理 | 非结构化+2:4 sparse | 配合A100, 2x peak perf |
| MCU端关键词检测 | 非结构化(pruning重训练) | 模型缩小8x |
| LLM部署 | Wanda/SparseGPT | 50%稀疏, 几乎无损 |

## 6. PyTorch 实现思路

### 6.1 幅度剪枝（非结构化）

```python
import torch
import torch.nn as nn

def magnitude_prune(weight, sparsity):
    """
    对单个权重张量进行幅度剪枝
    sparsity: 目标稀疏度 (0~1), 0=不剪, 1=全剪
    """
    weight_abs = torch.abs(weight)
    # 找到阈值: 排序后第k个值
    k = int(sparsity * weight.numel())
    threshold = torch.kthvalue(weight_abs.view(-1), k).values
    
    # 生成mask: 绝对值大于阈值的保留
    mask = weight_abs > threshold
    return weight * mask, mask

# 使用
w = torch.randn(64, 64)
w_pruned, mask = magnitude_prune(w, sparsity=0.5)
print(f"稀疏度: {(w_pruned == 0).float().mean():.2%}")
```

### 6.2 通道剪枝（结构化）

```python
import torch.nn.utils.prune as prune

def channel_prune(model, prune_ratio=0.3):
    """
    对卷积层进行通道剪枝
    按权重的Frobenius范数选择不重要的通道
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # 计算每个输入通道的F范数
            importance = torch.norm(module.weight.data, p='fro', dim=(1,2,3))
            k = int(module.in_channels * (1 - prune_ratio))
            _, idx = torch.topk(importance, k)
            
            # 保留重要的通道
            module.weight.data = module.weight.data[idx]
```

### 6.3 完整剪枝—微调流水线

```python
def prune_and_finetune(model, train_loader, prune_ratios):
    """
    完整剪枝流程:
    1. 逐层敏感度分析
    2. 根据敏感度分配prune_ratio
    3. 剪枝
    4. 微调恢复准确率
    """
    # 1. 敏感度分析
    sensitivity = {}
    for layer_name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            for ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
                acc = evaluate_with_pruning(model, layer_name, ratio)
                sensitivity[(layer_name, ratio)] = acc
    
    # 2. 基于敏感度剪枝 (简化: 不敏感层多剪)
    # ...
    
    # 3. 微调
    for epoch in range(5):  # 一般5个epoch就够了
        for data, target in train_loader:
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            optimizer.zero_grad()
            loss.backward()
            # 重要！重新应用mask，让被剪掉的权重保持为0
            apply_pruning_mask(model)
            optimizer.step()
    
    return model
```

## 7. TinyML / Edge AI 部署意义

剪枝在 TinyML 中至关重要：
- MCU的Flash只有几百KB → 必须把模型降到Flash能容下
- 非结构化剪枝配合 **TinyEngine**（后续讲座）可以利用稀疏性
- 结构化剪枝适合有 SIMD 或 DSP 指令的 MCU

## 8. 常见误区

1. **"剪完就完事了"** — 剪枝后必须微调（fine-tune），否则准确率暴跌
2. **"所有层剪一样多"** — 不同层敏感度差10倍，用统一比例是浪费
3. **"稀疏度越高越好"** — 超过某个拐点，准确率会断崖式下跌
4. **"剪枝只减少计算量"** — 也减少内存占用和IO带宽需求
5. **"一次剪到位"** — 迭代式剪枝（每次剪一点→微调→再剪）效果更好

## 9. 面试问题

**Q1**: "非结构化剪枝和结构化剪枝的区别？什么时候用哪个？"

**A1**: 
- 非结构化：去单个权重，压缩率更高，但需要稀疏硬件支持才真正加速
- 结构化：去整个通道/卷积核，压缩率低些但能在任何硬件上直接加速
- 选择：部署到通用CPU/GPU → 结构化；有稀疏硬件支持(A100) → 非结构化或2:4

**Q2**: "如何确定每层的最优剪枝率？"

**A2**: 敏感度分析。逐层尝试不同剪枝率，绘制"剪枝率 vs 准确率"曲线，然后用启发式算法（如AMC的RL方法）或简单的搜索（给敏感层少剪，不敏感层多剪）分配。

## 10. 本讲总结

- **剪枝的目标**: 减少参数和计算量，同时保持准确率
- **核心矛盾**: 稀疏度 vs 硬件加速效果
- **关键洞察**: 不是所有参数都同等重要——有些层的参数删掉90%都没事，有些删10%就崩了
- **工业实践**: 结构化剪枝是工业界首选，因为硬件友好；非结构化剪枝用于极限压缩场景

剪枝回答了："哪些参数是冗余的？"下一讲量化回答："能不能用更少比特表示参数？"
