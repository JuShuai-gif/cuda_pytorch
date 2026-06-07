# Lecture 04: 剪枝进阶 - 稀疏训练、自动剪枝率与系统支持

## 1. 本讲核心问题

> 如何自动决定每层剪多少？剪完后如何让硬件真正加速？稀疏矩阵在GPU上到底怎么跑？

## 2. 通俗解释

**生活类比 — 修剪花园**：

上一讲教了你怎么"剪"，这一讲教你怎么"决定剪多少"。

想象你是园艺师，面前有一排不同品种的树：
- 松树（前面几层）长得茂盛，需要轻剪
- 灌木（中间层）随便剪，春天自己会长回来
- 盆景（最后几层）剪错一根枝就毁了

你不能对所有树用同一把剪刀。你需要：
1. 先轻轻试剪（敏感度分析）
2. 确定每棵树的容忍度
3. 再正式下剪

这就是**自动确定剪枝率**的核心。

## 3. 关键公式

### 3.1 迭代剪枝 (Iterative Pruning)

与其一刀切，不如反复"剪一点→训练→再剪一点"：

$$W_t = \text{Prune}(W_{t-1}, s_t)$$

$$W_t = \text{Finetune}(W_t)$$

每轮剪枝率递增：$s_1 < s_2 < ... < s_T = S_{target}$

**好处**: 给网络时间"适应"损失，逐步将重要信息迁移到剩余的权重中。

### 3.2 2:4 结构化稀疏

NVIDIA Ampere 架构支持的稀疏模式：

$$\text{每4个连续元素中，恰好2个为0}$$

```
[0.5  0   0.3  0  ]    ← 满足2:4模式
[0    0.7 0    0.8]    ← 满足2:4模式
```

配合 Tensor Core → **2倍峰值吞吐**。

### 3.3 稀疏矩阵存储：CSR 格式

一个 $4 \times 4$ 稀疏矩阵：
```
[1  0  0  2]
[0  3  0  0]
[0  0  4  0]
[5  0  6  0]
```

CSR 存储（三数组）：
- `values = [1, 2, 3, 4, 5, 6]` — 非零值
- `col_idx = [0, 3, 1, 2, 0, 2]` — 列索引
- `row_ptr = [0, 2, 3, 4, 6]` — 每行起始位置

存储节省：$16 \times 4 = 64$ bytes → $6 \times 4 + 6 \times 4 + 5 \times 4 = 68$ bytes
**当稀疏度 > 50% 时才开始真正节省存储！**

### 3.4 训练时剪枝：Gradual Magnitude Pruning

从稠密网络开始，逐步增加稀疏度：

$$s_t = s_f + (s_i - s_f) \cdot \left(1 - \frac{t}{T}\right)^3$$

- $s_f$: 起始稀疏度（如 0%）
- $s_f$: 最终稀疏度（如 90%）
- $T$: 总训练步数
- 立方衰减让前期变化快，后期精细

## 4. 公式背后的直觉

**为什么 2:4 稀疏性正好？**

这是 NVIDIA 工程师精心挑选的"甜点"：
- 2:4 意味着 50% 稀疏度
- 刚好可以用 2 个 Tensor Core 周期完成（正常稠密需要 4 个周期）
- 硬件实现简单：只需在 fetch 数据时跳过零值
- 同时，50% 稀疏度对准确率损失很小

**为什么迭代剪枝比一次性剪枝好？**

想象让你立刻负重减半 → 你可能直接趴下。但如果每周减一点 → 你的肌肉会适应。神经网络同理：逐步剪枝让剩余权重有"学习补偿"的机会。

## 5. 工业界用途

| 技术 | 适用场景 | 硬件要求 |
|------|----------|----------|
| 2:4 Structured Sparsity | 云端推理 (A100/H100) | A100/H100 GPU |
| Channel Pruning | 手机/边缘设备 | 任何硬件 |
| Block Sparsity | FPGA/ASIC | 定制加速器 |
| Sparse Training | 减少训练成本 | 需要稀疏训练框架 |
| Lottery Ticket Hypothesis | 寻找最优子网络 | 研究用途，工业较少 |

### 实际案例

- **NVIDIA TensorRT**: 支持 2:4 稀疏模型的自动优化和部署
- **Apple CoreML**: 支持权重剪枝后模型的稀疏存储
- **Google TensorFlow Lite**: 提供 training-time pruning API

## 6. PyTorch 实现思路

### 6.1 训练时逐步剪枝

```python
import torch
import torch.nn as nn
from torch.nn.utils import prune

def gradual_pruning_schedule(current_step, total_steps, 
                              initial_sparsity=0.0, final_sparsity=0.9):
    """计算当前步骤的目标稀疏度（立方衰减）"""
    t = current_step / total_steps
    return final_sparsity + (initial_sparsity - final_sparsity) * (1 - t) ** 3

class GradualPruner:
    def __init__(self, model, initial_sparsity=0.0, final_sparsity=0.9,
                 total_steps=10000):
        self.model = model
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.total_steps = total_steps
        self.current_step = 0
        
        # 初始化所有可剪枝参数的mask
        self.prune_params = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prune.identity(module, 'weight')
                self.prune_params.append((module, 'weight'))
    
    def step(self):
        """每次训练步调用, 更新稀疏度并重新生成mask"""
        target_sparsity = gradual_pruning_schedule(
            self.current_step, self.total_steps,
            self.initial_sparsity, self.final_sparsity
        )
        
        for module, name in self.prune_params:
            prune.l1_unstructured(module, name=name, amount=target_sparsity)
        
        self.current_step += 1
    
    def remove_masks(self):
        """训练完成后固化稀疏结构"""
        for module, name in self.prune_params:
            prune.remove(module, name)
```

### 6.2 稀疏矩阵乘法性能测试

```python
import torch
import time

def benchmark_sparse_matmul(size=1024, sparsity=0.9):
    """对比稠密和稀疏矩阵乘法性能"""
    # 生成稀疏矩阵
    dense = torch.randn(size, size)
    mask = torch.rand(size, size) > sparsity
    sparse_dense = dense * mask  # 实际仍是稠密存储
    sparse_csr = sparse_dense.to_sparse_csr()
    
    vec = torch.randn(size)
    
    # 稠密矩阵乘法
    t0 = time.perf_counter()
    for _ in range(100):
        _ = dense @ vec
    t1 = time.perf_counter()
    
    # CSR稀疏矩阵乘法
    t2 = time.perf_counter()
    for _ in range(100):
        _ = sparse_csr @ vec
    t3 = time.perf_counter()
    
    print(f"Dense: {(t1-t0)/100*1000:.3f}ms | "
          f"CSR: {(t3-t2)/100*1000:.3f}ms")
    print(f"Sparsity: {sparsity:.0%}")

benchmark_sparse_matmul(sparsity=0.5)
benchmark_sparse_matmul(sparsity=0.9)
benchmark_sparse_matmul(sparsity=0.99)
```

## 7. TinyML / Edge AI 部署意义

- **MCU 场景**: 非结构化剪枝+微调 → 模型大小可以压缩 5-10x → 才能装进几百KB的Flash
- **TinyEngine**: 专门为稀疏矩阵设计的推理引擎，有效利用MCU上剪枝后的稀疏性
- **内存墙**: 在MCU上，内存访问功耗远超计算功耗。剪枝减少权重读取 → 直接省电

## 8. 常见误区

1. **"剪枝后权重就永久为0"** — 不，你需要训练过程中持续遮罩（re-masking），否则梯度更新会"复活"被剪掉的权重
2. **"剪枝只能在训练后做"** — 训练中逐步剪枝（gradual pruning）通常比训练后剪枝效果好
3. **"彩票假说(Lottery Ticket Hypothesis)找到了最优子网络"** — LTH 需要从头训练子网络，成本极高，工业界更常用的是 iterative magnitude pruning
4. **"稀疏度超过90%模型就不可能准"** — Lottery Ticket Hypothesis 证明了在某些初始化下，极稀疏子网络可以匹配原网络准确率

## 9. 面试问题

**Q1**: "Lottery Ticket Hypothesis 是什么？为什么它很重要？"

**A1**: LTH 指出：在稠密网络中，存在一个子网络（winning ticket），从头训练这个子网络可以达到或超过原网络的准确率。重要在于它揭示了"大网络中存在本质更小但同样强大的子结构"，这为剪枝提供了理论基础。

**Q2**: "用PyTorch的torch.nn.utils.prune做剪枝时，remove()函数做了什么？"

**A2**: `remove()` 将剪枝后的mask与权重永久合并（weight = weight * mask），然后删除mask相关的钩子。之后该参数被视为普通参数，不再有剪枝相关的额外状态。

## 10. 本讲总结

剪枝进阶的核心收获：
- **自动剪枝率**: 敏感度分析 → 逐层差异化剪枝
- **迭代剪枝**: 反复"剪→训练→剪"比一次到位效果好
- **硬件支持**: 2:4 稀疏是算法-硬件协同设计的典范
- **稀疏存储**: CSR格式分析 → 稀疏度>50%才真正省存储

剪枝是模型压缩的"减法"。下一讲我们做"除法"：用量化降低每个权重占用的比特数。
