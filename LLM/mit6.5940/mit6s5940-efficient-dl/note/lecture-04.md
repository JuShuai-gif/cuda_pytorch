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

### 大厂剪枝进阶实战

- **NVIDIA 官方 2:4 稀疏 Benchmark**: 在 A100 上，ResNet50 (2:4 sparse) + INT8 = 对比 FP32 dense，推理吞吐提升 5.3x。关键发现：必须同时使用 2:4 sparse + INT8 才能达到最佳效果，单独用 2:4 sparse 只加速 1.5x（因为 memory-bound），单独用 INT8 只加速 2.1x。两者叠加产生了超线性的加速效果，因为 2:4 稀疏减少了 INT8 Tensor Core 的 GEMM 输入量。

- **OpenAI Triton Inference Server 动态批处理 + 剪枝**: 将 BERT-Large 剪枝至 50% 稀疏度，配合 dynamic batching（攒 16 个请求一起推理），GPU 利用率从 23% 提升到 78%。但如果 batch 攒得太小（batch<8），2:4 稀疏几乎无加速 — 因为小 batch 下 kernel launch overhead 和 I/O 绑定主导了延迟。

- **华为 MindSpore Golden Stick 自动剪枝**: 手机端 Super-Resolution 模型，自动搜索出的剪枝率为：前 2 层 12%，中间 12 层 45-65%，最后 2 层 8%。人工设定的 uniform 50% 剪枝导致 PSNR 下降 1.2dB，而自动搜索的方案只降 0.2dB。区别在于自动搜索发现了"模型的首尾两层对剪枝极敏感"这一规律。

### 稀疏训练的产业现实

- **稀疏训练的工程挑战**: Facebook's GLT (2021) 在 512 张 V100 上做 sparse training，理论稀疏 90% 但实际训练吞吐只有 dense 训练的 60%。原因：(1) 稀疏 mask 的动态更新(prune-regrow)引入了大量 host-device 同步；(2) 稀疏梯度通信在 all-reduce 时需要 padding 和非零值压缩，NVLink 带宽利用不充分。
- **现实**: 2024 年工业界主流的稀疏训练仍停留在 **静态稀疏模式**（如 2:4 固定 pattern），动态稀疏（训练过程中改变 mask）虽然学术界效果好，但工程落地困难，训练速度太慢。

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

### 生产环境稀疏推理的正确姿势

```python
import torch
import time

def production_sparse_inference_setup(dense_weight: torch.Tensor, 
                                        sparsity: float = 0.5):
    """Production-ready sparse weight preparation.
    
    Key insight: CSR format only saves memory at sparsity > ~50%.
    Below 50%, the index overhead (col_idx + row_ptr) exceeds the savings.
    
    At sparsity=50% (2:4), CSR requires 2× the memory of the dense tensor:
      - dense: N × 4 bytes (FP32)
      - CSR: 0.5N × 4 (values) + 0.5N × 4 (col_idx) + (N/block) × 4 (row_ptr)
           ≈ 4N bytes = SAME as dense!
    
    Moral: Dense + 2:4 hardware support beats CSR at 50% sparsity.
    """
    original_size = dense_weight.numel() * dense_weight.element_size()
    
    # Generate 2:4 sparse mask (NVIDIA Ampere pattern)
    dense_reshaped = dense_weight.view(-1, 4)
    # In each group of 4, find the 2 smallest by magnitude
    sorted_mag, _ = dense_reshaped.abs().sort(dim=1)
    threshold = sorted_mag[:, 1:2]  # Keep top 2, prune bottom 2
    mask_2_4 = (dense_reshaped.abs() >= threshold).float()
    sparse_weight = (dense_reshaped * mask_2_4).view_as(dense_weight)
    
    # Store as CSR for portability (but TensorRT uses native 2:4)
    sparse_csr = sparse_weight.to_sparse_csr()
    
    csr_size = (
        sparse_csr.values().numel() * sparse_csr.values().element_size() +
        sparse_csr.col_indices().numel() * sparse_csr.col_indices().element_size() +
        sparse_csr.crow_indices().numel() * sparse_csr.crow_indices().element_size()
    )
    
    return {
        'dense_size_mb': original_size / (1024**2),
        'csr_size_mb': csr_size / (1024**2),
        'sparsity': (sparse_weight == 0).float().mean().item(),
        'break_even': csr_size < original_size,
    }

# Key lesson: Benchmark on TARGET hardware with realistic batch sizes
# A CSR matmul that's 3× faster than dense on paper may be 0.5× slower
# on a real GPU with batch=1 due to irregular memory access patterns
```

## 7. TinyML / Edge AI 部署意义

- **MCU 场景**: 非结构化剪枝+微调 → 模型大小可以压缩 5-10x → 才能装进几百KB的Flash
- **TinyEngine**: 专门为稀疏矩阵设计的推理引擎，有效利用MCU上剪枝后的稀疏性
- **内存墙**: 在MCU上，内存访问功耗远超计算功耗。剪枝减少权重读取 → 直接省电

### 硬件级稀疏支持与应用约束

- **ARM Cortex-M7 SIMD 指令**: `SMLAD` (Signed Multiply Accumulate Long Dual) 可以一次处理 2 个 16-bit 乘法累加，但前提是输入数据必须是稠密连续排列的。非结构化剪枝产生的稀疏权重破坏了这种连续性 → SIMD 利用率从 95% 降到 30%。**结论**: 在 ARM Cortex-M 上部署剪枝模型，结构化剪枝比非结构化剪枝实际快 2-3x，即使后者 FLOPs 更少。

- **STM32H743 (1MB SRAM, 480MHz)**: 图像分类模型通过 gradual magnitude pruning 达到 80% 稀疏度。模型存储从 512KB 降到 102KB + CSR 索引 71KB = 173KB（节省 66%）。但推理延迟只因理论 FLOPs 从 56M 降到 11M（降 80%）实际加速 1.8x，原因：(1) CSR 解码的循环控制流导致分支预测失败率 27%（Cortex-M7 的分支预测较弱）；(2) 随机内存访问模式使数据 cache miss rate 从 2% 升到 18%。

- **GAP8 (GreenWaves, 8核 RISC-V + 硬件卷积加速器)**: 原生支持 1-bit（二值）和 2-bit（三值）权重的硬件加速。在 50MHz、功耗仅 35mW 下，二值网络的前向推理可达到 15 GOPS 的有效吞吐。这使得在纽扣电池供电设备上连续运行 24小时的 sound event detection 成为可能（功耗预算仅 2mAh/day）。

## 8. 常见误区

1. **"剪枝后权重就永久为0"** — 不，你需要训练过程中持续遮罩（re-masking），否则梯度更新会"复活"被剪掉的权重
2. **"剪枝只能在训练后做"** — 训练中逐步剪枝（gradual pruning）通常比训练后剪枝效果好
3. **"彩票假说(Lottery Ticket Hypothesis)找到了最优子网络"** — LTH 需要从头训练子网络，成本极高，工业界更常用的是 iterative magnitude pruning
4. **"稀疏度超过90%模型就不可能准"** — Lottery Ticket Hypothesis 证明了在某些初始化下，极稀疏子网络可以匹配原网络准确率

### 生产环境剪枝进阶事故

5. **"Gradual pruning schedule 设错了 → 前期剪太猛, 模型崩了回不来"** — 立方衰减的 schedule 中，前 20% 的训练步应该只达到最终稀疏度的约 6%。如果在第 1000 步就剪了 50% → 剩余权重的梯度尚未收敛 → 后续微调无法恢复精度 → 整个 pruning run 废掉。**实战经验**: 从头训练 + 渐近剪枝一条龙的流程中，在前 30% 的步数几乎不做剪枝，让网络先建立稳定的特征表示。

6. **"2:4 sparse 推理用了通用 CSR 而不是 TensorRT 原生 sparse kernel"** — TensorRT 对 2:4 稀疏有其专用的 kernel 实现，直接利用了 Ampere SM 的硬件指令。如果你把 2:4 稀疏模型存成 ONNX（ONNX 不原生支持 2:4 sparse 格式），再通过 ONNX Runtime 跑 → 不会触发硬件加速 → 推理速度和 dense 没区别。**正确做法**: 必须用 TensorRT 的 `trtexec --sparsity=structured` 从头 build engine。

7. **"用剪枝的 ResNet50 做特征提取器的下游任务全崩了"** — 预训练模型剪枝后，通常只微调了 ImageNet 分类头。但如果你下游任务是目标检测（Faster R-CNN）或分割（FCN），剪枝后的 backbone 提取的特征图发生了系统性的偏移 — 某些通道缺失导致 FPN 的特征金字塔不再完备。**解决方案**: 剪枝后必须在下游任务的数据集上至少做 2-3 个 epoch 的端到端微调，不能直接拿 ImageNet 微调好的剪枝模型即插即用。

## 9. 面试问题

**Q1**: "Lottery Ticket Hypothesis 是什么？为什么它很重要？"

**A1**: LTH 指出：在稠密网络中，存在一个子网络（winning ticket），从头训练这个子网络可以达到或超过原网络的准确率。重要在于它揭示了"大网络中存在本质更小但同样强大的子结构"，这为剪枝提供了理论基础。

**Q2**: "用PyTorch的torch.nn.utils.prune做剪枝时，remove()函数做了什么？"

**A2**: `remove()` 将剪枝后的mask与权重永久合并（weight = weight * mask），然后删除mask相关的钩子。之后该参数被视为普通参数，不再有剪枝相关的额外状态。

**Q3 (NVIDIA 面试真题)**: "请描述 A100 的 2:4 结构化稀疏在 Tensor Core 中的具体硬件执行流程。为什么必须恰好是 2:4 而不是 1:4 或 3:4？"

**参考答案**: 

在 A100 的 SM 中，2:4 稀疏通过以下流程执行：
1. 每个 warp scheduler 发射一条稀疏 MMA 指令
2. 输入矩阵 A 和 B 分别以 4 元素为组，每组附带 2-bit metadata（指示哪 2 个元素是非零值）
3. Tensor Core 在读取这 4 个元素时，根据 metadata 只取 2 个非零值，送入乘法阵列
4. 因此实际送入 16×16×16 Tensor Core 的是 16×8 的有效数据（而非 16×16），只需要 2 个周期而不是 4 个

为什么是恰好 2:4？

- **1:4 (75% 稀疏)**: 虽然理论上能压缩更多，但硬件 metadata 仍是 2-bit（4 选 1）。Tensor Core 的有效输入只有 1/4，很难维持乘法阵列的利用率 → 功耗/面积效率低。而且 75% 稀疏度下，训练精度损失难以接受。
- **3:4 (25% 稀疏)**: Tensor Core 仍需要接近 4 个周期的全流程，加速空间小。metadata 2-bit 描述了"哪 3 个非零"，但硬件实现复杂度大增（需要支持跳过 1 个元素的能力，但不如 2:4 统一）。

**2:4 是"甜点"**: 50% 稀疏度恰好让 Tensor Core 周期减半，metadata 设计简洁（4 选 2 = 6 种组合，用 3-bit 编码中的 6 个有效值），精度损失在可接受范围（CNN < 0.5%，Transformer < 1%）。

**Q4 (字节跳动面试真题)**: "你负责训练一个稀疏度 90% 的 BERT-Large 用于搜索排序。训练到第 80 万步时 loss 突然从 0.8 跳到 2.3，之后再也不收敛。逐层排查后发现 Attention 层的 QKV projection 中，query 的稀疏 pattern 和 key 的稀疏 pattern 产生了"正交效应" — 被剪掉的 query 维度恰好对应被保留的 key 维度。从数学上解释为什么会发生，以及如何防止。"

**参考答案**: 

这个问题本质上是稀疏子空间的对齐问题。Attention 的计算是 `Q @ K^T`。如果 Q 的第 i 个输出维度被剪掉（全为 0），而 K 的第 j 个输入维度被保留（非零），那 `Q_i @ K_j` 这一项不受影响（因为 Q_i=0 所以此项为 0）。但如果反过来 — Q 中被保留的维度对应 K 中被剪掉的维度 → 这些维度的贡献完全消失，相当于 attention score 的有效维度坍缩。

**数学上**: 回忆 Q 和 K 都是 `d_model × d_head` 的权重矩阵。剪枝后的有效维度 = `(非零行数_Q) ∩ (非零列数_K)`。如果两者互不相交 → attention score 变成常数（全零），softmax 变成均匀分布 → 每个 token 平等地 attend 所有 token → 信息完全丢失 → loss 爆炸。

**防止方法**: 
1. **Group Lasso 正则化**: 在剪枝训练中对 Q 的 col 和 K 的 row 加 group sparsity 约束 → 鼓励 Q 和 K 的相同维度被同时剪掉或同时保留
2. **Coupled pruning**: 不在 Q 和 K 上独立做 magnitude pruning，而是用 `||W_Q[:,i]|| × ||W_K[i,:]||` 作为联合重要性指标
3. **Dimensionality budget**: 设定 `min_active_dims = d_head // 4`，保证至少 25% 的注意力维度保留非零对

**Q5 (快手面试真题)**: "你用 iterative magnitude pruning 训练了一个剪枝 MobileNetV3 用于端上推理。你发现每轮剪枝后微调 3 个 epoch，第 7 轮微调的精度恢复（accuracy recovery）只有第 1 轮的 1/3。这说明什么？怎么改进？"

**参考答案**: 

这说明了**剪枝饱和效应 (pruning saturation)**。在早期轮次，被剪掉的权重基本都是真正的"脂肪"（magnitude 接近 0），剩余权重通过重新分配可以轻松补偿。但后期轮次的剪枝开始切到"肌肉"（magnitude 中等但有功能的权重），剩余的可学习容量不足，微调无法完全补偿。

**改进方案**: 
1. **学习率 warm restart**: 每轮微调不是从 0 开始降低 lr，而是用 cosine annealing with warm restart（每轮先升后降）→ 帮助剩余权重跳出局部最优
2. **Knowledge distillation**: 在微调 loss 中加入原始模型的 soft label（KD loss），让剪枝模型不仅学 ground truth，还要学会模仿原始模型的输出分布 → 给剩余权重更丰富的梯度信号
3. **Regrowth (Dynamic Sparse Training)**: 不是一次剪掉后永久保留 mask，而是允许一部分被剪的权重在微调中"复活"（regrowth），同时剪掉另一些。SET (Sparse Evolutionary Training) 和 RigL 就是这个思路 — 每 N 步重新评估所有权重的重要性并更新 mask
4. **检查是否该停了**: 如果第 7 轮的 accuracy recovery < 0.1%，可能剪枝率已经接近该架构的理论上限 → 需要换更激进的架构（如从 MobileNetV3 换 EfficientNet）

## 10. 本讲总结

剪枝进阶的核心收获：
- **自动剪枝率**: 敏感度分析 → 逐层差异化剪枝
- **迭代剪枝**: 反复"剪→训练→剪"比一次到位效果好
- **硬件支持**: 2:4 稀疏是算法-硬件协同设计的典范
- **稀疏存储**: CSR格式分析 → 稀疏度>50%才真正省存储

剪枝是模型压缩的"减法"。下一讲我们做"除法"：用量化降低每个权重占用的比特数。

## 11. 工业落地 Checklist

| 检查项 | 说明 | 不做的后果 |
|--------|------|-----------|
| Gradual schedule 验证 | 前 30% 训练步数稀疏度 < 最终值的 20%，避免前期剪太猛 | 模型崩了回不来，整个 pruning run 废掉 |
| 剪枝后下游任务微调 | 剪枝的 backbone 接到检测/分割等下游任务必须先做 2-3 epoch 端到端微调 | 特征图偏移导致下游任务精度暴跌 10-20% |
| 2:4 sparse 用 TensorRT 原生 kernel | 必须用 `trtexec --sparsity=structured` build engine，不能用 ONNX Runtime | 稀疏模型跑出和 dense 一样的速度 |
| 剪枝率上限探测 | 对每层逐步增加剪枝率直到精度断崖，该点即为该层的剪枝上限 | 超过上限的剪枝不可逆地破坏模型 |
| 动态稀疏的工程可行性 | 训练中的 dynamic regrowth 虽然精度好，但训练吞吐可能只有 dense 的 30-50% | 投入 3x 训练成本换 2% 精度提升→性价比低 |
| 稀疏度 vs CSR 存储的实际收益 | 稀疏度 < 50% 时 CSR 存储可能比 dense 还大，计算 index overhead | 以为省了存储，实际更占地方 |
| Q/K 剪枝对齐 | Attention 的 Q 和 K 必须做 coupled pruning，防止稀疏子空间正交 | Attention 变成均匀分布 → loss 爆炸 |

## 12. 学习闭环补充：结构化剪枝必须改模型结构

### 12.1 工业核心

结构化剪枝的目标是改变 dense operator shape，让标准 kernel 直接变小。例如 Conv2d 从 `C_out=128` 变成 `C_out=80`，后续层的 `C_in` 也要同步变化。只把整个 channel 权重置零，仍然不会减少 dense Conv 的计算。

### 12.2 Dependency Graph

真实网络中通道不是孤立的：

| 结构 | 依赖 |
|---|---|
| Conv -> BN -> ReLU | BN 的 gamma/beta/running stats 也要裁剪 |
| Conv -> Conv | 下一层 input channel 要同步裁剪 |
| Residual Add | 两个分支 channel 必须对齐 |
| Concat | 后续层 channel index 需要重映射 |
| Group/Depthwise Conv | group 数和 channel 数有整除约束 |

工业工具通常需要 dependency graph，例如 Torch-Pruning、NNI、TensorRT Model Optimizer 或自定义 FX graph pass。

### 12.3 对应代码实验

```bash
python src/lecture-04/main.py
```

运行后不要只看参数量，还要检查：

- 剪枝后模型的 Conv/Linear shape 是否真的改变？
- BN 参数是否同步裁剪？
- latency 是否比原模型下降？
- ONNX 导出是否仍然成功？

### 12.4 本讲验收问题

1. 结构化剪枝和非结构化剪枝的上线收益为什么不同？
2. 为什么 residual connection 让 channel pruning 变复杂？
3. BN gamma pruning 的直觉是什么？
4. 剪枝后为什么必须 finetune 或 distill？
5. 结构化剪枝如何和 INT8 量化组合？

## 13. Python 代码补充：通道重要性排序

结构化剪枝的第一步通常是给每个输出通道打分。下面代码用 L2/Frobenius norm 计算 Conv2d 输出通道重要性。

```python
import torch
import torch.nn as nn

@torch.no_grad()
def conv_out_channel_importance(conv: nn.Conv2d):
    # weight shape: [out_channels, in_channels, kh, kw]
    return conv.weight.detach().flatten(1).norm(p=2, dim=1)

@torch.no_grad()
def select_channels_to_keep(conv: nn.Conv2d, keep_ratio: float):
    score = conv_out_channel_importance(conv)
    keep = max(1, int(score.numel() * keep_ratio))
    return torch.topk(score, keep).indices.sort().values

conv = nn.Conv2d(16, 32, 3, padding=1, bias=False)
keep_idx = select_channels_to_keep(conv, keep_ratio=0.7)
print("keep channels", keep_idx.tolist())
```

注意：工业级结构化剪枝不只是选择 `keep_idx`，还必须同步裁剪 BN、下一层输入通道、残差分支和 concat 依赖。单独把未保留通道置零不等于真正结构化剪枝。

