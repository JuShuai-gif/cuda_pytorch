# Learning both Weights and Connections for Efficient Neural Networks

> Song Han et al., NIPS 2015

## 1. 论文解决什么问题

深度神经网络（如 AlexNet、VGGNet）虽然精度高，但参数量和计算量巨大。例如 AlexNet 包含约 6100 万个参数，ConvNet 推理消耗数亿次浮点运算。在移动端、嵌入式设备和数据中心低延迟场景下，需要大幅降低模型的存储和计算开销——但直接设计小模型（如 SqueezeNet）往往以牺牲精度为代价。本文提出了**通过剪枝消除冗余连接**的方法，在不损失精度的前提下将模型压缩数倍至数十倍。

核心问题是：**网络中的哪些连接是冗余的？如何在不破坏网络表达能力的前提下安全移除它们？**

## 2. 核心方法

论文提出了经典的三步剪枝法（Three-Step Pruning Pipeline），这是该领域的奠基性工作：

**Step 1: 正常训练（Train Connectivity）**
首先以常规方式训练一个大网络直到收敛，建立网络初始的连接模式（connectivity）。这一步确定"哪些连接在起作用"。

**Step 2: 剪枝（Prune Small Weights）**
对所有层的权重取绝对值排序，设定一个百分比阈值 $p$（例如 90%），将绝对值最小的 $p \times 100\%$ 的权重置零并从网络中移除。直觉是：绝对值越小的权重对输出的贡献越小，移除它们对精度的影响也最小。

这一步得到的是稀疏权重矩阵，原始密集连接变为不规则稀疏模式（irregular sparsity）。关键操作是将这些被移除的连接对应的底层参数永久删除，而非简单 mask。

**Step 3: 微调/重训练（Retrain Remaining Weights）**
剪枝后网络精度会下降。此时对剩余的非零权重进行若干 epoch 的重训练（fine-tuning），让剩余权重"学习"补偿被移除权重的作用。这一步至关重要——如果直接剪枝不重训练，精度会大幅下降；通过重训练，剩余权重能重新组织到新的局部最优。

**迭代剪枝**：Step 2 和 Step 3 可以反复迭代，每次剪掉一部分权重后重训练，逐步提高稀疏度。相比于一次性剪掉大量权重，迭代方式更温和，精度恢复更好。

## 3. 关键公式（LaTeX）

**剪枝操作（Magnitude-based Pruning）**：

$$
W'_{ij} = \begin{cases}
W_{ij}, & \text{if } |W_{ij}| > \tau \\
0, & \text{otherwise}
\end{cases}
$$

其中阈值 $\tau$ 由目标稀疏度 $p$ 确定：$\tau = \text{percentile}(|W|, p)$。

**参数减少率（Compression Ratio）**：

$$
\text{CR} = \frac{\text{\#Params}_{\text{original}}}{\text{\#Params}_{\text{pruned}}} = \frac{1}{1 - p}
$$

当稀疏度 $p=0.9$ 时，$\text{CR} = 10\times$。

**重训练目标（微调阶段 Loss）**：

$$
\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \ell(f(x_i; W_{\text{sparse}}), y_i)
$$

其中 $W_{\text{sparse}}$ 仅包含非零权重，被剪掉的权重不参与前向/反向传播。

**迭代剪枝策略**：

$$
p_t = p_0 + \frac{t}{T}(p_{\text{target}} - p_0)
$$

其中 $p_t$ 为第 $t$ 轮迭代的目标稀疏度，$T$ 为总迭代轮数，$p_{\text{target}}$ 为最终目标稀疏度。

## 4. 实验结论

- **AlexNet 压缩效果**（ImageNet）：
  - 卷积层参数减少 **9×**（从 60M 降至约 6.7M）
  - 全连接层参数减少 **18×**（从 58.6M 降至约 3.2M）
  - 总体：Top-1 精度损失 <1%（原始 57.2% → 剪枝后 56.6%）
- **VGG-16 压缩效果**（ImageNet）：
  - 总参数减少 **13×**（从 138M 降至约 10.3M）
  - 全连接层参数减少 **49×**（从 123.6M 降至约 2.5M）
  - Top-5 精度损失 <0.5%
- **层敏感度分析**（Layer Sensitivity）：
  - 不同层对剪枝的敏感度差异显著：第一层卷积层（提取底层特征）对剪枝最敏感，全连接层（分类头）对剪枝最鲁棒
  - 均匀剪枝（所有层相同稀疏度）效果差于按层敏感度分区剪枝
- **迭代 vs 一次性剪枝**：迭代剪枝（每次少量剪枝+重训练）在同等稀疏度下精度高 2-3 个百分点
- **L1 vs L2 正则化**：L1 正则化在训练过程中自然产生更多接近零的权重，配合剪枝效果更好

## 5. 工业价值

这篇论文是神经网络剪枝领域的**开山之作**，其影响力体现在：

- **成为 AI 模型压缩的标准范式**：三步法（训练→剪枝→重训练）至今仍是多数剪枝工作的基础框架
- **催生了后续大量工作**：结构化剪枝（channel pruning）、动态剪枝、基于梯度的剪枝准则（如 SNIP、GraSP）、彩票假说（Lottery Ticket Hypothesis）等均源自此工作的启发
- **实际部署**：TensorFlow Lite、PyTorch Mobile 等框架中的模型优化工具内置了 magnitude-based pruning；NVIDIA TensorRT 的稀疏推理后端也支持此范式
- **与量化、蒸馏等方法的联合使用**：成为 Deep Compression 流水线的第一步（剪枝 + 量化 + 霍夫曼编码），是端到端模型压缩的起点
- **产业落地**：华为、高通等手机芯片厂商的 AI 引擎中，剪枝是标准优化手段之一

## 6. 与课程 lecture 的关系

- **Lecture 03（Pruning I）**：本文是 Lecture 03 的核心论文。Lecture 03 完整讲解了 magnitude-based pruning 的原理、迭代剪枝策略、层敏感度分析以及稀疏存储格式（CSR/CSC）。随后 Lecture 04（Pruning II）将讨论更高级的剪枝方法——结构化剪枝、彩票假说、基于梯度的剪枝准则等——这些都是在这篇工作的基础上发展而来的。

## 7. 我应该如何复现

1. **PyTorch 实现**：
   ```python
   import torch
   import torch.nn.utils.prune as prune

   # Step 1: 训练原始模型
   model = train_model()

   # Step 2: 全局剪枝
   parameters_to_prune = []
   for module_name, module in model.named_modules():
       if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
           parameters_to_prune.append((module, 'weight'))

   prune.global_unstructured(
       parameters_to_prune,
       pruning_method=prune.L1Unstructured,
       amount=0.9,  # 剪掉90%
   )

   # Step 3: 重训练
   for epoch in range(finetune_epochs):
       train_one_epoch(model)
   ```

2. **使用 torch-pruning 库**（更灵活）：
   ```python
   import torch_pruning as tp
   # 支持多种剪枝策略和结构化/非结构化剪枝
   DG = tp.DependencyGraph().build_dependency(model, example_inputs)
   ```

3. **复现关键实验**：
   - 在 CIFAR-10 上用 VGG-16 复现剪枝+重训练流程，观察不同稀疏度（50%/70%/80%/90%/%95%）下的精度变化
   - 绘制层敏感度曲线：分别对各层独立剪枝不同比例，观察精度下降
   - 对比迭代剪枝与一次性剪枝的精度差异

4. **主要超参数**：
   - 初始训练: SGD, lr=0.1, momentum=0.9, weight_decay=1e-4, epoch=90
   - 重训练: SGD, lr=0.001（降低学习率），epoch=20-40
   - 每层剪枝比例：根据层敏感度分配（卷积层低，全连接层高）

5. **常见坑**：
   - 剪枝后要用 `torch.nn.utils.prune.remove()` 永久移除 mask 才能获得真正的稀疏加速
   - 非结构化剪枝在 GPU 上实际加速有限（不规则内存访问），需要专用稀疏库（如 cuSPARSE）或特定硬件（2:4 结构化稀疏）
   - 重训练 epoch 数不够会导致精度无法恢复
