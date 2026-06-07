# 第七讲：神经架构搜索 I — 搜索空间、搜索策略与 One-Shot NAS

## 1. 本讲核心问题

手工设计神经网络架构越来越像一门"玄学"——依赖经验、直觉和大量试错。本讲介绍**如何让 AI 自己设计 AI 架构**：

- 什么是神经架构搜索（NAS）？NAS 的三个核心组件：**搜索空间、搜索策略、性能评估策略**是什么？
- 搜索空间如何设计？**cell-level**（搜索单元结构然后堆叠）和**network-level**（搜索整个网络拓扑）各有什么优劣？
- 搜索策略有哪些？网格搜索、随机搜索、强化学习、进化算法、梯度方法（如 DARTS）各自的特点？
- **权重共享 (Weight Sharing)** 如何把 NAS 搜索时间从数千 GPU 天降到几个 GPU 天？
- **One-Shot NAS** 的核心思想：训练一个超级网络（supernet），子网络直接继承权重，无需从头训练？
- 为什么"手动设计架构的时代已经结束了"？NAS 在哪些方面超越了人类设计？

## 2. 通俗解释

想象你要开一家连锁餐厅，需要设计厨房的布局（这就是神经架构）：

- **手工设计**：请一个有 20 年经验的大厨来规划厨房——洗菜池放哪、灶台几个、冰箱怎么摆。大厨凭经验画图，效果可能不错，但永远不知道有没有更好的方案。
- **网格/随机搜索**：你雇了 100 个实习生，每人随便画一个厨房平面图，然后挑一个最好的。运气好的话能找到不错的，但大量方案都是垃圾。
- **强化学习搜索**：你让一个 AI 厨师不断尝试不同的厨房布局。每次试完，根据出餐速度和菜品质量给一个分数（奖励）。AI 厨师逐渐学会什么布局好、什么不好。问题是每种布局都要真的建一个厨房来试，成本极高。
- **进化算法搜索**：你"繁殖"厨房布局：拿两个好布局 A 和 B，各取一半组合成新布局 C（交叉），再随机改一点（变异）。一代一代"进化"，越来越优。
- **One-Shot NAS**：你需要一个"万能厨房"——一个超级巨大的厨房，里面有所有可能的设备位置。你不需要为每种布局重新建厨房，只需要在这个超级厨房里"关掉"一些设备，模拟出子厨房的样子。子厨房做好一道菜试吃（验证精度），发现不行就换个"关法"。一个超级厨房，试出最优子厨房——这就是**权重共享**的本质：所有子网络共享超级网络的权重。

- **DARTS**：你不是在离散空间里挑布局（"灶台要么在左要么在右"），而是把每种可能性都赋予一个"概率"，然后在连续空间里用梯度下降优化这些概率——最后选概率最高的那几个，形成最终布局。这是对 NAS 的**连续松弛**，使得搜索可以用梯度下降，效率飞升。

## 3. 关键公式

### NAS 问题的数学表述

给定搜索空间 A，目标是找到最优架构 a*：

$$a^* = \arg\min_{a \in \mathcal{A}} \mathcal{L}_{val}(w^*(a), a)$$

其中 w*(a) 是架构 a 训练至收敛的最优权重：
$$w^*(a) = \arg\min_w \mathcal{L}_{train}(w, a)$$

### DARTS 的连续松弛

对于每一层，引入架构参数 α = {α^(i,j)} 作为候选操作 o 的混合权重：

$$\bar{o}^{(i,j)}(x) = \sum_{o \in \mathcal{O}} \frac{\exp(\alpha_o^{(i,j)})}{\sum_{o' \in \mathcal{O}} \exp(\alpha_{o'}^{(i,j)})} \cdot o(x)$$

### DARTS 的 bilevel 优化

$$\min_\alpha \mathcal{L}_{val}(w^*(\alpha), \alpha)$$
$$\text{s.t. } w^*(\alpha) = \arg\min_w \mathcal{L}_{train}(w, \alpha)$$

对 α 求导（近似，假设 w* 已最优）：
$$\nabla_\alpha \mathcal{L}_{val}(w^*(\alpha), \alpha) \approx \nabla_\alpha \mathcal{L}_{val}(w - \xi \nabla_w \mathcal{L}_{train}(w, \alpha), \alpha)$$

### One-Shot NAS 的关键假设

超网络权重 W 被所有子架构共享。子架构 a 的精度由其继承的权重 W_a 决定：

$$\mathcal{L}_{val}(W_a, a) \approx \mathcal{L}_{val}(W|_{a}, a)$$

其中 W 是训练好的超网络权重，W|ₐ 表示从 W 中取出属于架构 a 的那部分。

## 4. 公式背后的直觉

- **双层优化（Bilevel Optimization）**：NAS 本质上是一个"嵌套"问题——你需要找到最优架构 a，但要知道 a 好不好，你得先把它训练到最优（找到 w*(a)）。这就是 NAS 最贵的部分：评估每个候选架构都需要从头训练。

- **DARTS 的连续松弛直觉**：传统的架构搜索是在离散集合里挑选操作（比如从 {3x3 卷积, 5x5 卷积, max pooling} 中选一个），这无法用梯度优化。DARTS 的洞见是：给每个候选操作分配一个"混合权重" α，网络的输出是**所有操作的加权和**。在搜索过程中，α 逐渐分化——好的操作权重趋近于 1，差的操作趋近于 0，最终取 argmax 就得到离散架构。这就像是在做一个"软选择"，搜索完再做"硬选择"。

- **近似梯度的必要性**：精确计算 ∇_α 需要先完全训练 w 到收敛，这成本太高。DARTS 使用了一步近似：假设 w 已经"足够接近"最优，用一个 inner 梯度步来近似 w* 对 α 的依赖。这在实践中效果很好，虽然理论上不精确。

- **权重共享的革命性意义**：在传统 NAS 中，评估 10000 个候选架构需要训练 10000 个模型。One-Shot NAS 的核心洞见是：这些架构之间有很多重叠的结构（比如都有一些 3x3 卷积），为什么不共享权重呢？训练一个超级网络，子网络直接从超网继承权重，评估时间从几小时降到几秒钟。这个想法的代价是：子网络的精度评估不如独立训练准确（因为权重是在"共享环境"下训练的，不是为某个子网络定制的），但相关性足够用于排名。

## 5. 工业界用途

- **Google AutoML / Cloud AutoML**：最早的商业化 NAS 产品，用 RL + 权重共享自动搜索图像分类模型，在 ImageNet 上超过了手工设计的 NasNet。
- **EfficientNet (Google)**：虽然不是全自动 NAS，但使用了复合缩放策略，核心思想来自 NAS 的搜索经验——"宽度、深度、分辨率"应该同时缩放，而非单独调整。
- **MobileNetV3 (Google)**：结合了 NAS（搜索宏观架构）和 NetAdapt（搜索微观层参数），是 Platform-aware NAS 的工业级案例。
- **Facebook D2Go**：基于 NAS 的移动端目标检测系统，在手机上实现了实时检测。
- **华为 MindSpore / 百度 PaddleSlim**：内置了基于 DARTS 和 One-Shot 的 NAS 模块，支持开发者自动搜索适合特定硬件的模型架构。
- **AutoML-Zero**：Google 的一个极端项目，连激活函数和优化器都不用预设——从基本数学运算开始进化整个机器学习算法。展示了 NAS 思想的终极拓展。
- **Deci AI**：以色列创业公司，核心产品就是用 NAS 为特定 GPU/CPU 自动发现最优模型架构，实现 2-10x 推理加速。

## 6. PyTorch 实现思路

### 简化的 DARTS 搜索实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 候选操作集合
OPS = {
    'none': lambda C, stride: Zero(stride),
    'skip_connect': lambda C, stride: Identity() if stride==1 else FactorizedReduce(C, C),
    'conv_3x3': lambda C, stride: ReLUConvBN(C, C, 3, stride, 1),
    'conv_5x5': lambda C, stride: ReLUConvBN(C, C, 5, stride, 2),
    'dil_conv_3x3': lambda C, stride: DilConv(C, C, 3, stride, 2, 2),
    'max_pool_3x3': lambda C, stride: nn.MaxPool2d(3, stride, 1),
    'avg_pool_3x3': lambda C, stride: nn.AvgPool2d(3, stride, 1),
}

class MixedOp(nn.Module):
    """DARTS 的核心：混合操作，每个操作有权重 α"""
    def __init__(self, C, stride):
        super().__init__()
        self._ops = nn.ModuleList()
        for op_name in OPS:
            self._ops.append(OPS[op_name](C, stride))

    def forward(self, x, weights):
        # weights 是 softmax(α)，维度为 num_ops
        # 输出是所有操作的加权和
        return sum(w * op(x) for w, op in zip(weights, self._ops))

class DARTSCell(nn.Module):
    """一个 DARTS 搜索单元，包含多个节点和混合边"""
    def __init__(self, steps, C):
        super().__init__()
        self._steps = steps  # 中间节点数
        self._ops = nn.ModuleList()

        # 为每对有向边 (i, j) 创建一个 MixedOp
        for i in range(steps):
            for j in range(2 + i):
                stride = 2 if j < 2 and i == steps-1 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, weights[offset + j])
                    for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        return torch.cat(states[2:], dim=1)  # concat所有中间节点


# 架构参数（需要梯度）和网络权重（需要梯度）
alpha_normal = nn.Parameter(torch.randn(k, len(OPS)).cuda(), requires_grad=True)
alpha_reduce = nn.Parameter(torch.randn(k, len(OPS)).cuda(), requires_grad=True)

# Bilevel 优化：交替更新 α 和 w
arch_optimizer = torch.optim.Adam([alpha_normal, alpha_reduce], lr=3e-4)

for epoch in range(search_epochs):
    for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(train_loader, val_loader)):
        # Phase 1: 更新架构参数 α
        arch_optimizer.zero_grad()
        logits = model(val_X)  # 在验证集上
        arch_loss = criterion(logits, val_y)
        arch_loss.backward()
        arch_optimizer.step()

        # Phase 2: 更新网络权重 w
        optimizer.zero_grad()
        logits = model(trn_X)  # 在训练集上
        loss = criterion(logits, trn_y)
        loss.backward()
        optimizer.step()
```

### One-Shot NAS 概念代码

```python
class SuperNet(nn.Module):
    """One-Shot 超网络，包含所有可能的子网结构"""
    def __init__(self):
        super().__init__()
        # 每个位置有一组候选操作
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                conv3x3(in_c, out_c),
                conv5x5(in_c, out_c),
                skip_connect(in_c, out_c),
            ]) for in_c, out_c in zip(channel_config[:-1], channel_config[1:])
        ])

    def forward(self, x, architecture):
        """根据 architecture（由 0/1 组成的选择向量）激活特定路径"""
        for block_idx, (block, arch_choice) in enumerate(zip(self.blocks, architecture)):
            x = block[arch_choice](x)
        return x

# 训练超网络：随机采样子网络
for step in range(total_steps):
    arch = torch.randint(0, 3, (num_blocks,))  # 随机选择一个子网
    optimizer.zero_grad()
    output = supernet(data, arch)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

# 搜索：在训练好的超网上评估候选子网
best_acc = 0
best_arch = None
for arch in candidate_architectures:
    acc = evaluate(supernet, arch, val_loader)
    if acc > best_acc:
        best_acc = acc
        best_arch = arch
```

## 7. TinyML / Edge AI 部署意义

- **NAS 让 "针对特定硬件定制模型" 成为可能**：不同的 MCU（ARM Cortex-M4, RISC-V, ESP32）有不同的计算图和内存特性，NA为每个芯片自动找到最优架构。手工设计每个平台、每个延迟约束的模型是不可能的。
- **One-Shot NAS 大幅降低搜索成本**：传统 NAS 搜一个架构要几千 GPU 天，One-Shot NAS 只需几天。降低了门槛后，中小团队也可以做商业部署前的架构定制。
- **搜索空间设计直接决定部署效率**：如果搜索空间只包含"理论上高效"的操作（如 depthwise conv, inverted bottleneck），NAS 搜出来的模型天然适合移动端。这就是为什么 MobileNetV3 的搜索空间排除了标准卷积——它太慢了。
- **MCUNet 的成功（后续讲）** 正是 NAS + TinyML 结合的典范：在极小的 SRAM（256KB）和 Flash（1MB）约束下，NA自动发现了比手工设计 MCU-friendly 架构好得多的模型。
- **硬件感知 NAS**让"精度-延迟-能耗"三维帕累托最优成为可能：不仅搜出最小模型，还能直接搜出"在目标设备上跑得最快"的模型。

## 8. 常见误区

1. **"NAS 找到的架构是不可解释的"**：虽然搜索到的架构确实复杂，但模式是可辨识的。例如 DARTS 倾向于使用 separable conv 和 skip connection——这在手工设计的模型中也是好设计。NAS 本质上是在加速"好设计的发现"。
2. **"DARTS 总是比 RL/进化算法好"**：DARTS 的优势是速度快，但在极端搜索空间下，RL 和进化算法可能找到更好的结果。DARTS 的连续松弛本身就是一种近似，有时近似效果不好。
3. **"权重共享没有代价"**：有代价。权重共享使得子网络的精度排名不如独立训练准确（ranking correlation 通常 0.6-0.8，不是 1.0）。这对"选出最好的一个架构"影响不大，但如果要做精确的精度排序，可能需要"微调"步骤。
4. **"搜索空间越大越好"**：不是。搜索空间太大意味着有更多"坏架构"淹没了"好架构"，搜索效率反而下降。好的搜索空间是"小而精"——只包含有潜力的候选操作。
5. **"NAS 替代了人类 ML 工程师"**：NAS 自动化了架构设计这一个环节，但问题定义、数据工程、损失函数设计、部署优化等仍然需要人。NAS 是工具，不是替代品。
6. **"DARTS 的 skip connection 偏好"**：DARTS 在后期容易过度选择 skip connection（identity），导致网络变浅、精度下降。这是因为 skip connection 在训练早期学得更快（因为不需要学习），在 bilevel 优化中被"错误的奖励"了。这是 DARTS 的一个已知缺陷，多个后续工作（P-DARTS, PC-DARTS）尝试解决。

## 9. 面试问题

**Q1：DARTS 的核心贡献是什么？它是如何让 NAS "可微分"的？**

DARTS 的核心贡献是连续松弛：将离散的架构选择（从候选操作集合中选一个）转换为对每个候选操作赋予一个连续的权重参数 α。网络输出变成所有候选操作的 softmax 加权和。这样，架构搜索从离散组合优化问题变成了可微分的连续优化问题，可以使用梯度下降。搜索完成后，取 argmax（最大的 α 对应操作赢）得到离散架构。这使 NAS 搜索时间从数千 GPU 天降到了几个 GPU 天。

**Q2：One-Shot NAS 中权重共享的根本假设是什么？什么情况下这个假设会失效？**

根本假设是：子网络从超网络中继承的权重接近该子网络独立训练到收敛的权重。即超网络权重的"子集质量"足够高。这个假设在以下情况下会失效：(1) 子网络与超网络差异过大（如宽度相差 4x+），继承的权重偏差很大；(2) 超网络训练不充分，子网权重还没有形成良好的"子结构"；(3) 候选架构之间的操作类型差异太大（比如一个用卷积一个用 attention），权重共享的统计特性完全不同。

**Q3：为什么在 NAS 中使用 cell-level 搜索而非直接搜索整个网络？**

Cell-level 搜索的优势是：(1) 搜索空间大幅缩小（搜一个 cell 的结构 vs 搜几十层的安排），但可以通过多次堆叠相同的 cell 来构建任意深度的网络；(2) 发现的 cell 结构可迁移——在 CIFAR-10 上搜到的 cell 可以直接用在 ImageNet 上；(3) 这种模块化设计与人类设计理念吻合（ResNet 的 residual block, Inception 的 multi-branch block 都是"单元堆叠"）。缺点是限制了网络级多样性（所有单元相同），可能不是全局最优。

## 10. 本讲总结

NAS 代表了深度学习从"手工设计"到"自动设计"的范式转变：

- NAS 的核心三要素是**搜索空间、搜索策略、性能评估策略**——三者相互制约。
- **搜索空间**的设计决定了搜索效率：cell-level 搜索通过模块化降低复杂度，是当前主流。
- **搜索策略**从暴力搜索（网格/随机）到智能搜索（RL/进化/DARTS）不断进化，DARTS 的连续松弛用梯度方法使搜索效率有了质的飞跃。
- **One-Shot NAS** 通过权重共享彻底改变了 NAS 的成本结构：超网络一次训练，无限子网络随后评估——从"搜索一个架构"变成"从一个训练好的超级网络中提取最优子网"。
- NAS + TinyML 的结合（如 MCUNet）正在重新定义边缘 AI 的可能性——不是"模型适应硬件"，而是"为硬件从头定制模型"。

一句话总结：NAS 不是在"寻找魔法的架构配方"，而是在"自动化架构搜索的工程流程"——权重共享和可微分搜索是这个工程化的关键突破。DARTS 和 One-Shot NAS 把 "NAS = 烧钱" 变成了 "NAS = 几块 GPU 跑几天"。
