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

### 生产级案例分析

- **Google EfficientNet 与 Pixel 手机**：Google 用 NAS 发现了 EfficientNet 系列架构（核心是 MBConv block 的搜索），该架构替代了手工设计的 MobileNetV2，成为 Pixel 手机相机 App 中的实时图像处理模型。在 Pixel 4 上，EfficientNet-EdgeTPU 以 < 30ms 延迟完成 HDR+ 渲染中的场景理解，功耗比 MobileNetV2 低 40%，精度反超 1.5 个百分点。这直接证明了"NAS 搜出来的架构优于人类设计"在消费级硬件上是成立的。
- **华为 Noah's Ark**：华为的 Vega AutoML 框架在华为 Mate 30 的 NPU 上使用硬件感知 NAS 搜索了相机 ISP pipeline 中的多个模型（降噪、超分、场景识别）。关键数据点：相比手工设计的 MobileNetV3，NAS 搜出的模型在 Kirin 990 NPU 上推理延迟降低 35%，同时精度提升 0.8%。这背后是"NPU 算子的真实延迟不是线性的"——某些看似高效的 op 在 NPU 上反而慢，NAS 能自动规避这些坑。
- **Microsoft Azure ML**：Azure 的 Automated ML 服务将 NAS 包装成 SaaS 产品，用户在 Azure 上提交数据，系统自动搜索最优架构。据微软公开的白皮书，典型客户（如零售行业的商品识别）使用 AutoML 比雇佣 ML 工程师手工调参节省 60% 的人力和 3 周时间。但 cost 是搜索一次需要约 $500-2000 的云计算费用（取决于搜索空间大小）。

| 方法 | 搜索耗时 (GPU Days) | ImageNet Top-1 | 硬件 | 适合场景 |
|------|---------------------|----------------|------|----------|
| 手工设计 MobileNetV3 | N/A | 75.2% | 手机 | 基线 |
| MnasNet (RL NAS) | 288 | 76.1% | Pixel 手机 (TPU) | 追求极致硬件适配 |
| DARTS (梯度) | 4 | 73.3% (CIFAR-10 cell) | 通用 GPU | 快速原型验证 |
| One-Shot NAS (OFA) | 1200 (超网训练一次) | 80.0% | 任意 (派生子网) | 多设备产品线 |
| EfficientNet-B0 (NAS+复合缩放) | ~200 (搜索阶段) | 77.1% | 手机/Edge TPU | 平衡精度与效率的工业标准 |

> **成本洞察**：MnasNet 搜索一次的成本（288 GPU 天 × $2/GPU-小时 ≈ $13,800）对一个产品线是合理的——一旦搜出的架构部署在百万台设备上，单设备成本可忽略不计。但对初创公司或学术团队，DARTS 的 4 GPU 天方案才是经济可行的。

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

### 生产级常见陷阱（来自真实部署经验）

7. **"搜索空间大于 10^10 时，随机搜索和进化算法表现接近"**：这是 Facebook 在 D2Go 项目中得到的血泪教训。他们把搜索空间设计得非常大（所有可能的 kernel size × channel × depth 的组合，约 10^12），结果发现随机搜索选出的 top-10 架构和进化算法选出的 top-10 架构，在 ImageNet 上精度中位数只差 0.3%。原因：当空间大到一定程度，好架构的分布趋于均匀，搜索算法本身的"智能"被稀释了。**工程启示**：把精力花在缩小和精炼搜索空间上，比花在选择搜索算法上收益更大。
8. **"One-Shot NAS 的 ranking correlation 在 ImageNet 上是 0.75，在 COCO 检测任务上只有 0.5"**：这是商汤科技在内部 NAS 平台上的实测数据。分类任务中权重共享的有效性尚可，但目标检测中——因为检测头的结构和 backbone 强耦合——超网继承的权重对检测子网的预测能力大幅下降。**工程启示**：如果你的下游任务和预训练任务差异大（分类 → 检测/分割），不要直接信任超网的精度排名，至少对 top-10 候选做一次完整微调再决策。
9. **"DARTS 的搜索架构在 batch size=64 时表现好，但 batch size=512 时就塌了"**：这是来自 Google Brain 内部实验的一个微妙问题。DARTS 在搜索阶段用小 batch size（受限于 GPU 显存，因为 MixedOp 显存开销大），搜出来的架构隐含地对小 batch 的 BN 统计特性有偏好。当后续在大 batch 下重新训练时，BN 的行为变化可能让精度下降 1-2%。**工程启示**：搜到的架构必须在目标部署配置（batch size、优化器、数据增强）下重训验证，不能直接信任搜索结果的精度数字。

## 9. 面试问题

**Q1：DARTS 的核心贡献是什么？它是如何让 NAS "可微分"的？**

DARTS 的核心贡献是连续松弛：将离散的架构选择（从候选操作集合中选一个）转换为对每个候选操作赋予一个连续的权重参数 α。网络输出变成所有候选操作的 softmax 加权和。这样，架构搜索从离散组合优化问题变成了可微分的连续优化问题，可以使用梯度下降。搜索完成后，取 argmax（最大的 α 对应操作赢）得到离散架构。这使 NAS 搜索时间从数千 GPU 天降到了几个 GPU 天。

**Q2：One-Shot NAS 中权重共享的根本假设是什么？什么情况下这个假设会失效？**

根本假设是：子网络从超网络中继承的权重接近该子网络独立训练到收敛的权重。即超网络权重的"子集质量"足够高。这个假设在以下情况下会失效：(1) 子网络与超网络差异过大（如宽度相差 4x+），继承的权重偏差很大；(2) 超网络训练不充分，子网权重还没有形成良好的"子结构"；(3) 候选架构之间的操作类型差异太大（比如一个用卷积一个用 attention），权重共享的统计特性完全不同。

**Q3：为什么在 NAS 中使用 cell-level 搜索而非直接搜索整个网络？**

Cell-level 搜索的优势是：(1) 搜索空间大幅缩小（搜一个 cell 的结构 vs 搜几十层的安排），但可以通过多次堆叠相同的 cell 来构建任意深度的网络；(2) 发现的 cell 结构可迁移——在 CIFAR-10 上搜到的 cell 可以直接用在 ImageNet 上；(3) 这种模块化设计与人类设计理念吻合（ResNet 的 residual block, Inception 的 multi-branch block 都是"单元堆叠"）。缺点是限制了网络级多样性（所有单元相同），可能不是全局最优。

**Q4（高难度）：DARTS 的 bilevel optimization 在数学上是不精确的。具体来说，∇_α L_val(w*(α), α) 的精确计算需要对 w*(α) 做隐函数求导——这需要对 Hessian ∇²_w L_train 求逆。DARTS 做的一步近似在什么条件下会严重偏离真实梯度？这对搜索有什么影响？**

精确的架构梯度 ∇_α L_val(w*(α), α) 由隐函数定理给出：∇_α L_val = ∇_α L_val(w*, α) - ∇_w L_val(w*, α) · (∇²_w L_train(w*, α))⁻¹ · ∇_α ∇_w L_train(w*, α)。DARTS 用恒等矩阵近似这个 Hessian 逆——相当于假设 ∇²_w L_train 是 identity。这个近似在以下情况下严重偏离：(1) 训练早期，网络权重远未收敛到任何局部最优，Hessian 的谱不是均匀的（condition number >> 1）；(2) 搜索空间中有 skip connection 时，这些路径上的 Hessian 对角元素接近零（因为 skip 路径不依赖权重），恒等近似把它放大到了 1，导致 DARTS 系统性地高估了 skip connection 的梯度，最终使架构退化（collapse to skip）。这就是为什么 DARTS 在后期偏好 skip connection 的**数学根源**。PC-DARTS 用 partial channel connection 部分缓解了这个问题（减少 skip 的"便宜"优势），而 R-DARTS 通过添加正则化项直接惩罚上述 Hessian 逆的近似误差。

**Q5（高难度）：权重共享的超网络训练中，"multi-path sampling"和"sandwich sampling"（先最大再最小再随机）哪个更好？从优化理论的角度解释。**

Sandwich sampling（OFA 的方法）更好。从优化理论看：(1) 最大子网（maximal model）的损失函数曲面的最优点是最"平坦"的——它有最多的参数和最大的容量，能最好地拟合训练分布，它的最优点附近 Hessian 的 trace 最小（即局部几何更光滑）；(2) 最小子网（minimal model）从最大子网的最优点继承权重然后微调，相当于在"由最大超网定义的、已经接近全局最优的流形"上做 fine-tuning，而不是从随机初始化的广阔空间中搜索；(3) 随机中间子网从已训练好的超网中采样，继承了经过(sandwich)平滑的权重，这些权重已经包含了从大到小的梯度信息。而纯 multi-path random sampling 的问题是：小网络在训练早期就被采样到了，它们从"尚未训练好的超网"中继承权重——这些权重本质上是噪声，导致小网络永远无法收敛好。**工业实践数据（来自 OFA 原论文的消融实验）**：用 sandwich sampling 的子网 ImageNet top-1 精度中位数比 random sampling 高 3.2 个百分点，且方差更小（标准差 1.1% vs 2.7%）。

**Q6（高难度）：如果让你为一家自动驾驶公司（如 Waymo）设计 NAS 方案以在 Xavier 芯片上部署检测模型，你会用哪种 NAS 方法？为什么？列出 3 条最关键的设计决策。**

这是一个典型的"硬件感知 NAS + 多目标优化"问题。三条最关键的设计决策：

**(1) 选择 One-Shot 方法而非 RL-based NAS**：Waymo 的感知模型（如 PointPillars + CenterNet）非常大，每次从头训练一个候选架构在 Xavier 上需要 3-5 天。One-Shot NAS 只用训练一次超网络（约 1 周），后续所有的子网评估都在秒级完成。Xavier 有 32 TOPS INT8 算力，超网络的训练可以在数据中心 GPU 集群完成，推理测试在 Xavier 上跑真实延迟。

**(2) 延迟预测器必须针对 Xavier 的硬件特性定制**：Xavier 有 GPU（Volta）+ DLA（专用推理加速器）+ 两个不同的内存池（共享 DDR + 专用 SRAM）。不能用一个全局延迟查找表——因为同一个 op 在 GPU 上和在 DLA 上的延迟可能差 3-5 倍。需要为每个候选架构决定"哪些层跑 GPU、哪些层跑 DLA"，这本身就是一个二次搜索。这是大多数 NAS 论文忽略的工程复杂性。

**(3) 搜索目标必须包含"worst-case latency"而非"average latency"**：自动驾驶是硬实时系统（hard real-time），99 百分位延迟比平均延迟重要得多。由于 Xavier 上的 GPU 和 DLA 共享内存带宽，某些架构组合可能导致带宽竞争，worst-case 延迟飙升。在搜索的目标函数中加了 Latency_p99 < 33ms 的硬约束（对应 30fps），违反的候选直接淘汰。这在学术界几乎没人做，但 Waymo 内部工程师必须面对。

## 10. 本讲总结

NAS 代表了深度学习从"手工设计"到"自动设计"的范式转变：

- NAS 的核心三要素是**搜索空间、搜索策略、性能评估策略**——三者相互制约。
- **搜索空间**的设计决定了搜索效率：cell-level 搜索通过模块化降低复杂度，是当前主流。
- **搜索策略**从暴力搜索（网格/随机）到智能搜索（RL/进化/DARTS）不断进化，DARTS 的连续松弛用梯度方法使搜索效率有了质的飞跃。
- **One-Shot NAS** 通过权重共享彻底改变了 NAS 的成本结构：超网络一次训练，无限子网络随后评估——从"搜索一个架构"变成"从一个训练好的超级网络中提取最优子网"。
- NAS + TinyML 的结合（如 MCUNet）正在重新定义边缘 AI 的可能性——不是"模型适应硬件"，而是"为硬件从头定制模型"。

一句话总结：NAS 不是在"寻找魔法的架构配方"，而是在"自动化架构搜索的工程流程"——权重共享和可微分搜索是这个工程化的关键突破。DARTS 和 One-Shot NAS 把 "NAS = 烧钱" 变成了 "NAS = 几块 GPU 跑几天"。

## 11. 工业落地checklist

| 检查项 | 说明 | 不做的后果 |
|--------|------|-----------|
| 搜索空间必须剔除不兼容目标硬件的操作 | 目标芯片是 ARM Cortex-M4（无 FPU）时，搜索空间必须排除 FP32 操作和标准 BN，只保留 INT8 友好的 depthwise conv、ReLU6 | NAS 搜出的架构在目标芯片上无法部署，推理延迟超预算 3-10x |
| DARTS 搜索的验证集不能复用训练集的 augmentation pipeline | 验证集必须用与训练集不同的 augmentation（如 cutout 比例、颜色扰动强度），否则 α 参数过拟合数据增强 | 搜索到的 cell 在 ImageNet 迁移时精度下降 3-5 个百分点 |
| 搜索空间 > 1e10 时先用随机搜索做 baseline | Facebook D2Go 项目验证：搜索空间 ~1e12 时随机搜索 top-10 与进化算法 top-10 精度中位数仅差 0.3% | 花两周用 RL/进化算法搜出的架构精度只比随机搜索高 0.3%，浪费数千 GPU 小时 |
| One-Shot NAS 子网评估精度必须做 re-train validation | 商汤实测：超网继承权重在 COCO 检测上的排名 correlation 仅 0.5，远低于 ImageNet 分类的 0.75 | 直接信任超网排名部署，COCO mAP 可能比独立训练低 3-6 个点 |
| DARTS 搜索的 batch size 须与部署训练 batch size 对齐 | DARTS 搜索受限于 MixedOp 显存，通常用小 batch（64），搜出架构隐含对小 batch BN 统计的偏好 | Google Brain 验证：大 batch（512）重训精度下降 1-2% |
| 硬件感知 NAS 必须用真实硬件测延迟，不能纯靠 LUT 叠加 | MCU 上 Flash→SRAM swap 开销和 DMA 行为受操作顺序影响，LUT 线性叠加误差可达 20-30% | 搜出的"理论最优"架构在 MCU 上实测延迟超预算 30%+ |
| NAS 搜出的架构必须在目标部署配置下完整 re-train 验证精度 | 搜索阶段超网继承精度不是最终模型精度，须用目标 batch size/优化器/数据增强重训 | 实际模型精度比搜索报告值低 2-5%，产品 SLA 不达标需回滚重搜 |

## 12. 学习闭环补充：NAS 的工业价值在 Pareto Frontier

### 12.1 工业核心

NAS 不是为了找到“唯一最优模型”，而是生成一组满足不同硬件约束的 Pareto 最优候选。真实产品往往需要多档模型：旗舰机、普通手机、低端设备、服务端 fallback。

### 12.2 Search Space 设计

| 维度 | 示例 |
|---|---|
| depth | block 数、layer 数 |
| width | channel/hidden size |
| kernel | 3x3、5x5、7x7、depthwise |
| expansion | MobileNet inverted bottleneck ratio |
| resolution | 输入分辨率 |
| attention | head 数、window size、token keep ratio |

Search space 太大，搜索成本爆炸；太小，找不到好模型。工业上常用经验结构加小范围搜索。

### 12.3 对应代码实验

```bash
python src/lecture-07/main.py
```

输出应至少包含：sampled architectures、估计 accuracy/cost、best candidate。

### 12.4 本讲验收问题

1. NAS 的 search space、search strategy、performance estimator 分别是什么？
2. 为什么 random search 常常是强 baseline？
3. Weight sharing 会带来什么 bias？
4. NAS 为什么不能只优化 FLOPs？
5. 如何把目标硬件 latency 加入目标函数？

## 13. Python 代码补充：随机搜索 NAS 的最小闭环

NAS 的 baseline 应该先实现 random search。很多复杂搜索算法如果打不过 random search，就没有工程价值。

```python
import random

def sample_arch():
    return {
        "depth": random.choice([2, 3, 4]),
        "width": random.choice([16, 32, 64]),
        "kernel": random.choice([3, 5]),
        "resolution": random.choice([96, 128, 160]),
    }

def proxy_accuracy(arch):
    return 0.70 + 0.02 * arch["depth"] + 0.001 * arch["width"] - 0.0002 * arch["resolution"]

def proxy_latency_ms(arch):
    return arch["depth"] * arch["width"] * arch["kernel"] ** 2 * arch["resolution"] ** 2 / 1e6

results = []
for _ in range(50):
    arch = sample_arch()
    acc = proxy_accuracy(arch)
    lat = proxy_latency_ms(arch)
    results.append((acc, lat, arch))

best = max(results, key=lambda x: x[0] - 0.02 * x[1])
print("best", best)
```

工业解读：真实 NAS 应该把 `proxy_latency_ms` 换成目标硬件实测 latency table，而不是 FLOPs 公式。

