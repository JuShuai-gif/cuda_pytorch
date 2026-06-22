# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware

> Han Cai et al., ICLR 2019

## 1. 论文解决什么问题

神经网络架构搜索（NAS）虽然能自动发现高效架构，但存在两个重大瓶颈：(1) **代理任务（proxy task）偏差**：之前的 NAS 方法（如 NASNet、AmoebaNet、DARTS、MnasNet）在小型代理任务（如 CIFAR-10 或小数据集子集）上搜索，然后将搜索到的架构迁移到目标大任务（ImageNet），但这种迁移不能保证最优——代理任务上的好架构在目标任务上不一定好；(2) **代理硬件（proxy hardware）不匹配**：在 GPU 上以 FLOPs 作为效率指标进行搜索，但实际部署设备的延迟（latency）与 FLOPs 并不线性相关（受内存带宽、并行度等因素影响），导致搜索结果在实际设备上效率不高。(3) DARTS 类方法需要在搜索时保留所有候选操作在 GPU 内存中，导致巨大内存开销，无法直接在大型任务上搜索。

本文提出了 **ProxylessNAS**，首个无需代理（proxy-free）的 NAS 方法——直接在目标任务（ImageNet）和目标硬件（手机/GPU/CPU）上搜索，同时通过二值化路径（binarized path）技术大幅降低搜索内存。

## 2. 核心方法

ProxylessNAS 的核心创新包含两个方面：

### 2.1 基于 One-Shot NAS 的搜索空间

构建一个 "super net"（过参数化网络），其中每个层包含多个候选操作（如不同 kernel size 的卷积、identity skip、池化等）。搜索过程的目标是从每个层中选出一个最佳操作，使得最终子网络的精度和硬件效率最优。

将每个层的架构选择建模为可学习的架构参数 $\alpha$，通过 softmax 输出每个候选操作的概率。

### 2.2 二值化路径（Binarized Path）降低 GPU 内存

DARTS-style 搜索需要在训练 super net 时同时计算所有候选操作的梯度，内存消耗随候选操作数量线性增长。ProxylessNAS 的解决方案是：

- 在每次前向传播时，根据 $\alpha$ 的 softmax 概率对每个层**只采样两条路径**（二值化门控：两条路径以互补概率被激活）
- 利用 Gumbel Softmax 进行可微采样
- 仅被采样的路径参与前向和反向计算，未被采样路径的梯度为零

这将 GPU 内存从 $O(N)$ 降低到 $O(1)$（与候选操作数无关），使得直接在 ImageNet 上搜索成为可能。

### 2.3 硬件感知的延迟优化

不使用 FLOPs 作为效率代理，而是直接测量每个候选操作在目标硬件上的实际延迟。在搜索过程中，将延迟作为正则化项加入损失函数：

- 对每种候选操作在目标设备上跑多次延迟测量，构建操作的延迟查表（latency lookup table）
- 使用 **REINFORCE 梯度**来优化延迟目标，因为延迟对架构参数 $\alpha$ 不可微（离散硬件测量）

### 2.4 搜索算法

交替优化网络权重 $W$ 和架构参数 $\alpha$：
- 训练集优化 $W$（正常梯度下降）
- 验证集优化 $\alpha$（REINFORCE + 梯度下降的混合）
- 最终从收敛后的 $\alpha$ 中选出概率最高的操作作为最终架构

## 3. 关键公式（LaTeX）

**架构参数的 softmax 概率**：

$$
P(o_j | \alpha_i) = \frac{\exp(\alpha_{i,j})}{\sum_{k} \exp(\alpha_{i,k})}
$$

其中 $\alpha_i$ 为第 $i$ 层的架构参数向量，$o_j$ 为第 $j$ 个候选操作。

**二值化路径采样（Gumbel Softmax 简化版）**：

$$
m_j = \begin{cases}
1, & j = \arg\max_k (\log P(o_k) + g_k) \\
0, & \text{otherwise}
\end{cases}, \quad g_k \sim \text{Gumbel}(0, 1)
$$

为了稳定梯度，实际使用两条路径的二值化版本：

$$
\tilde{P}(o_j) = \frac{\exp(\alpha_{i,j})}{\exp(\alpha_{i,a}) + \exp(\alpha_{i,b})}
$$

每次对每个层选择两条路径 $a, b$，其余路径不被激活。

**REINFORCE 梯度（延迟优化）**：

$$
\nabla_\alpha \mathbb{E}_{P(\alpha)}[LAT(m)] = \mathbb{E}_{P(\alpha)}[LAT(m) \cdot \nabla_\alpha \log P(m)]
$$

其中 $LAT(m)$ 是采样架构 $m$ 在目标硬件上的实测延迟。

**总损失函数**：

$$
\mathcal{L} = \mathcal{L}_{\text{CE}}(W, \alpha) + \lambda_1 \cdot LAT(\alpha) \cdot \left[\frac{LAT(\alpha)}{LAT_{\text{target}}}\right]^w
$$

其中 $\lambda_1$ 控制延迟惩罚强度，$LAT_{\text{target}}$ 为目标延迟，$w$ 为强化目标约束的指数。

## 4. 实验结论

- **直接搜索 vs 代理搜索**：ProxylessNAS 在 ImageNet 上直接搜索得到的架构（Proxyless-GPU/Mobile/CPU）在同等延迟约束下精度始终优于或持平 MnasNet（代理搜索）
- **ImageNet 精度**：ProxylessNAS-Mobile 在 300M FLOPs 下达到 74.6% Top-1（MobileNetV2 同量级为 72.0%），ProxylessNAS-GPU 在 425M FLOPs 下达到 75.1%
- **实测延迟**（Pixel 1 手机）：ProxylessNAS-Mobile 延迟 78ms vs MnasNet-A1 84ms，但精度高 0.5%
- **搜索效率**：直接在 ImageNet 上搜索仅需约 **200 GPU hours**（8×V100 约 1 天），而 NASNet 在代理任务上就需 1800 GPU days
- **记忆开销**：二值化路径将 GPU 内存需求从 DARTS 的 >15GB 降低到 <8GB（单卡 V100 即可搜索）
- **不同硬件平台的搜索结果不同**：CPU 延迟最优架构 ≠ GPU 延迟最优架构 ≠ 手机延迟最优架构，证明了在目标硬件上直接搜索的必要性

## 5. 工业价值

- **使 NAS 实用化**：之前 NAS 需要几千 GPU days 在代理任务上搜索，ProxylessNAS 使 NAS 可在实际项目中被采用——200 GPU hours 在商业环境中完全可承受
- **硬件感知设计范范**：将部署延迟直接融入搜索目标，使得 AI 芯片公司和手机厂商（华为、高通、苹果）可以将 NAS 用于定制芯片上的模型优化
- **推动了 Once-for-All（OFA）的诞生**：ProxylessNAS 的作者后续提出 OFA，进一步将搜索效率提升到"一次训练，多个硬件部署"的级别
- **被工业界广泛采用**：微软的 NNIt、Google 的 Cloud AutoML 均采用了类似的可微 NAS 架构搜索方案
- **证明了代理无关（proxy-free）搜索的优越性**：影响了后续工作（如 FBNet, MobileNetV3 架构设计）直接在目标任务上优化

## 6. 与课程 lecture 的关系

- **Lecture 08（NAS II - Hardware-aware NAS）**：本文是 Lecture 08 的核心论文之一。课程从 NAS 的发展脉络（NASNet → ENAS → DARTS → ProxylessNAS → OFA）讲解，ProxylessNAS 是连接"纯精度搜索"和"硬件感知搜索"两个时代的关键桥梁。Lecture 08 会重点讨论 binarized path 的内存优化技巧和 REINFORCE 的硬件延迟优化。

## 7. 我应该如何复现

1. **核心代码框架**（PyTorch）：
   ```python
   class MixedOperation(nn.Module):
       def __init__(self, candidate_ops):
           super().__init__()
           self.ops = nn.ModuleList(candidate_ops)
           self.alpha = nn.Parameter(torch.randn(len(candidate_ops)))

       def forward(self, x):
           # Binarized path: sample 2 ops
           probs = F.softmax(self.alpha, dim=0)
           # Gumbel-based sampling
           ...
           return output
   ```

2. **使用开源代码**：
   - 官方实现：`https://github.com/MIT-HAN-LAB/ProxylessNAS`（PyTorch）
   - 搜索空间定义在 `models/super_nets/` 目录下
   - 延迟表生成脚本在 `scripts/` 目录

3. **简化复现路线**：
   - **Phase 1（理解机制）**：在 CIFAR-10 上用 8 层 CNN 超网实现二值化路径搜索，不加入硬件延迟约束
   - **Phase 2（硬件感知）**：在 ImageNet 100 子集上搜索，用 `torch.cuda.Event` 测量每个候选操作在 GPU 上的延迟，构建 latency lookup table，加入 REINFORCE 延迟正则化
   - **Phase 3（完整复现）**：在 ImageNet 上搜索 ProxylessNAS-Mobile 或 ProxylessNAS-GPU

4. **关键超参数**：
   - 架构参数学习率：$\text{lr}_{\alpha}=3\times10^{-3}$（远大于权重学习率），Adam 优化器
   - 权重学习率：$\text{lr}_{W}=0.1$，cosine decay，SGD momentum=0.9
   - 温度衰减（Gumbel Softmax temperature）：$\tau$ 从 5 线性衰减到 1
   - 延迟正则化权重 $\lambda_1$：根据目标延迟调整（通常 0.1-0.5）
   - 搜索 epoch：约 300 epochs 在 ImageNet 上

5. **常见坑**：
   - 延迟测量需要 warm-up（前几次推理的延迟不稳定）
   - REINFORCE 梯度方差大，需要与交叉熵梯度的合理平衡
   - Gumbel Softmax 的温度衰减速度影响搜索稳定性
   - 多设备延迟测量的同步误差
