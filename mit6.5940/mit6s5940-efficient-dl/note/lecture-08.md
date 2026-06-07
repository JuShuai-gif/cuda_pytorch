# 第八讲：NAS II — 硬件感知 NAS、精度预测器与 Once-for-All 网络

## 1. 本讲核心问题

上一讲的 NAS 只关注精度，但在实际部署中，**延迟、功耗、内存占用**同样重要——甚至更重要。本讲聚焦：

- 什么是**硬件感知 NAS (Hardware-Aware NAS)**？如何将延迟/能耗约束纳入搜索目标？
- 直接在每个候选架构上测量真实延迟太慢——**精度预测器 (Accuracy Predictor)** 如何避免全训练评估？
- 什么是**零样本 NAS (Zero-Shot NAS)**？不看任何训练结果，如何评估架构的潜力？
- **Once-for-All (OFA) 网络**如何一次训练，支持无限多种设备约束（不同延迟、不同硬件）？
- OFA 的**渐进式收缩 (Progressive Shrinking)** 训练策略如何工作？和普通训练有什么区别？
- **MCUNet** 作为 NAS+TinyML 的结合范例，如何解决 MCU 极受限内存下的推理问题？
- 什么是**神经-硬件协同搜索 (Neural-Hardware Co-Search)**？为什么比单纯搜模型或硬件更优？

## 2. 通俗解释

想象你是一家汽车公司的设计师：

- **精度-only NAS**：你只关心这辆车跑多快——谁快选谁，不管你油耗多少、能不能坐人。搜出来一辆 F1 赛车，极速 350km/h，但你没法开着它买菜接孩子。
- **硬件感知 NAS**：你现在有三个指标——速度（精度）、油耗（能耗）、价格（延迟）。你要找的不是最快的车，而是在给定预算下综合最优的车。"20 万以内，油耗低于 7L/100km，最舒适的家用车"——这就是硬件感知 NAS。
- **精度预测器**：你不想每设计一辆车都真的造一辆出来测。于是你建了一个"预测模型"——输入设计图纸，输出预测的性能。这个模型是拿之前真实造过的 1000 辆车的数据训练的。现在你可以在纸上画 10000 种设计，预测器告诉你哪种最好，然后只需要真的造前 10 名来验证。
- **零样本 NAS**：更激进——连真实造车数据都不需要。你直接看图纸上的某些"结构特征"（比如用了多少个大零件？是否有对称设计？重量分布均匀吗？），就能推断这个设计大概好不好。这类似于经验丰富的工程师只看图纸就能挑出明显差的设计。
- **OFA**：你想造一个"变形金刚"车。一次制造，但可以根据需要变成 SUV、跑车、货车。对于不同的客户（不同设备约束），你从这个变形金刚车上拆下不同的零件，拼出最适合他们的定制车型。生产一台变形金刚车（训练一个 OFA 超网络）的成本虽高，但一次投入后，可以为任何客户快速派生子型号。
- **渐进式收缩**：你不可能一天造出变形金刚。你先造一个完整版的标准跑车（最大的子网），训练好它。然后你让它"缩小"——拆掉一些零件变成家用车，微调让它也能跑好。再缩小变成微型车，再微调。这种从大到小逐步训练的方式，比同时学所有变形高效得多。

## 3. 关键公式

### 硬件感知 NAS 的多目标优化

Pareto 最优问题——最大化精度同时约束延迟：

$$\max_{a \in \mathcal{A}} \text{Acc}(a) \quad \text{s.t. } \text{Latency}(a) \leq T$$

或使用加权的目标函数：
$$a^* = \arg\max_{a \in \mathcal{A}} \left[\text{Acc}(a) - \beta \cdot \log\left(\frac{\text{Latency}(a)}{T}\right)\right]$$

### 精度预测器

训练一个回归器 f 从架构编码 x_a 预测精度：

$$\hat{\text{Acc}}(a) = f_\theta(x_a)$$

训练目标（最小化预测误差）：
$$\mathcal{L} = \sum_{(a, \text{Acc}(a)) \in \mathcal{D}} \|\hat{\text{Acc}}(a) - \text{Acc}(a)\|^2$$

x_a 通常是架构的向量表示，可以用 GCN、LSTM 或简单的 one-hot 编码。

### OFA 的渐进式收缩训练

训练顺序（从大到小）：

SuperNet training ——→ kernel-wise shrinking ——→ depth-wise shrinking ——→ width-wise shrinking

每个阶段：
$$\min_{W_{sub}} \mathcal{L}_{train}(W_{sub}), \quad W_{sub} \subset W_{super}$$

其中 W_super 是上一阶段学好的超网权重，W_sub 是当前阶段需要微调的子网权重子集。

### 零样本 NAS 指标（如 Zen-Score）

通过分析网络的表达能力和复杂度在初始化时预测性能：

$$\text{ZenScore} = \mathbb{E}_{x \sim \mathcal{D}}\left[\|\nabla_x \text{BN}(f_\theta(x))\|_F^2\right]$$

其中 BN 是 batch norm 层的输出。ZenScore 衡量了"网络对输入的敏感度"——对输入变化越敏感（梯度范数越大），表达能力越强。

## 4. 公式背后的直觉

- **硬件感知搜索的根本难题**：精度可以通过在验证集上跑一轮得到（很快），但延迟需要在真实硬件上跑才能准确。而 NAS 要评估成千上万个候选架构——无法为每个都测真实延迟。解决方案：建一个**延迟查找表 (Latency Lookup Table)**。测量每种操作（conv 3x3, depthwise conv 5x5, etc.）在不同输入形状下的延迟，然后一个模型的延迟 ≈ 各层延迟之和。虽然有误差（忽略了内存传输等全局开销），但对架构排名够用了。

- **精度预测器的直觉**："预测精度"比"训练到收敛测精度"快几个数量级。关键问题是：(1) 预测器和真实训练的相关性必须高；(2) 不能过拟合到已见过的架构区域。好的精度预测器通常只需要几百到几千个真实训练的样本来训练——相比于 NAS 要评估的数万架构，节省巨大。但预测器也有风险：在搜索空间的边界区域（远离训练样本），预测可能很不准。

- **零样本 NAS 的激进假设**：Zoph et al. 提出的 NAS 搜索成本很高，Zero-Shot NAS 试图**完全跳过训练**。核心思想：一个好的架构，在随机初始化时就应该表现出某些良好的结构特性——比如梯度流顺畅（没有梯度消失/爆炸）、表达能力与参数量的比值高等。虽不如精度预测器准确，但零样本方法在粗筛阶段很有用——快速排除明显差的架构。

- **OFA 为什么有效**：传统 NAS 为每个设备约束都要重新搜索一次。OFA 的洞见：这些搜索是高度冗余的——所有子网络共享相同的基本操作类型。如果你训练一个足够大的超网络，任何子网络都可以从中抽取权重，精度也不错。渐进式收缩进一步保证：先是最大的子网络学好（有足够容量拟合数据），然后逐步减少容量但保留大部分精度（因为小网络可以"知识蒸馏"大网络的经验）。

- **协同搜索的哲学**：传统上，算法和硬件是分开设计的——算法团队设计网络，硬件团队设计芯片。协同搜索认为：最优的算法依赖于硬件特性，最优的硬件也依赖于算法需求。联合搜索可以找到"算法-硬件"联合最优解，而非两个次优解的组合。

## 5. 工业界用途

- **OFA 在 Samsung**：Samsung 使用 OFA 为其多款手机（从高端 Galaxy S 到中端 A 系列）自动派生出最合适的模型代号。同一个 OFA 超网络，不同芯片（Exynos, Snapdragon）派生出不同子网。
- **Once-for-All 的 GitHub**：OFA 代码已在 GitHub 开源（mit-han-lab/once-for-all），支持 CNN 和 Transformer，在 ImageNet 上派生的子模型延迟从 5ms 到 100ms+ 全覆盖。
- **Apple**：Apple 的 Core ML 模型优化工具链内置了硬件感知压缩，虽然不是全自动 NAS，但对不同 Apple 芯片（A14, A15, M1, M2）使用不同的压缩配置。
- **Google Edge TPU**：Google 的 Edge TPU Compiler 支持对特定模型做算子级别的延迟预测和调优，硬件感知 NAS 的思路被内嵌在编译优化中。
- **NVIDIA TensorRT + TAO**：NVIDIA 的 TAO Toolkit 结合了 NAS 思路（architecture adaptation）和 TensorRT 部署，在不同 GPU 世代间迁移模型。
- **华为 Noah's Ark Lab 的 **Vega****：一个工业级 AutoML 框架，集成了硬件感知 NAS，支持在华为昇腾芯片上搜索最优架构。
- **自动驾驶**：Waymo、Tesla 等公司的感知模型需要在不同算力的车规级芯片上跑（不同车型配置不同），硬件感知 NAS 用于为每款车自动定制模型。

## 6. PyTorch 实现思路

### 简化的延迟查找表

```python
class LatencyLookupTable:
    """硬件感知 NAS 的核心：每种操作的延迟表"""
    def __init__(self):
        self.table = {}

    def measure_op(self, op_name, input_shape, in_channels, out_channels, stride):
        """在目标硬件上实际测量（通常是预先测好的）"""
        key = (op_name, input_shape, in_channels, out_channels, stride)
        # 在真实硬件上运行 100 次取平均——这一步离线完成
        self.table[key] = measured_latency

    def predict_latency(self, architecture):
        """预测一个架构的总延迟 = 各层延迟之和"""
        total = 0
        for layer in architecture:
            total += self.table[layer.signature]
        return total

# 硬件感知搜索
def hardware_aware_search(supernet, latency_table, target_latency, alpha=0.2):
    best_arch = None
    best_score = -float('inf')
    for arch in candidate_architectures:
        acc = supernet.evaluate(arch, val_loader)
        lat = latency_table.predict_latency(arch)
        if lat > target_latency:
            continue  # 违反延迟约束，跳过
        # 使用带权重的评分函数
        score = acc - alpha * (lat / target_latency)
        if score > best_score:
            best_score = score
            best_arch = arch
    return best_arch
```

### OFA 渐进式收缩的简化实现

```python
class OFASuperNet(nn.Module):
    """OFA 超网络，支持 kernel-wise, depth-wise, width-wise 的弹性"""

    def __init__(self, max_kernel=7, max_depth=20, max_width=1.0):
        super().__init__()
        self.kernel_choices = [3, 5, 7]
        self.depth_choices = [2, 3, 4]  # 每个 stage 的层数选择
        self.width_choices = [0.5, 0.75, 1.0]  # 宽度乘子
        # 使用最大配置构建超网络
        self.blocks = self._build_blocks(max_kernel, max_depth, max_width)

    def forward(self, x, arch_config):
        """根据 arch_config 动态选择路径"""
        # arch_config = {'kernel': 5, 'depth': 3, 'width': 0.75}
        # 只激活对应配置的通道和层
        ...
        return x

# 渐进式收缩训练
def progressive_shrinking(supernet, train_loader):
    # Stage 1: 训练最大子网络（全量 kernel + 全深度 + 全宽度）
    train_with_config(supernet, train_loader,
                      {'kernel': 7, 'depth': 4, 'width': 1.0})

    # Stage 2: 渐进收缩 kernel size（固定 depth 和 width）
    for k in [5, 3]:
        train_with_config(supernet, train_loader,
                          {'kernel': k, 'depth': 4, 'width': 1.0})

    # Stage 3: 渐进收缩 depth
    for d in [3, 2]:
        for k in [7, 5, 3]:
            train_with_config(supernet, train_loader,
                              {'kernel': k, 'depth': d, 'width': 1.0})

    # Stage 4: 渐进收缩 width
    for w in [0.75, 0.5]:
        for k in [7, 5, 3]:
            for d in [4, 3, 2]:
                train_with_config(supernet, train_loader,
                                  {'kernel': k, 'depth': d, 'width': w})
```

### 精度预测器的训练和使用

```python
class AccuracyPredictor(nn.Module):
    """基于架构编码预测精度的神经网络"""
    def __init__(self, arch_encoding_dim, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(arch_encoding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出预测精度
        )

    def forward(self, arch_encoding):
        return self.fc(arch_encoding)

# 编码架构（简化：每个 block 的选择 one-hot 拼接）
def encode_architecture(arch):
    encoding = []
    for block_choice in arch:
        encoding.extend([1 if i == block_choice else 0
                         for i in range(num_choices)])
    return torch.tensor(encoding, dtype=torch.float32)

# 收集训练数据：随机采样 + 真实训练
train_data = []
for _ in range(1000):  # 采样 1000 个架构
    arch = random_architecture()
    true_acc = train_and_evaluate(arch, train_loader, val_loader)
    train_data.append((arch, true_acc))

# 训练预测器
predictor = AccuracyPredictor(arch_encoding_dim)
for arch, acc in train_data:
    encoding = encode_architecture(arch)
    pred = predictor(encoding)
    loss = F.mse_loss(pred, torch.tensor([acc]))
    loss.backward()
```

## 7. TinyML / Edge AI 部署意义

- **硬件感知 NAS 是 TinyML 的"必需品"而非"奢侈品"**：MCU 的约束极其严格（SRAM < 512KB, Flash < 2MB），稍微超一点就部署不了。硬件感知 NAS 确保搜出来的架构一定满足硬件约束，而不是"搜完发现装不下"。
- **OFA 的部署灵活性**：一个产品线可能有多种 MCU（STM32F4 vs STM32H7 vs ESP32），OFA 一个超网络可以为每种 MCU 自动派生子模型，省去为每种芯片重新训练的代价。
- **精度预测器降低 MCU 的搜索成本**：MCU 上评估一个模型很慢（没有 GPU），精度预测器可以在服务端快速筛选候选架构，只把前几名拿到 MCU 上真测。这叫"两阶段搜索"。
- **零样本 NAS 在 TinyML 场景的局限**：大多数零样本指标（梯度范数、表达能力等）是在 GPU 上随机初始化后计算的，而 TinyML 模型极小、训练极度受限——零样本指标的相关性通常不如精度预测器。
- **协同搜索 = TinyEngine 的思想前身**：MCUNet 的成功不仅仅是搜了一个好模型（TinyNAS），还在同时优化了推理引擎（TinyEngine）。这就是协同搜索哲学：模型和引擎（硬件抽象层）一起优化。

## 8. 常见误区

1. **"硬件感知 = 只看延迟"**：延迟只是硬件约束之一。实际部署中，**能耗、内存占用峰值、编译时间、推理精度稳定性**都可能成为约束。特别是在电池供电设备上，能耗约束可能比延迟约束更关键。
2. **"精度预测器可以完美替代真实训练"**：精度预测器的误差通常在 0.5-2%。对于挑选前几名的架构，这个误差可以接受。但如果两个架构的真实精度只差 0.1%，预测器不一定能正确排序。
3. **"延迟查找表的加法假设没有误差"**：把各层延迟加起来忽略了很多因素——层间的数据搬运、内核启动开销、缓存竞争等。在 GPU 上这些开销相对小（< 5%），但在 MCU 上可能高达 20-30%（因为 SRAM 小，频繁 swap 到 Flash）。
4. **"OFA 的渐进式收缩可以用随机采样替代"**：实验证明，完全随机采样训练的效果远不如渐进式收缩。因为小网络从大网络"蒸馏"学得更好，而这种从大到小的知识传递正是渐进式收缩的核心。
5. **"协同搜索中，先搜模型再搜硬件和联合搜索一样"**：不一样。模型和硬件之间存在"鸡和蛋"的耦合。仅优化一方时，可能落入"局部最优"——比如搜出一个依赖大缓存（SRAM 大）的模型，但实际芯片的 SRAM 很小。
6. **"零样本 NAS 完全无用"**：在粗筛阶段很有用——快速排除掉明显差的架构，把计算资源集中在有希望的候选上。只是不应该作为唯一标准。

## 9. 面试问题

**Q1：OFA 的渐进式收缩为什么比直接随机采样训练超网络更好？**

渐进式收缩的核心优势是知识传递：(1) 大网络先学好了，然后在保持大部分精度的情况下逐步缩小——小网络能学到更好的初始化；(2) 从大到小的训练路径比随机采样平滑——每一步缩小幅度小，网络更容易适应；(3) 避免了"小网络拖累大网络"——如果随机混合训练，大网络还没有学好，就需要支持小网络配置，导致整体训练效率下降。实验结果：渐进式收缩训练的超网络，子网络精度比随机采样训练高 2-3 个百分点。

**Q2：精度预测器在 NAS 中如何避免过拟合？有哪些实用技巧？**

关键技巧：(1) 使用足够多样化的训练架构，覆盖搜索空间的不同区域；(2) 使用简单的预测器架构（2-3 层 MLP 即可，不需要深层网络），越深越容易过拟合；(3) 做 k-fold 交叉验证来评估预测器在未见架构上的泛化能力；(4) 使用"uncertainty-aware"预测（如 Bayesian 神经网络或 ensemble），对低置信度的预测进行"真实验证"；(5) 在搜索过程中持续更新预测器——用真实训练结果不断修正。

**Q3：MCUNet 是怎么把 NAS 用在 TinyML 上的？它的搜索空间和常规 NAS 有什么不同？**

MCUNet 的搜索空间专门针对 MCU 的内存约束设计：(1) 排除了所有占用内存大的操作（如大 kernel、大通道数的标准卷积），只保留 memory-efficient 操作（如 3x3 depthwise conv, inverted bottleneck）；(2) 每一层的通道数和输入分辨率被限制为特定的离散值，确保激活内存峰值不超过 SRAM 限制；(3) 搜索目标不仅包含精度，还包括"内存峰值 < SRAM"和"模型大小 < Flash"的硬约束——任何违反的候选架构直接被淘汰。这种"约束优先"的搜索空间设计与常规 NAS（精度优先，约束软约束）有本质区别。

## 10. 本讲总结

本讲将 NAS 从"学术探索"推向了"工业部署"——硬件感知是 NAS 落地的关键：

- **硬件感知 NAS**将延迟、能耗等真实约束纳入搜索目标，从"只搜精度最优"变成"搜约束下的精度最优"。
- **精度预测器**和**零样本 NAS**从不同角度降低评估成本——预测器需要少量训练样本但更准确，零样本完全免训练但精度较低。
- **OFA**代表了 NAS 工程化的巅峰：一个超网络，无限派生子模型，渐进式收缩保证了子模型质量。它是"一次训练，到处部署"的极致体现。
- **MCUNet**展示了 NAS + TinyML 的协同效应：在 MCU 的极端约束下，专门设计的搜索空间 + 硬件感知搜索 = 超越手工设计的结果。
- **协同搜索**揭示了一个更深层的洞察：最优的系统设计需要"算法-硬件-编译器"三者联合优化，而非各自孤立。

一句话总结：如果 NAS I 回答了"如何找到好架构"，NAS II 回答了"如何找到能在真实设备上跑得又准又快的好架构"——硬件感知 NAS 把 NAS 从一个学术玩具变成了工业级生产工具。OFA 的"一炼多派"模式正在成为行业标准。
