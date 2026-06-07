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

### 生产级案例分析

- **Samsung 的 OFA 产品线实践**：Samsung 的 Galaxy 手机系列（从旗舰 S 系列到中端 A 系列）使用的芯片跨越 Exynos 2200、Snapdragon 8 Gen 1、Snapdragon 778G 三代架构。OFA 超网络在 ImageNet 22K 上训练一次（约 1200 V100 GPU 天，成本约 $60K），然后为每款芯片自动派生子模型（延迟从 5ms 到 80ms 全覆盖）。关键数字：相比为每款芯片手工设计模型，OFA 节省了约 60% 的工程人力（从 ~6 人月降到 ~2 人月），同时子模型的 ImageNet 精度比单独训练的 MobileNetV3 高 1-2%。
- **NVIDIA TAO Toolkit**：TAO（Train, Adapt, Optimize）内部使用了精度预测器来加速"模型适配"。用户上传一个预训练模型，TAO 在一个月内自动尝试约 10000 种剪枝/蒸馏/量化组合，精度预测器在 < 1 分钟内筛选出 top-50，然后在真实 GPU 上测。据 NVIDIA GTC 2023 公开数据，一个 ResNet-50 目标检测模型的适配从人工调参的 2 周缩短到 2 天，且最终 TensorRT 部署精度与人工调参持平。
- **Qualcomm AIMET**：高通骁龙的 AI 引擎工具链 AIMET 内置了硬件感知 NAS（叫 AutoQuant），专门为 Hexagon DSP 的 INT8 推理搜索最优量化方案。涉及 8640 种 per-channel vs per-tensor、symmetric vs asymmetric 的量化配置组合——手动调不可能。

| 方法 | 搜索维度 | 部署目标 | 实测延迟估计 | 生产成熟度 |
|------|---------|---------|------------|-----------|
| OFA (MIT) | kernel + depth + width | 手机/Edge GPU | Latency LUT | 高（Samsung 使用） |
| MnasNet (Google) | op type + skip + filter | Pixel 手机 TPU | 真实硬件测量 | 高（Google 内部） |
| FBNet (Facebook) | block type + channel+stride | 手机 CPU/GPU | Latency LUT | 高（Meta 内部） |
| ProxylessNAS | op + channel | 手机 GPU | 真实硬件 (TRT) | 中（开源） |
| Neural-Hardware Co-Search | 模型架构 + 硬件参数 | FPGA/ASIC | 硬件仿真器 | 低（研究阶段） |

> **关键工程教训**：延迟查找表（LUT）在 GPU 上误差 < 5%，但在 MCU 上误差可达 20-30%（因为 MCU 的 Flash→SRAM swap 开销和 DMA 行为受操作顺序影响巨大，不是简单的线性叠加）。如果用 LUT 而非真实硬件测量做 MCU 搜索，搜出来的"最优架构"可能在实际芯片上不是最优。

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

### 生产级常见陷阱

7. **"精度预测器在搜索空间的不同区域，预测准确度差异巨大"**（来自 Facebook FBNet 团队的教训）：他们在 FBNetV3 项目中训练了一个精度预测器，在全搜索空间上的 MSE 只有 0.4%。但当他们 drill down 后发现——在"宽浅型网络"区域（low depth, wide channels），预测器的误差只有 0.2%；而在"窄深型网络"区域（high depth, narrow channels），误差飙到 2.1%。原因：训练预测器时采样的 2000 个架构主要是宽浅型的（随机采样天然偏向中等配置），窄深型的样本极少。**工程启示**：预测器需要 stratified sampling——按架构维度（depth, width, resolution）分层采样，确保搜索空间的每个角落都有足够的训练样本。FBNetV3 团队用这一修正后，预测器在最坏区域（窄深型）的 MSE 从 2.1% 降到 0.6%。

8. **"OFA 的超网络训练完成后，如果不做 fine-tuning 直接派生子网，精度比独立训练差 3-5%"**：这是一个经常被忽视的工程细节。OFA 论文中报告的子网精度是经过 25 epochs 微调后的结果，不是直接从超网抽取的权重。在 Samsung 的实际部署中，每个派生的子模型还要在目标任务的少量数据上微调——这一步不能跳过。如果跳过，子网精度可能无法满足产品 SLA。**工程启示**：OFA 解决的是"避免为每个子网做完整预训练"的问题，但不解决"避免微调"的问题。在计算 budget 中留出子网微调的时间和 GPU 资源。

9. **"硬件感知 NAS 的延迟约束必须是'软约束'，否则搜索过程会陷入局部最优"**（来自 Google MnasNet 团队的经验）：如果延迟约束是硬约束（比如 Latency < 15ms，超过就淘汰），搜索过程中一个好的"种子架构"可能因为一步随机变异导致延迟略超 15ms 而被丢弃，而它再变异一步就可能回到约束内并更优。实践中用 soft penalty（如 latency violation × 10 作为负 reward）效果远好于 hard cutoff。MnasNet 的消融实验：soft penalty 搜到的最优架构精度比 hard cutoff 高 0.6%。

## 9. 面试问题

**Q1：OFA 的渐进式收缩为什么比直接随机采样训练超网络更好？**

渐进式收缩的核心优势是知识传递：(1) 大网络先学好了，然后在保持大部分精度的情况下逐步缩小——小网络能学到更好的初始化；(2) 从大到小的训练路径比随机采样平滑——每一步缩小幅度小，网络更容易适应；(3) 避免了"小网络拖累大网络"——如果随机混合训练，大网络还没有学好，就需要支持小网络配置，导致整体训练效率下降。实验结果：渐进式收缩训练的超网络，子网络精度比随机采样训练高 2-3 个百分点。

**Q2：精度预测器在 NAS 中如何避免过拟合？有哪些实用技巧？**

关键技巧：(1) 使用足够多样化的训练架构，覆盖搜索空间的不同区域；(2) 使用简单的预测器架构（2-3 层 MLP 即可，不需要深层网络），越深越容易过拟合；(3) 做 k-fold 交叉验证来评估预测器在未见架构上的泛化能力；(4) 使用"uncertainty-aware"预测（如 Bayesian 神经网络或 ensemble），对低置信度的预测进行"真实验证"；(5) 在搜索过程中持续更新预测器——用真实训练结果不断修正。

**Q3：MCUNet 是怎么把 NAS 用在 TinyML 上的？它的搜索空间和常规 NAS 有什么不同？**

MCUNet 的搜索空间专门针对 MCU 的内存约束设计：(1) 排除了所有占用内存大的操作（如大 kernel、大通道数的标准卷积），只保留 memory-efficient 操作（如 3x3 depthwise conv, inverted bottleneck）；(2) 每一层的通道数和输入分辨率被限制为特定的离散值，确保激活内存峰值不超过 SRAM 限制；(3) 搜索目标不仅包含精度，还包括"内存峰值 < SRAM"和"模型大小 < Flash"的硬约束——任何违反的候选架构直接被淘汰。这种"约束优先"的搜索空间设计与常规 NAS（精度优先，约束软约束）有本质区别。

**Q4（高难度）：你正在为 NVIDIA Orin（车规级 SoC，275 TOPS）设计一个多任务感知模型（同时做检测、分割、深度估计），需要用硬件感知 NAS 搜索骨干网络。你会如何设计搜索空间和评估策略？列出 3 个与传统 NAS 截然不同的设计决策。**

(1) **多任务精度不能用简单加权和**：三个任务的 loss 量级和收敛速度差异巨大（检测 loss 通常是分割 loss 的 10-50x）。如果用加权和作为搜索 reward，NAS 会全力讨好检测任务而牺牲分割。工程方案是用"多目标 Pareto 前沿"——对每个候选架构，在三个任务上的精度组成一个 3D 向量，只保留非支配解（对任意任务，不存在另一个解在所有三个维度上都 >= 它且至少一个严格 >）。最终由产品经理从 Pareto 前沿中选一个 trade-off。这在学术界几乎不用（通常只用单目标），但 Waymo 和 Cruise 的感知团队就是这样做的。

(2) **搜索空间必须包含"多任务解耦决策"**：共享 backbone 的哪些层、从哪一层分出 head——这本身就应该纳入搜索空间。从第 8 层分出的检测 head 和从第 12 层分出的效果可能差距很大，但推理延迟也随之变化。需要在搜索空间中显式编码"split point"作为一个超参数——大多数 NAS 论文根本不考虑这一维度。

(3) **评估策略必须使用"硬件在环"而非 LUT**：Orin 的 GPU（Ampere 架构）+ DLA（Deep Learning Accelerator，最多 5 TOPS）+ 共享 DRAM 带宽（204.8 GB/s）。由于 DLA 和 GPU 共享 DRAM 带宽，两个加速器同时跑推理时会竞争带宽——LUT 的"单算子延迟叠加"假设完全失效。需要在每次搜索评估时，在真实 Orin 板子上跑完整模型推理，测量端到端延迟。这大大增加搜索时间，但对自动驾驶这种 $100M+ 的硬件投入来说，额外的几天搜索时间完全值得。Tesla 的 FSD 团队在 Twitter 上透露他们的 NAS 每次评估都是在真实 FSD 芯片上跑的，不是模拟——这是硬实时系统的现实。

**Q5（高难度）：Zero-Shot NAS 中的 Zen-Score 和 NASWOT 等指标的理论基础是什么？为什么在 Transformer 搜索空间中这些指标的 ranking correlation 显著下降（从 CNN 的 0.7 降到 Transformer 的 0.4）？**

Zen-Score 的核心是计算 BN 层输入梯度（∇_x BN(f(x))）的 Frobenius 范数的期望——它近似测量了网络在随机权重下对输入的"敏感度"，这和一个众所周知的经验现象相关：训练初期损失下降快的网络，最终精度通常更高。但关键是这个"敏感度"是一个**聚合量**——它不区分梯度来自 skip connection 还是卷积路径。在 CNN 中，skip 和 conv 的梯度大致处于同一量级，但 Transformer 中 attention 的梯度范数通常比 FFN 的小 1-2 个数量级（因为 attention 有 softmax 归一化，限制了输出范围）。因此 Zen-Score 在 Transformer 中几乎只反映了 FFN 的结构，完全忽视了 attention 模式——而 attention 模式的差异（head 数、head dim）恰恰是 Transformer 架构搜索中最重要的设计变量。NASWOT 有类似的问题：它基于"不同输入产生不同激活模式"的直觉，用激活矩阵的秩来估计网络表达能力——但 Transformer 中不同输入的激活模式差异本身就比 CNN 小（因为 token mixing），信噪比低。

**Q6（高难度）：如果你是一家 AI 芯片创业公司的 CTO，你会选择"神经-硬件协同搜索"来同时设计 AI 加速器架构和最优模型吗？分析优劣势和现实可行性。**

从 CTO 角度看，协同搜索在理论上是完美的，但在商业现实中几乎不可行：

**不利因素（为什么现实中极少有公司做）**：(1) **时间窗口灾难**：芯片设计周期 18-24 个月（从架构到 tape-out），而模型架构每 6 个月就迭代一代。协同搜索意味着让你的芯片架构绑定一款"今天"最优的模型——等到芯片流片出来，这个模型可能已被替代。(2) **搜索成本过高**：协同搜索的状态空间 = 硬件参数空间（PE 阵列大小、SRAM buffer 配置、带宽分配）× 模型架构空间（op type、depth、width）= 乘积级爆炸。每次评估需要硬件仿真器跑一次完整的模型推理（几小时），总计搜索时间可能是月级别。(3) **单一客户风险**：芯片是为"一个模型"优化的，如果客户想跑不同的模型，可能完全无法发挥芯片优势。Google 的 TPU 最初就面临这个问题——为 Google 内部模型优化得太极端，导致外部客户觉得"通用性不足"。

**有利因素（什么时候该做）**：(1) 如果芯片专门为一个封闭生态做（如 Apple Neural Engine 只为 Apple 自己的 Core ML 模型服务），协同搜索值得做——因为模型和芯片都在同一公司控制下。(2) 对固定任务的 ASIC（如智能音箱的 KWS 芯片，一辈子只跑 KWS），协同搜索能找到远超通用芯片的能效比——但要确保 KWS 这个任务会稳定存在 5 年以上。

**我的 CTO 决策**：只用轻量级协同设计——在硬件设计时预留一定的灵活性（可配置的 PE 阵列维度、可配置的 SRAM partition），让 NAS 在这个"受限硬件空间"内搜索模型。不做完全从头开始的硬件-模型联合搜索。这是 Google TPU v4、Apple A17 Pro 实际采用的策略：硬件提供弹性，软件搜索最佳利用方式。

## 10. 本讲总结

本讲将 NAS 从"学术探索"推向了"工业部署"——硬件感知是 NAS 落地的关键：

- **硬件感知 NAS**将延迟、能耗等真实约束纳入搜索目标，从"只搜精度最优"变成"搜约束下的精度最优"。
- **精度预测器**和**零样本 NAS**从不同角度降低评估成本——预测器需要少量训练样本但更准确，零样本完全免训练但精度较低。
- **OFA**代表了 NAS 工程化的巅峰：一个超网络，无限派生子模型，渐进式收缩保证了子模型质量。它是"一次训练，到处部署"的极致体现。
- **MCUNet**展示了 NAS + TinyML 的协同效应：在 MCU 的极端约束下，专门设计的搜索空间 + 硬件感知搜索 = 超越手工设计的结果。
- **协同搜索**揭示了一个更深层的洞察：最优的系统设计需要"算法-硬件-编译器"三者联合优化，而非各自孤立。

一句话总结：如果 NAS I 回答了"如何找到好架构"，NAS II 回答了"如何找到能在真实设备上跑得又准又快的好架构"——硬件感知 NAS 把 NAS 从一个学术玩具变成了工业级生产工具。OFA 的"一炼多派"模式正在成为行业标准。

## 11. 工业落地checklist

| 检查项 | 说明 | 不做的后果 |
|--------|------|-----------|
| 精度预测器必须做 stratified sampling 覆盖搜索空间各角落 | Facebook FBNetV3 发现：随机采样偏向宽浅型网络，窄深型区域预测误差从 0.2% 飙到 2.1% | 窄深型候选架构精度预测严重失真，搜索可能错过最优架构 |
| OFA 超网派生的子模型必须做 fine-tuning，不能直接部署抽取权重 | Samsung 实际部署中每个派生模型需 25 epochs 微调；跳过则精度比独立训练差 3-5% | 子模型精度不满足产品 SLA，线上用户体验质量下降 |
| 延迟查找表在 MCU 上误差远超 GPU，必须用真实硬件 profile | GPU 上 LUT 误差 <5%，但 MCU 上因 Flash→SRAM swap 和 DMA 非线性能达 20-30% | 搜出的"最优架构"在 STM32F4 实测延迟比预估高 30%，帧率不达标 |
| 硬件感知 NAS 的延迟约束应使用软约束（soft penalty）而非硬截断 | Google MnasNet 消融：soft penalty 搜出的架构精度比 hard cutoff 高 0.6% | 好的种子架构因子代变异略微超约束被淘汰，搜索陷入精度次优的局部最优 |
| OFA 渐进式收缩完成后必须验证所有 target latency 的子网都达标 | 超网训练可能对某些 kernel × depth × width 组合覆盖不足，极端配置子网精度塌陷 | 产品线上某款低端芯片的子模型精度断崖式下降，需紧急回滚 |
| 协同搜索（算法+硬件）仅适用于封闭生态或固定任务的 ASIC | Google TPU、Apple ANE 采用"硬件提供弹性 + NAS 在受限空间搜索"的轻量协同设计 | 芯片设计周期 18-24 月 vs 模型迭代 6 月——绑定当前模型的芯片流片后已过时 |
| Zero-Shot NAS 指标（Zen-Score/NASWOT）在 Transformer 搜索空间的 ranking correlation 仅 0.4 | Transformer 中 attention 梯度范数比 FFN 小 1-2 个数量级，Zen-Score 几乎只反映 FFN 结构 | 用 Zen-Score 筛选 ViT 架构时可能误淘汰 attention 设计优秀的候选，精度损失 2-3% |
