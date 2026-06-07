# Lecture 21: 端侧训练与迁移学习

## 1. 本讲核心问题

推理可以在端侧高效完成（量化、剪枝），但**训练**为什么这么难？本讲回答三个核心问题：

1. **端侧训练为何比推理难得多？** 反向传播需要存储所有中间激活值，内存爆炸；而推理只需存储当前层
2. **联邦学习如何保护隐私？** FedAvg 算法、梯度泄露风险、差分隐私防御
3. **如何让端侧训练变得可行？** Tiny Transfer Learning（TinyTL）、Sparse Back-Propagation、PockEngine 稀疏训练引擎

## 2. 通俗解释

**端侧训练 vs 推理的核心差异**：想象你在做一道复杂的数学题。推理是**从第一行写到最后一行的过程**——你只需要记住当前步骤。训练是**做完之后，倒着检查每一步有没有算错**——你需要记住**所有步骤的中间结果**，从最后一步一步步倒推回去。这些"中间结果"就是**激活值（activations）**，它们必须在反向传播时被重新使用来计算梯度。

对于一个 10M 参数的模型，推理可能需要 40MB 显存；但训练可能需要 400MB+——大部分都被激活值占据。在手机上，这翻倍的显存需求就是致命的。

**梯度泄露的直觉**：如果你把梯度发给服务器，服务器可能**从梯度反向推出你的原始数据**。举个例子：如果模型第一层是 Embedding，梯度就包含了"哪个词被激活了"的信息。对于图片分类模型，梯度甚至可以用来**重建原始输入图片**（Deep Leakage from Gradients）。联邦学习的目的就是"数据不出设备，只有模型参数和梯度在传"——但梯度本身也是隐私风险。

**FedAvg 的直觉**：100 个手机用户，每人手机上有自己的数据。服务器先发一个通用模型给所有人，每人在本地用自己的数据训练几个 epoch，然后把**模型权重的变化量**（而不是原始数据或梯度）传回服务器。服务器把所有人的权重变化量取加权平均，更新全局模型。重复这个过程。这就叫"数据不动模型动"。

**TinyTL 的直觉**：传统的迁移学习需要更新整个模型（或至少整个分类头）来做微调，这在端侧内存不够。TinyTL 的发现是：**只微调 bias 项和新增的轻量残差模块**，不更新大权重矩阵，也能达到接近全量微调的效果。原因？预训练权重已经抓住了通用特征，bias 和残差足以适配新任务。

**PockEngine 的直觉**：PockEngine 把训练看作一个"稀疏化"问题——不是所有神经元都需要做反向传播。它**动态决定哪些层需要真正的反向传播，哪些层可以直接跳过或近似计算**。这在推理时做一次快速的"重要性评估"，然后只训练重要的部分。

## 3. 关键公式

**训练内存分解**（$L$ 层网络，激活值大小 $a_\ell$）：
$$
M_{\text{training}} = \underbrace{|\theta|}_{\text{参数}} + \underbrace{|\mathbf{g}|}_{\text{梯度}} + \underbrace{M_{\text{opt}}}_{\text{优化器}} + \underbrace{\sum_{\ell=1}^{L} a_\ell}_{\text{激活值}}
$$

对于 Transformer，激活值内存估算：
$$
M_{\text{activations}} \approx B \times L \times S \times H \times \text{bytes\_per\_elem} \times \text{num\_layer}
$$
其中 $B$ 为 batch size，$L$ 为层数，$S$ 为序列长度，$H$ 为隐藏维度

**推理 vs 训练的内存倍数**：
$$
\frac{M_{\text{training}}}{M_{\text{inference}}} \approx 1 + \frac{M_{\text{opt}}}{|\theta|} + \frac{\sum a_\ell}{|\theta|}
$$
对于典型的小模型（5M 参数, Adam）：
$$
= 1 + \frac{12|\theta|}{|\theta|} + \frac{5|\theta|}{|\theta|} \approx 18\times
$$
**训练需要推理 18 倍的内存！**

**FedAvg 算法**：
$$
\mathbf{w}_{t+1} = \sum_{k=1}^{K} \frac{n_k}{n} \mathbf{w}_{t+1}^{(k)}
$$
其中 $\mathbf{w}_{t+1}^{(k)} = \mathbf{w}_t - \eta \nabla F_k(\mathbf{w}_t)$ 是客户端 $k$ 本地训练后的权重，$n_k$ 是其数据量

**梯度泄露（DLG - Deep Leakage from Gradients）**：
$$
\mathbf{x}^* = \arg\min_{\mathbf{x}} \left\| \frac{\partial \mathcal{L}(F(\mathbf{x}, \mathbf{w}), \mathbf{y}^*)}{\partial \mathbf{w}} - \nabla \mathbf{w} \right\|^2
$$
攻击者不知道 $(\mathbf{x}, \mathbf{y})$，但知道梯度 $\nabla \mathbf{w}$，通过优化一个**虚拟输入**使虚拟梯度与真实梯度匹配，从而重建输入。

**TinyTL 的内存节省**：
$$
M_{\text{TinyTL}} = \underbrace{|\theta_{\text{frozen}}|}_{\text{冻结（不存梯度）}} + \underbrace{|\theta_{\text{trainable}}|}_{\text{可训练}} + \underbrace{|\mathbf{g}_{\text{trainable}}|}_{\text{可训练梯度}} + \underbrace{M_{\text{opt}}^{\text{partial}}}_{\text{部分优化器}}
$$
因 $\theta_{\text{trainable}} \ll \theta_{\text{frozen}}$（仅 biases + 残差），优化器内存大幅降低

**Sparse Back-Propagation**：
$$
\frac{\partial \mathcal{L}}{\partial x_\ell} = \frac{\partial \mathcal{L}}{\partial y_\ell} \odot \mathbf{m}_\ell \cdot \frac{\partial f_\ell}{\partial x_\ell}
$$
其中 $\mathbf{m}_\ell \in \{0, 1\}$ 是稀疏掩码，只对"激活"的神经元做反向传播

## 4. 公式背后的直觉

- **激活值为什么占内存**：前向传播时，每一层的输出（激活值）必须保存到内存中，因为反向传播需要它们计算梯度。以一个 6 层 CNN 为例：第 1 层输出 32×112×112 的特征图（FP32, ~1.5MB），第 6 层输出可能 256×14×14（~0.2MB）。总共 ~10MB 激活值，而模型参数可能才 ~5MB。**激活值往往是最大的显存占用者**。

- **梯度泄露为什么可能**：梯度是损失函数对参数的偏导，它包含了数据和标签的信息。具体来说，$\frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial y} \cdot x^T$——梯度实际上等于输入 $x$ 乘以某个系数。有了梯度和模型权重，攻击者可以**优化一个虚拟输入，使它的梯度与截获的梯度一致**——如果优化成功，虚拟输入就等于真实输入。

- **TinyTL 的"记忆力"直觉**：预训练模型的权重像一本百科全书——已经涵盖了大量通用知识。微调 bias 就像在百科全书边角加几个批注，而不需要重写整本书。残差模块（Residual Adapter）提供了一种轻量的"个性化"方式，只增加很少参数即可适配新任务。

- **Sparse Back-Propagation 的直觉**：反向传播时，不是所有神经元对梯度都有显著贡献。有些神经元的梯度幅度极低（"死神经元"），反向传播它们相当于浪费计算。通过设置阈值 $\tau$，只对 $|\frac{\partial \mathcal{L}}{\partial y}| > \tau$ 的神经元做反向传播，可以节省 30-70% 的计算而不显著影响精度。

- **PockEngine 的"动态选择"**：传统训练中"哪些层需要反向传播"是静态决定的（要么全做，要么冻结）。PockEngine 在**每次迭代中动态评估**每层的重要性，对"不重要"的层跳过反向传播或用廉价近似代替。这就像考试时先扫一遍题目，只认真做高分值的题，简答题快速略过。

## 5. 工业界用途

| 技术 | 内存节省 | 应用场景 | 代表实现 |
|------|---------|---------|---------|
| TinyTL | 5-10× | 手机端个性化微调 | TinyTL (NeurIPS 2020) |
| Sparse BackProp | 30-70% FLOPs | 端侧持续学习 | MeProp, SWAT |
| PockEngine | 2-15× 内存 | 微控制器上的训练 | PockEngine (MLSys 2023) |
| FedAvg | N/A (隐私保护) | 手机键盘预测、医疗 | TensorFlow Federated, Flower |
| 差分隐私 SGD | 梯度加噪 | 合规的联邦学习 | Opacus (Meta), TF Privacy |

**具体实践**：
- **Gboard（Google 键盘）**：使用联邦学习训练下一个词预测模型，每日数百万设备参与，本地数据不出设备
- **Apple Siri/QuickType**：设备端联邦学习用于个性化语音识别和文本建议
- **Healthcare（NVIDIA FLARE）**：医院间联邦学习，在不共享患者数据的前提下训练疾病诊断模型
- **TinyML 持续学习**：智能家居设备（如 Nest 恒温器）根据用户行为持续在线微调预测模型

## 6. PyTorch 实现思路

```python
# ====================== 端侧训练内存分析 ======================
def profile_training_memory(model, input_shape, batch_size=1):
    """分析训练时的内存占用"""
    from torch.profiler import profile, ProfilerActivity

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(batch_size, *input_shape)

    # Hook 记录激活值内存
    activation_sizes = {}
    def hook_fn(name):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                activation_sizes[name] = out.numel() * out.element_size()
            elif isinstance(out, (list, tuple)):
                for i, o in enumerate(out):
                    if isinstance(o, torch.Tensor):
                        activation_sizes[f"{name}_{i}"] = o.numel() * o.element_size()
        return hook

    hooks = []
    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(hook_fn(name)))

    # 前向
    y = model(x)
    loss = y.sum()

    # 记录激活值总内存
    total_activation_mb = sum(activation_sizes.values()) / 1e6
    print(f"Total activation memory: {total_activation_mb:.1f} MB")

    # 反向
    loss.backward()

    # 清理 hooks
    for h in hooks:
        h.remove()

    return total_activation_mb

# ====================== TinyTL (只训练 bias + 残差) ======================
class TinyTransferLearning:
    def __init__(self, pretrained_model, num_classes):
        self.backbone = pretrained_model
        # 冻结所有预训练权重
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 新增轻量残差适配器（仅少量参数可训练）
        self.adapters = nn.ModuleList([
            ResidualAdapter(in_ch, reduction=4)
            for in_ch in self._get_channel_list()
        ])

        # 不替换分类头，只微调 bias
        # 将原分类层 bias 设为可训练
        pretrained_model.fc.bias.requires_grad = True

    def forward(self, x):
        # 前向：backbone 输出特征 + 适配器残差
        features = self.backbone.features(x)
        for i, adapter in enumerate(self.adapters):
            features[i] = features[i] + adapter(features[i])
        return self.backbone.fc(features[-1])

class ResidualAdapter(nn.Module):
    """轻量残差模块：conv 1x1 降维 + ReLU + conv 1x1 升维"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
        )
        # 初始化为接近恒等映射
        nn.init.zeros_(self.adapter[2].weight)

    def forward(self, x):
        return self.adapter(x)

# ====================== Sparse Back-Propagation ======================
class SparseBackpropLayer(nn.Module):
    """只有部分神经元参与反向传播的层"""
    def __init__(self, in_features, out_features, sparsity=0.5):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.sparsity = sparsity  # 反向传播的稀疏率

    def forward(self, x):
        return self.linear(x)

    def backward_with_sparsity(self, grad_output):
        """稀疏反向传播：只对 top-k 梯度最大的神经元计算"""
        # 选择梯度最大的 k 个神经元
        k = max(1, int(grad_output.shape[-1] * (1 - self.sparsity)))
        threshold = torch.topk(
            grad_output.abs().mean(dim=0), k
        ).values[-1]

        # 掩码：哪些神经元需要完整反向传播
        mask = (grad_output.abs().mean(dim=0) >= threshold).float()

        # 只对掩码神经元做完整梯度计算，其余设为 0
        sparse_grad = grad_output * mask.unsqueeze(0)
        # ... (后续正常计算，但计算量减少)

# ====================== 联邦学习 (FedAvg) ======================
class FederatedLearning:
    def __init__(self, global_model, num_clients=10, fraction=0.5):
        self.global_model = global_model
        self.num_clients = num_clients
        self.fraction = fraction  # 每轮参与训练的客户端比例

    def server_round(self, client_data_loaders):
        """服务器端：分发模型 + 收集更新 + 聚合"""
        selected = random.sample(
            range(self.num_clients),
            int(self.num_clients * self.fraction)
        )

        client_updates = []
        total_samples = 0
        for client_id in selected:
            # 发送全局模型给客户端
            local_model = copy.deepcopy(self.global_model)
            # 客户端本地训练
            n_samples, delta_w = self.client_train(
                local_model, client_data_loaders[client_id]
            )
            client_updates.append((n_samples, delta_w))
            total_samples += n_samples

        # 加权平均聚合
        for param_name in self.global_model.state_dict().keys():
            weighted_sum = sum(
                n * update[param_name]
                for n, update in client_updates
            )
            self.global_model.state_dict()[param_name] = weighted_sum / total_samples

    def client_train(self, model, dataloader, local_epochs=5):
        """客户端：本地 SGD 训练，返回权重变化量"""
        n_samples = len(dataloader.dataset)
        initial_weights = {
            k: v.clone() for k, v in model.state_dict().items()
        }

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for epoch in range(local_epochs):
            for x, y in dataloader:
                optimizer.zero_grad()
                loss = F.cross_entropy(model(x), y)
                loss.backward()
                optimizer.step()

        # 计算权重变化量 Δw = w' - w
        delta_w = {}
        for k in initial_weights:
            delta_w[k] = model.state_dict()[k] - initial_weights[k]

        return n_samples, delta_w

# ====================== 梯度泄露攻击 (仅用于理解安全风险) ======================
def gradient_leakage_attack(model, true_gradient, input_shape, steps=1000):
    """
    注意：此代码仅用于学术理解，请勿用于攻击
    从梯度重建原始输入数据
    """
    # 初始化随机虚拟输入
    dummy_input = torch.randn(1, *input_shape, requires_grad=True)
    dummy_label = torch.zeros(1, dtype=torch.long)  # 假设已知或可猜测

    optimizer = torch.optim.LBFGS([dummy_input], lr=0.1)

    for step in range(steps):
        def closure():
            optimizer.zero_grad()
            output = model(dummy_input)
            loss = F.cross_entropy(output, dummy_label)
            dummy_gradient = torch.autograd.grad(
                loss, model.parameters(), create_graph=True
            )
            # 匹配真实梯度
            gradient_diff = sum(
                ((dg - tg) ** 2).sum()
                for dg, tg in zip(dummy_gradient, true_gradient)
            )
            dummy_input.grad = torch.autograd.grad(
                gradient_diff, dummy_input
            )[0]
            return gradient_diff

        optimizer.step(closure)

    # 此时的 dummy_input 已经接近原始输入
    return dummy_input.detach()

# ====================== PockEngine 稀疏训练的简化思路 ======================
class PockEngineSparseTrainer:
    """动态选择哪些层做完整反向传播"""
    def __init__(self, model, sparsity_target=0.5):
        self.model = model
        self.sparsity_target = sparsity_target
        self.layer_importance = {}  # 每层的重要性分数

    def compute_layer_importance(self):
        """评估每层对最终梯度的贡献"""
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # 梯度范数作为重要性指标
                importance = param.grad.norm()
                self.layer_importance[name] = importance

    def select_active_layers(self):
        """选择最重要的层做反向传播"""
        sorted_layers = sorted(
            self.layer_importance.items(),
            key=lambda x: x[1], reverse=True
        )
        num_active = int(len(sorted_layers) * (1 - self.sparsity_target))
        return set(name for name, _ in sorted_layers[:num_active])

    def training_step(self, x, y):
        # 前向（全部层参与）
        output = self.model(x)
        loss = F.cross_entropy(output, y)
        # 全量反向传播（获取所有梯度）
        loss.backward(retain_graph=True)
        self.compute_layer_importance()
        active_layers = self.select_active_layers()

        # 只对活跃层的参数应用优化器
        for name, param in self.model.named_parameters():
            if name in active_layers and param.grad is not None:
                # 更新该层参数
                param.data -= self.lr * param.grad
```

## 7. TinyML / Edge AI 部署意义

端侧训练是 TinyML 的终极形态——不仅是"小模型在边缘跑推理"，而是"小模型在边缘持续学习和适应"：

- **个性化**：每个人有不同的行为模式（打字风格、日常路线），端侧训练使模型适配个人
- **持续的分布漂移适应**：传感器数据随季节变化（温度、光照），端侧训练使模型自动适应
- **隐私保护**：敏感数据（健康指标、私人对话）不需要离开设备
- **离线学习**：设备在无网络连接时也能从新数据中学习

**当前挑战与进展**：
- **内存**：端侧训练需要 ~18× 推理内存，是最大瓶颈。TinyTL 和 PockEngine 将差距缩小到 2-5×
- **功耗**：反向传播的功耗是前向的 3-5×。微控制器（如 Cortex-M）上训练仍不现实——目前主要集中在手机/平板级别
- **框架支持**：TensorFlow Lite for Microcontrollers 初步支持训练（~100KB 内存），PyTorch Mobile 支持推理但不支持训练
- **硬件演进**：新型 AI 加速器（如 Hailo-8、Edge TPU v2）开始支持片上训练，但主流仍不支持

## 8. 常见误区

1. **"端侧训练只需要存梯度就行"** — 最大的内存消耗者是**激活值**（activations），不是梯度。激活值必须在整个前向过程中保存，直到反向传播使用它们。Gradient Checkpointing 可以用计算换内存（在前向时不存激活值，反向时重新计算），但增加了 33% 的计算量。

2. **"联邦学习就是安全的"** — FedAvg 只保证原始数据不出设备，但梯度本身可能泄露隐私（DLG 攻击、成员推断攻击）。差分隐私（DP-SGD）通过给梯度加噪来防御，但噪声会降低模型质量。安全和精度的 trade-off 是联邦学习的核心难题。

3. **"TinyTL 就是冻结权重 + 微调分类头"** — 不准确。TinyTL 的创新在于**不换分类头**（保持原预训练的分类头的权重矩阵不变），只更新 bias + 新增的残差适配器。这比"冻结 backbone + 重新训练分类头"更省内存，因为分类头的权重矩阵通常很大（如 MobileNetV2 的分类头有 1.2M 参数）。

4. **"稀疏反向传播等于随机丢弃"** — Sparse Back-Propagation 通常基于**梯度幅度**来选择重要神经元，不是随机丢弃。随机丢弃（如 Dropout）会严重损害训练质量，而基于重要性的选择保证关键的梯度路径不被中断。

5. **"端侧训练不适合大模型"** — 端侧训练确实不适合从头训练大模型。但结合 PEFT（Parameter-Efficient Fine-Tuning）方法如 LoRA、Adapter，可以在端侧微调大模型的小部分参数（如 0.1% 参数），实现有效的个性化适配。

6. **"PockEngine 在所有情况下都能加速"** — PockEngine 的加速取决于模型的结构稀疏性。如果模型本身很紧凑（如 MobileNet），可稀疏化的空间有限。此外，动态评估层重要性的开销可能超过稀疏反向传播的节省，需要仔细的工程权衡。

## 9. 面试问题

**Q1: 为什么端侧训练比推理需要更多内存？列出主要贡献来源。**
A: 推理只需存储当前层的参数和激活值（一层处理完即释放）。训练需要：(1) 所有层的完整激活值（用于反向传播计算梯度）(2) 所有参数的梯度 (3) 优化器状态（如 Adam 的 $m, v$ 状态 + FP32 参数副本 = 12 bytes/参数）(4) 中间计算缓冲区。以 5M 参数模型为例，推理 ~20MB，训练 ~360MB（约 18×）。

**Q2: 梯度泄露攻击（DLG）的原理是什么？如何防御？**
A: DLG 通过优化随机虚拟输入，使虚拟输入产生的梯度与截获的真实梯度匹配，当两个梯度一致时，虚拟输入即为重建的原始输入。防御手段：(1) 差分隐私 SGD——在梯度上加高斯噪声 (2) 梯度压缩——只传 Top-k 或 1-bit 梯度 (3) 安全聚合（Secure Aggregation）——服务器只能看到聚合后的梯度 (4) 梯度裁剪——限制单样本梯度的最大范数。

**Q3: FedAvg 和传统分布式 SGD 的区别？**
A: 传统分布式 SGD 中，所有 worker 从同一个起点开始做一步 SGD，然后 AllReduce 梯度。FedAvg 中，每个客户端从全局模型开始做**多步**（多个 epoch）的本地 SGD，然后上传**权重差（Δw）**而非梯度。FedAvg 可以看作"多步本地更新 + 周期性同步"，通信效率更高但收敛可能不稳定（因为本地更新可能偏离全局方向——即 Non-IID 数据的 client drift 问题）。

**Q4: TinyTL 如何通过只更新 bias 来实现有效微调？**
A: 对于预训练的 MobileNet/ResNet，权重矩阵捕获了通用的视觉特征（边缘、纹理、形状），这些对新任务也是有用的。Bias 控制的是每层特征的"基线激活水平"——不同的任务（如从猫狗分类切到车辆分类）需要不同的激活模式，而调整 bias 可以显著改变激活分布。加上残差适配器提供轻量非线性变换，TinyTL 用 < 5% 的可训练参数达到了接近全量微调的效果。

**Q5: 如何评估稀疏反向传播对模型精度的影响？**
A: (1) 测量不同稀疏率下的验证集准确率曲线 (2) 确保稀疏率不超过某个临界点（通常 50-70% 稀疏率时精度下降 < 1%）(3) 验证收敛性：稀疏反向传播是否仍能达到与密集反向传播相同的最终 loss (4) 检查是否有某些层/模块被过度稀疏化导致梯度消失。

## 10. 本讲总结

端侧训练是高效深度学习的终极前沿——它使模型在部署后仍能持续学习和适应。本讲围绕三个核心挑战展开：

1. **内存瓶颈**：训练需要推理 ~18× 的内存（激活值 + 梯度 + 优化器状态）。解决方案包括 TinyTL（只微调 bias 和轻量适配器）、Sparse Back-Propagation（选择性反向传播）、PockEngine（动态评估层重要性）。

2. **隐私与安全**：联邦学习（FedAvg）让数据不出设备，但梯度本身可能泄露输入信息（DLG 攻击）。差分隐私提供理论保障，但需权衡精度。

3. **算法与系统的协同设计**：单独的内存优化或单独的通信优化都不够——需要在算法层面（稀疏训练、参数高效微调）和系统层面（激活值压缩、梯度检查点）同时进行优化。

端侧训练的最终愿景是：每个设备都有一个**持续学习、不断适配用户行为、完全保护隐私**的 AI 模型。当前技术离这个愿景还有距离，但 TinyTL、联邦学习和稀疏训练正在铺设通往终点的道路。
