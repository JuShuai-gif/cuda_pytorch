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

**真实案例分析**：

**案例 1：Google Gboard 联邦学习 — 百万设备规模的生产系统**
Google Gboard 是全球最大的联邦学习生产系统之一。系统架构：每台 Android 设备上运行一个轻量级 LSTM 模型（~2M 参数），用于预测下一个词。训练流程：每天凌晨设备充电 + WiFi 连接时，Gboard 从服务器拉取全局模型，使用本地输入历史（仅最近 24 小时的打字数据）做本地 SGD 训练（3-5 个 epoch），然后上传模型更新量 $\Delta w$ 到 Google 的 FL 服务器。服务器端使用 **FedAvg with Secure Aggregation** 聚合数千到数万个客户端的更新。关键工程细节：(1) **设备选择策略**：不是所有设备都参与每轮训练——Google 使用基于设备的"充电 + WiFi + 空闲"状态的筛选器，确保训练不影响用户体验；(2) **通信压缩**：$\Delta w$ 在上传前经过随机量化（stochastic rounding to INT8），将通信量从 ~8MB（FP32）降到 ~2MB；(3) **差分隐私**：每个 $\Delta w$ 在本地经过高斯噪声注入（$\sigma = 0.5$）后再上传，结合 Secure Aggregation（服务器只能看到聚合结果不能看到单用户更新），实现端到端隐私保护——即使 Google 内部员工也无法恢复单个用户的打字内容；(4) **Non-IID 处理**：不同用户的打字风格差异极大（英语 - 西班牙语混合、正式/非正式、年龄差异），Google 使用 **FedProx**（添加 proximal term 限制本地模型不要偏离全局模型太远）而非标准 FedAvg，有效缓解了 Non-IID 发散问题。截至 2024 年，Gboard FL 系统每天处理来自 5000 万+ 设备的更新，模型在离线个性化过程中从未有过用户数据泄露事件。

**案例 2：Apple Siri 唤醒词训练 — 联邦学习 + 差分隐私的隐私保护标杆**
Apple 在 WWDC 2019 上公布了 Siri 唤醒词的联邦学习系统，这是目前公开的最严格的隐私保护 + 端侧训练系统。核心挑战：Siri 的"Hey Siri"唤醒模型需要不断适应新的语音环境和口音，但语音数据是极其敏感的（包含私人对话内容）。Apple 的方案：(1) 设备端训练使用本地音频（当用户说"Hey Siri"时捕捉的 1-2 秒音频片段），不做上传；(2) 本地训练后，模型更新 $\Delta w$ 经过 **高斯差分隐私（$\epsilon = 2, \delta = 10^{-6}$）**处理——每个 update 被加噪到无法反推单个训练样本的程度；(3) 服务器端使用 **Secure Aggregation with Shamir Secret Sharing**——服务器在没有重构任何单个客户端的更新向量前提下，计算聚合后的更新。性能指标：参与设备数 100 万+，每轮通信量 < 5MB per device，模型准确率在 6 个月持续训练中提升了 15%（误唤醒率从 5% 降到 2%）。Apple 的经验：**FL+DP 的组合是可行的，但 $\epsilon$ 不能太小**——$\epsilon < 0.5$ 时模型完全不收敛，$\epsilon = 2$ 是一个好的 engineering sweet spot（平衡隐私和精度）。

**案例 3：NVIDIA FLARE — 医疗领域的跨机构联邦学习**
NVIDIA FLARE（Federated Learning Application Runtime Environment）是面向医疗和企业级联邦学习的框架。一个典型案例：多家医院联合训练 X 光片分类模型（COVID-19 检测）。每家医院的 PACS 系统中存储了数万到数十万张 X 光片，但由于 HIPAA/GDPR 隐私法规不能共享原始数据。使用 FLARE：(1) 每家医院在本地（on-premise GPU 集群）训练 ResNet-50 模型；(2) 每轮训练后，模型权重（而非梯度）通过 TLS 加密上传到中央 FL 服务器；(3) 服务器执行 FedAvg 聚合后分发新模型。关键挑战：**不同医院的 X 光片质量差异极大**（不同设备、不同曝光参数、不同标注标准），Non-IID 问题非常严重。FLARE 的解决方案：(1) 使用 **FedProx** 而非 FedAvg，通过 $\mu \|\mathbf{w} - \mathbf{w}_t\|^2$ 正则项防止本地模型漂移；(2) **联邦数据增强**：服务器生成合成数据（通过 StyleGAN）发送给数据量小的医院，补充其本地训练；(3) 模型聚合时使用 **加权 FedAvg**——数据质量高的医院（由本地验证集 AUC 衡量）获得更高的聚合权重。最终效果：联邦训练的模型 AUC（0.94）接近集中式训练（0.95），且没有违反任何隐私法规。

**案例 4：TinyTL 在手机端个性化图像分类 — 10× 内存节省的实战**
一家移动应用公司在他们的照片整理 App 中使用 TinyTL 做个性化场景分类（用户自己的照片分类：家庭、工作、旅行等）。技术栈：预训练 MobileNetV3（5.4M 参数，95MB），使用 TinyTL 方法——冻结所有权重矩阵，仅训练 bias + 新增的轻量残差适配器（每层 2 个 1×1 conv + ReLU，总可训练参数 ~200K）。在 iPhone 14（6GB RAM）上：(1) 传统微调方案（更新分类头 1.2M 参数 + backbone 最后 3 层）峰值内存 ~850MB；TinyTL 方案峰值内存 ~120MB（7× 节省）；(2) 50 张用户照片的训练时间：传统方案 12s，TinyTL 方案 2.5s；(3) 分类准确率：传统方案 91.2%，TinyTL 方案 89.8%（差距 1.4%——用户完全感知不到）。关键教训：**bias 的初始值非常关键**——TinyTL 中将 bias 初始化为预训练模型的原始值（而非零初始化），否则前几个 epoch 的激活分布会剧烈变化导致训练不稳定。

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

### 生产环境 P0 级故障实录

7. **"FedAvg 在 Non-IID 客户端数据上发散 → 全局模型完全不可用"** — 这是联邦学习在生产环境中最致命的 P0 故障。当客户端数据分布高度不同时（例如：一个用户的照片全是狗，另一个用户的全是猫），每个客户端本地训练的模型会严重漂移（client drift）——本地模型朝着各自的局部最优方向更新，聚合后的全局模型在两个方向上来回震荡，最终收敛到无用状态。真实案例：某手机厂商在 10 万设备上做联邦情感分类训练，用户 A 发的大多是正面消息（朋友圈），用户 B 发的大多是负面消息（吐槽群），5 轮 FedAvg 后全局模型准确率从 85% 跌到 52%（随机猜测水平）。解决方案：(1) **FedProx**：在本地 loss 中添加 proximal term $\frac{\mu}{2}\|\mathbf{w} - \mathbf{w}_t\|^2$，防止本地模型偏离全局模型太远，$\mu$ 通常设为 0.001-0.1；(2) **SCAFFOLD**：服务器维护一个全局"修正方向"（control variate），客户端在本地训练时用修正方向校正 client drift——但需要客户端存储额外的 state 向量，增加了端侧内存负担；(3) **FedNova**：在聚合时根据客户端的本地更新步数进行归一化，消除不同客户端训练步数差异带来的偏差。**关键教训：FedAvg 的简单加权平均假设了所有客户端在同一个 loss landscape 内优化——Non-IID 打破了这个假设**。

8. **"联邦学习中的 DP 噪声过大 → 模型完全不收敛"** — 差分隐私（DP-SGD）在联邦学习中是"必须品"（保护用户隐私），但噪声 $\sigma$ 和隐私预算 $\epsilon$ 的选择极其关键。生产经验：(1) $\epsilon < 0.5$ 时，噪声水平 $\sigma$ 通常超过梯度本身的幅度，模型根本无法学习；(2) $\epsilon = 2$ 是一个 engineering sweet spot——在百万级客户端下，单个用户的隐私得到较好保护，同时模型可正常收敛；(3) 噪声裁剪（per-sample gradient clipping）的阈值 $C$ 的选择比 $\sigma$ 的选择更关键——$C$ 太小，信息丢失严重；$C$ 太大，噪声需求增大。Apple 的经验是使用 **adaptive clipping**：根据每轮梯度的百分位数动态调整 $C$（如设为第 90 百分位数），而非固定值。一个 P0 事故：某团队将 clipping threshold 设为固定值 0.1，但训练后期梯度范数已降到 0.01 以下，导致几乎所有梯度被裁剪掉，模型完全停滞。

9. **"端侧训练中内存估算不足 → App 被系统 OOM Killer 杀死"** — 移动端的内存环境与服务器完全不同。iOS 对单个 App Extension（如键盘扩展）的内存限制仅为 50-100MB（视设备而定），Android 后台服务限制类似。即使 TinyTL 将训练内存从 850MB 降到 120MB，仍然可能超出 iOS 键盘扩展的 48MB 限制（iPhone 8 及更早型号）。真实教训：(1) 必须使用 `vmmap` 和 Xcode Memory Graph 工具分析实际内存占用——`torch.cuda.memory_allocated()` 只报告 PyTorch 分配的内存，不包括系统开销（iOS Metal 驱动 buffer、模型文件 mmap 等）；(2) 量化 + TinyTL 组合：先 INT8 量化模型（模型内存 4×↓），再使用 TinyTL（可训练参数 50×↓），组合后训练峰值内存可降至 ~30MB；(3) **分批加载**：将模型按层拆分，前向计算第 $i$ 层时只加载第 $i$ 层和相邻层的参数到内存，其余层保持在 Flash 存储上——用 I/O 换内存。

10. **"梯度泄露攻击（DLG）在生产环境中的实际威胁被低估"** — 学术界常认为 DLG 需要"知道真实标签"等强假设才能工作。但实际的攻击手段更灵活：(1) **iDLG（Improved DLG）**：可以从梯度中直接推断标签（分类任务中，ground-truth label 对应梯度向量中唯一为负的分量），不需要假设已知标签；(2) **Inverting Gradients (IG)**：使用余弦相似度而非 L2 距离作为优化目标，对高分辨率图片的重建质量远超 DLG；(3) 即使在 FL 中使用 Secure Aggregation（服务器看不到单个梯度），**恶意的客户端**可以在收到全局模型后通过观察模型更新来推断其他客户端的数据特征（成员推断攻击）。防御方面的教训：**仅靠 Secure Aggregation 不够**，必须在本地训练中就注入 DP 噪声——"先加噪声，再安全聚合"。

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

**Q6: FedAvg、FedProx 和 SCAFFOLD 三者在 Non-IID 场景下的收敛行为有何本质不同？各自的适用场景是什么？**（高级联邦学习面试）

A: 三者的核心差异在于如何处理 **client drift**——客户端本地训练导致的模型偏离全局方向。

**FedAvg（基线）**：
- 机制：客户端本地做多步 SGD，服务器加权平均模型权重变化量
- Non-IID 行为：当客户端数据分布差异大时，本地模型的更新方向 $\Delta \mathbf{w}_k$ 指向各自的局部最优，聚合后的全局模型是这些不同方向的"折中"。如果各方向之间夹角 > 90°（严重 Non-IID），折中后的更新可能停滞不前——这就是 divergence 的数学本质
- 适用：客户端数据 IID 或轻微 Non-IID，客户端数量 $\gg$ 数据分布类别数（大数据量客户端稀释了 Non-IID 效应）

**FedProx**：
- 机制：在 FedAvg 的本地 loss 中添加 proximal term：$\mathcal{L}_k(\mathbf{w}) + \frac{\mu}{2}\|\mathbf{w} - \mathbf{w}_t\|^2$
- Non-IID 行为：proximal term 像一根"弹性绳"，将本地模型拴在全局模型附近——本地模型可以偏离，但不能偏离太远。$\mu$ 控制弹性强度：$\mu \to 0$ → FedAvg，$\mu \to \infty$ → 本地不训练
- 关键创新：允许客户端执行**不同步数的本地更新**（统计异质性，systems heterogeneity）——计算快的客户端可以多做几步，计算慢的可以少做几步，proximal term 确保步数差异不会导致不公平的聚合
- 适用：客户端计算能力异构 + 数据 Non-IID 的场景（如手机 + 平板 + IoT 设备混合联邦学习）
- 局限：$\mu$ 需要手动调参，且对所有客户端一视同仁——但不同客户端的"偏离程度"天然不同

**SCAFFOLD**：
- 机制：服务器维护全局 control variate $\mathbf{c}$（修正方向），客户端维护本地 control variate $\mathbf{c}_k$。本地更新时：$\mathbf{w}_{k}^{(t+1)} = \mathbf{w}_k^{(t)} - \eta(\mathbf{g}_k - \mathbf{c}_k + \mathbf{c})$，其中 $\mathbf{c}_k - \mathbf{c}$ 项纠正了 client drift
- Non-IID 行为：control variate 追踪了"本地梯度方向与全局梯度方向的系统性偏差"，并在每次本地更新时消除这个偏差。数学上，SCAFFOLD 的收敛率与 Non-IID 程度**无关**（在强凸假设下），这是 FedProx 无法做到的
- 代价：每个客户端需要存储一个与模型等大的 control variate state 向量——对于 5M 参数模型，这是额外 ~20MB 内存（FP32），在手机端可能是不可接受的
- 适用：客户端有能力存储额外 state、数据 Non-IID 严重的场景（如跨机构医疗 FL）
- 局限：control variate 的方差在高维参数空间中很大，需要大量客户端才能稳定估计

**面试完整答案**：如果客户端资源充足且 Non-IID 严重，SCAFFOLD 理论最优；如果客户端异构且 Non-IID 中等，FedProx 最实用；如果数据接近 IID，FedAvg 最简单有效。**工程上常用的是 FedProx，因为它不需要客户端存储额外 state，且 $\mu$ 的调参直觉清晰**。

**Q7: 联邦学习中差分隐私（DP）的 $\epsilon$ 和客户端数量如何影响模型精度？为什么百万级用户的 DP-FL 比千级用户的 DP-FL 更容易训练？**（隐私 + ML 交叉面试）

A: 差分隐私在 FL 中的噪声分为两级：(1) 客户端本地的 per-sample DP 噪声（保护单个训练样本的隐私）；(2) 服务器端的聚合隐私（保护单个客户端的贡献不被泄露）。两者的关系是累积的。

**$\epsilon$ 与噪声的关系**：DP-SGD 中，梯度噪声的标准差 $\sigma = \frac{C \cdot \sqrt{2 \log(1.25/\delta)}}{\epsilon}$，其中 $C$ 为 clipping threshold。$\epsilon$ 越小 → $\sigma$ 越大 → 噪声越大 → 精度越低。

**客户端数量的"稀释"效应**：FL 聚合时，$N$ 个客户端的 DP 噪声在聚合后以 $1/N$ 的速率衰减（因为噪声是独立同分布的）：
$$\text{Var}[\text{Aggregated Noise}] = \frac{\text{Var}[\text{Individual Noise}]}{N}$$

**具体数字对比**：

| 场景 | $N=1000$ | $N=10^6$ |
|------|---------|---------|
| $\epsilon=2$ 需要的 $\sigma$（per-user） | ~5.0 | ~5.0 |
| 聚合后有效噪声 $\sigma_{\text{eff}}$ | $5.0/\sqrt{1000} \approx 0.158$ | $5.0/1000 \approx 0.005$ |
| 对梯度信噪比的影响 | 中等（~6% 精度损失） | 极小（<0.1% 精度损失） |

这就是为什么 Google Gboard（5000 万+ 用户）和 Apple Siri（100 万+ 用户）可以做 DP-FL 且精度几乎无损——**海量客户端的统计效应自然稀释了 DP 噪声**。但医疗 FL（10-100 家医院，每家医院是一个"客户端"）中，$N$ 太小，DP 噪声无法被有效稀释。NVIDIA FLARE 的解决方案是使用**客户端级别的 subsampling**——不是所有医院每轮都参与，而是随机抽样——subsampling 引入了额外的隐私放大效应（Privacy Amplification by Subsampling），可以将有效 $\epsilon$ 降低 3-5×。

**Q8: 端侧训练的能耗预算是多少？在 iOS 后台训练一个 TinyML 模型（5M 参数，10 epoch，100 samples），是否会触发系统的后台任务限制？**（系统工程面试）

A: 这是一道典型的"从理论翻到工程翻到系统"的综合题。

**能耗分析**：
- 推理（前向）：5M 参数 × 2 FLOPs/param × 100 samples × 10 epochs = 10G FLOPs
- 训练（前向 + 反向）：反向传播约为前向的 2× FLOPs → 总计 10G × 3 = 30G FLOPs
- A15 Bionic (iPhone 13) 的 Neural Engine：15.8 TOPS (INT8)，能效比 ~6 TOPS/W
- 30G FLOPs / 15.8 TOPS ≈ **1.9ms（纯计算）**，但实际受内存带宽限制约需 50-100ms
- 功耗：~2W (ANE + CPU + DRAM)，持续 100ms → 能耗约 0.056 mWh

表面上看微不足道，但问题在系统层：

**iOS 后台限制**：
1. **Background Task 时间限制**：iOS 给 App 的后台执行时间通常只有 30 秒（`beginBackgroundTask`），之后系统会挂起进程。10 epoch × 100 samples 的训练如果每 sample 300ms（包括 I/O 等），总时间 300s = 5 分钟，**远超 30 秒限制**。
2. **功耗预算（Energy Budget）**：iOS 的后台任务功耗预算极低（< 1% 电池/小时）。如果充电 + WiFi 时训练，功耗预算更宽松。
3. **热限制（Thermal Throttling）**：连续 5 分钟的神经网络计算会导致 SoC 升温，触发 DVFS（降频），训练时间可能膨胀到 10 分钟。

**解决方案**—这是 Google Gboard 和 Apple Siri 实际使用的策略：
1. **分片训练**：将 10 epoch 拆成 10 个 30 秒的后台任务，每个任务间间隔 5 分钟（让 SoC 冷却 + 不触发系统 watchdog）
2. **触发条件**：只在"充电 + WiFi + 屏幕关闭 + 电量 > 80%"时启动训练——这些条件确保功耗预算和网络连接
3. **提前停止**：监控电池温度和电量，SoC 温度 > 40°C 或电量 < 70% 时立即挂起训练并保存 checkpoint
4. **模型量化**：训练时使用 INT8 量化模型（不是 FP32），训练 FLOPs 减少 4×，内存带宽需求减半

**Q9: 梯度泄露攻击（DLG/iDLG/IG）的防御手段中，哪种最有效？梯度裁剪、DP 噪声、梯度压缩、Secure Aggregation 各自的攻防边界是什么？**（安全 + ML 交叉面试）

A: 四类防御手段的原理和效果：

**(1) 梯度裁剪（Gradient Clipping）**：
- 防御原理：限制每样本梯度的最大范数，防止异常样本产生过大的梯度信号
- 攻击绕过：DLG/iDLG 优化的是"相对形状"而非"绝对幅度"——裁剪后的梯度虽然范数变小，形状信息仍然保留，攻击者仍然可以重建输入
- 有效性：**弱**（单独使用几乎无效，只在与其他技术组合时才贡献一份力）

**(2) 差分隐私噪声（DP-SGD）**：
- 防御原理：在梯度上注入高斯噪声，使得任何单个训练样本对梯度的影响在统计上不可区分
- 效果：当 $\epsilon < 8$ 时，DLG 重建的输入与真实输入的 PSNR 通常 < 10dB（人眼完全无法辨认）；$\epsilon < 2$ 时 PSNR < 5dB（基本是随机噪声）——**这是目前最强且唯一有理论保证的防御**
- 代价：模型精度下降（$\epsilon=2$ 时通常下降 2-5% accuracy）

**(3) 梯度压缩（DGC/1-Bit SGD）**：
- 防御原理：只传输部分梯度（Top-k 或 1-bit sign），攻击者只能看到不完整的梯度
- 攻击绕过：iDLG 已被证明可以从 DGC 的 Top-1% 梯度中重建输入（因为被选中的梯度往往是最大幅度的，包含最多信息）。1-Bit SGD 安全性更高——只保留符号信息，但符号本身仍然可以泄露标签信息
- 有效性：**中等**（可以防御简单的 DLG 但无法防御专门针对压缩梯度设计的攻击）

**(4) Secure Aggregation（安全聚合）**：
- 防御原理：服务器只能看到聚合后的梯度 $\sum_{k} \mathbf{g}_k$，无法访问单个客户端的梯度
- 效果：直接阻断了 DLG（因为 DLG 需要单个客户端的精确梯度来做匹配优化）
- 攻击绕过：**不防御恶意服务器（honest-but-curious server 是 SA 的标准威胁模型）**，且不防御"聚合后"的信息泄露——聚合梯度仍然可能泄露数据集的统计特性（如属性分布、类别分布）
- 有效性：**强（对单客户端攻击）但有限（对聚合后攻击）**

**最优组合**：生产系统中（如 Apple Siri），防御策略是**DP-SGD（本地加噪）+ Secure Aggregation（传输中保护）**的双层防护。DP 在本地消除样本级信息，SA 确保服务器无法访问单用户更新。这套组合是目前已知的最强 FL 隐私保护方案，在百万级客户端下 $\epsilon=2$ 即可同时保证隐私和精度。

## 10. 本讲总结

端侧训练是高效深度学习的终极前沿——它使模型在部署后仍能持续学习和适应。本讲围绕三个核心挑战展开：

1. **内存瓶颈**：训练需要推理 ~18× 的内存（激活值 + 梯度 + 优化器状态）。解决方案包括 TinyTL（只微调 bias 和轻量适配器）、Sparse Back-Propagation（选择性反向传播）、PockEngine（动态评估层重要性）。

2. **隐私与安全**：联邦学习（FedAvg）让数据不出设备，但梯度本身可能泄露输入信息（DLG 攻击）。差分隐私提供理论保障，但需权衡精度。

3. **算法与系统的协同设计**：单独的内存优化或单独的通信优化都不够——需要在算法层面（稀疏训练、参数高效微调）和系统层面（激活值压缩、梯度检查点）同时进行优化。

端侧训练的最终愿景是：每个设备都有一个**持续学习、不断适配用户行为、完全保护隐私**的 AI 模型。当前技术离这个愿景还有距离，但 TinyTL、联邦学习和稀疏训练正在铺设通往终点的道路。

## 11. 工业落地checklist

| 检查项 | 说明 | 不做的后果 |
|--------|------|-----------|
| 联邦学习 Non-IID 场景不能用纯 FedAvg，必须升级为 FedProx 或 SCAFFOLD | 某手机厂商 10 万设备 FL：用户 A 只发正面消息、用户 B 只发负面 → FedAvg 5 轮后全局模型准确率从 85%→52%（随机水平） | 联邦训练后的全局模型完全不可用，百万设备参与的训练白费 |
| 差分隐私 FL 的 ε 不能设太小（< 0.5），百万级客户端是 DP 噪声自然稀释的前提 | Apple Siri 经验：ε < 0.5 模型完全不收敛；ε = 2 是 engineering sweet spot；医疗 FL 中 N=10-100 无法稀释 DP 噪声 | ε 太小=模型无法学习，ε 太大=隐私泄露风险——两难困境中选错方向 |
| 端侧训练内存估算不能用 torch.cuda.memory_allocated()，必须用 vmmap/Xcode Memory Graph | iOS 键盘扩展内存限制仅 48MB（iPhone 8），PyTorch 报告 120MB 可能实际占用 180MB（含 iOS Metal 驱动 buffer）→ OOM kill | App 被系统杀死，用户体验极差，crash rate 飙升 |
| TinyTL 中 bias 的初始化必须保持预训练原始值，不能零初始化 | 某照片 App 将 TinyTL bias 零初始化 → 前几个 epoch 激活分布剧烈变化 → 训练不稳定，精度比保持原始 bias 低 3% | "省内存"的优化变成了精度退化，用户感知个性化效果差 |
| 联邦学习的 DP 噪声裁剪阈值 C 必须用 adaptive clipping（如第 90 百分位数）而非固定值 | 某团队固定 C=0.1，训练后期梯度范数已 <0.01 → 几乎所有梯度被裁剪掉 → 模型完全停滞 | 联邦训练"正常运行"但模型精度不再提升，浪费数周时间 |
| 梯度泄露攻击（DLG）的防御不能仅靠 Secure Aggregation——必须在本地先注入 DP 噪声 | SA 只保护传输过程，恶意客户端仍可通过观察全局模型更新推断其他用户数据特征（成员推断攻击） | 以为"安全聚合就是安全的"，实则用户隐私仍可被推断——合规风险 |
| 移动端后台训练必须分片（每次 30s）+ 仅在充电+WiFi+电量>80% 时启动 | iOS Background Task 限制 30 秒，连续 5 分钟训练会被系统挂起；SoC 温度 > 40°C 触发降频 | 训练未完成即被系统杀死，checkpoint 损坏，用户永远看不到个性化效果 |
