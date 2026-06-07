# Lecture 22: 课程总结 — 高效深度学习全景回顾

## 1. 本讲核心问题

MIT 6.5940 贯穿一个核心问题：**如何在有限资源下（计算、内存、带宽、功耗）训练和部署最有效的深度学习模型？** 本讲总结：

1. **高效推理技术全景**：剪枝、量化、神经架构搜索（NAS）、知识蒸馏
2. **高效训练技术全景**：分布式训练、端侧训练、混合精度
3. **特定应用优化**：LLM、Vision Transformer、扩散模型
4. **算法 + 系统协同设计（Algorithm-System Co-design）**：本课程的核心哲学
5. **EE 与 CS 的跨领域连接**：硬件意识（hardware-aware）的算法设计

## 2. 通俗解释

如果把深度学习比作"造车"，本课程教的就是"如何造一辆省油又跑得快的车"：

- **剪枝 = 减重**：把车上没用的零件（接近 0 的权重）拆掉，车子更轻，跑得更快
- **量化 = 用低精度零件**：把 32 位高精度螺丝换成 8 位够用的螺丝，成本更低，装配更快
- **NAS = 自动设计车身**：让算法自动搜索最佳车身结构（网络架构），而不是靠工程师手工设计
- **蒸馏 = 老司机带新手**：让大模型（老师）教小模型（学生），学生学会老师的"驾驶技巧"
- **分布式训练 = 流水线生产**：一辆车由多个工人同时组装，并行制造
- **端侧训练 = 车辆自学习**：车在路上跑的时候自己学会适应路况，不需要回工厂重造

**核心理念：算法 + 系统协同设计**——你不能单独优化算法（如设计一个低精度友好的训练方法）或单独优化系统（如购买更快的内存），而是需要**两者一起设计**。比如：你设计了 4-bit 量化方案，但硬件没有 4-bit 乘法器，那就需要软硬件协同——算法端用 group-wise quantization 来适配硬件，硬件端提供 INT4 矩阵乘法单元。

**EE 与 CS 的连接**：传统上，EE（电子工程）关注硬件效率（芯片设计、功耗优化），CS（计算机科学）关注算法效率（复杂度、收敛率）。本课程将两者融合——每个算法的设计都要考虑硬件的真实约束（内存层次结构、算术强度、通信带宽），每个硬件的设计也要考虑算法的真实需求（稀疏性支持、低精度支持）。

## 3. 关键公式

**四大推理优化技术的统一视角**：

剪枝（权重稀疏化）：
$$
\tilde{\mathbf{W}} = \mathbf{W} \odot \mathbf{M}, \quad \|\mathbf{M}\|_0 = k
$$
目标：减少 $\|\mathbf{W}\|_0$ 的同时最小化精度损失 $\mathcal{L}(\tilde{\mathbf{W}}) - \mathcal{L}(\mathbf{W})$

量化（低精度表示）：
$$
\hat{\mathbf{W}} = s \cdot \text{round}\left(\frac{\text{clamp}(\mathbf{W}, c_{\min}, c_{\max})}{s}\right), \quad s = \frac{c_{\max} - c_{\min}}{2^b - 1}
$$
通信/存储压缩比 $= \frac{32}{b}$

NAS 的双层优化：
$$
\alpha^* = \arg\min_\alpha \mathcal{L}_{\text{val}}(w^*(\alpha), \alpha)
$$
$$
\text{s.t. } w^*(\alpha) = \arg\min_w \mathcal{L}_{\text{train}}(w, \alpha)
$$

知识蒸馏（KD）：
$$
\mathcal{L}_{\text{KD}} = \mathcal{L}_{\text{CE}}(y, \hat{y}) + \lambda \cdot T^2 \cdot \text{KL}\left(\sigma\left(\frac{z_t}{T}\right), \sigma\left(\frac{z_s}{T}\right)\right)
$$

**分布式训练的统一计算模型**：
$$
T_{\text{total}} = T_{\text{compute}} + T_{\text{communication}} = \frac{C_{\text{model}}}{P \cdot F} + \frac{K_{\text{model}}}{B_{\text{effective}}}
$$
其中 $C_{\text{model}}$ 为总计算量，$P$ 为 GPU 数，$F$ 为单 GPU 算力，$K_{\text{model}}$ 为通信量，$B_{\text{effective}}$ 为有效带宽

**Arithmetic Intensity（算术强度）与 Roofline 模型**：
$$
\text{AI} = \frac{\text{FLOPs}}{\text{Bytes Moved}}
$$
当 $\text{AI} > \frac{\text{Peak FLOPS}}{\text{Peak Bandwidth}}$ 时，算子为**计算瓶颈（compute-bound）**；否则为**访存瓶颈（memory-bound）**。

**端侧效率的全系统方程**：
$$
\text{Efficiency} = \frac{\text{Accuracy}}{\text{Latency} \times \text{Energy} \times \text{Memory} \times \text{Bandwidth}}
$$
目标：在给定的资源约束下最大化精度

## 4. 公式背后的直觉

- **剪枝和量化为什么能 work**：深度学习模型天然存在**过参数化（over-parameterization）**——参数数量远超学习任务所需。这就像用 100 个参数拟合一条直线，其中 98 个是冗余的。剪枝和量化就是**识别并移除冗余**，无论是结构性冗余（整个通道都不重要）还是数值冗余（32 位精度远超所需）。

- **Arithmetic Intensity 的实际意义**：卷积层的 AI 通常为 100-1000 FLOPs/Byte（计算密集型），全连接层的 AI 通常为 50-200，而 Element-wise 操作（ReLU, Add）的 AI 可能只有 1-5（访存密集型）。这意味着：**激活函数（如 ReLU）不是算不过来的问题，是数据搬不过来的问题**——优化它的关键是减少内存访问（如算子融合），而不是买更快的 GPU。

- **知识蒸馏中温度 $T$ 的作用**：$T > 1$ 时，softmax 输出变得更"软"——非目标类别的概率变大。这提供了**类间相似性的信息**（如"3"这个数字，不仅正确答案是"3"，老师还告诉你"它长得有点像 8"）。这种"暗知识"（dark knowledge）是蒸馏的核心价值。

- **端侧效率方程的"乘积"特性**：注意分母是乘积而非加法——这意味着任何一个维度变差，整体效率就急剧下降。一个精度 99% 但延迟 10 秒的模型，和一个精度 90% 但延迟 10ms 的模型，后者的实际可用性更高。**只追求精度而忽视效率是学术界常见的陷阱**。

- **协同设计的必要性**：假设你花一个月优化了一个极好的 4-bit 量化方案，但目标硬件只能做 INT8 矩阵乘法。在软件层模拟 4-bit（pack/unpack）的开销完全抵消了理论收益。反过来，如果你设计了一个支持 4-bit 的硬件，但没有相应的训练算法，模型精度会大幅下降。**软硬件必须同时设计**——这就是协同设计的本质。

## 5. 工业界用途

### 5.1 技术谱系总览

| 阶段 | 技术 | 资源节省 | 代表应用 |
|------|------|---------|---------|
| 推理优化 | 剪枝 | 模型大小 10-90% ↓ | MobileNet + 剪枝 → 手机端实时推理 |
| 推理优化 | 量化 | 模型大小 4× ↓, 延迟 2-4× ↓ | INT8 BERT → 毫秒级 NLP 推理 |
| 推理优化 | NAS | 自动找到 Pareto 最优架构 | EfficientNet, MobileNetV3 |
| 推理优化 | 蒸馏 | 小模型获得大模型精度 | DistilBERT (6层 = BERT 12层精度的 95%) |
| 训练优化 | 混合精度 (FP16/BF16) | 训练速度 2-3× ↑, 内存 50% ↓ | A100/H100 训练的默认配置 |
| 训练优化 | ZeRO/FSDP | 支持 175B+ 模型训练 | GPT-3, LLaMA 系列 |
| 训练优化 | 梯度压缩 | 通信量 270-600× ↓ | 低带宽集群训练 |
| 端侧优化 | TinyTL + 联邦学习 | 端侧训练内存 5-10× ↓ | 手机键盘预测、个性化 |

### 5.2 跨应用洞察

**LLM（大语言模型）**：
- 瓶颈：KV Cache 内存（自回归推理时存储所有历史 Key/Value）
- 优化：Multi-Query Attention（多查询注意力，多 head 共享 KV）、GQA（分组查询注意力）、KV 量化（INT8/INT4 KV cache）
- 部署：vLLM（PagedAttention 管理 KV cache 碎片）、Speculative Decoding（用小模型预测 + 大模型验证加速自回归）

**Vision Transformer（视觉 Transformer）**：
- 瓶颈：$O(N^2)$ 的注意力复杂度（$N$ 为 patch 数量）
- 优化：Swin Transformer（窗口注意力）、Token Merging（合并相似 token）、稀疏注意力
- 量化挑战：ViT 的 LayerNorm + GELU 的激活值分布特殊（值与值之间差异大），需专门校准

**扩散模型**：
- 瓶颈：1000 步迭代 + 每次 UNet 前向
- 优化：LCM 蒸馏（1-4 步）、DDIM（50 步）、潜空间操作（Latent Diffusion）、UNet 量化 + 剪枝
- 部署挑战：移动端实时扩散生成（< 2 秒）仍是 open problem

### 5.3 产业实践

- **NVIDIA TensorRT**：推理优化框架，集成了层融合、kernel 自动调优、INT8/FP16 量化、动态形状支持
- **Qualcomm AI Engine**：移动端推理，支持 INT4/INT8/FP16 混合精度，针对 Snapdragon Adreno GPU 和 Hexagon DSP 做后端正交优化
- **Apple CoreML + ANE**：Apple Neural Engine 专门加速 CoreML 模型，支持 FP16 推理和训练
- **Google TPU + XLA**：TPU v5p 支持 BF16 训练和 INT8 推理，XLA 编译器自动优化算子融合和内存布局
- **PyTorch 2.0 `torch.compile`**：通过 `torch.compile` + `Inductor` 后端实现自动 kernel 融合和代码生成

## 6. PyTorch 实现思路

```python
# ====================== 统一的高效推理管道 ======================
class EfficientInferencePipeline:
    """从预训练模型到部署最优模型的完整管道"""
    def __init__(self, pretrained_model):
        self.model = pretrained_model

    def step1_prune(self, sparsity=0.5):
        """步骤 1: 非结构化剪枝（移除权重）"""
        import torch.nn.utils.prune as prune
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=sparsity)
                prune.remove(module, 'weight')  # 永久化剪枝掩码

    def step2_quantize(self, backend='fbgemm'):
        """步骤 2: 训练后量化 INT8"""
        self.model.eval()
        self.model.qconfig = torch.ao.quantization.get_default_qconfig(backend)
        self.model = torch.ao.quantization.prepare(self.model)

        # 校准（需要少量代表性数据）
        for data in calibration_dataloader:
            self.model(data)

        self.model = torch.ao.quantization.convert(self.model)

    def step3_distill(self, teacher_model, student_model, dataloader, T=4.0):
        """步骤 3: 知识蒸馏（大模型教小模型）"""
        teacher_model.eval()
        student_model.train()

        for data, labels in dataloader:
            with torch.no_grad():
                teacher_logits = teacher_model(data)

            student_logits = student_model(data)

            # 硬标签损失
            hard_loss = F.cross_entropy(student_logits, labels)
            # 软标签损失（KL 散度）
            soft_loss = F.kl_div(
                F.log_softmax(student_logits / T, dim=1),
                F.softmax(teacher_logits / T, dim=1),
                reduction='batchmean'
            ) * (T * T)

            total_loss = 0.5 * hard_loss + 0.5 * soft_loss
            total_loss.backward()
            optimizer.step()

    def step4_export_torchscript(self, example_input):
        """步骤 4: 导出为 TorchScript 用于移动端部署"""
        traced = torch.jit.trace(self.model, example_input)
        traced.save("efficient_model.pt")
        return traced

# ====================== Arithmetic Intensity 分析 ======================
def analyze_arithmetic_intensity(module, input_shape):
    """分析算子的算术强度（FLOPs / Bytes）"""
    total_flops = 0
    total_bytes = 0

    for name, param in module.named_parameters():
        params = param.numel()
        # 参数从内存读到计算单元
        total_bytes += params * param.element_size()
        # 权重更新时写回
        total_bytes += params * param.element_size()

    # 估算激活值内存（粗略）
    activation_bytes = input_shape.numel() * 4  # FP32

    # FLOPs 估算（以 Conv2d 为例）
    if isinstance(module, nn.Conv2d):
        flops_per_elem = 2 * module.kernel_size[0] * module.kernel_size[1] * module.in_channels
        total_flops = flops_per_elem * module.out_channels * input_shape[-2] * input_shape[-1]

    ai = total_flops / (total_bytes + activation_bytes)
    print(f"Arithmetic Intensity: {ai:.1f} FLOPs/Byte")
    # AI > 100: Compute-bound (GPU 优化)
    # AI < 10: Memory-bound (算子融合/量化为关键)

    return ai

# ====================== 跨课程技术的协同应用示例 ======================
class HardwareAwareLLMInference:
    """
    综合运用课程中多种技术：量化 + KV Cache 压缩 + 推测解码
    """
    def __init__(self, model, quant_config='int8', kvcache_config='int4'):
        self.model = model
        self.quant_config = quant_config
        self.kv_cache = {}  # Key-Value 缓存压缩存储

    def prefill_phase(self, prompt_tokens):
        """预填充阶段：处理输入 prompt，生成初始 KV Cache"""
        with torch.no_grad():
            hidden_states = self.model.embed(prompt_tokens)
            for layer_idx, layer in enumerate(self.model.layers):
                # 量化感知注意力计算
                q, k, v = layer.self_attn(hidden_states)
                # KV Cache 量化存储（INT4）
                self.kv_cache[layer_idx] = (
                    self._quantize_kv(k, bits=4),
                    self._quantize_kv(v, bits=4)
                )
                hidden_states = layer(hidden_states, use_cache=True)

    def speculative_decode_step(self, draft_model, n_spec=5):
        """推测解码：小模型生成 n_spec 个候选 token，大模型验证"""
        # 小模型（draft）快速生成多个候选
        draft_tokens = draft_model.generate(n_tokens=n_spec)

        # 大模型一次前向验证所有候选
        with torch.no_grad():
            logits = self.model.forward_with_kv_cache(draft_tokens, self.kv_cache)

        # 接受与 draft 模型一致的 token
        accepted = self._verify_tokens(draft_tokens, logits)
        return accepted

    def _quantize_kv(self, tensor, bits=4):
        """KV Cache 量化：减少自回归生成的内存瓶颈"""
        qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        scale = tensor.abs().max() / qmax
        return (tensor / scale).round().clamp(qmin, qmax).to(torch.int8), scale

    def _verify_tokens(self, draft_tokens, target_logits):
        """验证 draft 模型生成的 token"""
        accepted = []
        for i, token in enumerate(draft_tokens):
            target_probs = F.softmax(target_logits[i], dim=-1)
            draft_prob = 1.0  # draft 模型确定性输出
            target_prob = target_probs[token].item()
            # Rejection sampling
            if random.random() < min(1.0, target_prob / draft_prob):
                accepted.append(token)
            else:
                # 拒绝后从调整后的分布采样
                adjusted_probs = F.relu(target_probs - F.softmax(draft_logits[i]))
                accepted.append(torch.multinomial(adjusted_probs, 1))
                break
        return accepted

# ====================== Roofline 分析与优化指导 ======================
def roofline_guided_optimization(model, input_tensor, hardware_specs):
    """
    基于 Roofline 模型决定优化策略
    hardware_specs: {'peak_flops': 312e12, 'peak_bandwidth': 2000e9}
    """
    from fvcore.nn import FlopCountAnalysis, parameter_count

    flops = FlopCountAnalysis(model, input_tensor).total()
    params = parameter_count(model)['']
    act_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) * 3  # 粗略估计：参数 + 梯度 + 激活值

    ai = flops / act_bytes
    ridge_point = hardware_specs['peak_flops'] / hardware_specs['peak_bandwidth']

    if ai > ridge_point:
        return "Compute-bound: 提高并行度、减少 FLOPs (剪枝、轻量架构)"
    else:
        return "Memory-bound: 算子融合、量化、减少内存访问 (in-place ops)"

# ====================== 课程技术速查表 ======================
TECHNIQUE_CHEATSHEET = {
    # 推理
    "剪枝": {"节省": "10-90% 参数", "精度损失": "< 1%", "工具": "torch.nn.utils.prune"},
    "量化 (INT8)": {"节省": "4× 存储/带宽", "精度损失": "< 1%", "工具": "torch.ao.quantization"},
    "量化 (INT4)": {"节省": "8× 存储/带宽", "精度损失": "1-3%", "工具": "AutoGPTQ, AWQ"},
    "NAS": {"节省": "2-10× FLOPs", "精度损失": "< 2%", "工具": "NNI, Archai"},
    "蒸馏": {"节省": "自定义", "精度损失": "取决于师生差距", "工具": "手动实现"},

    # 训练
    "AMP (FP16)": {"加速": "1.5-2×", "精度影响": "极小", "工具": "torch.cuda.amp"},
    "ZeRO-3": {"内存节省": f"N× (N GPU)", "通信开销": "中等", "工具": "FSDP, DeepSpeed"},
    "DGC": {"通信压缩": "270-600×", "精度影响": "几乎无", "工具": "手动实现"},
    "1-Bit SGD": {"通信压缩": "~32×", "精度影响": "可忽略", "工具": "手动实现"},

    # 端侧
    "TinyTL": {"内存节省": "5-10×", "适用": "迁移学习", "工具": "手动实现"},
    "联邦学习": {"隐私": "数据不出设备", "通信": "周期性同步", "工具": "Flower, TFF"},
}
```

## 7. TinyML / Edge AI 部署意义

本课程中所有技术最终都服务于一个目标：**让 AI 在资源受限的设备上运行**。从边缘视角回顾：

| 技术 | 对边缘的价值 | 当前局限 |
|------|------------|---------|
| 剪枝 | 减少存储（Flash）和内存（SRAM）占用 | 非结构化剪枝在 CPU/GPU 上加速有限 |
| 量化 | INT8 在 Cortex-M 上有原生支持 | INT4/INT2 需要软件模拟或专用硬件 |
| NAS | 自动为特定硬件搜索最优架构 | 搜索成本高，通常只做一次 |
| 蒸馏 | 使小模型达到可接受精度 | 需要先有大模型教师 |
| 联邦学习 | 隐私保护的分布式训练 | Non-IID 数据导致收敛困难 |
| 端侧训练 | 持续学习、个性化 | 内存和功耗仍是瓶颈 |
| 模型并行/ZeRO | 云端训练大模型 → 生成小模型 | 不直接用于边缘 |

**边缘部署的决策框架**：
1. 模型能否在目标设备上运行？（内存、计算、功耗）
2. 如果不能，优先尝试**量化**（几乎无损，最大收益）
3. 然后尝试**剪枝**（结构化剪枝 > 非结构化以获得实际加速）
4. 最后尝试**蒸馏 + NAS**（需要额外工作但可能大幅改进）
5. 如果需要个性化 → 端侧训练（当前对 MCU 不现实，手机/平板可行）

## 8. 常见误区

1. **"这些技术可以任意组合"** — 技术之间存在**非平凡的相互作用**。剪枝后的模型再做量化，激活值分布会改变，需要重新校准。蒸馏 + 剪枝：先蒸馏还是先剪枝？大多数情况下，蒸馏教师模型 → 剪枝学生模型的效果最好。

2. **"精度下降 1% 无所谓"** — 在 ImageNet 上从 76.4% → 75.4%（下降 1%）看起来很小，但在关键任务（医学诊断、自动驾驶）中，这 1% 可能意味着漏诊。此外，不同技术的精度损失可能叠加（剪枝 1% + 量化 1% + 蒸馏 1% ≠ 3% 而是可能更大）。

3. **"边缘部署就是缩小模型"** — 缩小模型是必要条件但不是充分条件。还需要考虑：推理框架的运行时开销（Python vs. C++）、硬件特定优化（如 CMSIS-NN for Cortex-M）、内存对齐、DMA 传输、中断处理等**系统工程**问题。

4. **"大模型的优化经验适用于小模型"** — 不完全。大模型过参数化程度高，对剪枝/量化的容忍度高。小模型（< 5M 参数）本身已经很紧凑，过度压缩容易导致显著的精度下降。

5. **"只要 FLOPs 降低，速度就会变快"** — FLOPs 减少不代表延迟线性减少。Element-wise 操作（FLOPs 低但 Memory-bound）和矩阵乘法（FLOPs 高但 Compute-bound）的延迟特性完全不同。**Roofline 分析**是连接 FLOPs 和实际延迟的桥梁。

6. **"硬件自动会越来越快，不需要优化"** — 摩尔定律已放缓（Dennard Scaling 终结，芯片功耗密度达到物理极限），单纯依赖硬件进步不再可行。**算法效率的改进对实际应用的贡献越来越大**——Transformer 的发明（2017）带来的进步可能远超 GPU 从 V100 到 H100 的硬件升级。

## 9. 面试问题

**Q1: 描述一个完整的"预训练模型 → 移动端部署"的优化管道。**
A: (1) 从预训练模型（如 ResNet-50）开始 (2) 结构化剪枝：移除不重要的通道（如 30% channels），精调恢复 (3) 量化感知训练（QAT）或训练后量化（PTQ）转 INT8 (4) 导出为 ONNX 或 TorchScript (5) TensorRT/ONNX Runtime 做算子融合和 kernel 调优 (6) 部署到目标设备。关键决策点：是否需要 QAT（精度敏感用 QAT，不在意 1-2% 精度损失用 PTQ）。

**Q2: 你有一个 7B 参数的 LLM，想在 4 张 A100（80GB）上做全参数微调。怎么做？**
A: (1) 计算内存需求：7B × 16 bytes/参数（FP16 param + gradient + FP32 optimizer）≈ 112GB × 1.3（激活值开销）≈ 146GB，远超单卡 80GB (2) 使用 FSDP/ZeRO-3：将参数/梯度/优化器分片到 4 张卡，每卡 ~36GB (3) 梯度检查点（Gradient Checkpointing）减少激活值内存 (4) 使用 Flash Attention 减少注意力计算的内存 (5) 混合精度训练（BF16 + FP32 optimizer）。如果仍不够，加入 LoRA 做参数高效微调而非全参数微调。

**Q3: 剪枝、量化和蒸馏三者的适用场景和优先级？**
A: **量化**优先级最高——几乎无精度损失、4× 内存和带宽节省、硬件广泛支持。**剪枝**次之——对于过参数化的大模型效果显著，但非结构化剪枝的实际加速需要稀疏硬件支持。**蒸馏**最后——需要额外的训练，但可以同时减小模型并保持精度，且可与前两者叠加。实践中通常是"量化 + 剪枝"或"蒸馏 + 量化"组合。

**Q4: 为什么 LLM 推理的瓶颈不是 FLOPs 而是内存带宽？**
A: 自回归推理每次只生成一个 token，属于**极低 batch size** 的场景。此时 arithmetic intensity 非常低（每搬一个参数只做 2 次乘加），操作变成 memory-bound。具体来说：一个 7B 模型从内存加载所有参数（~14GB FP16）只为了生成一个 token（~几万 FLOPs），内存带宽成为瓶颈。这就是为什么量化（减少参数字节数）和 KV Cache 压缩对 LLM 推理加速效果显著。

**Q5: 协同设计（co-design）是什么意思？请举例说明。**
A: 协同设计是指算法和硬件**不是先后独立设计的**，而是在设计过程中持续相互反馈。例如：设计 INT4 量化方案时，不仅要考虑算法精度（如何校准 scale/zero-point），还要考虑：目标硬件是否有 INT4 ALU（如果没有，模拟开销会抵消收益）、INT4 如何打包对齐缓存行（内存布局）、是否需要非对称量化（对称量化硬件友好但精度差）。反过来，设计 AI 加速器时，需要考虑：是否需要稀疏矩阵乘法器（支持结构化剪枝）、是否需要混合精度单元（支持训练中的 FP16 前向 + FP32 累加）。这就是 EE（硬件设计）和 CS（算法设计）的交叉点。

## 10. 本讲总结

MIT 6.5940 覆盖了从算法到系统、从训练到推理、从云端到边缘的全栈高效深度学习技术。课程的**统一主题**是：

**效率 = 精度 / 资源消耗**，而"资源"包括计算（FLOPs）、内存（Bytes）、通信（Bandwidth）、功耗（Watts）四个维度。

**四大洞察**：
1. **过参数化是优化的空间**：深度网络天然存在冗余，剪枝和量化就是利用这种冗余
2. **硬件是算法的镜子**：Roofline 模型告诉我们，不同的优化手段适用于不同的瓶颈（compute-bound vs. memory-bound）
3. **训练与推理的对称性**：训练的内存瓶颈（激活值 + 优化器）与推理的延迟瓶颈（内存带宽 + 自回归）是对称的问题，许多优化技术（量化、稀疏化）可以跨阶段应用
4. **协同设计是终极武器**：最好的优化来自算法结构和硬件特性之间的深度匹配，而非单独优化任何一方

**EE 与 CS 的融合**体现在课程的每个角落：剪枝（算法）需要稀疏矩阵乘法器（硬件）才能获得实际加速；量化（算法）需要 INT8 计算单元（硬件）来兑现理论收益；分布式训练的通信拓扑需要匹配 GPU 物理互联（NVLink, InfiniBand）。一个优秀的"高效 AI 工程师"必须同时理解算法和系统——这也是 6.5940 的核心教育目标。

**如果你只能记住一件事**：永远不要只优化算法或只优化硬件——效率的真正来源是两者之间的"匹配"。
