# Paper 04: AWQ (Lin et al., MLSys 2024)

> 论文全称：**AWQ: Activation-aware Weight Quantization for On-Device LLM Inference**
> 发表会议：MLSys 2024
> 作者：Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han（MIT HAN Lab）

---

## 1. 论文解决什么问题

大语言模型（LLM）推理面临严重的显存和带宽瓶颈。以 LLaMA-7B 为例，FP16 下权重大小约 14GB，需要高端 GPU 才能运行。

现有方法的问题：
- **Round-to-Nearest (RTN) 量化**：量化为 INT4 时，精度严重下降，尤其对 Llama、OPT 等模型
- **GPTQ 等重建方法**：虽能保持精度，但需要大量 GPU 内存做 weight 重建，对大模型不友好
- **SmoothQuant**：需要为每个模型手工调参（migration strength α），通用性差

AWQ 的**核心洞察**：权重中存在约 1% 的显著通道（salient channels），这些通道的激活值很大，但在量化时被均匀处理导致较大误差。AWQ 通过**按通道等价缩放**来保护这些显著权重。

---

## 2. 核心方法

### 核心洞察：权重显著通道（Salient Channels）
- 观察发现：权重矩阵中约 1% 的通道对应的**激活值绝对值显著更大**（通常是输入 embedding 的某些维度）
- 这些通道的权重量化误差被放大（因为乘以大的激活值），导致最终输出误差很大
- 传统 RTN 对所有通道一视同仁，没有保护机制

### AWQ 的解决方案：等效缩放（Equivalent Scaling）
关键思想：如果将显著通道的**权重放大 s 倍，同时将对应的输入激活缩小 s 倍**，数学上是等价的（矩阵乘法结果不变），但权重放大后量化相对误差变小。

$$Y = WX = (W \cdot \text{diag}(s)) \cdot (\text{diag}(s)^{-1} \cdot X)$$

其中 $s$ 是 per-channel 的缩放因子向量。

### 缩放因子 $s$ 的计算
对每个输出通道 $j$：

$$s_j = \max(|X_j|)^{\alpha}$$

其中 $\alpha \in [0, 1]$ 是超参数：
- $\alpha = 0$：所有通道平等 → 退化为 RTN
- $\alpha = 1$：激进的缩放 → 过大缩放可能导致其他通道精度下降
- 论文实验中 $\alpha = 0.5$ 效果最优

### 高效校准
- 仅需要一小批校准数据（通常 128 个样本 token）
- 仅在 linear 层中需要做 AWQ 处理
- 校准后直接执行 per-channel INT4 分组量化（group size = 128）

### TinyChat 推理引擎
- 配套实现的高效推理引擎，支持 GPU (CUDA) 和 CPU 后端
- 实现了 INT4 分组量化的高效 GEMM kernel
- 相比 FP16 推理加速 2-3 倍

---

## 3. 关键公式

### 等效变换
缩放矩阵乘法参数以实现按通道保护：

$$Y = WX = (W \cdot \text{diag}(s)) \cdot (\text{diag}(s)^{-1} \cdot X) = \hat{W} \cdot \hat{X}$$

其中 $\hat{W}$ 是缩放后的权重（量化前转换为 INT4），$\hat{X}$ 是缩放后的输入。

### 量化误差分析
量化误差在缩放后被重新分配：

$$\|WX - Q(\hat{W})\hat{X}\| \leq \sum_{j} \frac{\|X_j\|}{s_j} \cdot \| \text{quant_error}_j \|$$

可以看到，$s_j$ 越大，该通道的量化误差对输出影响越小。这就是 AWQ 通过缩放来保护显著通道的数学依据。

### 搜索 $\alpha$ 的目标
通过 grid search 最小化校准数据上的误差：

$$\alpha^* = \arg\min_{\alpha} \sum_{l} \|W_l X_l - Q_{\alpha}(\hat{W}_l) \hat{X}_l\|^2$$

---

## 4. 实验结论

### LLaMA 系列模型（INT4 Group=128）

| 模型 | 指标 | FP16 | RTN INT4 | GPTQ INT4 | **AWQ INT4** |
|------|------|------|----------|-----------|-------------|
| LLaMA-7B | PPL ↓ | 5.68 | 14.05 | 5.95 | **5.86** |
| LLaMA-13B | PPL ↓ | 5.09 | 11.13 | 5.31 | **5.22** |
| LLaMA-30B | PPL ↓ | 4.10 | 7.73 | 4.76 | **4.54** |
| LLaMA-65B | PPL ↓ | 3.56 | 5.31 | 4.10 | **3.97** |

- **AWQ 全面超越 RTN INT4**，精度接近 FP16
- **与 GPTQ 相当但速度快 4 倍**（不需要重建过程）
- 校准仅需 128 个样本，几分钟完成
- 在 GPU 推理引擎上实现 2-3× 加速

---

## 5. 工业价值

- **LLM 边缘部署的关键技术**：将 7B 模型从 14GB 压缩到约 4GB，可在消费级 GPU（RTX 3060）上运行
- **vLLM 集成**：vLLM 推理框架已原生支持 AWQ 量化模型
- **开源生态繁荣**：Hugging Face 上有大量 AWQ 量化的开源模型可直接使用
- **移动端 LLM**：结合 4-bit 量化 + 手机端推理引擎，实现了手机端 LLM 部署（如 TinyChat）

---

## 6. 与课程 Lecture 的关系

- **Lecture 4 (Quantization)**：AWQ 是激活感知量化（activation-aware quantization）的代表性工作，直接延伸了 Lecture 4 的量化基础知识
- **Lecture 11 (LLM Efficiency)**：本论文的核心主题——大语言模型的量化部署
- **Lecture 10 (Transformer) + Lecture 7 (Co-design)**：权重-激活协同优化体现了 co-design 思想
- **Lab 4（手工量化）**：复现 AWQ 需要理解 per-channel vs per-tensor 量化的区别

---

## 7. 我应该如何复现

1. **环境准备**：安装 `autoawq==0.2.5` 或 `llm-awq` 官方库
2. **加载模型**：`from transformers import AutoModelForCausalLM`，加载 LLaMA-7B
3. **校准数据**：准备 128 个 token 片段（可用 Wikitext-2 或 Pile 数据集）
4. **执行 AWQ**：
   - 对每个 linear 层，读取输入激活的统计值
   - 计算 `s_j = max(|X_j|)^0.5`（用 α=0.5）
   - 将权重乘以 `s_j`，将输入除以 `s_j`
   - 执行 INT4 per-channel 量化
5. **验证**：在 Wikitext-2 上测 perplexity
6. **关键代码**：
   ```python
   # 伪代码：AWQ 缩放
   x_max = torch.max(torch.abs(x_calib), dim=0)[0]
   scaling = torch.pow(x_max, alpha)  # alpha = 0.5
   w_scaled = w * scaling.unsqueeze(-1)
   w_quant = quantize_per_channel(w_scaled, n_bits=4)
   ```
7. **简化试跑**：如果 GPU 内存不够，用 LLaMA-1B 或 TinyLlama 测试
8. **注意事项**：AWQ 只对 linear 层（注意不要量化 layer norm 和 embedding），group_size=128 是常见的平衡精度和压缩的选择

