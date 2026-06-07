# Paper 18: EfficientViT — Multi-Scale Linear Attention for High-Resolution Dense Prediction (Cai et al., ICCV 2023)

> 论文全称：**EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction**
> 发表会议：ICCV 2023
> 作者：Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han（MIT HAN Lab）

---

## 1. 论文解决什么问题

Vision Transformer（ViT）在图像分类上表现优秀，但全局 softmax 注意力在**高分辨率密集预测任务**（语义分割、超分辨率、目标检测）上计算量爆炸：$O(N^2)$ 的注意力在 1024×1024 图像上产生百万量级的 token 序列，内存和计算都无法承受。同时，现有的轻量级 ViT（如 MobileViT、EdgeViT）缺乏多尺度特征金字塔，不适合密集预测。EfficientViT 提出 **ReLU 线性注意力**替代 softmax 注意力，并设计了**硬件友好的 sandwich 布局**和多尺度融合策略，在移动端芯片上实现极高效率。

---

## 2. 核心方法

### ReLU 线性注意力

标准 softmax 注意力：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

计算 $QK^T$ 需 $O(N^2 d)$，是大分辨率场景的主要瓶颈。

EfficientViT 的做法：用 **ReLU 激活函数** 替代 softmax 归一化，然后改变计算顺序：

$$\text{LinAttn}(Q, K, V) = \frac{\text{ReLU}(Q) \left(\text{ReLU}(K)^T V\right)}{\text{ReLU}(Q) \left(\text{ReLU}(K)^T \mathbf{1}\right)}$$

关键：先计算 $\text{ReLU}(K)^T V$（复杂度 $O(N d^2)$），再乘以 $\text{ReLU}(Q)$（复杂度 $O(N d^2)$）。总复杂度 $O(N d^2)$，在 $N \gg d$ 时远小于 $O(N^2 d)$。

### 为什么 ReLU 而非 Softmax？

- Softmax 要求整个 $QK^T$ 矩阵才能计算归一化，无法交换计算顺序
- ReLU 天然非负，提供非线性的同时去除负值噪声，可以拆解成 "先聚合 key-value 再乘 query" 的计算模式
- ReLU 不强制所有注意力权重归一化，给了更多自由（类似稀疏化效果）

### Hardware-Efficient Sandwich Layout

传统 ViT 的 FFN 在注意力之后。EfficientViT 的 sandwich 结构：

```
Input → MBConv (depthwise) → Linear Attention (N heads) → MBConv (depthwise) → Output
         ↑ FFN 前置                    ↑ Attn 中间              ↑ FFN 后置
```

- 前后两个 MBConv block（MobileNetV2 的倒残差模块）提供局部特征提取
- 中间的线性注意力负责全局信息交互
- MBConv 在移动端（CPU/NPU）比标准卷积更高效（depthwise separable）

### 多尺度特征金字塔

EfficientViT 分多个 stage，每个 stage 后分辨率减半、通道数翻倍（类似 ResNet 的 stage 设计），天然生成多尺度特征图 $P_2, P_3, P_4, P_5$，可直接喂入 FPN 做密集预测。

---

## 3. 关键公式

### 线性注意力的计算等价性

由于 ReLU 的非负性，计算可分两步：

步骤 1 — 全局上下文聚合（$O(N d^2)$）：
$$S = \text{ReLU}(K)^T V \in \mathbb{R}^{d \times d}$$

步骤 2 — 逐 token 查询（$O(N d^2)$）：
$$O_i = \text{ReLU}(Q_i) S$$

归一化（$O(N d^2)$）：
$$Z_i = \text{ReLU}(Q_i) \cdot \text{ReLU}(K)^T \mathbf{1}$$

最终输出：
$$\text{LinAttn}_i = \frac{O_i}{Z_i}$$

### 计算复杂度对比

| 注意力类型 | 复杂度 | N=1024,d=64 时的 FLOPs |
|------------|--------|------------------------|
| Softmax Attention | $O(N^2 d)$ | ~67M |
| ReLU Linear Attention | $O(N d^2)$ | ~4.2M |

**约 16× FLOPs 减少**（当 $N \gg d$ 时放大）。

### Sandwich 布局的 FLOPs 分配

设 MBConv 的 FLOPs 为 $F_M$，线性注意力的 FLOPs 为 $F_A$，总 FLOPs 为：

$$F_{\text{total}} = 2F_M + F_A$$

其中 $F_A \ll F_M$（因为线性注意力是 $O(N d^2)$，depthwise conv 是 $O(N H W C K^2)$，后者在高分辨率时更大）。

---

## 4. 实验结论

| 模型 | Params | MACs | ImageNet Top-1 | ADE20K mIoU | COCO AP |
|------|--------|------|----------------|-------------|---------|
| MobileNetV3-Large | 5.4M | 219M | 75.2% | — | — |
| Swin-Tiny | 28M | 4.5G | 81.3% | 44.5 | 46.0 |
| EfficientViT-B0 | **3.5M** | **79M** | **74.3%** | **34.6** | **28.7** |
| EfficientViT-B1 | 8.4M | 202M | **79.3%** | **40.3** | **38.1** |
| EfficientViT-B2 | 21M | 516M | **81.6%** | **43.7** | **43.2** |
| EfficientViT-B3 | 49M | 1.2G | **83.1%** | **46.8** | **46.4** |

- EfficientViT-B2 以 **21M 参数、516M MACs** 达到与 Swin-Tiny（28M、4.5G MACs）接近的精度，MACs 减少 **8.7×**
- 在移动端骁龙 8 Gen 1 CPU 上，EfficientViT-B1 推理延迟 **14ms**（Swin-Tiny 需 200+ms），真正做到了手机端可用
- 在 ADE20K 语义分割和 COCO 目标检测上，EfficientViT 以极低的计算量获得有竞争力的精度，验证了多尺度设计的有效性
- ReLU vs Softmax 消融实验：ReLU 精度提升 0.3%（因为 ReLU 天然稀疏化 + 非负性），且计算量降低一个量级

---

## 5. 工业价值

- **移动端视觉 backbone**：EfficientViT 被高通、联发科等芯片厂商采纳为 AI Engine 的推荐 backbone
- **实时 AR/VR**：14ms 的延迟满足 AR 眼镜对实时语义分割的需求（端侧 SLAM + 场景理解）
- **设计范式**：线性注意力 + sandwich 布局的设计模式启发了 MobileViTv3、FastViT、EdgeSAM 等工作
- **实际部署**：在 Jetson Orin / 高通 RB5 等边缘设备上，EfficientViT 是少数能实时运行语义分割的 ViT 方案

---

## 6. 与课程 Lecture 的关系

- **Lecture 16（Vision Transformer）**：EfficientViT 是课程中 ViT 效率优化的代表论文，直接回应了 ViT 在高分辨率场景下计算不可行的挑战
- **Lecture 1（Efficiency Metrics）**：论文从 Params、MACs、延迟、mIoU 多维度评价模型，是效率指标在视觉任务中的实战应用
- **Lecture 7（NAS）**：EfficientViT 的 sandwich 布局和多尺度 stage 设计可视为手工设计的网络架构搜索（NAS 的思想来源）
- **Lecture 2（Efficiency Metrics Deep Dive）**：线性注意力与 softmax 注意力的复杂度分析是本节内容的直接延伸

---

## 7. 我应该如何复现

1. **环境准备**：PyTorch 2.0+，timm（预训练权重），mmsegmentation（语义分割）
2. **实现 ReLU 线性注意力**：
   ```python
   def linear_attention(q, k, v):
       # q, k, v: [B, N, H, D]
       q = torch.relu(q)
       k = torch.relu(k)
       kv = torch.einsum('bnhd,bnhe->bhde', k, v)  # [B, H, D, D]
       z = k.sum(dim=1, keepdim=True)               # [B, 1, H, D]
       out = torch.einsum('bnhd,bhde->bnhe', q, kv)  # [B, N, H, D]
       norm = torch.einsum('bnhd,bhd->bnh', q, z.squeeze(1))
       return out / (norm.unsqueeze(-1) + 1e-6)
   ```
3. **构建 sandwich block**：
   - 前 MBConv：expand_ratio=4, kernel_size=3
   - 中间：线性注意力（num_heads=4 或 8）
   - 后 MBConv：expand_ratio=4, kernel_size=3
4. **构建多尺度 backbone**：4 个 stage，分辨率 [H/4, H/8, H/16, H/32]，通道 [32, 64, 128, 256]
5. **训练配置**：
   - 分类：ImageNet-1K，300 epochs，AdamW, lr=2e-3, cosine decay
   - 分割：在 ADE20K 上用 Semantic FPN head，80k iterations
   - 检测：在 COCO 上用 RetinaNet head
6. **关键注意事项**：
   - 线性注意力的归一化分母可能为 0，需要加 epsilon（1e-6）
   - ReLU 的非负性对于计算等价性至关重要（如果改为 GELU 则不成立）
   - 高分辨率输入（1024×1024）时，线性注意力的优势才明显
