# Paper 14: TSM — Temporal Shift Module for Efficient Video Understanding (Lin et al., ICCV 2019)

> 论文全称：**TSM: Temporal Shift Module for Efficient Video Understanding**
> 发表会议：ICCV 2019
> 作者：Ji Lin, Chuang Gan, Song Han（MIT HAN Lab）

---

## 1. 论文解决什么问题

视频理解任务中，传统 3D 卷积（如 C3D、I3D）虽然能同时建模空间和时间信息，但计算量巨大：一个 3D 卷积核的参数量和 FLOPs 是 2D 卷积的 $K_T$ 倍（$K_T$ 为时间维度大小）。这使得 3D CNN 难以部署到移动端和边缘设备。TSM 提出一个**零参数、零 FLOP 开销**的方案，使普通的 2D CNN 能获得时序建模能力，在保持 2D CNN 效率的同时达到接近 3D CNN 的精度。

---

## 2. 核心方法

### 核心思想：Temporal Shift

TSM 的核心洞察是：时序建模可以"免费"获得——只需要沿时间维度移动部分通道的特征图即可。

给定视频输入 $X \in \mathbb{R}^{T \times C \times H \times W}$（$T$ 帧，$C$ 通道），对于每个卷积层：
- 将通道分为三组：前向移位（向后一帧移动）、后向移位（向前一帧移动）、保持不变
- 多出来的/空出来的帧用零填充

### 具体操作（以 1/4 移位比例为例）

对于第 $t$ 帧的通道 $c$：
- 前 1/8 通道：取第 $t-1$ 帧的值（未来信息）
- 后 1/8 通道：取第 $t+1$ 帧的值（过去信息）
- 中间 3/4 通道：保持当前帧不变

### 两种变体
- **online TSM**：仅使用过去帧，适用于实时推理（在线视频流）
- **offline TSM**：同时使用过去和未来帧，适用于离线视频分析

### 关键优势
- **零参数增加**：不做任何可学习变换，只是内存移动
- **零 FLOP 增加**：仅在内存中 shift 指针，不增加乘法/加法运算
- **即插即用**：可插入任何 2D CNN（ResNet、MobileNet 等）中

---

## 3. 关键公式

### 传统 3D 卷积的 FLOPs

对于 3D 卷积核 $K_T \times K_H \times K_W$：

$$FLOPs_{3D} = T \times C_{in} \times C_{out} \times K_T \times K_H \times K_W \times H_{out} \times W_{out}$$

### TSM 的 Temporal Shift 操作

设 $X_t^{(c)}$ 为第 $t$ 帧第 $c$ 通道的特征图，移位操作定义为：

$$X_t^{(c)} \leftarrow \begin{cases} X_{t-1}^{(c)} & c \in [0, \frac{C}{8}) \\ X_{t+1}^{(c)} & c \in [\frac{C}{8}, \frac{C}{4}) \\ X_t^{(c)} & c \in [\frac{C}{4}, C) \end{cases}$$

### TSM 模型的总 FLOPs

$$FLOPs_{TSM} = FLOPs_{2D\_CNN} + \underbrace{0}_{\text{shift cost}}$$

TSM 相对于 3D 卷积的加速比：

$$Speedup = \frac{FLOPs_{3D}}{FLOPs_{2D}} \approx K_T \approx 3-7\times$$

与 I3D 对比在 ResNet-50 上达 **10× FLOPs 减少**。

---

## 4. 实验结论

| 方法 | Backbone | FLOPs | Kinetics-400 Top-1 | Something-Something V1 Top-1 |
|------|----------|-------|---------------------|-------------------------------|
| TSN (2D baseline) | ResNet-50 | 33G × 8 × 10 | 70.6% | 19.5% |
| I3D (3D) | ResNet-50 | 108G × N/A × N/A | 72.1% | 41.6% |
| TSM (offline) | ResNet-50 | 33G × 8 × 10 | **74.1%** | **47.2%** |
| TSM (online) | ResNet-50 | 33G × 8 × 1 | 73.0% | 46.0% |

- TSM 在 Kinetics-400 上仅用 **33G FLOPs/view**（每个 view 8 clips × 10 crops 即取平均），超过 I3D 的精度（72.1% vs 74.1%），同时 FLOPs 仅为 I3D 的 **~1/3**
- 在 Something-Something V1（时序敏感数据集）上，TSM **47.2%** 远超 I3D **41.6%**，验证了 shift 操作能有效捕捉时序关系
- **移位比例为 1/4 时最佳**：过大导致空间特征受损，过小导致时序建模不足
- 与 TSN（纯 2D）对比，TSM 在 Something-Something 上提升 **+27.7%**，证明时序建模的必要性
- 在移动端模型 MobileNetV2 上，TSM 同样有效，仅增加 <1% FLOPs

---

## 5. 工业价值

- **已被业界广泛采用**：TSM 代码（GitHub 7k+ stars）被字节跳动、快手等视频应用产品集成
- **移动端部署**：在手机上运行实时视频理解（如手势识别），延迟 <30ms
- **设计哲学**："零成本时序建模"的理念启发了后续 Temporal Difference、TDN、VideoMAE 等工作
- **硬件友好**：Shift 操作仅涉及内存指针移位，不需要专用硬件加速器，在 CPU/GPU/移动 NPU 上均可高效执行

---

## 6. 与课程 Lecture 的关系

- **Lecture 17（Efficient Video Understanding）**：本文是最核心的论文之一，展示了如何在不引入额外参数/计算的前提下赋予 2D CNN 时序建模能力，是效率导向视频理解的代表作
- **Lecture 1（Efficiency Metrics）**：TSM 是"零 FLOP 开销获得性能提升"的经典案例，体现了效率指标的深层含义——并非所有有用操作都需要用 FLOPs 衡量
- **Lecture 7（Algorithm-System Co-design）**：Shift 是一种硬件友好的操作，可以被编译为高效的 memmove 指令，体现了算法层面的创新如何与系统层优化配合

---

## 7. 我应该如何复现

1. **环境准备**：安装 PyTorch、OpenCV、mmaction2 或直接使用官方仓库 `github.com/mit-han-lab/temporal-shift-module`
2. **实现 Shift 操作**：核心代码仅 10 行 —— 使用 `torch.roll` 沿时间维度滚动 tensor：
   ```python
   def temporal_shift(x, n_segment=8, shift_div=8):
       nt, c, h, w = x.size()
       n_batch = nt // n_segment
       x = x.view(n_batch, n_segment, c, h, w)
       fold = c // shift_div
       out = torch.zeros_like(x)
       out[:, :-1, :fold] = x[:, 1:, :fold]
       out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold]
       out[:, :, 2*fold:] = x[:, :, 2*fold:]
       return out.view(nt, c, h, w)
   ```
3. **替换 Backbone**：在 ResNet-50 的每个残差块中，在第一个 conv1 之前插入 `temporal_shift`
4. **训练配置**：
   - 数据集：Something-Something V2（时序敏感，更适合验证）
   - 8 帧采样（`num_segments=8`），`shift_div=8`
   - SGD with momentum=0.9，初始 lr=0.01（30 epochs 衰减），batch_size=64（8 GPU）
5. **验证**：在 Kinetics-400 和 Something-Something V2 上评估 Top-1/Top-5 精度
6. **关键注意事项**：
   - `n_segment` 和 `shift_div` 必须匹配数据加载器的帧采样设置
   - 在 BN 层之前插入 shift 效果最好（作者消融实验验证）
   - 推理时如需 real-time，使用 online 模式（仅向过去 shift）
