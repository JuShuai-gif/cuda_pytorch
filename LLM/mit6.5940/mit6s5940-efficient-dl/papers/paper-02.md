# Paper 02: MCUNet (Lin et al., NeurIPS 2020)

> 论文全称：**MCUNet: Tiny Deep Learning on IoT Devices**
> 发表会议：NeurIPS 2020
> 作者：Ji Lin, Wei-Ming Chen, Yujun Lin, John Cohn, Chuang Gan, Song Han（MIT HAN Lab）

---

## 1. 论文解决什么问题

将深度学习部署到**微控制器（MCU）**级别设备（如 ARM Cortex-M 系列）面临两个核心瓶颈：
- **内存极小**：典型 MCU 只有 256KB SRAM 和 1MB Flash，一个常规 MobileNetV2 就需要 >2MB 内存
- **缺乏系统支持**：现有的深度学习推理引擎（TF-Lite Micro、ARM CMSIS-NN）需要大量手工优化

MCUNet 提出了一套**联合设计（Co-design）**方案：TinyNAS 自动搜索适合微控制器的网络结构，TinyEngine 自动生成针对该网络结构的最优推理代码，将 ImageNet 级别的图像分类带到了 Cortex-M 设备上。

---

## 2. 核心方法

MCUNet 由两大组件构成：

### TinyNAS（网络架构搜索）
- **搜索空间设计**：基于 MobileNetV2 的 inverted residual block，搜索空间包括：
  - 每个 block 的 expansion ratio
  - kernel size
  - 输出通道数（宽度）
  - 是否使用 skip connection
- **内存约束**：搜索过程中以 peak SRAM 使用量为硬约束（例如 ≤ 256KB）
- **Two-stage NAS**：
  - Stage 1: 用 weight-sharing supernet 训练一个"超网"
  - Stage 2: 在超网上用进化算法（Evolutionary Search）搜索满足内存约束的最优子网

### TinyEngine（代码生成推理引擎）
- 接收 TinyNAS 搜索出的网络结构，自动生成 C 语言级推理代码
- 关键优化技术：
  - **In-place depthwise convolution**：depthwise 卷积直接在原地操作，节省 SRAM
  - **Patch-based inference**：将大图像切分成 patch 分块处理，减少峰值内存
  - **Loop tiling & reordering**：自动优化循环嵌套顺序以利用 cache
  - **Memory planning**：自动分析各层的 tensor 生命周期，复用内存空间

---

## 3. 关键公式

### 内存约束搜索
搜索目标是在内存约束下最大化精度：

$$\max_{\alpha \in \mathcal{A}} \text{Acc}(N_\alpha) \quad \text{s.t.} \quad \text{PeakSRAM}(N_\alpha) \leq M_{\text{limit}}$$

其中 $\alpha$ 是网络架构参数，$M_{\text{limit}}$ 是 MCU 的内存上限。

### Peak SRAM 估算
对于每一层 $l$：

$$\text{PeakSRAM} \approx \sum_{l} \max(\text{Input}_l, \text{Output}_l, \text{Weight}_l)$$

TinyEngine 使用更精确的 lifetime 分析来估算实际峰值内存。

### In-place 卷积
将激活张量 $X$ 和输出 $Y$ 共享同一块内存区域：

$$Y = \text{DepthwiseConv}(X, W) \quad \text{with} \quad \text{ptr}(Y) = \text{ptr}(X)$$

---

## 4. 实验结论

### ImageNet 分类结果（MCUNet-320MB/32KB）

| 模型 | Flash (KB) | SRAM (KB) | Top-1 Acc | 延迟 (ms) |
|------|------------|-----------|-----------|-----------|
| MobileNetV2-0.35 | 552 | 204 | 60.2% | 未报告 |
| ProxylessNAS 搜索 | 428 | 321 | 超出 MCU 内存 | - |
| **MCUNet-320MB** | 320 | 293 | 60.7% | 110ms |
| **MCUNet-512MB** | 512 | 256 | 64.2% | 144ms |

- MCUNet 是第一个在 MCU 上跑通 ImageNet 分类的工作
- TinyEngine 生成的代码比 TF-Lite Micro 快 1.7-2.4 倍
- 内存使用比 CMSIS-NN 优化后的 TensorFlow 低 3-4 倍

---

## 5. 工业价值

- **TinyML 里程碑**：将 ImageNet 级别分类带到了 2 美元成本的 MCU 上
- **实际产品落地**：被集成到多家 IoT 公司的产品中（智能家居传感器、可穿戴设备）
- **开源影响力**：MCUNet GitHub 仓库有 2000+ stars，TinyEngine 被社区广泛使用
- **AutoML for Edge**：证明了架构搜索可以针对特定硬件约束自动优化

---

## 6. 与课程 Lecture 的关系

- **Lecture 5 (Neural Architecture Search)**：TinyNAS 的 two-stage weight-sharing NAS 是本课程 NAS 部分的直接延伸
- **Lecture 7 (System + Algorithm Co-design)**：MCUNet 是 algorithm-hardware co-design 的经典案例
- **Lecture 9 (MCU & TinyML)**：本论文直接对应 MCU 部署 lecture 的核心内容
- **Lecture 6 (Automated Pruning)**：超网训练和子网搜索受到 auto-pruning 思想的启发

---

## 7. 我应该如何复现

1. **环境准备**：STM32F746 Nucleo 开发板（或 QEMU 模拟器）+ ARM GCC 交叉编译工具链
2. **TinyNAS 复现**：
   - 使用 PyTorch 构建 MobileNetV2-based 超网
   - 实现 memory cost model（估算每个 block 的 SRAM 使用）
   - 用进化算法在超网上搜索满足 256KB SRAM 约束的子网
3. **TinyEngine 复现**：
   - 用 Python 写一个代码生成器，将搜索到的网络转成 C 代码
   - 实现 in-place depthwise conv 和 patch-based inference
   - 交叉编译到 ARM Cortex-M4
4. **验证**：
   - 在 ImageNet 验证集上测精度
   - 用串口连接到 MCU 板，测量实际推理时间
   - 用 `arm-none-eabi-size` 分析编译后二进制大小
5. **简化复现**：如果无硬件，可以用 QEMU 模拟 Cortex-M4，或直接在 PyTorch 里模拟内存约束计算

