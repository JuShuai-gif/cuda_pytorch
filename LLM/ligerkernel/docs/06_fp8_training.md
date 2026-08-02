# FP8 训练指南

> 背景：Liger-Kernel 是训练算子仓库，但只支持 fp32 / fp16 / bf16 与 AMP 混合精度，
> 本身**不做 fp8/fp4**。fp8 训练需要另一套精度管理栈。本文总结 fp8 训练的实现路径、
> 关键机制与可借鉴的开源仓库。

## 一、什么是 fp8 训练

fp8 训练指让**计算路径（尤其是 GEMM）使用 8 位浮点**进行训练。它不是"全 fp8"，
而是典型的**混合精度**：只有前向（有时含反向）的矩阵乘用 fp8，参数、优化器状态仍
保持高精度。代表性案例是 DeepSeek-V3 的 FP8 训练。

与 Liger 这类"算子融合/省显存"优化是**正交**的：两者可以共存，但量化本身要靠下述
专门的库。

## 二、硬件与数据格式

### 硬件要求

- 支持 fp8 GEMM 的 GPU：
  - **Ampere / Hopper**：支持 E4M3 / E5M2（`torch.float8_e4m3fn` / `torch.float8_e5m2`）
  - **Blackwell**：进一步支持更高精度 fp8 与 MXFP4
- 没有硬件时只能软件模拟，性能无意义（可作教学/验证用）。

### 两个 fp8 格式

| 格式 | 特点 | 业界常见用途 |
| --- | --- | --- |
| **E4M3**（4 位指数 + 3 位尾数） | 精度好、动态范围小（最大约 448） | 前向激活与权重 |
| **E5M2**（5 位指数 + 2 位尾数） | 动态范围大、精度差 | 反向梯度（见下节） |

### torch 中的五种 float8 类型

torch 暴露了 5 种 8 位浮点 dtype，命名规则：`float8_e<指数位>m<尾数位>[fn][uz]`。

| 类型 | 指数位 | 尾数位 | 最大范围 | 精度 | 特点 |
| --- | --- | --- | --- | --- | --- |
| `float8_e4m3fn` | 4 | 3 | ±448 | 较好 | **最常用**，FP8 推理/训练的主力格式 |
| `float8_e5m2` | 5 | 2 | ±57344 | 较差 | 范围大精度差，常用于反向梯度 |
| `float8_e4m3fnuz` | 4 | 3 | 略小于 e4m3 | 较好 | `uz` = unsigned zero（零无正负号） |
| `float8_e5m2fnuz` | 5 | 2 | 大 | 较差 | 同上 + 无符号零 |
| `float8_e8m0fnu` | 8 | 0 | 超大 | 无精度 | **只存缩放因子**，本身不算数 |

要点：

- **`fn`（finite only）**：不支持 inf/NaN，省出的 bit 模式换来更多动态范围。
- **`uz`（unsigned zero）**：零只有一种表示（没有 +0/-0），简化硬件。
- **`float8_e8m0fnu`**：8 位全是指数、0 位尾数，**不能表达数值精度，只用来存每个张量的缩放因子（scale）**，是 Hopper/Blackwell 硬件为 FP8 量化配套的特殊格式。一个张量往往由"E4M3/E5M2 数据 + E8M0 scale"共同表示。

实际使用（以 FlashRT 推理引擎为例，`pi05_thor.py` 中的 `fp8 = torch.float8_e4m3fn`）：量化推理选 **E4M3**，因为它"范围够用、精度最高"，是权重和激活的主力格式；E5M2 通常只用于对精度不敏感、但数值可能很大的地方（如训练时的梯度）。

## 三、核心机制

1. **缩放因子（scaling factor）**
   - fp8 是定点近似，每个 tensor / tile 需按 `max(abs(x))` 计算归一化系数。
   - 使用**延迟一个 step 的缩放系数（delayed scaling）**，保证反向时系数已就绪。
2. **fp32 累加**
   - GEMM 乘算用 fp8，**累加必须在 fp32 进行**，否则数值发散。
3. **高精度 master weight 与优化器状态**
   - 参数（master weight）与 Adam 状态通常仍存 fp32 / bf16。
   - 这是"混合精度"的本质：低精度只作用于计算路径，不直接存成唯一状态。

## 三.5、E4M3 还是 E5M2？（附 DeepSeek-V3 的纠正）

**业界标准做法**是混合使用：**前向 Fprop 用 E4M3，反向 Dgrad/Wgrad 用 E5M2**
（NVIDIA Transformer Engine 等）。原因：

- 前向的激活/权重动态范围适中，E4M3 尾数多、精度好 → 用 E4M3。
- 反向的**梯度张量常有离群值、动态范围大**，E4M3 容易溢出 → 改用指数位更多、
  "更扛动态范围"的 E5M2（代价是精度低一点）。

**注意一个常见误解：DeepSeek-V3 并不是"前向 E4M3、反向 E5M2"。**
DeepSeek-V3 技术报告 §3.3.2 明确说明：它**对所有张量统一用 E4M3**，以获得更高精度。
其可行性来自**细粒度量化**：

- 激活按 **1×128 tile**（每 token 每 128 通道）分组缩放
- 权重按 **128×128 block** 分组缩放

每组用独立缩放系数把数值对齐到 E4M3 的可表示范围，相当于在小分组内"共享指数位"，
消除了离群值导致的溢出问题，于是反而能用精度更高的 E4M3 全程训练。
配套手段还有：GEMM 累加提升到 FP32（每 128 元素提升到 CUDA Cores）、在线计算缩放系数。

| | 格式 | 关键原因 |
| --- | --- | --- |
| 业界标准（TE 等） | Fprop E4M3 / 反向 E5M2 | 用更大指数范围保护易溢出的梯度 |
| DeepSeek-V3 | **全程 E4M3** | 细粒度 tile/block 缩放消除溢出，享受 E4M3 更高精度 |

## 四、实现路径

### 路径 A：直接用成熟框架（推荐）

- **NVIDIA Transformer Engine (TE)**：`fp8_autocast` 包装一次，GEMM 自动走 fp8，
  缩放因子自动管理。常配 Megatron-LM / NeMo。
- **Microsoft MS-AMP**：自动插入低精度，支持"参数 fp8、梯度 fp8、优化器状态分块低精度"
  的多级方案，代码改动最小。
- **DeepSpeed**：`ds_config` 开启 fp8，配合 ZeRO 使用。

### 路径 B：用 torch 原生 fp8 自定义

```python
# torch.float8_e4m3fn 在 matmul 时自动 fp32 累加
a_fp8 = a.to(torch.float8_e4m3fn)          # 手动算好 scale
out = torch._scaled_mm(a_fp8, b_fp8, scale_a, scale_b)
```

适合教学、完全可控，但缩放系数调度要自己实现。

### 路径 C：从零实现（学习）

1. 只做前向 GEMM 的 per-tensor 量化
2. 加入 delayed scaling
3. 加入反向（梯度用 E5M2）
4. 接进训练循环
5. 先保证数值收敛，再谈性能

## 四.5、一个 fp8 训练 step 的实现细节

核心要点：**fp8 只出现在"计算的那一刻"，参数本身永远不真正只以 fp8 存在**。
一个训练 step 的流程：

```text
1. 参数（master weight）和优化器状态始终存 fp32 / bf16
2. 前向：把激活、权重量化成 fp8（带缩放因子）→ fp8 GEMM（fp32 累加）→ 输出转回 bf16/fp32
3. 反向：Dgrad / Wgrad 同样走 fp8 GEMM，梯度张量量化（E4M3 或 E5M2）
4. 梯度缩放 / loss scaling：像 AMP 一样，确保梯度落在 fp8 可表示范围内
5. 参数更新：用 fp32 梯度 + 高精度 master weight 做 AdamW，更新完进入下一轮
6. 工程层：用 hook 包装 Linear（如 TE 的 fp8_autocast）、激活缓存存 fp8、
   分布式 all-reduce 前转高精度
```

工程实现的关键机制：

- **量化时机**：多数框架用 **delayed scaling**（用上一 step 的 max 值算系数，避免本 step
  同步开销）；DeepSeek 用**在线**逐 tile/block 计算。
- **缩放因子管理**：per-tensor / per-tile / per-block，粒度越细越稳，代价是额外计算。
- **哪些算子保持高精度**：归一化、注意力 softmax、embedding、输出头、gate——这些对
  低精度敏感，留在 bf16/fp32。

## 五、可借鉴的 GitHub 仓库

| 仓库 | 用途 |
| --- | --- |
| `NVIDIA/TransformerEngine` | fp8 GEMM、scaling 管理、`fp8_autocast`（必看） |
| `NVIDIA/Megatron-LM` | 大规模训练框架，看它如何与 TE 集成 |
| `microsoft/DeepSpeed` | fp8 + ZeRO 训练工程化 |
| `microsoft/MS-AMP` | 多级低精度混合训练方案 |
| `pytorch/pytorch` | 原生 `float8_e4m3fn/e5m2`、`torch._scaled_mm` |
| `deepseek-ai/DeepSeek-V3` | 论文 + 开源代码，FP8 大规模训练实战（per-tile scaling） |
| MoE 相关（如 Megatron-LM / megablocks 的 fp8 分支） | fp8 在 MoE 上的落地 |
| 教学向小仓库（如 `fp8-training-tutorial`、`scaled_fp8_emulation`） | 最小可读实现 |

## 六、起步建议

1. 先读 **DeepSeek-V3 技术报告**（arXiv 2412.19437）的 FP8 训练章节，理解 per-tile
   scaling，以及 DeepSeek 为什么能全程用 E4M3（对比业界"前向 E4M3、反向 E5M2"）。
2. 跑通 TE 的 `examples/fp8`，对比 fp32 收敛曲线。
3. 用 torch 原生 fp8 写一个 **per-tensor delayed scaling 的最小示例**（一个 Linear 层），
   验证 loss 收敛。
4. 最后再考虑接进 Megatron / DeepSpeed 做大规模训练。

## 六.5、VLA（视觉-语言-动作模型）能用 fp8 训练吗

**能，而且收益往往比纯文本更大。**

VLA（如 OpenVLA、Pi0、RT-2）本质还是 **transformer + GEMM**：视觉编码器
（SigLIP / DINOv2 / ViT）→ 大语言模型（Llama / Qwen）→ action head。fp8 训练发生在
**GEMM 精度栈**这一层，和"文本还是视觉"无关，因此：

- 视觉编码器、多模态投影、LLM 的 MLP/attention 全是大矩阵乘，**都可以走 fp8**。
- 需要留高精度的还是归一化、softmax、embedding 那些——与文本 LLM 完全一致。

需要注意的差异：

- 视觉特征 / action 标签的**数值分布和文本不同**（可能有更多离群值），量化策略
  （per-tile / per-block scaling）需要重新校准，但这属于通用调参问题，不是 VLA 特有。
- 现有主流 VLA 训练栈（OpenVLA、Pi0/openpi 等）默认是 **bf16**，要上 fp8 需要自己接
  TE / MS-AMP，或复用 DeepSeek 那套 per-tile scaling 方案。
- VLA 常处理**视频/长序列**，显存和吞吐瓶颈比纯文本更突出 → fp8 的"显存减半 + GEMM
  提速"对 VLA 其实**更有吸引力**。

## 七、常见误区

- "fp8 训练 = 全部用 fp8"：错。master weight、优化器状态、累加仍是高精度。
- "Liger 能做 fp8"：Liger 本身不做量化，只是计算融合；要 fp8 需配 TE / MS-AMP / DeepSpeed。
- "有 fp8 硬件就能快"：还需要 scaling 调度、kernel 对齐、分布式精度对齐，否则会掉点或发散。
- "DeepSeek-V3 是前向 E4M3、反向 E5M2"：错。那是业界标准做法；DeepSeek-V3 全程用 E4M3，
  靠细粒度量化保证精度。
