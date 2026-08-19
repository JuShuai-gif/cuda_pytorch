# 01｜量化基础：公式、粒度、精度体系与 PTQ

## 本模块解决的问题

量化是推理优化里"精度换效率"的核心手段。但要正确使用它，必须先理解：量化公式是什么、scale 从哪来、不同粒度差多少、FP8/INT8/INT4 和 FP16/FP32 到底差在哪。本章用可复现的数字回答：

```text
量化公式 x_q = round(x/scale) 的 scale / zero-point / clipping 怎么定？
为什么 per-channel 精度通常更高？为什么 group-wise 常用于 LLM？
为什么 INT8 不一定比 FP16 快？为什么量化后可能 memory-bound？
为什么 dequant kernel 可能反而成为瓶颈？
```

配套代码：`src/compression/quantization/`（`quantize.py`、`dtypes.py`、`ptq.py`）。

---

## 1. 量化公式

对称 int8 量化（本文用对称，省去 zero-point）：

```text
量化：  x_q = clamp(round(x / scale), -127, 127)
反量化：x  ~= scale * x_q
```

三个关键量：

| 量 | 作用 | 怎么定 |
|---|---|---|
| `scale` | 每个量化步长代表多少实数值 | 由数据范围决定，如 `max(|x|)/127` |
| `zero-point` | 非对称量化里 0 对应的整数 | 补偿非零均值分布 |
| `clipping` | 截断范围 | 越紧精度越高，但截断 outliers 引入误差 |

**scale 是误差的核心来源**：量化误差约 `± scale/2`，而 scale 越大（数据范围越大），误差越大。所以"让每个 scale 覆盖的范围尽量小"是量化精度的第一原则——这正是粒度要解决的问题。

---

## 2. 精度体系：FP32 / TF32 / FP16 / BF16 / INT8

| 格式 | bits | mantissa | exponent | 相对精度 | 硬件路径 |
|---|---|---|---|---|---|
| FP32 | 32 | 23 | 8 | 最高 | CUDA core / 慢 |
| TF32 | 32 | 10 | 8 | 中 | Tensor Core（fp32 输入） |
| FP16 | 16 | 10 | 5 | 中 | Tensor Core（快） |
| BF16 | 16 | 7 | 8 | 低 | Tensor Core（快，范围同 fp32） |
| INT8 | 8 | - | - | 低（依赖 scale） | INT8 Tensor Core |

**TF32 是"fp32 的 10 位尾数版"**：Tensor Core 把 fp32 输入的尾数截到 10 位做矩阵乘，速度接近 fp16 但保留 fp32 的指数范围。所以 tf32 是"不想改代码时的第一档加速"。

### 本机实测（1024³ GEMM，相对 fp32-ieee）

```text
fp32-ieee  471us   误差 2e-4    (1.0x)
tf32       145.6us 误差 0.12    (3.24x)
fp16        32.7us 误差 0.081   (14.4x)
bf16        32.6us 误差 0.579   (14.5x)
```

读法：

1. **fp16/bf16 快 14x**：Tensor Core 的 fp16 吞吐是 fp32 CUDA core 的数倍，而 GEMM 是 compute-bound，所以速度直接兑现。
2. **bf16 速度同 fp16 但误差 7x**：bf16 尾数只有 7 位（fp16 是 10 位），精度明显更差。bf16 的价值在**指数范围**（和 fp32 一样，不易 overflow），所以训练用 bf16，推理用 fp16。
3. **tf32 是折中**：3.2x 加速 + 中等精度损失，不改模型直接吃。

**关键结论**：低精度加速只在 **compute-bound** 的 GEMM 上兑现。如果是 memory-bound 的 elementwise，fp16 不会比 fp32 快（带宽一样，搬的字节数一样）——这引出第 5 节的"为什么 INT8 不一定快"。

---

## 3. 量化粒度：per-tensor / per-channel / per-token / per-group

一个 tensor 用**多少个 scale**，就是粒度：

```text
per-tensor   整个 tensor 一个 scale
per-channel  每个 channel（输出列）一个 scale
per-token    每个 token（行）一个 scale
per-group    每 group_size 个元素一个 scale（LLM 常用 group_size=128）
```

### 本机实测（含 outlier 列的权重，MSE）

```text
per-tensor     1.716e-3
per-channel    1.3e-5    （130x 优于 per-tensor）
per-token      4.44e-4
per-group(128) 1.10e-4
```

### 为什么 per-channel 精度更高

真实 LLM 的权重/激活有 **outlier channel**：少数 channel 的幅度远大于其他。per-tensor 的一个 scale 被 outlier 拉大（`max(|x|)/127`），导致所有 normal 元素的量化步长都变大，精度全面下降。per-channel 给每个 channel 独立 scale，outlier channel 用大 scale、normal channel 用小 scale，互不干扰。

### 为什么 per-group 有时反而不如 per-channel（本模块的关键发现）

本实验里 per-group(128) 的 MSE 是 per-channel 的 **8.5 倍**。原因：outlier 是"整列"的，per-channel 的 scale 恰好对齐 outlier 的结构；而 per-group 按 128 列切块，一个含 outlier 列的 group 里，outlier 会把整个 group 的 scale 拉大，污染同组其他 127 个 normal 列。

**教训：粒度不是越细越好，要看 outlier 的空间结构。**

- 列级 outlier（activation outlier）→ per-channel 最优。
- 元素级 outlier（weight 里零星大值）→ per-group 更细的粒度才有效。
- per-group 用于 LLM 的真实原因：**weight-only 量化时，权重矩阵的 outlier 是零散元素**，group 越细越能局部对齐，且 group-wise 的 scale 可以离线预计算、不增加推理开销。

---

## 4. weight-only INT8 PTQ

PTQ（Post-Training Quantization）：不重训练，直接对训练好的模型算 scale 并量化。

```text
1. 对每个 Linear 权重算 per-channel scale（max(|w|, dim)/127）
2. 权重存 int8（1 byte，fp16 的一半）
3. 推理时 dequant（scale * w_q）再 matmul（activation 保持 fp16）
```

### 本机实测（残差 MLP 4 层）

```text
max_abs_diff = 0.0098   （输出误差，很小）
weight size  = 16.8MB → 8.4MB   （0.50x，减半）
```

weight-only 的精度损失小，因为**权重本身的分布通常比较均匀**（没有 activation 那种极端 outlier），per-channel 量化就能很好地保留。而 activation 的 outlier 才是量化精度的主要敌人（这是 SmoothQuant 的动机，Stage 8 后续）。

---

## 5. 回答几个"反直觉"问题

### 为什么 INT8 不一定比 FP16 快？

三个原因：

1. **memory-bound 的算子不会因 int8 变快**：int8 省的是字节数，如果算子是 memory-bound（elementwise、reduction），int8 让搬的字节减半，理论快 2x；但 GEMM 是 compute-bound，int8 Tensor Core 吞吐是 fp16 的 2x（理想）。所以 int8 的收益取决于算子类型。

2. **dequant 开销抵消收益**：weight-only int8 的 GEMM 要在 kernel 里先 dequant（int8→fp16）再做 fp16 或 int8 计算。dequant 是额外的访存和计算，如果 dequant 是独立 kernel（物化 dequant 后的 weight），反而比直接 fp16 gemm 慢——本仓库 Stage 6 的 `dequant_gemm` 实测 **0.59x（更慢）** 就是这个原因。

3. **精度损失可能不可接受**：int8 的精度取决于 scale，outlier 严重时 int8 的精度崩溃（per-tensor int8 量化带 outlier 的 activation 会灾难性掉点）。

### 为什么量化后可能 memory-bound？

量化把权重从 2 字节（fp16）压到 1 字节（int8），显存带宽压力减半。但如果模型的瓶颈原本是 **activation 的访存**（activation 没量化），量化权重后，GEMM 变成"读 int8 权重 + 读 fp16 激活"，激活的访存占比上升，整个算子反而变成 activation 访存主导的 memory-bound。这就是"weight-only 量化省了权重带宽，但 activation 带宽成为新瓶颈"。

### 为什么 dequant kernel 可能成为瓶颈？

如果 dequant 是**独立 kernel**（先 int8→fp16 物化，再 gemm），那么：

```text
dequant kernel：读 int8(1B) + 写 fp16(2B) = 3B/element 的访存
fp16 gemm     ：读 fp16 weight(2B) + activation

总访存 = 3B（dequant）+ 2B（gemm 读 dequant 后的 weight）= 5B/element
直接 fp16 gemm = 2B/element
```

dequant 的独立 kernel 反而**增加了**访存。正确做法是 **fused dequant+gemm**（在 SRAM 里 dequant，不物化）——但 fused 实现里 dequant 的计算要重复 K/BLOCK_K 次（每个 K tile 都 dequant 一遍），这也是开销。真正的工业解是 **INT8 Tensor Core 直接吃 int8 输入**（mma.s8s8s32），不经过 fp16。

---

## 6. 本模块闭环小结

```text
问题：如何用低精度换推理效率，精度损失可控
      ↓
原理：x_q = round(x/scale)；scale 越小误差越小；粒度决定 scale 的局部性
      ↓
实测：per-channel 比 per-tensor 好 130x；fp16 快 14x；int8 减半模型
      ↓
结论：量化是"精度-速度-显存"三方权衡，粒度选择和 fused dequant 是关键
      ↓
下一步：SmoothQuant（迁移 activation outlier）、AWQ（保护重要权重）、
      GPTQ（逐层重建）、FP8/FP4 现代硬件低精度
```

要继续就说「继续」。
