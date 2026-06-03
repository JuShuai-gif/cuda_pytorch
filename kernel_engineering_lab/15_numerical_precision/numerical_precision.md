# GPU Kernel 数值精度完全指南

> 面向 GPU Kernel 工程师的数值精度实战手册。覆盖 IEEE 754 基础、浮点陷阱、精度分析方法、工业界常见坑、以及针对每种 Kernel 算子的精度特征分析。

---

## 目录

1. [IEEE 754 浮点表示基础](#1-ieee-754-浮点表示基础)
2. [浮点运算的非结合性与误差传播](#2-浮点运算的非结合性与误差传播)
3. [混合精度训练中的算子精度对比](#3-混合精度训练中的算子精度对比)
4. [精度验证方法学](#4-精度验证方法学)
5. [提高精度的工程技术](#5-提高精度的工程技术)
6. [工业界常见的精度坑](#6-工业界常见的精度坑)
7. [各类型 Kernel 的精度特征分析](#7-各类型-kernel-的精度特征分析)
8. [精度测试工程实践](#8-精度测试工程实践)

---

## 1. IEEE 754 浮点表示基础

### 1.1 浮点数的二进制表示

浮点数由三个字段组成：

```
浮点数 = (-1)^sign × 2^(exponent - bias) × (1 + mantissa)
```

| 精度 | 总位数 | 符号位 | 指数位 | 尾数位 | Bias |
|------|--------|--------|--------|--------|------|
| fp32 (float) | 32 | 1 | 8 | 23 | 127 |
| fp16 (half) | 16 | 1 | 5 | 10 | 15 |
| bf16 (bfloat16) | 16 | 1 | 8 | 7 | 127 |

**为什么 fp16 最大值只有 65504？**

fp16 的指数范围是 [-14, 15]（排除全 0 和全 1 的特殊值），因此：

```
fp16_max = (2 - 2^(-10)) × 2^15 = 65504
```

当中间结果超过 65504 时，fp16 会 overflow 到 `+inf`，这是 fp16 最致命的限制。

**为什么 bf16 范围与 fp32 相同但精度更低？**

bf16 保持与 fp32 相同的 8 位指数（动态范围相同），但尾数只有 7 位（vs fp32 的 23 位）。因此 bf16 可以表示 `-3.4e38` 到 `3.4e38`，不会像 fp16 那样频繁溢出。bf16 的精度约为 2-3 位有效十进制数字。

### 1.2 机器精度 ε

机器精度（machine epsilon）是两个相邻浮点数之间的最小间距，约等于尾数的最后一个有效位：

| 精度 | ε | 有效十进制位数 |
|------|---|---------------|
| fp32 | 1.19 × 10⁻⁷ | ~7 位 |
| fp16 | 9.77 × 10⁻⁴ | ~3 位 |
| bf16 | 7.81 × 10⁻³ | ~2-3 位 |
| fp64 | 2.22 × 10⁻¹⁶ | ~16 位 |

**实际意义**：对任意实数 x，其 fp16 表示的相对误差约为 ±0.05%（在非溢出范围内）。

### 1.3 舍入模式 (Rounding Modes)

IEEE 754 定义了四种舍入模式：

| 模式 | 名称 | 行为 | 使用场景 |
|------|------|------|----------|
| Round to nearest, ties to even | RNE | 舍入到最近值，平局取偶数 | **默认模式**，最常用 |
| Round toward zero (truncation) | RTZ | 向零舍入 | C/C++ 整数转换时的默认行为 |
| Round toward +∞ | RUP | 向上舍入 | 区间算术上界 |
| Round toward -∞ | RDN | 向下舍入 | 区间算术下界 |

**GPU 上的舍入行为**：
- 大多数 CUDA PTX 指令默认使用 RNE
- 某些硬件近似指令（如 `__fdividef`）使用非 IEEE 兼容的舍入
- CUDA 的 `cudaSetDeviceFlags` 不支持修改舍入模式（与 x86 不同）
- FMA 指令 `fma.rn` 显式使用 RNE 舍入

### 1.4 特殊值

| 值 | 位模式（fp32） | 说明 |
|----|---------------|------|
| +0 | 0x00000000 | 所有位为 0 |
| -0 | 0x80000000 | 符号位为 1，其余为 0 |
| +inf | 0x7F800000 | 指数全 1，尾数全 0 |
| -inf | 0xFF800000 | 符号位为 1，指数全 1，尾数全 0 |
| NaN (quiet) | 0x7FC00000 起 | 尾数最高位为 1 |
| NaN (signaling) | 0x7F800001 起 | 尾数最高位为 0，其他位非 0 |
| subnormal | 指数=0，尾数≠0 | 表示 2⁻¹²⁶ 量级的极小数 |

**subnormal 数的问题**：
- 性能损失：许多 GPU（如 Ampere 之前）处理 subnormal 数时会触发 microcode 辅助，导致 50-100x 性能下降
- Google TPU 和许多 AI 加速器不支持 subnormal，会 flush 到 0
- CUDA 可通过 `__cudaFlushDenormals()` 将 subnormal 刷到 0 来避免性能损失

### 1.5 fp16 vs bf16 选择策略

| 场景 | 推荐精度 | 原因 |
|------|----------|------|
| 反向传播的权重梯度 | fp16 | 范围不需要很大，精度更关键 |
| 前向传播的激活值 | bf16 | 需要大范围，精度容忍度更高 |
| 损失值存储 | fp32 | 精度和范围都重要 |
| 中间累加 | fp32 或 fp64 | 必须保持高精度 |
| Image/Video 输入 | fp16 | 像素值范围有限 (0-255) |
| Embedding 表 | fp16 | 权重值通常较小，fp16 精度足够 |

---

## 2. 浮点运算的非结合性与误差传播

### 2.1 为什么 (a+b)+c ≠ a+(b+c) ？

浮点加法不满足结合律。根本原因：**每次运算都涉及舍入，舍入的位置不同导致最终结果不同**。

**经典示例 (fp32)**：

```
a = 1e20, b = -1e20, c = 1.0

(a + b) + c = (1e20 + -1e20) + 1.0 = 0.0 + 1.0 = 1.0    ✓ 正确
a + (b + c) = 1e20 + (-1e20 + 1.0) = 1e20 + (-1e20) = 0.0  ✗ 错误！
```

**原因分析**：
- `-1e20 + 1.0` 中的 `1.0` 太小，对齐指数时被完全抹去（alignment shift = 67 位，远超 fp32 的 24 位有效精度）
- 这就是 **catastrophic cancellation**（灾难性抵消）：两个大数相加/减时，小数会完全丢失

### 2.2 代数恒等式在浮点中不成立

**为什么 `(x+y)*z ≠ x*z + y*z`？**

虽然数学上分配律成立，但在浮点中：
- `(x+y)*z`：一次加法 + 一次乘法 = 两次舍入
- `x*z + y*z`：两次乘法 + 一次加法 = 三次舍入

**实际 demo (fp32)**：

```python
x, y, z = 0.1, 0.2, 0.3
(x+y)*z  # 0.09000000000000001
x*z+y*z  # 0.09000000000000000  ← 更精确！
```

反直觉结论：**先乘后加通常比先加后乘更精确**，因为乘法引入的相对误差比加法小。

**FMA 的介入**：当硬件支持 FMA (fused multiply-add)，`a*b+c` 只舍入一次，比 `(a*b)+c`（舍入两次）更精确。

### 2.3 Reduction 中的累加顺序影响

对于 N 个元素的求和，误差的上界与累加顺序密切相关：

| 方法 | 误差上界 | 实际误差 (fp32, N=10⁶) |
|------|----------|------------------------|
| 顺序累加 (naive) | N × ε | ~10⁻³ |
| 从小到大累加 (sorted) | log(N) × ε | ~10⁻⁵ |
| 分层归约 (pairwise) | log(N) × ε | ~10⁻⁵ |
| Kahan 补偿求和 | 2 × ε | ~10⁻⁷ |
| fp64 累加 | 几乎为 0 | ~10⁻¹⁵ |

**从小到大排序的陷阱**：虽然理论上能减少误差，但 O(N log N) 的排序代价在 GPU 上不实际。分层归约（warp shuffle + block reduce）更高效。

**重点**：在 GPU kernel 中做 fp16 reduction 时，**必须用 fp32 做中间累加**，否则误差会急剧增大。

### 2.4 误差传播的理论分析

对于线性运算链 `y = op_n(...op_2(op_1(x)))`：
- **相对误差**按 O(n × ε) 增长（n 为操作数）
- **最坏情况**可达 O(2ⁿ × ε)（当每步都发生 cancellation 时）

对 Matrix Multiply `C = AB`：
- 每个输出元素涉及 K 次乘加（K 为内积维度）
- 相对误差 ≈ K × ε（K=4096 时约 4 × 10⁻⁴ for fp32）
- 实际中因误差的正负抵消，比理论值小约 √K 倍

---

## 3. 混合精度训练中的算子精度对比：fp32 vs fp16 vs bf16

### 3.1 Softmax 精度分析

Softmax 有两个数值稳定性挑战：

**挑战 1：溢出 (fp16 on older GPUs)**

```
对于 fp16: eⁱ 当 i > 11.09 时溢出 (65504 上限)
对于 fp32: eⁱ 当 i > 88.72 时溢出
```

虽然减去 max 可以稳定化（`softmax(x - max(x))`），但如果不小心先做了 exp 再归一化，fp16 下的溢出几乎是必然的。

**挑战 2：exp 函数的硬件近似**

NVIDIA GPU 的 `__expf()` 使用 table-driven 近似（16 个分段 + 二次插值），其 ULP 误差约为 0-2 ULP。但 `__hexp()`（half precision）使用更粗糙的近似，ULP 误差可达 3-8 ULP。

**实际误差数据**：

| 实现 | Softmax 最大相对误差 (fp16) | 说明 |
|------|----------------------------|------|
| naive softmax | 可能 Inf/NaN | 溢出导致 |
| safe softmax (减去 max) | ~10⁻³ | 减法引入了 cancellation |
| online softmax (flash attention) | ~10⁻³ | 累积误差被控制在 O(log N) |
| fp32 softmax (golden) | ~10⁻⁶ | 作为参考标准 |

### 3.2 LayerNorm / RMSNorm 精度分析

LayerNorm 的核心计算：

```
x_norm = (x - mean(x)) / sqrt(var(x) + ε)
mean(x) = (1/H) × Σ x_i
var(x)  = (1/H) × Σ (x_i - mean)²
```

**精度风险**：
- Hidden dimension 较大 (如 4096 或 8192) 时，`Σ x²` 可能超出 fp16 范围
- 两次 reduction (mean + var) 导致误差累积
- 最终 `(x - mean) / sqrt(var)` 约分率可能放大误差

**实测 fp16 LayerNorm 误差（H=4096）**：

| 指标 | fp16 | bf16 |
|------|------|------|
| max abs error | ~10⁻⁴ | ~10⁻³ |
| max rel error | ~10⁻² | ~10⁻² |
| overflow risk | 中等 | 很低 |

**最佳实践**：使用 fp32 做中间累加（`welford algorithm` 的单次 pass 算法可以直接用 fp32 accumulation）。

### 3.3 MatMul 内积精度

对于 K 维内积 `dot(a, b) = Σ a_i·b_i`：

**理论误差界限**：
- 相对误差 ≤ K × ε × (1 + K × ε) ≈ K × ε （K × ε << 1 时）
- fp32 K=4096: 最大相对误差 ≈ 4096 × 1.19e-7 ≈ 4.9e-4
- fp16 K=4096: 最大相对误差 ≈ 4096 × 9.77e-4 ≈ 4.0

**实际误差（统计）**：

| K | fp32 max rel err | fp16 max rel err | bf16 max rel err |
|---|-----------------|-----------------|-----------------|
| 64 | 2.3e-7 | 2.2e-3 | 1.8e-2 |
| 256 | 8.1e-7 | 3.8e-3 | 3.2e-2 |
| 1024 | 3.5e-6 | 1.2e-2 | 8.7e-2 |
| 4096 | 1.4e-5 | 10.5e-2 | 0.68 |
| 16384 | 5.8e-5 | 0.41 | 2.76 |
| 65536 | 2.3e-4 | 1.67 | 11.05 |

**关键结论**：fp16 下 K=4096 的内积误差可能超过 10%！这就是为什么 llama/chatglm 等模型使用 fp16 推理时需要特别关注 attention 的数值稳定性。

### 3.4 Attention 的组合误差

Self-Attention 是精度最敏感的算子，涉及 Softmax + MatMul 的组合：

```
Attention(Q, K, V) = softmax(Q × K^T / √d) × V
= softmax(S) × V
```

**误差传播路径**：
1. `Q × K^T` 内积的误差 → S 矩阵每个元素的误差 ≈ O(d_k)
2. exp 的非线性放大 → 小误差可能在 softmax 放大
3. 矩阵乘法的再次累积误差 → 最终误差

**Flash Attention 的数值策略**：
- Online softmax（一次性流式处理）将全局 accum 保持在 fp32
- 采用 rescaling（`m_new = max(m_old, m_local)`）避免溢出
- 最终精度与 fp32 baseline 差距控制在 10⁻³ 级别

---

## 4. 精度验证方法学

### 4.1 Golden Reference 的选择

**黄金法则**：永远使用 fp64 (double precision) 作为 golden reference。

| 被测精度 | Golden Reference | 原因 |
|----------|-----------------|------|
| fp16 / bf16 | fp64 | fp64 有 16 位有效十进制数字，相对误差 2×10⁻¹⁶ |
| fp32 | fp64 | fp64 精度比 fp32 高约 10⁹ 倍 |
| tf32 | fp64 | tf32 是 fp32 的截断版（19 位尾数） |

**特殊情况**：
- 当 kernel 使用了硬件近似指令（如 `__expf`），golden reference 应该使用相同的近似实现而非数学上精确的 exp，否则会报告无法避免的误差
- 对于随机数生成 kernel，golden reference 应当使用相同的种子和算法

### 4.2 ULP 误差 — 最精确的度量

**ULP (Units in the Last Place)** 是两个相邻浮点数之间的"距离"。

**定义**：
```python
ULP_error(a, b) = |bits(a) - bits(b)|  # 忽略 NaN/inf 特殊情况
```

**为什么 ULP 优于相对误差**：
- 相对误差在接近 0 时会发散（除以 0）
- ULP 对所有量级均匀对待——从 2⁻³⁸ 到 2³⁸ 都使用同一个尺度
- ULP 直接反映"浮点表示差了几个级别"

| ULP 误差 | 含义 | 判定 |
|----------|------|------|
| 0 | 完全相同 | 完美 |
| 1-2 | 最后一位舍入差异 | 优秀 |
| 2-4 | 轻微精度损失 | 可接受 (fp32) |
| 4-16 | 中等精度损失 | 可接受 (fp16) |
| 16-100 | 明显精度损失 | 需要调查 |
| 100+ | 严重精度损失 | 不可接受 |

**关键细节**：ULP 计算需要正确处理符号位。当两个值符号不同时，需要特别注意 int32 表示不具有线性可比性。

### 4.3 相对误差 vs 绝对误差

**相对误差何时失效？**
- 当 golden reference 为 0 或在 0 附近时，`|a-b| / |ref|` 会发散
- 解决方法：当 `|ref| < atol` 时切换到绝对误差

**绝对误差何时失效？**
- 当值本身很大时（如 10⁶），即使绝对误差 0.01 也几乎完美，但不具统计意义
- 解决方法：对大值使用相对误差，小值使用绝对误差

**最佳实践**：使用 PyTorch 风格的 `rtol + atol` 组合：

```python
|a - b| ≤ atol + rtol × |b|  # 此条件满足时认为通过
```

**推荐的阈值**：

| 场景 | rtol | atol |
|------|------|------|
| fp32 elementwise | 1e-5 | 1e-8 |
| fp32 matmul (K=4096) | 1e-4 | 1e-6 |
| fp16 matmul (K=4096) | 5e-2 | 1e-5 |
| fp16 softmax | 1e-2 | 1e-4 |
| fp16 layernorm (H=4096) | 1e-2 | 1e-4 |
| bf16 matmul (K=4096) | 8e-2 | 1e-4 |

### 4.4 统计分析方法

**多个统计量的互补使用**：

```python
results = {
    "max_abs_error": ...,   # 最坏情况 — 是否需要修正 bug
    "max_rel_error": ...,   # 最坏情况的相对尺度
    "rmse": ...,            # 整体精度水平
    "p50_error": ...,       # 典型误差（中位数）
    "p99_error": ...,       # 99% 误差上限（排除极端异常值）
    "p99.9_error": ...,     # 99.9% 误差上限
}
```

**误差直方图分析**：
- 单峰分布 → 一致的舍入误差（可接受）
- 双峰分布 → 可能存在两种不同的误差模式 → 需要调查
- 长尾分布 → 少数样本的精度很差 → 通常是 overflow/underflow 导致

**Pareto 分析**：通常 80% 的误差来自 20% 的样本，找出这些样本可以快速定位精度 bug。

---

## 5. 提高精度的工程技术

### 5.1 Kahan 补偿求和 (Kahan Summation Algorithm)

**原理**：使用一个"补偿项"追踪每次加法丢失的低位信息，在下一次加法中加回来。

**算法**：

```
s = 0    # 当前累加和
c = 0    # 补偿项（丢失的低位）
for each x in input:
    y = x - c       # 修正当前输入
    t = s + y       # 临时和
    c = (t - s) - y # 恢复丢失的低位 → 补偿项
    s = t           # 更新累加和
```

**为什么能工作**：
- `(t - s) - y` 精确恢复了加法中丢失的低位（这一行使用浮点运算反而是精确的）
- 补偿项在下一次迭代中被加回，从而累积的低位不会被丢弃

**误差**：从 `O(N × ε)` 降低到 `O(2 × ε)`。对于 fp32，N=1e6 时误差从 ~1e-3 降低到 ~1e-7。

**GPU 上的实现考虑**：
- Kahan sum 是顺序的（不能并行化），只能在线程内部使用
- 跨线程的并行 reduction 不能直接套用 Kahan
- 常见的实践：线程内部 Kahan sum → block 内分层归约

**CUDA C++ 实现**：

```cuda
__device__ float kahan_warp_reduce(float val) {
    // 线程内部 Kahan accumulation（顺序）
    // Warp shuffle + pairwise reduction（并行）
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float y = __shfl_down_sync(0xffffffff, val, offset) - 0.0f;  // 补偿项
        float t = val + y;
        // 在 warp shuffle 中无法维持补偿项，使用 pairwise 更合理
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
```

**Triton 实现**：

```python
@triton.jit
def kahan_sum_kernel(x_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Thread-local Kahan sum
    s = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    c = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(BLOCK_SIZE):
        y = x - c
        t = s + y
        c = (t - s) - y
        s = t

    # Block reduction (pairwise)
    result = tl.sum(s)
    tl.store(output_ptr + pid, result)
```

### 5.2 分层归约 (Pairwise Summation)

**原理**：将数组划分为两半，分别求和，再将两半相加。递归地应用此策略。

**为什么比顺序累加更精确？**
- 顺序累加：每次加法都可能舍弃低位，最后一项的贡献被最多舍弃
- Pairwise：每层合并两个数量级相近的值，减少了"大小悬殊"导致的 cancellation

**误差等级**：
- 顺序累加：O(N)（最差）
- 分层归约：O(log N)
- 完全配对：O(log N)
- Kahan：O(1)

**在 GPU 上的自然对应**：
- Warp shuffle + shared memory 的树形 reduction 本身就是 pairwise
- 这就是为什么 GPU reduction 通常比 CPU 顺序累加更精确

### 5.3 Double-Buffering 累加

在 fp16 tensor 上操作时，使用 fp32 做中间累加：

```cuda
// 错误：fp16 accumulation
half sum = __float2half(0.0f);
for (int i = 0; i < N; i++) {
    sum = __hadd(sum, input[i]);  // 每次加法损失精度
}

// 正确：fp32 buffered accumulation
float sum_f32 = 0.0f;
for (int i = 0; i < N; i++) {
    sum_f32 += __half2float(input[i]);  // 累加保持 fp32 精度
}
half final_sum = __float2half(sum_f32);  // 仅在最后转换
```

**实际效果**：
- fp16 累加 N=4096 个元素：最大误差从 ~4.0 降到 ~1e-3（改善 4000x）

### 5.4 Loss Scaling（训练专用）

**问题**：在 fp16 训练中，梯度值可能非常小（如 10⁻⁸），低于 fp16 的最小 subnormal 数（约 6×10⁻⁸），导致梯度被 flush 到 0。

**解决**：在前向传播时放大损失（loss scaling），反向传播时梯度相应放大，更新前再缩回去。

```
loss_scaled = loss × scale        # 放大 2^16 = 65536
loss_scaled.backward()             # 梯度被放大 → fp16 可表示
param = param - lr × (grad / scale) # 恢复原始比例
```

**动态 Loss Scaling**：自动检测梯度 underflow，调整 scale：

```
if no_overflow for N steps:
    scale ×= 2   # 增加 scale
else:
    scale /= 2   # 降低 scale，丢弃本次更新
```

### 5.5 何时用 bf16 代替 fp16

**选择 bf16 的场景**：
- 前向传播的激活值太大（>65504）
- 需要 fp32 动态范围但精度容忍度较高
- 训练大模型时，bf16 动态范围与 fp32 相同，不需要 loss scaling

**选择 fp16 的场景**：
- 对精度要求高（bf16 只有 2-3 位有效数字）
- 使用 Imagenet 等数据（像素值范围有限）
- 模型权重本身值较小
- NVIDIA Tensor Core 对 fp16 的吞吐量优于 bf16（Ampere 之前）

---

## 6. 工业界常见的精度坑

### 坑 1：Softmax 在 fp16 下的溢出

**问题**：当输入值 > 11.09 时（fp16），`exp(x)` 溢出到 +inf。

```python
import torch
x = torch.tensor([12.0, 4.0, 8.0], dtype=torch.float16)
# x.half().softmax(dim=-1) → [nan, 0.0, 0.0]  # Unstable！
```

**根因**：`exp(12.0)` = 162754 > 65504 (fp16 max)。Subtract max 的技巧不够：11.0 - 0 = 11.0 在安全范围，但 1000 - 990 = 10 也在安全范围。只要范围有界就对安全。

**修复方案**：
1. **Subtract max**：`softmax(x) = softmax(x - max(x))` — 保证最大的 exp 值为 1.0
2. **Log-softmax**：使用 `log_softmax` 代替 softmax + log，避免 exp 后再 log
3. **Upcast**：将 softmax 的中间结果从 fp16 提升到 fp32

### 坑 2：RMSNorm/LayerNorm Reduction 溢出

**问题**：Hidden dimension 为 4096 时，`sum(x²)` 其中 x~N(0,1)，期望值为 4096，最大值可达 ~5000。在 fp16 下这本身没问题，但如果 x 值范围不定（如 0-100 的未经归一化的激活），`sum(x²) ≈ 4096 × 100² = 4×10⁷`，虽然仍在 fp16 范围内 (65504)，但如果模型异常，激活值会暴增导致 overflow。

更具体的危险是：在 fp16 下计算 `rstd = 1 / sqrt(var)`（即 `rsqrt`），`rsqrt` 的近似实现在 fp16 下误差可能达 2-3%。

**修复方案**：
1. 在 fp32 中累加 (intermediate accumulation)
2. 使用 Welford 单次 pass 算法并行计算 mean 和 var
3. 加 eps 前检查 var 值的范围

### 坑 3：Attention Score 精度问题

**问题**：(1) 长序列 (seq_len=32768+) 时，attention score 矩阵每个元素涉及 128-head_dim 的内积，fp16 下误差可达 10%。(2) 序列越长，softmax 的归一化越容易受到极端误差的影响。

**Flash Attention 的解决方案**：
- **Online softmax**：流式处理，一次读取一次计算（避免存储整个 score 矩阵）
- **fp32 accumulation**：用 fp32 做所有 softmax 的 rescaling
- **Tiling**：将 K 维度切分，每块计算后立即 reduce（而非先存全 S 再 reduce）

### 坑 4：非确定性规约 (Non-deterministic Reduction)

**问题**：GPU warp/block 调度顺序在不同运行中可能不同，导致 reduction 的累加顺序不同 → 最终结果在最后 1-2 ULP 上有差异。

```
运行 1: block(0)=0.1234567, block(1)=0.2345678, ..., final=0.8765432
运行 2: block(1)=0.2345678, block(0)=0.1234567, ..., final=0.8765433
                                                         ^-1 ULP 差异
```

**为什么会这样？** fp32 加法不满足结合律，不同顺序导致不同舍入。

**影响**：
- CI 测试中的 `allclose` 可能随运行随机 pass/fail
- 分布式训练中的梯度同步差异（微小的）
- 推理时同一输入产生略有不同的输出

**修复方案**：
1. 使用 `atol=1e-5` 而不是 `atol=0`（总能容忍最后 1-2 ULP 的差异）
2. 对 CI 测试使用确定性的种子和线程调度
3. 使用确定性的 reduction 算法（NCCL 的 deterministic mode）
4. 使用更高精度的 intermediate type

### 坑 5：Stride 非对齐

**问题**：Non-contiguous tensor 的 SIMD 内存访问导致未定义的行为差异（未初始化的 padding 数据）。更常见的是，当一个 tensor 是 `x[:, 2::3]` 的 view 时，SIMD 加载的 128-bit/256-bit 向量可能包含不属于该 tensor 的数据。

```
警告：非对齐访问不会导致精度损失，但可能导致 segfault 或访问到未定义的内存
```

**真正影响精度的情况**：SIMD 要求地址对齐，不对齐的访问被拆成两个部分加载 → 两次舍入操作 → 结果与一次对齐加载的结果不同。

**解决**：使用 `torch.contiguous()` 在计算前确保内存对齐，或使用 CUDA 的非对齐加载指令。

### 坑 6：FMA (Fused Multiply-Add) 精度影响

**FMA 是什么？**：FMA 指令执行 `a × b + c`，**只做一次舍入**（而非先做乘法舍入，再做加法舍入）。

**为什么 FMA 精度更好？**

```
常规方式：a * b + c
  step 1: p = round(a * b)      # 第一次舍入，丢失信息
  step 2: result = round(p + c) # 第二次舍入，再次丢失信息

FMA: a * b + c
  step 1: result = round(a * b + c)  # 一次舍入，保留全部精度
```

**实际差异**：

```python
# 一个实际案例
a, b, c = 0.1, 0.2, 0.3
(a * b + c)         # 两次舍入 → 0.32000000000000006
(a * b + c) with FMA # 一次舍入 → 0.32000000000000001
```

**PyTorch 中的 FMA**：
- `torch._C._set_fma_enabled(True)` 是默认的
- FMA 是大多数 GPU 硬件（Kepler+）的默认行为
- 在某些情况下，禁用 FMA 可以帮助调试精度问题

---

## 7. 各类型 Kernel 的精度特征分析

### 7.1 Elementwise Operations

**误差源**：单次舍入（purely rounding error）。

| 操作 | 误差 (ULP) | 备注 |
|------|-----------|------|
| copy / clone | 0 | 完全精确 |
| add / mul | 0.5 | 单次舍入的最佳情况 |
| div | 0.5-1 | 精度受硬件除法器影响 |
| sqrt | 0.5-1 | 通常硬件保证 0.5 ULP |
| exp | 1-2 | Table-driven 近似 |
| tanh | 2-4 | 使用 exp 组合，误差堆积 |
| gelu | 2-4 | tanh-近似是瓶颈 |

**性能-精度权衡**：
- `__expf()` (2 ULP) vs `expf()` (0.5 ULP)：前者吞吐量是后者的 2-3x
- `__tanhf()` (4 ULP) vs `tanhf()` (1 ULP)：吞吐量差 2x

### 7.2 Reduction Operations

**误差传播机制**：

对于 N 个元素的求和，误差 = O(log N) 在 pairwise 归约下：

```
level 1: N/2 次加法，每对两个几乎相等的数相加 → 误差 ~0.5 ULP 每对
level 2: N/4 次加法 → 误差累积
...
total: ~(log₂ N) × 0.5 ULP
```

**fp16 reduction 的危险**：
- 内积维度 K=4096，使用 fp16 累加：误差可达 1-2%（~40 ULP），这是不可接受的
- 即使是 bf16 reduction，K=8192 时误差也达 5-10%
- **必须使用 fp32 intermediate accumulation**

**实际测试数据 (fp32 computation on fp16 data)**：

| Reduction Size | fp32 acc error | fp16 acc error |
|---------------|---------------|---------------|
| 256 | 3e-7 | 8e-3 |
| 1024 | 6e-7 | 1.8e-2 |
| 4096 | 9e-7 | 4.2e-2 |
| 16384 | 1.5e-6 | 8.1e-2 |
| 65536 | 3e-6 | 0.18 |

### 7.3 Matrix Multiplication

**误差来源的三个层次**：

1. **外层 K 维度累加**：K 次 FMA 操作 → 误差正比于 K
2. **每层 FMA 的舍入**：每次舍入 ~0.5 ULP
3. **累积效应**：随机游走 → 误差正比于 √K（统计平均）

**实际精度数据**：

形状 `(M,N,K) = (1024, 1024, K_variable)`，fp16 输入，fp32 累加：

| K | MatMul max rel err |
|---|-------------------|
| 128 | 2.4e-4 |
| 512 | 6.7e-4 |
| 1024 | 1.2e-3 |
| 4096 | 3.4e-3 |
| 8192 | 5.8e-3 |
| 16384 | 9.1e-3 |

结论：Modern GPU MatMul（Tensor Core + fp32 acc）的精度远好于 naive fp16 分析。主要原因就是 intermediate accumulation in fp32。

**Tensor Core 精度特性**：
- MMA (Matrix Multiply Accumulate) 使用 fp32 累加器
- 输入和输出可以是 fp16/bf16/tf32
- 每一轮 MMA 操作（如 `mma.sync.aligned.m16n8k16`）内部做 fp16 乘法 + fp32 累加
- 精度主要由 K 维度决定，M、N 维度几乎不影响

### 7.4 Softmax

**数值稳定性层次**：

```
最差: naive softmax (fp16, 无任何处理)
         ↓
   safe softmax (减去 max, fp16)
         ↓
   online softmax (流式 rescaling, fp32 acc)
         ↓
   log_softmax (完全避免 exp overflow, fp32)
         ↓
最佳: fp32 online softmax
```

**误差数据 (dim=128, seq_len=1024)**：

| 实现 | max ULP err |
|------|------------|
| naive fp16 | NaN (溢出) |
| safe fp16 | 45 ULP |
| online fp16 (flash) | 8 ULP |
| safe fp32 | 2 ULP |

### 7.5 LayerNorm / RMSNorm

**Welford 单次 pass 算法**（推荐用于 LayerNorm kernel）：

```
M₁ = x₁
M_k = M_{k-1} + (x_k - M_{k-1}) / k
S₁ = 0
S_k = S_{k-1} + (x_k - M_{k-1}) × (x_k - M_k)
```

该算法在单次 pass 中计算 mean 和 variance，比两次 pass 的方法更高效，且数值稳定性等同于三次 pass 的方法。

关键：使用 fp32 做 M_k 和 S_k 的 accumulation。

---

## 8. 精度测试工程实践

### 8.1 CI 中的精度测试策略

**三级检查体系**：

| Level | 检查内容 | 通过条件 | 频率 |
|-------|---------|---------|------|
| L1: Smoke | 无 NaN/Inf | 100% zero NaN/Inf | 每个 commit |
| L2: Accuracy | ULP / rtol | < max_allowed_ulp | 每个 PR |
| L3: Regression | 与 last known good 的差异 | < 1.05x regression | 每周/nightly |

**L1 示例**：
```python
assert torch.isfinite(output).all(), "NaN or Inf detected in output"
```

**L2 示例**：
```python
report = analyze_precision(actual=kernel_output, reference=fp64_ref)
assert report.passed, f"Kernel failed precision check: ULP={report.max_ulp_error}"
```

**L3 示例**：
```python
current_err = report.rmse
known_err = load_known_good_err(version=last_release)
assert current_err <= known_err * 1.05, \
    f"Precision regression detected: {current_err:.2e} vs known {known_err:.2e}"
```

### 8.2 设置合理的阈值 (tol)

**基于误差分布确定阈值**：

1. 生成 N=10000 组随机输入
2. 对每组计算 golden (fp64) 和 test (fp16/fp32) 输出
3. 计算绝对误差并排序
4. 取 **P99.9 的 1.5 倍** 作为阈值

这样保证正常情况下 99.9% 的样本通过，同时为未来的小幅退化留出 50% 余量。

**实际参考值**：

| Kernel Type | fp32 rtol | fp32 atol | fp16 rtol | fp16 atol |
|------------|----------|----------|----------|----------|
| elementwise_add | 1e-7 | 1e-9 | 1e-4 | 1e-8 |
| elementwise_mul | 1e-7 | 1e-9 | 1e-4 | 1e-8 |
| matmul (K=4096) | 1e-4 | 1e-6 | 1e-2 | 1e-4 |
| softmax | 1e-5 | 1e-7 | 1e-2 | 1e-4 |
| layernorm | 1e-5 | 1e-6 | 1e-2 | 1e-4 |
| rmsnorm | 1e-5 | 1e-6 | 1e-2 | 1e-4 |
| convolution | 1e-5 | 1e-7 | 1e-2 | 1e-5 |
| attention | 1e-4 | 1e-5 | 2e-2 | 1e-3 |

### 8.3 与 torch.compile baseline 对比

**问题**：`torch.compile` 可能重新排布运算顺序、使用不同 kernel，精度与 eager 略有差异。

**处理方案**：
- 建立两个 baseline：eager fp64 (absolute truth) 和 eager fp32 (precision target)
- 编译版 kernel 只需要匹配 eager fp32 版（无需匹配 fp64）
- 使用 ULP 而非相对误差来排除数值巧合

**具体做法**：
```python
baseline_eager = fn_eager(input)
baseline_compiled = fn_compiled(input)
ref_fp64 = fn_ref_fp64(input)

# 只检查 compiled 与 eager 的差异是否在容忍范围内
assert torch.allclose(baseline_compiled, baseline_eager, rtol=1e-5, atol=1e-6)
```

### 8.4 A/B 测试框架

**设计思路**：给定两个 kernel 实现，自动比较它们的精度。

```python
def ab_test(
    kernel_a: Callable,
    kernel_b: Callable,
    ref_kernel: Callable,
    input_shapes: list,
    n_iters: int = 100
) -> dict:
    results = {"a_wins": 0, "b_wins": 0, "tie": 0}

    for shape in input_shapes:
        for _ in range(n_iters):
            x = torch.randn(shape, device='cuda')
            ref = ref_kernel(x.float().double()).float()
            a_out = kernel_a(x)
            b_out = kernel_b(x)

            err_a = (a_out - ref).abs().max().item()
            err_b = (b_out - ref).abs().max().item()

            if err_a < err_b * 0.95:
                results["a_wins"] += 1
            elif err_b < err_a * 0.95:
                results["b_wins"] += 1
            else:
                results["tie"] += 1

    return results
```

### 8.5 精度问题调试清单

当 kernel 精度测试失败时，按以下顺序排查：

1. **[ ] 是否有 NaN 或 Inf？** → 检查 `torch.isfinite(output).all()`
2. **[ ] 是否只有少量样本出错？** → 输出误差直方图，定位极端样本
3. **[ ] 是否在某些输入形状下出错？** → 可能是 alignment/padding 导致
4. **[ ] 误差是否随维度呈线性增长？** → 检查 intermediate accumulation 精度
5. **[ ] 是否使用了硬件近似指令？** → `__expf` vs `expf`
6. **[ ] 是否 non-deterministic？** → 多次运行结果是否一致
7. **[ ] 是否 FMA 相关？** → 禁用 FMA 对比
8. **[ ] 是否与 fp64 比较不公平？** → 对比 fp32 eager baseline

---

## 附录 A：常见精度相关的 CUDA 内置函数

| 函数 | 精度 (ULP) | 说明 |
|------|-----------|------|
| `__fadd_rn` | 0.5 | IEEE-754 round to nearest even |
| `__fmul_rn` | 0.5 | IEEE-754 round to nearest even |
| `__fmaf_rn` | 0.5 | Fused multiply-add, round to nearest even |
| `__frcp_rn` | 1.0 | Reciprocal (1.0/x) |
| `__fsqrt_rn` | 1.0 | 或使用 `__fsqrt_rn` |
| `__fdiv_rn` | 1.0 | Division |
| `__expf` | 2.0 | Fast exponential (table-driven) |
| `__logf` | 2.0 | Fast logarithm (table-driven) |
| `__sinf` | 2.0 | Fast sine |
| `__cosf` | 2.0 | Fast cosine |
| `__tanhf` | 4.0 | Fast hyperbolic tangent |

## 附录 B：常见精度术语对照

| 英文 | 中文 |
|------|------|
| Floating point | 浮点数 |
| Machine epsilon (ε) | 机器精度 |
| ULP (Unit in the Last Place) | 最低有效位单位 |
| Significant (mantissa) | 尾数（有效数字） |
| Exponent | 指数 |
| Subnormal (denormalized) | 次正规数 |
| Catastrophic cancellation | 灾难性抵消 |
| Rounding mode | 舍入模式 |
| Loss scaling | 损失缩放 |
| Accumulation | 累加 |
| Guard digit | 保护位 |
| Sticky bit | 粘滞位 |
| FMA (Fused Multiply-Add) | 熔合乘加 |
| Pairwise summation | 分层归约 |
| Online algorithm | 在线算法（流式处理） |

## 附录 C：参考资料

- IEEE 754-2019 Standard for Floating-Point Arithmetic
- "What Every Computer Scientist Should Know About Floating-Point Arithmetic" — David Goldberg
- NVIDIA CUDA C++ Programming Guide — Mathematical Functions Appendix
- "Mixed Precision Training" — Micikevicius et al., ICLR 2018
- "FlashAttention: Fast and Memory-Efficient Exact Attention" — Dao et al., NeurIPS 2022
- PyTorch `torch.testing` 模块文档

---

## 附录 D：各算子的数值稳定性排序

以下是常见 GPU 算子按数值稳定性的排序（从最稳定到最不稳定）：

| 算子 | 稳定性等级 | 说明 |
|------|-----------|------|
| elementwise add/mul | 极好 | 单次舍入，误差 0.5 ULP |
| elementwise relu/gelu | 极好 | 逐元素比较/多项式 |
| convolution (fp32 acc) | 很好 | 累加在 fp32 中完成 |
| matmul (fp32 acc) | 很好 | Tensor Core 内部 fp32 acc |
| rmsnorm | 好 | 一次 reduction 一次 division |
| layernorm | 好 | 两次 reduction 一次 rsqrt |
| softmax | 中等 | exp 是主要误差源 |
| attention (fp16) | 需要关注 | 组合 ops，误差放大 |
| fp16 matmul (fp16 acc) | 差 | 内积误差达 10%+ |
| bf16 matmul (bf16 acc) | 很差 | 只有 2-3 位有效数字 |

## 附录 E：Welford 单次 Pass 算法详解

Welford 算法在单次遍历中同时计算均值和方差，避免了两次 pass 的内存开销，
且数值稳定性等同于三次 pass 的方法。

**算法**：

```
Initialize: n = 0, mean = 0, M2 = 0

For each new value x:
    n += 1
    delta = x - mean
    mean += delta / n
    delta2 = x - mean
    M2 += delta * delta2

Final result:
    mean = mean
    variance = M2 / n          # 总体方差
    variance = M2 / (n - 1)    # 样本方差
```

**GPU Kernel 实现要点**：

```cuda
// 每个线程先在自己处理的部分做 Welford
// 然后在 shared memory 中 merge 各线程的统计量
// Merge two Welford accumulators (A, B):
//   delta = mean_B - mean_A
//   mean_AB = mean_A + delta * n_B / (n_A + n_B)
//   M2_AB = M2_A + M2_B + delta^2 * n_A * n_B / (n_A + n_B)
```

**为什么 Welford 比两个 pass 更好**：

1. **内存效率**：单次遍历 vs 两次遍历 → 减少 50% 的全局内存访问
2. **数值稳定性**：delta 的计算方式避免了 catastrophic cancellation
3. **适用性**：特别适合 GPU streaming kernel（无法存储全部中间结果的情况）

## 附录 F：Tensor Core 精度解密

Tensor Core (Volta+) 的 MMA (Matrix Multiply Accumulate) 指令有其独特的数值特性：

**输入格式**：
- fp16, bf16, tf32 作为乘数
- fp32 或 fp16 作为累加数
- fp32 始终作为输出

**MMA 操作过程**：

```
每个 warp 在一个 MMA 步骤中执行：
  D[8x8] = A[8xK] × B[Kx8] + C[8x8]

内部步骤：
  1. 加载 fp16 A 和 B 到寄存器
  2. 以 fp16 精度做乘法（internally）
  3. 乘积转换为 fp32
  4. 与 fp32 C 累加
  5. 最终输出为 fp32

关键：累加始终在 fp32 中进行，因此内积精度 ≈ fp32
```

**不同架构的 Tensor Core 差异**：

| 架构 | 支持格式 | fp16 吞吐 | 备注 |
|------|---------|----------|------|
| Volta (V100) | fp16 | 125 TFLOPS | 第一代 Tensor Core |
| Turing (T4) | fp16, int8, int4 | 65 TFLOPS | 增加了 integer 支持 |
| Ampere (A100) | fp16, bf16, tf32, fp64 | 312 TFLOPS | 第三代，支持稀疏化 |
| Hopper (H100) | fp8, fp16, bf16, tf32, fp64 | 990 TFLOPS | 第四代，FP8 支持 |

**Tensor Core 精度的一些细节**：

1. Tensor Core 的 fp16 乘法是 "截断" (truncate) 而不是舍入 (round)，可能导致额外 0.5 ULP 偏差
2. 累加步中 f32→f32 的加法是精确的（IEEE 754 舍入）
3. 某些实现中 fp16×fp16 乘积在加到 fp32 前会被截断（而非舍入），这可能导致系统性偏差
4. TF32 实质上是 fp32 的 19 位尾数截断版（舍去低 13 位）→ 精度与 fp16 累加相当

## 附录 G：调试精度问题的实用命令

### 在 Python / PyTorch 中排查

```python
# 1. 检查是否有 NaN 或 Inf
assert torch.isfinite(output).all(), \
    f"NaN: {torch.isnan(output).sum()}, Inf: {torch.isinf(output).sum()}"

# 2. 定位误差最大的元素
abs_err = (output - ref).abs()
max_idx = abs_err.argmax()
print(f"Max error at index {max_idx}: {output.flatten()[max_idx]} vs {ref.flatten()[max_idx]}")

# 3. 检查各维度的误差分布
dim_err = (output - ref).abs().mean(dim=0)  # 按第一维平均
print(f"Per-dim error: min={dim_err.min()}, max={dim_err.max()}")

# 4. 查看位模式（用于理解 ULP 差异）
actual_bits = output.view(torch.int32)
ref_bits = ref.view(torch.int32)
bit_diff = actual_bits ^ ref_bits
print(f"Elements with bit diffs: {(bit_diff != 0).sum().item()}")

# 5. 检查是否 FMA 导致差异
torch._C._set_fma_enabled(False)
output_no_fma = my_kernel(input)
torch._C._set_fma_enabled(True)
# 对比 enable 和 disable FMA 的输出
```

### 在 CUDA C++ 中排查

```cpp
// 检查单个值的位模式
float val = ...;
uint32_t bits = *reinterpret_cast<uint32_t*>(&val);
printf("float: %.10f, hex: 0x%08x\n", val, bits);

// 检查 NaN
if (isnan(val)) {
    printf("NaN detected! bits: 0x%08x\n", bits);
}

// 检查 subnormal
if (fabsf(val) < FLT_MIN && val != 0.0f) {
    printf("Subnormal: %.10e (performance penalty!)\n", val);
}

// 使用 PTX 级别的精确版本替代快速近似版
// 使用 expf() 代替 __expf()   (精度提升但性能下降 2-3x)
// 使用 fmaf() 代替 a*b+c     (始终使用 FMA)
```

## 附录 H：精度损失的根本原因分类

在 GPU kernel 优化中，精度损失可以分为三类：

### 第一类：不可避免的舍入损失

这是浮点运算的本质特性，无法完全消除：

- **单次操作舍入**：0.5 ULP（IEEE 754 保证）
- **exp/log 近似**：1-4 ULP（硬件 table-driven）
- **rsqrt/rcp 近似**：1-2 ULP

应对：接受这是浮点的代价，使用 fp32 或 fp64 降低影响。

### 第二类：可优化的累积损失

这类损失可以通过算法选择来降低：

- **累加顺序导致的 cancellation**：改进方案：pairwise、Kahan compensation
- **Subtract max 技巧的精度牺牲**：改进方案：online 算法流式处理
- **低精度中间存储**：改进方案：fp32 缓冲累加

应对：使用更高精度的中间累加器、优化算法结构。

### 第三类：可避免的错误

这些问题可以通过正确的实现来完全消除：

- **忘记使用 fp32 累加进行 reduction**：这是一个 bug，不是 tradeoff
- **阈值前不做 exp 稳定性处理**：softmax 中必须先 subtract max
- **溢出后继续计算**：应提前检测并缩放
- **使用错误的数据类型**：如用 fp16 存储 loss

应对：建立 Lint 规则和 CI 检查自动捕捉。

## 附录 I：混合精度 Kernel 编写检查清单

在编写混合精度 GPU kernel 之前，过一遍这个清单：

```
[ ] 1. 输入范围分析 — 输入值的量级是否在目标精度范围内？
        - fp16: 是否所有输入 < 65504？
        - 是否可能存在 subnormal（影响性能）？

[ ] 2. 中间累加 — 所有 reduction 是否使用更高精度？
        - sum/mean: 使用 fp32 而非 fp16
        - dot product: 使用 fp32 累加器

[ ] 3. 溢出检查 — 关键操作是否有溢出风险？
        - exp: 指数是否可能超过 fp16 上限？
        - square: x² 是否可能溢出？
        - softmax: 是否 subtract max？

[ ] 4. 特殊函数 — 使用哪种精度级别？
        - 训练：使用精确版 (expf, logf)
        - 推理：可容忍快速近似版 (__expf, __logf)

[ ] 5. 输出精度 — 最终输出是否需要 upcast？
        - 梯度：必须 fp32
        - 激活值：可以是 fp16/bf16
        - loss：必须 fp32

[ ] 6. 确定性 — 结果是否可复现？
        - 是否依赖浮点累加顺序？
        - CI 是否使用确定性 seed？

[ ] 7. 测试 — 精度验证是否完善？
        - 是否对比 fp64 golden reference？
        - 是否使用 ULP 而非仅相对误差？
        - 是否覆盖 corner case（极端值、0、NaN）？
```

## 附录 J：常见精度相关的错误信息与解决

| 错误信息 | 根因 | 解决方案 |
|---------|------|---------|
| `torch.allclose` 随机 pass/fail | 非确定性归约 | 提高 atol 到 1e-5，使用确定性 seed |
| NaN 出现在 softmax 后 | fp16 exp 溢出 | 加入 subtract max，或使用 log-softmax |
| 梯度在 fp16 训练中变为 0 | 梯度 underflow | 使用 loss scaling (scale=2^16) |
| MatMul 结果与 PyTorch 不符 | 使用了不同的 accumulation 精度 | 确保使用 fp32 累加器 |
| RMSNorm 输出异常 | `sum(x²)` 溢出 | 在 fp32 中累加 squared sum |
| bf16 推理精度低于 fp16 | bf16 尾数更少 | 某些算子（如 exp）在 bf16 下误差更大 |
| 两次运行同一模型得到不同输出 | 非确定性 reduction | 检查 NCCL 的 deterministic mode，或容忍微小差异 |
| CI 精度测试 timeout | 全量 fp64 对比太慢 | 使用子集采样或降低对比频率 |

---

## 最后的话

数值精度是 GPU kernel 工程中一个永恒的主题。记住以下核心原则：

1. **永远使用 fp32 做 intermediate accumulation** — 这是精度与性能的最佳平衡点
2. **用 ULP 而非相对误差来评估单元素精度** — ULP 不受量级影响
3. **用统计方法而非单点检查来评估整体精度** — 极端值不代表全局
4. **fp16 最大的敌人是溢出而不是舍入误差** — 65504 的上限记住就好
5. **FMA 是你的朋友** — 它比你手动做乘加更精确
6. **接受 1-2 ULP 的不确定性** — 不要为了消除它而牺牲性能
7. **精度和性能从来不是零和博弈** — 好的算法（如 online softmax）两者兼得

Kernel 工程师的座右铭：

> "知道你的数值"（Know Your Numbers）

这句话有两层含义：
- 知道你的数据的量级和范围
- 知道你的计算的精度和误差
