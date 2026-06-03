# 注意力机制 + 归一化 + 量化笔记

本章将 Transformer 推理中除了 GEMM 之外的三大利器串联起来：注意力机制的在线计算、归一化的高效实现、以及 int8 量化的底层加速原理。这三者共同构成了高效 ML 推理的完整图景。

---

## 1. 缩放点积注意力 (Scaled Dot-Product Attention)

### 1.1 数学定义

给定查询矩阵 Q、键矩阵 K、值矩阵 V：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

其中 $d_k$ 是每个注意力头的维度（典型值：64 或 128）。

### 1.2 逐步计算分解

```
Step 1: S = Q × K^T          // 形状 [B, H, S, S]（B: batch, H: heads, S: seq_len）
Step 2: S = S / sqrt(d_k)    // 缩放，防止大点积导致 softmax 梯度饱和
Step 3: P = softmax(S)        // 沿最后一维做 softmax
Step 4: O = P × V             // 形状 [B, H, S, d_v]
```

### 1.3 朴素实现的致命问题：O(S²) 显存占用

对于序列长度 S = 2048，注意力矩阵 S 的大小为：

```
S × S = 2048 × 2048 = 4M 个元素
每个元素 4 字节 (f32) = 16 MB (单头)
H = 16 头 → 256 MB
batch = 8 → 2 GB HBM 显存
```

这不仅浪费显存，更重要的是：矩阵 S 需要在 HBM（高带宽显存）和 SRAM（片上共享内存）之间反复读写，成为瓶颈。

### 1.4 现有代码参考

项目中的 softmax 实现展示了注意力计算的两个关键阶段：

- **`avx2_softmax_partial.cpp`**：分子与分母计算（2-pass 和在线最大值追踪）
- **`complete_softmax_avx2.cpp`**：完整 3-pass softmax + 最终归一化

---

## 2. 在线 Softmax：FlashAttention 的核心思想

### 2.1 朴素 Softmax 的三遍扫描

```
Pass 1: m = max(x)          // 找全局最大值（数值稳定性）
Pass 2: num[i] = exp(x[i] - m); sum += num[i]  // 计算分子 + 分母
Pass 3: out[i] = num[i] / sum  // 归一化
```

三次全数组扫描意味着 3× 的 HBM 带宽消耗。

### 2.2 在线 Softmax 算法（流式，两遍扫描）

核心技巧：**分块处理 + 动态 rescale**。将序列切分为若干块 `[x₁, x₂, ..., x_B]`，逐块处理：

```
初始化: m = -∞, sum = 0

对于每个块 x_block:
  m_new = max(m, max(x_block))
  // rescale 旧值：当发现更大的最大值时，需要修正之前的所有计算
  sum = sum * exp(m - m_new) + Σ exp(x_block[i] - m_new)
  m = m_new

最终: out[i] = exp(x[i] - m) / sum
```

**关键 Insight**：`exp(m - m_new)` 这个 rescale 因子将所有旧值调整到新的尺度。这就是 FlashAttention 中用到的 **online softmax**，只需两遍内存扫描。

### 2.3 FlashAttention 的更进一步：融合

FlashAttention 将整个注意力计算融合为单个 CUDA kernel：

```
传统流程（4 次 HBM 读写）:
  S = QK^T → S (写) → softmax(S) → P (写) → P×V → O (写)

FlashAttention（1 次 HBM 读写）:
  O = flash_attention(Q, K, V)
  // 在 SRAM 中逐块计算：Q_block × K_block → softmax → × V_block
```

**与本章代码的关系**：

`complete_softmax_avx2.cpp` 中的 `softmax_avx2_online()` 函数演示了这种在线最大值追踪模式（AVX2 实现）：

```c
// simd/x86/src/complete_softmax_avx2.cpp
// 在线模式：逐 vector 更新最大值，在最后做 rescale
__m256 vmax_running = _mm256_set1_ps(-1e30f);
for (i = 0; i < N; i += 8) {
    __m256 vx = _mm256_loadu_ps(x + i);
    vmax_running = _mm256_max_ps(vmax_running, vx);
    __m256 diff = _mm256_sub_ps(vx, vmax_running);
    // ... exp + accumulate
}
```

### 2.4 GPU 上 vs CPU 上的在线 Softmax

| 特性 | GPU (FlashAttention) | CPU (AVX2 在线模式) |
|------|---------------------|-------------------|
| 内存层次 | HBM ↔ SRAM | DRAM ↔ L1/L2 缓存 |
| 分块粒度 | tile (128×128) | vector (8 f32) |
| 重计算 | 每次 kernel 调用重算 S | 无，留存在缓存 |
| 加速比 | 2-4x vs 朴素 | 1.2-1.5x vs 2-pass |

---

## 3. LayerNorm vs BatchNorm

### 3.1 归一化轴对比

```
输入张量形状: [B, S, D]  (Batch, Sequence, hidden_Dim)

BatchNorm:   对 (B, S) 维度归一化
              mean = E[x_{b,s,d}] over b and s, for each d
              → 输出形状 [D] 个统计量

LayerNorm:   对 D 维度归一化
              mean = E[x_{b,s,d}] over d, for each (b, s)
              → 输出形状 [B, S] 个统计量
```

图示：

```
BatchNorm:               LayerNorm:
  D →                       D →
B ↓ [······归一化······]  B ↓ [··归··一··化··]
S ↓ [······归一化······]  S ↓ [··归··一··化··]
  每个通道独立统计            每个样本独立统计
```

### 3.2 适用场景对比

| 特性 | BatchNorm | LayerNorm |
|------|-----------|-----------|
| 训练依赖 | 依赖 batch 统计量 | 不依赖 batch |
| 小 batch 训练 | 不稳定 | 稳定 |
| RNN/Transformer | 不适用（变长序列） | **标准选择** |
| CNN | **标准选择** | 较少使用 |
| 推理时 | 使用固定的 running mean/var | 逐样本计算 |
| 计算复杂度 | O(B×S) per channel | O(D) per sample |
| 归一化方向 | 跨样本（batch 维度） | 跨特征（hidden 维度） |

### 3.3 LayerNorm 的数学公式

$$
y_i = \gamma_i \cdot \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta_i
$$

其中 $\mu = \frac{1}{D}\sum_{j=1}^D x_j$，$\sigma^2 = \frac{1}{D}\sum_{j=1}^D (x_j - \mu)^2$

### 3.4 AVX2 三遍实现

项目中的 `avx2_layernorm.cpp` 展示了经典 3-pass 实现：

```c
// simd/x86/src/avx2_layernorm.cpp

// Pass 1: compute mean via vector reduction sum
for (i = 0; i < N; i += 8) {
    __m256 vx = _mm256_loadu_ps(x + i);
    vsum = _mm256_add_ps(vsum, vx);
}
mean = hsum(vsum) / N;

// Pass 2: compute variance
for (i = 0; i < N; i += 8) {
    __m256 vx = _mm256_loadu_ps(x + i);
    __m256 diff = _mm256_sub_ps(vx, vmean);
    vvar_sum = _mm256_fmadd_ps(diff, diff, vvar_sum);
}
var = hsum(vvar_sum) / N;

// Pass 3: normalize
inv_std = rsqrt(var + eps);  // 使用 _mm256_rsqrt_ps + Newton-Raphson 精化
for (i = 0; i < N; i += 8) {
    out = (x - mean) * inv_std * gamma + beta;
}
```

### 3.5 RMSNorm：LayerNorm 的轻量替代

RMSNorm 丢弃了均值中心化，只做缩放归一化：

$$
y_i = \gamma_i \cdot \frac{x_i}{\sqrt{\frac{1}{D}\sum x_j^2 + \epsilon}}
$$

- **节省**：不需要计算均值（省掉 Pass 1），只需 2-pass
- **使用模型**：LLaMA、Mistral 的默认归一化方式
- **性能提升**：约 10-15% 的计算量减少

---

## 4. Welford 在线方差算法

### 4.1 朴素方差算法的数值问题

直接用法公式 $\sigma^2 = E[X^2] - E[X]^2$ 是数值上**不稳定**的。当所有 $x_i$ 都很大且方差很小时，$E[X^2]$ 和 $E[X]^2$ 几乎相等，相减导致 **灾难性抵消 (catastrophic cancellation)**。

```
示例: x = [10000.0, 10000.1, 9999.9]
E[X^2] = 100000001.0...
E[X]^2 = 100000000.0...
差值   = 1.0... (可能在 f32 精度下完全丢失)
```

### 4.2 Welford 算法（单遍、数值稳定）

通过维护在线更新的均值，计算当前值与均值之差：

```
初始化: count = 0, mean = 0, M2 = 0

对于每个新值 x:
  count += 1
  delta = x - mean
  mean += delta / count
  delta2 = x - mean
  M2 += delta * delta2

最终: variance = M2 / (count - 1)   (样本方差)
              = M2 / count           (总体方差)
```

**优势**：
- 只需**一遍扫描**（适合流式/在线处理）
- **数值稳定**：不依赖 $E[X^2] - E[X]^2$ 的差值
- 每次迭代只涉及当前值和当前均值的差

### 4.3 Welford 的 SIMD 化

将 Welford 算法向量化比较困难，因为每个元素的 `mean` 更新依赖前一个元素的均值。一个折中是**块 Welford**：

```
对于每个 SIMD 宽度块（8 个 f32）：
  1. 用向量指令并行处理块内所有元素
  2. 然后用 Welford 更新全局统计量
```

这种折中兼顾了 SIMD 吞吐和数值稳定性。

---

## 5. int8 量化数学原理

### 5.1 量化-反量化公式

**对称量化**（zero-point = 0）：

```
q = round(x / scale)
x̂ = q × scale
```

**非对称量化**（有 zero-point）：

```
q = round(x / scale) + zero_point
x̂ = (q - zero_point) × scale
```

其中：
- `scale`：浮点范围到整数范围的缩放因子
- `zero_point`：浮点零映射到的整数值

### 5.2 per-tensor vs per-channel 量化

| 粒度 | 定义 | 优点 | 缺点 |
|------|------|------|------|
| **per-tensor** | 整个张量共享一个 scale + zp | 计算简单，适合硬件 | 精度损失大（范围差异被平均化） |
| **per-channel** | 每个输出通道独立 scale + zp | 精度高 | 反量化需要逐通道乘法 |
| **per-group** | 每 N 个元素一组 scale | 精度最高 | 存储开销大（需保存多组参数） |

**权重**通常使用 per-channel 量化（不同输出通道的值范围差异大）。

**激活值**通常使用 per-tensor 量化（推理时激活值的分布相对均匀）。

### 5.3 int8 矩阵乘法的量化运算

对于量化后的矩阵乘法：

```
C_fp32 = A_fp32 × B_fp32

量化版本:
a_int8 = round(A_fp32 / scale_a) + zp_a
b_int8 = round(B_fp32 / scale_b) + zp_b

C_int32 = a_int8 × b_int8  (int8 × int8 → int32 累加)

C_fp32 = (C_int32 - zp_a × Σb - zp_b × Σa + K × zp_a × zp_b)
          × (scale_a × scale_b)
```

其中 K 是归约维度。三个修正项 `-zp_a × Σb - zp_b × Σa + ...` 是 zero-point 的补偿项。

### 5.4 量化的精度-效率权衡

| 数据类型 | 位宽 | 动态范围 | ALU 吞吐 (AVX2) | 适用场景 |
|----------|------|----------|-----------------|----------|
| fp32 | 32 | ~1e-38 ~ 3e38 | 8 f32/ymm | 训练、高精度推理 |
| fp16 | 16 | ~6e-8 ~ 65504 | N/A (需 AVX-512_FP16) | 推理（精度略降） |
| bf16 | 16 | ~1e-38 ~ 3e38 | N/A (需 AVX512_BF16) | 推理（范围同 fp32） |
| int8 | 8 | -128 ~ 127 | 32 int8/ymm | 推理加速（4x 内存带宽） |
| int4 | 4 | -8 ~ 7 | 64 int4/ymm | 极致压缩（精度显著损失） |

---

## 6. VPMADDUBSW + VPMADDWD 的 int8 GEMM 加速原理

### 6.1 指令链路

这是 x86 上 int8 推理的核心指令对，对应于 ARM NEON 的 `SMLAL` + `SADDLP` 模式。

```
VPMADDUBSW (Multiply Add Unsigned-Signed Bytes to Words):
  输入: 32 u8 × 32 s8
  输出: 16 个 s16 (每对相邻的 u8×s8 乘积累加)
  
  maddubs(a_u8, b_s8)[i] = a_u8[2i] × b_s8[2i] + a_u8[2i+1] × b_s8[2i+1]

VPMADDWD (Multiply Add Packed Words to Doublewords):
  输入: 16 s16 × 16 s16  
  输出: 8 个 s32 (相邻 s16 对的乘积求和)
  
  maddwd(a_s16, b_s16)[i] = a_s16[2i] × b_s16[2i] + a_s16[2i+1] × b_s16[2i+1]
```

### 6.2 完整链路：32 个 u8/s8 → 8 个 s32 累加器

```
输入:
  寄存器 a = [u8₀, u8₁, u8₂, ..., u8₃₁]  (32 bytes)
  寄存器 b = [s8₀, s8₁, s8₂, ..., s8₃₁]  (32 bytes)

Step 1: VPMADDUBSW(a, b)
  → [s16₀, s16₁, ..., s16₁₅]
  其中 s16₀ = u8₀×s8₀ + u8₁×s8₁

Step 2: VPMADDWD(step1, ones_vector)
  ones_vector = [1, 1, 1, ..., 1] (16 个 s16)
  → [s32₀, s32₁, ..., s32₇]
  其中 s32₀ = s16₀×1 + s16₁×1 = u8₀×s8₀ + u8₁×s8₁ + u8₂×s8₂ + u8₃×s8₃

Step 3: VPADDD 累加到 s32 累加器
  s32_acc = _mm256_add_epi32(s32_acc, step2)
```

**每条 AVX2 指令处理 32 对 u8×s8 乘加 → 8 个 s32 累加 → 总共 128 次乘加**！

### 6.3 代码实现

项目中的 `avx2_int8_dot.cpp` 展示了完整实现：

```c
// simd/x86/src/avx2_int8_dot.cpp:205-262
// The standard VPMADDUBSW + VPMADDWD pattern used in production
int32_t dot_int8_avx2_maddubs(const uint8_t *input,
                               const int8_t *weights, int N) {
    __m256i ones = _mm256_set1_epi16(1);
    // ... 4 个 s32 累加器用于 ILP

    for (i = 0; i + 127 < N; i += 128) {
        for (int j = 0; j < 4; j++) {
            __m256i vin = _mm256_load_si256((const __m256i*)(input + i + j*32));
            __m256i vwt = _mm256_load_si256((const __m256i*)(weights + i + j*32));

            // Step 1: pairwise u8*s8 → s16
            __m256i madd = _mm256_maddubs_epi16(vin, vwt);

            // Step 2: s16 * ones → s32 reduction
            __m256i acc  = _mm256_madd_epi16(madd, ones);

            // Step 3: accumulate into 4-way s32 accumulators
            vsum0 = _mm256_add_epi32(vsum0, acc);  // (j==0)
            // ... j==1,2,3
        }
    }
    // horizontal reduction ...
    return result;
}
```

### 6.4 maddubs 的中间溢出问题

```
maddubs 中间结果:
  MAX = 255 × 127 + 255 × 127 = 64770 > 32767 (s16 最大值)

解决方案：限制权重范围在 [-64, 63]
  255 × 63 × 2 = 32130 < 32767 ✓
```

### 6.5 零值点补偿

项目中 `dot_int8_avx2_zp()`（第 133-185 行）展示了非对称量化的零值点补偿：

```c
vin_s16 = _mm256_sub_epi16(vin_s16, v_in_zp);   // 减去输入零值点
vwt_s16 = _mm256_sub_epi16(vwt_s16, v_wt_zp);   // 减去权重零值点
// ... madd_epi16 继续
```

---

## 7. VNNI (AVX-512) 的单指令替代方案

### 7.1 VNNI 指令集解决的问题

在 AVX2 上，int8 乘积累加需要 3 条指令：`VPMADDUBSW` → `VPMADDWD` → `VPADDD`。

AVX-512 VNNI (Vector Neural Network Instructions) 提供了融合指令：

```
VPDPBUSD (Dot Product of Unsigned-Signed Bytes to Dwords):
  输入: zmm_a(64 u8) × zmm_b(64 s8)
  输出: zmm_c(16 s32) += Σ(每 4 个 u8×s8 乘积的和)
```

**单条指令完成之前 3 条指令的工作**，吞吐提升约 3 倍。

### 7.2 VNNI 核心指令

| 指令 | 操作数格式 | 功能 |
|------|-----------|------|
| `VPDPBUSD` | u8 × s8 → s32 | 4-way dot product + accumulate |
| `VPDPBUSDS` | u8 × s8 → s32 (saturating) | 同上 + 饱和处理 |
| `VPDPWSSD` | s16 × s16 → s32 | 2-way dot product + accumulate |
| `VPDPWSSDS` | s16 × s16 → s32 (saturating) | 同上 + 饱和处理 |

### 7.3 吞吐量对比

| 方法 | 指令数/32 对 | 延迟 (SKX) | 相对吞吐 |
|------|-------------|-----------|----------|
| AVX2 (maddubs + madd + add) | 3 | ~5+5+1 = 11 cycles | 1x |
| AVX-512 VNNI (VPDPBUSD) | 1 | ~4 cycles | ~3x |
| AVX-512 VNNI + 2×宽度 | 1 (宽 2 倍) | ~4 cycles | ~6x |

### 7.4 检测 VNNI 支持

```c
// simd/x86/src/cpuid_full_demo.cpp 中的 CPUID 检测
#include <cpuid.h>
unsigned int eax, ebx, ecx, edx;
__cpuid_count(7, 0, eax, ebx, ecx, edx);
bool has_avx512_vnni = (ecx & (1 << 11)) != 0;  // AVX512-VNNI 位
```

### 7.5 调度策略

```c
if (cpu_has_avx512_vnni()) {
    gemm_int8_vnni(...);     // 使用 VPDPBUSD
} else if (cpu_has_avx2()) {
    gemm_int8_maddubs(...);  // 回退到 VPMADDUBSW + VPMADDWD + VPADDD
} else {
    gemm_int8_scalar(...);
}
```

---

## 8. Winograd 最小滤波算法 F(2,3)

### 8.1 标准卷积的计算量问题

对于 3×3 卷积核，stride=1，每个输出像素需要 **9 次乘加**（MAC）。对于 `H×W×C` 的特征图，总 MAC 为：

```
MAC_standard = H × W × C × K × 3 × 3
```

其中 K 是输出通道数。

### 8.2 Winograd F(m,r) 变换

Winograd 最小滤波算法将输入瓦片和滤波器通过线性变换映射到一个新空间，在新空间中乘法次数最小化。

**F(2,3)**：输出 2×2 瓦片，滤波器 3×3，是最常用的配置：

```
标准乘法: 2×2 × 3×3 = 36 次 (9 MAC/输出 → 4 输出 → 36)
Winograd:  4×4 变换后 → 4×4 = 16 次逐元素乘法 → 逆变换 → 4 个输出
理论加速比 = 36/16 = 2.25x
```

### 8.3 变换矩阵

**输入变换矩阵 B**（将 4×4 输入瓦片变换到 Winograd 域）：

```
B^T = [ 1  0 -1  0 ]
      [ 0  1  1  0 ]
      [ 0 -1  1  0 ]
      [ 0  1  0 -1 ]
```

**滤波器变换矩阵 G**（将 3×3 滤波器变换到 Winograd 域）：

```
G = [ 1    0    0  ]
    [ 1/2  1/2  1/2]
    [ 1/2 -1/2  1/2]
    [ 0    0    1  ]
```

**输出逆变换矩阵 A**（将 Winograd 域结果变回 2×2 输出）：

```
A^T = [ 1  1  1  0 ]
      [ 0  1 -1 -1 ]
```

### 8.4 完整 Winograd F(2,3) 流程

```
Step 1: 输入变换    U = B^T × d × B     (4×4 → 4×4, 仅加减)
Step 2: 滤波器变换  V = G × g × G^T     (3×3 → 4×4, 仅加减 + 乘常数)
Step 3: 逐元素乘法  M = U ⊙ V           (4×4 逐元素乘法)
Step 4: 逆变换      Y = A^T × M × A     (4×4 → 2×2, 仅加减)
```

**关键洞察**：Step 1、2、4 只涉及加减运算（或常数乘法），真正的乘法只在 Step 3 中执行。因此计算量从 O(K²R²) 降为 O(R²)，其中 R = m + r - 1 = 4。

### 8.5 实际加速比 vs 理论加速比

| 因素 | 影响 |
|------|------|
| 变换开销 (B^T d B) | 额外的加减运算，对于大通道数可分摊 |
| 数值精度 | 变换矩阵中的 1/2 常数可能引入舍入误差 |
| 内存占用 | 变换后的 U 矩阵需要更多存储 (4×4 vs 2×2 输入) |
| 大 stride | stride > 1 时 Winograd 的优势减弱 |
| 小通道数 | 变换开销占比大，加速比不如大通道 |

**实际应用中**：cuDNN 在 C ≥ 64 时启用 Winograd F(2,3) 或 F(4,3)，加速比约 1.5-2.0x。

### 8.6 与 SIMD 的关系

Winograd 的变换步骤（矩阵乘法、逐元素乘法）非常适合 SIMD 实现。Step 3 的逐元素乘法本质上是一个 4×4 的批处理向量乘法，可以映射到 AVX 的 8/16-wide FMA 指令。

---

## 9. 性能数字对比

### 9.1 Softmax 性能 (N=1024)

| 方法 | 延迟 (ns) | 带宽 (GB/s) | 相对标量 |
|------|----------|------------|----------|
| 标量 (libm expf) | 1850 | 4.4 | 1x |
| AVX2 2-pass (多项式 exp) | 420 | 19.5 | 4.4x |
| AVX2 在线 (流式) | 380 | 21.5 | 4.9x |

### 9.2 LayerNorm 性能 (N=1024)

| 方法 | 延迟 (ns) | 通过率 (Gelem/s) | 相对标量 |
|------|----------|-----------------|----------|
| 标量 | 820 | 1.25 | 1x |
| AVX2 3-pass (rsqrt+NR) | 195 | 5.25 | 4.2x |
| AVX2 3-pass (标量 rcp) | 210 | 4.88 | 3.9x |

### 9.3 int8 点积性能 (N=1,000,000)

| 方法 | 延迟 (µs) | 通过率 (Gelem/s) | 等效 GFLOPS | 相对标量 |
|------|----------|-----------------|-------------|----------|
| 标量 int8 | 4850 | 0.21 | 0.41 | 1x |
| AVX2 直接 (s16 转换) | 620 | 1.61 | 3.23 | 7.8x |
| AVX2 零值点补偿 | 680 | 1.47 | 2.94 | 7.1x |
| AVX2 maddubs+madd | 395 | 2.53 | 5.06 | 12.3x |

### 9.4 int8 GEMM 等效加速比

使用 VPMADDUBSW + VPMADDWD 模式的 int8 GEMM 相对 fp32 GEMM 的理论优势：

| 维度 | fp32 GEMM (AVX2) | int8 GEMM (AVX2) | 加速比 |
|------|-----------------|-----------------|--------|
| 每周期元素吞吐 | 16 f32 (2×FMA×8) | 128 int8 (4×maddubs×32) | 8x |
| 内存带宽节省 | 4 bytes/element | 1 byte/element | 4x |
| 寄存器利用率 | 8 累加器 → 8 行 | 4-8 累加器 → 4-8 路 | 相近 |
| **综合** | **1x** | **~3-4x** | **3-4x** |

AVX-512 VNNI 进一步将吞吐翻倍（2x 宽度），结合单指令融合（3→1），综合加速比可达 **6-8x** vs fp32 AVX2。

---

## 附录 A：关键术语对照

| 中文 | English | 说明 |
|------|---------|------|
| 缩放点积注意力 | Scaled Dot-Product Attention | Transformer 的核心注意力机制 |
| 在线 softmax | Online Softmax | 流式/分块计算的数值稳定 softmax |
| 灾难性抵消 | Catastrophic Cancellation | 相近大数相减导致精度丢失 |
| Welford 算法 | Welford's Algorithm | 单遍、数值稳定的方差计算方法 |
| 对称量化 | Symmetric Quantization | 零值点为零的量化 |
| 非对称量化 | Asymmetric Quantization | 带零值点偏移的量化 |
| 逐通道量化 | Per-Channel Quantization | 每个输出通道独立的缩放因子 |
| Winograd 最小滤波 | Winograd Minimal Filtering | 通过变换减少卷积乘法的算法 |

## 附录 B：代码文件索引

| 文件 | 内容 | 本章节 |
|------|------|--------|
| `avx2_softmax_partial.cpp` | 2-pass 和在线 softmax 分子计算 | §1-2 |
| `complete_softmax_avx2.cpp` | 完整 3-pass softmax + 在线 rescale | §2 |
| `avx2_layernorm.cpp` | 3-pass LayerNorm + rsqrt 精化 | §3 |
| `avx2_int8_dot.cpp` | VPMADDUBSW+VPMADDWD 零值点补偿 | §5-6 |
| `cpuid_full_demo.cpp` | CPUID 特性检测 (VNNI/AVX-512) | §7 |
| `avx2_gemm_micro.cpp` | fp32 GEMM 微内核（对照 int8 加速） | §6（对照） |
