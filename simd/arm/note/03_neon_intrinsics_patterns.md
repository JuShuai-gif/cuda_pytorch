# NEON Intrinsics 实战模式

本章介绍 7 种在 NEON 开发中反复出现的编程模式。每个模式包含：问题的描述、NEON 实现、性能考量、常见陷阱。

---

## 模式 1：Map（逐元素操作）

**问题**：对数组中的每个元素应用函数 f，输出同形状数组。

```
输入:  [a0 a1 a2 a3 ... aN-1]
输出: [f(a0) f(a1) f(a2) f(a3) ... f(aN-1)]
```

### 1.1 基本算术 Map

```c
// 向量加法: C = A + B
void map_add(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i <= n - 16; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t a3 = vld1q_f32(a + i + 12);

        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        float32x4_t b2 = vld1q_f32(b + i + 8);
        float32x4_t b3 = vld1q_f32(b + i + 12);

        vst1q_f32(c + i,      vaddq_f32(a0, b0));
        vst1q_f32(c + i + 4,  vaddq_f32(a1, b1));
        vst1q_f32(c + i + 8,  vaddq_f32(a2, b2));
        vst1q_f32(c + i + 12, vaddq_f32(a3, b3));
    }
    for (; i < n; i++) c[i] = a[i] + b[i];
}

// expected speedup: 3-4x on A76
```

### 1.2 ReLU 激活函数

```c
// ReLU(x) = max(0, x)
inline float32x4_t relu_f32x4(float32x4_t x) {
    const float32x4_t zero = vdupq_n_f32(0.0f);
    return vmaxq_f32(x, zero);
}

void map_relu(const float* src, float* dst, int n) {
    for (int i = 0; i <= n - 16; i += 16) {
        float32x4_t x0 = vld1q_f32(src + i);
        float32x4_t x1 = vld1q_f32(src + i + 4);
        float32x4_t x2 = vld1q_f32(src + i + 8);
        float32x4_t x3 = vld1q_f32(src + i + 12);

        vst1q_f32(dst + i,      relu_f32x4(x0));
        vst1q_f32(dst + i + 4,  relu_f32x4(x1));
        vst1q_f32(dst + i + 8,  relu_f32x4(x2));
        vst1q_f32(dst + i + 12, relu_f32x4(x3));
    }
    for (; i < n; i++) dst[i] = (src[i] > 0.0f) ? src[i] : 0.0f;
}
```

**替代写法**（使用位选择）：

```c
inline float32x4_t relu_v2(float32x4_t x) {
    const float32x4_t zero = vdupq_n_f32(0.0f);
    uint32x4_t mask = vcgeq_f32(x, zero);  // x >= 0 ?
    return vbslq_f32(mask, x, zero);
}
// 这两种写法性能相同。选择更易读的。
```

### 1.3 Clamp

```c
// clamp(x, lo, hi) = min(max(x, lo), hi)
inline float32x4_t clamp_f32x4(float32x4_t x, float32x4_t lo, float32x4_t hi) {
    return vminq_f32(vmaxq_f32(x, lo), hi);
}
```

### 1.4 Sigmoid 近似

精确的 sigmoid 是 `1/(1+exp(-x))`，但 `exp` 在 NEON 中没有直接指令。生产中使用多项式或有理近似：

```c
// sigmoid 近似: y = 0.5 + x * (1 - |x|) / 8,  for |x| in [-2, 2]
// 对于超出范围的 x，用比较+选择钳制

float32x4_t sigmoid_approx_f32x4(float32x4_t x) {
    const float32x4_t half   = vdupq_n_f32(0.5f);
    const float32x4_t eighth = vdupq_n_f32(0.125f);  // 1/8
    const float32x4_t two    = vdupq_n_f32(2.0f);
    const float32x4_t zero   = vdupq_n_f32(0.0f);
    const float32x4_t one    = vdupq_n_f32(1.0f);

    float32x4_t  abs_x = vabsq_f32(x);
    uint32x4_t   in_range = vcleq_f32(abs_x, two);  // |x| <= 2

    float32x4_t approx = vmlsq_f32(half, x, vmulq_f32(abs_x, eighth));
    // approx ≈ 0.5 + x * (1 - 0.125 * |x|)

    // 超出范围的钳制: 负时→0, 正时→1
    uint32x4_t pos_mask = vcgeq_f32(x, two);
    float32x4_t result  = vbslq_f32(pos_mask, one, zero);  // pos→1, neg→0
    result = vbslq_f32(in_range, approx, result);           // in range→approx

    return result;
}
```

对于更高精度需求（如 ML 推理），可使用编译期预计算的多项式系数表：

```c
// 更精确的 sigmoid，使用 3 阶多项式拟合
// 分段拟合: [0, 1], [1, 3], [3, 5], [5, +∞)
// 这里简化展示核心模式
float32x4_t sigmoid_poly_f32x4(float32x4_t x) {
    // 多项式系数 (针对 x >= 0 范围)
    const float32x4_t c0 = vdupq_n_f32(0.5f);
    const float32x4_t c1 = vdupq_n_f32(0.2125f);
    const float32x4_t c3 = vdupq_n_f32(-0.004f);

    float32x4_t x2 = vmulq_f32(x, x);
    float32x4_t x3 = vmulq_f32(x2, x);

    float32x4_t y = vmlaq_f32(c0, c1, x);    // c0 + c1*x
    y = vmlaq_f32(y, c3, x3);                 // + c3*x^3

    // Sigmoid 是对称的: σ(-x) = 1 - σ(x)
    uint32x4_t neg_mask = vcltq_f32(x, vdupq_n_f32(0.0f));
    float32x4_t neg_result = vsubq_f32(vdupq_n_f32(1.0f), y);
    return vbslq_f32(neg_mask, neg_result, y);
}
```

---

## 模式 2：Reduce（归约）

**问题**：将数组归约为一个或少数几个值（sum, max, dot product, etc.）。

```
输入:  [a0 a1 a2 ... aN-1]
输出:  scalar (sum/max/dot product)
```

### 2.1 向量的水平和

```c
// 归约求和
float vector_sum_neon(const float* data, int n) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    int i;
    for (i = 0; i <= n - 16; i += 16) {
        acc0 = vaddq_f32(acc0, vld1q_f32(data + i));
        acc1 = vaddq_f32(acc1, vld1q_f32(data + i + 4));
        acc2 = vaddq_f32(acc2, vld1q_f32(data + i + 8));
        acc3 = vaddq_f32(acc3, vld1q_f32(data + i + 12));
    }

    // 合并 4 个累加器
    acc0 = vaddq_f32(acc0, acc1);
    acc2 = vaddq_f32(acc2, acc3);
    acc0 = vaddq_f32(acc0, acc2);

    float sum = vaddvq_f32(acc0);  // 只在最后做一次 vaddv

    // 标量尾部
    for (; i < n; i++) sum += data[i];
    return sum;
}
```

**性能考量**：
- 4个累加器是为了隐藏 `vaddq_f32` 的 4 周期延迟
- `vaddvq_f32` 调用次数从 O(n/4) 降为 O(1)（归约操作仅一次）
- 预期加速：3-4x

### 2.2 浮点点积

```c
// dot(A, B) = Σ A[i] * B[i]
float dot_product_neon(const float* a, const float* b, int n) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    int i;
    for (i = 0; i <= n - 16; i += 16) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i),      vld1q_f32(b + i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4),  vld1q_f32(b + i + 4));
        acc2 = vfmaq_f32(acc2, vld1q_f32(a + i + 8),  vld1q_f32(b + i + 8));
        acc3 = vfmaq_f32(acc3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }

    acc0 = vaddq_f32(acc0, acc1);
    acc2 = vaddq_f32(acc2, acc3);
    acc0 = vaddq_f32(acc0, acc2);

    float dot = vaddvq_f32(acc0);

    for (; i < n; i++) dot += a[i] * b[i];
    return dot;
}
```

**注意**：`vfmaq_f32(acc, a, b)` 执行 `acc = acc + a * b`（fused multiply-add，无中间舍入）。对于需要精确位一致性的程序，`vfma`（fused）和 `vmla`（非 fused）的结果在最后几位可能不同。

### 2.3 Int8 点积（ARMv8.2+）

```c
// 仅在支持 vdotq_s32 的 CPU 上可用
// vdotq_s32: 将两个 int8x16 的对应元素相乘，各取 4 个乘积之和作为 32-bit 结果
//
// vdotq_s32(acc, a8x16, b8x16):
//   acc[0] += a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
//   acc[1] += a[4]*b[4] + a[5]*b[5] + a[6]*b[6] + a[7]*b[7]
//   acc[2] += a[8]*b[8] + a[9]*b[9] + a[10]*b[10] + a[11]*b[11]
//   acc[3] += a[12]*b[12] + a[13]*b[13] + a[14]*b[14] + a[15]*b[15]

int32_t dot_product_int8_neon(const int8_t* a, const int8_t* b, int n) {
    int32x4_t acc = vdupq_n_s32(0);
    int i;

    for (i = 0; i <= n - 16; i += 16) {
        int8x16_t va = vld1q_s8(a + i);
        int8x16_t vb = vld1q_s8(b + i);

        // 一条指令完成 16 个乘积累加到 4 个 32-bit 累加器
        acc = vdotq_s32(acc, va, vb);
    }

    int32_t result = vaddvq_s32(acc);

    for (; i < n; i++) result += (int32_t)a[i] * (int32_t)b[i];
    return result;
}

// 预期加速: 8-10x vs 标量 (Cortex-A76)
// 因为 vdotq_s32 每周期吞吐 1 条，每条 = 16 MAC = 32 ops/cycle
```

### 2.4 最大值归约

```c
float vector_max_neon(const float* data, int n) {
    float32x4_t max_vec = vld1q_f32(data);
    int i;
    for (i = 4; i <= n - 4; i += 4) {
        max_vec = vmaxq_f32(max_vec, vld1q_f32(data + i));
    }
    float result = vmaxvq_f32(max_vec);
    for (; i < n; i++) {
        if (data[i] > result) result = data[i];
    }
    return result;
}
```

---

## 模式 3：Filter/Select（条件筛选）

**问题**：根据条件筛选或替换元素。

### 3.1 替换 NaN

```c
void replace_nan_neon(float* data, int n, float replacement) {
    const float32x4_t repl = vdupq_n_f32(replacement);
    for (int i = 0; i <= n - 4; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        // NaN != NaN → 对 NaN: vceqq 返回 0 (false)
        uint32x4_t is_nan = vmvnq_u32(vceqq_f32(v, v));  // NOT(==) → NaN → 1
        v = vbslq_f32(is_nan, repl, v);
        vst1q_f32(data + i, v);
    }
}
```

### 3.2 阈值过滤

```c
// 保留大于 threshold 的元素，小于的记录为0
void threshold_above_neon(float* data, int n, float threshold) {
    const float32x4_t thresh = vdupq_n_f32(threshold);
    const float32x4_t zero   = vdupq_n_f32(0.0f);
    for (int i = 0; i <= n - 4; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        uint32x4_t  mask = vcgtq_f32(v, thresh);  // v > threshold ?
        v = vbslq_f32(mask, v, zero);              // 如果大于则保持，否则归零
        vst1q_f32(data + i, v);
    }
}

// 双阈值（带内）过滤
void in_range_filter_neon(float* data, int n, float lo, float hi) {
    const float32x4_t vlo = vdupq_n_f32(lo);
    const float32x4_t vhi = vdupq_n_f32(hi);
    const float32x4_t zero = vdupq_n_f32(0.0f);
    for (int i = 0; i <= n - 4; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        uint32x4_t mask_lo = vcgeq_f32(v, vlo);
        uint32x4_t mask_hi = vcleq_f32(v, vhi);
        uint32x4_t mask = vandq_u32(mask_lo, mask_hi);  // lo <= v <= hi
        v = vbslq_f32(mask, v, zero);
        vst1q_f32(data + i, v);
    }
}
```

---

## 模式 4：卷积

### 4.1 一维卷积（kernel 大小 3）

```
对于每个输出位置 t:
  y[t] = w0 * x[t-1] + w1 * x[t] + w2 * x[t+1]
```

**寄存器轮转技术**（避免重复加载）：

```c
// 寄存器轮转法实现 1D 卷积
void conv1d_k3_neon(const float* x, int n, const float w[3], float* y) {
    float32x4_t w0 = vdupq_n_f32(w[0]);
    float32x4_t w1 = vdupq_n_f32(w[1]);
    float32x4_t w2 = vdupq_n_f32(w[2]);

    // 加载前3个向量
    float32x4_t x_prev = vld1q_f32(x);       // t=0..3
    float32x4_t x_curr = vld1q_f32(x + 1);   // t=1..4
    float32x4_t x_next;                       // 迭代中填充

    for (int t = 0; t < n - 4; t += 3) {
        x_next = vld1q_f32(x + t + 2);  // t+2..t+5

        // y[t+0..t+3] = w0 * x[t-1..t+2] + w1 * x[t..t+3] + w2 * x[t+1..t+4]
        float32x4_t y_vec = vmulq_f32(w1, x_curr);
        y_vec = vfmaq_f32(y_vec, w0, x_prev);
        y_vec = vfmaq_f32(y_vec, w2, x_next);

        vst1q_f32(y + t + 1, y_vec);

        // 轮转寄存器：prev ← curr, curr ← next
        x_prev = x_curr;
        x_curr = x_next;
    }

    // 处理边界（第一个和最后一个输出位置）
    if (n >= 2) {
        y[0] = w[1]*x[0] + w[2]*x[1];
        y[n-1] = w[0]*x[n-2] + w[1]*x[n-1];
    }
}
```

**寄存器轮转的优势**：
- 每个输出只需要加载 4 个 float（一次 `vld1q`），而不是 12 个
- 缓存友好：数据流平滑前进
- 适用于更大的 kernel：kernel 大小 5 需要 5 个寄存器

---

## 模式 5：量化 Int8 推理

### 5.1 量化卷积中的 Int8 点积

```c
// 量化方案: q = round(x / scale) + zero_point
// 卷积: y_int32 = Σ (qA - zpA) * (qB - zpB)
//       = Σ qA*qB - zpA*Σ qB - zpB*Σ qA + N*zpA*zpB

// NEON int8 推理的基本算子:
// vdotq_s32: 将 int8×16 乘积累加到 int32×4

// 矩阵乘法的一个 tile (使用预包装的权重)
void gemm_int8_4x4_neon(
    const int8_t* A_packed,   // 预包装的激活（int8，4行×K列）
    const int8_t* B_packed,   // 预包装的权重（int8，K列×4列）
    int32_t* C,                // 输出（int32，4×4）
    int K)
{
    int32x4_t c0 = vdupq_n_s32(0);
    int32x4_t c1 = vdupq_n_s32(0);
    int32x4_t c2 = vdupq_n_s32(0);
    int32x4_t c3 = vdupq_n_s32(0);

    #pragma GCC unroll 4
    for (int k = 0; k < K; k += 16) {
        int8x16_t a0 = vld1q_s8(A_packed + 0 * K + k);
        int8x16_t a1 = vld1q_s8(A_packed + 1 * K + k);
        int8x16_t a2 = vld1q_s8(A_packed + 2 * K + k);
        int8x16_t a3 = vld1q_s8(A_packed + 3 * K + k);

        int8x16_t b0 = vld1q_s8(B_packed + k * 4 + 0 * 16);
        int8x16_t b1 = vld1q_s8(B_packed + k * 4 + 1 * 16);
        int8x16_t b2 = vld1q_s8(B_packed + k * 4 + 2 * 16);
        int8x16_t b3 = vld1q_s8(B_packed + k * 4 + 3 * 16);

        // 4×4 dot products
        c0 = vdotq_s32(c0, a0, b0);
        c1 = vdotq_s32(c1, a1, b1);
        c2 = vdotq_s32(c2, a2, b2);
        c3 = vdotq_s32(c3, a3, b3);
    }

    // 存储结果
    vst1q_s32(&C[0], c0);
    vst1q_s32(&C[4], c1);
    vst1q_s32(&C[8], c2);
    vst1q_s32(&C[12], c3);
}
```

### 5.2 零点处理

```c
// 预计算零点修正项
// y_int32 = Σ qA∗qB - zpA*sum_qB - zpB∗sum_qA + k*zpA∗zpB
//
// 在运行时: sum_qB 和 sum_qA 可预计算（权重固定时）

void gemm_int8_with_zero_point(
    const int8_t* A, const int8_t* B,
    int32_t* C, int M, int N, int K,
    int32_t zpA, int32_t zpB,
    const int32_t* sum_A,   // 预计算: sum_A[i] = Σ A[i][k]
    const int32_t* sum_B)   // 预计算: sum_B[j] = Σ B[k][j]
{
    // ... GEMM 核心 ...

    // 零点修正
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i*N + j] += sum_A[i] * zpB;
            C[i*N + j] += sum_B[j] * zpA;
            C[i*N + j] -= K * zpA * zpB;
        }
    }
}
```

**关键优化**：`sum_A` 和 `sum_B` 可预计算并缓存。K * zpA * zpB 是常数。这是 NCNN/QNNPACK 等框架的标准做法。

---

## 模式 6：FP16 半精度

### 6.1 FP16 基本操作

```c
// fp16 优势: 2x 吞吐量, 1/2 内存带宽
// 限制: 需要 ARMv8.2+ 硬件支持

// fp32 → fp16 转换
float16x8_t vcvt_f16_f32(float32x4_t lo, float32x4_t hi);

// fp16 → fp32 转换
float32x4_t vcvt_f32_f16(float16x4_t a);  // 低64位

// fp16 点积
float dot_product_f16_neon(const __fp16* a, const __fp16* b, int n) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);

    for (int i = 0; i <= n - 16; i += 16) {
        // 加载 16 个 fp16 → 2 个 f32x4
        float16x8_t a_lo = vld1q_f16(a + i);
        float16x8_t a_hi = vld1q_f16(a + i + 8);
        float16x8_t b_lo = vld1q_f16(b + i);
        float16x8_t b_hi = vld1q_f16(b + i + 8);

        // 转换 + FMA
        acc0 = vfmaq_f32(acc0,
            vcvt_f32_f16(vget_low_f16(a_lo)),
            vcvt_f32_f16(vget_low_f16(b_lo)));
        acc1 = vfmaq_f32(acc1,
            vcvt_f32_f16(vget_high_f16(a_lo)),
            vcvt_f32_f16(vget_high_f16(b_lo)));
        // ... 对 hi 做同样的操作
    }
    return vaddvq_f32(vaddq_f32(acc0, acc1));
}
```

### 6.2 直接用 FP16 做矩阵乘法

```c
// ARMv8.2+ 有直接 FP16 FMA: vfmaq_f16
void gemm_fp16_4x4_neon(const __fp16* A, const __fp16* B, __fp16* C,
                         int M, int N, int K, int lda, int ldb, int ldc) {
    for (int i = 0; i < M; i += 4) {
        for (int j = 0; j < N; j += 4) {
            float16x8_t c00 = vdupq_n_f16(0);

            for (int k = 0; k < K; k += 8) {
                float16x8_t a0 = vld1q_f16(&A[i * lda + k]);
                float16x8_t b0 = vld1q_f16(&B[k * ldb + j]);
                // WARNING: 需要广播和 lane 乘法，此处简化
                c00 = vfmaq_f16(c00, a0, b0);
            }
            vst1q_f16(&C[i * ldc + j], c00);
        }
    }
}
```

**FP16 的工业应用**：
- 移动端 ML 推理（int8 量化之外的另一选择）
- 精度损失小，但带宽节省 2x
- Apple A11+ 的 Neural Engine、高通 Hexagon DSP 等多用 fp16

---

## 模式 7：循环优化技术

### 7.1 循环展开（Unrolling）

```c
// 不展开: 4 floats/iteration, 依赖链延迟限制
for (int i = 0; i < n; i += 4) {
    float32x4_t v = vld1q_f32(src + i);
    acc = vfmaq_f32(acc, v, weight);
}

// 2x 展开: 8 floats/iteration, 两个独立的 acc 隐藏延迟
for (int i = 0; i <= n - 8; i += 8) {
    float32x4_t v0 = vld1q_f32(src + i);
    float32x4_t v1 = vld1q_f32(src + i + 4);
    acc0 = vfmaq_f32(acc0, v0, w0);
    acc1 = vfmaq_f32(acc1, v1, w1);
}

// 4x 展开: 16 floats/iteration, 最优展开度
for (int i = 0; i <= n - 16; i += 16) {
    float32x4_t v0 = vld1q_f32(src + i);
    float32x4_t v1 = vld1q_f32(src + i + 4);
    float32x4_t v2 = vld1q_f32(src + i + 8);
    float32x4_t v3 = vld1q_f32(src + i + 12);
    acc0 = vfmaq_f32(acc0, v0, w0);
    acc1 = vfmaq_f32(acc1, v1, w1);
    acc2 = vfmaq_f32(acc2, v2, w2);
    acc3 = vfmaq_f32(acc3, v3, w3);
}
```

**展开度选择指南**：
- `vfmaq_f32` 延迟 = 4 cycles (A76) → 最少 4 个独立累加器
- 多于 4 个累加器：收益递减，寄存器压力增大
- 经验法则：展开到恰好隐藏 FMA 延迟的程度

### 7.2 软件流水（Software Pipelining）

将 load、compute、store 分离，使它们在流水线中重叠：

```c
// 软件流水 SAXPY: y = a*x + y
void saxpy_pipelined(int n, float a, const float* x, float* y) {
    const float32x4_t va = vdupq_n_f32(a);

    // 预取并加载第一批数据
    float32x4_t x0 = vld1q_f32(x);
    float32x4_t y0 = vld1q_f32(y);

    for (int i = 4; i <= n - 4; i += 4) {
        // 加载下一批数据（同时当前批正在计算）
        float32x4_t x1 = vld1q_f32(x + i);
        float32x4_t y1 = vld1q_f32(y + i);

        // 计算当前批
        y0 = vfmaq_f32(y0, x0, va);

        // 存储上一批
        vst1q_f32(y + i - 4, y0);

        // 轮转
        x0 = x1;
        y0 = y1;
    }
    // 存储最后一批
    y0 = vfmaq_f32(y0, x0, va);
    vst1q_f32(y + n - 4, y0);
}
```

这个技术对于受 load 延迟限制的代码（如内存带宽瓶颈的 SAXPY）收益有限（因为 load 延迟完全被硬件预取器隐藏）。但对于 L1 中的计算密集型代码，可提供 10-15% 的提升。

### 7.3 交错独立操作

```c
// 不当: 顺序执行 load1 → compute1 → store1 → load2 → compute2 → store2
// 每步等待上一步完成

// 更好: 交错 load/compute/store 减少停顿
float32x4_t compute_next(const float32x4_t* src, const float32x4_t* params) {
    // 提前加载（利用 load 延迟）
    float32x4_t v0 = vld1q_f32(src);
    float32x4_t v1 = vld1q_f32(src + 4);

    // 执行与 v0/v1 无关的数学运算
    float32x4_t p = vld1q_f32(params);  // 参数加载

    // v0 现在可用（load 延迟已过）
    p = vfmaq_f32(p, v0, v1);

    return p;
}
```

---

## NEON 性能杀手总结

| 问题 | 症状 | 解决方法 |
|------|------|----------|
| 每迭代都 vaddv/vmaxv | 延迟 3+ cycles，占用唯一执行口 | 累积多个向量，最后归约 |
| 过多 vld3/vld4 | 每指令 3-4 微操作，阻塞 load 端口 | 用 vld1 + 手动解交错 |
| 标量循环在 NEON 之外 | 分支预测开销，标量延迟 | 用 NEON compare+select 处理条件 |
| 非 SoA 布局 | 必须 gather（NEON 无法直接做） | 预转换为 SoA |
| 在浮点循环中混入整数 NEON | 端口竞争（A76） | 分循环或确保比例合理 |
| 循环中创建多余的向量常量 | 多余的 vdup/load，浪费指令 | 提到循环外 |
| 忽略对齐 | 跨 cache line/page 开销 | alignas(64) 或 aligned_alloc |

---

## 推荐：用 Godbolt 验证生成的汇编

对于关键的 NEON 循环，建议在 [godbolt.org](https://godbolt.org) 上用 ARM GCC 或 Clang 查看生成的汇编，验证：
1. 所有操作确实是 NEON 指令（没有回退到标量）
2. 循环展开了
3. 常量被提到了循环外
4. 寄存器分配合理（没有过度溢出到栈）

```bash
# 本地也行
arm-linux-gnueabihf-gcc -O3 -march=armv8-a+simd -S -fverbose-asm neon.c
# 或
aarch64-linux-gnu-gcc -O3 -march=armv8.2-a+simd -S -fverbose-asm neon.c
```

下一节进入 SVE 的 VLA 编程世界：谓词、无尾部循环、一次编写处处最优。
