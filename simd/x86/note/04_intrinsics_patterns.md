# x86 SIMD Intrinsic 生产级编程模式

## 1. Map（逐元素操作）

### 1.1 ReLU（Rectified Linear Unit）

```c
// AVX2 ReLU: f(x) = max(x, 0)
void relu_avx2(const float* src, float* dst, int n) {
    __m256 zero = _mm256_setzero_ps();
    int i;
    for (i = 0; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, _mm256_max_ps(v, zero));
    }
    for (; i < n; i++) dst[i] = src[i] > 0 ? src[i] : 0;
}

// AVX-512 ReLU with tail masking
void relu_avx512(const float* src, float* dst, int n) {
    __m512 zero = _mm512_setzero_ps();
    int i;
    for (i = 0; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(src + i);
        _mm512_storeu_ps(dst + i, _mm512_max_ps(v, zero));
    }
    if (i < n) {
        __mmask16 tail = (1u << (n - i)) - 1;
        __m512 v = _mm512_maskz_loadu_ps(tail, src + i);
        _mm512_mask_storeu_ps(dst + i, tail, _mm512_max_ps(v, zero));
    }
}
```

### 1.2 LeakyReLU

```c
// AVX2 LeakyReLU: f(x) = x if x > 0 else alpha * x
void leaky_relu_avx2(const float* src, float* dst, int n, float alpha) {
    __m256 zero = _mm256_setzero_ps();
    __m256 va = _mm256_set1_ps(alpha);
    for (int i = 0; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        __m256 mask = _mm256_cmp_ps(v, zero, _CMP_GT_OS);
        __m256 neg = _mm256_mul_ps(v, va);
        _mm256_storeu_ps(dst + i, _mm256_blendv_ps(neg, v, mask));
    }
}

// AVX-512 LeakyReLU with k-register mask
void leaky_relu_avx512(const float* src, float* dst, int n, float alpha) {
    __m512 zero = _mm512_setzero_ps();
    __m512 va = _mm512_set1_ps(alpha);
    for (int i = 0; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(src + i);
        __mmask16 pos = _mm512_cmp_ps_mask(v, zero, _CMP_GT_OQ);
        __m512 neg = _mm512_mul_ps(v, va);
        _mm512_storeu_ps(dst + i, _mm512_mask_mov_ps(neg, pos, v));
    }
}
```

### 1.3 Clamp

```c
__m256 clamp_avx2(__m256 x, float lo, float hi) {
    return _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(lo)), _mm256_set1_ps(hi));
}

__m512 clamp_avx512(__m512 x, float lo, float hi) {
    return _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(lo)), _mm512_set1_ps(hi));
}
```

### 1.4 GELU 近似（Transformer 关键算子）

GELU 公式: `gelu(x) ≈ 0.5x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))`

```c
// AVX2 GELU approximation using tanh
static inline __m256 gelu_avx2(__m256 x) {
    const float c1 = 0.7978845608028654f;   // sqrt(2/pi)
    const float c2 = 0.044715f;
    const float half = 0.5f;

    __m256 vc1 = _mm256_set1_ps(c1);
    __m256 vc2 = _mm256_set1_ps(c2);
    __m256 vhalf = _mm256_set1_ps(half);
    __m256 one = _mm256_set1_ps(1.0f);

    // x^2 and x^3
    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 x3 = _mm256_mul_ps(x2, x);

    // tanh(c1 * (x + c2 * x^3))
    __m256 inner = _mm256_fmadd_ps(vc2, x3, x);    // x + c2 * x^3
    __m256 tanh_arg = _mm256_mul_ps(vc1, inner);

    // Now we need tanh(tanh_arg). Use polynomial approx:
    // tanh(x) ≈ x * (27 + x^2) / (27 + 9*x^2)  -- Padé [3/2] approximant
    __m256 t2 = _mm256_mul_ps(tanh_arg, tanh_arg);
    __m256 num = _mm256_fmadd_ps(t2, tanh_arg, _mm256_mul_ps(tanh_arg, _mm256_set1_ps(27.0f)));
    __m256 den = _mm256_fmadd_ps(t2, _mm256_set1_ps(9.0f), _mm256_set1_ps(27.0f));
    __m256 tanh_val = _mm256_div_ps(num, den);

    // 0.5 * x * (1 + tanh_val)
    __m256 one_plus = _mm256_add_ps(one, tanh_val);
    return _mm256_mul_ps(vhalf, _mm256_mul_ps(x, one_plus));
}
```

Tanh 近似使用 Padé 有理逼近，比指数函数快得多（避免调用 `_mm256_exp_ps`）。精度损失 < 1e-4，对训练和推理几乎无影响。

### 1.5 Swish / SiLU

```c
// SiLU: f(x) = x * sigmoid(x)
// sigmoid(x) = 1 / (1 + exp(-x))

// Using polynomial approximation for sigmoid to avoid exp():
// sigmoid(x) ≈ 0.5 + 0.25 * x / (1 + |0.25*x|)  -- fast approx, ~1% error
static inline __m256 silu_fast_avx2(__m256 x) {
    __m256 quarter = _mm256_set1_ps(0.25f);
    __m256 half = _mm256_set1_ps(0.5f);

    __m256 ax = _mm256_mul_ps(x, quarter);
    __m256 abs_ax = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), ax);
    __m256 denom = _mm256_add_ps(_mm256_set1_ps(1.0f), abs_ax);
    __m256 sig = _mm256_add_ps(half, _mm256_div_ps(ax, denom));
    return _mm256_mul_ps(x, sig);
}
```

## 2. Reduce（归约）

### 2.1 AVX2 水平求和（经典 6 指令序列）

```c
float reduce_sum_avx2(__m256 v) {
    // hadd twice within each 128-bit lane
    __m256 h = _mm256_hadd_ps(v, v);          // [a0+a1, a2+a3, a0+a1, a2+a3 | a4+a5, a6+a7, a4+a5, a6+a7]
    h = _mm256_hadd_ps(h, h);                  // [sum0123, sum0123, sum0123, sum0123 | sum4567, sum4567, sum4567, sum4567]

    // Extract high 128 and add to low 128
    __m128 lo = _mm256_extractf128_ps(h, 0);
    __m128 hi = _mm256_extractf128_ps(h, 1);
    __m128 s = _mm_add_ps(lo, hi);             // [sum0123+sum4567, ...]

    // Reduce 4 floats to 1
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}
```

### 2.2 AVX-512 归约

```c
// Built-in reduce (compiler generates optimal sequence)
float reduce_sum_avx512(__m512 v) {
    return _mm512_reduce_add_ps(v);
}

// Manual AVX-512 reduction for understanding
float reduce_sum_avx512_manual(__m512 v) {
    __m256 lo = _mm512_castps512_ps256(v);
    __m256 hi = _mm512_extractf32x8_ps(v, 1);
    __m256 s = _mm256_add_ps(lo, hi);
    return reduce_sum_avx2(s);  // reuse AVX2 reduction
}

// AVX-512 horizontal max
float reduce_max_avx512(__m512 v) {
    return _mm512_reduce_max_ps(v);
}
```

### 2.3 Argmax（找最大值及其索引）

```c
// AVX2: track index alongside value
int argmax_avx2(const float* arr, int n) {
    __m256 best_val = _mm256_set1_ps(-INFINITY);
    __m256 best_idx = _mm256_setzero_ps();
    __m256 base_idx = _mm256_setr_ps(0,1,2,3,4,5,6,7);
    __m256 stride = _mm256_set1_ps(8.0f);

    for (int i = 0; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(arr + i);
        __m256 idx = _mm256_add_ps(base_idx, _mm256_set1_ps((float)i));

        // compare: which lanes have bigger values?
        __m256 mask = _mm256_cmp_ps(v, best_val, _CMP_GT_OS);
        best_val = _mm256_blendv_ps(best_val, v, mask);
        best_idx = _mm256_blendv_ps(best_idx, idx, mask);
    }

    // Horizontal reduction: find max among the 8 lanes
    // Extract all 8 values and their indices, find global max
    float vals[8], idxs[8];
    _mm256_storeu_ps(vals, best_val);
    _mm256_storeu_ps(idxs, best_idx);

    float max_val = vals[0];
    int arg = (int)idxs[0];
    for (int j = 1; j < 8; j++) {
        if (vals[j] > max_val) { max_val = vals[j]; arg = (int)idxs[j]; }
    }
    return arg;
}
```

**AVX-512 argmax** 可以利用 `_mm512_cmp_ps_mask` 和 `_mm512_mask_blend_ps` 更简洁：

```c
int argmax_avx512(const float* arr, int n) {
    __m512 best_val = _mm512_set1_ps(-INFINITY);
    __m512i best_idx = _mm512_setzero_si512();
    __m512i base = _mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);

    for (int i = 0; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(arr + i);
        __m512i idx = _mm512_add_epi32(base, _mm512_set1_epi32(i));

        __mmask16 better = _mm512_cmp_ps_mask(v, best_val, _CMP_GT_OQ);
        best_val = _mm512_mask_blend_ps(better, best_val, v);
        best_idx = _mm512_mask_blend_epi32(better, best_idx, idx);
    }

    // Horizontal reduction shortcut
    float vals[16]; int idxs[16];
    _mm512_storeu_ps(vals, best_val);
    _mm512_storeu_si512((__m512i*)idxs, best_idx);
    int arg = idxs[0];
    for (int j = 1; j < 16; j++)
        if (vals[j] > vals[arg]) arg = idxs[j];
    return arg;
}
```

## 3. Dot Product（点积/内积）

### 3.1 AVX2 点积

```c
// dot = Σ a[i] * b[i] for i in [0, n)
float dot_product_avx2(const float* a, const float* b, int n) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();  // 第二个累加器，隐藏 FMA 延迟

    int i;
    for (i = 0; i + 16 <= n; i += 16) {
        __m256 a0 = _mm256_loadu_ps(a + i);
        __m256 b0 = _mm256_loadu_ps(b + i);
        sum0 = _mm256_fmadd_ps(a0, b0, sum0);

        __m256 a1 = _mm256_loadu_ps(a + i + 8);
        __m256 b1 = _mm256_loadu_ps(b + i + 8);
        sum1 = _mm256_fmadd_ps(a1, b1, sum1);
    }

    // Combine two accumulators
    sum0 = _mm256_add_ps(sum0, sum1);

    // Tail: scalar
    float result = reduce_sum_avx2(sum0);
    for (; i < n; i++) result += a[i] * b[i];
    return result;
}
```

**关键优化**：两个累加器 `sum0` 和 `sum1` 交错使用，打破了 FMA 的依赖链。FMA 延迟为 4 个周期，但吞吐为 0.5 个周期（每周期 2 条）。两个累加器可以交错隐藏延迟。

### 3.2 AVX-512 点积

```c
float dot_product_avx512(const float* a, const float* b, int n) {
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();

    int i;
    for (i = 0; i + 32 <= n; i += 32) {
        __m512 a0 = _mm512_loadu_ps(a + i);
        __m512 b0 = _mm512_loadu_ps(b + i);
        sum0 = _mm512_fmadd_ps(a0, b0, sum0);

        __m512 a1 = _mm512_loadu_ps(a + i + 16);
        __m512 b1 = _mm512_loadu_ps(b + i + 16);
        sum1 = _mm512_fmadd_ps(a1, b1, sum1);
    }

    sum0 = _mm512_add_ps(sum0, sum1);
    float result = _mm512_reduce_add_ps(sum0);
    for (; i < n; i++) result += a[i] * b[i];
    return result;
}
```

## 4. 量化 int8 推理

### 4.1 为什么 int8 在 CPU 上重要

在云端推理中，int8 量化可以将模型体积和计算量减少 4x（fp32→int8）。AVX2 通过 `_mm256_maddubs_epi16` 和 `_mm256_madd_epi16` 提供了 int8→int32 的累加流水线。

### 4.2 AVX2 int8 点积内核

```c
// Compute dot product of two int8 arrays with zero point = 0
// Process 32 elements per iteration
int32_t dot_product_i8_avx2(const int8_t* a, const int8_t* b, int n) {
    __m256i acc_even = _mm256_setzero_si256();  // even accumulator
    __m256i acc_odd = _mm256_setzero_si256();   // odd accumulator

    for (int i = 0; i + 64 <= n; i += 64) {
        // Load 32 bytes from each array
        __m256i va0 = _mm256_loadu_si256((__m256i*)(a + i));
        __m256i vb0 = _mm256_loadu_si256((__m256i*)(b + i));
        // _mm256_maddubs_epi16: multiply unsigned * signed 8-bit, accumulate to 16-bit
        // va0 as unsigned (u8), vb0 as signed (s8)
        __m256i prod0 = _mm256_maddubs_epi16(va0, vb0);

        // _mm256_madd_epi16: multiply pairs of 16-bit, accumulate to 32-bit
        // This gives us int32 partial sums
        acc_even = _mm256_add_epi32(acc_even, _mm256_madd_epi16(prod0, _mm256_set1_epi16(1)));

        // Second half (next 32 elements)
        __m256i va1 = _mm256_loadu_si256((__m256i*)(a + i + 32));
        __m256i vb1 = _mm256_loadu_si256((__m256i*)(b + i + 32));
        __m256i prod1 = _mm256_maddubs_epi16(va1, vb1);
        acc_odd = _mm256_add_epi32(acc_odd, _mm256_madd_epi16(prod1, _mm256_set1_epi16(1)));
    }

    // Horizontal sum of 8-way i32 accumulator
    acc_even = _mm256_add_epi32(acc_even, acc_odd);
    __m128i lo = _mm256_extracti128_si256(acc_even, 0);
    __m128i hi = _mm256_extracti128_si256(acc_even, 1);
    __m128i sum128 = _mm_add_epi32(lo, hi);

    // 4-way i32 → scalar
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    return _mm_extract_epi32(sum128, 0);
}
```

### 4.3 AVX-512 VNNI：一条指令完成 u8×s8→i32 点积

这是 AVX-512 VNNI 的杀手锏：

```c
// AVX-512 VNNI int8 dot product
// _mm512_dpbusd_epi32: unsigned u8 × signed s8 → accumulated i32
// Single instruction replaces the _mm256_maddubs_epi16 + _mm256_madd_epi16 pipeline!
int32_t dot_product_i8_vnni(const int8_t* a, const int8_t* b, int n) {
    __m512i acc0 = _mm512_setzero_si512();
    __m512i acc1 = _mm512_setzero_si512();

    for (int i = 0; i + 128 <= n; i += 128) {
        __m512i va0 = _mm512_loadu_si512((__m512i*)(a + i));
        __m512i vb0 = _mm512_loadu_si512((__m512i*)(b + i));
        acc0 = _mm512_dpbusd_epi32(acc0, va0, vb0);

        __m512i va1 = _mm512_loadu_si512((__m512i*)(a + i + 64));
        __m512i vb1 = _mm512_loadu_si512((__m512i*)(b + i + 64));
        acc1 = _mm512_dpbusd_epi32(acc1, va1, vb1);
    }

    acc0 = _mm512_add_epi32(acc0, acc1);
    return _mm512_reduce_add_epi32(acc0);
}
```

**性能对比（per element throughput）**：

| 指令集 | 每周期元素数 | 相对 fp32 |
|--------|-----------|-----------|
| fp32 scalar | 1 | 1x |
| AVX2 fp32 | 16 (2 FMA/cycle × 8) | 4x |
| AVX2 int8 | 64 (2 maddubs/cycle × 32) | 8x |
| AVX-512 VNNI int8 | 256 (2 dpbusd/cycle × 128) | ~16x |

## 5. LayerNorm 与 Softmax

### 5.1 LayerNorm: 两遍算法

LayerNorm 的公式：`y = (x - μ) / √(σ² + ε) * γ + β`

其中 μ 是均值，σ² 是方差。需要两遍扫描：
1. 第一遍：计算 μ 和 σ²
2. 第二遍：归一化

```c
// AVX2 LayerNorm with Welford's online variance
void layernorm_avx2(const float* x, float* y, int n, float eps,
                     const float* gamma, const float* beta) {
    // Pass 1: compute mean and variance using Welford's online algorithm
    __m256 mean = _mm256_setzero_ps();
    __m256 m2 = _mm256_setzero_ps();
    int count = 0;

    for (int i = 0; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        count += 8;

        // Welford update: delta = x - mean; mean += delta / count; m2 += delta * (x - mean)
        __m256 delta = _mm256_sub_ps(v, mean);
        __m256 delta_count = _mm256_set1_ps(1.0f / (float)count);
        __m256 delta_n = _mm256_mul_ps(delta, delta_count);
        mean = _mm256_add_ps(mean, delta_n);
        __m256 delta2 = _mm256_sub_ps(v, mean);
        m2 = _mm256_add_ps(m2, _mm256_mul_ps(delta, delta2));
    }

    // Reduce mean and m2 to scalars
    float mean_scalar = reduce_sum_avx2(mean) / (float)n;
    float var_scalar = reduce_sum_avx2(m2) / (float)n;

    // Pass 2: normalize
    __m256 vmean = _mm256_set1_ps(mean_scalar);
    __m256 inv_std = _mm256_set1_ps(1.0f / sqrtf(var_scalar + eps));

    for (int i = 0; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        v = _mm256_sub_ps(v, vmean);
        v = _mm256_mul_ps(v, inv_std);

        if (gamma && beta) {
            __m256 vg = _mm256_loadu_ps(gamma + i);
            __m256 vb = _mm256_loadu_ps(beta + i);
            v = _mm256_fmadd_ps(v, vg, vb);  // v * gamma + beta
        }
        _mm256_storeu_ps(y + i, v);
    }
}
```

**Welford 算法说明**：相比简单先把所有值加起来再求方差（需要 2 遍），Welford 可以在单遍中在线更新均值和 M2（离差平方和），且数值稳定性更好。

### 5.2 Softmax: 三遍算法

Softmax: `y[i] = exp(x[i] - max(x)) / Σ exp(x[j] - max(x))`

```c
// AVX2 Softmax (three-pass)
void softmax_avx2(const float* x, float* y, int n) {
    // Pass 1: find max value (for numerical stability)
    __m256 max_vec = _mm256_set1_ps(-INFINITY);
    int i;
    for (i = 0; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        max_vec = _mm256_max_ps(max_vec, v);
    }
    float max_val = reduce_max_avx2(max_vec);
    for (; i < n; i++) max_val = fmaxf(max_val, x[i]);

    // Pass 2: compute exp sum
    __m256 sum_vec = _mm256_setzero_ps();
    __m256 max_vec_broadcast = _mm256_set1_ps(max_val);
    for (i = 0; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        v = _mm256_sub_ps(v, max_vec_broadcast);  // shift for stability
        v = exp_approx_avx2(v);  // use polynomial approx for exp
        sum_vec = _mm256_add_ps(sum_vec, v);
        _mm256_storeu_ps(y + i, v);  // store exp values for pass 3
    }
    float sum_val = reduce_sum_avx2(sum_vec);
    for (; i < n; i++) { y[i] = expf(x[i] - max_val); sum_val += y[i]; }

    // Pass 3: normalize
    __m256 inv_sum = _mm256_set1_ps(1.0f / sum_val);
    for (i = 0; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(y + i);
        _mm256_storeu_ps(y + i, _mm256_mul_ps(v, inv_sum));
    }
    for (; i < n; i++) y[i] /= sum_val;
}
```

**两遍变体**（在线跟踪最大值）：

```c
// Two-pass Softmax: track max online, compute exp on the fly
void softmax_avx2_2pass(const float* x, float* y, int n) {
    // Pass 1: find global max
    float max_val = -INFINITY;
    for (int i = 0; i < n; i++) max_val = fmaxf(max_val, x[i]);

    // Pass 2: compute exp and sum simultaneously
    __m256 sum_vec = _mm256_setzero_ps();
    __m256 max_vec = _mm256_set1_ps(max_val);
    int i;
    for (i = 0; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        v = _mm256_sub_ps(v, max_vec);
        v = exp_approx_avx2(v);
        _mm256_storeu_ps(y + i, v);
        sum_vec = _mm256_add_ps(sum_vec, v);
    }
    float sum_val = reduce_sum_avx2(sum_vec);
    for (; i < n; i++) { y[i] = expf(x[i] - max_val); sum_val += y[i]; }

    // Normalize in-place
    __m256 inv = _mm256_set1_ps(1.0f / sum_val);
    for (i = 0; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_loadu_ps(y + i), inv));
    }
    for (; i < n; i++) y[i] /= sum_val;
}
```

### 5.3 Flash Attention 概念（Tiling Softmax）

在长序列上，完整的 softmax 需要 O(N) 内存。Flash Attention 通过 tiling（分块）将复杂度降低为 O(1)，同时保持数学等价。核心思想：

```
对每个块 B_j：
  m_prev = m_new
  计算块内: max_local = max(B_j)
  m_new = max(m_prev, max_local)
  
  重新缩放: sum_exp *= exp(m_prev - m_new)
  计算块内: local_sum = Σ exp(x_k - m_new)
  sum_exp += local_sum
  
  重新缩放: output *= exp(m_prev - m_new)
  计算块内: output_k = exp(x_k - m_new) * v_k
  output += output_k
```

AVX-512 实现这个 tiled softmax 非常适合，因为 `_mm512_reduce_max_ps` 和 `_mm512_reduce_add_ps` 快速完成块内归约。

## 6. 卷积（Convolution）

### 6.1 1D 卷积

```c
// 1D convolution: y[i] = Σ_{k=0}^{K-1} x[i+k] * w[k]  (stride=1, no padding)
void conv1d_avx2(const float* x, float* y, int n,
                  const float* w, int kernel_size) {
    // Broadcast each kernel weight and FMADD with the corresponding input window
    // y[i..i+7] = Σ_k w[k] * x[i+k..i+k+7]
    for (int i = 0; i + 8 <= n - kernel_size + 1; i += 8) {
        __m256 acc = _mm256_setzero_ps();
        for (int k = 0; k < kernel_size; k++) {
            __m256 w_broadcast = _mm256_set1_ps(w[k]);
            __m256 x_window = _mm256_loadu_ps(x + i + k);
            acc = _mm256_fmadd_ps(w_broadcast, x_window, acc);
        }
        _mm256_storeu_ps(y + i, acc);
    }
}
```

### 6.2 Im2col + GEMM 的 2D 卷积

对于 2D 卷积，标准 SIMD 方法是将输入展开成列矩阵（im2col），然后做矩阵乘法：

```c
// Simplified im2col for 3x3 conv, stride 1
// Input: HxW image, Output: (H-2)*(W-2) result
void conv2d_3x3_im2col_avx2(const float* input, float* output,
                              const float* kernel,
                              int H, int W) {
    int OH = H - 2;
    int OW = W - 2;

    for (int oh = 0; oh < OH; oh++) {
        for (int ow = 0; ow + 8 <= OW; ow += 8) {
            __m256 acc = _mm256_setzero_ps();

            // Unroll the 3x3 kernel manually
            for (int kh = 0; kh < 3; kh++) {
                const float* row = input + (oh + kh) * W + ow;
                __m256 k0 = _mm256_set1_ps(kernel[kh * 3 + 0]);
                __m256 k1 = _mm256_set1_ps(kernel[kh * 3 + 1]);
                __m256 k2 = _mm256_set1_ps(kernel[kh * 3 + 2]);

                __m256 r0 = _mm256_loadu_ps(row);
                __m256 r1 = _mm256_loadu_ps(row + 1);
                __m256 r2 = _mm256_loadu_ps(row + 2);

                acc = _mm256_fmadd_ps(k0, r0, acc);
                acc = _mm256_fmadd_ps(k1, r1, acc);
                acc = _mm256_fmadd_ps(k2, r2, acc);
            }
            _mm256_storeu_ps(output + oh * OW + ow, acc);
        }
    }
}
```

### 6.3 Winograd F(2,3)

对于 3×3 卷积，Winograd 最小滤波算法可以将乘法次数从 9 次减少到 4 次（代价是额外的变换）：

```
Winograd F(2,3): 对每 2x2 输出块，只需要 4x4 = 16 次乘法
原始卷积:       对每 2x2 输出块，需要 9×4 = 36 次乘法
理论加速: 2.25x
```

Winograd 变换矩阵是固定的（可以在编译时预计算），变换运算使用 SIMD 指令批量处理。

## 7. fp16 和 bf16 转换

### 7.1 fp16 转换（AVX2 F16C）

```c
// fp32 → fp16 (pack 8 fp32 into 128-bit of 8 fp16)
__m128i fp32_to_fp16_avx2(__m256 a) {
    return _mm256_cvtps_ph(a, _MM_FROUND_TO_NEAREST_INT);
}

// fp16 → fp32 (unpack to 256-bit)
__m256 fp16_to_fp32_avx2(__m128i a) {
    return _mm256_cvtph_ps(a);
}

// Batch conversion
void convert_fp32_to_fp16(const float* src, uint16_t* dst, int n) {
    for (int i = 0; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        _mm_storeu_si128((__m128i*)(dst + i), _mm256_cvtps_ph(v, 0));
    }
}
```

### 7.2 bf16 模拟转换

bf16 是"截断"的 fp32——保留高 16 位（指数部分不变，尾数从 23 位截断到 7 位）：

```c
// bf16 conversion: truncate lower 16 bits of fp32
// AVX2: use 16-bit right shift to truncate
__m256i fp32_to_bf16_avx2(__m256 a) {
    // Shift all 32-bit lanes right by 16 bits to get bf16
    __m256i ai = _mm256_castps_si256(a);
    return _mm256_srli_epi32(ai, 16);  // keep upper 16 bits
}

// bf16 to fp32: shift left 16 (restore to fp32)
__m256 bf16_to_fp32_avx2(__m256i a) {
    __m256i ai = _mm256_slli_epi32(a, 16);
    return _mm256_castsi256_ps(ai);
}

// Memory-efficient bf16 storage: pack two bf16 into one 32-bit slot
void pack_bf16(const float* src, uint16_t* dst, int n) {
    for (int i = 0; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        __m256i truncated = _mm256_srli_epi32(_mm256_castps_si256(v), 16);
        // Pack 8 x 32-bit lanes into 128-bit of 8 x 16-bit
        __m128i packed = _mm_packus_epi32(
            _mm256_extracti128_si256(truncated, 0),
            _mm256_extracti128_si256(truncated, 1));
        _mm_storeu_si128((__m128i*)(dst + i), packed);
    }
}
```

**bf16 dot product with AVX-512 BF16** (Cooper Lake+, Sapphire Rapids):

```c
float bf16_dot_avx512(const uint16_t* a, const uint16_t* b, int n) {
    __m512 acc = _mm512_setzero_ps();
    for (int i = 0; i + 32 <= n; i += 32) {
        __m512i va = _mm512_loadu_si512((__m512i*)(a + i));
        __m512i vb = _mm512_loadu_si512((__m512i*)(b + i));
        // _mm512_dpbf16_ps: bf16 * bf16 → fp32 accumulation
        // Single instruction processes 32 bf16 elements
        acc = _mm512_dpbf16_ps(acc, (__m512bh)va, (__m512bh)vb);
    }
    return _mm512_reduce_add_ps(acc);
}
```

## 8. GEMM/GEMV 微内核

### 8.1 GEMV（矩阵-向量乘法）

```c
// y = A * x + y  (A: MxN matrix)
// AVX2 GEMV with 4-row unrolling for L1 cache tiling
void gemv_avx2(const float* A, const float* x, float* y, int M, int N) {
    // For each 4 rows
    for (int i = 0; i + 4 <= M; i += 4) {
        __m256 y0 = _mm256_setzero_ps();  // accumulates 8 columns for row i
        __m256 y1 = _mm256_setzero_ps();
        __m256 y2 = _mm256_setzero_ps();
        __m256 y3 = _mm256_setzero_ps();

        for (int j = 0; j + 8 <= N; j += 8) {
            __m256 x_vec = _mm256_loadu_ps(x + j);

            // Broadcast each element of x and FMADD with corresponding A row
            __m256 x_bc = _mm256_broadcast_ss(x + j);
            // ... complex FMADD sequence ...
        }
        // ... horizontal reduce and store ...
    }
}
```

**AVX-512 GEMV** 因为 32 个寄存器 + 16 宽向量 + 掩码，可以实现非常激进的展开：

```c
void gemv_avx512(const float* A, const float* x, float* y, int M, int N) {
    for (int i = 0; i + 8 <= M; i += 8) {
        __m512 y_acc[8] = {};  // 8 accumulators, one per row
        // ...
    }
}
```

### 8.2 GEMM 微内核（8x8 AVX2）

```c
// C_{m×n} += A_{m×k} * B_{k×n}
// Micro-kernel: 8 rows of A × 8 columns of B
void gemm_microkernel_8x8_avx2(
    const float* A, const float* B, float* C,
    int k, int lda, int ldb, int ldc)
{
    // Load 8 rows of A into registers (one YMM per row)
    __m256 c0 = _mm256_loadu_ps(C + 0 * ldc);  // row 0
    __m256 c1 = _mm256_loadu_ps(C + 1 * ldc);
    __m256 c2 = _mm256_loadu_ps(C + 2 * ldc);
    __m256 c3 = _mm256_loadu_ps(C + 3 * ldc);
    __m256 c4 = _mm256_loadu_ps(C + 4 * ldc);
    __m256 c5 = _mm256_loadu_ps(C + 5 * ldc);
    __m256 c6 = _mm256_loadu_ps(C + 6 * ldc);
    __m256 c7 = _mm256_loadu_ps(C + 7 * ldc);

    // Loop over k dimension
    for (int p = 0; p < k; p++) {
        // Load 8 elements from B row (one column of k dimension)
        __m256 b_vec = _mm256_loadu_ps(B + p * ldb);

        // Load one element of A at a time and broadcast
        __m256 a0 = _mm256_broadcast_ss(A + 0 * lda + p);
        c0 = _mm256_fmadd_ps(a0, b_vec, c0);

        __m256 a1 = _mm256_broadcast_ss(A + 1 * lda + p);
        c1 = _mm256_fmadd_ps(a1, b_vec, c1);

        __m256 a2 = _mm256_broadcast_ss(A + 2 * lda + p);
        c2 = _mm256_fmadd_ps(a2, b_vec, c2);

        __m256 a3 = _mm256_broadcast_ss(A + 3 * lda + p);
        c3 = _mm256_fmadd_ps(a3, b_vec, c3);

        __m256 a4 = _mm256_broadcast_ss(A + 4 * lda + p);
        c4 = _mm256_fmadd_ps(a4, b_vec, c4);

        __m256 a5 = _mm256_broadcast_ss(A + 5 * lda + p);
        c5 = _mm256_fmadd_ps(a5, b_vec, c5);

        __m256 a6 = _mm256_broadcast_ss(A + 6 * lda + p);
        c6 = _mm256_fmadd_ps(a6, b_vec, c6);

        __m256 a7 = _mm256_broadcast_ss(A + 7 * lda + p);
        c7 = _mm256_fmadd_ps(a7, b_vec, c7);
    }

    // Store results
    _mm256_storeu_ps(C + 0 * ldc, c0);
    _mm256_storeu_ps(C + 1 * ldc, c1);
    _mm256_storeu_ps(C + 2 * ldc, c2);
    _mm256_storeu_ps(C + 3 * ldc, c3);
    _mm256_storeu_ps(C + 4 * ldc, c4);
    _mm256_storeu_ps(C + 5 * ldc, c5);
    _mm256_storeu_ps(C + 6 * ldc, c6);
    _mm256_storeu_ps(C + 7 * ldc, c7);
}
```

**寄存器使用分析**：
- 8 个 YMM 累加器（c0-c7）+ 1 个 B 向量 + 1 个 A 广播 = 10 个活跃寄存器
- 16 个 YMM 总寄存器中还有 6 个空闲（可用于更激进的展开，如 12x8 微内核）
- 如果同时 broadcast A 的 8 个元素（需要 8 个 load 端口冲突），可以考虑提前将 A 元素加载到 GPR 再 broadcast

**AVX-512 16x8 GEMM 微内核** 优势更明显：32 个 ZMM 允许 16 个累加器 + 8 个临时 = 24，还有 8 个空闲。

### 8.3 Cache 分块参数

```c
// GEMM blocking parameters (empirically tuned for Skylake-SP)
// L1: 32KB → ~4KB for A block + ~4KB for B block + 4KB for C block
#define MC 256   // rows of C in L2 cache
#define KC 256   // k-dimension block
#define NC 4096  // columns of B in L3 cache (large L3)

// Micro-kernel dimensions
#define MR 8    // rows in micro-kernel
#define NR 8    // columns in micro-kernel
```

## 9. 总结：模式选择指南

```
需要逐元素操作？ → Map 模式 (ReLU, clamp, GELU, SiLU)
需要将向量归约为标量？ → Reduce (sum, max, argmax)
需要两个向量的内积？ → Dot Product + FMA
需要 int8 推理？ → AVX2 maddubs/madd 或 AVX-512 VNNI
需要归一化？ → LayerNorm/Softmax 多遍算法
需要矩阵乘法？ → GEMM 微内核 + cache 分块
需要降低带宽？ → fp16 转换 或 bf16 truncation
```
