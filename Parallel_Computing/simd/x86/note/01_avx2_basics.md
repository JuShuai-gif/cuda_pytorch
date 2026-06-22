# AVX2 Intrinsics 基础指南

## 1. 数据类型全览

AVX2 在 C/C++ 中通过 `immintrin.h` 头文件提供 intrinsic 函数。核心数据类型：

| 类型 | 内容 | 元素数 | 字节数 |
|------|------|--------|--------|
| `__m256` | f32（单精度浮点） | 8 | 32 |
| `__m256d` | f64（双精度浮点） | 4 | 32 |
| `__m256i` | 整数（各种宽度） | 取决于解释 | 32 |

`__m256i` 是一个"万能"整数类型。同一寄存器可以解释为：
- 32× `int8_t`（`char`）
- 16× `int16_t`（`short`）
- 8× `int32_t`（`int`）
- 4× `int64_t`（`long long`）

具体行为由操作指令决定，编译器不帮你做类型检查——你调 `_mm256_add_epi16` 参数传的是 `__m256i`，编译器假设你知道它们在逻辑上代表 16 个 `int16_t`。

## 2. 加载与存储（Load/Store）

### 2.1 对齐加载/存储

```c
// 要求 ptr 必须 32 字节对齐，否则出错（不 crash，但慢或结果错误于旧 CPU）
__m256 _mm256_load_ps(float const *ptr);       // 4x f32 → __m256
__m256d _mm256_load_pd(double const *ptr);      // 4x f64 → __m256d
__m256i _mm256_load_si256(__m256i const *ptr);  // 256 位整数加载

// 反方向存储
void _mm256_store_ps(float *ptr, __m256 a);
void _mm256_store_pd(double *ptr, __m256d a);
void _mm256_store_si256(__m256i *ptr, __m256i a);
```

```c
// 分配 32 字节对齐内存
#include <malloc.h>  // _mm_malloc
float *data = (float*)_mm_malloc(n * sizeof(float), 32);
// ... 使用 ...
_mm_free(data);  // 必须用 _mm_free 释放，不能用 free
```

### 2.2 非对齐加载/存储

```c
// 不要求对齐，现代 CPU 性能接近对齐版本
__m256 _mm256_loadu_ps(float const *ptr);
__m256d _mm256_loadu_pd(double const *ptr);
__m256i _mm256_loadu_si256(__m256i const *ptr);

void _mm256_storeu_ps(float *ptr, __m256 a);
void _mm256_storeu_pd(double *ptr, __m256d a);
void _mm256_storeu_si256(__m256i *ptr, __m256i a);
```

**现代 CPU 上的非对齐代价**：
- 不跨 cache line（64 字节边界）：零额外代价（Haswell+ 的非对齐加载/存储和前对齐一样在同一个端口执行）
- 跨 cache line（64 字节边界）：额外 1-2 个周期
- 跨 4KB 页面边界：触发页面遍历，代价约 100-150 个周期

**实践准则**：为了提高代码可移植性和降低维护成本，16或32字节对齐数组，然后使用非对齐加载/存储 `loadu`/`storeu`。现代 CPU 上的性能差异在多数情况下微乎其微。

### 2.3 非时间存储（Non-temporal Store）

```c
// 绕过 cache，直接写入内存
// 适用于：数据不会被立即再次读取，且数据量 > 50% L1d 大小
void _mm256_stream_ps(float *ptr, __m256 a);
void _mm256_stream_pd(double *ptr, __m256d a);
void _mm256_stream_si256(__m256i *ptr, __m256i a);

// 非时间存储是弱序的（weakly-ordered），如果需要确保可见性：
_mm_sfence();  // Store Fence，确保所有之前的 store 对后续可见
```

**使用场景**：
```c
// 大型 memcpy 风格的操作：直接将数据从源搬到目的地，不污染 cache
void memcpy_stream_256(const float* src, float* dst, int n) {
    for (int i = 0; i < n; i += 8) {
        __m256 data = _mm256_loadu_ps(src + i);
        _mm256_stream_ps(dst + i, data);  // 写直达，不占用 cache
    }
    _mm_sfence();  // 保证所有流存储完成
}
```

### 2.4 屏蔽加载/存储（Masked Load/Store，AVX2 有限支持）

AVX2 仅有以下掩码加载/存储（只能屏蔽最高位，即 f32 的符号位）：

```c
// 根据掩码的最高位（MSB）选择性地加载
// 如果对应位置的掩码 MSB=0，该 lane 保持原值（maskload）或设为 0（maskz_load）
__m128 _mm_maskload_ps(float const *ptr, __m128i mask);
void _mm_maskstore_ps(float *ptr, __m128i mask, __m128 a);

// 256 位版本
__m256 _mm256_maskload_ps(float const *ptr, __m256i mask);
void _mm256_maskstore_ps(float *ptr, __m256i mask, __m256 a);
```

**掩码生成**（只看最高位）：
```c
__m256 values = _mm256_loadu_ps(src);
__m256 zero = _mm256_setzero_ps();
// _CMP_GT_OS 比较 a > b，结果全 1 表示 true，全 0 表示 false
// 掩码的高位（MSB）自然为 1 或 0
__m256 cmp_result = _mm256_cmp_ps(values, zero, _CMP_GT_OS);

// 加载时，掩码 MSB=1 的位置才真正从内存加载；MSB=0 保留目标的原值
__m256 loaded = _mm256_maskload_ps(src, (__m256i)cmp_result);
```

**这是 AVX2 掩码的一个重大局限**：你无法直接指定任意 k-bit 掩码，必须回到比较→全 0/1→MSB 的迂回路径。AVX-512 直接解决了这个问题（k 寄存器）。

## 3. 设置与初始化（Set/Initialize）

### 3.1 置零与广播

```c
// 全零
__m256 zero = _mm256_setzero_ps();
__m256d zero_d = _mm256_setzero_pd();
__m256i zero_i = _mm256_setzero_si256();

// 广播：将一个值复制到所有 lane
__m256 broad = _mm256_set1_ps(3.14f);     // 所有 8 个 lane 都是 3.14
__m256d broad_d = _mm256_set1_pd(2.718);  // 所有 4 个 lane 都是 2.718

// 从内存广播（一条指令，不加载再广播）
__m256 broad_mem = _mm256_broadcast_ss(&scalar);    // 1 个指令！
__m256d broad_mem_d = _mm256_broadcast_sd(&scalar);
```

**性能提示**：`_mm256_broadcast_ss` 比 `_mm256_set1_ps(*ptr)` 更好，因为前者从内存直接广播，后者需要先加载标量到 GPR/xmm 再广播。

### 3.2 指定所有 lane（注意逆序！）

```c
// _mm256_set_ps 的参数顺序是反直觉的：最后一个参数是最低地址（lane 0）
// 签名：_mm256_set_ps(e7, e6, e5, e4, e3, e2, e1, e0)
//                                          ^^^^^^^^^^^^^^^^
//                                          第一个 = lane 0
__m256 v = _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f);
// v[0] = 0.0, v[1] = 1.0, ..., v[7] = 7.0

// 同样，_mm256_set_epi32 也是最后一个参数是最低 32 位
__m256i vi = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
// vi.u32[0] = 0, vi.u32[7] = 7
```

**记住规则**：Intel intrinsic 命名中 `set` 系列（以及所有带 `_r` 后缀的 reverse 版本）逆序写入。`setr` 系列**正序**写入：

```c
// setr = reverse set，参数顺序符合直觉
__m256 v = _mm256_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f);
// v[0] = 0.0, v[1] = 1.0, ...

__m256i vi = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
// vi.u32[0] = 0, vi.u32[7] = 7
```

## 4. 算术运算

### 4.1 基本浮点运算

```c
__m256 a = _mm256_loadu_ps(src1);
__m256 b = _mm256_loadu_ps(src2);

__m256 sum  = _mm256_add_ps(a, b);    // c[i] = a[i] + b[i]
__m256 diff = _mm256_sub_ps(a, b);    // c[i] = a[i] - b[i]
__m256 prod = _mm256_mul_ps(a, b);    // c[i] = a[i] * b[i]
__m256 quot = _mm256_div_ps(a, b);    // c[i] = a[i] / b[i]
```

**div 性能警告**：`_mm256_div_ps` 的延迟约为 11-13 个周期（Skylake），吞吐约每 5 周期一个。极其昂贵。如果除以常量，应预计算倒数后用乘法代替：

```c
// 差：反复除以常量
for (int i = 0; i < n; i += 8) {
    __m256 v = _mm256_loadu_ps(src + i);
    v = _mm256_div_ps(v, constant);  // 11-13 周期延迟
}

// 好：预计算倒数
__m256 inv = _mm256_set1_ps(1.0f / scalar_constant);
for (int i = 0; i < n; i += 8) {
    __m256 v = _mm256_loadu_ps(src + i);
    v = _mm256_mul_ps(v, inv);       // 4 周期延迟
}
```

### 4.2 Fused Multiply-Add（FMA）

FMA 是 AVX2 生态中最强大的单指令之一：

```c
// a * b + c，一次舍入，一个 µop
__m256 r = _mm256_fmadd_ps(a, b, c);   // r = a*b + c
__m256 r = _mm256_fmsub_ps(a, b, c);   // r = a*b - c
__m256 r = _mm256_fnmadd_ps(a, b, c);  // r = -(a*b) + c
__m256 r = _mm256_fnmsub_ps(a, b, c);  // r = -(a*b) - c

// 双精度版本
__m256d r = _mm256_fmadd_pd(a, b, c);
```

**为什么 FMA 如此重要**：

```c
// 向量点积（dot product）：FMA 天然匹配 accumulate 模式
__m256 acc = _mm256_setzero_ps();
for (int i = 0; i < n; i += 8) {
    __m256 x = _mm256_loadu_ps(vec1 + i);
    __m256 y = _mm256_loadu_ps(vec2 + i);
    acc = _mm256_fmadd_ps(x, y, acc);  // acc += x*y，一个指令完成！
}
// 如果不用 FMA：
// acc = _mm256_add_ps(acc, _mm256_mul_ps(x, y));  // 两条指令，两次舍入
```

FMA 在现代 Intel CPU（Haswell+）上的吞吐：**每周期 2 条**（端口 0 和 1 各一条）。即每周期可以完成 2×8=16 次浮点乘加，对应 32 GFLOPS/GHz（单精度）。

### 4.3 整数运算

```c
// 32 位整数
__m256i a = _mm256_loadu_si256((__m256i*)(src1));
__m256i b = _mm256_loadu_si256((__m256i*)(src2));

__m256i sum  = _mm256_add_epi32(a, b);  // 8 个 i32 加法
__m256i diff = _mm256_sub_epi32(a, b);
__m256i maxv = _mm256_max_epi32(a, b);  // 逐元素最大值
__m256i minv = _mm256_min_epi32(a, b);

// 16 位整数
__m256i add16 = _mm256_add_epi16(a, b);    // 16 个 i16
__m256i mulhi16 = _mm256_mulhi_epi16(a, b); // 高位相乘

// 8 位整数
__m256i add8  = _mm256_add_epi8(a, b);    // 32 个 i8
__m256i sub8  = _mm256_sub_epi8(a, b);

// 64 位整数（操作更受限）
__m256i add64 = _mm256_add_epi64(a, b);
```

**AVX2 缺少的 32 位整数乘法**（重要坑！）：

```c
// 错误：_mm256_mullo_epi32 在 AVX2 中根本不存在！
// __m256i prod32 = _mm256_mullo_epi32(a, b);  // 编译错误

// 更糟糕：_mm256_mul_epi32 存在，但它是做什么的？
// _mm256_mul_epi32 的行为：取每个 128 位 lane 中的第 0 和第 2 个 i32，
// 将它们扩展为 i64，然后做 64 位乘法。只有 32→64 位结果有用。
// 对于一般的 32 位乘法，你需要自己实现。

// 替代方案 1：使用 16 位乘法并组合（如果数据允许）
// 方案 2：降级到 128 位 _mm_mullo_epi32 然后组合
// 方案 3：升级到 AVX-512（_mm512_mullo_epi32 存在）
```

## 5. 比较操作

### 5.1 浮点比较

```c
__m256 a = _mm256_loadu_ps(src1);
__m256 b = _mm256_loadu_ps(src2);

// 比较结果：逐 lane 全 1（true）或全 0（false）
// __m256 的每个 lane 是 32 位，结果也是 32 位全 1/全 0
__m256 eq = _mm256_cmp_ps(a, b, _CMP_EQ_OQ);      // a == b
__m256 ne = _mm256_cmp_ps(a, b, _CMP_NEQ_OQ);     // a != b
__m256 lt = _mm256_cmp_ps(a, b, _CMP_LT_OS);      // a < b
__m256 le = _mm256_cmp_ps(a, b, _CMP_LE_OS);      // a <= b
__m256 gt = _mm256_cmp_ps(a, b, _CMP_GT_OS);      // a > b
__m256 ge = _mm256_cmp_ps(a, b, _CMP_GE_OS);      // a >= b
__m256 uo = _mm256_cmp_ps(a, b, _CMP_UNORD_Q);    // NaN

// 后缀含义：
//   O = Ordered（正确处理 NaN）
//   Q = Quiet（不触发浮点异常）
//   S = Signaling（可能触发浮点异常，NAN 会报错）
// 实践：始终用 _OQ 后缀（安静、有序比较），除非有特殊需要
```

**将比较结果转为整数掩码**：

```c
// _mm256_movemask_ps：提取 8 个 lane 的最高位（每个 f32 的符号位）
// 返回一个 8 位整数，第 i 位 = 第 i 个 lane 的符号位
int mask = _mm256_movemask_ps(cmp_result);

// 示例：检查数组中哪些元素 > 0
__m256 v = _mm256_loadu_ps(arr);
int pos_mask = _mm256_movemask_ps(
    _mm256_cmp_ps(v, _mm256_setzero_ps(), _CMP_GT_OS));
// pos_mask 的第 i 位 = 1 表示 arr[i] > 0
while (pos_mask) {
    int idx = __builtin_ctz(pos_mask);  // 找最低位的 1 的位置
    // 处理 arr[i + idx]
    pos_mask &= pos_mask - 1;  // 清除最低位的 1
}
```

### 5.2 整数比较

```c
__m256i eq32 = _mm256_cmpeq_epi32(a, b);   // 每个 i32 lane：相等返回全 1
__m256i gt32 = _mm256_cmpgt_epi32(a, b);   // a > b（有符号）

// 也有 16 位和 8 位版本
__m256i eq16 = _mm256_cmpeq_epi16(a, b);
__m256i eq8  = _mm256_cmpeq_epi8(a, b);    // 32 路并行比较！
```

## 6. 混合与选择（Blend / Select）

### 6.1 按掩码选择

```c
// _mm256_blendv_ps: 根据掩码（每个 lane 的最高位/MSB）选择 a 或 b
// 掩码 MSB=1 → 选 b 的值
// 掩码 MSB=0 → 选 a 的值
__m256 mask = _mm256_cmp_ps(cond, zero, _CMP_GT_OS);
__m256 result = _mm256_blendv_ps(a, b, mask);
// result[i] = mask[i]最高位为1 ? b[i] : a[i]
```

```c
// _mm256_blend_ps: 用立即数（imm8）常数值掩码选择
// imm8 的 bit 0-7 分别对应 lane 0-7，bit=1 选 b，bit=0 选 a
__m256 r = _mm256_blend_ps(a, b, 0b10101010);  
// 偶数 lane 选 a，奇数 lane 选 b
```

**整数版本的混合**：

```c
// _mm256_blendv_epi8: 逐字节混合（掩码也必须是逐字节的）
// 这是 AVX2 中最灵活的混合——逐字节控制！
__m256i mask_bytes = _mm256_cmpeq_epi8(indices, pattern);
__m256i blended = _mm256_blendv_epi8(default_val, replaced_val, mask_bytes);
```

### 6.2 条件计算的经典模式

```c
// ReLU: f(x) = max(x, 0)
__m256 relu(__m256 x) {
    return _mm256_max_ps(x, _mm256_setzero_ps());
}

// LeakyReLU: f(x) = x > 0 ? x : alpha * x
__m256 leaky_relu(__m256 x, float alpha) {
    __m256 zero = _mm256_setzero_ps();
    __m256 mask = _mm256_cmp_ps(x, zero, _CMP_GT_OS);
    __m256 neg_part = _mm256_mul_ps(x, _mm256_set1_ps(alpha));
    return _mm256_blendv_ps(neg_part, x, mask);  // mask=1 选 x，mask=0 选 neg_part
}

// Clamp: f(x) = min(max(x, lo), hi)
__m256 clamp(__m256 x, float lo, float hi) {
    __m256 vlo = _mm256_set1_ps(lo);
    __m256 vhi = _mm256_set1_ps(hi);
    return _mm256_min_ps(_mm256_max_ps(x, vlo), vhi);
}
```

## 7. 置换与重排（Permute / Shuffle）

这是 AVX2 最复杂也最强大的部分。理解 lane 边界是核心。

### 7.1 关键概念：128 位 Lane

AVX2 的 256 位寄存器内部分为两个 128 位 lane：

```
256-bit YMM register:
│←──── 128-bit low lane (lane 0) ────│←──── 128-bit high lane (lane 1) ────│
│ f0 │ f1 │ f2 │ f3 │ f4 │ f5 │ f6 │ f7 │
```

**核心规则**：大多数 shuffle/permute 指令**在 128 位 lane 内部独立操作**，不跨 lane 交换数据。这是 AVX2 和 AVX-512 最大的架构差异之一。

### 7.2 Lane 内 Shuffle

```c
// _mm256_shuffle_ps: 在每个 128 位 lane 内重排
// imm8: 每 2 位选择一个 lane 内的源元素（0-3）
// lane 0 从 a[3:0] 和 b[3:0] 中选择；lane 1 从 a[7:4] 和 b[7:4] 中选择
__m256 shuf = _mm256_shuffle_ps(a, b, _MM_SHUFFLE(3, 2, 1, 0));

// _mm256_permute_ps: 在每个 128 位 lane 内重排（仅从 a 中选择）
__m256 perm = _mm256_permute_ps(a, _MM_SHUFFLE(3, 1, 2, 0));
// lane 0: a[0], a[2], a[1], a[3]   (注意 shuffle 宏中的位置也是反向的)
// lane 1: a[4], a[6], a[5], a[7]

// _mm256_unpacklo_ps: 交错排列每个 lane 的低半部分
// 结果 lane 0: a[0], b[0], a[1], b[1]
// 结果 lane 1: a[4], b[4], a[5], b[5]
__m256 lo = _mm256_unpacklo_ps(a, b);

// _mm256_unpackhi_ps: 交错排列每个 lane 的高半部分
// 结果 lane 0: a[2], b[2], a[3], b[3]
// 结果 lane 1: a[6], b[6], a[7], b[7]
__m256 hi = _mm256_unpackhi_ps(a, b);
```

### 7.3 跨 Lane 操作（AVX2 新增）

```c
// _mm256_permutevar8x32_ps: 完全按索引重排（可跨 lane！）
// idx 的每个 i32 指定 a 中哪个位置的元素被放到结果对应位置
// idx[i] 的范围是 0-7，可以跨 128 位边界！
__m256i idx = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);  // 反转
__m256 rev = _mm256_permutevar8x32_ps(a, idx);

// _mm256_insertf128_ps: 将 128 位数据插入到指定位置
// 将 b（__m128）放入结果的高位或低位
__m256 r = _mm256_insertf128_ps(a, b, 0);  // b → 低 128 位
__m256 r = _mm256_insertf128_ps(a, b, 1);  // b → 高 128 位

// _mm256_extractf128_ps: 提取 128 位数据
__m128 lo = _mm256_extractf128_ps(a, 0);  // 提取低 128 位
__m128 hi = _mm256_extractf128_ps(a, 1);  // 提取高 128 位
```

### 7.4 字节级 Shuffle（PSHUFB）

```c
// _mm256_shuffle_epi8: 逐字节重排——SIMD 中的瑞士军刀
// table[i] = 查找表（16/32 个字节）
// indices[i] = 索引（每个字节），若最高位为 1 → 输出 0
__m256i table = _mm256_loadu_si256((__m256i*)lookup);
__m256i indices = _mm256_loadu_si256((__m256i*)input);
__m256i result = _mm256_shuffle_epi8(table, indices);

// 常用模式：大小写转换
// 将 'A'..'Z' 映射到 'a'..'z'
// 查表：table[c] = tolower(c) for c in 0..255
__m256i tolower(__m256i input) {
    // table 数组的第 X 个字节 = tolower(X)
    __m256i table = _mm256_loadu_si256((__m256i*)tolower_table);
    return _mm256_shuffle_epi8(table, input);
}
```

**注意**：`_mm256_shuffle_epi8` 也有 128 位 lane 限制！每个 128 位 lane 独立使用自己的低 16 字节查表。不能用它实现真正的跨 lane 32 字节查表。如需跨 lane，需先用 `_mm256_permutevar8x32_epi32` 交换 lane 内容后再 shuffle。

## 8. 水平操作与归约

### 8.1 _mm256_hadd_ps 及阴影

```c
// _mm256_hadd_ps: 在每个 128 位 lane 内做水平加（相邻对）
// lane 0 结果: a[0]+a[1], a[2]+a[3], b[0]+b[1], b[2]+b[3]
// lane 1 结果: a[4]+a[5], a[6]+a[7], b[4]+b[5], b[6]+b[7]
__m256 h = _mm256_hadd_ps(a, b);
```

水平加指令虽然方便，但在关键路径上有两个问题：
1. 通常解码为 2 个 µop（比垂直指令慢）
2. 只能在单个端口（port 5）执行，是瓶颈

### 8.2 归约到标量（Reduction）

**AVX2 水平求和**（经典序列）：

```c
float reduce_sum_avx2(__m256 v) {
    // Step 1: 两个 128 位 lane 各自水平加
    // v[0]=v[0]+v[1], v[1]=v[2]+v[3], v[2]=v[4]+v[5], v[3]=v[6]+v[7]
    // v[4]=v[0]+v[1], v[5]=v[2]+v[3], v[6]=v[4]+v[5], v[7]=v[6]+v[7] (同一序列在 lane 1 重复)
    v = _mm256_hadd_ps(v, v);
    // v: [a0+a1, a2+a3, a0+a1, a2+a3, a4+a5, a6+a7, a4+a5, a6+a7]
    
    v = _mm256_hadd_ps(v, v);
    // v: [sum_low4, sum_low4, sum_low4, sum_low4, sum_high4, sum_high4, sum_high4, sum_high4]
    
    // Step 2: 提取高 128 位并加到低 128 位
    __m128 lo = _mm256_extractf128_ps(v, 0);  // sum_low4
    __m128 hi = _mm256_extractf128_ps(v, 1);  // sum_high4
    __m128 sum128 = _mm_add_ps(lo, hi);       // sum_low4 + sum_high4
    
    // Step 3: 32 位标量求和
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    
    float result;
    _mm_store_ss(&result, sum128);
    return result;
}
```

**更好的替代方案**（使用 permute + add，减少 µop 和 port 5 依赖）：

```c
// 使用 permute + add 替代 hadd，可以同时在 port 0/1 和 port 5 执行
// 但代码量更大，适合性能最关键的循环

float reduce_sum_avx2_fast(__m256 v) {
    // 交换高 128 位和低 128 位
    __m128 lo = _mm256_extractf128_ps(v, 0);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    
    // 在 128 位内做水平加
    __m128 shuf = _mm_movehdup_ps(sum128);  // 复制 lane 1 到 lane 0
    sum128 = _mm_add_ps(sum128, shuf);       // [a+b, c+d, a+b, c+d]
    shuf = _mm_shuffle_ps(sum128, sum128, _MM_SHUFFLE(1, 1, 1, 1));
    sum128 = _mm_add_ss(sum128, shuf);        // 标量结果
    
    return _mm_cvtss_f32(sum128);
}
```

## 9. 类型转换

### 9.1 f32 ↔ i32

```c
// f32 → i32（截断，向零舍入）
__m256i ints = _mm256_cvttps_epi32(floats);

// i32 → f32
__m256 floats = _mm256_cvtepi32_ps(ints);

// f32 → i32（舍入到最近偶数）
__m256i ints_round = _mm256_cvtps_epi32(floats);
```

### 9.2 不同整数宽度之间的转换

```c
// 有符号扩展：i8 → i16, i16 → i32, i32 → i64
// 注意：这些操作通常只在低 128 位有效
__m256i i16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m256i*)src));  // 16 i8 → 16 i16
__m256i i32 = _mm256_cvtepi16_epi32(_mm_loadu_si128((__m256i*)src)); // 8 i16 → 8 i32

// 打包（收缩）：i16 → i8, i32 → i16（饱和或截断）
__m256i packed = _mm256_packs_epi32(a, b);     // 8+8 i32 → 2×8 i16 (饱和)
__m256i packed_u = _mm256_packus_epi32(a, b);  // 8+8 i32 → 2×8 u16 (无符号饱和)
```

## 10. Gather 操作

```c
// _mm256_i32gather_ps: 从非连续地址加载 8 个 f32
// base: 基地址
// index: 8 个 i32 偏移（以元素为单位）
// scale: 偏移乘以的倍数（1, 2, 4, 8）
__m256 gathered = _mm256_i32gather_ps(base, indices, 4);  
// gathered[i] = *(float*)(base + indices[i] * 4)
```

**性能真相**：虽然 `_mm256_i32gather_ps` 是一条指令，但它被解码为多个 µop（每加载一个元素一个 µop）。延迟通常为 20-30 个周期，是对齐加载的 5-10 倍。**仅在无法转换为连续访问时使用 gather**。

```c
// 同样的加载，手动实现也比 gather 快（因为编译器知道访问模式）
for (int i = 0; i < 8; i++)
    tmp[i] = base[indices[i]];
__m256 result = _mm256_loadu_ps(tmp);
// 上述代码通常会被优化成等效于 gather 的序列，或更优
```

## 11. 常见陷阱与最佳实践

### 11.1 缺失的 32 位整数乘法

前面已经提到，但值得再次强调：

```c
// AVX2 没有 _mm256_mullo_epi32
// 解决方案：如果只需要低 32 位结果，可以这样做：
static inline __m256i _mm256_mullo_epi32_emu(__m256i a, __m256i b) {
    // 提取每个 i32 到 i64，做乘法，取低 32 位
    // 复杂的序列，需要 6-8 条指令
    // 不推荐手动实现，建议改用 AVX-512 或重新设计算法
}
```

### 11.2 Lane 边界陷阱

```c
// _mm256_shuffle_ps 不在 128 位 lane 之间交换数据
// 如果需要跨 lane shuffle，必须使用 _mm256_permutevar8x32_ps
// 或 _mm256_insertf128_ps + _mm256_extractf128_ps 组合
```

### 11.3 非对齐加载的页面跨越

```c
// 安全：使用非对齐加载
// 现代 CPU 上的代价几乎为零（只要不跨页边界）
// 谨慎：跨越 4KB 页面边界会导致约 100 个周期的额外延迟
// 如果 n 很大（> 4096），建议 32 字节对齐或至少 64 字节对齐
```

### 11.4 循环尾部处理

```c
// 循环处理主循环未覆盖的余数元素
void vec_add(const float* a, const float* b, float* c, int n) {
    int i = 0;
    // 主循环：处理 8 个元素一组
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(c + i, _mm256_add_ps(va, vb));
    }
    // 尾部处理：逐元素
    for (; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}
```

### 11.5 编译器优化陷阱

```c
// 不要这样：编译器可能将 scatter/gather 自动向量化，但效率很低
for (int i = 0; i < n; i++)
    result[indices[i]] += values[i];

// 更好的：先按索引排序，再做连续访问
// 或使用 _mm256_i32gather_ps 显式控制
```

## 12. 完整的 AVX2 向量加法示例

```c
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

// 向量加法：c = a + b（所有元素都处理）
void vec_add_avx2(const float* a, const float* b, float* c, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(c + i, vc);
    }
    // 处理余下的元素
    for (; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 1024;
    float *a = (float*)_mm_malloc(n * sizeof(float), 32);
    float *b = (float*)_mm_malloc(n * sizeof(float), 32);
    float *c = (float*)_mm_malloc(n * sizeof(float), 32);
    
    for (int i = 0; i < n; i++) {
        a[i] = (float)i;
        b[i] = (float)(n - i);
    }
    
    vec_add_avx2(a, b, c, n);
    
    for (int i = 0; i < 4; i++)
        printf("c[%d] = %.1f\n", i, c[i]);
    
    _mm_free(a); _mm_free(b); _mm_free(c);
    return 0;
}
```

编译运行：
```bash
gcc -mavx2 -mfma -O2 -o vec_add vec_add.c
./vec_add
# c[i] = 1024.0 for all i（验证正确性）
```
