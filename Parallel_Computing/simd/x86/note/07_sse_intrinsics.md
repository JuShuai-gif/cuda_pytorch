# SSE 内联函数完整指南

```
-------------------------------------------------------------------------------
目标指令集:     SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2
最低基准:       所有 x86-64 CPU（SSE2 由 x86-64 ABI 保证）
实用性:         仍用于生产环境，包括:
                   - Chromium/Skia（最低要求 SSE3）
                   - 旧硬件兼容路径
                   - 256 位宽度无优势时的 128 位操作
                   - 端口压力较低的横向操作（hadd/hsub）
性能:           2倍标量吞吐量（每条指令 4 个浮点数）
                1倍内存带宽 vs 标量（128 位加载宽度相同）
-------------------------------------------------------------------------------
```

---

## 1. 为什么 SSE 仍然重要

SSE2 是 **x86-64 的最低基准指令集**。AMD64/Intel64 ABI 强制要求
SSE2 支持。当你在 x86-64 上写 `float a = b + c;` 时，编译器可能
会使用 `addss`（SSE 标量指令）。SSE2 指令用于：
- 在寄存器中传递 float/double 参数（System V 中使用 XMM0-XMM7）
- `memcpy`/`memset`（glibc 使用 SSE/AVX 分发）
- 128 位原子操作（`cmpxchg16b`）

**SSE 是基础。** 理解 SSE 之后，AVX/AVX-512 将是自然而然的进阶，
而非未知的飞跃。

### SSE 指令集时间线

| 指令集 | 年份 | 关键新增内容 |
|-----|------|---------------|
| SSE (Katmai) | 1999 | 70 条指令, 128 位 XMM, 单精度浮点 |
| SSE2 (Willamette) | 2000 | 双精度浮点, 整数（8/16/32/64 位）, 缓存控制 |
| SSE3 (Prescott) | 2004 | 横向加减, `lddqu`, `movsldup`/`movshdup` |
| SSSE3 (Merom) | 2006 | `pshufb`（杀手级指令）, `palignr`, `pmaddubsw` |
| SSE4.1 (Penryn) | 2008 | `blendvps`, `dpps`, `pmulld`（32 位整数乘法！）, `roundps` |
| SSE4.2 (Nehalem) | 2008 | `pcmpestri`, `pcmpistri`（字符串操作）, `crc32` |

---

## 2. SSE 数据类型

```c
#include <xmmintrin.h>  // SSE  (__m128)
#include <emmintrin.h>  // SSE2 (__m128d, __m128i)
#include <pmmintrin.h>  // SSE3
#include <tmmintrin.h>  // SSSE3
#include <smmintrin.h>  // SSE4.1 + SSE4.2

// 以下头文件会传递性地包含上述所有：
#include <immintrin.h>  // 包含以上所有 + AVX/AVX2/AVX-512
```

| 类型 | 内容 | 元素数量 | 总位数 |
|------|---------|---------------|------------|
| `__m128` | f32（单精度浮点） | 4 | 128 |
| `__m128d` | f64（双精度浮点） | 2 | 128 |
| `__m128i` | 整数（解释方式可变） | 最多 16×u8 | 128 |

`__m128i` 可以表示:
- 16 × `int8_t` / `uint8_t`
- 8 × `int16_t` / `uint16_t`
- 4 × `int32_t` / `uint32_t`
- 2 × `int64_t` / `uint64_t`

---

## 3. 加载/存储操作

### 3.1 对齐加载/存储（需要 16 字节对齐）

```c
// float（4× f32）
__m128  _mm_load_ps(const float* ptr);       // 需要 16 字节对齐的 ptr
void    _mm_store_ps(float* ptr, __m128 a);

// double（2× f64）
__m128d _mm_load_pd(const double* ptr);
void    _mm_store_pd(double* ptr, __m128d a);

// 整数（128 位）
__m128i _mm_load_si128(const __m128i* ptr);
void    _mm_store_si128(__m128i* ptr, __m128i a);
```

### 3.2 非对齐加载/存储（无对齐要求）

```c
__m128  _mm_loadu_ps(const float* ptr);         // f32
__m128d _mm_loadu_pd(const double* ptr);        // f64
__m128i _mm_loadu_si128(const __m128i* ptr);    // int

void _mm_storeu_ps(float* ptr, __m128 a);
void _mm_storeu_pd(double* ptr, __m128d a);
void _mm_storeu_si128(__m128i* ptr, __m128i a);
```

**性能说明**：在 Haswell+（2013 年及以后）上，缓存行内的非对齐加载
与对齐加载性能相同。`load` 与 `loadu` 的区别主要是为了文档说明以及
非常旧的 CPU（Core 2 及更早）。

### 3.3 特殊加载

```c
// SSE3：加载非对齐整数 128（针对可能的跨缓存行加载优化）
__m128i _mm_lddqu_si128(const __m128i* ptr);

// 仅加载低元素，其余清零
__m128  _mm_load_ss(const float* ptr);       // 加载 1 个 f32，高位 3 个清零
__m128d _mm_load_sd(const double* ptr);      // 加载 1 个 f64，高位 1 个清零

// 从流式（非时间局部）缓冲区加载
__m128i _mm_stream_load_si128(const __m128i* ptr); // SSE4.1
```

### 3.4 非时间局部（流式）存储

```c
// 绕过缓存，直接写入内存。用于大型只写缓冲区。
void _mm_stream_ps(float* ptr, __m128 a);
void _mm_stream_pd(double* ptr, __m128d a);
void _mm_stream_si128(__m128i* ptr, __m128i a);
// 对于 64 字节区域的 128 位流式存储，现代 CPU 不需要 _mm_sfence。
// 对于多次 NT 存储序列，在最后使用一次 _mm_sfence() 即可。
```

---

## 4. 设置和初始化

### 4.1 置零和广播

```c
__m128  zero_f = _mm_setzero_ps();            // 全部 4 个通道 = 0.0f
__m128d zero_d = _mm_setzero_pd();            // 全部 2 个通道 = 0.0
__m128i zero_i = _mm_setzero_si128();         // 全部字节 = 0

__m128  bcast_f = _mm_set1_ps(3.14f);         // 4× 3.14f
__m128d bcast_d = _mm_set1_pd(2.718);         // 2× 2.718
__m128i bcast_i8 = _mm_set1_epi8(0x7F);       // 16× 0x7F
__m128i bcast_i16 = _mm_set1_epi16(42);       // 8× 42
__m128i bcast_i32 = _mm_set1_epi32(100);      // 4× 100
__m128i bcast_i64 = _mm_set1_epi64x(999LL);   // 2× 999
```

### 4.2 设置单个元素（注意顺序是反的！）

```c
// 警告：set 函数以反向顺序接受参数！
// _mm_set_ps(e3, e2, e1, e0) → 通道 0 = e0, 通道 3 = e3

__m128 v1 = _mm_set_ps(4.f, 3.f, 2.f, 1.f);
// v1[0]=1.0, v1[1]=2.0, v1[2]=3.0, v1[3]=4.0

// _mm_setr_ps 以正向顺序接受参数（r = 反向）：
__m128 v2 = _mm_setr_ps(1.f, 2.f, 3.f, 4.f);
// v2[0]=1.0, v2[1]=2.0, v2[2]=3.0, v2[3]=4.0

// 整数变体遵循相同模式：
__m128i vi = _mm_set_epi32(4, 3, 2, 1);       // 通道 0 = 1, 通道 3 = 4
__m128i vr = _mm_setr_epi32(1, 2, 3, 4);      // 通道 0 = 1, 通道 3 = 4
```

### 4.3 移动和转换

```c
// 将标量移入/移出 XMM（保留高位元素）
__m128  _mm_move_ss(__m128 a, __m128 b);    // a 的底部元素来自 b
float   _mm_cvtss_f32(__m128 a);            // 提取底部 f32 为标量

// 类型间转换（零开销，仅重新解释位模式）
__m128i _mm_castps_si128(__m128 a);         // f32 位 → int 位
__m128  _mm_castsi128_ps(__m128i a);        // int 位 → f32 位
__m128d _mm_castps_pd(__m128 a);            // f32 位 → f64 位
```

---

## 5. 算术操作

### 5.1 浮点算术（4× f32 打包）

```c
__m128 c = _mm_add_ps(a, b);    // c[i] = a[i] + b[i]
__m128 c = _mm_sub_ps(a, b);    // c[i] = a[i] - b[i]
__m128 c = _mm_mul_ps(a, b);    // c[i] = a[i] * b[i]
__m128 c = _mm_div_ps(a, b);    // c[i] = a[i] / b[i]  （开销很大！）
__m128 c = _mm_sqrt_ps(a);      // c[i] = sqrt(a[i])   （开销很大！）

// 倒数近似（12 位精度，1-2 周期延迟 vs 除法约 14 周期）：
__m128 c = _mm_rcp_ps(a);       // c[i] ≈ 1.0f / a[i]   （快速，低精度）
__m128 c = _mm_rsqrt_ps(a);     // c[i] ≈ 1.0f / sqrt(a[i])

// 标量变体（仅操作最低位元素）：
__m128 c = _mm_add_ss(a, b);    // c[0] = a[0] + b[0], c[1..3] = a[1..3]
__m128 c = _mm_sub_ss(a, b);
__m128 c = _mm_mul_ss(a, b);
__m128 c = _mm_div_ss(a, b);

// 最小值/最大值：
__m128 c = _mm_min_ps(a, b);    // c[i] = fminf(a[i], b[i])
__m128 c = _mm_max_ps(a, b);    // c[i] = fmaxf(a[i], b[i])
```

### 5.2 双精度算术（2× f64）

```c
__m128d c = _mm_add_pd(a, b);
__m128d c = _mm_sub_pd(a, b);
__m128d c = _mm_mul_pd(a, b);
__m128d c = _mm_div_pd(a, b);
__m128d c = _mm_sqrt_pd(a);
__m128d c = _mm_min_pd(a, b);
__m128d c = _mm_max_pd(a, b);
```

**注意**：SSE 没有 FMA 指令。FMA 随 AVX（FMA3）在 Haswell（2013 年）
引入。如果你在 SSE 上需要 `a*b+c`，必须使用两条指令：
```c
__m128 c = _mm_add_ps(_mm_mul_ps(a, b), c);  // 2 微操作, 2 次舍入
```

### 5.3 整数算术

```c
// 8 位
__m128i c = _mm_add_epi8(a, b);     // 16 × i8 加法
__m128i c = _mm_sub_epi8(a, b);
__m128i c = _mm_max_epi8(a, b);     // SSE4.1
__m128i c = _mm_min_epi8(a, b);     // SSE4.1
__m128i c = _mm_avg_epu8(a, b);     // (a[i] + b[i] + 1) >> 1  （舍入均值）

// 16 位
__m128i c = _mm_add_epi16(a, b);    // 8 × i16 加法
__m128i c = _mm_sub_epi16(a, b);
__m128i c = _mm_mullo_epi16(a, b);  // 乘法的低 16 位
__m128i c = _mm_mulhi_epi16(a, b);  // 乘法的高 16 位
__m128i c = _mm_max_epi16(a, b);
__m128i c = _mm_min_epi16(a, b);

// 32 位
__m128i c = _mm_add_epi32(a, b);    // 4 × i32 加法
__m128i c = _mm_sub_epi32(a, b);
__m128i c = _mm_mullo_epi32(a, b);  // SSE4.1：4 × i32 乘法（低 32 位）
__m128i c = _mm_max_epi32(a, b);    // SSE4.1
__m128i c = _mm_min_epi32(a, b);    // SSE4.1

// 64 位
__m128i c = _mm_add_epi64(a, b);    // 2 × i64 加法
__m128i c = _mm_sub_epi64(a, b);

// 绝对值
__m128i c = _mm_abs_epi8(a);        // SSSE3：16 × i8 绝对值
__m128i c = _mm_abs_epi16(a);       // SSSE3：8 × i16 绝对值
__m128i c = _mm_abs_epi32(a);       // SSSE3：4 × i32 绝对值

// 符号操作
__m128i c = _mm_sign_epi8(a, b);    // SSSE3：按掩码进行有符号取反
__m128i c = _mm_sign_epi16(a, b);
__m128i c = _mm_sign_epi32(a, b);
```

### 5.4 横向算术（SSE3）

```c
// 横向加法：相邻对求和
__m128  c = _mm_hadd_ps(a, b);
// c[0] = a[0] + a[1],  c[1] = a[2] + a[3]
// c[2] = b[0] + b[1],  c[3] = b[2] + b[3]

__m128d c = _mm_hadd_pd(a, b);
// c[0] = a[0] + a[1],  c[1] = b[0] + b[1]

// 横向减法
__m128  c = _mm_hsub_ps(a, b);
__m128d c = _mm_hsub_pd(a, b);

// 整数横向加减（SSSE3）
__m128i c = _mm_hadd_epi16(a, b);
__m128i c = _mm_hadd_epi32(a, b);
__m128i c = _mm_hsub_epi16(a, b);
__m128i c = _mm_hsub_epi32(a, b);

// 交替加/减元素（SSSE3）
__m128i c = _mm_addsub_epi16(a, b);  // 偶数：加, 奇数：减
```

**性能警告**：`haddps` 在 Intel Haswell+ 上解码为 2 条微操作，
延迟约 5 周期，吞吐量为每 2 周期 1 次（端口 5 瓶颈）。
对于性能关键代码中的横向归约，考虑使用 shuffle + add 替代。

---

## 6. 比较操作

### 6.1 浮点比较

```c
// SSE 比较：结果为真的通道全 1（0xFFFFFFFF），为假的通道全 0
__m128 mask_eq = _mm_cmpeq_ps(a, b);    // a == b
__m128 mask_lt = _mm_cmplt_ps(a, b);    // a < b
__m128 mask_le = _mm_cmple_ps(a, b);    // a <= b
__m128 mask_gt = _mm_cmpgt_ps(a, b);    // a > b
__m128 mask_ge = _mm_cmpge_ps(a, b);    // a >= b
__m128 mask_neq = _mm_cmpneq_ps(a, b);  // a != b
__m128 mask_ord = _mm_cmpord_ps(a, b);  // !isnan(a) && !isnan(b)
__m128 mask_unord = _mm_cmpunord_ps(a, b); // isnan(a) || isnan(b)

// 将符号位提取为 4 位整数掩码
int mask = _mm_movemask_ps(mask_eq);
// 位 i = 通道 i 的符号位（float 符号位 = 最高有效位，因此比较结果为真
// （即 0xFFFFFFFF，其 MSB=1）时，该位为 1）
```

### 6.2 整数比较

```c
// 8 位
__m128i mask = _mm_cmpeq_epi8(a, b);    // 16 路相等
__m128i mask = _mm_cmpgt_epi8(a, b);    // 有符号大于

// 16 位
__m128i mask = _mm_cmpeq_epi16(a, b);   // 8 路相等
__m128i mask = _mm_cmpgt_epi16(a, b);   // 有符号大于

// 32 位
__m128i mask = _mm_cmpeq_epi32(a, b);   // 4 路相等
__m128i mask = _mm_cmpgt_epi32(a, b);   // 有符号大于

// 64 位（SSE4.1）
__m128i mask = _mm_cmpeq_epi64(a, b);   // 2 路相等

// 提取字节级掩码
int mask16 = _mm_movemask_epi8(mask);   // 从每个字节的 MSB 生成 16 位掩码
```

---

## 7. 混合和选择（SSE4.1）

### 7.1 浮点混合

```c
// 基于掩码混合：掩码位为 1 时从 B 选取，为 0 时从 A 选取
__m128 r = _mm_blend_ps(a, b, 0b0101);    // 通道: A[0], B[1], A[2], B[3]
__m128d r = _mm_blend_pd(a, b, 0b10);     // 通道: A[0], B[1]

// 可变混合（由掩码的符号位选取）：
__m128 r = _mm_blendv_ps(a, b, mask);
// r[i] = (mask[i] < 0) ? b[i] : a[i]
```

### 7.2 整数混合

```c
// 立即数混合（SSE4.1）：
__m128i r = _mm_blend_epi16(a, b, 0b10101010); // 8× i16, imm8 选择

// 可变混合（SSE4.1）：
__m128i r = _mm_blendv_epi8(a, b, mask);  // 16× i8, 由掩码 MSB 选择
```

### 7.3 条件插入/提取（SSE4.1）

```c
// 将标量插入到指定位置
__m128 r = _mm_insert_ps(a, b, 0b00011001);
// 将 b[0] 插入到 a[imm6] 位置，可选地清零其他通道

// 从指定位置提取标量
float f = _mm_cvtss_f32(_mm_shuffle_ps(x, x, _MM_SHUFFLE(0,0,0,2)));
// 更好的方式：使用 SSE4.1 的 _mm_extract_ps（提取为 int，再转为 float）
int bits = _mm_extract_ps(x, 2);  // 提取通道 2 作为 int 位模式
float f; memcpy(&f, &bits, 4);
```

---

## 8. 洗牌和置换

### 8.1 PSHUFB——瑞士军刀（SSSE3）

`_mm_shuffle_epi8` 是功能最强大的单一 SSE 指令。它可以任意置换
16 个字节，并可将任意输出字节置零：

```c
// _mm_shuffle_epi8(lookup_table, indices):
// 对于 indices 中的每个字节 i：
//   if (indices[i] & 0x80) → result[i] = 0
//   else → result[i] = lookup_table[indices[i] & 0x0F]
//
// 注意：在每个 128 位通道内独立操作。查找表仅 16 字节！
// 对于 32 字节操作使用 _mm256_shuffle_epi8（但仍然是按 128 位通道操作的）。

// 示例：反转字节顺序
__m128i reverse_byte_order(__m128i x) {
    __m128i indices = _mm_setr_epi8(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
    return _mm_shuffle_epi8(x, indices);
}

// 示例：小写转大写
__m128i toupper_ascii(__m128i input) {
    // table[i] = (i >= 'a' && i <= 'z') ? i - 32 : i
    uint8_t table[16] __attribute__((aligned(16)));
    for (int i = 0; i < 16; i++)
        table[i] = (i >= 'a' && i <= 'z') ? (uint8_t)(i - 32) : (uint8_t)i;
    __m128i lut = _mm_load_si128((__m128i*)table);
    return _mm_shuffle_epi8(lut, input);
}
```

### 8.2 SHUFPS——32 位元素洗牌

```c
// _mm_shuffle_ps(a, b, _MM_SHUFFLE(sel_a3, sel_a2, sel_b1, sel_b0)):
// Result[0] = b[sel_b0]
// Result[1] = b[sel_b1]
// Result[2] = a[sel_a2]
// Result[3] = a[sel_a3]
//
// 宏 _MM_SHUFFLE 以反向顺序接受参数（高通道到低通道）：
_MM_SHUFFLE(3, 2, 1, 0) // 通道3=a[3], 通道2=a[2], 通道1=b[1], 通道0=b[0]

__m128 r = _mm_shuffle_ps(a, b, _MM_SHUFFLE(3, 1, 2, 0));
// r[0]=b[0], r[1]=b[2], r[2]=a[1], r[3]=a[3]

// 常用模式：
__m128 broadcast_lane0 = _mm_shuffle_ps(x, x, _MM_SHUFFLE(0, 0, 0, 0));
// r[0..3] = x[0]

__m128 swap_halves = _mm_shuffle_ps(x, x, _MM_SHUFFLE(1, 0, 3, 2));
// r = [x[2], x[3], x[0], x[1]]
```

### 8.3 解包（交织）

```c
// _mm_unpacklo_ps：交织低半部分
__m128 r = _mm_unpacklo_ps(a, b);
// r = [a[0], b[0], a[1], b[1]]

// _mm_unpackhi_ps：交织高半部分
__m128 r = _mm_unpackhi_ps(a, b);
// r = [a[2], b[2], a[3], b[3]]

// 整数变体：
__m128i r = _mm_unpacklo_epi8(a, b);   // 交织字节
__m128i r = _mm_unpackhi_epi8(a, b);
__m128i r = _mm_unpacklo_epi16(a, b);  // 交织 16 位字
__m128i r = _mm_unpackhi_epi16(a, b);
__m128i r = _mm_unpacklo_epi32(a, b);  // 交织 32 位双字
__m128i r = _mm_unpackhi_epi32(a, b);
__m128i r = _mm_unpacklo_epi64(a, b);  // 交织 64 位四字
__m128i r = _mm_unpackhi_epi64(a, b);
```

### 8.4 其他洗牌操作

```c
// SSE3：移动/复制
__m128 r = _mm_movehdup_ps(a);     // a[1],a[1],a[3],a[3]
__m128 r = _mm_moveldup_ps(a);     // a[0],a[0],a[2],a[2]

// SSSE3：对齐两个寄存器（字节级移位+合并）
__m128i r = _mm_alignr_epi8(a, b, N);
// r = 拼接(a,b) >> (N*8)，取低 128 位
// 示例 N=1：r = [b[1], b[2], ... b[15], a[0]]

// SSE2：按字节或位移动整个 128 位寄存器
__m128i r = _mm_slli_si128(a, N);  // 左移 N 字节（0 ≤ N ≤ 15）
__m128i r = _mm_srli_si128(a, N);  // 右移 N 字节
```

---

## 9. 类型转换

### 9.1 浮点 ↔ 整数

```c
// f32 → i32（向零截断）
__m128i ints = _mm_cvttps_epi32(floats);

// f32 → i32（舍入到最近偶数）
__m128i ints = _mm_cvtps_epi32(floats);

// i32 → f32
__m128 floats = _mm_cvtepi32_ps(ints);

// f64 ↔ i32
__m128i ints = _mm_cvttpd_epi32(doubles);
__m128d doubles = _mm_cvtepi32_pd(ints);

// f32 ↔ f64
__m128d d = _mm_cvtps_pd(lo_2_floats);     // 将低 2 个 f32 提升为 f64
__m128  f = _mm_cvtpd_ps(two_doubles);     // 将 2 个 f64 降级为 f32

// 标量转换（仅操作最低位元素）
int   i = _mm_cvttss_si32(xmm);            // f32 → int（截断）
int   i = _mm_cvtss_si32(xmm);             // f32 → int（舍入）
__m128 x = _mm_cvtsi32_ss(xmm, i);         // int → f32
```

### 9.2 整数宽度转换

```c
// SSE4.1：有符号/零扩展（仅使用源的低半部分！）
__m128i r = _mm_cvtepi8_epi16(src);   // 8× i8 → 8× i16（使用 src 的低 8 字节）
__m128i r = _mm_cvtepi8_epi32(src);   // 4× i8 → 4× i32
__m128i r = _mm_cvtepi8_epi64(src);   // 2× i8 → 2× i64
__m128i r = _mm_cvtepi16_epi32(src);  // 4× i16 → 4× i32
__m128i r = _mm_cvtepi16_epi64(src);  // 2× i16 → 2× i64
__m128i r = _mm_cvtepi32_epi64(src);  // 2× i32 → 2× i64

// 零扩展变体（SSE4.1）：
__m128i r = _mm_cvtepu8_epi16(src);   // 8× u8 → 8× i16（零扩展）
__m128i r = _mm_cvtepu8_epi32(src);   // 4× u8 → 4× i32
__m128i r = _mm_cvtepu16_epi32(src);  // 4× u16 → 4× i32
__m128i r = _mm_cvtepu16_epi64(src);  // 2× u16 → 2× i64
__m128i r = _mm_cvtepu32_epi64(src);  // 2× u32 → 2× i64

// 打包（饱和窄化）：更宽 → 更窄
__m128i r = _mm_packs_epi32(a, b);    // 4+4 i32 → 8 i16（饱和, 有符号）
__m128i r = _mm_packs_epi16(a, b);    // 8+8 i16 → 16 i8（饱和, 有符号）
__m128i r = _mm_packus_epi32(a, b);   // 4+4 i32 → 8 u16（饱和, 无符号）
__m128i r = _mm_packus_epi16(a, b);   // 8+8 i16 → 16 u8（饱和, 无符号）
```

---

## 10. 按位操作

```c
// 浮点类型（适用于 __m128 和 __m128d）：
__m128 r = _mm_and_ps(a, b);     // 按位与
__m128 r = _mm_or_ps(a, b);      // 按位或
__m128 r = _mm_xor_ps(a, b);     // 按位异或
__m128 r = _mm_andnot_ps(a, b);  // (~a) & b

// 整数类型：
__m128i r = _mm_and_si128(a, b);
__m128i r = _mm_or_si128(a, b);
__m128i r = _mm_xor_si128(a, b);
__m128i r = _mm_andnot_si128(a, b);  // (~a) & b

// 常用惯用写法：
__m128 zero = _mm_xor_ps(x, x);            // 将寄存器置零（1 微操作）
__m128 abs_x = _mm_andnot_ps(_mm_set1_ps(-0.0f), x);  // 清除符号位 = fabsf
__m128 neg_x = _mm_xor_ps(x, _mm_set1_ps(-0.0f));     // 翻转符号位 = -x
```

**专业提示**：使用 `_mm_andnot_ps` 配合 `_mm_set1_ps(-0.0f)` 高效地计算
`fabsf(x)`——它清除每个浮点通道的符号位。这比 `x < 0 ? -x : x` 快得多。

---

## 11. 数学和特殊函数

### 11.1 舍入（SSE4.1）

```c
// SSE4.1：打包浮点舍入
__m128 r = _mm_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
__m128 r = _mm_round_ps(x, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);  // floor
__m128 r = _mm_round_ps(x, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);  // ceil
__m128 r = _mm_round_ps(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);     // trunc
```

### 11.2 点积（SSE4.1）

```c
// _mm_dp_ps(a, b, imm8)：带选择性累加的点积
// imm8：高半字节 = 将点积结果广播到哪些结果通道
//       低半字节 = 对哪些输入通道进行乘法
//
// 示例：dp = a[0]*b[0] + a[1]*b[1]，复制到全部 4 个通道
__m128 r = _mm_dp_ps(a, b, 0xFF);
// 0xFF = 将结果广播到所有通道，使用全部 4 个输入通道
// 等价于：broadcast(dot(a,b))

// 示例：dp = a[0]*b[0] + a[1]*b[1]，仅存储到通道 0
__m128 r = _mm_dp_ps(a, b, 0x71);
// 0x71 = 掩码 0111_0001：通道 0,1,2,3 → 1, 使用输入 0,1 → 仅 result[0]

// 性能说明：_mm_dp_ps 延迟较高（约 12-14 周期），通常比手动
// mul+hadd 更慢。仅在非热路径中为代码清晰性而使用。
```

### 11.3 绝对差值和（SSE2）

```c
// _mm_sad_epu8(a, b)：计算 8 字节组上的 Σ|a[i] - b[i]|
// Result[0..7] = Σ|a[0..7] - b[0..7]|  （作为 u16）
// Result[8..15] = Σ|a[8..15] - b[8..15]|
//
// 用于视频编解码器的运动估计（SAD 度量）
__m128i sad = _mm_sad_epu8(block_a, block_b);
int total_sad = _mm_extract_epi16(sad, 0) + _mm_extract_epi16(sad, 4);
```

---

## 12. 字符串和文本处理（SSE4.2）

### 12.1 CRC32C 硬件

```c
#include <nmmintrin.h>  // 或 <smmintrin.h>

// 硬件 CRC32C（Castagnoli 多项式：0x1EDC6F41）
uint32_t crc = 0xFFFFFFFF;
crc = _mm_crc32_u8(crc, byte);       // 处理 1 字节
crc = _mm_crc32_u16(crc, word);      // 处理 2 字节
crc = _mm_crc32_u32(crc, dword);     // 处理 4 字节
crc = _mm_crc32_u64(crc, qword);     // 处理 8 字节（仅 x86-64）
crc ^= 0xFFFFFFFF;  // 最终 XOR
```

### 12.2 字符串比较（SSE4.2）

```c
// _mm_cmpestri：比较显式长度字符串，带范围控制
// 返回第一个匹配或不匹配的索引（签名极其复杂）
int idx = _mm_cmpestri(a, la, b, lb, _SIDD_CMP_EQUAL_ORDERED);
// idx = b 中第一个等于 a 中任意字符的位置

// _mm_cmpistri：隐式长度字符串（以 null 结尾）
int idx = _mm_cmpistri(a, b, _SIDD_CMP_EQUAL_ANY);
// idx = b 中也出现在 a 中的第一个字符的位置

// 用于 simdjson 和快速字符串搜索库
```

---

## 13. 完整 SSE 示例：向量加法 + 基准测试

```c
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <time.h>

// SSE 向量加法：c = a + b
__attribute__((noinline))
void vec_add_sse(const float* a, const float* b, float* c, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 vc = _mm_add_ps(va, vb);
        _mm_storeu_ps(c + i, vc);
    }
    for (; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 1000003;  // 质数，用于测试尾部处理
    float *a = (float*)_mm_malloc(n * sizeof(float), 16);
    float *b = (float*)_mm_malloc(n * sizeof(float), 16);
    float *c = (float*)_mm_malloc(n * sizeof(float), 16);

    for (int i = 0; i < n; i++) {
        a[i] = (float)i;
        b[i] = (float)(n - i);
    }

    // 基准测试
    clock_t start = clock();
    for (int iter = 0; iter < 1000; iter++)
        vec_add_sse(a, b, c, n);
    clock_t end = clock();

    // 验证
    int errors = 0;
    for (int i = 0; i < n; i++)
        if (c[i] != a[i] + b[i]) errors++;
    printf("SSE 向量加法：%d 错误, 耗时 %.3f ms\n",
           errors, (double)(end - start) * 1000 / CLOCKS_PER_SEC);

    _mm_free(a); _mm_free(b); _mm_free(c);
    return errors;
}
```

编译：
```bash
# SSE4.1 广泛可用（2008 年起）
gcc -msse4.1 -O2 -o vec_add_sse vec_add_sse.c

# 或目标 SSE2 基线
gcc -msse2 -O2 -o vec_add_sse vec_add_sse.c
```

---

## 14. SSE vs AVX/AVX-512 对比

| 特性 | SSE | AVX2 | AVX-512 |
|---------|-----|------|---------|
| 寄存器宽度 | 128 位 | 256 位 | 512 位 |
| f32 元素数 | 4 | 8 | 16 |
| 寄存器数（x86-64） | 16 | 16 | 32 |
| FMA | 无（mul+add = 2 操作） | 有（vfmadd = 1 操作） | 有 |
| 3 操作数编码 | 否 | 是（VEX） | 是（EVEX） |
| 掩码寄存器 | 无（使用 blend） | 无（使用 blend） | 有（k0-k7） |
| CPU 支持 | 所有 x86-64 | Haswell+（2013） | Skylake-X+（2017） |
| 适用场景 | 遗留兼容, 仅 128 位操作 | 通用用途 | ML 推理, HPC |

**在新代码中何时使用 SSE（2024 年起）：**
- 需要兼容 2013 年之前的硬件（如果必须支持的话）
- 4 宽度是最佳宽度的操作（例如处理 3D 向量如 xyz 坐标——
  4 个浮点数完美匹配 3D+填充模式）
- 代码大小受限（SSE 指令比 VEX/EVEX 编码的指令更短）

**对于其他所有情况，应将 AVX2 作为你的基准目标。**

---

## 15. 参考表：SSE 头文件

```c
#include <xmmintrin.h>   // SSE:   __m128, _mm_*_ps, _mm_load_ps 等
#include <emmintrin.h>   // SSE2:  __m128d, __m128i, _mm_*_pd, _mm_*_si128
#include <pmmintrin.h>   // SSE3:  _mm_hadd_ps, _mm_lddqu_si128 等
#include <tmmintrin.h>   // SSSE3: _mm_shuffle_epi8, _mm_alignr_epi8 等
#include <smmintrin.h>   // SSE4.1: _mm_blendv_*, _mm_dp_ps, _mm_mullo_epi32
                          // SSE4.2: _mm_crc32_*, _mm_cmpestri
#include <nmmintrin.h>   // SSE4.2（更详细的分解）
#include <immintrin.h>   // 以上所有 + AVX/AVX2/AVX-512/FMA/BMI

// 在实践中，只需使用：
#include <immintrin.h>
```

（文件结束 - 共 488 行）