# AVX-512 Intrinsics 深入指南

## 1. 数据类型与寄存器

### 1.1 向量类型

| 类型 | 内容 | 元素数 | 总位宽 |
|------|------|--------|--------|
| `__m512` | f32 浮点 | 16 | 512 |
| `__m512d` | f64 浮点 | 8 | 512 |
| `__m512i` | 通用整数 | 取决于指令 | 512 |

**与 AVX2 的对比**：
```
AVX2 __m256:   [f0|f1|f2|f3|f4|f5|f6|f7]          (256-bit, 8xf32)
AVX-512 __m512: [f0|f1|f2|f3|f4|f5|f6|f7|f8|f9|f10|f11|f12|f13|f14|f15]  (512-bit, 16xf32)
```

### 1.2 掩码类型（k-register）

```c
// 掩码寄存器的 C 类型
typedef unsigned char  __mmask8;   // 8 位掩码（用于 8/16 位元素宽度）
typedef unsigned short __mmask16;  // 16 位掩码（用于 32 位元素宽度，如 __m512）
typedef unsigned int   __mmask32;  // 32 位掩码（较少使用）
typedef unsigned long long __mmask64;  // 64 位掩码（用于 8 位元素宽度，如 64 个 i8）

// 掩码的每个 bit 对应向量中的一个元素
// bit = 1: 该元素参与操作
// bit = 0: 该元素被屏蔽
```

**实践中**：
- `__m512`（16×f32）→ 掩码类型是 `__mmask16`（16 bits）
- `__m512d`（8×f64）→ 掩码类型是 `__mmask8`（8 bits）
- `__m512i` 作为 64×i8 → 掩码类型是 `__mmask64`

## 2. 掩码操作：AVX-512 的核心竞争力

### 2.1 比较生成掩码（直接到 k 寄存器）

这是 AVX-512 相比 AVX2 最大的提升之一。AVX2 中比较结果是一个全 0/1 的 256 位向量，需要 `movemask` 提取到 GPR 或 `blendv` 消费。AVX-512 直接生成 k 寄存器：

```c
// 比较 → __mmask16，一条指令
__m512 a = _mm512_loadu_ps(src);
__m512 b = _mm512_set1_ps(0.0f);

__mmask16 mask_eq  = _mm512_cmp_ps_mask(a, b, _CMP_EQ_OQ);   // a == b
__mmask16 mask_gt  = _mm512_cmp_ps_mask(a, b, _CMP_GT_OQ);   // a > b
__mmask16 mask_ge  = _mm512_cmp_ps_mask(a, b, _CMP_GE_OQ);   // a >= b
__mmask16 mask_lt  = _mm512_cmp_ps_mask(a, b, _CMP_LT_OQ);   // a < b
__mmask16 mask_neq = _mm512_cmp_ps_mask(a, b, _CMP_NEQ_OQ);  // a != b

// 也可比较整数（返回 __mmask16 或其他宽度）
__mmask16 mask_i32_eq = _mm512_cmpeq_epi32_mask(a_i, b_i);
__mmask16 mask_i32_gt = _mm512_cmpgt_epi32_mask(a_i, b_i);
```

**注意后缀**：`_mask` 后缀表示结果直接进入 k 寄存器，而不是向量寄存器。

### 2.2 掩码算术：零化与合并

AVX-512 有三种掩码模式，体现在 intrinsic 命名中：

```c
__m512 a = _mm512_loadu_ps(src1);
__m512 b = _mm512_loadu_ps(src2);
__mmask16 mask = 0b1010101010101010;  // 只有偶数 lane 参与

// 模式 1: zero-masking（零化掩码）
// 被屏蔽的 lane 写入 0
__m512 r_z = _mm512_maskz_add_ps(mask, a, b);
// r_z[i] = mask[i] ? a[i]+b[i] : 0.0f

// 模式 2: merge-masking（合并掩码）
// 被屏蔽的 lane 保留第一个源的值
__m512 r_m = _mm512_mask_add_ps(a, mask, a, b);
// 注意：第一个参数 a 同时作为"无效时保留的值"和"被加数"
// r_m[i] = mask[i] ? a[i]+b[i] : a[i]

// 模式 3: 无掩码（全 active），即正常的 AVX-512 指令
__m512 r = _mm512_add_ps(a, b);
// 等同于 mask=0xFFFF，所有 lane 参与
```

**merge-masking 的高级用法**：第一个源不一定是 `a`，可以是单独的数据：

```c
__m512 fallback = _mm512_set1_ps(-1.0f);  // 默认值
__m512 result = _mm512_mask_add_ps(fallback, mask, a, b);
// result[i] = mask[i] ? a[i]+b[i] : -1.0f
```

**性能**：zero-masking 比 merge-masking 略快，因为不需要读取第一个源寄存器的"不参与"部分。zero-masking 可使用消除寄存器依赖的机制（dependency breaking idiom）。

### 2.3 k 寄存器逻辑操作

```c
__mmask16 m1 = _mm512_cmp_ps_mask(x, zero, _CMP_GT_OQ);   // x > 0
__mmask16 m2 = _mm512_cmp_ps_mask(x, one, _CMP_LT_OQ);    // x < 1

// 逻辑与：在两个条件都满足的位置操作
__mmask16 m_and = _kand_mask16(m1, m2);  // 0 < x < 1

// 逻辑或
__mmask16 m_or = _kor_mask16(m1, m2);

// 逻辑非
__mmask16 m_not = _knot_mask16(m1);

// 逻辑异或
__mmask16 m_xor = _kxor_mask16(m1, m2);

// 也可以直接使用 C 位操作（因为 __mmask16 就是 unsigned short）
__mmask16 m_combined = m1 & m2 | m_extra;
// 两种方式生成相同的机器码
```

**使用场景**：构建复杂条件：

```c
// 在 (0, 1) 区间内执行加法，否则保留原值
__mmask16 active = _kand_mask16(
    _mm512_cmp_ps_mask(x, zero, _CMP_GT_OQ),
    _mm512_cmp_ps_mask(x, one, _CMP_LT_OQ)
);
x = _mm512_mask_add_ps(x, active, x, _mm512_set1_ps(10.0f));
```

### 2.4 移动掩码到 GPR

```c
// 场景：需要知道哪些 lane 满足条件（例如 break/continue）
__mmask16 mask = _mm512_cmp_ps_mask(data, threshold, _CMP_GT_OQ);

if (mask != 0) {
    // 有 lane 满足条件
    // 遍历所有置位 bit
    while (mask) {
        int idx = __builtin_ctz(mask);  // 最低置位 bit 的索引
        // 处理 data[idx]
        mask &= mask - 1;  // 清除最低置位
    }
}
```

### 2.5 掩码加载/存储

```c
// 掩码加载：只有掩码 bit=1 的 lane 才从内存加载
__m512 data;
__mmask16 load_mask = (1 << n) - 1;  // n 个元素

// merge-masking 加载：未加载的 lane 保留原值
data = _mm512_mask_loadu_ps(data, load_mask, src);

// zero-masking 加载：未加载的 lane 设为 0
data = _mm512_maskz_loadu_ps(load_mask, src);

// 掩码存储：只有掩码 bit=1 的 lane 才写入内存
_mm512_mask_storeu_ps(dst, store_mask, data);
```

**处理不定长数组尾部**的优雅方案：

```c
// AVX2 方法：需要单独的标量循环处理尾部
// AVX-512 方法：用掩码一次性处理
void vec_add_avx512(const float* a, const float* b, float* c, int n) {
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        _mm512_storeu_ps(c + i, _mm512_add_ps(va, vb));
    }
    if (i < n) {
        int remaining = n - i;
        __mmask16 mask = (1 << remaining) - 1;
        __m512 va = _mm512_maskz_loadu_ps(mask, a + i);
        __m512 vb = _mm512_maskz_loadu_ps(mask, b + i);
        _mm512_mask_storeu_ps(c + i, mask, _mm512_add_ps(va, vb));
    }
}
```

## 3. Compress 和 Expand：稀疏数据处理

### 3.1 Compress（压缩）

将掩码选中的元素从稀疏排列变为连续排列：

```c
// _mm512_mask_compress_ps: 将 mask=1 的元素紧凑地存入内存
// 输入：a = [A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P]
// mask = 0b1001001001001001  (每 4 个选 1 个)
// 输出 dst = [A, E, I, M]  (仅 4 个元素，连续存放)

__mmask16 mask = /* 稀疏掩码 */;
float dst[16];
_mm512_mask_compressstoreu_ps(dst, mask, data);
// dst 的前 popcount(mask) 个元素被填入 mask 选中的值
```

**使用场景**：过滤操作。例如从数组中选出所有满足条件的元素：

```c
// 稀疏过滤：找出所有 > 0 的元素，紧凑存放
void filter_positive(const float* src, float* dst, int n, int* out_count) {
    int written = 0;
    for (int i = 0; i < n; i += 16) {
        __m512 v = _mm512_loadu_ps(src + i);
        __mmask16 mask = _mm512_cmp_ps_mask(v, _mm512_setzero_ps(), _CMP_GT_OQ);
        _mm512_mask_compressstoreu_ps(dst + written, mask, v);
        written += __builtin_popcount(mask);
    }
    *out_count = written;
}
```

### 3.2 Expand（展开）

compress 的逆操作：将连续数据展开到稀疏位置：

```c
// _mm512_mask_expand_ps: 将紧凑输入展开到 mask=1 的位置
// 输入 src = [A, B, C, D, E]  (连续)
// mask = 0b1001001001001001
// 输出：mask=1 的位置依次填入 src[0], src[1], ...

__m512 result = _mm512_maskz_expandloadu_ps(mask, src);
// result[mask=1 的第 0 位] = A
// result[mask=1 的第 1 位] = B
// ...
```

**使用场景**：从紧凑的索引/值列表中恢复稀疏向量表示。

### 3.3 性能考量

compress/expand 是**微码实现**的复杂操作（不是单周期 µop），延迟 ~15-20 个周期。但它们的等效标量循环更慢（需要判断、分支、压缩存储），所以 SIMD compress 在大规模数据过滤中仍然有明显的净收益。

## 4. 冲突检测（Conflict Detection）

### 4.1 基本概念

`_mm512_conflict_epi32` 检测 scatter 操作中的写冲突（多个 lane 试图写同一个索引）：

```c
// indices = [3, 7, 3, 5, 2, 7, 1, 8, 0, 4, 6, 9, 2, 5, 8, 1]
// 冲突：
//   indices[0] = 3, indices[2] = 3  → lane 0 和 lane 2 冲突
//   indices[1] = 7, indices[5] = 7  → lane 1 和 lane 5 冲突

__m512i indices = _mm512_loadu_si512(index_data);
__m512i conflicts = _mm512_conflict_epi32(indices);
// conflicts[i] 的 bit j 为 1 表示 lane i 和 lane j 有相同的 index 值
// conflicts[2] 的 bit 0 为 1（lane 2 和 lane 0 都写 index=3）
// conflicts[5] 的 bit 1 为 1（lane 5 和 lane 1 都写 index=7）
```

### 4.2 直方图/哈希表应用

```c
// 安全地 scatter-add 到直方图
void histogram_add_safe(int* histogram, const int* indices, const float* values, int n) {
    for (int i = 0; i + 16 <= n; i += 16) {
        __m512i idx = _mm512_loadu_si512((__m512i*)(indices + i));
        __m512 val = _mm512_loadu_ps(values + i);
        
        // 检测冲突
        __m512i conflict = _mm512_conflict_epi32(idx);
        __mmask16 no_conflict = _mm512_testn_epi32_mask(conflict, conflict);
        
        // 无冲突：直接 scatter
        if (no_conflict == 0xFFFF) {
            _mm512_i32scatter_ps(histogram, idx, val, 4);
        } else {
            // 有冲突：逐元素安全处理
            for (int j = 0; j < 16; j++) {
                histogram[indices[i + j]] += values[i + j];
            }
        }
    }
}
```

## 5. 嵌入式舍入和 SAE

### 5.1 什么是 SAE

SAE = Suppress All Exceptions（抑制所有浮点异常）+ 指定舍入模式。允许在不修改全局 MXCSR 寄存器的情况下改变舍入行为：

```c
// 标准加法：使用 MXCSR 中的默认舍入模式（通常为最近偶数）
__m512 c = _mm512_add_ps(a, b);

// SAE 加法：指定舍入模式且抑制异常
__m512 c_trunc = _mm512_add_round_ps(a, b, 
    _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);

// 可用舍入模式：
// _MM_FROUND_TO_NEAREST_INT     - 舍入到最近偶数（默认）
// _MM_FROUND_TO_NEG_INF         - 向负无穷（floor）
// _MM_FROUND_TO_POS_INF         - 向正无穷（ceil）
// _MM_FROUND_TO_ZERO            - 向零（trunc）
// _MM_FROUND_CUR_DIRECTION      - 使用 MXCSR 当前设置
```

### 5.2 使用场景

**高精度求和**（Kahan summation 风格的补偿）：

```c
// 使用向负无穷舍入计算低估值，使用向正无穷舍入计算高估值
// 两者之间的差值就是舍入误差的上界
__m512 lo = _mm512_add_round_ps(a, b,
    _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
__m512 hi = _mm512_add_round_ps(a, b,
    _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
```

**神经网络量化**（模拟低精度舍入行为）：

```c
// 将 fp32 截断到 int8 范围并舍入
__m512 x = _mm512_loadu_ps(data);
x = _mm512_mul_round_ps(x, scale, 
    _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);  // 明确截断
```

## 6. AVX-512 与其他指令集的差异

### 6.1 寄存器数量翻倍

```
AVX2：  16 个 256 位 YMM 寄存器 (ymm0-ymm15)
AVX-512：32 个 512 位 ZMM 寄存器 (zmm0-zmm31)
```

这改变了很多代码设计：

```c
// AVX2 GEMM：寄存器极其紧张，需要频繁的 evict/reload
// 6x16 微内核中，6 个 A 寄存器 + 6 个累加器 + 加载和广播的临时寄存器
// 经常溢出 16 个 YMM 寄存器，导致栈溢出 (spill)

// AVX-512 GEMM：寄存器充足，可以更激进地展开
// 12x16 微内核中，12 个 A 寄存器 + 12 个累加器 + 临时寄存器
// 32 个 ZMM 寄存器很少有溢出
```

### 6.2 掩码替代 Blend

```c
// AVX2 条件赋值：
__m256 mask = _mm256_cmp_ps(x, zero, _CMP_GT_OS);
__m256 result = _mm256_blendv_ps(neg_part, x, mask);
// 需要：cmp → blendv（2 条指令）

// AVX-512 等价：
__mmask16 mask = _mm512_cmp_ps_mask(x, zero, _CMP_GT_OQ);
__m512 result = _mm512_mask_mul_ps(neg_part, mask, x, factor);
// cmp_mask 是"免费的"（结果直接进入 k 寄存器，不占用向量端口）
// mask_mul 一条指令完成"条件乘法"（被屏蔽的位置保留 neg_part）
```

### 6.3 更强大的 Permute

AVX-512 有原生跨 512 位全宽度的 permute：

```c
// _mm512_permutexvar_ps: 在整个 512 位范围内按索引重排
// 没有 128/256 位 lane 边界限制！
__m512i idx = _mm512_setr_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
__m512 rev = _mm512_permutexvar_ps(idx, data);  // 完全反转 16 个元素

// AVX2 实现同样的反转需要：
// extract high 128 + reverse + insert + extract low 128 + reverse + insert + merge
// 至少 6-8 条指令！
```

## 7. AVX-512 上的寄存器压力与函数调用

### 7.1 调用约定

x86-64 System V ABI 中：
- zmm0-zmm7 用于参数传递（前 8 个 `__m512` 参数通过寄存器传递）
- zmm0-zmm7 是调用者保存的（caller-saved）
- zmm8-zmm31 是被调用者保存的（callee-saved）

```c
// 这个函数的前 8 个 __m512 参数通过 zmm0-zmm7 传递
__m512 compute(__m512 a, __m512 b, __m512 c, __m512 d,
               __m512 e, __m512 f, __m512 g, __m512 h) {
    return _mm512_add_ps(_mm512_add_ps(a, b),
           _mm512_add_ps(_mm512_add_ps(c, d),
           _mm512_add_ps(_mm512_add_ps(e, f),
           _mm512_add_ps(g, h))));
}
// 第 9 个及更多参数通过栈传递
```

### 7.2 Callee-saved 寄存器利用

```c
// 在内部循环中可以使用 zmm16-zmm31 而不需要保存/恢复
// 但在跨越函数调用时必须保存
__m512 inner_loop(__m512 input) {
    // zmm16-zmm31 在此函数中可以自由使用
    // 但调用其他函数后，它们的值可能被破坏
    __m512 temp1 = _mm512_loadu_ps(data);
    __m512 temp2 = _mm512_loadu_ps(data2);
    // 调用外部库函数 → 需要手动保存 zmm16-zmm31 或在调用前存到内存
    return _mm512_add_ps(temp1, temp2);
}
```

## 8. 降频问题：真相与对策

### 8.1 历代 CPU 的 AVX-512 频率表现

| CPU | 微架构 | AVX-512 频率表现 |
|-----|--------|-----------------|
| Skylake-X/Skylake-SP | Skylake (2017) | 显著降频（几百 MHz），首次实现有热密度问题 |
| Cascade Lake | Cascade Lake (2019) | 改善但仍存在 |
| Ice Lake | Sunny Cove (2019) | 大幅改善，降频通常 < 100 MHz |
| Tiger Lake | Willow Cove (2020) | 进一步改善 |
| Sapphire Rapids | Golden Cove (2023) | 基本不是问题 |
| AMD Zen4 | Zen4 (2022) | 几乎不降频（双泵 256 位） |

### 8.2 工程建议

1. **不要因为"AVX-512 降频"的古老恐惧而放弃使用它**
2. 在 2020+ 的硬件上，AVX-512 的 2x 吞吐优势通常远大于任何频率降低
3. 如果非常在意功耗/发热（如无风扇边缘设备），可以考虑限制 AVX-512 的使用频率
4. AMD Zen4 几乎没有 AVX-512 频率惩罚

## 9. 完整示例：AVX-512 向量归约

### 9.1 AVX2 vs AVX-512 归约对比

```c
// AVX2 水平求和（需要约 8 条指令）
float reduce_sum_avx2(__m256 v) {
    v = _mm256_hadd_ps(v, v);           // 1
    v = _mm256_hadd_ps(v, v);           // 2
    __m128 lo = _mm256_extractf128_ps(v, 0); // 3
    __m128 hi = _mm256_extractf128_ps(v, 1); // 4
    __m128 s = _mm_add_ps(lo, hi);           // 5
    s = _mm_hadd_ps(s, s);                   // 6
    s = _mm_hadd_ps(s, s);                   // 7
    return _mm_cvtss_f32(s);
}

// AVX-512 水平求和（编译器内建的 reduce）
float reduce_sum_avx512(__m512 v) {
    return _mm512_reduce_add_ps(v);  // 一条 intrinsic，编译器生成最优指令序列
}

// 自己实现 AVX-512 归约（了解底层原理）
float reduce_sum_avx512_manual(__m512 v) {
    // Step 1: 将 512 位分成两半，逐元素相加
    __m256 lo = _mm512_castps512_ps256(v);        // 低 256 位
    __m256 hi = _mm512_extractf32x8_ps(v, 1);     // 高 256 位
    __m256 sum256 = _mm256_add_ps(lo, hi);
    
    // Step 2: 在 256 位内做 hadd + extract
    sum256 = _mm256_hadd_ps(sum256, sum256);
    sum256 = _mm256_hadd_ps(sum256, sum256);
    __m128 lo128 = _mm256_castps256_ps128(sum256);
    __m128 hi128 = _mm256_extractf128_ps(sum256, 1);
    __m128 sum128 = _mm_add_ps(lo128, hi128);
    // sum128[0] = 全和
    return _mm_cvtss_f32(sum128);
}
```

### 9.2 掩码归约（只对掩码选中的元素求和）

```c
float masked_reduce_sum(__m512 v, __mmask16 mask) {
    // 方法 1: zero masking 后归约（未选中的被清零，不影响和）
    __m512 masked = _mm512_maskz_mov_ps(mask, v);
    return _mm512_reduce_add_ps(masked);
    
    // 方法 2: 仅在 mask=1 的位置执行加法
    // 更复杂但更通用（例如当 mask=0 的位置有 NaN 时不会被归约激活）
}
```

## 10. 编译与调试

### 10.1 编译器支持

```bash
# GCC/Clang 启用 AVX-512 
gcc -mavx512f -mavx512bw -mavx512vl -mavx512dq -O2 -o prog prog.c

# 启用 VNNI（神经网络推理指令）
gcc -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -O2 -o prog prog.c

# 检查生成的汇编
gcc -mavx512f -S -o prog.s prog.c

# 检查代码是否使用了 AVX-512 指令
objdump -d prog | grep -E 'vaddps|vfmadd|vmovaps|knot|kmov'
```

### 10.2 运行时特性检测

```c
#include <cpuid.h>

int cpu_has_avx512f(void) {
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return (ebx & (1u << 16)) != 0;  // AVX-512F
    }
    return 0;
}

int cpu_has_avx512bw(void) {
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return (ebx & (1u << 30)) != 0;  // AVX-512BW
    }
    return 0;
}

int cpu_has_avx512vl(void) {
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return (ebx & (1u << 31)) != 0;  // AVX-512VL
    }
    return 0;
}
```

### 10.3 函数多版本（运行时自动选择）

```c
// 默认版本：纯 C 实现
__attribute__((target("default")))
float compute_norm(const float* x, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) sum += x[i] * x[i];
    return sqrtf(sum);
}

// AVX2 优化版本
__attribute__((target("avx2,fma")))
float compute_norm(const float* x, int n) {
    __m256 sum = _mm256_setzero_ps();
    int i;
    for (i = 0; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        sum = _mm256_fmadd_ps(v, v, sum);
    }
    float result = reduce_sum_avx2(sum);
    for (; i < n; i++) result += x[i] * x[i];
    return sqrtf(result);
}

// AVX-512 优化版本
__attribute__((target("avx512f")))
float compute_norm(const float* x, int n) {
    __m512 sum = _mm512_setzero_ps();
    int i;
    for (i = 0; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(x + i);
        sum = _mm512_fmadd_ps(v, v, sum);
    }
    float result = _mm512_reduce_add_ps(sum);
    for (; i < n; i++) result += x[i] * x[i];
    return sqrtf(result);
}
```

### 10.4 常见编译器优化陷阱

```c
// 陷阱：编译器可能重新排序掩码操作，破坏正确性
// 解决方案：使用内联汇编或 __asm__ volatile 阻止不想要的优化

// 陷阱：局部变量声明为 __m512 而不初始化
__m512 v;  // 未初始化！可能被优化掉
v = _mm512_loadu_ps(src + i);  // 正确

// 陷阱：_mm512_setzero_ps() 不会真的清零——它依赖的是一条异或自身指令
// 编译器可能消除"无用"的代码。确保所有计算都产生可见的副作用。
```

## 11. AVX-512 设计的完整循环示例

```c
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

// 使用 AVX-512 掩码优雅处理任意长度数组
// c[i] = a[i] > 0 ? a[i] : 0   (ReLU)
void relu_avx512(const float* src, float* dst, int n) {
    int i = 0;
    __m512 zero = _mm512_setzero_ps();
    
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(src + i);
        __mmask16 pos = _mm512_cmp_ps_mask(v, zero, _CMP_GT_OQ);
        _mm512_mask_storeu_ps(dst + i, pos, v);
        // mask=0（非正数）的位置不写入 dst，dst 保留原值
        // 注意：这里假设 dst 已经被初始化为零
    }
}

// 带初始化的 ReLU（零化掩码）
void relu_avx512_init(const float* src, float* dst, int n) {
    int i = 0;
    __m512 zero = _mm512_setzero_ps();
    
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(src + i);
        __mmask16 pos = _mm512_cmp_ps_mask(v, zero, _CMP_GT_OQ);
        _mm512_mask_storeu_ps(dst + i, pos, v);
    }
    
    // 尾部处理（通过掩码，无需单独循环）
    if (i < n) {
        __mmask16 tail_mask = (1u << (n - i)) - 1;
        __m512 v = _mm512_maskz_loadu_ps(tail_mask, src + i);
        __mmask16 pos = _mm512_cmp_ps_mask(v, zero, _CMP_GT_OQ);
        __mmask16 store_mask = tail_mask & pos;  // 只处理确实在范围内的正数
        // 实际需要初始化尾部元素为 0（这里简化处理）
        _mm512_mask_storeu_ps(dst + i, pos, v);
    }
}

int main() {
    int n = 100;
    float *src = (float*)_mm_malloc(n * sizeof(float), 64);
    float *dst = (float*)_mm_malloc(n * sizeof(float), 64);
    
    for (int i = 0; i < n; i++) src[i] = (float)(i - 50);  // -50..49
    
    // 初始化 dst 为 0
    for (int i = 0; i < n; i++) dst[i] = 0.0f;
    
    relu_avx512(src, dst, n);
    
    for (int i = 0; i < 5; i++)
        printf("ReLU(%+.0f) = %+.0f\n", src[i], dst[i]);
    
    _mm_free(src); _mm_free(dst);
    return 0;
}
```

编译运行：
```bash
gcc -mavx512f -mavx512bw -mavx512vl -O2 -o relu512 relu512.c
./relu512
```

## 12. AVX-512 学习要领

### 12.1 从 AVX2 迁移的优先事项

1. **先用掩码替代 blend**：这是收益/成本比最高的改进
2. **然后利用寄存器翻倍**：减少 spill/fill，提高展开因子
3. **再使用 compress/expand**：替代手写的稀疏循环
4. **最后用 VNNI**：如果工作负载是 ML 推理

### 12.2 必须注意的陷阱

- EVEX 编码的指令比 VEX 编码长（6 vs 3-4 字节），轻微影响指令缓存密度
- 使用 `_mm512_maskz_*` 而不是 `_mm512_mask_*` 当不需要保留旧值时，可以断开依赖链
- 在 AMD Zen4 上，512 位指令被拆成两个 256 位 µop，性能行为有所不同
- 不要盲目 512 位化所有代码：有些算法在 256 位宽度下因为更好的时钟频率而相同或更快
