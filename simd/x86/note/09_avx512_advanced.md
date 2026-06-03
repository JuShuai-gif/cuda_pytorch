# AVX-512 高级模式

```
-------------------------------------------------------------------------------
目标指令集:     AVX-512F, AVX-512BW, AVX-512VL, AVX-512DQ, AVX-512CD
硬件平台:       Skylake-X/SP, Cascade Lake, Ice Lake, Sapphire Rapids, Zen4+
参考文献:       Modern X86 Assembly Language Programming, 2nd Ed, 第 9-11 章
关键特性:       冲突检测, 压缩/扩展, 嵌入式舍入, FP16
-------------------------------------------------------------------------------
```

---

## 1. 冲突检测 (AVX-512CD)

### 1.1 问题：分散写入冲突

分散操作使用每个通道的索引写入内存。当两个通道具有相同的索引时，就会发生写入冲突：

```c
// 内存映射: histogram[0..15] = 0
// indices = [3, 7, 3, 5, 7, 2, 1, 8]
// 通道 0 写入 histogram[3], 通道 2 也写入 histogram[3]
// → 数据竞争！结果取决于哪个通道先执行。
```

`_mm512_conflict_epi32` 通过将每个通道的索引与其他所有通道进行比较，并生成冲突位图来检测这种情况。

### 1.2 API 与语义

```c
#include <immintrin.h>  // 使用 -mavx512cd 编译时包含 AVX-512CD

// 结果的每个通道 i 包含一个位掩码:
//   bit j = 1  当 indices[i] == indices[j] 且 j <= i
// 对于通道 0，indices[0]=3，且通道 2 也是 3:
//   conflict[0] = 0b00000101 (第 0 位和第 2 位被设置)
//   conflict[2] = 0b00000101 (第 0 位和第 2 位被设置)

__m512i conflicts = _mm512_conflict_epi32(indices);

// 检测是否有任何通道存在冲突（与更早的通道）
// testn = test NOT: 在 (a & b) == 0 的掩码位置返回 1
// 与自身冲突时第 0 位被设置，所以我们需要的是 ~(conflict & conflict)
// 实际上: _mm512_testn_epi32_mask(conflict, broadcast) 按通道检查
// 更实用的做法: 检查 conflict 是否只设置了自身位
__m512i self_only = _mm512_set1_epi32(1); // 仅第 0 位
// 对于每个通道，最低置位应该是它自身
// 最简单的方法: 检查是否有通道 0 以外的冲突位
__m512i others = _mm512_srli_epi32(conflicts, 1); // 移出自身位
__mmask16 has_conflict = _mm512_test_epi32_mask(others, others);
if (has_conflict) {
    // 回退到此数据块的标量路径
}
```

### 1.3 使用冲突检测的安全分散操作

```c
// 生产级直方图更新，带冲突检测
void histogram_add_safe_avx512(int* histogram, const int* indices,
                               const float* values, int n) {
    for (int i = 0; i + 16 <= n; i += 16) {
        __m512i idx = _mm512_loadu_si512((__m512i*)(indices + i));
        __m512 val = _mm512_loadu_ps(values + i);

        // 检测冲突
        __m512i conflict = _mm512_conflict_epi32(idx);
        // 如果任何通道与其他通道有冲突:
        // conflict[j] & (conflict[j]-1) 移除自身位 (第 j 位)
        // 如果结果 != 0，则存在冲突
        // 按通道检查:
        __m512i shifted = _mm512_srli_epi32(conflict, 1);
        __mmask16 has_conflict = _mm512_test_epi32_mask(shifted, shifted);

        if (has_conflict == 0) {
            // 无冲突: 使用快速分散加法
            _mm512_i32scatter_ps(histogram, idx, val, 4);
        } else {
            // 存在冲突: 标量回退
            for (int j = 0; j < 16; j++) {
                histogram[indices[i + j]] += (int)values[i + j];
            }
        }
    }
}
```

### 1.4 性能特征

| 操作 | 延迟 | 吞吐量 | 备注 |
|-----------|---------|------------|-------|
| `_mm512_conflict_epi32` | ~3 周期 | 1/周期 | 轻量级，始终值得检查 |
| `_mm512_i32scatter_ps` | ~20-25 周期 | 1/6 周期 | 开销大；尽可能避免使用 |
| 标量回退 | 16x 内存操作 | 可变 | 仅用于冲突通道 |

**经验法则**: 先检查冲突；仅在确保无冲突时使用分散操作。冲突检测增加的开销可以忽略不计（16 个通道约 3 个周期），但在冲突情况下可节省 10 倍以上的周期。

---

## 2. 压缩与扩展 (AVX-512F/VL)

### 2.1 压缩：从稀疏掩码到密集打包

`_mm512_mask_compress_ps` 取出 `mask[i] = 1` 的元素，并将它们连续地打包到内存中：

```c
// 输入:  data  = [A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P]
//         mask  =  1  0  1  0  0  1  0  0  0  1  0  0  1  0  0  0
// 输出: dst   = [A, C, F, J, M]  (仅 5 个元素，连续排列)

__m512 data = _mm512_loadu_ps(src);
__mmask16 mask = _mm512_cmp_ps_mask(data, zero, _CMP_GT_OQ);
float dst[16];
_mm512_mask_compressstoreu_ps(dst, mask, data);
size_t count = __builtin_popcount(mask);
// dst[0..count-1] 包含压缩后的值
```

### 2.2 扩展：从密集输入到稀疏放置

逆操作：取出密集数组并将其分散回掩码指定的位置：

```c
// 输入:  src  = [A, B, C, D, E]  (密集, 5 个值)
//         mask =  1  0  1  0  0  1  0  0  0  1  0  0  1  0  0  0
// 输出: dst  = [A, 0, B, 0, 0, C, 0, 0, 0, D, 0, 0, E, 0, 0, 0]

float src[16] = {A, B, C, D, E};
__mmask16 mask = 0b0000001001001001; // 5 位被设置
__m512 result = _mm512_maskz_expandloadu_ps(mask, src);
```

### 2.3 用例：过滤

```c
// 过滤: 保留大于阈值的元素，返回计数
size_t filter_positive_avx512(const float* src, float* dst, size_t n) {
    const __m512 zero = _mm512_setzero_ps();
    size_t written = 0;

    for (size_t i = 0; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(src + i);
        __mmask16 pos = _mm512_cmp_ps_mask(v, zero, _CMP_GT_OQ);
        _mm512_mask_compressstoreu_ps(dst + written, pos, v);
        written += __builtin_popcount((unsigned int)pos);
    }

    // 尾部: 标量处理
    for (size_t i = n - (n % 16); i < n; i++) {
        if (src[i] > 0.0f) dst[written++] = src[i];
    }
    return written;
}
```

### 2.4 性能

| 操作 | 延迟 | 吞吐量 |
|-----------|---------|------------|
| 压缩存储 | ~15 周期 | ~1/3 周期 |
| 扩展加载 | ~15 周期 | ~1/3 周期 |

压缩/扩展是**微码实现**的（非单条微操作），但仍比等效的标量循环快 3-10 倍，因为它们消除了分支开销并最大限度地减少了标量内存操作。

---

## 3. 嵌入式舍入与 SAE (AVX-512F)

### 3.1 抑制所有异常 (SAE)

SAE 抑制浮点异常（下溢、上溢、不精确、无效、非规格化数），并允许指定舍入模式——所有这些都不需要修改 MXCSR 寄存器：

```c
// 标准加法: 使用 MXCSR 舍入模式
__m512 c = _mm512_add_ps(a, b);

// SAE 配合舍入到最近值
__m512 c_near = _mm512_add_round_ps(a, b,
    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

// SAE 配合向零截断
__m512 c_trunc = _mm512_add_round_ps(a, b,
    _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);

// SAE 配合向下取整 (趋向负无穷)
__m512 c_floor = _mm512_add_round_ps(a, b,
    _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);

// SAE 配合向上取整 (趋向正无穷)
__m512 c_ceil = _mm512_add_round_ps(a, b,
    _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
```

### 3.2 用例

**区间算术**: 同时计算下界和上界：
```c
// 计算 lo ≤ 精确结果 ≤ hi
__m512 lo = _mm512_add_round_ps(a, b,
    _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
__m512 hi = _mm512_add_round_ps(a, b,
    _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
// 真实结果在 [lo, hi] 区间内
```

**机器学习量化模拟**: 截断中间结果以模拟 int8 精度：
```c
// 模拟 int8 量化: 截断而非舍入
__m512 q = _mm512_mul_round_ps(x, scale,
    _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
q = _mm512_add_round_ps(q, zero_point,
    _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
```

**无需修改 MXCSR 的 Kahan 求和**:
```c
// 高精度求和，无需全局修改 MXCSR
__m512 sum = _mm512_setzero_ps();
__m512 c   = _mm512_setzero_ps();  // 补偿项
for (...) {
    __m512 y = _mm512_sub_round_ps(v, c,
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 t = _mm512_add_round_ps(sum, y,
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    c = _mm512_add_round_ps(
        _mm512_sub_round_ps(t, sum,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC),
        _mm512_xor_ps(y, _mm512_set1_ps(-0.0f)),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    sum = t;
}
```

### 3.3 可用的舍入模式

| 宏 | 值 | 含义 |
|-------|-------|---------|
| `_MM_FROUND_TO_NEAREST_INT` | 0x00 | 舍入到最近偶数（默认） |
| `_MM_FROUND_TO_NEG_INF` | 0x01 | 趋向 -∞ 舍入（向下取整） |
| `_MM_FROUND_TO_POS_INF` | 0x02 | 趋向 +∞ 舍入（向上取整） |
| `_MM_FROUND_TO_ZERO` | 0x03 | 趋向零舍入（截断） |
| `_MM_FROUND_NO_EXC` | 0x08 | 抑制所有异常 (SAE) |

---

## 4. FP16 支持 (AVX-512FP16, Sapphire Rapids+)

### 4.1 FP16 转换 (AVX-512F, 所有 AVX-512 CPU 均可用)

```c
// f32 → f16: 将 16 个浮点数转换为 16 个半精度值
__m256i f16 = _mm512_cvtps_ph(floats, _MM_FROUND_TO_NEAREST_INT);

// f16 → f32: 将 16 个半精度值转换为 16 个浮点数
__m512 f32 = _mm512_cvtph_ps(f16);
```

### 4.2 原生 FP16 操作 (Sapphire Rapids+)

在支持 AVX-512FP16 的 CPU 上，可以直接对半精度进行算术运算：

```c
// 编译选项: gcc -mavx512fp16
__m512h a = _mm512_loadu_ph(src);      // 加载 32 个 fp16 值
__m512h b = _mm512_loadu_ph(src2);
__m512h c = _mm512_add_ph(a, b);       // 32 × fp16 加法!
__m512h d = _mm512_mul_ph(a, b);       // 32 × fp16 乘法
__m512h e = _mm512_fmadd_ph(a, b, c);  // 32 × fp16 FMA

// 吞吐量是 fp32 的 2 倍 (每条指令 32 次操作 vs 16 次)
```

### 4.3 BF16 支持 (AVX-512BF16, Cooper Lake+)

```c
// BF16 点积: bf16 × bf16 → fp32 累加
// _mm512_dpbf16_ps: 每个 32 位通道计算 2 个 bf16 点积
__m512i a = _mm512_loadu_si512(bf16_data);
__m512i b = _mm512_loadu_si512(bf16_weights);
__m512 acc = _mm512_dpbf16_ps(acc, a, b);
// 结果: acc[i] += a[2i]*b[2i] + a[2i+1]*b[2i+1]

// BF16 格式: bfloat16 = IEEE float32 截断低 16 位
// 尾数: 7 位 (vs fp16 的 10 位, fp32 的 23 位)
// 指数: 8 位 (与 fp32 相同, vs fp16 的 5 位)
// 动态范围: 与 fp32 相同
```

---

## 5. 置换与混洗 (AVX-512F)

### 5.1 全 512 位跨通道置换

AVX2 将大多数混洗操作限制在 128 位通道内。AVX-512 取消了这一限制：

```c
// _mm512_permutexvar_ps: 跨全部 16 个通道的任意置换
__m512i idx = _mm512_setr_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
__m512 rev = _mm512_permutexvar_ps(idx, data);  // 完全反转，仅 1 条指令!

// AVX2 需要 6-8 条指令才能完成同样的操作
```

### 5.2 各种置换指令

```c
// 在 128 位通道内置换 (与旧的 SSE/AVX 类似，但为 512 位)
__m512 r = _mm512_permute_ps(data, _MM_SHUFFLE(3,1,2,0));
// 每个 128 位通道独立混洗

// 在 256 位半区内置换 64 位元素
__m512d r = _mm512_permute_pd(data, 0b01010101);

// 跨两个源的 32 位混洗
__m512 r = _mm512_shuffle_ps(a, b, _MM_SHUFFLE(3,1,2,0));
```

### 5.3 双寄存器置换

```c
// 从两个 512 位寄存器中选择任意 8 个 64 位元素
__m512d r = _mm512_permutex2var_pd(a, idx, b);
// idx 指定从合并后的 16 个元素中选择哪些 (0-15)

// 适用于表查找、交错、矩阵转置
```

---

## 6. K 寄存器高级操作

### 6.1 K 寄存器与通用寄存器互转

```c
// 将 k 寄存器掩码移至通用寄存器
__mmask16 k = _mm512_cmp_ps_mask(a, zero, _CMP_GT_OQ);
unsigned int bits = (unsigned int)k;  // 隐式转换

// 或显式转换: _cvtmask16_u32 (_cvtu32_mask16 的逆操作)
unsigned short kmask = _cvtmask16_u32(k);  // 其实不需要，直接强制转换即可

// 从整数构建掩码
__mmask16 k2 = _cvtu32_mask16(0b1010101010101010);
```

### 6.2 K 寄存器逻辑

```c
__mmask16 k1 = _mm512_cmp_ps_mask(a, zero, _CMP_GT_OQ);   // a > 0
__mmask16 k2 = _mm512_cmp_ps_mask(a, one,  _CMP_LT_OQ);   // a < 1

// 使用逻辑操作组合
__mmask16 k_and = _kand_mask16(k1, k2);    // 0 < a < 1
__mmask16 k_or  = _kor_mask16(k1, k2);     // a < 0 或 a > 1
__mmask16 k_xor = _kxor_mask16(k1, k2);
__mmask16 k_not = _knot_mask16(k1);

// 直接使用组合后的掩码
__m512 result = _mm512_maskz_mul_ps(k_and, a, scale);
// 只有满足 0 < a < 1 的元素才会乘以 scale

// 实践中，直接使用 C 语言的位运算 (__mmask16 是整数类型)
__mmask16 combined = k1 & k2 | k_extra;
__mmask16 flipped  = ~k1 & 0xFFFF;
// 生成相同的机器码，可读性更好
```

### 6.3 K 寄存器的位计数

```c
__mmask16 mask = _mm512_cmp_ps_mask(data, threshold, _CMP_GT_OQ);
int count = __builtin_popcount((unsigned int)mask);  // 为真的通道数

// 查找第一个置位
int first = __builtin_ctz((unsigned int)mask);

// 查找最后一个置位  
int last = 31 - __builtin_clz((unsigned int)mask);
```

---

## 7. 生产级 AVX-512 循环模式

### 7.1 掩码尾部循环（零开销）

```c
// AVX-512: 无需单独的标量尾部!
void vec_process_avx512(const float* src, float* dst, size_t n,
                         float scale, float bias) {
    const __m512 vscale = _mm512_set1_ps(scale);
    const __m512 vbias  = _mm512_set1_ps(bias);
    size_t i = 0;

    // 主循环: 完整的 16 元素向量
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(src + i);
        v = _mm512_fmadd_ps(v, vscale, vbias);
        _mm512_storeu_ps(dst + i, v);
    }

    // 掩码尾部: 使用硬件掩码处理剩余元素
    if (i < n) {
        size_t remaining = n - i;
        __mmask16 mask = (__mmask16)((1u << remaining) - 1u);
        __m512 v = _mm512_maskz_loadu_ps(mask, src + i);
        v = _mm512_fmadd_ps(v, vscale, vbias);
        _mm512_mask_storeu_ps(dst + i, mask, v);
    }
}
```

### 7.2 多累加器归约

```c
// 4 路累加器以获得最大 FMA 吞吐量
float sum_avx512_4acc(const float* x, size_t n) {
    __m512 s0 = _mm512_setzero_ps();
    __m512 s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps();
    __m512 s3 = _mm512_setzero_ps();

    size_t i = 0;
    for (; i + 64 <= n; i += 64) {
        s0 = _mm512_add_ps(s0, _mm512_loadu_ps(x + i +  0));
        s1 = _mm512_add_ps(s1, _mm512_loadu_ps(x + i + 16));
        s2 = _mm512_add_ps(s2, _mm512_loadu_ps(x + i + 32));
        s3 = _mm512_add_ps(s3, _mm512_loadu_ps(x + i + 48));
    }

    s0 = _mm512_add_ps(s0, s1);
    s2 = _mm512_add_ps(s2, s3);
    s0 = _mm512_add_ps(s0, s2);

    float result = _mm512_reduce_add_ps(s0);
    for (; i < n; i++) result += x[i];
    return result;
}
```

---

## 8. AVX-512 性能优化总结

| 技术 | 收益 | 代价 |
|-----------|---------|------|
| 掩码寄存器 | 消除 blend 指令 (~3 uop → 1 uop) | 无 |
| 32 个寄存器 | 减少溢出/填充，增加展开次数 | 无 |
| 512 位宽度 | 相比 AVX2 吞吐量提升 2× | 可能触发频率降频（旧 CPU） |
| 冲突检测 | 快速的分散安全检测 | ~3 周期 |
| 压缩/扩展 | 无分支的稀疏操作 | ~15 周期（微码实现） |
| SAE | 逐指令的舍入控制 | 无 |
| 双寄存器置换 | 消除多指令混洗序列 | 无 |
| 分散/聚集 | 简化索引访问（需谨慎使用） | ~20-25 周期 |

**结论**: 在现代 x86 硬件（2020+）上，只要有 AVX-512 就应使用它。即使 Zen4 采用了双泵 256 位实现，2× 寄存器文件 + 2× 宽度 + 掩码寄存器仍能带来超过 2 倍的性能提升，这得益于寄存器压力缓解以及 blend/混洗链的消除。

---

## 9. 快速参考：AVX-512 内建函数速查表

```c
// ===== 掩码操作 =====
__mmask16 k = _mm512_cmp_ps_mask(a, b, _CMP_GT_OQ);
__mmask16 k = _mm512_cmpeq_epi32_mask(ai, bi);
__mmask16 k = _kand_mask16(k1, k2);
__mmask16 k = _knot_mask16(k1);
int count = __builtin_popcount((unsigned int)k);

// ===== 掩码算术 =====
__m512 r = _mm512_mask_add_ps(src, mask, a, b);    // 合并
__m512 r = _mm512_maskz_add_ps(mask, a, b);        // 置零
__m512 r = _mm512_mask_blend_ps(mask, a, b);       // 选择

// ===== 压缩/扩展 =====
_mm512_mask_compressstoreu_ps(dst, mask, data);     // 压缩
__m512 r = _mm512_maskz_expandloadu_ps(mask, src);  // 稀疏

// ===== 冲突检测 =====
__m512i c = _mm512_conflict_epi32(indices);

// ===== 嵌入式舍入 =====
__m512 r = _mm512_add_round_ps(a, b,
    _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);

// ===== 跨通道置换 =====
__m512 r = _mm512_permutexvar_ps(idx, data);
__m512d r = _mm512_permutex2var_pd(a, idx, b);

// ===== 归约 =====
float s = _mm512_reduce_add_ps(v);
float m = _mm512_reduce_max_ps(v);

// ===== FP16/BF16 =====
__m256i h = _mm512_cvtps_ph(f32, _MM_FROUND_TO_NEAREST_INT);
__m512 f  = _mm512_cvtph_ps(h);
__m512 acc = _mm512_dpbf16_ps(acc, a, b);          // BF16 点积
```

(文件结束 - 共 409 行)
