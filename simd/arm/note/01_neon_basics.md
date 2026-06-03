# NEON 基础：寄存器、类型、操作与陷阱

## 1. NEON 寄存器文件

### AArch64 模式下的寄存器布局

```
AArch64 NEON/浮点寄存器文件（32 个 128-bit 寄存器）:

                   128 bits
   Q0:  [ +--------------------------------+ ]
        [ |  D0 (lower 64 bits)  | D1 (upper 64 bits) | ]
        [ +--------------------------------+ ]
   Q1:  [ |  D2                | D3                | ]
   ...
   Q31: [ |  D62               | D63               | ]

   不同视角:
   Qi  = 128-bit 向量寄存器  (Q register view)
   Di  = 64-bit  向量寄存器  (D register view, 共64个)
   Si  = 32-bit  标量浮点     (S register view, Di的低/高32位)
```

**重要**：NEON 和浮点共用同一个物理寄存器文件。切换上下文的代价很小（Linux lazy FP/NEON 保存），但不要因此滥用。

### Lane 概念

Lane 是向量中一个元素的位置。不同数据类型有不同 lane 数：

```
float32x4_t  v = { 1.0f, 2.0f, 3.0f, 4.0f };
                   ^      ^      ^      ^
                 lane0  lane1  lane2  lane3

int8x16_t    v = { 0, 1, 2, 3, ..., 14, 15 };
                   ^                      ^
                 lane0                  lane15
```

lane 索引始终从 0 开始，低位 lane 在内存低地址。

---

## 2. NEON 向量类型

### 整数类型

```c
#include <arm_neon.h>

// 有符号整数
int8x8_t   a8;    // 8-bit × 8  lanes  (64-bit vector)
int8x16_t  a16;   // 8-bit × 16 lanes  (128-bit vector)
int16x4_t  b4;    // 16-bit × 4 lanes  (64-bit)
int16x8_t  b8;    // 16-bit × 8 lanes  (128-bit)
int32x2_t  c2;    // 32-bit × 2 lanes  (64-bit)
int32x4_t  c4;    // 32-bit × 4 lanes  (128-bit)
int64x1_t  d1;    // 64-bit × 1 lane   (64-bit)
int64x2_t  d2;    // 64-bit × 2 lanes  (128-bit)

// 无符号整数（命名规则相同，int 替换为 uint）
uint8x16_t  u8x16;
uint16x8_t  u16x8;
uint32x4_t  u32x4;
uint64x2_t  u64x2;
```

### 浮点类型

```c
// 半精度 (fp16)
float16x4_t  f16x4;   // 4 × fp16 (64-bit)
float16x8_t  f16x8;   // 8 × fp16 (128-bit)

// 单精度 (fp32)
float32x2_t  f32x2;   // 2 × f32 (64-bit)
float32x4_t  f32x4;   // 4 × f32 (128-bit)  ← 最常用

// 双精度 (fp64) ── 仅在 AArch64 下可用
float64x1_t  f64x1;   // 1 × f64 (64-bit)
float64x2_t  f64x2;   // 2 × f64 (128-bit)
```

### 多向量复合类型（用于 interleaved load/store）

```c
// 包含 2/3/4 个相同元素类型的向量
float32x4x2_t  buf2;   // { val[0], val[1] }   2×4个 f32
float32x4x3_t  buf3;   // { val[0], val[1], val[2] }  3×4个 f32
float32x4x4_t  buf4;   // { val[0], val[1], val[2], val[3] }  4×4个 f32
```

---

## 3. 加载与存储

### 连续加载（最常用）

```c
// 从连续内存加载 128-bit 数据
float32x4_t vld1q_f32(const float32_t *ptr);     // 无对齐要求
int32x4_t   vld1q_s32(const int32_t *ptr);
uint8x16_t  vld1q_u8(const uint8_t *ptr);

// 64-bit 加载
float32x2_t vld1_f32(const float32_t *ptr);

// 存储
void vst1q_f32(float32_t *ptr, float32x4_t val);
void vst1q_s32(int32_t *ptr, int32x4_t val);

// 单 lane 存储（写入1个元素，不修改其他内存）
void vst1q_lane_f32(float32_t *ptr, float32x4_t val, const int lane);
```

**关于对齐**：`vld1q` 族指令在现代 Cortex-A 核心（A72, A76, X1）上对未对齐地址几乎无性能损失。但如果一条 load 跨 cache line 边界（64-byte），会产生额外周期。建议 SoA 布局中确保 64 字节对齐。

### 交错加载（Interleaved/Strided Load）

```c
// 从交错排列的内存中加载:
// 内存: [R0 G0 B0 R1 G1 B1 R2 G2 B2 R3 G3 B3 ...]
// 结果分离为平面数组
float32x4x3_t rgb = vld3q_f32(ptr);   // rgb.val[0]=R向量, val[1]=G, val[2]=B

float32x4x2_t vld2q_f32(const float32_t *ptr);   // 2路交错
float32x4x3_t vld3q_f32(const float32_t *ptr);   // 3路交错（RGB）
float32x4x4_t vld4q_f32(const float32_t *ptr);   // 4路交错（RGBA）

// 对应存储
void vst2q_f32(float32_t *ptr, float32x4x2_t val);
void vst3q_f32(float32_t *ptr, float32x4x3_t val);
void vst4q_f32(float32_t *ptr, float32x4x4_t val);
```

**性能警告**：`vld3q_f32` / `vld4q_f32` 在 Cortex-A76 上需要 2 个 load 端口（每条指令发出 3/4 个微操作）。对于大吞吐场景，用 `vld1q` 一次加载整个块再用 `vtrn`/`vzip` 手动解交错往往更快。

### 其他有用加载模式

```c
// 从内存加载并复制到所有 lane
float32x4_t vld1q_dup_f32(const float32_t *ptr);    // 加载1个float复制到4个lane

// 从另一个向量的某个 lane 加载
float32x4_t vld1q_lane_f32(const float32_t *ptr, float32x4_t src, const int lane);

// 重复一个值到所有 lane
float32x4_t vdupq_n_f32(float32_t val);
int32x4_t   vdupq_n_s32(int32_t val);
```

---

## 4. 算术指令

### 基本运算

```c
float32x4_t vaddq_f32(float32x4_t a, float32x4_t b);   // a + b
float32x4_t vsubq_f32(float32x4_t a, float32x4_t b);   // a - b
float32x4_t vmulq_f32(float32x4_t a, float32x4_t b);   // a * b
float32x4_t vnegq_f32(float32x4_t a);                   // -a
float32x4_t vabsq_f32(float32x4_t a);                   // |a|

// 整数对应
int32x4_t vaddq_s32(int32x4_t a, int32x4_t b);
int32x4_t vsubq_s32(int32x4_t a, int32x4_t b);
int32x4_t vmulq_s32(int32x4_t a, int32x4_t b);
```

### 乘法与乘加（FMA）

```c
// vmla: a = a + b * c  (multiply-accumulate, 破坏性写入 a)
float32x4_t vmlaq_f32(float32x4_t a, float32x4_t b, float32x4_t c);
int32x4_t   vmlaq_s32(int32x4_t a, int32x4_t b, int32x4_t c);

// vfma (fused multiply-add, 无中间舍入): a = a + b * c
float32x4_t vfmaq_f32(float32x4_t a, float32x4_t b, float32x4_t c);

// vmls: a = a - b * c  (multiply-subtract)
float32x4_t vmlsq_f32(float32x4_t a, float32x4_t b, float32x4_t c);
```

**FMA 性能至关重要**：在 Cortex-A76 上，`vfmaq_f32` 延迟 4 周期，吞吐量 2 条/周期。这意味着每个周期可以执行 2×4×2 = 16 FLOP（因为 FMA 算 2 次浮点运算）。GEMM 优化中，FMA 是核心指令。

### 除法与倒数近似

```c
// 精确除法 ── 非常昂贵！
float32x4_t vdivq_f32(float32x4_t a, float32x4_t b);  // 延迟 ~9 cycles @ A76

// 快速倒数近似 (1/x)
float32x4_t vrecpeq_f32(float32x4_t a);   // 初始近似，相对误差 ~2^-8
// 一次 Newton-Raphson 迭代提高精度:
//   x_new = x * (2 - a * x)
// 两次迭代得到近乎完全的精度

// 快速平方根倒数近似 (1/sqrt(x))
float32x4_t vrsqrteq_f32(float32x4_t a);  // 初始近似
// Newton-Raphson: y = y * (3 - a * y * y) / 2

// 平方根
float32x4_t vsqrtq_f32(float32x4_t a);
```

**NEON 除法优化实例**：

```c
// 用倒数近似实现快速除法，适合精度容忍大的场景
inline float32x4_t fast_divq_f32(float32x4_t a, float32x4_t b) {
    float32x4_t recip = vrecpeq_f32(b);
    // 2次 Newton-Raphson 迭代
    float32x4_t step1 = vrecpsq_f32(b, recip);   // 2 - b * recip (只需一次)
    recip = vmulq_f32(recip, step1);
    float32x4_t step2 = vrecpsq_f32(b, recip);
    recip = vmulq_f32(recip, step2);
    return vmulq_f32(a, recip);
}
```

---

## 5. 归约操作

```c
// 跨 lane 求和
float32_t vaddvq_f32(float32x4_t a);   // sum(a[0..3]) 返回标量
int32_t   vaddvq_s32(int32x4_t a);

// 获取最大值/最小值
float32_t vmaxvq_f32(float32x4_t a);
float32_t vminvq_f32(float32x4_t a);

// 扩大归约（64-bit 结果累加器，避免溢出）
int64_t vaddlvq_s32(int32x4_t a);   // sum 用 int64 累加
// vaddlvq_s32 内部拆成 int64×2，返回2个值求和

// 配对加法
float32_t vpadds_f32(float32x2_t a);   // a[0]+a[1]
float32x2_t vpadd_f32(float32x2_t a, float32x2_t b);  // {a[0]+a[1], b[0]+b[1]}
```

**关键陷阱**：`vaddvq_f32` 每次调用至少 3 个周期延迟（A76），且只有一个执行单元。在循环内每次迭代都做归约会严重拖慢性能。应当累积多个结果后再归约。

---

## 6. 比较运算

```c
// 比较结果为 lane-wise 全 1（真）或全 0（假）

float32x4_t vceqq_f32(float32x4_t a, float32x4_t b);   // a == b
float32x4_t vcgtq_f32(float32x4_t a, float32x4_t b);   // a > b
float32x4_t vcgeq_f32(float32x4_t a, float32x4_t b);   // a >= b
float32x4_t vcltq_f32(float32x4_t a, float32x4_t b);   // a < b (实际上是vcgt的交换参数)
float32x4_t vcleq_f32(float32x4_t a, float32x4_t b);   // a <= b

// 整数对应（与浮点相同的语义）
uint32x4_t vceqq_s32(int32x4_t a, int32x4_t b);   // 返回 uint32x4_t
uint32x4_t vcgtq_s32(int32x4_t a, int32x4_t b);
```

**重要**：NEON 没有直接的浮点类型比较返回 bool 类型。比较结果是一个向量，每个 lane 要么是全 1 (`0xFFFFFFFF`) 要么是全 0 (`0x00000000`)。这个向量可以直接用作位选择的掩码。

---

## 7. 掩码和位选择

```c
// 位选择 (bitwise select): result = (mask & a) | (~mask & b)
float32x4_t vbslq_f32(uint32x4_t mask, float32x4_t a, float32x4_t b);

// 按位逻辑运算（处理掩码）
uint32x4_t vandq_u32(uint32x4_t a, uint32x4_t b);
uint32x4_t vorrq_u32(uint32x4_t a, uint32x4_t b);
uint32x4_t veorq_u32(uint32x4_t a, uint32x4_t b);
uint32x4_t vmvnq_u32(uint32x4_t a);   // NOT

// 组合掩码
uint32x4_t vandq_u32(mask1, mask2);   // AND 两个条件
uint32x4_t vorrq_u32(mask1, mask2);   // OR 两个条件
```

**实际使用模式**：

```c
// 条件赋值: 将 x 中小于0的元素替换为0 (ReLU)
float32x4_t relu(float32x4_t x) {
    float32x4_t zero = vdupq_n_f32(0.0f);
    uint32x4_t  mask = vcgeq_f32(x, zero);   // x >= 0 ?
    return vbslq_f32(mask, x, zero);          // 非负则保持, 负则置0
}

// NaN 替换
float32x4_t replace_nan(float32x4_t x, float32x4_t replacement) {
    uint32x4_t nan_mask = vceqq_f32(x, x);    // NaN != NaN, 所以 if NaN: false
    return vbslq_f32(nan_mask, x, replacement);
}
```

---

## 8. 类型转换

```c
// float ↔ int 转换
int32x4_t   vcvtq_s32_f32(float32x4_t a);     // f32 → i32 (向零舍入)
float32x4_t vcvtq_f32_s32(int32x4_t a);       // i32 → f32

// 带舍入模式的转换
int32x4_t vcvtnq_s32_f32(float32x4_t a);      // 向最近偶数舍入
int32x4_t vcvtpq_s32_f32(float32x4_t a);      // 向正无穷舍入（ceil）
int32x4_t vcvtmq_s32_f32(float32x4_t a);      // 向负无穷舍入（floor）

// fp16 ↔ fp32 转换（ARMv8.2+ 支持）
float16x4_t vcvt_f16_f32(float32x4_t a);       // f32×4 → f16×4 (截断高64位)
float32x4_t vcvt_f32_f16(float16x4_t a);       // f16×4 → f32×4

// vmovn: 变窄 (128-bit → 64-bit, 取低半部分)
int16x4_t  vmovn_s32(int32x4_t a);            // i32×4 → i16×4
uint8x8_t  vmovn_u16(uint16x8_t a);           // u16×8 → u8×8

// vqmovn: 饱和变窄
uint8x8_t  vqmovn_u16(uint16x8_t a);          // 超出255的饱和为255
// vqmovun: 有符号饱和变窄为无符号
uint8x8_t  vqmovun_s16(int16x8_t a);          // 负值→0, 超出255→255

// vmovl: 变宽 (64-bit → 128-bit)
int32x4_t  vmovl_s16(int16x4_t a);            // i16×4 → i32×4
uint32x4_t vmovl_u16(uint16x4_t a);
```

**类型双关（reinterpret）**：同一寄存器内容的重新解释，无指令开销

```c
float32x4_t vreinterpretq_f32_s32(int32x4_t a);    // int32 bit pattern → float32
int32x4_t   vreinterpretq_s32_f32(float32x4_t a);  // float32 bit pattern → int32
uint32x4_t  vreinterpretq_u32_f32(float32x4_t a);  // float32 bit pattern → uint32
```

这在需要同时做整数位操作和浮点算术时非常有用。

---

## 9. NEON Intrinsics 命名规范

```
v    op    [q]  _  type
│     │     │      │
│     │     │      └── Lane 元素类型: f32, s32, u8, f16...
│     │     └── 可选 q = 128-bit; 无q = 64-bit
│     └── 操作名: add, sub, mul, mla, ld1, st1, cvt...
└── vector 前缀
```

额外后缀：
- `_n`: 第二个操作数是标量  `vaddq_n_f32` 加一个标量到每个lane
- `_lane`: 从另一向量的特定lane获取标量  `vmulq_lane_f32`
- `_laneq`: 从128-bit向量的特定lane（适用范围不同与_lane）
- `_high`/`_low`: 取128-bit的高64 / 低64

```c
// 示例
float32x4_t vaddq_n_f32(float32x4_t a, float32_t b);       // a + b (broadcast)
float32x4_t vmulq_lane_f32(float32x4_t a, float32x2_t b,   // a * b[lane]
                           const int lane);
```

---

## 10. 常见陷阱与最佳实践

### 陷阱1：循环内做归约

```c
// 不好的写法：每4个元素做一次 vaddv
for (int i = 0; i < n; i += 4) {
    float32x4_t v = vld1q_f32(&data[i]);
    float sum = vaddvq_f32(v);   // 3+ cycle 延迟, 每次循环
    total += sum;
}

// 更好的写法：向量累积，最后归约
float32x4_t acc = vdupq_n_f32(0.0f);
for (int i = 0; i < n; i += 4) {
    float32x4_t v = vld1q_f32(&data[i]);
    acc = vaddq_f32(acc, v);
}
float total = vaddvq_f32(acc);   // 只在最后做一次归约
```

### 陷阱2：盲目使用 vld3/vld4

```c
// RGB交错数据: [R0 G0 B0 R1 G1 B1 ...]
// vld3q_f32 看起来很直接:
float32x4x3_t rgb = vld3q_f32(ptr);  // 加载并分离

// 但对大吞吐场景，手动 vld1q + 解交错有时更快:
float32x4_t chunk0 = vld1q_f32(ptr + 0);
float32x4_t chunk1 = vld1q_f32(ptr + 4);
float32x4_t chunk2 = vld1q_f32(ptr + 8);
// 然后用 vtrn/vzip 手动分离...
```

**规则**：在基准测试中测量。不要假设 `vld3` 比手动加琐碎操作快。

### 陷阱3：NEON 除法不是向量化的

```c
// 不要写:
float32x4_t result = vdivq_f32(a, b);   // 内部是标量除法循环

// 在性能和精度允许时用:
float32x4_t result = fast_divq_f32(a, b);  // 倒数近似
```

`vdivq_f32` 在硬件层面并不是真正的全向量除法单元，而是通过多次迭代实现的。对于大吞吐量应用，使用 `vrecpeq + Newton-Raphson` 是最佳替代。

### 陷阱4：NEON/浮点上下文切换

在 AArch64 上，NEON 和浮点状态在上下文切换时是延迟保存/恢复的（lazy save/restore）。但如果频繁在浮点代码和系统调用之间切换，可能引发额外的保存/恢复开销。

```c
// 避免在热循环内调用可能触发上下文切换的函数
// 如果必须，考虑使用更多的 NEON 寄存器以减少溢出
```

### 陷阱5：忽略数据对齐

```c
// 尽管 vld1q 在未对齐地址上工作，但：
// 1. 跨 cache line 边界仍有 1-3 周期额外开销
// 2. 跨 4KB 页边界可能触发额外的 TLB miss

// 推荐做法:
float data[1024] __attribute__((aligned(64)));  // 64-byte 对齐
// 或 C11:
alignas(64) float data[1024];
```

### 陷阱6：混合浮点和整数 NEON

```c
// Cortex-A76 有独立的浮点和整数 NEON 流水线
// 在浮点循环中插入整数操作可能导致端口竞争

// 如果可能，将浮点和整数 NEON 操作分在不同的循环中执行
// 或者确保它们的比例匹配硬件端口
```

---

## 11. 一个完整示例：向量加法

```c
#include <arm_neon.h>
#include <stdio.h>

// NEON 向量加法: C = A + B
void vector_add_neon(const float* a, const float* b, float* c, int n) {
    int i;
    // 主循环：每次处理 16 个 float（4 个 Q 寄存器）
    for (i = 0; i <= n - 16; i += 16) {
        float32x4_t a0 = vld1q_f32(&a[i + 0]);
        float32x4_t a1 = vld1q_f32(&a[i + 4]);
        float32x4_t a2 = vld1q_f32(&a[i + 8]);
        float32x4_t a3 = vld1q_f32(&a[i + 12]);

        float32x4_t b0 = vld1q_f32(&b[i + 0]);
        float32x4_t b1 = vld1q_f32(&b[i + 4]);
        float32x4_t b2 = vld1q_f32(&b[i + 8]);
        float32x4_t b3 = vld1q_f32(&b[i + 12]);

        float32x4_t c0 = vaddq_f32(a0, b0);
        float32x4_t c1 = vaddq_f32(a1, b1);
        float32x4_t c2 = vaddq_f32(a2, b2);
        float32x4_t c3 = vaddq_f32(a3, b3);

        vst1q_f32(&c[i + 0], c0);
        vst1q_f32(&c[i + 4], c1);
        vst1q_f32(&c[i + 8], c2);
        vst1q_f32(&c[i + 12], c3);
    }
    // 标量尾部（处理不足16个的元素）
    for (; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// 编译: gcc -O3 -march=armv8-a+simd -o add add.c
```

这个例子展示了：
1. 循环展开（4x 展开，处理 16 个元素/迭代），增加指令级并行
2. 标量尾部循环处理不足完整向量的元素
3. 使用 `vld1q` 加载和 `vst1q` 存储

现在你有了编写 NEON 代码的全部基础知识。下一节将深入内存布局话题，这是决定 SIMD 性能上限的关键因素。
