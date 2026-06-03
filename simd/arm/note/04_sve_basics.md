# SVE 基础：向量长度无关的编程范式

## 0. 为什么需要 SVE

NEON 有一个根本限制：**向量宽度是固定的 128-bit**。这意味着：

1. **代码不可移植**：为 Cortex-A76（128-bit NEON）写的代码在 Graviton3（256-bit SVE）上只能利用一半宽度
2. **必须维护多个版本**：不同向量宽度需要不同的展开因子和尾部循环
3. **尾部循环是 bug 温床**：标量尾部处理是 SIMD 代码最常见的事故源

SVE 通过**向量长度无关**（Vector-Length Agnostic, VLA）解决所有三个问题。

```
NEON 方式（固定 128-bit）:
  for (i = 0; i < n - 4; i += 4) { ... }  // 主循环: 4 floats
  for (; i < n; i++) { ... }              // 尾部: scalar

SVE 方式（VLA）:
  while (svwhilelt_b32(i, n)) {            // 主循环 + 尾部 = 一个循环!
      自动处理任意向量长度
  }
```

---

## 1. SVE 寄存器

### 向量寄存器

```
Z0-Z31: 可伸缩向量寄存器（32 个）
  宽度: VL bits (128, 256, 512, 1024, 2048...)
  
  Z0 的 128-bit 低部分 = V0 (NEON 的 Q0 寄存器)
  Z0 的 64-bit  低部分 = D0 (NEON 的 D0 寄存器)

  这意味着 SVE 和 NEON 共用寄存器文件
  你可以在一个函数中混用 NEON 和 SVE（但不推荐）
```

### 谓词寄存器

```
P0-P15: 谓词寄存器（16 个）
  宽度: VL/8 bits (每个 lane 1 bit)
  
  功能: 控制每个 lane 的启用/禁用
  - P0: 通常用于 ALU 操作（ptrue）
  - P1-P3: 用于数据操作
  - P4-P7: 通用谓词
  - P8-P15: 用于循环控制
```

### 与 NEON 的寄存器映射

```
SVE                 NEON
───────────────────────────
Z0  [VL-1:0]  ←→   V0  [127:0]  (低128位)
Z1  [VL-1:0]  ←→   V1  [127:0]
...
P0  [VL/8-1:0]       无对应

注意: Z0 的高位（VL-1 到 128）在 NEON 中没有对应
在混用 NEON 和 SVE 时必须谨慎
```

---

## 2. 谓词的概念

### 什么是谓词

谓词是一个位掩码，每个 bit 控制向量中对应 lane 的操作：

```
Z0 (256-bit, VL=256, float32x?):
  [ lane7 | lane6 | lane5 | lane4 | lane3 | lane2 | lane1 | lane0 ]

P0 (256/8 = 32-bit 谓词, 每个 float lane 用 1 bit):
  [   0   |   1   |   1   |   0   |   1   |   1   |   1   |   1   ]
      ↑       ↑       ↑       ↑       ↑       ↑       ↑       ↑
     禁用    启用    启用    禁用    启用    启用    启用    启用
```

### 谓词创建

```c
#include <arm_sve.h>

// 方式 1: 循环计数器模式（最常用）
svbool_t pg = svwhilelt_b32(uint64_t i, uint64_t n);
// 含义: 对 lane j, 如果 (i + j) < n 则启用, 否则禁用
// 这是构建无尾部循环的关键

// 方式 2: 全真谓词
svbool_t pg_all = svptrue_b32();   // 所有 lane 启用
// 等价于:
svbool_t pg_all = svptrue_pat_b32(SV_ALL);

// 方式 3: 模式谓词
svbool_t pg_vl1  = svptrue_pat_b32(SV_VL1);  // 只有 lane0 启用
svbool_t pg_vl2  = svptrue_pat_b32(SV_VL2);  // lane0-1 启用
svbool_t pg_vl3  = svptrue_pat_b32(SV_VL3);  // lane0-2 启用
svbool_t pg_vl4  = svptrue_pat_b32(SV_VL4);  // lane0-3 启用
svbool_t pg_vl5  = svptrue_pat_b32(SV_VL5);
svbool_t pg_vl6  = svptrue_pat_b32(SV_VL6);
svbool_t pg_vl7  = svptrue_pat_b32(SV_VL7);
svbool_t pg_vl8  = svptrue_pat_b32(SV_VL8);  // lane0-7 启用
svbool_t pg_vl16 = svptrue_pat_b32(SV_VL16);
svbool_t pg_odd  = svptrue_pat_b32(SV_VL256);  // 等等

// 方式 4: 比较谓词（从数据生成）
svbool_t pg_gt = svcmpgt_f32(pg_all, a, b);   // a > b 的 lane 为 true
svbool_t pg_eq = svcmpeq_f32(pg_all, a, b);   // a == b 的 lane 为 true
```

### 谓词逻辑运算

```c
// SVE2 提供位级谓词运算
svbool_t pg_and = svand_b_z(pg1, pg2, pg3);    // pg1 & pg2
svbool_t pg_or  = svorr_b_z(pg1, pg2, pg3);    // pg1 | pg2
svbool_t pg_not = svnot_b_z(pg1, pg2);          // ~pg2
```

---

## 3. 向量类型

```c
// SVE 向量类型（VLA，宽度在编译时未知）
svfloat32_t   f32_vec;   // VL/32 个 float
svfloat64_t   f64_vec;   // VL/64 个 double
svint32_t     i32_vec;   // VL/32 个 int32
svuint32_t    u32_vec;   // VL/32 个 uint32
svint16_t     i16_vec;   // VL/16 个 int16
svint8_t      i8_vec;    // VL/8 个 int8
svuint8_t     u8_vec;

// 谓词类型
svbool_t  pg;            // VL/8 位的谓词

// 获取向量长度（运行时值，不是常量）
uint64_t vl = svcntb();  // 向量中的字节数 (VL/8)
uint64_t vl = svcnth();  // 向量中的半字数 (VL/16)
uint64_t vl = svcntw();  // 向量中的字数 (VL/32)
uint64_t vl = svcntd();  // 向量中的双字数 (VL/64)
```

---

## 4. 谓词控制的加载/存储

这是 SVE 最强大的特性。谓词控制的 load/store **消除了对标量尾部循环的需求**。

### 基本谓词 Load/Store

```c
// 谓词 load: 只加载 pg 掩码启用的 lane
svfloat32_t svld1_f32(svbool_t pg, const float32_t *base);
svint32_t   svld1_s32(svbool_t pg, const int32_t *base);

// 谓词 store: 只存储 pg 掩码启用的 lane（不会越界写入！）
void svst1_f32(svbool_t pg, float32_t *base, svfloat32_t data);
void svst1_s32(svbool_t pg, int32_t *base, svint32_t data);

// 获取第一个 faulting 地址（用于内存故障处理）
// svldff1_f32: first-faulting load，在越界时不触发段错误
// 适用于处理未被完全映射的页面的尾部
```

### 无尾部循环的向量加法

```c
// SVE 版向量加法：无标量尾部，自然适应任意 n 和任意 VL
void vector_add_sve(const float* a, const float* b, float* c, uint64_t n) {
    uint64_t i = 0;

    // 主循环：自动处理主处理和尾部
    while (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);   // 哪些 lane 有效？
        svfloat32_t va = svld1_f32(pg, &a[i]);  // 只加载有效 lane
        svfloat32_t vb = svld1_f32(pg, &b[i]);
        svfloat32_t vc = svadd_f32_m(pg, va, vb);  // additive merge
        svst1_f32(pg, &c[i], vc);                 // 只存储有效 lane

        i += svcntw();   // i += VL/32（当前硬件上的实际 lane 数）
    }
}
```

**和 NEON 对比**：
```c
// NEON: 需要主循环 + 尾部循环
for (i = 0; i <= n - 4; i += 4) { ... }  // 主循环
for (; i < n; i++) c[i] = a[i] + b[i];   // 尾部

// SVE: 一个循环处理一切
// 不仅代码更简单，而且避免了尾部处理的标量开销
```

---

## 5. 算术指令的三种变体

SVE 谓词操作的三种后缀：

```c
// _m (merge): 对禁用 lane，保留第一个操作数的值
svfloat32_t svadd_f32_m(svbool_t pg, svfloat32_t op1, svfloat32_t op2);
// pg=1 的 lane: result = op1 + op2
// pg=0 的 lane: result = op1       ← 保留第一个操作数

// _z (zero): 对禁用 lane，结果为零
svfloat32_t svadd_f32_z(svbool_t pg, svfloat32_t op1, svfloat32_t op2);
// pg=1 的 lane: result = op1 + op2
// pg=0 的 lane: result = 0         ← 置零

// _x (don't care): 对禁用 lane，结果不确定（通常最快）
svfloat32_t svadd_f32_x(svbool_t pg, svfloat32_t op1, svfloat32_t op2);
// pg=1 的 lane: result = op1 + op2
// pg=0 的 lane: result = undefined  ← 可能保留旧值，可能为任意值
```

### 使用场景

```c
// _m: 累加模式（保持累加器值不变）
svfloat32_t acc = svdup_f32(0);
for (...) {
    svbool_t pg = svwhilelt_b32(i, N);
    svfloat32_t data = svld1_f32(pg, ptr + i);
    acc = svadd_f32_m(pg, acc, data);  // 只更新有效 lane 的累加
    i += svcntw();
}

// _z: 初始化或清零模式
svfloat32_t masked_result = svadd_f32_z(pg, a, b);
// 不活跃 lane 的结果为零，可以直接存储

// _x: 速度优先模式（不需要关心不活跃 lane 的值）
svfloat32_t tmp = svadd_f32_x(pg, a, b);
// 随后会再次被谓词覆盖时使用
```

---

## 6. FMA（融合乘加）

```c
// SVE FMA: result = acc + a × b
svfloat32_t svmla_f32_m(svbool_t pg, svfloat32_t acc,
                         svfloat32_t a, svfloat32_t b);
svfloat32_t svmla_f32_z(svbool_t pg, svfloat32_t acc,
                         svfloat32_t a, svfloat32_t b);

// 点积加速（SVE，类似 NEON 的 vdotq_s32）
svint32_t svdot_s32(svint32_t acc, svint8_t a, svint8_t b);
// int8 × int8 → int32 累加
```

---

## 7. 归约

```c
// 求和
svfloat32_t svaddv_f32(svbool_t pg, svfloat32_t op);
// 返回标量: 所有活动 lane 的和

// 最大值/最小值
svfloat32_t svmaxv_f32(svbool_t pg, svfloat32_t op);
svfloat32_t svminv_f32(svbool_t pg, svfloat32_t op);

// 跨 lane 求和实例
float sum_sve(const float* data, uint64_t n) {
    svfloat32_t acc = svdup_f32(0.0f);
    uint64_t i = 0;

    while (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t vec = svld1_f32(pg, &data[i]);
        // 用 _m 模式累加，保护不活跃 lane 的累加器值
        acc = svadd_f32_m(pg, acc, vec);
        i += svcntw();
    }

    // 归约：对所有 lane 求和（谓词控制的归约）
    svbool_t pg_all = svptrue_b32();
    return svaddv_f32(pg_all, acc);
}
```

---

## 8. 比较和选择

```c
// 比较
svbool_t svcmpgt_f32(svbool_t pg, svfloat32_t a, svfloat32_t b);  // a > b
svbool_t svcmpge_f32(svbool_t pg, svfloat32_t a, svfloat32_t b);  // a >= b
svbool_t svcmpeq_f32(svbool_t pg, svfloat32_t a, svfloat32_t b);  // a == b
svbool_t svcmplt_f32(svbool_t pg, svfloat32_t a, svfloat32_t b);  // a < b
svbool_t svcmple_f32(svbool_t pg, svfloat32_t a, svfloat32_t b);  // a <= b
svbool_t svcmpne_f32(svbool_t pg, svfloat32_t a, svfloat32_t b);  // a != b

// 条件选择 → 类似于 NEON 的位选择
svfloat32_t svsel_f32(svbool_t pg, svfloat32_t a, svfloat32_t b);
// pg=1: result = a,  pg=0: result = b

// 实现 ReLU: max(0, x)
svfloat32_t relu_sve(svfloat32_t x, svbool_t pg) {
    const svfloat32_t zero = svdup_f32(0.0f);
    svbool_t gt_mask = svcmpge_f32(pg, x, zero);
    return svsel_f32(gt_mask, x, zero);
}
```

---

## 9. 循环控制模式

### 经典模式：svwhilelt 升序循环

```c
// 处理 [0, n) 中的元素
void process_sve(const float* src, float* dst, uint64_t n) {
    uint64_t i = 0;
    while (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t v = svld1_f32(pg, &src[i]);

        // ... 处理 ...

        svst1_f32(pg, &dst[i], v);
        i += svcntw();  // 按实际 lane 数推进
    }
}
```

### 倒序循环（某些场景更优）

```c
void process_sve_countdown(const float* src, float* dst, int64_t n) {
    int64_t i = n;
    do {
        svbool_t pg = svwhilelt_b32(0, i);    // while 0 < i
        i -= svcntw();                          // 递减（带符号）

        svfloat32_t v = svld1_f32(pg, &src[i]);
        // ... 处理 ...
        svst1_f32(pg, &dst[i], v);
    } while (i > 0);
}
```

倒序循环的优势：
1. `svwhilelt_b32(0, i)` 的谓词生成和指针偏移更简单
2. 递减到零的判断指令比递增比较更快（`subs` + `b.ne` vs `cmp` + `b.lt`）
3. 适合编译器做更激进的优化

---

## 10. SVE2 附加指令

SVE2（ARMv9）提供的额外能力：

```c
// 复数乘法（信号处理核心）
// 将复数数组分为实部和虚部做乘法

// 加宽的整数乘加
svint32_t svmlalb_s32(svint32_t acc, svint16_t a, svint16_t b);
// int16×int16 → int32 累加

// 位操作谓词
svbool_t svand_b_z(svbool_t pg, svbool_t pn, svbool_t pm);   // pn & pm
svbool_t svorr_b_z(svbool_t pg, svbool_t pn, svbool_t pm);   // pn | pm

// 增强的 scatter/gather
svint32_t svld1_gather_s32index_s32(svbool_t pg, const int32_t* base,
                                     svint32_t indices);
// 按 int32 索引从 base 采集数据
```

---

## 11. SVE 与 NEON 共存

### 运行时检测

```c
#include <sys/auxv.h>
#include <asm/hwcap.h>

enum SIMDLevel {
    SIMD_NONE      = 0,
    SIMD_NEON      = 1,
    SIMD_NEON_DOT  = 2,   // ARMv8.2 int8 dot product
    SIMD_SVE       = 3,   // ARMv8.2+
    SIMD_SVE2      = 4    // ARMv9
};

SIMDLevel detect_simd_level() {
    unsigned long hwcap  = getauxval(AT_HWCAP);
    unsigned long hwcap2 = getauxval(AT_HWCAP2);

    if (hwcap2 & HWCAP2_SVE2)    return SIMD_SVE2;
    if (hwcap2 & HWCAP2_SVE)     return SIMD_SVE;
    if (hwcap  & HWCAP_ASIMDDP)  return SIMD_NEON_DOT;
    if (hwcap  & HWCAP_ASIMD)    return SIMD_NEON;
    return SIMD_NONE;
}
```

### 函数多版本（Function Multi-Versioning）

```c
// 根据 CPU 特性选择不同实现
// GCC 的 target_clones 属性会自动生成多版本
// 但这只在目标架构正确时才有效

#if defined(__ARM_FEATURE_SVE)
// SVE 版本
void my_kernel_sve(const float* A, const float* B, float* C, int N) {
    // ... SVE 实现 ...
}
#else
// NEON 回退版本
void my_kernel_neon(const float* A, const float* B, float* C, int N) {
    // ... NEON 实现 ...
}
#endif

// 运行时调度
typedef void (*kernel_fn)(const float*, const float*, float*, int);

kernel_fn select_kernel() {
    unsigned long hwcap2 = getauxval(AT_HWCAP2);
    if (hwcap2 & HWCAP2_SVE) {
        return my_kernel_sve;
    }
    return my_kernel_neon;
}
```

### 混用 NEON 和 SVE 的注意事项

```c
// ⚠ SVE 和 NEON 共用寄存器文件
// Z0 的低 128 位 = V0
//
// 如果在一个函数中混用，需要遵守 ABI 规则：
//   1. NEON 代码只修改 V0-V7（调用者保存）
//   2. SVE 代码可能修改 Z0-Z31
//   3. 从 SVE 切换到 NEON 前，保存所有 SVE 状态
//
// 推荐：一个函数只用一种 SIMD，避免混用
```

---

## 12. 编译器标志和限制

```bash
# SVE 编译
gcc -march=armv8.2-a+sve   -O3 -o prog prog.c
gcc -march=armv8.2-a+sve   -O3 -mbig-endian ...  # 大端模式
gcc -march=armv9-a         -O3 ...               # SVE2

# 在运行时指定 SVE 向量长度（用于 QEMU 模拟或模拟不同 VL）
# 环境变量:
export SVE_VECTOR_LENGTH=256   # 模拟 256-bit VL
# 或
/sys/devices/system/cpu/sve/vl  # 查看硬件 VL

# QEMU 模拟
qemu-aarch64 -cpu max,sve=on,sve256=on ./prog
```

---

## 13. SVE 在实际中的局限性

### 为什么 SVE 还不是主流

1. **硬件部署有限**
   - 目前只有 Apple M4、Fujitsu A64FX、AWS Graviton3、Neoverse V1 支持 SVE
   - 绝大多数手机（A76, A78, X1）仍是 NEON only

2. **生态不成熟**
   - 开源库大多只有 NEON 优化路径
   - SVE 的调试和性能分析工具有限

3. **编译器优化**
   - GCC 和 Clang 的 SVE 自动向量化逐渐改善但仍不如 NEON 成熟
   - 手写 SVE intrinsics 需要比较深入的理解

4. **无法确定最优展开因子**
   - NEON 可以硬编码展开 4x（因为 128-bit 固定）
   - SVE 在 VL=256 与 VL=512 上最优展开不同
   - 需要在不同 VL 上基准测试

### 何时用 SVE

- **云端 ARM 服务器**：Graviton3 的 256-bit SVE 比 NEON 高 2x 吞吐
- **HPC**：Fujitsu A64FX 的 512-bit SVE 提供巨大向量宽度
- **需要可移植二进制**：同一份二进制在 128-2048 bit 上都最优

### 何时坚持用 NEON

- **移动端应用**：几乎所有手机都是 NEON only
- **需要确定性性能**：128-bit 固定宽度，展开深度已知
- **生态更成熟**：更多示例、库、调试工具

---

## 14. 完整示例：SVE 矩阵乘法微核心

```c
// SVE GEMM 微核心 (256-bit VL 示例)
// C += A × B, 其中 C 是 4×4, A 是 4×K, B 是 K×4
void sve_gemm_4x4_vl256(const float* A, const float* B, float* C,
                         int K, int lda, int ldb, int ldc) {
    // 为 4 行 C 分配累加器（行优先，需要 4 个 svfloat32_t）
    svfloat32_t c0 = svdup_f32(0.0f);
    svfloat32_t c1 = svdup_f32(0.0f);
    svfloat32_t c2 = svdup_f32(0.0f);
    svfloat32_t c3 = svdup_f32(0.0f);

    int k = 0;
    svbool_t pg = svptrue_b32();  // 全量谓词

    while (k < K) {
        // 加载 A 的第 k 列（广播到所有 lane）
        svfloat32_t a0 = svld1rq_f32(pg, &A[0 * lda + k]);  // 注意: 需要broadcast
        svfloat32_t a1 = svld1rq_f32(pg, &A[1 * lda + k]);
        svfloat32_t a2 = svld1rq_f32(pg, &A[2 * lda + k]);
        svfloat32_t a3 = svld1rq_f32(pg, &A[3 * lda + k]);

        // 加载 B 的一行
        svfloat32_t bk = svld1_f32(pg, &B[k * ldb]);

        // FMA
        c0 = svmla_f32_m(pg, c0, a0, bk);
        c1 = svmla_f32_m(pg, c1, a1, bk);
        c2 = svmla_f32_m(pg, c2, a2, bk);
        c3 = svmla_f32_m(pg, c3, a3, bk);

        k++;
    }

    // 存储 C（如果 C 小于 8 个 float，需要用谓词存储）
    svst1_f32(pg, &C[0 * ldc], c0);
    svst1_f32(pg, &C[1 * ldc], c1);
    svst1_f32(pg, &C[2 * ldc], c2);
    svst1_f32(pg, &C[3 * ldc], c3);
}
```

**注意**：SVE 的 GEMM 优化比 NEON 复杂得多，因为：
1. 最优分块大小依赖于 VL
2. 在 VL=128（最小）和 VL=512（较大）上需要不同的展开策略
3. 通常需要一个运行时初始化的分块策略

---

## 15. SVE 学习路线图

```
第 1 步: 理解谓词概念
  - svwhilelt 是理解 SVE 的钥匙
  - 研究无尾部循环的等价物: NEON 主循环 + 标量尾部

第 2 步: 改写简单的 NEON 程序为 SVE
  - 向量加法、SAXPY、点积
  - 观察代码从 "主循环+尾部" 简化为 "一个 while 循环"

第 3 步: 理解 _m/_z/_x 后缀
  - 这对高性能 SVE 代码至关重要
  - 错误地使用 _z 会导致不活跃 lane 的累加器被清零

第 4 步: 跨不同 VL 的基准测试
  - QEMU 支持模拟 VL=128/256/512
  - 验证代码在不同 VL 下的正确性和性能

第 5 步: 深入 SVE2
  - 复数乘法、位谓词操作、更强的 scatter/gather
  - 只在 ARMv9 (Neoverse V2, Cortex-X4) 上可用
```

---

**下一节**：将 NEON 和 SVE 的知识应用到 7 个真实工业场景中：图像/音频处理、ML 推理、数据压缩、网络包处理。
