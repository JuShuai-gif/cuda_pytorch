# 工业级 GEMM 设计笔记

GEMM（General Matrix Multiply）是 HPC 和 ML 推理中最核心的算子。从全连接层到注意力投影，从卷积 im2col 到推荐系统嵌入内积，背后都是矩阵乘法。本文深入拆解如何从标量三重循环一直优化到和 BLIS/oneDNN 对标的工业级实现。

---

## 1. GEMM 在 ML 推理中的核心地位

### 1.1 三大典型场景

| 场景 | 矩阵形状 | 调用频率 | 说明 |
|------|----------|----------|------|
| **全连接层** | `C[M×N] = A[M×K] × B[K×N]` | 每层一次 | MLP block、FFN 中的两层线性变换 |
| **注意力投影** | QKV 三组 `[B×S×D] × [D×D]` | 每层三次 | Self-Attention 的 Q、K、V 投影 |
| **卷积 im2col** | 将 `[H×W×C]` 展开成 GEMM | 每卷积层一次 | `im2col` 将卷积变成矩阵乘法 |

这三个场景覆盖了 Transformer 和 CNN 推理中 90% 以上的浮点运算量。以 BERT-Large 为例，单次前向传播：

```
总 FLOPs ≈ 70G
其中 GEMM ≈ 62G（~88%）
注意力 softmax + LayerNorm 等 ≈ 8G（~12%）
```

### 1.2 为什么 GEMM 值得极致优化

- **计算密度高**：单次 `C[i][j] += A[i][k] * B[k][j]` 需要 K 次乘加，数据复用量大
- **缓存友好**：通过分块可以让 O(MNK) 内存流量降为 O(MK + KN + MN)
- **硬件亲和**：完美匹配 SIMD FMA 指令和寄存器阻塞模式

---

## 2. 缓存层次分块策略：MC×KC×NC 三级分块

### 2.1 分块的动机

标量 GEMM 对每个 `(i,j)` 输出元素，遍历整个 K 维度：

```
for i in 0..M:
  for j in 0..N:
    for k in 0..K:
      C[i][j] += A[i][k] * B[k][j]
```

问题：A 和 B 被反复加载。每个 A 元素被重复使用 N 次，每个 B 元素被重复使用 M 次，但如果数据溢出缓存，硬件预取也无法补救。

### 2.2 BLIS 风格的五层循环

工业级 GEMM 将原始三重循环展开为 **五层循环**：

```
for ic in 0..M step MC:         // L3 缓存分块（B 面板驻留 L3）
  for pc in 0..K step KC:       // L2 缓存分块（A 面板驻留 L2）
    pack B panel: B[pc:pc+KC][:N] → B_packed  // 列优先打包
    for jc in 0..N step NC:     // L1 缓存分块
      pack A panel: A[ic:ic+MC][pc:pc+KC] → A_packed  // 行优先打包
      for jr in 0..NC step NR:  // 列微瓦片循环（寄存器阻塞）
        for ir in 0..MC step MR:// 行微瓦片循环
          micro_kernel: C[ir:ir+MR][jr:jr+NR] +=
            A_packed[ir:ir+MR][0:KC] × B_packed[0:KC][jr:jr+NR]
```

### 2.3 三级分块原理

```
                    +-----------+-----------+
                    |           |           |
                    |   B面板   |           |
                    |   KC×N    |           |
                    |   (L3)    |           |
                    |           |           |
                    +-----------+-----------+
                    |           |
                    |  A面板    |  C瓦片
                    |  MC×KC   |  MC×NC
                    |  (L2)    |  (寄存器)
                    |           |
                    +-----------+
```

| 分块 | 大小 | 目标缓存 | 复用模式 |
|------|------|----------|----------|
| **MC** | ~256-2048 | L2 缓存 (256 KB-1 MB) | A 面板在 L2 中被 KC 循环复用 |
| **KC** | ~256-1024 | L2/L1 缓存 | B 面板在 L2 中被 NC 循环复用；A 面板在 L1 中复用 |
| **NC** | ~256-4096 | L3 缓存 (1-2 MB) | B 面板驻留 L3，避免反复从主存加载 |
| **MR** | 6-8 (AVX2) | ymm 寄存器 | 将 C 的行保存在寄存器中，消除 C 的重复加载/存储 |
| **NR** | 8 (AVX2) | ymm 寄存器 | 一行 8 个 f32 刚好占满一个 ymm（256-bit） |

### 2.4 典型分块参数

| 平台 | MC | KC | NC | MR | NR | 说明 |
|------|-----|-----|------|-----|-----|------|
| Haswell AVX2 | 256 | 256 | 4096 | 6 | 8 | BLIS 默认配置，MR=6 留寄存器余量 |
| Skylake AVX-512 | 256 | 256 | 4096 | 16 | 16 | zmm 寄存器 16 f32/lane |
| ARM A76 NEON | 128 | 256 | 2048 | 8 | 12 | NEON 128-bit，MR/NR 用不同因子 |

---

## 3. 打包策略

### 3.1 为什么需要打包

原始矩阵乘法中访问模式不友好：

- **A 矩阵**（M×K，行主序）：访问 `A[i][k]` —— 对于固定的 i，沿 k 是连续的 → **天然友好**
- **B 矩阵**（K×N，行主序）：访问 `B[k][j]` —— 对于固定的 j，沿 k 步长为 N，对 SIMD 是非连续 gather → **必须打包**

打包的核心思想：**将需要用 SIMD 宽度一次性加载的数据在内存中排成连续的一块**。

### 3.2 B 矩阵打包：列优先（K×NR 面板）

原始 B 矩阵布局（行主序）：

```
B = [b00 b01 b02 ... b0N-1]    ← row 0 (stride=N)
    [b10 b11 b12 ... b1N-1]    ← row 1
    ...
    [bK-1,0  ...  bK-1,N-1]   ← row K-1
```

要加载 8 个 B 值（同一行 k，相邻 8 列），需要 `_mm256_set_ps()` 做 8 次标量 gather。这会在端口 5 上产生严重瓶颈。

打包后的 B 面板（K×NR）：

```
B_packed[k * NR + j] = B[k][col_start + j]
```

内部循环中只需一条 `_mm256_loadu_ps(&B_packed[k * NR])` 即可加载 8 个连续的 B 值。

**对应代码**：`avx2_gemm_micro.cpp` 中的 `pack_B_Kx8()` 函数：

```c
// simd/x86/src/avx2_gemm_micro.cpp:187-196
static void pack_B_Kx8(float* B_packed, const float* B, int ldb,
                       int row_start, int col_start,
                       int K_len, int NR_use) {
    for (int k = 0; k < K_len; k++) {
        for (int j = 0; j < NR_use; j++) {
            B_packed[k * NR + j] = B[(k + row_start) * ldb + col_start + j];
        }
    }
}
```

### 3.3 A 矩阵打包：行优先（MR×KC 面板）

A 矩阵已经行主序，但打包仍有两个意义：

1. **将 MR 行连续复制出来**：使得微内核中 `A_packed[row * KC + k]` 访问完全线性
2. **对齐和预取**：保证 32 字节对齐，避免跨 cache line 访问

```c
// simd/x86/src/avx2_gemm_micro.cpp:198-207
static void pack_A_8xK(float* A_packed, const float* A, int lda,
                       int row_start, int col_start,
                       int K_len, int MR_use) {
    for (int i = 0; i < MR_use; i++) {
        for (int k = 0; k < K_len; k++) {
            A_packed[i * K_len + k] = A[(row_start + i) * lda + col_start + k];
        }
    }
}
```

### 3.4 打包开销分析

打包本质上是 **O(MK + KN)** 的额外数据搬运。对于大矩阵（M、N、K ≥ 1000），打包开销占比 < 5%。对于小矩阵，可以跳过打包直接使用原始布局。

**生产级优化**：BLIS 的打包函数被高度优化（展开、预取、非时间存储）。当 KC 较小时，A 面板可完全放入 L1 缓存（32 KB），使打包开销几乎为零。

---

## 4. 微内核设计

### 4.1 AVX2 8×8 微内核（本项目的实现）

在 `avx2_gemm_micro.cpp:296-350` 中实现了完整 8×8 寄存器阻塞微内核。该微内核计算：

```
C[0:8][0:8] += A_packed[0:8][0:K] × B_packed[0:K][0:8]
```

**寄存器分配**（AVX2：16 个 ymm 寄存器）：

| 寄存器 | 用途 | 数量 |
|--------|------|------|
| c0..c7 | C 累加器（每行一个 ymm） | 8 |
| b_vec | 加载 B 打包面板 | 1 |
| a_brd | 广播 A 值（复用） | 1 |
| **总计** | | **10/16 (63%)** |

**内部循环（每 k 次迭代）**：

```c
__m256 b_vec = _mm256_loadu_ps(&b[k * ldb]);           // 1 次加载
c0 = _mm256_fmadd_ps(_mm256_set1_ps(a[0*lda+k]), b_vec, c0);  // 广播+FMA
c1 = _mm256_fmadd_ps(_mm256_set1_ps(a[1*lda+k]), b_vec, c1);
// ... c2..c7 同上
// = 1 B 加载 + 8 A 广播 + 8 FMA = 17 条指令，16 次浮点运算
```

**运算强度 (Arithmetic Intensity)**：

```
FLOPs    = 8 FMA × 2 = 16 flops
Bytes    = (1 B 加载 + 8 A 广播) × 4 bytes = 36 bytes
AI       = 16 / 36 = 0.44 flops/byte
```

而标量三重循环的 AI = 2 / 8 = 0.25 flops/byte。寄存器阻塞将数据复用提升 1.76 倍。

### 4.2 AVX2 6×8 vs 8×8

| 特性 | 6×8 (BLIS Haswell) | 8×8 (本项目) |
|------|-------------------|--------------|
| C 累加器数 | 6 个 ymm | 8 个 ymm |
| 寄存器使用率 | 50% (8/16) | 63% (10/16) |
| 余量寄存器 | 8 个（K 展开 ×2 余量充足） | 6 个（K 展开余量紧张） |
| 理论峰值利用率 | 更高（更易隐藏延迟） | 较低但代码更简洁 |
| 适用场景 | 生产级（BLIS 默认） | 教学/演示 |

**为什么 BLIS 选 6×8 而不是 8×8？**

剩余的 8 个寄存器可以用来：
- **K 循环展开**：一次处理 2 个 K 元素（需要 2 个 b_vec 寄存器）
- **C 预取**：提前加载下一批 A 行广播值
- **减少寄存器溢出**：8×8 在需要 `_mm256_permute` 等操作时可能溢出到栈

### 4.3 AVX-512 16×16 微内核

AVX-512 将寄存器宽度翻倍至 512 位（zmm 寄存器），同时将寄存器数量翻倍至 32 个：

```
NR = 16（每个 zmm 容纳 16 个 f32）
MR = 8-16（32 个 zmm 寄存器提供充足余量）
```

**16×16 微内核的寄存器分配**：

| 寄存器 | 用途 | 数量 |
|--------|------|------|
| c0..c15 | C 累加器 | 16 |
| b_vec | B 打包面板加载 | 1 |
| a_brd | A 值广播 | 1 |
| K 展开第二组 | b_vec2 | 1 |
| **总计** | | **19/32 (59%)** |

内部循环每次 K 迭代执行 16 次 FMA = 32 flops，2 倍于 AVX2。加上 2 倍 FMA 单元（部分 SKX 型号），实际吞吐可达 AVX2 的 3-4 倍。

---

## 5. FMA 延迟隐藏：4 路累加器展开

### 5.1 FMA 指令的流水线特性

Skylake 微架构上 `vfmadd231ps` 的关键参数：

| 参数 | 值 |
|------|-----|
| 延迟 (Latency) | 4 个周期 |
| 吞吐 (Throughput) | 2 条/周期（端口 0+1） |
| 每个 FMA 的 uop 数 | 1（融合乘加，单 uop） |

**问题**：如果连续对同一个累加器执行 FMA，必须等前一条 FMA 的 4 个周期延迟结束才能发射下一条。

**解法**：用 4 个独立累加器交错执行：

```c
// 错误：串行依赖链
for (int k = 0; k < K; k++) {
    c0 = _mm256_fmadd_ps(a_brd, b_vec, c0);  // 等 c0 就绪（4 cycle）
}

// 正确：4 路交错
__m256 c0, c1, c2, c3;
for (int k = 0; k < K; k += 4) {
    c0 = _mm256_fmadd_ps(a_brd0, b_vec0, c0);  // 发射后不等
    c1 = _mm256_fmadd_ps(a_brd1, b_vec1, c1);  // 下一周期发射
    c2 = _mm256_fmadd_ps(a_brd2, b_vec2, c2);
    c3 = _mm256_fmadd_ps(a_brd3, b_vec3, c3);
    // c0 在 4 个周期后完成，此时我们正在处理 c3，刚好无等待
}
// 最后归约：c0 += c1 + c2 + c3
c0 = _mm256_add_ps(c0, c1);
c0 = _mm256_add_ps(c0, c2);
c0 = _mm256_add_ps(c0, c3);
```

### 5.2 8×8 微内核中的隐含 ILP

8×8 微内核的 8 个 C 累加器本就可以看作 8 路 ILP——每个 `c[p]` 是独立的：

```
c0 = FMA(a[0], b, c0)  // row 0
c1 = FMA(a[1], b, c1)  // row 1（与 c0 独立）
c2 = FMA(a[2], b, c2)  // row 2（与 c0、c1 独立）
```

但每轮 K 迭代中，8 个 FMA 使用同一个 `b_vec` 加载结果。B 加载的延迟（5 cycle）可以通过在 A 广播阶段重叠来隐藏。进一步优化可以在 K 维度上展开 2 倍：

```c
// K×2 展开（需要额外的 b_vec2 寄存器）
__m256 b_vec0 = _mm256_loadu_ps(&b[(k+0) * ldb]);
__m256 b_vec1 = _mm256_loadu_ps(&b[(k+1) * ldb]);
c0 = _mm256_fmadd_ps(_mm256_set1_ps(a[0*lda+k+0]), b_vec0, c0);
c0 = _mm256_fmadd_ps(_mm256_set1_ps(a[0*lda+k+1]), b_vec1, c0);
// ...
```

这样可以完全隐藏 B 加载延迟（在第 2 个 B 加载期间，第 1 组 FMA 在流水线中）。

---

## 6. 边缘处理：非对齐维度

当 M、N、K 不是微瓦片尺寸的整数倍时，需要处理余数块。

### 6.1 通用边缘处理策略

```c
static void gemm_micro_tiled(int M, int N, int K, ...) {
    for (int mi = 0; mi < M; mi += MR) {
        int mr_use = (mi + MR <= M) ? MR : (M - mi);  // 余数处理
        pack_A_8xK(A_packed, A, lda, mi, 0, K, mr_use);

        for (int nj = 0; nj < N; nj += NR) {
            int nr_use = (nj + NR <= N) ? NR : (N - nj);  // 余数处理
            pack_B_Kx8(B_packed, B, ldb, 0, nj, K, nr_use);

            // 微内核调用，mr_use 或 nr_use 可能 < MR/NR
            gemm_micro_8x8(K, A_packed, K, B_packed, NR,
                           &C[mi * ldc + nj], ldc);
        }
    }
}
```

### 6.2 微内核内处理非对齐列的技巧

对于 `nr_use < 8` 的情况，AVX2 可以使用 **masked store** 避免写入越界：

```c
// AVX2 没有原生 masked store，需要手动构造掩码
__m256i mask = _mm256_setr_epi32(
    (j+0 < nr_use) ? -1 : 0,
    (j+1 < nr_use) ? -1 : 0,
    // ... 共 8 个
);
// 先加载旧值，用掩码选择性覆盖
__m256 old = _mm256_loadu_ps(&C[row * ldc + j]);
__m256 blended = _mm256_blendv_ps(old, c_acc, _mm256_castsi256_ps(mask));
_mm256_storeu_ps(&C[row * ldc + j], blended);
```

AVX-512 则直接提供 `_mm512_mask_store_ps`：

```c
__mmask16 mask = (1 << nr_use) - 1;
_mm512_mask_store_ps(&C[row * ldc + j], mask, c_acc);
```

### 6.3 K 维度非对齐处理

K 维度不要求对齐到 SIMD 宽度。微内核内部循环是标量 K 递增的，天然支持任意 K。唯一的开销是最后一轮 K 迭代可能需要部分展开（用标量循环处理 K % 展开因子）。

---

## 7. 与 BLIS/oneDNN 的对应关系

### 7.1 BLIS 架构映射

| BLIS 概念 | 本项目对应 | 说明 |
|-----------|-----------|------|
| `gemm` API | `gemm_micro_tiled()` | 顶层分块循环 |
| `packm` (pack A) | `pack_A_8xK()` | A 面板行优先打包 |
| `packm` (pack B) | `pack_B_Kx8()` | B 面板列优先打包 |
| `macro-kernel` | 外层 `mi/nj` 循环 | MC×NC 面板级循环 |
| `micro-kernel` | `gemm_micro_8x8()` | 寄存器阻塞的 MR×NR 瓦片 |
| `KC` (K 面板大小) | 当前实现 K=64, 无 KC 分块 | 生产级需添加 |
| `MC`, `NC` | M=64/MR, N=64/NR | 当前为完整矩阵，未做 L2/L3 分块 |

### 7.2 oneDNN (MKL-DNN) 对应

oneDNN 的 GEMM 使用 **JIT (Just-In-Time)** 生成微内核代码：

- **`brgemm`** (batch-reduce GEMM)：将 K 维度的多步归约进行批处理，与 KC 分块对应
- **`int8 GEMM`**：VPMADDUBSW + VPMADDWD + VNNI（AVX-512），与第 12 章的内容对应
- **`packing 函数`**：使用 AVX-512 的 gather/scatter 指令加速打包流程

### 7.3 当前实现 vs 生产级的差距

| 特性 | `avx2_gemm_micro.cpp` | 生产级（BLIS/oneDNN） |
|------|----------------------|----------------------|
| K 展开 | 无（每轮 1 个 K） | 2-4 倍展开 |
| 预取 | 无 | `_mm_prefetch` 在合适位置插入 |
| KC 分块 | 无（K=64 完整放入 L1） | 是，任意 K 都有效 |
| 对齐 | 无显式保证 | 对齐加载 `_mm256_load_ps` |
| 线程并行 | 无 | OpenMP 并行外层循环 |
| 自动调优 | 无 | 参数搜索（BLIS 的 `bli_init`） |

---

## 8. 性能数字

### 8.1 不同 M/N/K 的理论 GFLOPS

以下数据基于单核心 Skylake AVX2（2.5 GHz，理论峰值 40 GFLOPS 双精度 / 80 GFLOPS 单精度（FMA））**估算**。实际值依赖缓存大小和内存带宽。

| M | N | K | FLOPs(2MNK) | 标量 GFLOPS | 简单 SIMD | 微瓦片 | 占峰值% |
|---|----|---|-------------|------------|----------|--------|---------|
| 64 | 64 | 64 | 524,288 | 0.8 | 3.2 | 8.5 | 10.6% |
| 128 | 128 | 128 | 4,194,304 | 1.2 | 6.8 | 18.3 | 22.9% |
| 256 | 256 | 256 | 33,554,432 | 1.5 | 12.4 | 28.7 | 35.9% |
| 512 | 512 | 512 | 268,435,456 | 1.8 | 16.1 | 34.2 | 42.8% |
| 1024 | 1024 | 1024 | 2,147,483,648 | 1.9 | 17.9 | 37.5 | 46.9% |
| 2048 | 2048 | 2048 | 17,179,869,184 | 1.9 | 18.4 | 38.6 | 48.3% |

### 8.2 性能瓶颈分析

| 问题规模 | 瓶颈类型 | 原因 |
|----------|----------|------|
| M,N,K ≤ 128 | **延迟绑定** | 矩阵完全放入 L1，FMA 成为瓶颈 |
| 128 < M,N,K ≤ 1024 | **缓存带宽** | L2/L3 带宽成为制约 |
| M,N,K > 1024 | **主存带宽** | DDR4 双通道 ~40 GB/s，已经饱和 |

### 8.3 优化路径总结

```
标量三重循环                                   基准
  │
  ├─ 向量化 + 打包 (消除 gather)                 ~2-3x
  │
  ├─ 寄存器阻塞 (消除 C 重复加载/存储)            ~1.5-2x
  │
  ├─ K 展开 (FMA 延迟隐藏)                       ~1.2-1.5x
  │
  ├─ L1/L2/L3 缓存分块 (任意大小矩阵)            ~1.2-2x
  │
  ├─ 预取 + 对齐加载 (消除 cache miss)            ~1.1-1.3x
  │
  └─ 多线程并行 (OpenMP/线程池)                  ~Nx (N=核心数)
                                                    ────
                                                 总计 ~10-50x
```

---

## 9. 代码引用

### 9.1 现有代码

所有实现位于 **`simd/x86/src/avx2_gemm_micro.cpp`**：

| 函数 | 行号 | 功能 | 优化等级 |
|------|------|------|----------|
| `scalar_gemm()` | 107-120 | 标量三重循环基准 | Level 0 |
| `gemm_naive_simd()` | 136-170 | 向量化但含 gather | Level 1 |
| `gemm_packed_simd()` | 226-254 | B 打包消除 gather | Level 2 |
| `gemm_micro_8x8()` | 296-350 | **8×8 寄存器阻塞微内核** | Level 3 |
| `gemm_micro_tiled()` | 367-395 | 瓦片化循环 + 打包 + 微内核 | Level 4 |
| `pack_A_8xK()` | 198-207 | A 面板行优先打包 | 辅助 |
| `pack_B_Kx8()` | 187-196 | B 面板列优先打包 | 辅助 |
| `hsum_ps()` | 89-98 | 水平归约（permute+add+hadd） | 辅助 |
| `compute_gflops()` | 401-406 | GFLOPS 计算 | 辅助 |

### 9.2 新增生产级实现：`avx2_gemm_production.cpp`

在 `avx2_gemm_micro.cpp` 基础上进一步优化，新增文件应包含：

```
simd/x86/src/avx2_gemm_production.cpp  （待创建）
```

**新增特性**：

1. **KC 分块**：支持任意 K 维度（当前 K=64 硬编码）
2. **K×2 展开**：在微内核中一次处理 2 个 K 元素
3. **软件预取**：在打包函数中插入 `_mm_prefetch`
4. **对齐加载**：使用 `_mm256_load_ps`（而非 `loadu`）保证对齐
5. **OpenMP 并行**：在外层 `ic` 循环上添加 `#pragma omp parallel for`
6. **AVX-512 调度**：通过运行时 CPUID 检测自动选择 AVX2 或 AVX-512 微内核
7. **动态分块**：基于 L1/L2/L3 缓存大小自动计算 MC/KC/NC

**与现有代码的继承关系**：

```
avx2_gemm_micro.cpp          avx2_gemm_production.cpp
    │                                  │
    ├─ gemm_micro_8x8()  ──────────►  沿用 + K×2 展开
    ├─ pack_A_8xK()      ──────────►  添加预取
    ├─ pack_B_Kx8()      ──────────►  添加预取 + 对齐
    ├─ gemm_micro_tiled() ──────────►  拆分为 MC/KC/NC 三层 + 并行
    └─ hsum_ps()          ──────────►  直接复用
```

---

## 附录：关键术语对照

| 中文 | English | 解释 |
|------|---------|------|
| 微内核 | micro-kernel | 寄存器阻塞的最内层 `MR×NR` 矩阵乘法 |
| 面板 | panel | 打包后的子矩阵（A 面板 `MC×KC`，B 面板 `KC×NC`） |
| 瓦片 | tile | 缓存层次中分块的基本单元 |
| 打包 | packing | 将数据重新排列为连续内存布局 |
| 广播 | broadcast | 将一个标量复制到 SIMD 寄存器的所有 lane |
| FMA | Fused Multiply-Add | `c = a × b + c`，单指令完成乘加 |
| ILP | Instruction-Level Parallelism | 通过独立指令交错来隐藏延迟 |
| AI | Arithmetic Intensity | 浮点运算次数 / 内存访问字节数 |
| 余数块 | fringe/remainder block | 维度不对齐时的边界处理瓦片 |
