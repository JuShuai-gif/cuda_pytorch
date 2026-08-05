# 20 矩阵分块与 Cache Blocking

> 对应 PDF：第 6.2.1 节 Optimizing Level 1 Data Cache Access 的子矩阵部分（PDFp50~51）、附录 A.1 Matrix Multiplication（PDFp97~98）、表 6.2
> 本篇回答：Cache Blocking（分块）是什么？为什么能同时压住 L1 与 LLC 的 miss？论文的六重循环分块怎么读？附录 A.1 的 SIMD 版怎么运作？

## 1. 本章要解决的问题

- 朴素矩阵乘法为什么会 miss 到飞起？
- 转置能救一时，为什么不通用？
- 分块（blocking）如何在不用拷贝的前提下保持局部性？
- 六重循环的结构怎么理解？SM 如何取？
- 附录 A.1：对齐 + 预取 + SSE2 的完整版怎么拼？

## 2. 前置知识

- note/18：朴素乘法、转置、表 6.2 优化链。
- note/05：L1d/LLC 大小、缓存行。
- SIMD（SSE2）基础：`__m128d`、`_mm_load_pd` 等。

## 3. 核心概念

- **Cache Blocking / Tiling（缓存分块）**：把矩阵拆成能放进某级缓存的小块，逐块计算。
- **SM（Sub-Matrix Size）**：分块尺寸，通常 = 缓存行大小 / sizeof(元素)。
- **Loop Blocking**：通过循环步长把访存局部化。
- **Tile（块）**：一次处理的小矩形。
- **`_mm_prefetch`**：显式预取指令（附录 A.1 使用 NTA 提示）。
- **`__attribute__((aligned(64)))`**：矩阵按缓存行对齐（附录 A.1）。

## 4. 硬件工作流程

### 4.1 朴素乘法的问题回顾

```text
for i: for j: for k:
   res[i][j] += mul1[i][k] * mul2[k][j];

mul1：按行顺序 → 好
mul2：按列顺序（k 变行号）→ 每条缓存行只用 1 个 double → 差
res ：按行顺序 → 好
```

- 内层每轮需要 3×1000 条缓存行 > 32kB L1d → 全 miss。
- `mul2` 的 `(0,0)`/`(0,1)` 本同缓存行，但等内层跑完一行早已被逐出。

### 4.2 转置的局限

- 转置后两矩阵都顺序访问（表 6.2：100% → 23.4%）。
- 但需要额外拷贝（tmp 矩阵）；矩阵太大/内存不足时不适用。

### 4.3 分块（Blocking）核心思想

```text
把乘法拆成 SM×SM 的小块：
   外层循环按 SM 步进遍历大矩阵
   内层循环处理一个块，块内数据能同时驻留 L1d
```

```text
for i step SM: for j step SM: for k step SM:
   for i2 in [0,SM): for k2 in [0,SM): for j2 in [0,SM):
       res[i+i2][j+j2] += mul1[i+i2][k+k2] * mul2[k+k2][j+j2]
```

- 块内 `mul1` 的 `[i+i2][k+k2]`（行连续）、`mul2` 的 `[k+k2][j+j2]`（行连续）都顺序。
- 块内数据在较短时间内被反复使用 → 时间局部性。
- 表 6.2：Sub-Matrix 17.3%（比转置再省 6.1%，且无需拷贝）。

### 4.4 附录 A.1 的完整版（PDFp97）

```cpp
#define N 1000
double res[N][N] __attribute__ ((aligned (64)));
double mul1[N][N] __attribute__ ((aligned (64)));
double mul2[N][N] __attribute__ ((aligned (64)));
#define SM (CLS / sizeof (double))

for (i = 0; i < N; i += SM)
  for (j = 0; j < N; j += SM)
    for (k = 0; k < N; k += SM)
      for (i2 = 0, rres = &res[i][j], rmul1 = &mul1[i][k];
           i2 < SM; ++i2, rres += N, rmul1 += N) {
        _mm_prefetch (&rmul1[8], _MM_HINT_NTA);
        for (k2 = 0, rmul2 = &mul2[k][j]; k2 < SM; ++k2, rmul2 += N) {
          __m128d m1d = _mm_load_sd (&rmul1[k2]);
          m1d = _mm_unpacklo_pd (m1d, m1d);   // 复制成向量 {a,a}
          for (j2 = 0; j2 < SM; j2 += 2) {
            __m128d m2 = _mm_load_pd (&rmul2[j2]);
            __m128d r2 = _mm_load_pd (&rres[j2]);
            _mm_store_pd (&rres[j2], _mm_add_pd (_mm_mul_pd (m2, m1d), r2));
          }
        }
      }
```

要点：

- 三数组都对齐 64B → 期望同行的元素确实同行。
- `rmul1[k2]` 从内层提出，用 `_mm_unpacklo_pd` 复制成 `{a,a}` 向量。
- `_mm_prefetch(&rmul1[8])` 提前 8 个元素预取，NTA 提示（不污染缓存）。
- `restrict` 声明指针无别名。

## 5. PDF 核心观点

> 来源：PDF 第 50~51、97~98 页；对应章节 6.2.1（子矩阵）、A.1、表 6.2。以下为概括。

1. **加法顺序无关允许重排**（PDFp50）：只要每个加数恰好出现一次，内层加法顺序可随意，这是分块的数学基础。
2. **内层循环的缓存行利用**（PDFp50）：一次处理 2 次中循环迭代 → 同一缓存行用两次，L1d miss 减半。
3. **展开次数由缓存行决定**（PDFp50）：64B/8B = 8 次中循环展开、8 次外循环展开（整行写 res）。
4. **六重循环的意义**（PDFp50）：外层按 SM 分块 → 局部性；内层三个循环处理块内；k2/j2 顺序因依赖不同而交换。
5. **gcc 对数组索引不聪明**（PDFp50）：用 `rres/rmul1/rmul2` 指针把公共表达式提出内层；C 的别名规则阻碍优化，`restrict` 有助（论文注：编译器仍未完全支持）；Fortran 因默认无别名而受数值计算偏爱。
6. **表 6.2 全链条**（PDFp50~51）：Original 100% → Transposed 23.4% → Sub-Matrix 17.3% → Vectorized 9.47%（318 MFLOPS → 3.35 GFLOPS）。
7. **向量化仍受 mul2 预取限制**（PDFp51）：mul2 在最终版仍无法完美预取，除非转置；3.19 GFLOPS 单线程已不错。
8. **附录 A.1 完整版**（PDFp97）：结构同 6.2.1 最终版，唯一变化是把 rmul1[k2] 提出内层并用 unpacklo 构向量；显式对齐三数组。

## 6. 通俗解释

Cache Blocking 就像**把大工程拆成桌面能放下的小任务**：

> 朴素做法：图纸（矩阵）太大，你每算一步都要去仓库搬数据（miss），搬回来又放不下桌面（缓存）。
> 分块：把图纸裁成桌面大小的块，一块块算。每块搬一次数据，就能在桌面上把它算完再换下一块。
> 于是"搬数据"（访存）的次数从每元素一次，变成每块一次。

为什么转置不够好？

> 转置 = 先把图纸重新复印一遍（拷贝 tmp），保证后来竖读变横读。复印要花钱花地方。
> 分块 = 不用复印，直接原地裁块算，省下复印费（6.1%）。

为什么循环变成六重？

> 原来的 i/j/k 三重循环把整个矩阵当一个大任务。分块后外层 i/j/k 用来"选哪一块"，
> 内层 i2/k2/j2 用来"算块里的哪个元素"——三重变六重，各司其职。

## 7. 示例分析

### 7.1 SM 的取值

- `SM = CLS / sizeof(double)`：64B / 8B = 8。
- 中循环展开 8 次 → mul1 一行 8 个元素恰好一条缓存行。
- 外循环展开 8 次 → res 一次写 8 个结果（整行写）。
- 若缓存行 32B：SM=4，代码依旧正确（两条缓存行都 100% 利用）。

### 7.2 转置 vs 分块

- 转置：多一次拷贝；两矩阵全顺序；1000 次非顺序访问的代价被覆盖。
- 分块：不拷贝；块内局部性好；输入矩阵可任意大（只要 res 能放下）。
- 表 6.2：转置 23.4% → 分块 17.3%（再省 6.1%）。

### 7.3 向量化的叠加

- SSE2 一次处理 2 个 double（`__m128d`）。
- `_mm_unpacklo_pd(a,a)` 把标量扩成 `{a,a}` 向量。
- `_mm_mul_pd + _mm_add_pd` 完成 `res += mul1·mul2` 两元素。
- 最终 318 MFLOPS → 3.35 GFLOPS。

## 8. 未优化代码

朴素三重循环矩阵乘法。

```cpp
// bad.cpp: 朴素矩阵乘法
#include <vector>

int main() {
    constexpr int N = 512;
    std::vector<std::vector<double>> A(N, std::vector<double>(N, 1.0));
    std::vector<std::vector<double>> B(N, std::vector<double>(N, 1.0));
    std::vector<std::vector<double>> C(N, std::vector<double>(N, 0.0));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < N; ++k)
                C[i][j] += A[i][k] * B[k][j];
    return C[0][0] == 0.0;
}
```

## 9. 优化后代码

分块 + 转置 B（无需 SIMD 也快很多）的版本。

```cpp
// good.cpp: 分块 + 转置 B
#include <vector>

int main() {
    constexpr int N = 512;
    constexpr int SM = 8;
    std::vector<double> A(N * N, 1.0), B(N * N, 1.0), T(N * N, 0.0), C(N * N, 0.0);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            T[j * N + i] = B[i * N + j];   // 转置

    for (int i = 0; i < N; i += SM)
        for (int j = 0; j < N; j += SM)
            for (int k = 0; k < N; k += SM)
                for (int i2 = 0; i2 < SM; ++i2)
                    for (int k2 = 0; k2 < SM; ++k2) {
                        double a = A[(i + i2) * N + (k + k2)];
                        for (int j2 = 0; j2 < SM; ++j2)
                            C[(i + i2) * N + (j + j2)] += a * T[(k + k2) * N + (j + j2)];
                    }
    return C[0] == 0.0;
}
```

> 分块大小扫描（8/16/32/64/128）见 src/10_cache_blocking；完整优化链见 src/28_integrated_project。

## 10. 为什么会更快

| 角度 | 朴素 | 分块(+转置) |
|---|---|---|
| L1d miss | 每元素一次（mul2 列访问） | 每缓存行一次（块内顺序） |
| 缓存行利用率 | mul2 ~12.5% | ~100% |
| 时间局部性 | 无（行被逐出才再用） | 块内短时间复用 |
| 预取 | mul2 无效 | 全部可预取 |
| 内存带宽 | 浪费 | 充分 |

论文数据（表 6.2）：转置 23.4%、分块 17.3%、向量化 9.47%（2007 Core 2，量级参考）。

## 11. 如何验证

```bash
./build/10_cache_blocking/cache_blocking       # 分块大小扫描
./build/28_integrated_project/integrated_project  # 完整优化链
./scripts/perf_stat.sh ./build/10_cache_blocking/cache_blocking
./scripts/cachegrind.sh ./build/10_cache_blocking/cache_blocking
```

## 12. 实验结果应该怎么看

- 分块大小扫描：最优 block 通常略小于 L1d/L2 大小（留余量给代码、栈与其他数据）。
- 对比不同 block size 的 cache-misses 与运行时间；过大/过小都会变差。
- 若启用 SIMD（需要 ENABLE_AVX2/AVX512 选项 + 运行时检测），对比 MFLOPS/GFLOPS。

## 13. 常见误区

- **误区 1：分块只对矩阵乘法有用**。任何"数据超缓存但可拆块重复使用"的算法都受益（卷积、BLAS、图像滤波等）。
- **误区 2：块越大越好**。块必须能驻留目标缓存；超出即 miss。
- **误区 3：转置是唯一解法**。分块无需拷贝、更通用；转置只是思路之一。
- **误区 4：编译器自动就能分块**。gcc 对数组索引/别名不聪明，常需手写或用库（BLAS）。
- **误区 5：向量化一定要配 AVX-512**。SSE2 也能 2 倍；先保证数据布局再谈向量化。

## 14. 实践练习

1. 运行 src/10，扫描 block 8/16/32/64/128，记录运行时间并找出最优。
2. 复现表 6.2：朴素/转置/分块/分块+向量化四档，记录相对时间与 FLOPS。
3. 解释"为什么 SM=缓存行大小/sizeof(double)"。
4. 用 Cachegrind 对比朴素与分版的缓存 miss 数量。
5. 讨论：若 mul2 无法转置且不能分块，还有什么办法？（预取、辅助线程等）

## 15. 本章总结

- 分块把"每元素一次访存"变成"每块一次"，同时压住 L1 与 LLC miss。
- 数学基础：加法顺序无关；无需拷贝即可重排。
- 六重循环：外层选块、内层算块；SM 由缓存行决定。
- 对齐（aligned(64)）+ 预取（_mm_prefetch）+ SIMD（unpacklo）层层叠加。
- 论文数据：100% → 23.4%（转置）→ 17.3%（分块）→ 9.47%（向量化）。

## 16. 对应代码

- src/10_cache_blocking/（分块大小扫描）
- src/28_integrated_project/（完整优化链，含 FLOPS 统计）
- src/09_matrix_traversal/（行/列遍历基础）
