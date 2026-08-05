# 09 SIMD与自动向量化

> 本笔记对应 PDF 第 12 章 Using vector operations（第 115～134 页）中的核心概念部分：12.1（AVX/YMM）、12.2（AVX-512/ZMM）、12.3（自动向量化）、12.6（串行代码向量化）、12.7（向量数学函数）、12.9（RGB/3D 对齐）、12.10（结论）。Intrinsics 与向量类部分（12.4、12.5、12.8）见 `note/10_Intrinsics编程.md`。

## 1. 本章解决什么问题

向量操作（SIMD，Single Instruction Multiple Data）让 CPU **一条指令同时处理多个数据**。本章回答：

1. 向量寄存器有哪些规格？（XMM/YMM/ZMM、SSE/AVX/AVX-512）
2. 编译器怎么自动向量化？（前提条件、选项、障碍）
3. 什么时候值得用向量？（有利/不利因素）

核心结论：**简单规则循环编译器能自动向量化；向量化收益取决于数据类型大小、数据量、可预测性和指令集；无法自动向量化时改用 Intrinsics（见 10 笔记）。**

## 2. 核心概念

| 术语 | 含义 | 出处 |
|------|------|------|
| SIMD | 单指令多数据：一条指令处理多个元素 | PDF 第 115 页 |
| XMM / YMM / ZMM | 128 / 256 / 512 位向量寄存器 | PDF 第 115 页 |
| SSE / SSE2 | 128 位向量指令集（浮点 / 整数与 double） | PDF 第 115 页 |
| AVX / AVX2 | 256 位浮点 / 整数向量 | PDF 第 117 页 |
| AVX-512 | 512 位向量 + mask 寄存器 | PDF 第 117 页 |
| 自动向量化 | 编译器把循环改写为向量指令 | PDF 第 118 页 |
| 掩码（mask） | AVX-512 按元素条件执行用的布尔向量 | PDF 第 117 页 |
| 归约（reduction） | 把多个元素合并成一个结果（如求和） | PDF 第 129 页 |
| gather | AVX2 按索引向量查表 | PDF 第 124 页 |

## 3. 工作原理

### 3.1 向量寄存器能装多少数据（PDF 第 115～116 页，表 12.1）

| 寄存器 | 宽度 | float | double | int32 |
|--------|------|-------|--------|-------|
| MMX | 64 位 | - | - | 2 |
| XMM（SSE/SSE2） | 128 位 | 4 | 2 | 4 |
| YMM（AVX/AVX2） | 256 位 | 8 | 4 | 8 |
| ZMM（AVX-512） | 512 位 | 16 | 8 | 16 |

一条 `a[i] = b[i] + c[i]` 的循环，AVX-512 下每周期处理 16 个元素。

### 3.2 向量化收益的本质

把 4/8/16 次独立运算合并成一条指令。收益来源：**执行单元吞吐**（每周期能发的向量指令数与标量相当）+ **更少的循环开销**（每向量一批元素才一轮循环控制）。

### 3.3 自动向量化的机制（PDF 第 118～119 页）

编译器扫描循环，判断能否证明"每轮独立 + 无别名 + 无副作用"，然后：

- 向量化主循环 + 处理尾部余数；
- 数据对齐（可对齐数组 vs 指针不确定对齐）；
- 指定指令集（`-msse2`/`-mavx2`/`-mavx512f` 等）决定用多大寄存器。

## 4. PDF 核心观点

### 12.1 AVX 与 YMM 寄存器（第 117 页）

- AVX 把 XMM 扩展到 256 位 YMM；AVX2 支持 256 位整数向量（PDF 第 117 页）。
- 从 AVX 代码切换到非 AVX 代码时有**性能惩罚**（YMM 寄存器状态变化）；过渡前应调用 `_mm256_zeroupper()`（PDF 第 117 页）。
- 需要 `_mm256_zeroupper()` 的场景：部分程序用 AVX 编译、部分不用；CPU dispatch 的 AVX 版本离开时；AVX 代码调用非 AVX 库函数前（PDF 第 117 页）。

### 12.2 AVX-512 与 ZMM 寄存器（第 117～118 页）

- ZMM 512 位；64 位模式下向量寄存器从 16 个增加到 **32 个**；AVX-512 代码应编译为 64 位（PDF 第 117 页）。
- 新增 **mask 寄存器**：按元素条件执行，使含分支的代码向量化更高效（PDF 第 117 页）。
- AVX-512 有多个子扩展：F（基础）、VL、BW、DQ、ER、CD、PF、VBMI、IFMA、FP16 等；**没有处理器支持全部**，选子集做分支（PDF 第 117～118 页）。

### 12.3 自动向量化（第 118～121 页）

- 简单循环（`a[i]=b[i]+2`）在指定 SSE2+ 后编译器自动向量化（PDF 第 118 页，Example 12.1a）。
- 指针访问的数组有对齐/别名问题，向量化代码更繁琐（PDF 第 118 页，Example 12.1b）。
- **自动向量化的有利条件**（PDF 第 119 页）：
  1. 好的编译器（GNU/Clang/Intel）；
  2. 新版本编译器；
  3. 指针访问用 `__restrict__` 声明无别名；
  4. 用选项启用目标指令集（`-mavx2` 等）；
  5. 宽松浮点选项：`-O2 -fno-trapping-math -fno-math-errno -fno-signed-zeros`（不建议 `-ffast-math`，会禁用 isnan 等）；
  6. 数组按 16（SSE2）/32（AVX）/64（AVX-512）对齐；
  7. 循环次数是向量宽度的倍数（可加哑元补足）；
  8. 指针数组对齐信息让编译器可见；
  9. 尽量减少向量元素级分支；
  10. 避免向量元素级查表。
- **向量化障碍**（PDF 第 119 页）：无法排除指针别名；未走分支可能产生异常/副作用；不知数组大小是否向量宽度倍数；不知对齐；数据需重排；代码太复杂；调用无向量版本的函数；用查表。
- 非循环也能向量化：连续 4 个 float 的结构可打包成 XMM（PDF 第 120 页，Example 12.2）。
- 分支向量化：计算两侧再合并；需要 `-fno-trapping-math`（PDF 第 120 页）。
- **数据越小越有利**：short 比 int、float 比 double 每向量装更多（PDF 第 120 页）。
- SSE2 不能做任意宽度整数乘法；向量没有整数除法指令（PDF 第 121 页）。

### 12.6 串行代码的向量化改造（第 129～131 页）

- 求和 `sum += a[i]` 是串行归约；按向量宽度 n 展开为 n 路部分和，再横向相加（PDF 第 129 页，Example 12.7）。
- 好的编译器在 fast-math + SSE2 下会自动把 12.7a 转成 12.7b（PDF 第 130 页）。
- 泰勒级数求 exp(x)：每项依赖上一项，无法直接向量化；用系数表存 `1/n!` 并每 4 项一组并行计算（PDF 第 130 页，Example 12.8）。
- 注意：展开后依赖链变成长度为 n 的"每 n 步一跳"，速度可能受乘法延迟而非吞吐限制（PDF 第 131 页）。

### 12.7 向量数学函数（第 131～132 页）

- 长向量库（一次处理整数组，如 MKL）与短向量库（一次处理一个寄存器，如 Sleef/SVML/libmvec/VCL）两种（PDF 第 131 页）。
- 短向量库中间结果留在寄存器、可直连下一步，常更快；但长依赖链时可能阻塞乱序（PDF 第 131 页）。
- GCC 用自带 `libmvec`；Clang 12+ 可用 `-fveclib=libmvec`；Intel SVML 函数最全（PDF 第 132 页）。
- 各库对特殊值（INF/NaN/subnormal）的处理差异很大；按应用场景选（PDF 第 132 页）。

### 12.9 RGB / 3D 向量对齐（第 133 页）

- RGB 三个值 / 三维向量装不满 4 宽向量。三种方案：**加第 4 个无用值**（占内存）；**每 4/8 个点一组**，R 归 R、G 归 G、B 归 B；**全部 R 在前、G 中、B 后**（SoA 式）（PDF 第 133 页）。
- 点数不是向量宽度倍数时补哑点（PDF 第 133 页）。

### 12.10 结论（第 133～134 页）

- 算法允许并行时向量化收益很大，取决于每向量元素数（PDF 第 133 页）。
- 优先依赖自动向量化；编译器做不到时再用 intrinsics/向量类/汇编（PDF 第 133 页）。
- 手写向量化后编译器还能继续优化（内联/CSE/常量传播），这是手写汇编做不到的（PDF 第 134 页）。
- **向量化有利因素**：小数据类型（char/int16/float）、大数组同类操作、数组大小是向量宽度倍数、不可预测的分支二选一、向量专属操作（min/max/饱和加法/近似倒数/平方根倒数/RGB 色差）、有 AVX2/AVX-512、向量数学库、GNU/Clang（PDF 第 134 页）。
- **不利因素**：大类型（int64/double）、未对齐数据、需要大量转换/重排、可预测分支可跳过大部分表达式、编译器不知对齐/别名、指令集缺操作（如 32 位整数乘法）、老 CPU 执行单元小于寄存器宽度（PDF 第 134 页）。
- 向量化代码更易错，宜放可复用、充分测试的库模块中（PDF 第 134 页）。

## 5. 简单示例

```cpp
// A loop that a good compiler vectorizes automatically (PDF p118, Ex 12.1a).
// Compile with -O3 -mavx2: 8 floats handled per instruction on this machine.
const int size = 1024;
float a[size], b[size];

void add_two() {
    for (int i = 0; i < size; ++i) {
        a[i] = b[i] + 2.0f;
    }
}
```

## 6. 未优化代码

阻碍自动向量化的典型：指针访问且无别名保证（PDF 第 118 页，Example 12.1b 的问题）：

```cpp
// The compiler cannot prove that aa and bb do not overlap/alias,
// and it does not know their alignment. Vectorization is still possible
// but the code has to be guarded, so it is less efficient.
void add_two_ptr(float *aa, float *bb, int n) {
    for (int i = 0; i < n; ++i) {
        aa[i] = bb[i] + 2.0f;
    }
}
```

## 7. 优化后代码

```cpp
// Tell the compiler the pointers do not alias (GCC/Clang __restrict__).
// Combined with -O3 -mavx2 this vectorizes cleanly.
void add_two_ptr(float *__restrict__ aa, float *__restrict__ bb, int n) {
    for (int i = 0; i < n; ++i) {
        aa[i] = bb[i] + 2.0f;
    }
}
```

## 8. 为什么会更快

- **向量化**：每条向量加法处理 4/8/16 个元素（SSE/AVX2/AVX-512），指令数下降一个量级（PDF 第 73、115 页）。
- **执行单元吞吐**：向量指令占用与标量相同的发射槽，但负载更大（PDF 第 115 页）。
- **循环开销**：每批元素一轮循环控制，分支/计数开销按比例摊薄。
- **数据大小**：float 比 double 每向量多一倍元素，所以"用小类型更有利"（PDF 第 120 页）。
- **`__restrict__` 的作用**：消除别名假设障碍，让编译器敢向量化（PDF 第 119 页）。

## 9. 如何验证

```bash
# 查看向量化报告（GCC）
g++ -O3 -std=c++17 -mavx2 -fopt-info-vec -fopt-info-vec-missed vec.cpp -o vec

# Clang 的等价报告（本机未装，仅供记录）
# clang++ -O3 -std=c++17 -mavx2 -Rpass=loop-vectorize -Rpass-missed=loop-vectorize vec.cpp

# 确认汇编里出现 ymm 寄存器 / vaddps 指令
g++ -O3 -std=c++17 -mavx2 -S -masm=intel vec.cpp -o /tmp/vec.s
grep -E "vaddps|ymm" /tmp/vec.s | head

# 对比不同指令集（本机最高 AVX2，无 AVX-512）
g++ -O3 -std=c++17 -mavx2 vec.cpp -o vec_avx2
g++ -O3 -std=c++17 vec.cpp -o vec_sse       # 默认 SSE2
./vec_avx2 && ./vec_sse
```

- 编译命令：`g++ -O3 -std=c++17 -mavx2`（本机 g++ 13.3.0，CPU 支持 AVX2）
- 运行命令：`./vec_avx2` / `./vec_sse`
- 向量化报告：GCC `-fopt-info-vec`；Clang `-Rpass=loop-vectorize`
- perf 命令：`sudo perf stat ./vec_avx2`（本机需 root）
- 查看汇编：`g++ -O3 -mavx2 -S -masm=intel`，搜 `ymm`/`vaddps`

## 10. 常见误区

- **误区一：-O3 就会自动向量化。** 需要指定指令集（默认只到 SSE2）并满足无别名/无副作用等条件（PDF 第 118～119 页）。
- **误区二：`-ffast-math` 是向量化必需品。** `-fno-trapping-math -fno-math-errno` 已足够，且保留 NaN 检测（PDF 第 119 页）。
- **误区三：int64/double 也适合向量化。** 每向量元素少一半，收益减半（PDF 第 134 页）。
- **误区四：向量化总比标量快。** 数据转换/重排多、未对齐、老 CPU 半宽执行单元时反而更慢（PDF 第 134 页）。
- **误区五：查表可以和向量化共存。** 查表是向量化障碍（PDF 第 119、124 页）。
- **误区六：本机有 AVX-512。** 当前 i9-14900HX 支持到 AVX2，无 AVX-512；`-mavx512f` 编译的程序在本机无法运行，必须用 CPUID 分发（见 11 笔记）。

## 11. 实践任务

1. 用 `-fopt-info-vec` 编译第 5 节循环，看 GCC 是否向量化；再试去掉 `__restrict__` 看报告变化。
2. 对比 `-mavx2` 与默认（SSE2）下同一循环的运行时间与汇编（ymm vs xmm）。
3. 把 `sum += a[i]` 改写成 8 路部分和（按 AVX2 宽度），与标量版本对比，验证 12.6 节的归约改造。
4. 写一个含 `if` 的循环（`a[i] = (b[i] > 0) ? b[i]*2 : b[i]`），看能否向量化；加 `-fno-trapping-math` 后再看。
5. 用 `alignas(64)` 声明大数组重跑，对比指针版本（对齐信息可见性）的性能。

## 12. 本章总结

- SIMD 一条指令处理多元素；XMM/YMM/ZMM 分别 128/256/512 位。
- 自动向量化需要：好编译器 + 指令集选项 + 无别名/副作用 + 宽松浮点选项 + 对齐 + 规则访问。
- 归约（求和）按向量宽度拆多路部分和；泰勒级数等串行算法用系数表 + 分组并行。
- 数据越小越有利；转换/重排/未对齐/老 CPU 是不利因素。
- 编译器做不了的再上 Intrinsics（见 10 笔记）。

## 13. 对应代码

本章对应实验（阶段三实现）：

- `src/12_auto_vectorization/` —— 可/不可向量化循环、restrict、GCC/Clang 向量化报告
- `src/13_intrinsics/` —— SSE/AVX2/AVX-512 intrinsics（衔接 10 笔记）
- `src/09_alignment/` —— 对齐对向量化的影响
- `src/08_memory_cache/` —— AoS/SoA 与向量化能力对比

> 状态：上述实验代码尚未实现（阶段三完成），届时更新本节链接。
