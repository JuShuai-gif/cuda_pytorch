# 10 Intrinsics编程

> 本笔记对应 PDF 第 12.4 节 Using intrinsic functions（第 121～124 页）、12.5 节 Using vector classes（第 125～128 页）、12.8 节 Aligning dynamically allocated memory（第 133 页）。SIMD 基础与自动向量化见 `note/09_SIMD与自动向量化.md`。

## 1. 本章解决什么问题

自动向量化不可预测：编译器可能不向量化、或向量化得很差。本章给出**程序员手动控制向量化**的三条路：

1. **Intrinsic 函数**（12.4）：直接写"几乎就是汇编"的向量函数，但保留编译器寄存器分配与进一步优化；
2. **向量类**（12.5）：用重载运算符把 intrinsics 包装成可读代码，性能相同；
3. **对齐内存**（12.8）：让数据满足向量加载要求。

核心结论：**intrinsics 与向量类性能相同、只是可读性差别；C++17 下 `new` 对齐类型化数组自动满足对齐要求。**

## 2. 核心概念

| 术语 | 含义 | 出处 |
|------|------|------|
| Intrinsic 函数 | 几乎一对一映射到机器指令的向量函数 | PDF 第 121 页 |
| `__m128i`/`__m128`/`__m128d` | 128 位整数/float/double 向量类型 | PDF 第 122 页 |
| `__m256*`、`__m512*` | 256/512 位向量类型 | PDF 第 122 页 |
| mask 指令 | AVX-512 掩码条件指令（`_mm512_mask_*`） | PDF 第 123 页 |
| 向量类（vector class） | 用类+运算符包装向量（Intel dvec.h / Agner VCL） | PDF 第 125 页 |
| gather | AVX2 按索引向量查表 | PDF 第 124 页 |
| permute | 寄存器内重排元素 | PDF 第 124 页 |
| 对齐内存 | 地址可被向量宽度（16/32/64）整除 | PDF 第 133 页 |

## 3. 工作原理

### 3.1 intrinsic 编译流程（PDF 第 121 页）

intrinsic 函数调用在编译时被翻译成具体的机器指令，但编译器仍负责寄存器分配、调用约定、指令重排、公共子表达式消除。相比手写汇编，它**更安全、且能被继续优化**。

包含头文件：`immintrin.h`（或 `x86intrin.h`）覆盖所有指令集（PDF 第 124 页）。

### 3.2 分支怎么在向量里表达

- 传统做法（SSE2）：比较生成全 1/全 0 掩码 → `AND`/`ANDNOT` + `OR` 选两支（PDF 第 122 页，Example 12.4b）；
- SSE4.1：一条 `_mm_blendv_epi8` 按掩码混合（PDF 第 123 页，Example 12.4c）；
- AVX-512：mask 寄存器 + 条件指令 `_mm512_mask_add_epi16`，一条指令完成"计算+选择"（PDF 第 123 页，Example 12.4d）。

### 3.3 对齐为何重要

SSE 时代对齐 load 更快，未对齐 load 在部分 CPU（Atom、老 Intel）上有显著惩罚；AVX/AVX-512 后对齐要求放宽。对齐访问是"免费的优化"，且不破坏可移植性（PDF 第 118、124 页）。

## 4. PDF 核心观点

### 12.4 使用 intrinsic 函数（第 121～124 页）

- 自动向量化结果不可预测；intrinsics 在"编译器不向量化或向量化差"时显式控制（PDF 第 121 页）。
- 向量化带分支的循环（Example 12.4a）用掩码 + AND/OR（SSE2）或 blend（SSE4.1）或 mask 指令（AVX-512）（PDF 第 121～124 页）。
- Example 12.4b（SSE2 掩码法）比标量快 **3-7 倍**，取决于分支可预测性（PDF 第 122 页）。
- 向量类型：`__m128i`（整数，可含 16×8/8×16/4×32/2×64 位）、`__m128`（4 float）、`__m128d`（2 double），对应 256/512 位为 `__m256*`、`__m512*`（PDF 第 122 页）。
- intrinsic 函数以 `_mm` 开头，列在 Intel 手册与 Intel Intrinsics Guide 中，数千个，需仔细挑选（PDF 第 122 页）。
- 用哪个指令集必须自己确保 CPU 支持，否则崩溃（PDF 第 124 页）；检测方法见第 13 章（11 笔记）。
- **对齐数据**：向量 load/store 前用 `alignas` 声明对齐数组（PDF 第 124 页，Example 12.5）。
- 查表向量化：AVX2 gather 支持按索引向量查表，但元素逐个读取、慢；小表用 permute 更快（PDF 第 124 页）。

### 12.5 使用向量类（第 125～128 页）

- intrinsic 代码冗长；向量类用运算符包装，机器码与 intrinsics 相同（PDF 第 125 页）。
- 优点：显式控制向量化、绕过自动向量化障碍、代码比 intrinsics 简洁、性能相同（PDF 第 125 页）。
- 两大库：Intel 向量类（`dvec.h`，仅 Intel/MSVC 编译器）与 Agner 的 **VCL**（`vectorclass.h`，支持 Intel/MSVC/GNU/Clang、Windows/Linux/Mac/BSD、Apache 2.0 许可）（PDF 第 125 页，表 12.2）。
- 表 12.3 列出了各宽度/类型的向量类名（如 `Vec4f`、`Vec8f`、`Vec16f` 对应 128/256/512 位 float）（PDF 第 126 页）。
- 64 位 MMX 向量与浮点代码不兼容，别用；128 位以上无此问题（PDF 第 126 页）。
- VCL 可按指令集用预处理宏 `INSTRSET` 选择实现，并能配合 CPU dispatch 一次编译四个版本（SSE2/SSE4.1/AVX2/AVX-512BW）（PDF 第 127～128 页，Example 12.6）。

> 补充说明：Agner VCL 向量类库开源地址 https://github.com/vectorclass（PDF 第 125 页原文引用）。本项目实验不依赖外部库，直接用 intrinsics，避免引入第三方依赖；VCL 用法仅作介绍。

### 12.8 对齐动态分配内存（第 133 页）

- `new`/`malloc` 只保证 8 或 16 字节对齐，旧标准下对齐向量分配要手工处理（PDF 第 133 页）。
- **C++17 起 `new` 类型化数组自动按 `alignof(T)` 对齐**：`__m512 *pp = new __m512[arraysize];`（PDF 第 133 页，Example 12.9）。
- 自定义对齐分配可用 `posix_memalign`/`aligned_alloc`（补充，见实验）。

## 5. 简单示例

SSE2 向量加法的最小例子（PDF 第 121～122 页的风格）：

```cpp
#include <immintrin.h>
#include <cstring>
#include <cstdio>

// Two vectors of 4 floats added with one instruction each (SSE2).
float add4(float *out, const float *a, const float *b) {
    __m128 va = _mm_loadu_ps(a);            // load 4 floats
    __m128 vb = _mm_loadu_ps(b);
    __m128 vc = _mm_add_ps(va, vb);         // one instruction: 4 adds
    _mm_storeu_ps(out, vc);
    return _mm_cvtss_f32(vc);               // low element, prevent elision
}
```

## 6. 未优化代码

带分支的循环，编译器可能不自动向量化（PDF 第 121 页，Example 12.4a）：

```cpp
// Branch in the loop: compiler may or may not vectorize this.
// PDF p121 (Example 12.4a): SelectAddMul
void select_add_mul(short int aa[], const short int bb[],
                    const short int cc[]) {
    for (int i = 0; i < 256; ++i) {
        aa[i] = (bb[i] > 0) ? (cc[i] + 2) : (bb[i] * cc[i]);
    }
}
```

## 7. 优化后代码

SSE2 掩码法（PDF 第 122 页，Example 12.4b 的核心）：

```cpp
#include <immintrin.h>

// Vectorized with SSE2: compare->mask, then AND/ANDNOT/OR to pick.
// PDF p122 (Example 12.4b), 8 elements at a time (int16).
void select_add_mul_sse2(short int aa[], const short int bb[],
                         const short int cc[]) {
    __m128i zero = _mm_setzero_si128();
    __m128i two  = _mm_set1_epi16(2);
    for (int i = 0; i < 256; i += 8) {
        __m128i b  = _mm_loadu_si128((const __m128i*)(bb + i));
        __m128i c  = _mm_loadu_si128((const __m128i*)(cc + i));
        __m128i c2 = _mm_add_epi16(c, two);       // cc[i] + 2
        __m128i bc = _mm_mullo_epi16(b, c);       // bb[i] * cc[i]
        __m128i m  = _mm_cmpgt_epi16(b, zero);    // mask: all 1 if >0
        c2 = _mm_and_si128(c2, m);                // keep c+2 where b>0
        bc = _mm_andnot_si128(m, bc);             // keep b*c elsewhere
        __m128i a  = _mm_or_si128(c2, bc);
        _mm_storeu_si128((__m128i*)(aa + i), a);
    }
}
```

## 8. 为什么会更快

- **向量化**：每轮处理 8 个 int16，指令数约降到标量的 1/8（PDF 第 122 页）。
- **分支消除**：标量版本含分支（可能误预测，惩罚 15-25 周期）；向量掩码法无分支，性能不受数据分布影响（PDF 第 122 页）。
- **可预测性依赖**：分支越不可预测，掩码法相对收益越大（3-7 倍区间，PDF 第 122 页）。

## 9. 如何验证

```bash
# 编译标量与 SSE2 版本
g++ -O3 -std=c++17 select_add_mul.cpp -o select_scalar
g++ -O3 -std=c++17 -msse2 select_add_mul.cpp -o select_sse2

# 运行对比（程序内置随机数据、校验和一致检查、多轮计时）
./select_scalar
./select_sse2

# 确认汇编含向量指令
g++ -O3 -std=c++17 -msse2 -S -masm=intel select_add_mul.cpp -o /tmp/sse2.s
grep -E "paddw|pmullw|pcmpgtw" /tmp/sse2.s | head

# 对比分支预测失败次数（标量 vs 向量）
sudo perf stat -e branch-misses ./select_scalar
```

- 编译命令：`g++ -O3 -std=c++17 -msse2`（本机 g++ 13.3.0；AVX2 版用 `-mavx2`）
- 运行命令：`./select_scalar` / `./select_sse2`
- 校验方法：两个版本对相同输入计算校验和，必须一致（防止错误向量化）
- perf 命令：`sudo perf stat -e branch-misses`（本机需 root）
- 查看汇编：`g++ -O3 -S -masm=intel`，搜 `paddw`/`pmullw`/`pcmpgtw`/`v*` 指令

> 当前环境说明：本机 CPU 支持 SSE4.2/AVX/AVX2，**不支持 AVX-512**。AVX-512 intrinsics 代码会编译（需 `-mavx512f` 等）但在本机运行会非法指令崩溃，必须配合 CPUID 分发（11 笔记）。AVX-512 部分在本项目实验中标"当前环境未验证，仅编译验证"。

## 10. 常见误区

- **误区一：intrinsics 代码 = 汇编，编译器不再优化。** 编译器仍做寄存器分配、CSE、调度（PDF 第 121 页）。
- **误区二：intrinsics 比向量类快。** 性能相同，只是可读性差别（PDF 第 125 页）。
- **误区三：`__m128` 可以随便 load 任意地址。** 对齐版本（`_mm_load_ps`）要求 16 字节对齐；不确定时用 `_mm_loadu_ps`（未对齐版本）。
- **误区四：intrinsic 调用会自动检查 CPU 支持。** 不会；不支持的 CPU 上会崩溃，需自己分发（PDF 第 124 页）。
- **误区五：掩码法永远最优。** 分支高度可预测时标量可能更快；按数据特性实测（PDF 第 122 页）。
- **误区六：AVX 代码能直接和普通代码混用。** 切换有惩罚，需 `_mm256_zeroupper()`（PDF 第 117 页）。

## 11. 实践任务

1. 运行 6/7 节示例，验证 SSE2 版结果与标量一致（校验和），对比耗时。
2. 把 `select_add_mul` 改写为 AVX2 版（`__m256i`，16 个 int16），用 `-mavx2` 编译运行。
3. 把 `sum` 归约用 `_mm_add_ps`/`_mm256_add_ps` 实现（横向相加 `_mm_hadd_ps` 或手动加），对比多累加器标量版。
4. 用 `_mm_min_ps`/`_mm_max_ps` 实现逐元素 min/max，对比标量循环。
5. 实现 RGB 图像亮度处理的向量版本（每像素 3 通道，按 12.9 节方法之一组织数据）。
6. 用 `posix_memalign` 分配 64 字节对齐数组，对比对齐/未对齐 load 的性能（本机对齐惩罚可能很小）。

## 12. 本章总结

- intrinsics 提供显式的向量控制，且编译器仍能继续优化；是"比汇编安全、比自动向量化可控"的中间层。
- 分支向量化的演进：AND/OR 掩码（SSE2）→ blend（SSE4.1）→ mask 指令（AVX-512）。
- 向量类让 intrinsics 代码可读，性能相同；开源首选 VCL。
- C++17 的 `new` 自动对齐类型化数组；自定义对齐用 `posix_memalign`/`aligned_alloc`。
- **用任何指令集前必须检测 CPU 支持**（见 11 笔记）。

## 13. 对应代码

本章对应实验（阶段三实现）：

- `src/13_intrinsics/` —— SSE/AVX2/AVX-512 intrinsics：向量加法、点积、reduction、min/max、RGB/3D 处理
- `src/09_alignment/` —— 对齐内存与向量 load/store
- `src/14_cpu_dispatch/` —— CPUID 检测与运行时选择（衔接 11 笔记）

> 状态：上述实验代码尚未实现（阶段三完成），届时更新本节链接。
