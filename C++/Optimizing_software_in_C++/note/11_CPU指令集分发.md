# 11 CPU指令集分发

> 本笔记对应 PDF 第 13 章 Making critical code in multiple versions for different instruction sets（第 135～143 页），覆盖 13.1～13.7，并衔接 12.1（AVX 过渡惩罚，第 117 页）。

## 1. 本章解决什么问题

用最新指令集编译的程序在旧 CPU 上无法运行。为了让程序**既能在新 CPU 上用尽新指令、又能在旧 CPU 上正常跑**，需要 CPU dispatch（指令集分发）：为关键代码写多个版本，运行时检测 CPU 能力并选择最合适的。

本章回答：

1. 分发策略有哪些？（按指令集而非品牌/型号）
2. 常见陷阱是什么？（正/负列表、未知处理器、虚拟化）
3. 怎么实现？（首次调用分发、GNU ifunc）

核心结论：**按"支持的指令集"而不是"CPU 品牌/型号"做分发；只对最关键的代码做多版本；未知 CPU 应给它最好的兼容分支。**

## 2. 核心概念

| 术语 | 含义 | 出处 |
|------|------|------|
| CPU dispatch | 为不同指令集编译多份关键代码，运行时选择 | PDF 第 135 页 |
| CPUID | 查询 CPU 特性（指令集、缓存大小）的指令 | PDF 第 137 页 |
| 指令集层级 | 80386→SSE→SSE2→…→AVX→AVX2→FMA3→AVX-512 | PDF 第 135 页，表 13.1 |
| 正列表/负列表 | 按"支持的型号"或"需要回避的型号"分发 | PDF 第 137 页 |
| ifunc | Linux/ELF 的加载时分发机制（GNU indirect function） | PDF 第 141 页 |
| 半宽执行单元 | 初代大寄存器 CPU 内部执行单元只有一半宽度 | PDF 第 138 页 |
| `_mm256_zeroupper()` | AVX→非 AVX 过渡前清空 YMM 状态 | PDF 第 117 页 |

## 3. 工作原理

### 3.1 指令集向后兼容（PDF 第 135 页，表 13.1）

`80386 → SSE → SSE2 → SSE3 → SSSE3 → SSE4.1 → SSE4.2 → AVX → AVX2 → FMA3 → AVX512F → AVX512BW/DQ/VL → AVX512-FP16`。按某指令集编译的代码，在支持该指令集或更高指令集的 CPU 上都能跑。

### 3.2 分发时机

- **每次调用**：每次调用都做 switch——简单但耗时；
- **首次调用**：函数指针初始指向 dispatcher，首次调用后改指向正确版本——"不调用就不花时间"；
- **初始化时**：程序启动时设置函数指针——响应时间一致；
- **加载时**：GNU ifunc，程序加载时由 loader 选版本（Linux/FreeBSD）；
- **安装时**：安装程序选 .so/.dll；
- **不同可执行文件**：指令集互不兼容时（如 32/64 位）分开发布（PDF 第 139～140 页）。

### 3.3 为什么"按指令集"而不是"按型号"

CPU 品牌/型号信息不可靠：虚拟化可能伪造型号；新型号未知；型号编号不连续。**只有 CPUID 的特性信息（支持哪些指令集、缓存多大）是可靠的**（PDF 第 137 页）。

## 4. PDF 核心观点

### 13.1 CPU dispatch 策略（第 135～137 页）

- 多版本代码开发/测试/维护成本高，值得做成**可复用库**；只对关键路径做（PDF 第 135 页）。
- 常见陷阱（PDF 第 135～137 页）：
  - 为当前处理器优化而不是未来——从写代码到用户用上要数年，彼时该型号已过时；
  - 按型号而非指令集思考——型号清单又长又难维护；
  - 假设型号数字连续——`N+1` 不一定更好；
  - 未知处理器处理不当——应给"支持该指令集的未知 CPU"最好的分支；
  - 低估更新分发器的成本——只在出现重大新指令集时才加分支；
  - 分支过多——通常**两个分支就够**：最新指令集 + 兼容 5-10 年前 CPU 的版本；
  - 忽视虚拟化——型号/厂商字符串可伪造，指令集特性可靠。
- 解决方案：**少数几个分支，按支持的指令集分发**（PDF 第 137 页）。
- 案例：Mathcad 用的旧 MKL 靠 CPU 家族号判断，把 CPUID 伪装成老 Pentium 4 后某些任务快了 33%（PDF 第 137 页）。
- 只对最关键的代码做分发，最好隔离成独立函数库（PDF 第 137 页）。

### 13.2 按型号分发（第 137～138 页）

- 某个型号上某实现特别差时，用**负列表**（回避的型号）而非正列表（优化的型号）——负列表几乎不用更新（新型号通常更好）（PDF 第 137～138 页）。

### 13.3 困难案例（第 138～139 页）

- **strlen**：位扫描指令（BSF）在某些老 CPU 上慢，但不值得为此做专门版本（每调用才一次）（PDF 第 138 页）。
- **半宽执行单元**：初代 256 位 CPU 把 256 位操作拆成两个 128 位；可"用 AVX 只在支持 AVX2 时"或负列表回避（PDF 第 138 页）。
- **高精度数学**：ADC 进位标志链依赖"部分标志停顿"，Intel/AMD 表现不同——少数值得按品牌分发的案例（PDF 第 138 页）。
- **内存拷贝**：memcpy 情况太多，交给有 CPU dispatch 的标准库即可（PDF 第 139 页）。
- 终极方案：运行时测速选最优版本；但频率动态变化、测量不稳定，需交替多次测（PDF 第 139 页）。

### 13.4 测试与维护（第 139 页）

- 测速要在各自目标 CPU 上测；**正确性测试只需最高指令集的 CPU**（低版本代码可在高版本 CPU 上跑）（PDF 第 139 页）。
- 代码里应有**覆盖分发、强制运行任意分支**的测试开关（PDF 第 139 页）。

### 13.5 实现（第 139～141 页）

- 六种分发时机（见 3.2）（PDF 第 139～140 页）。
- 用 asmlib 的 `InstructionSet()` 或 VCL 的 `instrset_detect()` 检测指令集（PDF 第 140 页）。
- **首次调用分发示例**（Example 13.1，PDF 第 140 页）：函数指针 `CriticalFunction` 初始指向 `CriticalFunction_Dispatch`，首次调用按 `level` 改写为 `_AVX`/`_SSE2`/`_386` 版本。
- 不同版本可放不同模块、各自按对应指令集编译（PDF 第 141 页）。

### 13.6 Linux 加载时分发（第 141～142 页）

- **GNU ifunc**：`__attribute__((ifunc("dispatcher")))`。程序加载时调用 dispatcher，把返回的函数指针放进 PLT（PDF 第 141～142 页）。
- 要求：ELF 平台（Linux/FreeBSD）、binutils 2.20+、glibc 2.11+（PDF 第 142 页）。
- 注意：dispatcher 在任何构造函数之前、程序启动前调用，**不能依赖任何已初始化状态或环境变量**；即使函数从未被调用，dispatcher 也会执行（PDF 第 142 页）。
- Example 13.2 展示基于 `instrset_detect()` 的 ifunc 实现（PDF 第 142 页）。

### 13.7 Intel 编译器中的分发（第 143 页）

- Intel icc（Classic，已停维护）支持 `-axAVX` 自动多版本；但**只对 Intel CPU 公平**，AMD/VIA 上性能差甚至崩溃（PDF 第 143 页）。
- LLVM 版 icx 不分发用户代码、只看指令集，行为与 Clang 几乎相同（PDF 第 143 页）。
- **结论：不要在面向非 Intel CPU 的软件里用 Classic icc**（PDF 第 143 页）。

### 衔接 12.1（AVX 过渡惩罚，第 117 页）

- AVX 代码离开时若可能回到非 AVX 代码，先调 `_mm256_zeroupper()`，否则有状态切换惩罚（PDF 第 117 页）。

## 5. 简单示例

首次调用分发（PDF 第 140 页，Example 13.1 的结构）：

```cpp
// Dispatch-on-first-call pattern (PDF p140, Example 13.1).
// A function pointer initially points to the dispatcher; after the first
// call it points to the best implementation for this CPU.

typedef int (*fn_t)(int);

int critical_scalar(int x);
int critical_sse2(int x);   // compiled with -msse2
int critical_avx2(int x);   // compiled with -mavx2

int critical_dispatch(int x);       // defined below
fn_t critical = &critical_dispatch; // entry point

// runtime detection: see src/common/cpu_info (CPUID-based)
int cpu_level();                     // 0=generic, 2=SSE2, 8=AVX2

int critical_dispatch(int x) {
    int level = cpu_level();
    if (level >= 8) critical = &critical_avx2;
    else if (level >= 2) critical = &critical_sse2;
    else critical = &critical_scalar;
    return critical(x);              // forward to chosen version
}
```

## 6. 未优化代码

只写一个 AVX2 版本、不检测 CPU（PDF 第 135 页"失去旧 CPU 兼容"）：

```cpp
// Compiles only with -mavx2; crashes on any CPU without AVX2.
// PDF p124: "The program will crash if it contains an instruction
// that the CPU does not support."
void process(float *a, float *b, int n) {
    // AVX2 intrinsics here...
    (void)a; (void)b; (void)n;
}
```

## 7. 优化后代码

```cpp
// Three versions (scalar / SSE2 / AVX2) selected at runtime by CPUID.
// Correctness can be verified by forcing each branch (PDF p139).
// Scalar version runs everywhere; AVX2 version only where supported.
```

实现细节与完整可运行代码见 `src/14_cpu_dispatch/`（阶段三）。

## 8. 为什么会更快

- **指令集优势**：AVX2 每次向量操作处理 8 个 float，是 SSE2（4 个）的两倍、标量的 8 倍（PDF 第 73、115 页）。
- **兼容性**：scalar 版本保证在旧 CPU 上仍正确；分发只花在首次调用的一次开销（PDF 第 139 页）。
- **可移植性 vs 性能的平衡**：多版本代码同时拿到"新 CPU 的最大性能"和"旧 CPU 的可用性"。

## 9. 如何验证

```bash
# 编译各版本（separate objects with different -m flags）
g++ -O3 -std=c++17 -mavx2 -c critical_avx2.cpp -o critical_avx2.o
g++ -O3 -std=c++17 -msse2 -c critical_sse2.cpp -o critical_sse2.o
g++ -O3 -std=c++17 -c critical_scalar.cpp -o critical_scalar.o
g++ -O3 -std=c++17 critical_main.cpp critical_*.o -o dispatch_demo

# 运行：程序打印检测到的指令集与选择的版本
./dispatch_demo

# 验证正确性：三种实现结果一致（程序内置校验和对比）
# 强制各分支（分发测试开关，PDF p139）
./dispatch_demo --force scalar
./dispatch_demo --force sse2
./dispatch_demo --force avx2
```

- 编译命令：各版本用不同 `-m` 选项编译，再链接（本机 g++ 13.3.0）
- 运行命令：`./dispatch_demo`；`--force <branch>` 强制分支
- 本机 CPU：i9-14900HX 支持 AVX2、无 AVX-512；`--force avx512` 若存在会崩溃（预期）
- 检测代码：`src/common/cpu_info.cpp`（CPUID 实现，阶段三）

## 10. 常见误区

- **误区一：按 CPU 型号分发。** 应看指令集；型号信息不可靠（虚拟化/未来型号）（PDF 第 136～137 页）。
- **误区二：未知 CPU 走最差分支。** 应给"支持对应指令集"的未知 CPU 最好分支（PDF 第 136 页）。
- **误区三：为每个新 CPU 型号加分支。** 两个分支通常足够；否则维护成本失控（PDF 第 136 页）。
- **误区四：只测最高版本的正确性。** 应能强制运行每个分支验证（PDF 第 139 页）。
- **误区五：ifunc 里能随便用全局状态。** 它在构造函数前运行，只能靠 CPUID 等自给自足（PDF 第 142 页）。
- **误区六：AVX2 代码和标量代码随意互相调用。** 需 `_mm256_zeroupper()`（PDF 第 117 页）。

## 11. 实践任务

1. 实现 `src/common/cpu_info.cpp` 的 CPUID 检测，打印本机支持的指令集。
2. 实现 scalar/SSE2/AVX2 三个版本的向量加法，用首次调用分发，验证三个版本结果一致。
3. 给程序加 `--force` 开关强制运行任意分支（PDF 第 139 页的测试建议）。
4. 尝试把 AVX2 分支在本机运行，并用 `--force avx512`（若无 AVX-512 应捕获到非法指令崩溃，说明保护的必要性）。
5. 用 GNU ifunc（`__attribute__((ifunc(...)))`）改写一个简单函数，确认加载时选对版本（需 ELF 平台，Linux 可用）。

## 12. 本章总结

- CPU dispatch：多版本关键代码 + 运行时按指令集选择。
- 分发依据是指令集特性，不是品牌/型号；陷阱包括正列表、未知 CPU、虚拟化。
- 实现方式：首次调用分发最常用；Linux 可用 ifunc 在加载时完成。
- 只对最关键的代码做多版本；正确性可用"强制分支"开关验证。
- 本机（i9-14900HX）实践上限是 AVX2，AVX-512 分支只能编译 + 理论验证。

## 13. 对应代码

本章对应实验（阶段三实现）：

- `src/14_cpu_dispatch/` —— CPUID 检测、scalar/SSE/AVX2 版本、运行时选择、`--force` 开关
- `src/common/cpu_info.h/.cpp` —— 共享的 CPU 检测模块
- `src/13_intrinsics/` —— 各指令集的实现版本（被分发器调用）

> 状态：上述实验代码尚未实现（阶段三完成），届时更新本节链接。
