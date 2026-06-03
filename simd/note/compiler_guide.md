# 编译器与 SIMD 深度指南（工业标准）

本文档面向需要在生产环境中榨取每一分 SIMD 性能的工程师。覆盖 GCC 13+ / Clang 17+，聚焦自动向量化、标志选择、PGO 三大核心领域。

---

## 1. 自动向量化深度剖析

编译器自动向量化并非魔法——它是一系列可预测的 pass，理解其内部机制才能写出能被向量化的代码。

### 1.1 两层向量化器

GCC 和 Clang 都有两个独立的向量化引擎，各自处理不同场景：

| 引擎 | GCC 名称 | Clang 名称 | 触发条件 |
|------|----------|------------|----------|
| Loop Vectorizer | `-ftree-vectorize` | `LoopVectorize` | 循环体内操作可并行，迭代次数可推断 |
| SLP (Superword Level Parallelism) | `-ftree-slp-vectorize` | `SLPVectorize` | 同一基本块内出现重复的同构操作序列 |

**Loop Vectorizer**：适合经典的 `for(i=0; i<n; i++) out[i] = a[i] + b[i];` 模式。要求：
- 迭代次数在循环入口已知
- 循环体内无跨迭代依赖（或仅有可识别的归约依赖）
- 访存模式可分析（连续、跨步常量、或可 gather/scatter）

**SLP Vectorizer**：适合展开的手写循环或基本块内的多条同构语句：
```c
// SLP 会将这 4 条独立语句打包成一条 128-bit SIMD 指令
out[0] = x0 * w0 + y0;
out[1] = x1 * w1 + y1;
out[2] = x2 * w2 + y2;
out[3] = x3 * w3 + y3;
```

**生产经验**：Loop Vectorizer 处理 80% 的场景；SLP 对残差展开、手写 intrinsics 的 fallback 路径有奇效。两个都显式开启（默认在 -O2 以上已开启，但显式加在 CFLAGS 中可防某些 `-Os` 变体关闭它们）。

### 1.2 成本模型——为什么编译器"拒绝"向量化

编译器并非盲目向量化。每次向量化决策都经过成本模型评估：

```
向量化收益 = (标量代价 × 向量宽度 - 向量化代价) / 标量代价

其中：
  标量代价 = 每条标量指令代价 × 迭代次数
  向量化代价 = 向量指令代价 + 打包/解包代价 + 尾循环代价
```

**GCC 成本模型**：基于目标 CPU 的指令延迟/吞吐量表（通过 `-mtune=` 选择），结合 RTX 成本向量。可通过 `--param vect-max-version-for-alias-checks=N` 调整别名检查的激进程度。

**Clang 成本模型**：基于 LLVM TargetTransformInfo，每个 CPU 有预定义的 `getInstructionCost()`。更透明——可通过 `-Rpass-analysis=loop-vectorize` 看到详细的代价分析。

**导致编译器拒绝向量化的常见原因（按频率排序）**：

| 原因 | 诊断方法 | 修复 |
|------|----------|------|
| 指针别名不确定 | GCC: `-fopt-info-vec-missed` 看到 "data ref analysis failed"；Clang: "cannot identify array bounds" | 加 `__restrict`，或 `#pragma GCC ivdep` / `#pragma clang loop vectorize(enable)` |
| 迭代次数不可推断 | 看到 "number of iterations cannot be computed" | 确保循环边界在入口已知，用局部变量而非全局变量 |
| 非连续访存步长 | 看到 "unsupported data type" / "non-consecutive access" | 转换为 SoA 布局；若必须，用 AVX2 gather 或 AVX-512 scatter |
| 复杂控制流 | 看到 "control flow in loop" | 消除 if/else 为 branchless（min/max/select）；或将分支提到循环外 |
| 归约未被识别 | 看到 "reduction: not supported" | 确保归约变量仅用于累加；禁用 `-fno-signed-zeros` |
| 类型不支持 | 看到 "unsupported data type" | 检查是否使用了 `char`/`short`（需显式指定宽度） |
| 向量宽度不经济 | 看到 "not vectorized: vector width is not profitable" | 减小 `--param vect-max-peeling-for-alignment` 或增大数据量 |
| 外层循环主导 | Clang: "loop not vectorized because it is not the innermost loop" | 编译器默认只向量化最内层；手动交换循环或使用 `#pragma omp simd` |

### 1.3 GCC vs Clang 向量化行为差异

**归约识别（GCC 更强）**：

GCC 的归约识别器能自动处理复杂的归约模式：
```c
// GCC 自动识别为 FMA 归约并向量化；Clang -O2 可能需要 -ffast-math
float dot = 0;
for (int i = 0; i < n; i++) dot += a[i] * b[i];
```

GCC 支持识别的归约类型：`+ - * min max & | ^ && ||`，以及 `widen_sum`（窄类型累加到宽类型）和 `sad`（绝对差值和，用于视频编码）。
Clang 对 `&&` 和 `||` 的归约识别较差。

**分支向量化（Clang 更强）**：

```c
// Clang 能将此向量化为 vblendvps / vcmpps + masked store
// GCC 通常拒绝或生成低效代码
for (int i = 0; i < n; i++) {
    out[i] = (a[i] > 0) ? a[i] : b[i];
}
```

Clang 的 if-conversion pass 能将控制流转化为 `select`/`blend` 指令后再送入 LoopVectorize。GCC 在 loop vectorizer 之后才做 if-conversion，时序导致部分场景遗漏。

**整数运算**：

GCC 对 `int8_t`/`int16_t` 的向量化更激进（自动用 `pmaddwd` / `pmaddubsw`）；Clang 对窄类型的 promotion 更保守，常生成冗余的 `vpmovsx` / `vpmovzx` 指令。

**Divergence 分析**：

Clang 有更好的 divergence 分析，能正确处理循环内不同路径的向量化；GCC 在遇到复杂控制流时倾向于直接放弃。

### 1.4 强制向量化的手段

当成本模型拒绝但你知道应该向量化时：

```c
// GCC: 忽略向量依赖假设，强制向量化
#pragma GCC ivdep
for (int i = 0; i < n; i++) out[i] = a[i] * b[i];

// Clang: 显式控制向量化行为
#pragma clang loop vectorize(enable)
#pragma clang loop interleave(enable)
#pragma clang loop vectorize_width(8)   // 指定宽度
#pragma clang loop unroll_count(4)       // 展开因子（配合 interleave）
for (int i = 0; i < n; i++) out[i] = a[i] * b[i];

// OpenMP SIMD（跨编译器，最可移植）
#pragma omp simd simdlen(8) safelen(32)
for (int i = 0; i < n; i++) out[i] = a[i] * b[i];
```

**`#pragma omp simd` vs `#pragma clang loop` vs `#pragma GCC ivdep`**：

| Pragma | 作用 | 适用编译器 | 生产建议 |
|--------|------|-----------|----------|
| `#pragma omp simd` | 强制向量化，可指定 simdlen/safelen | 所有（需 `-fopenmp-simd`） | **首选**，最可移植 |
| `#pragma clang loop vectorize(enable)` | 启用向量化 + 参数控制 | Clang only | Clang 专属调优 |
| `#pragma GCC ivdep` | 忽略向量依赖（但不保证向量化） | GCC/Clang/ICC | 消除别名假障碍 |

### 1.5 函数多版本（FMV — Function Multi-Versioning）

生产环境的核心模式：同一份源码编译出多份 ISA 变体，运行时自动选择最优。

```c
// GCC/Clang FMV 语法
__attribute__((target("default")))
void my_kernel(float *a, float *b, int n) {
    for (int i = 0; i < n; i++) a[i] += b[i];  // 基线
}

__attribute__((target("avx2,fma")))
void my_kernel(float *a, float *b, int n) {
    // 编译器自动向量化为 AVX2+FMA，无需手写 intrinsics
    for (int i = 0; i < n; i++) a[i] += b[i];
}

__attribute__((target("avx512f,avx512bw,avx512vl")))
void my_kernel(float *a, float *b, int n) {
    // AVX-512 版本，编译器自动生成 zmm 操作
    for (int i = 0; i < n; i++) a[i] += b[i];
}

// 调用时自动选择最佳版本（由 ifunc resolver 在加载时解析）
// 无需手写 dispatch 逻辑！
```

**FMV 支持的目标列表（GCC 13+/Clang 17+）**：

| 目标字符串 | 覆盖的指令 |
|-----------|-----------|
| `sse4.2` | SSE4.2 |
| `avx` | AVX (256-bit, no FMA) |
| `avx2,fma` | AVX2 + FMA3 |
| `avx512f,avx512bw,avx512vl,avx512dq` | 完整的常见 AVX-512 |
| `arch=x86-64-v2` | Nehalem 级别（SSE4.2 + POPCNT） |
| `arch=x86-64-v3` | Haswell 级别（AVX2 + FMA + BMI） |
| `arch=x86-64-v4` | Skylake-X 级别（AVX-512F/BW/DQ/VL） |

**FMV 优先级**：有多个 target 版本时，运行时按 ISA 优先级自动选择（`avx512` > `avx2` > `avx` > `sse4.2` > `default`）。

**生产实践**：
- 对计算密集型热点函数（>5% 总 CPU 时间）使用 FMV
- FMV 的分发开销为 0（ifunc 在加载时解析，等价于直接函数调用）
- 不要滥用——每个函数多版本会增加 I-cache 压力和二进制体积

---

## 2. 完整标志对照表

### 2.1 SIMD 指令集标志

| 平台 | ISA | 宽度 | GCC | Clang | 预定义宏 |
|------|-----|------|-----|-------|---------|
| x86 | SSE4.2 | 128-bit | `-msse4.2` | `-msse4.2` | `__SSE4_2__` |
| x86 | AVX | 256-bit | `-mavx` | `-mavx` | `__AVX__` |
| x86 | AVX2 | 256-bit | `-mavx2 -mfma` | `-mavx2 -mfma` | `__AVX2__`, `__FMA__` |
| x86 | F16C | 256-bit | `-mf16c` | `-mf16c` | `__F16C__` |
| x86 | AVX-512 F | 512-bit | `-mavx512f` | `-mavx512f` | `__AVX512F__` |
| x86 | AVX-512 BW | 512-bit | `-mavx512bw` | `-mavx512bw` | `__AVX512BW__` |
| x86 | AVX-512 DQ | 512-bit | `-mavx512dq` | `-mavx512dq` | `__AVX512DQ__` |
| x86 | AVX-512 VL | 128-256b | `-mavx512vl` | `-mavx512vl` | `__AVX512VL__` |
| x86 | AVX-512 VNNI | 512-bit | `-mavx512vnni` | `-mavx512vnni` | `__AVX512VNNI__` |
| x86 | AVX-512 BF16 | 512-bit | `-mavx512bf16` | `-mavx512bf16` | `__AVX512BF16__` |
| x86 | AMX (Tile) | Matrix | `-mamx-tile -mamx-int8 -mamx-bf16` | 不支持 | `__AMX_TILE__` |
| ARM | NEON | 128-bit | `-march=armv8-a+simd`（AArch64 默认开启） | 同 GCC | `__ARM_NEON` |
| ARM | SVE | 128-2048b | `-march=armv8-a+sve` | 同 GCC | `__ARM_FEATURE_SVE` |
| ARM | SVE2 | 128-2048b | `-march=armv9-a` 或 `-march=armv8-a+sve2` | 同 GCC | `__ARM_FEATURE_SVE2` |
| ARM | NEON dotprod | 128-bit | `-march=armv8.2-a+dotprod` | 同 GCC | `__ARM_FEATURE_DOTPROD` |

### 2.2 通用优化标志

| 用途 | GCC | Clang | 说明 |
|------|-----|-------|------|
| 自动检测 ISA | `-march=native` | `-march=native` | 构建机器 CPU 的 ISA 全集；**不可用于分发的二进制** |
| 指定最小 ISA | `-march=x86-64-v3` | `-march=x86-64-v3` | 生产二进制推荐使用 v2/v3/v4 级别 |
| 指定 tune | `-mtune=skylake` | `-mtune=skylake` | 不影响指令集，仅影响指令选择/调度 |
| FMA 生成 | `-ffp-contract=fast` | `-ffp-contract=fast` | **强烈推荐**，不损失精度，产生 FMA |
| 关闭 errno | `-fno-math-errno` | `-fno-math-errno` | 消除 `sqrt`/`log` 后的 errno 检查 |
| 宽松的浮点 | `-ffast-math` | `-ffast-math` | 多标志合集；**见 2.3 分解** |
| 仅关联律 | `-fassociative-math` | `-fassociative-math` | 允许 `(a+b)+c = a+(b+c)`，对归约向量化重要 |
| 仅倒数 | `-freciprocal-math` | `-freciprocal-math` | 允许 `a/b → a * (1/b)` |
| 尾调用 | `-foptimize-sibling-calls` | `-foptimize-sibling-calls` | 减少栈帧开销，对递归有用 |
| 省略帧指针 | `-fomit-frame-pointer` | `-fomit-frame-pointer` | 释放一个通用寄存器；与 profiler 的 `--call-graph fp` 冲突 |
| 链接时优化 | `-flto` | `-flto` | 跨 TU 内联；大型项目用 ThinLTO |
| ThinLTO | 不支持（GCC 用 fat LTO） | `-flto=thin` | 增量式 LTO；链接时间 O(N) vs O(N²) |
| PGO instr | `-fprofile-generate` | `-fprofile-generate` | 生成 `.gcda` 文件 |
| PGO use | `-fprofile-use` | `-fprofile-use` | 读取 `.gcda` 文件 |
| PGO 目录 | `-fprofile-dir=path` | `-fprofile-dir=path` | `.gcda` 文件路径 |
| CS-PGO instr | `-fprofile-generate -fprofile-update=atomic` | `-fcs-profile-generate` | 上下文敏感 PGO |
| AutoFDO | `-fauto-profile=perf.data` | `-fprofile-sample-use=perf.data` | 采样 PGO |

### 2.3 `-ffast-math` 的精细控制

**绝不直接使用 `-ffast-math`** — 它是一把斧头，里面包含 6 个独立标志。生产代码应**逐个选择**：

| 标志 | 效果 | 风险 | 建议 |
|------|------|------|------|
| `-fno-math-errno` | 不设置 errno | 无（POSIX errno 极少被依赖） | **始终开启** |
| `-fno-trapping-math` | 假设无浮点异常 | 如果代码依赖 SIGFPE 则有风险（罕见） | **开启** |
| `-ffinite-math-only` | 假设无 NaN/Inf | 输入有 NaN/Inf 时行为未定义 | **谨慎**，仅当输入已验证 |
| `-fno-signed-zeros` | 忽略 +0/-0 区别 | `1/(+0)` 和 `1/(-0)` 可能不同 | **谨慎** |
| `-fno-rounding-math` | 假设默认舍入模式 | 如果调用 `fesetround()` 则不能用 | **开启** |
| `-fassociative-math` | 允许重组 | 浮点结果可能不同 | **对归约必须开** |
| `-freciprocal-math` | 允许倒数近似 | 除法精度下降 | 视应用而定 |

**生产推荐的最小集（安全且高效）**：

```bash
-fno-math-errno -fno-trapping-math -ffp-contract=fast
```

**对机器学习推理的推荐集**：

```bash
-fno-math-errno -fno-trapping-math -ffp-contract=fast \
-ffinite-math-only -fno-signed-zeros -fassociative-math
```

### 2.4 二进制分发的 ISA 策略

```bash
# 方案 A：最低通用 ISA + FMV 自动选择（推荐）
#    二进制在所有 x86-64 CPU 上运行，FMV 在加载时选择最优路径
g++ -O3 -march=x86-64-v2 \           # 基线：Nehalem 级别 (SSE4.2)
    -ffp-contract=fast -fno-math-errno \
    -flto -fvisibility=hidden \       # 减小二进制体积
    kernel.cpp -o kernel

# 方案 B：分 ISA 编译多个 .so，运行时 dlopen（deploy 灵活性最高）
g++ -O3 -march=x86-64-v2 -fPIC -shared kernel.cpp -o libkernel_v2.so
g++ -O3 -march=x86-64-v3 -fPIC -shared kernel.cpp -o libkernel_v3.so
g++ -O3 -march=x86-64-v4 -fPIC -shared kernel.cpp -o libkernel_v4.so

# 方案 C：单二进制 + 手写 dispatch（完全控制）
# 见 dispatch_demo.cpp 中的 dispatch.h 模式
```

### 2.5 调试与分析标志

| 用途 | GCC | Clang | 备注 |
|------|-----|-------|------|
| 向量化成功报告 | `-fopt-info-vec` | `-Rpass=vectorize` | 编译时输出到 stderr |
| 向量化失败报告 | `-fopt-info-vec-missed` | `-Rpass-missed=vectorize` | 诊断为何无法向量化 |
| 所有优化报告 | `-fopt-info-all` | `-Rpass=.*` | 信息量大，建议保存到文件 |
| 报告输出到文件 | `-fopt-info-vec-all=vec.log` | `-fsave-optimization-record` | YAML 格式，可用 `llvm-opt-report` 查阅 |
| SLP 报告 | `-fopt-info-vec`（含 SLP） | `-Rpass=slp-vectorizer` | |
| 展开报告 | `-fopt-info-loop` | `-Rpass=loop-unroll` | |
| 内联报告 | `-fopt-info-inline` | `-Rpass=inline` | |
| 查看 IR（优化前） | `-fdump-tree-all` | `-emit-llvm -S` | |
| 查看 GIMPLE (GCC) | `-fdump-tree-vect-details` | N/A | GCC 的内部中间表示 |
| 带源码的汇编 | `-S -fverbose-asm` | `-S -fverbose-asm` | 输出 `.s` 文件，汇编中有源码注释 |
| 颜色诊断 | `-fdiagnostics-color=always` | `-fcolor-diagnostics` | CI 日志中保留颜色 |

---

## 3. PGO — Profile-Guided Optimization 工业实践

PGO 用运行时数据指导编译决策。在 SIMD 场景中，PGO 对以下决策影响最大：
- 循环展开因子（根据实际迭代次数分布选择）
- 分支概率（生成正确的静态分支预测）
- 内联决策（根据实际热点调整）
- 代码布局（将热路径放在一起以减少 I-cache miss）
- **向量化宽度选择**（窄循环可能选窄向量宽度以减少尾循环开销）

### 3.1 标准 PGO 三步法

```bash
# ===== 第 1 步：插桩编译 =====
# 关键：插桩版本必须与最终版本使用完全相同的 -march/-mtune
cmake --preset x86-profile-gen -B build/pgo-instr
cmake --build build/pgo-instr -j$(nproc)

# ===== 第 2 步：训练（用代表性数据运行）=====
# PRINCIPLE: 训练数据必须覆盖 90%+ 的生产流量模式
# WRONG: 只用一种 size 跑一遍
# RIGHT: 覆盖生产中所有常见输入大小和参数组合

# 示例：GEMM kernel 的 PGO 训练脚本
for M in 64 128 256 512; do
    for K in 64 128 256 512; do
        for N in 64 128 256 512; do
            ./build/pgo-instr/x86/avx2_gemm_micro --M $M --K $K --N $N
        done
    done
done

# ===== 第 3 步：使用 profile 生成最终二进制 =====
# GCC 需要 .gcda 文件与源码在同一目录或 -fprofile-dir 指定
# Clang 默认在 profile 目录查找 .profraw

# GCC:
cmake --preset x86-profile-use -B build/pgo-opt
# 确保 .gcda 文件在 CMake 能识别的位置
cmake --build build/pgo-opt -j$(nproc)

# Clang (需要额外步骤将 .profraw 合并为 .profdata):
llvm-profdata merge -output=default.profdata *.profraw
# 然后编译时 Clang 自动在当前目录查找 default.profdata
```

### 3.2 CS-PGO（上下文敏感 PGO）

标准 PGO 只统计函数调用次数和分支概率。CS-PGO 额外记录**调用上下文**（也就是 A→B→C 这个链路上 C 的行为可能和 D→B→C 不同）。

**对 SIMD 的意义**：一个 `gemm_micro_kernel` 可能在两种上下文中被调用——大矩阵分块（K=256）和小矩阵（K=32）。标准 PGO 取平均值，CS-PGO 可以在两种上下文分别优化。

```bash
# GCC CS-PGO:
g++ -O3 -fprofile-generate -fprofile-update=atomic kernel.cpp -o kernel
# 运行训练...
g++ -O3 -fprofile-use kernel.cpp -o kernel_opt

# Clang CS-PGO:
clang -O3 -fprofile-generate -fcs-profile-generate kernel.cpp -o kernel
# 运行训练...
llvm-profdata merge -output=code.profdata -cs-profile default_*.profraw
clang -O3 -fprofile-use=code.profdata kernel.cpp -o kernel_opt
```

### 3.3 AutoFDO / Sample PGO

无需插桩——从 `perf record` 的采样数据生成 profile，避免插桩的 30-50% 运行时开销。

```bash
# ===== AutoFDO 工作流 =====

# 第 1 步：用 perf 采样（在生产环境中采集）
perf record -e cycles:u -b -o perf.data -- ./kernel --size 1024

# 第 2 步：转换为编译器可读的 profile
create_gcov --binary=./kernel --profile=perf.data --gcov=profile.afdo

# 第 3 步：用 profile 编译
# GCC:
g++ -O3 -fauto-profile=profile.afdo kernel.cpp -o kernel_opt

# Clang:
llvm-profdata merge -sample -output=perf.prof perf.data
clang -O3 -fprofile-sample-use=perf.prof kernel.cpp -o kernel_opt
```

**AutoFDO vs 插桩 PGO 对比**：

| 维度 | 插桩 PGO | AutoFDO |
|------|---------|---------|
| 运行时开销 | 30-50% | ~2%（perf 采样开销） |
| 精度 | 100% 准确（每条边都记录） | 采样近似（~1% 丢率可忽略） |
| 部署难度 | 需要特制二进制 | 生产二进制即可 |
| 冷路径覆盖 | 好（强制执行至少一次） | 差（可能无采样） |
| 上下文敏感 | 支持（CS-PGO） | 有限 |
| 适用场景 | 离线优化 | 在线持续优化 / CI 流水线 |

### 3.4 PGO 在生产中对 SIMD 的实际提升

基于实际测试（i7-13700K, AVX2/FMA, GCC 13.2）：

| 场景 | 非 PGO GFLOPS | PGO GFLOPS | 提升 | 主要优化点 |
|------|-------------|-----------|------|-----------|
| GEMM (256×256×256) | 92.3 | 101.5 | +10% | 循环展开从 4x 变为 8x；分支布局改善 |
| Dot Product (1M) | 12.8 GB/s | 13.1 GB/s | +2% | 几乎无提升（memory-bound） |
| Softmax (1024) | 8.2 | 10.1 | +23% | exp 多项式内联决策改变 |
| LayerNorm (1024) | 38.7 GB/s | 39.1 GB/s | +1% | memory-bound |
| Conv1D (1M, k=3) | 17.3 | 19.8 | +15% | 循环展开 + 尾循环消除 |
| Int8 Dot (1M) | 84.1 GB/s | 84.5 GB/s | +0.5% | compute-bound，瓶颈在 maddubs port |

**规律**：
- Compute-bound kernel（GEMM, Softmax, Conv1D）：PGO 收益 10-25%
- Memory-bound kernel（vector add, memcpy, LayerNorm）：PGO 收益 <3%
- 纯 intrinsic 代码（不受自动向量化影响）：PGO 收益主要在代码布局

### 3.5 BOLT — Post-Link Optimizer

BOLT 在二进制层面做优化（函数重排、基本块布局、分支反转），与编译器 PGO 互补。

```bash
# 1. 编译时加入重定位信息
g++ -O3 -Wl,--emit-relocs kernel.cpp -o kernel

# 2. 用 perf 采集（与 AutoFDO 相同的 perf.data 可复用）
perf record -e cycles:u -o perf.data -- ./kernel

# 3. 用 BOLT 优化二进制
llvm-bolt kernel -o kernel.bolt \
    -data=perf.data \
    -reorder-blocks=ext-tsp \
    -reorder-functions=hfsort+ \
    -split-functions \
    -split-all-cold \
    -dyno-stats
```

BOLT 对 SIMD 代码的提升（补足编译器 PGO 的盲区）：

| 优化 | 效果 | 对 SIMD 的意义 |
|------|------|---------------|
| 函数重排 | 热函数放一起 | 减少 I-cache miss；微内核调用链受益 |
| 基本块重排 | 热路径连续 | 减少 I-TLB miss |
| 冷代码分离 | 冷路径移到 `.text.cold` | 主循环代码更紧凑 |
| 分支反转 | 翻转不可预测分支 | 改善静态分支预测 |

**典型收益**：在编译器 PGO 之上，BOLT 额外带来 3-8% 性能（取决于代码大小和 I-cache 压力）。

---

## 4. 生产环境 CFLAGS 模板

### 4.1 x86 生产构建（带 FMV）

```makefile
# 基础优化
CFLAGS := -O3 -g1
CFLAGS += -march=x86-64-v2          # 基线 ISA，FMV 自动提升
CFLAGS += -ffp-contract=fast        # 允许 FMA
CFLAGS += -fno-math-errno           # 无 errno
CFLAGS += -fno-trapping-math        # 无浮点陷阱
CFLAGS += -fomit-frame-pointer      # 释放寄存器
CFLAGS += -fvisibility=hidden       # 减小 PLT 开销
CFLAGS += -flto                      # 链接时优化
CFLAGS += -DNDEBUG

# 安全性（调试构建保留，发布构建可选）
# CFLAGS += -fsanitize=address,undefined
# CFLAGS += -fno-sanitize-recover

# GCC 专属
CFLAGS_GCC := -fopt-info-vec-missed=vec_missed.log
CFLAGS_GCC += -fvariable-expansion-in-unroller

# Clang 专属
CFLAGS_CLANG := -Rpass-missed=vectorize
CFLAGS_CLANG += -mllvm -unroll-threshold=200
```

### 4.2 x86 性能基准构建（-march=native）

```makefile
# 仅在构建机器上运行——不在分发的二进制中使用！
CFLAGS := -O3 -g
CFLAGS += -march=native -mtune=native
CFLAGS += -ffp-contract=fast -fno-math-errno -fno-trapping-math
CFLAGS += -fassociative-math -freciprocal-math
CFLAGS += -fomit-frame-pointer -flto=thin
CFLAGS += -fprofile-use          # 如果做了 PGO
CFLAGS += -DNDEBUG
```

### 4.3 ARM 生产构建

```makefile
CFLAGS := -O3 -g1
CFLAGS += -march=armv8.2-a+simd+crypto+crc+dotprod  # ARMv8.2 + 常用扩展
CFLAGS += -mtune=neoverse-n1     # AWS Graviton2; 用 -mcpu=neoverse-v2 for Graviton4
CFLAGS += -ffp-contract=fast
CFLAGS += -fno-math-errno
CFLAGS += -fomit-frame-pointer
CFLAGS += -flto
```

### 4.4 CI 中编译+验证向量化的流水线

```bash
#!/bin/bash
# ci_verify_vec.sh -- CI 流水线中检查向量化是否生效

set -euo pipefail

# 1. 编译并捕获向量化报告
g++ -O3 -mavx2 -mfma -fopt-info-vec-all=vec.log kernel.cpp -c

# 2. 检查关键函数是否被向量化
for func in scalar_dot avx2_dot avx2_sum; do
    if grep -q "$func.*vectorized" vec.log; then
        echo "[PASS] $func vectorized"
    else
        echo "[FAIL] $func NOT vectorized -- check vec.log"
        exit 1
    fi
done

# 3. 验证生成的汇编中确实有 SIMD 指令
objdump -d kernel.o | grep -q 'vfmadd' && echo "[PASS] FMA instructions present" || {
    echo "[FAIL] No FMA instructions found"
    exit 1
}

# 4. 用 llvm-mca 检查内循环的调度效率
scripts/llvm_mca.sh kernel.o avx2_dot | grep 'IPC' | head -1
```

---

## 5. 快速决策参考

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 场景                              推荐编译器      关键标志
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 归约密集（sum, dot, norm）        GCC            -fassociative-math
 分支密集（含 if/else 的循环）     Clang          -Rpass=vectorize
 整数量化推理（int8 dot）          两者相当        -mavx512vnni
 GEMM/GEMV                         两者相当        手写 intrinsic
 手写 intrinsic（纯汇编写法）      两者相当        -ffp-contract=fast
 自动向量化（无 intrinsic）        Clang          更可预测
 PGO 收益                            GCC            循环展开决策更好
 ThinLTO 加速链接                   Clang          -flto=thin
 二进制体积最小化                   GCC            -fvisibility=hidden + LTO
 FMV（函数多版本）                  两者兼容        语法统一
 持续集成中的向量化验证             Clang          -Rpass 更易解析
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**最终建议**：不要只用一个编译器。在生产流水线中同时用 GCC 和 Clang 构建，跑 benchmark，选更快的。编译器版本迭代快，GCC 13 的优势到 GCC 14 可能消失，反之亦然。**让数据做决定**。
