# SIMD 编译器对比指南

本文档对比 GCC 与 Clang 在 SIMD 自动向量化场景下的行为差异及最佳实践。

---

## 1. GCC vs Clang 自动向量化

### 1.1 核心差异

| 特性 | GCC | Clang/LLVM |
|------|-----|------------|
| 循环展开 | 更激进，倾向于展开多次迭代以增大向量化窗口 | 保守，需要 `-mllvm -unroll-threshold=` 手动调优 |
| 归约模式识别 | 强项，对 `sum`、`dot product` 等归约识别准确 | 部分场景需要手动使用 `#pragma clang loop vectorize(enable)` |
| 复杂控制流向量化 | 较弱，`if`/`else` 分支可能导致向量化失败 | 强项，LLVM 的 SLP vectorizer 对分支内代码向量化更好 |
| 可预测性 | 相同代码不同版本间行为差异较大 | 相对稳定，产出更可预测 |
| 向量宽度选择 | 倾向于使用最大可用宽度（如 AVX-512） | 更保守，有时需要显式 `-mprefer-vector-width=` |

### 1.2 常见存在差异的场景

**场景 1：含有条件分支的循环**

```c
for (int i = 0; i < n; i++) {
    if (a[i] > 0)
        b[i] = a[i] * 2;
    else
        b[i] = a[i];
}
```

- GCC：可能因分支复杂放弃向量化
- Clang：使用 masked move 或 select 指令生成向量化代码

**场景 2：归约操作**

```c
float sum = 0;
for (int i = 0; i < n; i++)
    sum += a[i] * b[i];
```

- GCC：通常识别为 FMA 归约，生成高效的 `vfmadd` 序列
- Clang：可能需要 `-ffast-math` 才能达到同等级别优化

**场景 3：非连续访存（gather/scatter）**

- GCC（≥12）：`-mavx512f` 下自动生成 gather 指令
- Clang：对 gather/scatter 的自动识别更保守，通常需要 intrinsic

### 1.3 查看向量化报告

**GCC：**

```bash
# 查看所有向量化信息（成功+失败）
gcc -O3 -fopt-info-vec -c file.c

# 只看失败的部分
gcc -O3 -fopt-info-vec-missed -c file.c

# 输出到文件
gcc -O3 -fopt-info-vec-all=vec_report.txt -c file.c
```

**Clang：**

```bash
# 查看向量化成功的记录
clang -O3 -Rpass=vectorize -c file.c

# 查看向量化失败的分析
clang -O3 -Rpass-missed=vectorize -c file.c

# 查看循环相关所有信息
clang -O3 -Rpass-analysis=loop-vectorize -c file.c

# 完整报告
clang -O3 -fsave-optimization-record -c file.c
```

---

## 2. 编译器标志对照表

| 用途 | GCC | Clang/LLVM |
|------|-----|------------|
| 启用 AVX2 | `-mavx2 -mfma` | `-mavx2 -mfma` |
| 启用 AVX-512 | `-mavx512f -mavx512dq -mavx512bw` | `-mavx512f -mavx512dq -mavx512bw` |
| 启用 SSE4.2 | `-msse4.2` | `-msse4.2` |
| 启用 ARM NEON | `-mfpu=neon` (ARM) / 自动 (AArch64) | 自动 (AArch64) |
| 启用 ARM SVE | `-march=armv8-a+sve` | `-march=armv8-a+sve` |
| 自动检测 ISA | `-march=native` | `-march=native` |
| 指定目标微架构 | `-march=znver4` / `-march=sapphirerapids` | `-march=znver4` / `-march=sapphirerapids` |
| 向量化报告 | `-fopt-info-vec` | `-Rpass=vectorize` |
| 向量化失败报告 | `-fopt-info-vec-missed` | `-Rpass-missed=vectorize` |
| 强制向量化 | `-ftree-vectorize` (需要 `-O2` 或更高) | `-fvectorize` (默认开启) |
| 关闭数学 errno | `-fno-math-errno` | `-fno-math-errno` |
| 快速数学（损失精度） | `-ffast-math` | `-ffast-math` |
| 单独允许倒数近似 | `-freciprocal-math` | `-freciprocal-math` |
| 浮点收缩（FMA 生成） | `-ffp-contract=fast` | `-ffp-contract=fast` |
| 允许关联数学变换 | `-fassociative-math` | `-fassociative-math` |
| 链接时优化 | `-flto` | `-flto` |
| ThinLTO | 不支持 | `-flto=thin` |
| PGO 生成阶段 | `-fprofile-generate` | `-fprofile-generate` |
| PGO 使用阶段 | `-fprofile-use` | `-fprofile-use` |
| 指定 PGO 数据目录 | `-fprofile-dir=/path` | `-fprofile-dir=/path` |
| 地址消毒器 | `-fsanitize=address` | `-fsanitize=address` |
| 未定义行为消毒器 | `-fsanitize=undefined` | `-fsanitize=undefined` |
| 指定向量宽度 | 通过 `-march` 隐含 | `-mprefer-vector-width=512` |
| 展开阈值 | `-funroll-loops --param max-unroll-times=8` | `-mllvm -unroll-threshold=300` |
| 优化报告保存 | `-fopt-info-vec-all=file.txt` | `-fsave-optimization-record` |

---

## 3. 如何验证向量化

### 3.1 objdump / disassembly 分析

最直接的方式——检查生成的汇编指令：

```bash
# 反编译目标文件，搜索 SIMD 指令
objdump -d -M intel build/x86/avx2_gemm_micro | grep -E 'v(?:add|mul|fma|mov|gather|scatter|broadcast)'

# 统计各 ISA 指令使用次数
objdump -d -M intel build/x86/avx2_gemm_micro | grep -c 'vfmadd'
objdump -d -M intel build/x86/avx2_gemm_micro | grep -c 'vmovaps'

# ARM 平台下检查 NEON 指令
objdump -d build/arm/neon_gemm | grep -E 'fmla|fadd|ld1|st1'
```

**关键观察点：**
- 向量加载/存储：`vmovaps` / `vmovups` / `vmovdqa` / `vmovdqu`
- 向量运算：`vaddps` / `vmulps` / `vfma***ps`
- 寄存器使用：`ymm`（256位）= AVX2，`zmm`（512位）= AVX-512，`xmm`（128位）= SSE
- ARM NEON：`v0.4s` / `v0.2d` 寄存器 + `fmla` / `fadd` 等指令
- ARM SVE：`p0` 谓词寄存器 + `fmla` 等指令

### 3.2 编译器向量化提示

**GCC `-fopt-info-vec` 输出解读：**

```
# 成功向量化的典型输出
note: loop vectorized
note: loop with 32 iterations vectorized using 32 byte vectors

# 失败的典型输出
note: not vectorized: number of iterations cannot be computed
note: not vectorized: unhandled data-ref
note: not vectorized: data ref analysis failed
```

**Clang `-Rpass` 输出解读：**

```
# 成功
remark: vectorized loop (vectorization width: 8, interleaved count: 4)

# 失败
remark: loop not vectorized: cannot identify array bounds
remark: loop not vectorized: value that could not be identified as reduction is used outside the loop
```

### 3.3 llvm-mca 静态分析

llvm-mca 可以在不运行程序的情况下分析指令序列的吞吐量：

```bash
# 1. 提取目标函数汇编
objdump -d build/x86/avx2_gemm_micro > kernel.s

# 2. 用 llvm-mca 分析（假设目标 CPU 为 skylake）
llvm-mca -mcpu=skylake -iterations=100 kernel.s

# 输出包含：
#   - Dispatch Width / Retire Width
#   - IPC（每周期指令数）
#   - Total Cycles / Total uOps
#   - 各功能单元压力（port 分布）
```

### 3.4 perf stat 硬件计数器验证

运行时通过性能计数器定量验证：

```bash
# 统计 SIMD 指令占比
perf stat -e fp_arith_inst_retired.256b_packed_single \
            -e fp_arith_inst_retired.512b_packed_single \
            -e instructions:u \
            -e cycles:u \
            -e cpu/event=0xc7,umask=0x01,name=FP_ARITH_INST_RETIRED_256B/ \
            ./build/x86/avx2_gemm_micro

# IPC 检查（< 1 说明受限于延迟或依赖链）
perf stat -e instructions,cycles ./build/x86/avx2_gemm_micro

# 分支预测检查
perf stat -e branches,branch-misses ./build/x86/avx2_gemm_micro

# 缓存检查
perf stat -e cache-references,cache-misses ./build/x86/avx2_gemm_micro
```

**关键计数器含义（Intel）：**
- `fp_arith_inst_retired.256b_packed_single`：256位（AVX2）标量单精度指令数
- `fp_arith_inst_retired.512b_packed_single`：512位（AVX-512）标量单精度指令数
- `uops_retired.all`：退役微操作总数
- `l1d_pend_miss.fb_full`：L1D 填充缓冲区满——向量化不够宽/延迟过长

---

## 4. 常见问题

### 4.1 `-ffast-math` 的副作用

`-ffast-math` 实质上是以下多个标志的集合：`-fno-math-errno -funsafe-math-optimizations -ffinite-math-only -fno-rounding-math -fno-signaling-nans -fcx-limited-range`

**具体影响：**
- 允许倒数近似：`x / 3.0` 可能变成 `x * 0.333333343f`，而非精确除法
- 允许关联重组：`(a + b) + c` 可能变成 `a + (b + c)`，浮点运算结果可能不同
- 允许零符号忽略：`-0.0` 和 `+0.0` 被视为等价
- 允许 NaN/Inf 不传播：`isnan()` 检测可能失效

**建议：** 在数值精度敏感的场景下，使用更细粒度的标志替代：
```bash
# 仅允许 FMA 和关联性，保留 errno/NaN 处理
-O3 -mavx2 -fno-math-errno -ffp-contract=fast -fno-trapping-math
```

### 4.2 向量化优化等级要求

| 优化等级 | 向量化状态 | 说明 |
|---------|-----------|------|
| `-O0` | 禁用 | 无任何优化 |
| `-O1` | 禁用 | GCC/Clang 均不自动向量化 |
| `-O2` | 启用 | 基础向量化，`-ftree-vectorize` 自动生效 |
| `-O3` | 启用+激进 | 增加循环展开、SLP 向量化等 |
| `-Os` | 可能启用 | 仅在代码体积不增加时向量化 |

### 4.3 指针别名阻碍向量化

这是自动向量化失败最常见的原因。当编译器无法确定两个指针不重叠时，会生成保守的标量代码：

```c
// 编译器无法证明 a 和 b 不重叠 → 可能不向量化
void foo(float *a, float *b, int n) {
    for (int i = 0; i < n; i++)
        a[i] = b[i] * 2.0f;
}
```

**解决方案：**

```c
// 方法1：使用 __restrict (C99) 或 __restrict__ (C++)
void foo(float * __restrict a, float * __restrict b, int n);

// 方法2：使用 #pragma（需要编译器支持）
#pragma GCC ivdep   // GCC: 忽略向量依赖
_Pragma("clang loop vectorize(assume_safety)")  // Clang

// 方法3：使用 const 参数传递只读语义
// 方法4：编译时加上 -fno-strict-aliasing（不推荐，全局影响）
```

### 4.4 GCC 归约 vs Clang 分支

| 场景 | 推荐编译器 | 原因 |
|------|-----------|------|
| 归约密集（sum/dot/norm） | GCC | 识别归约模式能力更强 |
| 分支密集（含 if/else 的循环） | Clang | LLVM SLP vectorizer 对控制流向量化更好 |
| 浮点密集型（GEMM 等） | 两者相当 | 关键在于手写 intrinsic 质量 |
| PGO 收益 | GCC | GCC 的 PGO 反馈对循环变换决策影响更大 |
| LTO 收益 | Clang | ThinLTO 支持增量式 LTO，大型项目速度更快 |

### 4.5 浮点收缩与 FMA 生成

`-ffp-contract` 控制是否将 `a * b + c` 融合为单条 `fma(a, b, c)` 指令：

| 值 | 含义 | 默认 |
|----|------|------|
| `off` | 永不收缩 | |
| `on` | 仅在源文件内收缩 | GCC 默认 |
| `fast` | 跨语句跨源文件收缩（LTO 下） | Clang 默认 |

FMA 的优势：一条指令完成乘加，减少延迟和舍入误差积累。
FMA 的风险：中间结果不产生舍入，严格 IEEE 754 环境下可能不兼容。

---

## 5. PGO（Profile-Guided Optimization）工作流

### 5.1 完整步骤

```bash
# ========== Step 1: instrument（插桩编译） ==========
cmake --preset x86-profile-gen -B build/profile-gen
cmake --build build/profile-gen -j$(nproc)

# ========== Step 2: train（训练运行） ==========
# 用代表性工作负载运行程序，生成 .gcda 文件
./build/profile-gen/x86/avx2_gemm_micro --size 1024 --iterations 100
./build/profile-gen/x86/avx2_gemm_micro --size 2048 --iterations 50
./build/profile-gen/x86/avx2_gemm_micro --size 512 --iterations 200

# 确认 .gcda 文件已生成
ls -la build/profile-gen/x86/*.gcda

# ========== Step 3: use profile（使用 profiling 数据编译） ==========
cmake --preset x86-profile-use -B build/profile-use
cmake --build build/profile-use -j$(nproc)

# ========== Step 4: 验证性能提升 ==========
# 对比非 PGO 版本
./build/x86-release/x86/avx2_gemm_micro --size 1024 --iterations 100 > baseline.txt
./build/profile-use/x86/avx2_gemm_micro --size 1024 --iterations 100 > pgo.txt
```

### 5.2 PGO 关键注意事项

1. **训练数据必须具有代表性**——覆盖主要代码路径和输入规模
2. **训练构建和最终构建需要相同的编译器标志**（`-march` 必须一致）
3. **GCC 和 Clang 的 `.gcda` 格式不兼容**，不要混用
4. **多线程程序**：GCC 需要在编译时禁用线程安全 profiling (`-fprofile-update=atomic`)
5. **PGO 对循环展开和代码布局的帮助最大**，对纯 intrinsic 代码帮助有限

### 5.3 自动化 PGO 脚本

```bash
#!/bin/bash
# scripts/pgo_build.sh -- automated PGO build pipeline

set -euo pipefail
BUILD_DIR="${1:-build/pgo}"
PRESET="x86-profile-gen"

echo "=== Phase 1: Instrumentation build ==="
cmake --preset "${PRESET}" -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}" -j"$(nproc)"

echo "=== Phase 2: Training runs ==="
for size in 512 1024 2048 4096; do
    "${BUILD_DIR}/x86/avx2_gemm_micro" --size "${size}" --iterations 100
done

echo "=== Phase 3: Profile-use build ==="
cmake --preset x86-profile-use -B "${BUILD_DIR}-opt"
cmake --build "${BUILD_DIR}-opt" -j"$(nproc)"

echo "=== Done: optimized binary at ${BUILD_DIR}-opt ==="
```

---

## 6. 快速参考

### 常用编译命令

```bash
# GCC: AVX2 发布构建
gcc -O3 -march=native -mavx2 -mfma -fopt-info-vec -DNDEBUG kernel.c -o kernel

# Clang: AVX2 发布构建
clang -O3 -march=native -mavx2 -mfma -Rpass=vectorize -DNDEBUG kernel.c -o kernel

# 检查向量化失败原因
gcc -O3 -march=native -fopt-info-vec-missed kernel.c -c 2>&1 | sort -u

# Clang 同等命令
clang -O3 -march=native -Rpass-missed=vectorize kernel.c -c 2>&1 | sort -u

# 生成带源码注释的优化报告
clang -O3 -march=native -fsave-optimization-record -foptimization-record-file=report.yaml kernel.c -c
```

### 快速决策：GCC or Clang？

```text
你的代码以归约模式为主？  → 优先 GCC
你的代码有大量条件分支？  → 优先 Clang
你在做 PGO 优化？        → 优先 GCC（循环展开决策更好）
你在用 LTO 加速编译？    → 优先 Clang + ThinLTO
你想尝试不同策略？       → 两个都跑，用 perf stat 对比
```
