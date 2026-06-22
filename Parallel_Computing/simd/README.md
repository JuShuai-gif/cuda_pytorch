# SIMD 教程工程 (Industrial-Grade SIMD Tutorial)

工业级 SIMD (Single Instruction Multiple Data) 优化教程工程项目，覆盖 x86 (SSE/AVX2/AVX-512) 和 ARM (NEON/SVE/SVE2) 平台的 SIMD 编程实战。

## 目录结构

```
simd/
├── CMakeLists.txt                  # 根 CMake 构建文件
├── README.md                       # 本文件
├── scripts/                        # 辅助脚本
│   ├── run_all_tests.sh            # 运行所有测试
│   ├── run_all_benchmarks.sh       # 运行所有性能基准测试
│   ├── inspect_asm.sh              # 反汇编查看生成代码
│   └── perf_stat.sh                # perf 性能统计
├── common/                         # 公共工具头文件（header-only）
│   ├── aligned_buffer.h            # 对齐内存分配
│   ├── benchmark.h                 # 基准测试框架
│   ├── check.h                     # 测试断言宏
│   ├── cpu_features.h              # CPU 特性检测
│   ├── random_data.h               # 随机数据生成
│   └── timer.h                     # 高精度计时器
├── notes/                          # 学习笔记（Markdown）
│   ├── 00_overview/                # 平台概览
│   ├── 01_x86_basics/              # x86 基础操作
│   ├── 02_memory_layout/           # 内存布局（1）
│   ├── 03_memory_layout/           # 内存布局（2）
│   ├── 04_intrinsics_patterns/     # Intrinsics 常见模式
│   ├── 05_industrial_cases/        # 工业级案例
│   ├── 06_arm_basics/              # ARM NEON 基础
│   └── 07_arm_advanced/           # ARM SVE 进阶
├── x86/                            # x86 SIMD 源码示例
│   ├── CMakeLists.txt
│   ├── sse/
│   ├── avx2/
│   └── avx512/
└── arm/                            # ARM SIMD 源码示例
    ├── CMakeLists.txt
    ├── neon/
    └── sve/
```

## 前置要求

| 工具      | 最低版本 | 说明                              |
| --------- | -------- | --------------------------------- |
| CMake     | 3.14+    | 构建系统                          |
| GCC       | 8.0+     | 推荐 GCC 12+ 以获得最佳 SIMD 支持 |
| Clang     | 10.0+    | 推荐 Clang 16+                    |
| Linux     | 4.0+     | x86-64 或 AArch64                 |
| perf      | (可选)   | 性能分析，通常随 Linux 内核安装   |

x86 平台需要 CPU 支持以下指令集：
- **SSE4.2** — 基线要求
- **AVX2 + FMA** — 推荐（Intel Haswell 2013+ / AMD Excavator 2015+）
- **AVX-512F** — 进阶（Intel Skylake-X 2017+ / AMD Zen4 2022+）

ARM 平台需要：
- **NEON/ASIMD** — ARMv8-A 基线要求，所有 AArch64 处理器均支持
- **SVE** — 可选（如富士通 A64FX、AWS Graviton3）
- **SVE2** — 可选（如 ARM Neoverse V1/N2）

## 学习路线（推荐阅读顺序）

### 第一阶段：理解工具

1. **`common/` 头文件** —— 阅读并理解 `timer.h`、`benchmark.h`、`check.h` 等工具的实现。这些是后续所有代码的基础设施。
2. 编译并运行 `cpu_features.h` 中的 `cpu_print_features()`，确认你的 CPU 支持哪些 SIMD 指令集。

### 第二阶段：平台概览

3. **`notes/00_overview/`** —— 了解 x86 和 ARM SIMD 的历史演进、寄存器宽度、命名约定和跨平台对比。

### 第三阶段：基础操作

4. **`notes/01_x86_basics/`** —— x86 SIMD 基础：load/store、算术运算、类型转换、比较和掩码。
5. **`notes/06_arm_basics/`** —— ARM NEON 基础：与 x86 对应的 NEON 操作对照。

### 第四阶段：内存布局

6. **`notes/02_memory_layout/`** —— 理解 AoS vs SoA、对齐要求、缓存行对齐、预取策略。
7. **`notes/03_memory_layout/`** —— 进阶内存布局技巧：结构体拆分、SOA 转置、流式存储。

### 第五阶段：Intrinsics 常见模式

8. **`notes/04_intrinsics_patterns/`** —— 归约、查表、条件执行、展开循环、指令级并行。

### 第六阶段：工业级案例

9. **`notes/05_industrial_cases/`** —— 真实场景：矩阵乘法、卷积、哈希、排序网络、字符串处理。

### 第七阶段：源码实战

10. 进入 `x86/` 目录，从 SSE 开始逐步构建和运行代码。
11. 使用 `objdump` 或 `inspect_asm.sh` 查看编译器生成的汇编指令。
12. 使用 `perf stat` 分析 cache miss、分支预测、IPC 等硬件计数器。
13. 进入 `arm/` 目录，对比 ARM NEON 的实现差异。

## 构建

### 构建全部平台

```bash
mkdir build && cd build
cmake .. \
    -DBUILD_ARM=ON \
    -DBUILD_ARM_SVE=ON \
    -DBUILD_X86=ON \
    -DBUILD_X86_AVX512=ON
make -j$(nproc)
```

### 构建特定平台

```bash
# 仅构建 x86 (默认)
cmake .. -DBUILD_X86=ON

# 仅构建 x86 并启用 AVX-512
cmake .. -DBUILD_X86=ON -DBUILD_X86_AVX512=ON

# 仅构建 ARM
cmake .. -DBUILD_ARM=ON

# 构建 ARM 并启用 SVE/SVE2
cmake .. -DBUILD_ARM=ON -DBUILD_ARM_SVE=ON
```

### 常用 CMake 参数

| 参数                  | 类型 | 默认值 | 说明                               |
| --------------------- | ---- | ------ | ---------------------------------- |
| `BUILD_X86`           | BOOL | ON     | 构建 x86 SIMD 目标（SSE/AVX2）     |
| `BUILD_X86_AVX512`    | BOOL | OFF    | 构建 x86 AVX-512 目标              |
| `BUILD_ARM`           | BOOL | OFF    | 构建 ARM SIMD 目标（NEON）         |
| `BUILD_ARM_SVE`       | BOOL | OFF    | 构建 ARM SVE 目标                  |
| `CMAKE_BUILD_TYPE`    | STR  | (无)   | Release / Debug / RelWithDebInfo   |
| `CMAKE_C_COMPILER`    | STR  | (自动) | 指定 C 编译器，如 `gcc-13` 或 `clang-16` |

CMake 构建系统会自动检测宿主机架构并设置对应的 `BUILD_*` 选项。只有在交叉编译或构建非宿主机平台时才需要手动指定。

## 运行测试

```bash
# 运行所有测试
./scripts/run_all_tests.sh

# 或手动进入 build 目录运行 ctest
cd build && ctest --output-on-failure
```

## 运行性能基准测试

```bash
# 运行所有基准测试
./scripts/run_all_benchmarks.sh

# 或手动运行特定基准测试
cd build
./x86/avx2/bench_add      # 向量加法基准测试
./x86/avx2/bench_mul      # 向量乘法基准测试
./x86/avx2/bench_dotprod  # 点积基准测试
```

基准测试输出示例：
```
   Name                         ns/element        GB/s     Speedup Samples
   ------------------------  ------------  ------------  ------------ -------
  0 scalar_add                     1.2345       162.03  1.00x (baseline)     100
  1 sse_add                        0.3001       666.44        4.11x          100
  2 avx2_add                       0.1502      1331.42        8.22x          100
  3 avx512_add                     0.0751      2662.85       16.44x          100
```

## 反汇编分析

```bash
# 查看生成的 SIMD 指令
./scripts/inspect_asm.sh <binary_name>

# 等价命令
objdump -d -M intel build/x86/avx2/bench_add | less

# 查看带源码交织的反汇编（需使用 -g 编译）
objdump -S -M intel build/x86/avx2/bench_add | less
```

关注以下内容：
- 是否产生了预期的 SIMD 指令（`vmulps`、`vaddps` 等）
- 循环是否被向量化
- 是否有不必要的 `vmovups` / spill/fill 操作
- FMA 指令是否被合并（`vfmadd213ps`）

## 性能分析

判断你的 kernel 是 **compute-bound** 还是 **memory-bound** 是优化的第一步。

### 快速分类（perf stat）

```bash
# 一键获取 IPC + cache miss + SIMD 比例 + 解释
./scripts/perf_stat.sh build/x86/avx2_dot_product
```

输出会自动告诉你：
- IPC > 2.0 → compute-bound
- IPC < 0.5 + cache miss > 10% → memory-bound
- 其余 → 混合型

### 完整 profile 流水线（perf record + annotate + cachegrind）

```bash
# 一键运行 record + annotate + cachegrind
./scripts/profile.sh all build/x86/avx2_dot_product

# 或者分步执行：

# 1. 采样热点：哪些函数占用最多 CPU 时间
./scripts/profile.sh record build/x86/avx2_dot_product

# 2. 指令级热点：哪条指令是瓶颈
./scripts/profile.sh annotate build/x86/avx2_dot_product

# 3. Cache 模拟：L1/L2/LLC miss 率
./scripts/profile.sh cache build/x86/avx2_dot_product

# 4. 火焰图：可视化调用栈
./scripts/profile.sh flame build/x86/avx2_dot_product

# 5. 内存延迟分布
./scripts/profile.sh mem build/x86/avx2_dot_product

# 6. Intel Top-Down 微架构分析（需要较新 Intel CPU）
./scripts/profile.sh topdown build/x86/avx2_dot_product
```

所有输出保存到 `benchmarks/profile_output/`。

### profile 决策矩阵

| 指标                         | 结论              | 行动                                    |
| ---------------------------- | ----------------- | --------------------------------------- |
| 高 IPC (>2) + 低 cache miss  | **Compute-bound** | 展开循环、加累加器、最大化 ILP        |
| 低 IPC (<0.5) + 高 cache miss | **Memory-bound**  | 优化访存布局、prefetch、kernel fusion   |
| 高 frontend stall             | 取指/解码瓶颈     | 减少代码大小、避免间接跳转               |
| 高 backend stall              | 执行/数据瓶颈     | 检查 cache miss / port 压力              |
| 高 branch miss                | 分支预测失败     | 消除分支或用 branchless 写法             |
| SIMD 指令占比 < 5%            | 未向量化          | 检查编译标志、添加 intrinsics / restrict |

### 计算受限 vs 内存受限：Roofline 模型

```
操作强度 = FLOPs / bytes

如果 操作强度 > 硬件峰值比 -> compute-bound
如果 操作强度 < 硬件峰值比 -> memory-bound

硬件峰值比 = 峰值 GFLOPS / 峰值内存带宽(GB/s)
例如 DDR5-5600 双通道 ~90 GB/s, CPU @ 500 GFLOPS
峰值比 = 500 / 90 ≈ 5.6 FLOPs/byte
```

| 操作       | 操作强度           | 判定          |
| ---------- | ----------------- | ------------- |
| 向量加法   | 0.17 FLOPs/byte   | memory-bound  |
| 向量内积   | 0.17 FLOPs/byte   | memory-bound  |
| 矩阵乘法   | ~N FLOPs/byte     | compute-bound |
| ReLU       | 0.083 FLOPs/byte  | memory-bound  |
| 卷积 3x3   | 2.25 FLOPs/byte   | 边界带        |

### 优化策略

**Memory-bound kernel:**
- 尽可能合并 pass（kernel fusion）
- 使用 non-temporal / streaming store 避免 cache pollution
- 考虑数据压缩（fp16、int8）
- 优化内存访问模式（连续访问、对齐、prefetch）

**Compute-bound kernel:**
- 最大化 ILP (Instruction Level Parallelism)，使用多个累加器
- 使用 FMA 指令
- 展开循环以减少循环开销
- 寄存器分块（register blocking/tiling）

### 方法论：从怀疑到证明

1. **hypothesis** — 猜测瓶颈是 compute、memory、branch 还是 decode
2. **perf stat** — 收集 IPC、cache miss、branch miss 宏观数据
3. **perf record/report** — 定位热点函数
4. **perf annotate** — 在汇编级别确认具体瓶颈指令
5. **cachegrind** — 仿真 cache 行为，确认数据布局问题
6. **fix & repeat** — 修改代码后重新 profile，验证改进效果

## 工业级优化检查清单

在声称一个 kernel "已优化" 之前，请逐项检查：

- [ ] 数据是否对齐到 SIMD 寄存器宽度（32 字节用于 AVX2，64 字节用于 AVX-512）
- [ ] 循环是否已展开（4x-8x 用于 memory-bound，更多用于 compute-bound）
- [ ] 是否消除了循环中的分支（branchless programming）
- [ ] 是否使用了 FMA 指令替代分离的 mul + add
- [ ] 是否使用了 prefetch（`_mm_prefetch` / `__builtin_prefetch`）
- [ ] load/store 是否使用了 aligned 版本（`_mm256_load_ps` 而非 `_mm256_loadu_ps`）
- [ ] 是否避免了 gather/scatter（除非硬件原生支持）
- [ ] 对 memory-bound kernel：是否使用了 non-temporal store
- [ ] 对 compute-bound kernel：register blocking 是否减少了内存流量
- [ ] 是否使用 `restrict` 关键字告知编译器指针无别名
- [ ] 是否在 L1 cache 中分块工作集
- [ ] 是否通过 `perf stat` / `profile.sh` 验证了 IPC 和 cache miss 率
- [ ] 是否通过 `perf annotate` 确认了热点汇编指令与预期一致

## 参考资源

### 官方文档
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html) — 交互式 intrinsics 查询工具
- [ARM NEON Programmer's Guide](https://developer.arm.com/architectures/instruction-sets/intrinsics/) — ARM NEON intrinsics 参考
- [ARM SVE Programmer's Guide](https://developer.arm.com/documentation/102699/latest/) — SVE/SVE2 编程指南
- [Intel 64 and IA-32 Architectures Optimization Reference Manual](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
- [AMD64 Architecture Programmer's Manual - Volume 5: 64-Bit Media and x87 Floating-Point Instructions](https://www.amd.com/en/support/tech-docs.html)

### 书籍
- *Computer Systems: A Programmer's Perspective* (CSAPP) — 第 5 章 优化程序性能
- *Agner Fog's Optimization Manuals* — [https://www.agner.org/optimize/](https://www.agner.org/optimize/)
- *Hacker's Delight* (2nd Edition) — Henry S. Warren, Jr.

### 在线资源
- [uops.info](https://uops.info/) — x86 指令延迟和吞吐量数据库
- [Compiler Explorer (godbolt.org)](https://godbolt.org/) — 在线反汇编和编译器对比
- [SIMD Everywhere (SIMDe)](https://github.com/simd-everywhere/simde) — 跨平台 SIMD 可移植头文件库

## 贡献

欢迎提交 Issue 和 Pull Request。请确保：
- 所有代码通过 `ctest` 测试
- 新 kernel 包含 scalar baseline 和性能基准测试
- 代码风格与现有代码保持一致
- 注释和文档使用英文

## 许可证

MIT License
