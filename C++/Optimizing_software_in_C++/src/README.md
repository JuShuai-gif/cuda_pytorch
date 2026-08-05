# src —— 实验代码

18 组 C++17 实验 + 共享模块，由 `CMakeLists.txt` 统一管理。每组实验对应一篇笔记（见 `note/README.md`）。

## 目录结构

```
src/
├── CMakeLists.txt         # 总配置：C++17、警告、构建类型、选项开关
├── common/                # 共享模块
│   ├── benchmark.h/.cpp   # 计时：预热 + 多轮 + min/median/mean + 防编译器消除
│   └── cpu_info.h/.cpp    # CPUID 检测（指令集级别、AVX2/AVX-512）
├── 01_profiling/          # 计时 / perf stat / perf record / perf report
├── 02_integer_float/      # int/float、除法、类型转换
├── 03_branch/             # 分支预测、branchless
├── 04_loop/               # 循环展开、多累加器、不变量外提
├── 05_function/           # 函数调用方式对比
├── 06_class_virtual/      # 类、虚函数、继承、RTTI
├── 07_container/          # 容器遍历、reserve、内存池
├── 08_memory_cache/       # 缓存层级扫描、AoS/SoA、转置竞争
├── 09_alignment/          # 内存对齐
├── 10_multithreading/     # 多线程归约、扩展性
├── 11_false_sharing/      # 伪共享
├── 12_auto_vectorization/ # 自动向量化、restrict、向量化报告
├── 13_intrinsics/         # SSE/AVX2/AVX-512 intrinsics
├── 14_cpu_dispatch/       # CPUID + 运行时选择实现
├── 15_lookup_table/       # 计算 vs 查表、表大小 vs 缓存
├── 16_division_optimization/  # 除法优化
├── 17_template_metaprogramming/  # 编译期计算
└── 18_benchmark/          # 性能测试陷阱
```

## 编译方法

```bash
# 推荐：一键构建（Release）
./scripts/build.sh

# 或手动
cmake -S src -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

构建类型：

| 类型 | 编译参数 | 用途 |
|------|----------|------|
| Debug | `-O0 -g` | 调试；**禁止用于测性能** |
| Release（默认） | `-O3 -g -fno-omit-frame-pointer` | 测性能 + perf 定位热点 |
| RelWithDebInfo | `-O2 -g -fno-omit-frame-pointer` | 折中 |

可选开关（配置时传入 `-D<开关>=ON`）：

| 开关 | 作用 |
|------|------|
| `USE_MARCH_NATIVE` | 启用 `-march=native`（默认 OFF，保证可移植） |
| `ENABLE_VECTORIZATION_REPORT` | GCC 向量化报告 `-fopt-info-vec -fopt-info-vec-missed` |

## 编译参数说明

- `-Wall -Wextra -Wpedantic`：开启全部常见警告，尽早暴露问题。
- `-O3`：最高常规优化（内联、循环优化、向量化候选）。
- `-g`：调试符号，供 `perf report`/gdb 使用。
- `-fno-omit-frame-pointer`：保留帧指针，`perf` 才能还原调用栈。
- `-std=c++17`：启用 C++17（`if constexpr` 等）。
- 各实验目录的 `CMakeLists.txt` 会为**个别目标**单独加指令集参数（如 `13_avx512_example` 用 `-mavx512f`）；总配置**不全局开启 AVX2/AVX-512**。

## 运行方法

```bash
# 运行全部示例（打印实验名，记录失败）
./scripts/run_all.sh

# 运行全部 benchmark 并保存带时间戳结果
./scripts/benchmark_all.sh

# 运行单个
./build/03_branch/03_benchmark
```

## 每个实验的标准结构

每组实验遵循：`baseline.cpp`（低性能写法）、`optimized.cpp`（优化写法）、`benchmark.cpp`（对比 + 校验和一致性）、`README.md`（运行方法/预期结果/注意事项）、`CMakeLists.txt`。

质量约定：

- 结果可复现（固定随机种子）；
- 预热 + 多轮，输出 min/median/mean；
- 被测代码结果折叠进 `volatile sink` 防编译器消除；
- baseline 与 optimized 输出校验和并核对一致；
- SIMD 代码带 CPU 能力运行时检测，无 AVX-512 的机器上不执行 AVX-512 路径。
