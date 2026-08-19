# Chapter 1: Building LLVM and Understanding the Directory Structure

## 核心概念（详细展开）

### LLVM 基础设施的本质
LLVM 是一组可组合的编译器库的集合，而不是一个单一的编译器。理解这一点对 AI 编译器工程师至关重要——MLIR 正是建立在同样的"库化编译器基础设施"哲学之上。LLVM 的核心理念是：每个组件（前端、中端优化、后端代码生成）都作为库存在，可以被任意组合。这使得 LLVM 成为业界事实上的代码生成标准：不仅用于 CPU（x86、AArch64、RISC-V），也广泛用于 GPU（AMDGPU、NVPTX）和各种 AI 加速器（Google TPU、Apple Neural Engine）。

**工业界背景**：Google 的 TensorFlow/XLA 编译器栈底层使用 LLVM 生成代码；Apple 的 Metal 编译器也是 LLVM-based；Meta 的 PyTorch Glow 编译器同样基于 LLVM 构建。这意味着如果你在 AI 编译器领域工作，LLVM 不是可选项，而是必修课。

**为什么需要深入理解构建系统**：在工业界，你几乎不会使用默认的 `cmake + ninja` 构建。你需要处理交叉编译（cross-compilation）、自定义 target triple、按需构建特定后端、Release+Assert 混合构建（assertions enabled in Release builds for production debugging）等场景。这些知识直接影响你的日常开发效率。

### MLIR vs LLVM 项目结构对比
MLIR 作为 LLVM 的兄弟项目，在目录结构上高度相似但有重要区别：

| 概念 | LLVM | MLIR |
|------|------|------|
| 核心 IR 库 | `llvm/lib/IR` | `mlir/lib/IR` |
| 优化 passes | `llvm/lib/Transforms` | `mlir/lib/Transforms` |
| Dialect 定义 | 无（LLVM IR 是单一 IR） | `mlir/lib/Dialect/*` |
| 构建配置 | `LLVM_ENABLE_PROJECTS="clang;mlir"` | 同样通过 CMake 变量 |
| 测试框架 | lit + FileCheck | 完全相同的 lit + FileCheck |
| 后端 | `llvm/lib/Target/<name>` | 通过 `mlir/lib/Conversion/*ToLLVM` 转换 |

对 AI 编译器工程师来说，理解 LLVM 的目录结构直接帮助理解 MLIR 的代码组织。Triton 编译器（OpenAI 开发）的目录结构也受此影响。

### JAX/XLA 编译管道概述
JAX 使用 XLA（Accelerated Linear Algebra）作为编译器后端，其编译管道：
1. JAX Python → HLO（High-Level Optimizer）IR — 前端生成
2. HLO → LHLO（Late HLO） — 缓冲区分配
3. LHLO → LLVM IR — 通过 XLA 的 LLVM IR emitter
4. LLVM IR → 机器码 — 使用 LLVM 后端（通常为 GPU PTX）

理解 LLVM 构建系统意味着你可以为 XLA 定制 LLVM 后端或添加自定义优化 pass。

### Triton 的构建系统
Triton 是一个用于编写 GPU 内核的 Python 语言和编译器。其构建系统：
- 使用 CMake 作为顶层构建系统
- 内部有自定义的 Triton IR → Triton GPU IR → LLVM IR 转换管道
- 通过 `LLVM_EXTERNAL_PROJECTS` 机制集成到 LLVM 构建树中
- 在 `llvm-project/mlir` 之外还依赖 `llvm-project/llvm`

工业实践中，许多 AI 编译器项目通过 `LLVM_EXTERNAL_PROJECTS` 或 `LLVM_ENABLE_PROJECTS` 机制与 LLVM 共存构建，这比独立维护构建系统要容易得多。

---

## LLVM / MLIR 流程（深入）

### 完整构建流程的 API 调用链

```
cmake 配置阶段:
  CMakeLists.txt (项目根)
    → llvm/CMakeLists.txt
      → llvm/cmake/modules/*.cmake (LLVM 自定义 CMake 模块)
        → 检测编译器特性 (CheckCXXCompilerFlag 等)
        → 设置 LLVM_TARGETS_TO_BUILD → 读取各 Target 的 CMakeLists
        → 生成 build.ninja (或 Makefile)

ninja 构建阶段:
  ninja clang
    → 编译 clang 源码
    → 链接 LLVMCore, LLVMSupport, LLVMTarget, ... 
    → 生成 clang 可执行文件

  ninja opt
    → 编译 opt 工具
    → 链接 LLVMCore, LLVMAnalysis, LLVMTransform*, ...
    → 生成 opt 可执行文件
```

**大型项目的 CMake 模式**：在生产级 LLVM-based 项目中，常见的 CMake 模式包括：
- `LLVM_EXTERNAL_PROJECTS`：将外部项目注入 LLVM 构建树（如 IREE、Triton）。
- `LLVM_ENABLE_PROJECTS`：启用 LLVM 官方子项目（如 `clang;mlir;lld`）。
- `CMAKE_BUILD_TYPE=RelWithDebInfo`：生产环境常用的折衷——开启优化同时保留调试符号。
- `LLVM_USE_LINKER=lld`：使用 LLD 替代系统链接器，大幅加速链接阶段。
- `LLVM_PARALLEL_COMPILE_JOBS` 和 `LLVM_PARALLEL_LINK_JOBS`：分别控制编译和链接的并行度。

### 优化流程的 Pass 顺序与数据流

LLVM 的默认优化管道（以 `-O2` 为例）遵循精心设计的顺序：
1. **IR 规范化 passes**：`instcombine`（指令合并）、`simplifycfg`（简化控制流）、`mem2reg`（内存提升为寄存器/SSA）
2. **分析 passes**：构建支配树、循环信息、别名分析
3. **循环优化**：`loop-rotate`（循环旋转）、`licm`（循环不变量外提）、`indvars`（归纳变量优化）
4. **内联**：`inline` — 在 O2/O3 中非常积极
5. **标量优化**：`gvn`（全局值编号）、`sccp`（稀疏条件常量传播）
6. **向量化**：`loop-vectorize`、`slp-vectorizer`
7. **清理 passes**：最后的 `instcombine`、`simplifycfg`、`dce`（死代码消除）

**MLIR 的对应流程**：MLIR 不直接使用 LLVM 的 Pass Pipeline，但有概念上对应的过程：
- `canonicalize` 对应 LLVM 的规范化 passes
- `cse`（公共子表达式消除）对应 `gvn`
- `inline` 对应内联
- 各种 Dialect 特有的 `-convert-*-to-*` passes 对应 LLVM 的 lowering

### 测试流程的全景

LLVM 测试基础设施是分层级的：
1. **单元测试**（gtest）：测试单个 API 或类的行为。位于 `llvm/unittests/`。
2. **回归测试**（lit + FileCheck）：测试特定优化 pass 的行为。位于 `llvm/test/`。
3. **集成测试**（LLVM test-suite）：编译和运行完整的 C/C++ 程序。位于独立的 `llvm-test-suite` 仓库。

对 AI 编译器开发者来说，理解 lit/FileCheck 至关重要，因为 MLIR 和 Triton 都使用完全相同的测试框架。

---

## 关键机制解析（工业视角）

### CMake 构建配置的工业实践

**Debug vs Release vs RelWithDebInfo**：
在生产环境中，你不会只使用 Debug 或 Release。常见配置：
- **Debug**：用于日常开发，断言全开，优化全关。编译器本身运行极慢（~10x slower than Release）。
- **Release**：用于最终用户编译的速度测试。断言关闭，优化全开。
- **RelWithDebInfo**：**生产环境调试的黄金配置**。保留调试符号但不关闭优化。结合 `LLVM_ENABLE_ASSERTIONS=ON` 获得带断言的优化构建。
- **MinSizeRel**：用于嵌入式目标，尺寸优化为优先。

**加速构建的工业技巧**：
```bash
# 生产环境的典型 CMake 配置
cmake -GNinja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_TARGETS_TO_BUILD="X86;AArch64;NVPTX" \
  -DLLVM_ENABLE_PROJECTS="clang;mlir;lld" \
  -DLLVM_OPTIMIZED_TABLEGEN=ON \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DLLVM_USE_LINKER=lld \
  -DLLVM_PARALLEL_LINK_JOBS=1 \
  ${LLVM_SRC}/llvm
```

关键点解释：
- `ccache` 缓存编译中间结果，对增量构建加速明显（需要手动安装和配置 cache 大小）。
- `lld` 是多线程链接器，对大型 LLVM 可执行文件链接速度提升 3-5x。
- `LLVM_OPTIMIZED_TABLEGEN=ON`：TableGen 在调试模式下运行极慢，强制优化构建它。
- `LLVM_PARALLEL_LINK_JOBS=1`：链接阶段内存消耗巨大（~8-16GB per link），限制并行链接数避免 OOM。

### Ninja 构建驱动的深度理解

Ninja 的核心理念是"最小化构建图的解析开销"。在 LLVM 项目中，构建图的依赖关系如下：
```
llc (最终可执行文件):
  ├── libLLVMCodeGen.a ← 编译 llvm/lib/CodeGen/*.cpp
  ├── libLLVMTarget.a ← 编译 llvm/lib/Target/*.cpp  
  ├── libLLVMAnalysis.a ← 编译 llvm/lib/Analysis/*.cpp
  ├── libLLVMIR.a ← 编译 llvm/lib/IR/*.cpp
  └── libLLVMSupport.a ← 编译 llvm/lib/Support/*.cpp
```

修改 `llvm/include/llvm/IR/Function.h`（公开头文件）会触发所有依赖 `libLLVMIR` 的库重新编译——这就是为什么修改核心头文件的增量构建仍然很慢。

**工业界的 Ninja 最佳实践**：
- `ninja -j $(nproc)` 不做并行度限制，让 Ninja 自行决定
- `ninja -k 0` 一次性看到所有编译错误（避免修一个错编译一次再遇到下一个错的循环）
- `ninja -n` 干跑模式：显示将要执行的命令而不实际构建，用于调试构建图
- `ninja -d explain` 显示为什么某些目标需要重新构建

### lit 测试框架的工业用法

**高级 lit 用法**：
```bash
# 只运行特定目录下的测试
./bin/llvm-lit -sv ../llvm/test/Transforms/InstCombine/

# 只运行文件名匹配特定模式的测试  
./bin/llvm-lit -sv ../llvm/test/Transforms/InstCombine/add.ll

# 指定并行 worker 数
./bin/llvm-lit -j 16 -sv ../llvm/test/CodeGen/AArch64/

# 输出完整的 RUN 命令（用于手动复现）
./bin/llvm-lit -a test/CodeGen/AArch64/arm64-csel.ll

# 过滤测试名（正则匹配）
./bin/llvm-lit -sv --filter='GlobalISel' test/CodeGen/AArch64/
```

**MLIR 测试中的 lit**：MLIR 使用完全相同的 lit 基础设施，区别只是文件扩展名：
- LLVM IR 测试文件：`.ll` 扩展名，注释以 `;` 开头
- MLIR 测试文件：`.mlir` 扩展名，注释以 `//` 开头
- MIR (Machine IR) 测试文件：`.mir` 扩展名

这使得学习 LLVM lit 测试的技能可以直接迁移到 MLIR。

### FileCheck 的工业级模式匹配

**常见 FileCheck 调试陷阱**：
1. **空格被忽略**：`CHECK: mov x0, x1` 可以匹配 `mov    x0,    x1`。如果不希望这样，使用 `CHECK-EXACT`。
2. **子串匹配**：`CHECK: mov` 会匹配 `mov`, `movn`, `movk` 等。需要使用 `[[:#]]` 边界锚点或更精确的正则。
3. **DAG 块边界**：连续的 `CHECK-DAG` 指令是一个块，遇到非 DAG 指令块结束。但 `CHECK-NOT` 不算"非 DAG"指令——它们是独立的哨兵。
4. **变量作用域**：`[[VAR:pattern]]` 定义的变量只能在定义之后使用，但同一行可以使用和定义。

**AI 编译器中的 FileCheck 使用**：
在 MLIR/Triton 的测试中，FileCheck 用于：
- 验证 dialect 转换后 IR 的正确性
- 检查 pass pipeline 的输出格式
- 验证生成的 LLVM IR 或 PTX 代码
- 确保优化不会引入不期望的指令

### LLVM 目录结构的设计哲学

核心 LLVM 项目目录的设计遵循"接口与实现分离"原则：

```
llvm/
├── include/llvm/         ← 公开头文件（其他项目可依赖）
│   ├── ADT/              ← 抽象数据类型（SmallVector, DenseMap, ...）
│   ├── Support/          ← 系统抽象层（文件系统、错误处理、命令行解析）
│   ├── IR/               ← LLVM IR 核心类（Module, Function, Instruction, ...）
│   ├── Analysis/         ← 分析 passes 头文件（DominatorTree, LoopInfo, ...）
│   ├── Transforms/       ← 变换 passes 头文件
│   ├── CodeGen/          ← 后端代码生成基础设施
│   └── Target/           ← 目标特定信息的通用接口
├── lib/
│   ├── IR/               ← IR 核心类的实现
│   ├── Analysis/         ← 分析 passes 实现
│   ├── Transforms/       ← 变换 passes 实现
│   ├── CodeGen/          ← 后端代码生成实现（寄存器分配、指令调度等）
│   └── Target/<arch>/    ← 各目标架构的后端实现
├── tools/                ← 面向开发者的可执行文件（opt, llc, llvm-mc, ...）
├── test/                 ← lit 回归测试
├── unittests/            ← gtest 单元测试
└── utils/                ← 辅助工具（FileCheck, lit, TableGen, ...）
```

**为什么 include/llvm 路径重复项目名**：
当 Clang 同时包含 LLVM 和 Clang 自己的头文件时，通过 `#include "llvm/IR/Function.h"` 和 `#include "clang/AST/Decl.h"` 可以立即区分来源项目，避免命名冲突。MLIR 遵循相同的约定：`mlir/include/mlir/IR/Operation.h`。

### C API 的战略意义

`include/llvm-c/` 目录包含 LLVM 的 C API。对 AI 编译器工程师来说：
- **稳定性**：C API 采用 best-effort 稳定策略，通常比 C++ API 更适合长期集成，但不是永久的源码/二进制兼容保证；跨版本升级仍要阅读 release notes 并跑兼容测试
- **语言绑定**：Python 的 llvmlite、Rust 的 llvm-sys 等绑定都基于 C API
- **性能权衡**：C API 不如 C++ API 功能丰富，但与 LLVM 版本的耦合度低
- **MLIR C API**：MLIR 也从 LLVM 借鉴了 C API 模式（`mlir/include/mlir-c/`）

---

## AI 编译器关联

### Triton 编译器与 LLVM 构建的集成

Triton 编译器通过以下方式与 LLVM 集成：
1. **共享构建树**：Triton 通过 `LLVM_EXTERNAL_PROJECTS=triton` 注入 LLVM 构建树
2. **依赖 LLVM 库**：Triton 的中间层使用 LLVM 的 ADT/Support 库
3. **LLVM IR 作为输出目标**：Triton IR → Triton GPU IR → LLVM IR → PTX/AMDGPU

这意味着你构建 LLVM 的经验（CMake 配置、Ninja 调优、ccache 集成）可以直接应用于 Triton 开发环境。

### MLIR 项目结构学习

MLIR 的目录结构刻意模仿 LLVM：
```
mlir/
├── include/mlir/
│   ├── IR/               ← MLIR 核心 IR（Operation, Block, Region, ...）
│   ├── Dialect/           ← 各 Dialect 定义（linalg, gpu, scf, ...）
│   ├── Transforms/        ← MLIR passes
│   └── Conversion/        ← Dialect 间转换 passes
├── lib/
│   ├── IR/               ← MLIR IR 实现
│   ├── Dialect/           ← 各 Dialect 实现
│   └── Conversion/        ← 转换 passes 实现
├── test/                  ← lit 测试
└── tools/                 ← mlir-opt, mlir-translate, ...
```

学习 LLVM 的目录结构后，你可以迅速定位 MLIR 中的对应组件。

### JAX/XLA 编译管道的构建

在实际的 JAX 开发中，你需要：
1. 从源码构建 LLVM（XLA 依赖特定版本的 LLVM）
2. 配置正确的 `LLVM_TARGETS_TO_BUILD`（包含你需要的 GPU 后端，如 NVPTX）
3. 理解 XLA 如何通过 LLVM 的 JIT 编译能力（`LLJIT`、`ORC JIT`）实现即时编译

### IREE 的 LLVM 依赖

IREE（Incubating MLIR-based compiler）底层依赖 LLVM：
- 使用 MLIR 的 `ConvertToLLVM` dialect 转换
- 通过 LLVM 后端生成最终机器码
- 构建 IREE 需要先构建 LLVM+MLIR

**学习路径建议**：
1. 先精通 LLVM 的构建和测试体系（本章内容）
2. 然后学习 MLIR 的构建和测试（几乎完全相同）
3. 再深入学习 LLVM IR 和 MLIR 的对应关系
4. 最后掌握 pass pipeline 和代码生成流水线

---

## 示例说明

本章的配套代码（`ch1/FileCheckExamples/ex3`）展示了 FileCheck 的核心用法。在实际项目中，你应该扩展这些模式：

### FileCheck 高级模式

```llvm
; 验证优化的正确性：删除无用分支
; RUN: opt -passes=simplifycfg -S %s | FileCheck %s

define i32 @test(i32 %x) {
  ; CHECK-LABEL: @test(
  ; CHECK-NEXT:    [[RES:%.*]] = add i32 %x, 1
  ; CHECK-NEXT:    ret i32 [[RES]]
  %cmp = icmp slt i32 %x, 0
  br i1 %cmp, label %ret, label %dead
dead:
  br label %ret
ret:
  %val = phi i32 [ 0, %entry ], [ 1, %dead ]
  %res = add i32 %x, %val
  ret i32 %res
}
```

### 扩展到 AI 编译器测试

在 MLIR 中，测试模式类似但用于 dialect 转换：
```mlir
// RUN: mlir-opt --convert-linalg-to-loops %s | FileCheck %s

// CHECK-LABEL: func @matmul
// CHECK:       scf.for
// CHECK-NEXT:    scf.for
// CHECK-NEXT:      load
func.func @matmul(%A: memref<8x8xf32>, %B: memref<8x8xf32>, %C: memref<8x8xf32>) {
  linalg.matmul ins(%A, %B : memref<8x8xf32>, memref<8x8xf32>)
               outs(%C : memref<8x8xf32>)
  return
}
```

---

## 工业落地：把“能构建”升级为“可复现”

原书示例基于 LLVM 20.1.1。本项目默认以 20.1.x 为兼容窗口，是因为 LLVM C++ API、
pass 名称和 MIR 格式会随版本演进。生产环境不能只记录“LLVM 20”，至少要固化：

```text
llvm-project release + commit
CMake/Ninja 版本与完整配置参数
host 编译器、链接器、stdlib 版本
LLVM_ENABLE_ASSERTIONS 与 CMAKE_BUILD_TYPE
LLVM_TARGETS_TO_BUILD / LLVM_ENABLE_PROJECTS
容器镜像 digest 或工具链制品号
```

一套可交付的构建至少经过以下门禁：

```bash
# 快速开发门禁
ninja check-llvm

# 只验证本次改动相关测试，失败时显示完整命令
llvm-lit -sv llvm/test/<affected-area>

# 后端改动还要覆盖对应 target
llvm-lit -sv llvm/test/CodeGen/<Target>
```

不要在同一构建目录里切换 LLVM 大版本、host compiler 或断言配置；CMake cache 会保留
旧探测结果。CI 应使用全新的 build 目录，并把 `CMakeCache.txt`、版本输出和失败命令作为制品保存。

## 总结

### 技术要点清单
- LLVM 是模块化的编译器库集合，非单体编译器；MLIR 继承相同哲学
- CMake 变量控制构建的每个方面：`LLVM_TARGETS_TO_BUILD` 是开发效率的关键
- Ninja 通过最小化依赖图解析加速增量构建；`ccache` 和 `lld` 是工业标配
- lit 是通用的测试执行器，MLIR/Triton 都使用它
- FileCheck 的 CHECK 语义精巧：空格被忽略、子串匹配、DAG 块边界都是常见陷阱源
- 公开头文件遵循 `<project>/include/<project>/<lib>/` 模式以避免命名冲突
- C API（`llvm-c`/`mlir-c`）提供稳定接口，适合长期集成项目
- Debug 构建运行编译器本身约慢 10x；`RelWithDebInfo + assertions` 是调试性能问题的最佳选择
- LLVM test-suite 提供端到端正确性测试，对新后端开发至关重要
- 通过 `LLVM_EXTERNAL_PROJECTS` 可将 AI 编译器项目（Triton、IREE）注入 LLVM 构建树

### 实践建议
1. **永远不要使用默认的"全后端"构建**：只构建你需要的 2-3 个后端，节省 80% 的编译时间
2. **学习使用 ccache**：设置 `CCACHE_DIR` 到 SSD 上，设置合理的 cache 大小（如 50G）
3. **掌握 lit -sv 组合**：这是日常测试的标配
4. **在自己的 fork 中进行实验**：不要担心搞坏东西，LLVM 的构建系统非常健壮
5. **编写测试时考虑向前兼容**：使用 `CHECK-DAG` 而非严格的 `CHECK-NEXT` 序列，除非顺序确实重要

### 进一步学习方向
- LLVM 官方文档：https://llvm.org/docs/GettingStarted.html
- MLIR 官方教程：https://mlir.llvm.org/docs/Tutorials/
- CMake 官方文档：https://cmake.org/documentation/ — LLVM 大量使用高级 CMake 特性
- 阅读 `llvm/CMakeLists.txt` 了解项目级的 CMake 组织方式
- 实践：从源码构建 LLVM+Clang+MLIR 并运行测试套件
- 探索 Buildbot（https://lab.llvm.org/buildbot/）了解 LLVM CI/CD 流程

### 工业界的实际案例
- **Google**：内部使用基于 LLVM 的工具链编译所有 C++ 代码（包括 TensorFlow 的核心）
- **Apple**：所有 Apple 平台的编译器（iOS、macOS、watchOS、visionOS）都基于 LLVM/Clang。Apple 也是 M1/M2 GPU 后端、GlobalISel、寄存器分配器等关键组件的维护者
- **Meta**：开发了 Glow（AI 推理编译器）和一系列 LLVM 贡献，包括 ThinLTO 和 PGO 基础设施
- **NVIDIA GPU 生态**：LLVM/Clang 的 CUDA 和 OpenMP offload 路径，以及 Triton 等项目，会使用 LLVM NVPTX 后端生成 PTX；不要据此把 NVIDIA 的专有 `nvcc` 实现简单等同为“LLVM 版本”
- **AMD**：ROCm 平台使用 LLVM 的 AMDGPU 后端生成 GPU 代码

### 对 AI 编译器工程师的核心建议
本章的内容看似"只是构建工具"，但实际上：
- 80% 的生产力问题来自于不合理的构建配置
- 测试基础设施（lit/FileCheck）是 AI 编译器开发的日常工具
- 理解目录结构帮助你快速定位 MLIR/Triton/IREE 代码中的相关组件
- CMake 集成为你的自定义编译器项目提供了标准化的构建骨架

### 常见构建问题和解决方案

**问题1：构建 OOM (Out of Memory)**
链接 LLVM 的可执行文件（尤其是 `clang`、`lld`）消耗大量内存（8-16GB）。解决方案：
```bash
# 限制并行链接数（最重要！）
cmake -DLLVM_PARALLEL_LINK_JOBS=1 ...

# 使用更高效的链接器
cmake -DLLVM_USE_LINKER=lld ...

# 或使用 gold 链接器
cmake -DLLVM_USE_LINKER=gold ...

# 如果还是 OOM，减少并行编译数
ninja -j $(($(nproc) / 2))
```

**问题2：调试构建运行极慢**
Debug 构建的 LLVM 编译器可能比 Release 慢 10-20x。如果你需要在调试器中单步执行编译器代码：
```bash
# 不要使用纯 Debug！使用带调试符号的优化构建
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DLLVM_ENABLE_ASSERTIONS=ON ...
# 这样既保留断言检查，又有合理的性能
```

**问题3：CMake 找不到正确的工具链**
这在交叉编译场景中常见：
```bash
# 明确指定 C/C++ 编译器
cmake -DCMAKE_C_COMPILER=/path/to/clang \
      -DCMAKE_CXX_COMPILER=/path/to/clang++ \
      -DCMAKE_C_COMPILER_TARGET=aarch64-linux-gnu \
      ...
```

**问题4：增量构建仍然很慢**
因为修改核心头文件（如 `Function.h`）触发大量重编译。缓解方案：
- 使用 ccache：`-DCMAKE_C_COMPILER_LAUNCHER=ccache`
- 将构建目录放在 SSD 上
- 使用 `ninja -d explain` 了解为什么某个 target 需要重建

### MLIR 构建的特殊注意事项

构建 MLIR 时，额外需要注意：
```bash
# MLIR 作为 LLVM 子项目构建
cmake -DLLVM_ENABLE_PROJECTS="mlir;clang" ...

# MLIR 的测试使用相同的 lit 基础设施
ninja check-mlir

# 构建 MLIR 的 Python 绑定（用于 Triton/IREE 等工具）
cmake -DMLIR_ENABLE_BINDINGS_PYTHON=ON ...
```

AI 编译器工程师通常需要同时构建 LLVM + MLIR + Clang：
```bash
cmake -GNinja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_TARGETS_TO_BUILD="X86;AArch64;NVPTX;AMDGPU" \
  -DLLVM_ENABLE_PROJECTS="clang;mlir;lld" \
  -DLLVM_OPTIMIZED_TABLEGEN=ON \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DLLVM_USE_LINKER=lld \
  ${LLVM_SRC}/llvm
```

这个配置构建时间约 30-60 分钟（取决于机器），产出约 20-30GB。这是 AI 编译器开发的"标准工作环境"。
