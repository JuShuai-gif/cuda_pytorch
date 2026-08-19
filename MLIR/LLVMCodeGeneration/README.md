# LLVM Code Generation 学习工程

基于 Packt Publishing《LLVM Code Generation》（Quentin Colombet & Kristof Beyls, 2025）的配套学习项目，覆盖 LLVM 后端开发全流程。

> **版本基线**：原书代码基于 LLVM 20.1.1；本目录默认支持 LLVM 20.1.x。LLVM C++ API、pass 名称和 MIR 格式不承诺跨大版本源码兼容，不能把“LLVM 17+ 有这些概念”理解成“本项目代码可以在 17-20 间直接编译”。

## 项目结构

```
LLVMCodeGeneration/
├── CMakeLists.txt          # 顶层构建文件
├── README.md               # 本文件
├── note/                   # 每章 Markdown 笔记
│   ├── Chapter0_Prerequisite_Knowledge.md    # 前置知识（体系结构/汇编/编译原理）
│   ├── Chapter1_Building_LLVM_and_Understanding_the_Directory_Structure.md
│   ├── Chapter2_Contributing_to_LLVM.md
│   ├── ... (共 22 章含 Ch0)
│   └── Chapter21_Getting_Started_with_the_Assembler.md
└── src/                    # 每章示例代码
    ├── Chapter1_Building_LLVM_and_Understanding_the_Directory_Structure/
    │   ├── example1.cpp
    │   └── example2.cpp
    ├── Chapter2_Contributing_to_LLVM/
    └── ...
```

## 章节索引

### Chapter 0: 前置知识
| 章节 | 主题 | 笔记 | 代码 |
|------|------|------|------|
| Ch0  | Prerequisite Knowledge — 计算机体系结构 / 汇编 / 编译原理 | [笔记](note/Chapter0_Prerequisite_Knowledge.md) | - |

### Part 1: Getting Started with LLVM
| 章节 | 主题 | 笔记 | 代码 |
|------|------|------|------|
| Ch1  | Building LLVM and Understanding the Directory Structure | [笔记](note/Chapter1_Building_LLVM_and_Understanding_the_Directory_Structure.md) | [代码](src/Chapter1_Building_LLVM_and_Understanding_the_Directory_Structure/) |
| Ch2  | Contributing to LLVM | [笔记](note/Chapter2_Contributing_to_LLVM.md) | [代码](src/Chapter2_Contributing_to_LLVM/) |
| Ch3  | Compiler Basics and How They Map to LLVM APIs | [笔记](note/Chapter3_Compiler_Basics_and_How_They_Map_to_LLVM_APIs.md) | [代码](src/Chapter3_Compiler_Basics_and_How_They_Map_to_LLVM_APIs/) |
| Ch4  | Writing Your First Optimization | [笔记](note/Chapter4_Writing_Your_First_Optimization.md) | [代码](src/Chapter4_Writing_Your_First_Optimization/) |
| Ch5  | Dealing with Pass Managers | [笔记](note/Chapter5_Dealing_with_Pass_Managers.md) | [代码](src/Chapter5_Dealing_with_Pass_Managers/) |
| Ch6  | TableGen – LLVM Swiss Army Knife for Modeling | [笔记](note/Chapter6_TableGen_LLVM_Swiss_Army_Knife_for_Modeling.md) | [代码](src/Chapter6_TableGen_LLVM_Swiss_Army_Knife_for_Modeling/) |

### Part 2: Middle-End – LLVM IR to LLVM IR
| 章节 | 主题 | 笔记 | 代码 |
|------|------|------|------|
| Ch7  | Understanding LLVM IR | [笔记](note/Chapter7_Understanding_LLVM_IR.md) | [代码](src/Chapter7_Understanding_LLVM_IR/) |
| Ch8  | Survey of the Existing Passes | [笔记](note/Chapter8_Survey_of_the_Existing_Passes.md) | [代码](src/Chapter8_Survey_of_the_Existing_Passes/) |
| Ch9  | Introducing Target-Specific Constructs | [笔记](note/Chapter9_Introducing_Target_Specific_Constructs.md) | [代码](src/Chapter9_Introducing_Target_Specific_Constructs/) |
| Ch10 | Hands-On Debugging LLVM IR Passes | [笔记](note/Chapter10_Hands_On_Debugging_LLVM_IR_Passes.md) | [代码](src/Chapter10_Hands_On_Debugging_LLVM_IR_Passes/) |

### Part 3: Introduction to the Backend
| 章节 | 主题 | 笔记 | 代码 |
|------|------|------|------|
| Ch11 | Getting Started with the Backend | [笔记](note/Chapter11_Getting_Started_with_the_Backend.md) | [代码](src/Chapter11_Getting_Started_with_the_Backend/) |
| Ch12 | Getting Started with the Machine Code Layer | [笔记](note/Chapter12_Getting_Started_with_the_Machine_Code_Layer.md) | [代码](src/Chapter12_Getting_Started_with_the_Machine_Code_Layer/) |
| Ch13 | The Machine Pass Pipeline | [笔记](note/Chapter13_The_Machine_Pass_Pipeline.md) | [代码](src/Chapter13_The_Machine_Pass_Pipeline/) |

### Part 4: LLVM IR to Machine IR
| 章节 | 主题 | 笔记 | 代码 |
|------|------|------|------|
| Ch14 | Getting Started with Instruction Selection | [笔记](note/Chapter14_Getting_Started_with_Instruction_Selection.md) | [代码](src/Chapter14_Getting_Started_with_Instruction_Selection/) |
| Ch15 | Instruction Selection: The IR Building Phase | [笔记](note/Chapter15_Instruction_Selection_The_IR_Building_Phase.md) | [代码](src/Chapter15_Instruction_Selection_The_IR_Building_Phase/) |
| Ch16 | Instruction Selection: The Legalization Phase | [笔记](note/Chapter16_Instruction_Selection_The_Legalization_Phase.md) | [代码](src/Chapter16_Instruction_Selection_The_Legalization_Phase/) |
| Ch17 | Instruction Selection: The Selection Phase and Beyond | [笔记](note/Chapter17_Instruction_Selection_The_Selection_Phase_and_Beyond.md) | [代码](src/Chapter17_Instruction_Selection_The_Selection_Phase_and_Beyond/) |

### Part 5: Final Lowering and Optimizations
| 章节 | 主题 | 笔记 | 代码 |
|------|------|------|------|
| Ch18 | Instruction Scheduling | [笔记](note/Chapter18_Instruction_Scheduling.md) | [代码](src/Chapter18_Instruction_Scheduling/) |
| Ch19 | Register Allocation | [笔记](note/Chapter19_Register_Allocation.md) | [代码](src/Chapter19_Register_Allocation/) |
| Ch20 | Lowering of the Stack Layout | [笔记](note/Chapter20_Lowering_of_the_Stack_Layout.md) | [代码](src/Chapter20_Lowering_of_the_Stack_Layout/) |
| Ch21 | Getting Started with the Assembler | [笔记](note/Chapter21_Getting_Started_with_the_Assembler.md) | [代码](src/Chapter21_Getting_Started_with_the_Assembler/) |

## 环境要求

- **LLVM 20.1.x**（推荐与原书一致的 20.1.1）
- **CMake 3.20+**（LLVM 20.1 官方最低要求）
- **Ninja**（推荐）或 Make
- **C++17** 编译器（GCC 9+ / Clang 9+）

先确认所有 LLVM 工具来自同一套安装：

```bash
llvm-config --version
clang --version
opt --version
llc --version
```

安装 LLVM（源码构建）：

```bash
git clone --depth 1 --branch llvmorg-20.1.1 https://github.com/llvm/llvm-project.git
cd llvm-project
cmake -GNinja -S llvm -B build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_TARGETS_TO_BUILD="AArch64;X86" \
  -DLLVM_ENABLE_PROJECTS="clang"
ninja -C build
```

开发阶段推荐 `RelWithDebInfo + assertions`：它既保留可用的调试信息和断言，又不会像纯 Debug 构建那样明显拖慢大型测试。

## 构建与运行

```bash
cd LLVMCodeGeneration

# 配置（指向你的 LLVM 安装或构建目录）
cmake -GNinja -S . -B build \
  -DLLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm

# 编译所有示例
ninja -C build
```

配置阶段会严格检查 LLVM 版本。若正在主动移植到其他版本，可以显式关闭门禁：

```bash
cmake -GNinja -S . -B build \
  -DLLVM_DIR=/path/to/other-llvm/lib/cmake/llvm \
  -DLLVM_CODEGEN_STRICT_VERSION=OFF
```

关闭门禁只表示“允许做兼容性实验”，不表示该版本已经通过验证。

## 运行单个示例

```bash
# 运行 Chapter 3 的示例 1（IR 构建）
./build/bin/Chapter3_Compiler_Basics_and_How_They_Map_to_LLVM_APIs/Chapter3_Compiler_Basics_and_How_They_Map_to_LLVM_APIs_example1

# 运行 Chapter 4 的常量传播示例
./build/bin/Chapter4_Writing_Your_First_Optimization/Chapter4_Writing_Your_First_Optimization_example1

# 运行 Chapter 5 的 Pass Manager 示例
./build/bin/Chapter5_Dealing_with_Pass_Managers/Chapter5_Dealing_with_Pass_Managers_example1
```

## 如何看待 src 中的示例

`src/` 同时包含可运行实验、LLVM API 骨架和概念模拟。后两类用于解释后端扩展点，不是可直接复制进产品的实现。是否“工业可用”至少还取决于：

- 固定 LLVM commit 与 target 配置；
- verifier、lit/FileCheck 和语义差分测试；
- ABI、对象文件、编译时间、代码大小及运行性能验证；
- 可诊断日志、灰度开关和回滚方案。

## 许可证

本项目仅用于学习目的。原书版权归 Packt Publishing 所有。
