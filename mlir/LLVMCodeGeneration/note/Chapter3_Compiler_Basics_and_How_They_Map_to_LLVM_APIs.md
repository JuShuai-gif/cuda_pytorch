# Chapter 3: Compiler Basics and How They Map to LLVM APIs

## 核心概念（详细展开）

### Target（目标架构）

**工业界定义**：Target 是编译器生成的代码将在其上运行的硬件架构。在 LLVM 中，一个 Target 不仅仅是 CPU 架构（如 x86_64、AArch64），也包括 GPU（NVPTX、AMDGPU）和 AI 加速器（如自定义 NPU）。每个 Target 在 `llvm/lib/Target/<name>` 下有完整的后端实现。

**与 MLIR 的对应**：MLIR 没有"Target"这个概念，取而代之的是 Dialect lowering chain。例如，从 `linalg` dialect → `gpu` dialect → `llvm` dialect → LLVM IR → Machine IR。MLIR 通过多层 lowering 而非单一后端来选择目标架构。

**工程陷阱**：
- Target Triple 的长度和格式在不同操作系统间有差异（如 `x86_64-apple-macosx` vs `x86_64-unknown-linux-gnu`），解析时必须处理所有变体
- 修改 Target 相关代码时必须验证所有相关后端不被破坏——使用 `LLVM_TARGETS_TO_BUILD=all` 做回归测试是必要的
- 在 JIT 场景中，host 和 target 必须一致（除非使用交叉编译 JIT）

### Host（宿主机）

Host 是运行编译器的设备。在以下场景中 Host 与 Target 不同：
- **交叉编译**：在 x86 上编译 ARM 代码（嵌入式开发、移动端开发常见）
- **GPU 编译**：Host CPU（如 x86）上运行编译器，Target 是 GPU（如 NVIDIA PTX）
- **云端编译**：在云服务器（x86）上编译 AI 加速器代码（如 Google TPU）

**AI 编译器的特殊性**：在 Triton 中，"host"通常是 CUDA driver 所在的 CPU，而代码运行在 GPU 上。Triton 编译器本身作为 Python 包的一部分在 host 上运行，生成的 PTX/CUBIN 被发送到 GPU 执行。这种情况下，编译器和运行时是两个不同的目标架构。

### Lowering（降级/下沉）

Lowering 是编译器中最核心的概念之一。它描述了 IR 从高层抽象逐步转换为低层表示的过程。

**LLVM 中的 lowering 链条**：
```
C/C++ 源码 (最高抽象)
  ↓ Clang 前端
LLVM IR (中高层抽象，类型丰富，SSA)
  ↓ opt (中端优化)
LLVM IR (优化后，更接近机器)
  ↓ llc (后端)
SelectionDAG / GlobalISel (DAG 形式，开始有目标特定信息)
  ↓ 指令选择
Machine IR (目标特定指令，寄存器分配前)
  ↓ 寄存器分配、指令调度
Machine IR (物理寄存器，指令顺序确定)
  ↓ MC 层
汇编/目标文件 (最低抽象)
```

**MLIR 中的对应概念**：MLIR 的 lowering 是多 dialect 的渐进转化：
```
Tensor 级别的操作 (linalg dialect)
  ↓ tiling + bufferization
MemRef 级别的操作 (memref dialect + scf loops)
  ↓ convert-linalg-to-loops
循环级别的操作 (scf dialect)
  ↓ convert-scf-to-cf
控制流级别的操作 (cf dialect)
  ↓ convert-to-llvm
LLVM dialect → LLVM IR
```

**工程中的 lowering 策略**：在 AI 编译器开发中，lowering 往往**不是一次性完成**的。例如，Triton 编译器先在 Triton IR 级别做 tiling 和 memory coalescing 优化，然后才 lower 到 Triton GPU IR，最后 lower 到 LLVM IR。这种"渐进式 lowering"允许在每个抽象层上做特定于该层的优化。

### Canonical Form（规范形式）

规范形式是编译器优化的基础。如果一个 IR 有 N 种等价表示，而编译器只能识别其中一种，那么大量优化机会将丢失。

**LLVM IR 中的规范形式示例**：
- 常量操作数在右侧：`add i32 %x, 5`（而非 `add i32 5, %x`）
- `icmp` 比较中的常量在右侧
- PHI 节点在基本块的开头（在 `getFirstNonPHI()` 之前）
- 基本块必须以 terminator（`ret`, `br`, `switch`, ...）结尾

**MLIR 中的规范形式**：MLIR 通过 `canonicalize` pass 实现规范化。常见的规范化模式：
- 常量折叠：`arith.addi %c1, %c2` → `arith.constant`
- 同一性消除：`arith.addi %x, %c0` → `%x`
- 分支简化：消除不可达的基本块

**工业界的教训**：不遵循规范形式可能导致 pass pipeline 中的下游 pass 无法识别你的 IR 模式，从而静默地跳过优化。这是一个非常难以调试的问题——编译器不会报错，但生成的代码质量下降。

### Build Time / Compile Time / Runtime

在编译器工程中，这三个时间概念的区分至关重要：

- **Build time（构建时间）**：即 "编译编译器的编译时间"。在 LLVM 开发中，全量构建可能耗时 1-3 小时。这直接影响开发迭代速度。
- **Compile time（编译时间）**：编译器处理用户代码的时间。对用户来说，这是"编译器的运行时间"。在 AI 编译器领域，JIT 编译时间直接影响用户体验——如果每次运行都需要等待几分钟的编译，开发者会拒绝使用。
- **Runtime（运行时）**：最终程序执行的时间。对编译器来说，这是优化的终极目标——生成更快的代码。

**AI 编译器的三时间权衡**：
在 JIT 场景（PyTorch、JAX、Triton）中，编译时间和运行时间的权衡非常微妙：
- 编译时间太长 → 用户不想等 → 糟糕的开发者体验
- 编译时间太短（优化不够）→ 运行时间太长 → 糟糕的模型性能
- 解决方案：分层编译（tiered compilation）——快速生成可运行的代码，然后逐步优化热点

### Backend vs Middle-End（后端的两种含义）

在 LLVM 语境中，"backend"有两个层次的含义：

1. **广义 Backend**：处理前端（Clang）输出后的所有阶段。包括中端优化（`opt` tool）和后端代码生成（`llc` tool）。
2. **狭义 Backend**：仅指目标特定的代码生成部分，通常在 `llvm/lib/Target/<name>/` 和 `llvm/lib/CodeGen/` 中实现。

**Middle-End**：位于前端和狭义后端之间的目标无关优化层。包括：
- Pass pipeline 中的标量优化
- 循环优化
- 向量化（虽然可能考虑目标特性）
- 内联

**MLIR 的处理方式**：MLIR 没有"middle-end"这个概念。所有转换都是 dialect 间的 lowering 和优化。目标特定性通过引入目标特定的 dialect（如 `nvvm` 对应 NVIDIA PTX）来实现。

### ABI（Application Binary Interface）

ABI 定义了函数在二进制级别如何通信。它是不同编译器（甚至同一编译器的不同版本）生成的代码之间互操作性的基础。

**ABI 涉及的内容**：
- 参数传递：哪些参数在寄存器中、哪些在栈上
- 返回值：如何返回不同大小的值
- 栈对齐：函数调用时的栈指针要求
- 寄存器保存：哪些寄存器是 caller-saved、哪些是 callee-saved
- 结构体布局：结构体成员的对齐和填充规则
- 重定位：动态链接时的符号解析

**AI 编译器的 ABI 挑战**：
在 GPU 编译中，ABI 通常比 CPU 简单（没有动态链接），但 GPU 有不同的调用约定限制：
- NVIDIA GPU：参数通常通过 constant memory 或 shared memory 传递
- Kernel launch 的参数传递完全不同于普通函数调用
- Triton 编译器需要将 Python-level 的参数映射到 GPU 的 kernel 调用约定

### Encoding（指令编码）

Encoding 描述汇编指令如何被编码为二进制机器码。这是编译器后端最低层的关注点：
- 操作码字段：哪个位域标识指令类型
- 操作数字段：寄存器编号、立即数等如何编码
- 寻址模式：不同的内存寻址模式的编码方式
- 指令前缀：如 x86 的 REX 前缀用于 64-bit 操作
- 指令长度：固定长度（如 ARM 的 4 字节）vs 可变长度（如 x86 的 1-15 字节）

---

## LLVM / MLIR 流程（深入）

### IR 构建的完整 API 调用链

从零构建 LLVM IR 的详细步骤：

```cpp
// Step 1: 创建 LLVMContext（整个编译器的上下文，负责类型/常量的 uniquing）
LLVMContext Context;

// Step 2: 创建 Module（编译单元，包含所有函数和全局变量）
Module M("my_module", Context);

// Step 3: 创建函数签名（返回类型 + 参数类型列表）
Type *Int32Ty = Type::getInt32Ty(Context);
FunctionType *FT = FunctionType::get(Int32Ty, {Int32Ty, Int32Ty}, false);
Function *F = Function::Create(FT, Function::ExternalLinkage, "add", M);

// Step 4: 创建基本块（以标签命名，entry 是入口基本块）
BasicBlock *EntryBB = BasicBlock::Create(Context, "entry", F);

// Step 5: 创建指令（需要 IRBuilder 来简化指令创建）
IRBuilder<> Builder(EntryBB);
Value *Arg0 = F->getArg(0);
Value *Arg1 = F->getArg(1);
Value *Sum = Builder.CreateAdd(Arg0, Arg1, "sum");

// Step 6: 添加 terminator 指令（基本块必须以 terminator 结尾）
Builder.CreateRet(Sum);

// Step 7: 验证 IR 的正确性（生产代码中必须执行）
verifyModule(M, &errs());

// Step 8: 输出 IR 到文件或控制台
M.print(outs(), nullptr);
```

**关键 API 的位置**：
- `LLVMContext`: `llvm/include/llvm/IR/LLVMContext.h`
- `Module`: `llvm/include/llvm/IR/Module.h`
- `Function`: `llvm/include/llvm/IR/Function.h`
- `IRBuilder`: `llvm/include/llvm/IR/IRBuilder.h`（推荐使用，封装了大量指令创建细节）

### MLIR Module/Function/Block vs LLVM 对比

| 概念 | LLVM IR | MLIR | 说明 |
|------|---------|------|------|
| 顶层容器 | `Module` | `ModuleOp` | 两者都是编译单元的顶层容器 |
| 函数 | `Function` | `func::FuncOp` | LLVM 的函数是 Module 的直接子元素 |
| 基本块 | `BasicBlock` | `Block` | 两者概念几乎相同：线性指令序列 |
| 指令 | `Instruction` | `Operation` | MLIR 的 Operation 比 LLVM 的 Instruction 更灵活（可变操作数、可变结果数） |
| 区域 | 无直接对应 | `Region` | MLIR 特有的概念：Operation 可以包含嵌套的 Block 列表 |
| SSA | 原生 SSA | 原生 SSA | 两者都使用 SSA 形式，但 MLIR 使用 block argument 传递值 |

**关键区别**：
- LLVM IR 是单一 IR（所有指令共享同一类型系统和操作码空间）
- MLIR 是多 dialect 的 IR——每个 dialect 定义自己的操作和类型
- MLIR 的 `Region` 概念远超 LLVM 的基本块嵌套——它允许任意的控制流结构

### MLIR Operations vs LLVM Instructions

MLIR 的 Operation 比 LLVM 的 Instruction 从根本上更灵活：

```cpp
// LLVM Instruction - 固定结构：单结果、类型系统已定
Instruction *Add = BinaryOperator::CreateAdd(LHS, RHS, "add", BB);

// MLIR Operation - 灵活结构：可变结果、自定义类型
// %result = "mydialect.myop"(%lhs, %rhs) {attr = 42} : (i32, i32) -> i32
Operation *Op = builder.create<MyDialect::MyOp>(loc, resultType, lhs, rhs);
```

**AI 编译器的启示**：
- 在 Triton 中，Triton IR 的操作语义上比 LLVM IR 更高层——例如，`tl.load` 和 `tl.store` 封装了复杂的 memory coalescing 逻辑
- 在 MLIR 中，你可以直接定义新的 dialect 和操作来精确表达你的 AI 模型结构

---

## 关键机制解析（工业视角）

### LLVMContext 的 Uniquing 机制

`LLVMContext` 是 LLVM IR 的全局上下文，负责唯一化（uniquing）类型和常量。这意味着：

```cpp
LLVMContext Ctx;
Type *T1 = Type::getInt32Ty(Ctx);
Type *T2 = Type::getInt32Ty(Ctx);
// T1 == T2 为 true（指针比较）——它们是同一个对象
```

**性能影响**：
- 常量比较变成 O(1) 指针比较（而非 O(n) 内容比较）
- 减少内存消耗——相同类型和常量只存储一份
- **但**：多线程场景下 `LLVMContext` 不是线程安全的。如果并行编译，每个线程需要自己的 `LLVMContext`

**MLIR 的对应**：MLIR 的 `MLIRContext` 承担类似角色。它管理 dialect 注册、类型 uniquing、属性存储等。

### PHI 节点与 SSA 的底层细节

PHI 指令是 SSA 形式的核心机制。在 LLVM IR 中：

```llvm
; 每个 PHI 操作数对应对应的前驱基本块
%result = phi i32 [ %val_from_bb1, %bb1_label ], [ %val_from_bb2, %bb2_label ]
```

**PHI 指令的约束**：
1. PHI 指令必须在基本块的最开头（任何非 PHI 指令之前）
2. 每个前驱基本块必须有一个对应的输入值
3. PHI 指令的语义：当控制流从某个前驱进入时，选择对应的值作为结果

**MLIR 的替代方案**：MLIR 不使用 PHI 指令，而是使用 **block arguments**：
```mlir
^bb1:
  %x = arith.addi %a, %b : i32
  cf.br ^bb3(%x : i32)   // 将 %x 传递给 bb3
^bb2:
  %y = arith.subi %a, %b : i32
  cf.br ^bb3(%y : i32)   // 将 %y 传递给 bb3
^bb3(%result : i32):     // block argument 作为"汇合点"
  // 在 bb3 中使用 %result
```

两种方式是语义等价的，但 block argument 风格在 IR 阅读和理解上更直观。

### 支配树的工业级使用

支配树（Dominator Tree）是编译器中最基础的数据结构之一。几乎所有优化 pass 都依赖支配关系：

```cpp
// 获取支配树分析
DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(F);

// 检查指令 a 是否支配指令 b
if (DT.dominates(InstA, InstB)) {
    // InstA 总是在 InstB 之前执行
}

// 找到两个指令的最近公共支配者
BasicBlock *ClosestDom = DT.findNearestCommonDominator(BB1, BB2);
```

**常见错误**：
- 修改 CFG 后没有更新支配树
- 在非 SSA 代码上做基于支配的优化
- 忘记支配树是建立在函数级别（不是整个 Module 级别）

### 关键边的实际处理

关键边（Critical Edge）是连接多后继源节点和多前驱目标节点的边。它之所以"关键"，是因为在不影响其他路径的前提下无法在这条边上插入代码。

**处理方式**：
```cpp
// 在特定边上插入新的基本块
BasicBlock *NewBB = SplitCriticalEdge(FromBB, ToBB);
```

**性能权衡**：拆分关键边会引入额外的跳转指令，在热点代码路径上可能导致性能下降。因此，优化 pass 应该在必要时才拆分关键边，并在优化完成后尽量合并回去。

### Machine IR 级别的关键区别

Machine IR（MIR）与 LLVM IR 的根本区别反映了它们所处的抽象层级：

| 特性 | LLVM IR | Machine IR |
|------|---------|------------|
| 类型系统 | 丰富（i1, i32, float, struct, ...） | 简化为寄存器类（GPR32, FPR64, ...） |
| 操作数 | 虚拟寄存器和常量 | 虚拟寄存器、物理寄存器、立即数、内存地址 |
| Terminator | 必须有一个 | 可以有零个或多个 |
| 前驱/后继 | 通过 CFG 分析获取 | 直接通过 `predXXX`/`succXXX` 遍历 |
| SSA | 强制 | 可选（支持 SSA 和非 SSA 形式） |
| 指令格式 | `def = opcode args` | `def0, def1, ... = opcode args`（支持多定义） |

对于 AI 编译器工程师，理解 Machine IR 相对不那么重要——大多数 AI 编译器（Triton、XLA、IREE）在更高层的 IR 上做优化，LLVM IR 才是主要交互层。但也有例外：如果你的 AI 编译器需要自定义 GPU 指令调度策略，Machine IR 的调度相关 API 就很重要。

---

## AI 编译器关联

### MLIR 的 Module/Function/Block 映射

当你从 LLVM IR 转到 MLIR 时，概念映射如下：

```
LLVM IR: Module → Function → BasicBlock → Instruction
MLIR:    ModuleOp → func::FuncOp → Block → Operation
```

MLIR 的关键差异：
1. **Region**：MLIR 中 Operation 可以包含 Region（一个或多个 Block 的列表），而 LLVM 的 Instruction 没有这个能力。这使得 MLIR 可以直接表达嵌套控制流（如 `scf.for` 的 body 是一个 Region）。
2. **Dialect 系统**：MLIR 的每个操作都属于某个 dialect。不同 dialect 的操作可以共存于同一个函数中。
3. **类型系统可扩展**：MLIR 类型可以按 dialect 自定义（如 `tensor<8x8xf32>`、`memref<?xf32>`）。

### Triton IR 的数据流模型

Triton 的 IR 设计受到 LLVM SSA 思想的深刻影响：
- Triton IR 使用 SSA 形式的虚拟寄存器
- 控制流使用结构化控制流（`scf.if`、`scf.for`）而非基本块+跳转
- Memory 操作（`tl.load`、`tl.store`）是高级别抽象——Triton 编译器自动处理 memory coalescing、bank conflicts 和 shared memory 分配

### SSA 在 AI 编译器中的意义

SSA 形式对 AI 编译器的优化至关重要：
- **数据流分析**：SSA 使 use-def 链的构建变为 O(1) 操作
- **公共子表达式消除（CSE）**：在 SSA 中，相同操作产生相同值的模式很容易检测
- **死代码消除（DCE）**：在 SSA 中，检查一条指令是否有 use 即可决定是否可删除
- **自动微分**：SSA 形式使得反向模式自动微分的实现更加简洁——每个前向操作对应一个反向操作，梯度通过 use-def 链反向传播

---

## 示例说明

### 示例1：从零构建 LLVM IR Module

```cpp
// 完整的 LLVM IR 构建示例
// 构建如下 IR:
// define i32 @add(i32 %a, i32 %b) {
// entry:
//   %sum = add i32 %a, %b
//   ret i32 %sum
// }

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

void buildSimpleIR() {
    LLVMContext Ctx;
    Module M("example", Ctx);
    
    // 创建函数类型：i32(i32, i32)
    Type *I32 = Type::getInt32Ty(Ctx);
    FunctionType *FT = FunctionType::get(I32, {I32, I32}, false);
    Function *F = Function::Create(FT, Function::ExternalLinkage, "add", M);
    
    // 命名参数（可选但有利于 IR 可读性）
    F->getArg(0)->setName("a");
    F->getArg(1)->setName("b");
    
    // 创建基本块
    BasicBlock *BB = BasicBlock::Create(Ctx, "entry", F);
    IRBuilder<> Builder(BB);
    
    // 创建加法指令
    Value *Sum = Builder.CreateAdd(F->getArg(0), F->getArg(1), "sum");
    
    // 创建返回指令
    Builder.CreateRet(Sum);
    
    // 验证并输出
    verifyModule(M, &errs());
    M.print(outs(), nullptr);
}
```

### 扩展：CFG 分析和 RPO 遍历

```cpp
// 使用 RPO 遍历函数的所有基本块
#include "llvm/ADT/PostOrderIterator.h"

void analyzeCFG(Function &F) {
    ReversePostOrderTraversal<Function*> RPOT(&F);
    int order = 0;
    for (BasicBlock *BB : RPOT) {
        outs() << "Block " << order++ << ": " << BB->getName() << "\n";
        outs() << "  Predecessors: ";
        for (BasicBlock *Pred : predecessors(BB)) {
            outs() << Pred->getName() << " ";
        }
        outs() << "\n  Successors: ";
        for (BasicBlock *Succ : successors(BB)) {
            outs() << Succ->getName() << " ";
        }
        outs() << "\n";
        
        // 检查关键边
        for (BasicBlock *Succ : successors(BB)) {
            if (isCriticalEdge(BB->getTerminator(), 
                               Succ->getFirstNonPHI()->getIterator())) {
                outs() << "  Critical edge: " << BB->getName() 
                       << " -> " << Succ->getName() << "\n";
            }
        }
    }
}
```

---

## 总结

### 技术要点清单
- LLVM IR 的层次结构：Module → Function → BasicBlock → Instruction（每层可通过 `getParent()` 向上遍历）
- `LLVMContext` 提供类型和常量的 uniquing，是多线程场景下需要注意的共享资源
- SSA 形式通过 PHI 指令解决不同控制流路径的值的汇合问题
- MLIR 使用 block arguments 替代 PHI 指令——概念等价但语法更清晰
- 支配关系是 SSA 的核心约束：定义必须支配其所有使用
- 关键边（Critical Edge）是优化的一大障碍——必要时可拆分但会影响性能
- 不可约图（Irreducible CFG）会破坏基于支配的分析——需要特殊处理
- Machine IR 与 LLVM IR 的根本区别在于抽象层级：Machine IR 更接近硬件
- RPO 遍历保证拓扑顺序——定义在使用之前遇到
- `IRBuilder` 是创建 IR 的推荐方式——自动处理指令插入位置和基本块管理

### 实践建议
1. **始终使用 IRBuilder**：手动创建 Instruction 极易出错，IRBuilder 封装了所有细节
2. **修改 IR 后验证**：使用 `verifyModule()` 或 `verifyFunction()` 检查 IR 合法性
3. **注意 use-def 链的意外跨函数遍历**：全局值的使用可能跨越函数边界
4. **在 Machine IR 中区分定义和使用**：使用 `MachineOperand::isDef()` 避免混淆
5. **处理不可约图时额外小心**：先检查 `containsIrreducibleCFG()` 再依赖支配分析
6. **生产代码中禁用断言会遮蔽 bug**：使用 `RelWithDebInfo + LLVM_ENABLE_ASSERTIONS=ON`

### 进一步学习方向
- LLVM Language Reference（https://llvm.org/docs/LangRef.html）——IR 语义的权威文档
- MLIR Language Reference（https://mlir.llvm.org/docs/LangRef/）
- "SSA-based Compiler Design" 教材——深入理解 SSA 理论的数学基础
- 阅读 `llvm/lib/IR/` 源码——理解 IR 类的实现细节
- 实践：用 LLVM C++ API 实现一个简单的 C 语言子集的编译器前端

### 工业界的实际应用
- **LLVM IR 作为中间格式**：许多编译器项目（Rust、Swift、Julia）使用 LLVM IR 作为 IR——他们只需实现自己的前端，LLVM 处理后端
- **MLIR 作为下一代编译器基础设施**：Google 的 TensorFlow、PyTorch 的 Torch-MLIR、IREE 都在使用 MLIR
- **Triton 的双层 IR**：Triton 有自定义的高层 IR（用于 tile-level 优化）和低层的 Triton GPU IR（接近 SSA 机器模型）
