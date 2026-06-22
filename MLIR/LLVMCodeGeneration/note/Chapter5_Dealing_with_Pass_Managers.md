# Chapter 5: Dealing with Pass Managers

## 核心概念（详细展开）

### Pass（优化遍）的工业定义

在 LLVM 中，Pass 是编译器优化的基本单元。一个 Pass 封装了：
- **变换逻辑**：实际修改 IR 的代码
- **依赖声明**：此 Pass 需要哪些分析（analyses）先运行
- **效果描述**：此 Pass 可能修改 IR 的哪些方面（用于分析失效决策）
- **作用域**：此 Pass 在哪个层级上运作（Module、Function、Loop 等）

**与 MLIR Pass 的对比**：
两者概念高度相似，但实现机制不同：

| 维度 | LLVM Pass | MLIR Pass |
|------|-----------|-----------|
| 作用域 | Module/CGSCC/Function/Loop/MachineFunction | 任意 Operation（可嵌套） |
| 依赖声明 | `getAnalysisUsage()` 或模板参数 | `getDependentDialects()` |
| 失效机制 | `PreservedAnalyses` 返回值 | 类似机制但更细粒度 |
| 管道构建 | `PassManager::addPass()` | `OpPassManager::addNestedPass()` |
| 分析存储 | Pass Manager 内部管理 | `AnalysisManager` 类似 |

**对 AI 编译器工程师的重要性**：在 MLIR/Triton 开发中，你几乎每天都在创建和修改 Pass。理解 LLVM Pass Manager 的设计理念直接帮助你理解 MLIR 的 Pass 基础设施。

### Pass Manager（遍管理器）的作用

Pass Manager 是 LLVM 编译管道的"指挥"。它的职责：

1. **调度执行**：按照注册顺序运行 Pass
2. **依赖解析**：确保 Pass 需要的分析在其之前运行
3. **分析缓存管理**：缓存分析结果以提高效率
4. **失效传播**：当 Pass 修改 IR 时，失效受影响的分析结果
5. **管道组合**：允许嵌套作用域（如 Module pass 可以包含 Function pass）

**为什么需要 Pass Manager**：
在大型编译器中，优化 pass 之间的依赖关系极其复杂。例如：
- `LoopUnroll` 需要 `LoopInfo` 分析
- `LoopInfo` 依赖 `DominatorTree`
- `LoopUnroll` 可能破坏 `LoopInfo`（因为它改变了循环结构）
- 但 `LoopUnroll` 可能保留 `DominatorTree`

手动管理这些依赖关系几乎不可能。Pass Manager 自动化了这一过程。

### Legacy vs New Pass Manager

LLVM 有两个 Pass Manager 实现，目前处于共存状态：

**Legacy Pass Manager（旧版）**：
```cpp
// 基于多态 + 静态注册
class MyLegacyPass : public FunctionPass {
    static char ID;
public:
    MyLegacyPass() : FunctionPass(ID) {}
    
    bool runOnFunction(Function &F) override {
        // 获取分析
        DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
        // 变换 IR...
        return Changed;
    }
    
    void getAnalysisUsage(AnalysisUsage &AU) const override {
        AU.addRequired<DominatorTreeWrapperPass>();
        AU.setPreservesAll();
    }
};

char MyLegacyPass::ID = 0;
static RegisterPass<MyLegacyPass> X("my-pass", "My Pass Description");
```

**New Pass Manager（新版）**：
```cpp
// 基于 CRTP + 模板
class MyNewPass : public PassInfoMixin<MyNewPass> {
public:
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
        // 获取分析
        DominatorTree &DT = AM.getResult<DominatorTreeAnalysis>(F);
        // 变换 IR...
        if (Changed)
            return PreservedAnalyses::allInSet<CFGAnalyses>();
        return PreservedAnalyses::all();
    }
};
```

**为什么有两个版本**：

| 特性 | Legacy PM | New PM |
|------|-----------|--------|
| 性能 | 较慢（虚函数调用开销） | 更快（模板化，编译时派发） |
| 分析缓存 | 手动管理 | 自动延迟缓存 |
| Machine IR 支持 | 支持（当前唯一） | 部分支持 |
| O0/O1/O2/O3 管道 | 内置 | 内置但不同 |
| 扩展性 | 通过宏注册 | 通过模板注册 |
| 当前状态 | CodeGen 默认 | IR 优化默认 |

**工业现状**：
- LLVM IR 到 LLVM IR 的优化管道（中端）已迁移到 New PM
- Machine IR 的 CodeGen 管道仍使用 Legacy PM（因为迁移工作量大）
- 新代码应使用 New PM，但学习两者都有必要（因为你会遇到两种代码）

### 分析（Analysis）的特殊性

Analysis 与普通 Pass 的关键区别：
- **Pass 变换 IR**，Analysis 只提供信息给 Pass
- **Analysis 结果被缓存**——只有在其依赖的 IR 被修改时才重新计算
- 在 New PM 中，Analysis 继承自 `AnalysisInfoMixin`，有自己的 Key 来标识

```cpp
// 定义 Analysis
class MyAnalysis : public AnalysisInfoMixin<MyAnalysis> {
    friend AnalysisInfoMixin<MyAnalysis>;
    static AnalysisKey Key;
public:
    struct Result {
        int Answer;
        Result(int A) : Answer(A) {}
    };
    
    Result run(Function &F, FunctionAnalysisManager &AM) {
        // 计算分析结果
        return Result(42);
    }
};

// 使用 Analysis
PreservedAnalyses MyPass::run(Function &F, FunctionAnalysisManager &AM) {
    auto &Result = AM.getResult<MyAnalysis>(F);
    // Result.Answer == 42
}
```

### PreservedAnalyses 的语义

`PreservedAnalyses` 描述 Pass 对 IR 做了什么样的修改，决定了哪些分析需要失效：

```cpp
// 没有修改任何东西
return PreservedAnalyses::all();

// 修改了所有东西（最保守——所有分析都需要重新计算）
return PreservedAnalyses::none();

// 只修改了控制流（保留了支配树和内存分析）
return PreservedAnalyses::allInSet<CFGAnalyses>();

// 自定义：保留某些分析，失效其他
PreservedAnalyses PA;
PA.preserve<DominatorTreeAnalysis>();
PA.preserve<LoopAnalysis>();
return PA;
```

**常见错误**：
- 返回 `PreservedAnalyses::all()` 但实际上修改了 IR → 导致陈旧（stale）分析被使用，产生难以调试的错误
- 返回 `PreservedAnalyses::none()` 但实际修改很小 → 导致不必要的分析重新计算，影响编译时间
- 使用 `preserveSet<>()` 和 `preserve<>()` 混淆 → 前者对应分析集（如 `CFGAnalyses`），后者对应单个分析

---

## LLVM / MLIR 流程（深入）

### Legacy Pass Manager 的完整 Pass Pipeline 构建

```cpp
// 构建 legacy pass pipeline
legacy::PassManager PM;

// 添加分析 passes（会作为依赖被自动运行）
PM.add(createDominatorTreeWrapperPass());
PM.add(createLoopInfoWrapperPass());

// 添加变换 passes
PM.add(createPromoteMemoryToRegisterPass());  // mem2reg
PM.add(createInstructionCombiningPass());     // instcombine
PM.add(createCFGSimplificationPass());        // simplifycfg
PM.add(createDeadCodeEliminationPass());      // dce

// 执行管道
Module M = ...;
PM.run(M);
```

**Legacy PM 的生命周期方法**：
- `doInitialization(Module &M)`：在管道的 module 级别开始时调用
- `runOnFunction(Function &F)`：对每个函数调用（函数级 pass）
- `doFinalization(Module &M)`：在管道的 module 级别结束时调用
- `releaseMemory()`：释放 pass 持有的资源（非 module 级别的状态）

### New Pass Manager 的完整 Pass Pipeline 构建

```cpp
// 构建 new pass pipeline
// 第一步：创建 analysis managers
LoopAnalysisManager LAM;
FunctionAnalysisManager FAM;
CGSCCAnalysisManager CGAM;
ModuleAnalysisManager MAM;

// 第二步：注册分析（建立跨层级的 proxy）
PassBuilder PB;
PB.registerModuleAnalyses(MAM);
PB.registerCGSCCAnalyses(CGAM);
PB.registerFunctionAnalyses(FAM);
PB.registerLoopAnalyses(LAM);
PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

// 第三步：创建 pass manager 并添加 passes
ModulePassManager MPM;
FunctionPassManager FPM;

// O2 级别的默认管道
MPM = PB.buildPerModuleDefaultPipeline(OptimizationLevel::O2);

// 或者自定义管道
FPM.addPass(InstCombinePass());
FPM.addPass(SimplifyCFGPass());
FPM.addPass(DeadCodeEliminationPass());
MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));

// 第四步：执行管道
MPM.run(M, MAM);
```

**New PM 的嵌套作用域**：
```cpp
ModulePassManager MPM;
{
    FunctionPassManager FPM;
    FPM.addPass(MyFunctionPass());
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
}
{
    CGSCCPassManager CGPM;
    CGPM.addPass(MyCGSCCPass());
    MPM.addPass(createModuleToCGSCCPassAdaptor(std::move(CGPM)));
}
```

### Pass Manager 内部工作原理

当 `PM.run(IR, AM)` 被调用时：

1. **依赖检查**：对于每个 pass，检查其声明的依赖分析是否可用
2. **分析调度**：如果依赖的分析尚未计算，Pass Manager 自动运行它
3. **分析缓存**：分析结果被缓存，后续 pass 可直接获取
4. **Pass 执行**：运行 pass 的 `run()` 方法
5. **失效传播**：根据 pass 返回的 `PreservedAnalyses`，失效受影响的分析
6. **重复**：对管道中剩余的 passes 重复 1-5

**失效传播的机制**：
- `CFGAnalyses` 集合包括：`DominatorTreeAnalysis`、`LoopAnalysis`、`BranchProbabilityAnalysis`、`BlockFrequencyAnalysis` 等
- 如果 pass 返回 `PreservedAnalyses::none()`（没保留任何分析），所有分析缓存被清空
- 如果 pass 返回 `PreservedAnalyses::allInSet<CFGAnalyses>()`，CFG 相关的分析被保留，但其他分析被失效

---

## 关键机制解析（工业视角）

### Legacy PM 的完整实现模板

```cpp
// MyLegacyPass.h
#pragma once
#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/Analysis/LoopInfo.h"

namespace llvm {

class MyLegacyPass : public FunctionPass {
public:
    static char ID;
    
    MyLegacyPass() : FunctionPass(ID) {}
    
    // 核心方法：执行变换
    bool runOnFunction(Function &F) override;
    
    // 声明依赖
    void getAnalysisUsage(AnalysisUsage &AU) const override;
    
private:
    // 可选的辅助方法
    bool simplifyBlock(BasicBlock &BB);
};

} // namespace llvm

// MyLegacyPass.cpp
#include "MyLegacyPass.h"
#include "llvm/InitializePasses.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

char MyLegacyPass::ID = 0;

// Legacy PM 的注册宏（这是必需的样板代码）
INITIALIZE_PASS_BEGIN(MyLegacyPass, "my-pass", 
                      "My Pass (Legacy)", false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_END(MyLegacyPass, "my-pass",
                    "My Pass (Legacy)", false, false)

bool MyLegacyPass::runOnFunction(Function &F) {
    bool Changed = false;
    
    // 获取依赖的分析
    LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    
    for (Loop *L : LI) {
        // 对每个循环做处理...
        Changed |= processLoop(L);
    }
    
    return Changed;
}

void MyLegacyPass::getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired<LoopInfoWrapperPass>();     // 需要循环信息
    AU.addPreserved<DominatorTreeWrapperPass>(); // 保留支配树
    AU.setPreservesCFG();                       // 不修改 CFG
}

// 注册 pass（使其可被 opt 使用）
static RegisterPass<MyLegacyPass> X("my-pass", "My Pass (Legacy)");

// 插件注册函数
void initializeMyLegacyPassPass(PassRegistry &);
```

**INITIALIZE_PASS 宏的参数解释**：
- `false, false`：第一个 `false` 表示不是 CFG-only pass（会修改控制流图），第二个 `false` 表示不是 analysis pass

### New PM 的完整实现模板

```cpp
// MyNewPass.h
#pragma once
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

namespace llvm {

class MyNewPass : public PassInfoMixin<MyNewPass> {
public:
    // 核心方法：执行变换
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
    
private:
    bool simplifyBlock(BasicBlock &BB);
};

} // namespace llvm

// MyNewPass.cpp
#include "MyNewPass.h"

using namespace llvm;

PreservedAnalyses MyNewPass::run(Function &F, 
                                   FunctionAnalysisManager &AM) {
    bool Changed = false;
    
    // 获取依赖的分析
    auto &LI = AM.getResult<LoopAnalysis>(F);
    
    for (Loop *L : LI) {
        // 只在需要时才获取其他分析
        auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
        Changed |= processLoop(L, DT);
    }
    
    // 声明保留/失效的分析
    if (Changed) {
        // 修改了循环结构，需要失效循环分析
        PreservedAnalyses PA;
        PA.preserve<DominatorTreeAnalysis>();  // 支配树未受影响
        return PA;  // 不保留 LoopAnalysis（默认失效）
    }
    return PreservedAnalyses::all();  // 没做任何修改
}

// 插件注册（使 pass 可通过 opt 的 -load-pass-plugin 加载）
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION, "MyNewPass", LLVM_VERSION_STRING,
        [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                    if (Name == "my-new-pass") {
                        FPM.addPass(MyNewPass());
                        return true;
                    }
                    return false;
                });
        }
    };
}
```

### Pipeline 调试的工业工具

**使用 opt 检查 pipeline**：

```bash
# 打印默认的 O2 管道结构
opt -O2 -disable-output -debug-pass-manager input.ll 2>&1

# 打印 pass 执行的时间统计
opt -O2 -time-passes input.ll -disable-output

# 在每个 pass 之后打印 IR（用于理解 IR 如何逐步变换）
opt -O2 -print-after-all input.ll -disable-output 2>&1 | less

# 只在特定 pass 之后打印 IR
opt -passes='instcombine,print-after-all,simplifycfg' input.ll -S

# 使用 passes 语法指定自定义管道
opt -passes='default<O2>' input.ll -S
opt -passes='mem2reg,instcombine,simplifycfg,dce' input.ll -S
```

**关键调试标志**：
- `-stats`：打印每个 pass 的统计信息（如"删除了多少指令"）
- `-time-passes`：打印每个 pass 的耗时
- `-debug-only=<pass-name>`：打印特定 pass 的调试输出

### 工业管道调度策略

在 IREE 等 MLIR-based AI 编译器中，pass pipeline 的调度非常精心设计：

```
IREE 编译管道（简化的全局流）:

1. 前端 import
   - 将外部表示（TF graph, PyTorch model）导入 MLIR
   
2. 全局优化
   - Inline 小函数（减少调用开销）
   - Canonicalize（规范化）
   - CSE（消除公共子表达式）
   
3. 计算调度 (Dispatch region formation)
   - 识别可融合的操作组
   - 形成 dispatch regions
   
4. Tile and Distribute
   - Linalg tiling（分块以利用缓存层次）
   - 分配到 workgroups/subgroups
   
5. Bufferization
   - 从 tensor 语义转换为 memref（显式内存管理）
   
6. Lowering
   - Linalg → Loops/GPU
   - Vectorization（如果目标架构支持 SIMD/SIMT）
   
7. 代码生成
   - Lower to LLVM dialect
   - Lower to NVVM（NVIDIA）/ROCDL（AMD）
   - LLVM codegen → PTX / AMD GPU assembly
```

**管道设计的关键原则**：
1. **规范化在先**：先规范化 IR，使后续 pattern 匹配更简单
2. **重复关键 passes**：`canonicalize` 和 `cse` 应该在每次大变换后运行
3. **渐进式 lowering**：不要一次 lower 到最底层；在每个层次上做优化
4. **融合策略**：融合（fusion）需要平衡 ILP（指令级并行）和内存带宽
5. **Profiling-driven**：在有 profiling 信息的情况下，管道可能完全不同

---

## AI 编译器关联

### MLIR Pass Manager vs LLVM Pass Manager

对于 AI 编译器工程师，理解两者的差异和共同点至关重要：

```
MLIR Pass Manager 的特点：

1. 嵌套 Operation 作用域
   - MLIR 的 Operation 可以嵌套（通过 Region）
   - Pass 可以运行在任意 Operation 上
   - 例如：func.func pass 可以包含 scf.for pass

2. Dialect 感知
   - Pass 需要声明依赖的 dialect
   - getDependentDialects() 确保 dialect 被加载

3. 更细粒度的 Pipeline 调度
   - OpPassManager::addNestedPass<MyPass>()
   - 支持按 Operation 名称调度

4. 动态 Pass Pipeline
   - 可以在运行期间添加/删除 passes（LLVM 很难做这个）
```

**代码对比**：

```cpp
// LLVM: FunctionPass 作用于每个 Function
class LLVMInstCombine : public PassInfoMixin<LLVMInstCombine> {
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

// MLIR: Pass 作用于任何 Operation
class MLIRCanonicalizer : public PassWrapper<MLIRCanonicalizer, 
                                              OperationPass<>> {
    void runOnOperation() override;
    
    // 声明依赖的 dialect
    void getDependentDialects(DialectRegistry &Registry) const override {
        Registry.insert<arith::ArithDialect, cf::ControlFlowDialect>();
    }
};
```

### Pipeline Scheduling in IREE

IREE 是最典型的 AI 编译器 pipeline 示例。其 Pass Pipeline 设计体现了工业 AI 编译器的复杂性和精密性：

```
IREE 的 Flow Pipeline（简化）:

// 1. Frontend materialization
InputConversionPipeline  // 将外部 IR 转换为 IREE 内部表示

// 2. Global optimization
GlobalOptimizationPipeline
  ├── IPO (Inter-Procedural): Inline
  ├── Canonicalization: clean up after inlining
  ├── Fusion: tile + fuse element-wise operations
  └── DispatchCreation: identify dispatch regions

// 3. Stream formation
StreamPipeline  // 管理异步执行和数据流

// 4. HAL (Hardware Abstraction Layer)
HALPipeline
  ├── Device-specific optimization
  ├── Memory planning
  └── Executable translation

// 5. Target-specific lowering
// Each target (CPU, Vulkan, CUDA, ROCm) has its own pipeline
for each target:
    TargetSpecificLoweringPipeline
    ├── Linalg → target ops
    ├── Bufferization
    ├── Vectorization (if applicable)
    └── Lower to LLVM/NVVM/ROCDL
```

**IREE Pipeline 的设计原则**：
1. **分层架构**：Flow → Stream → HAL，每层有清晰的职责边界
2. **可插拔后端**：Target-specific lowering 是独立管道，允许不同硬件共享上层优化
3. **Fusion 策略**：Tile + fuse 在 Flow 层完成，利用 `linalg` dialect 的 tile-and-fuse 能力
4. **渐进式 Lowering**：从不直接 lower 到最底层——中间保持多层次的 IR

### Triton 的 Pass Pipeline

Triton 编译器有自己的 Pass Pipeline，展示了 GPU 编译的独特需求：

```
Triton Compiler Pipeline:

1. Triton IR (Python frontend)
   ↓ TritonToTritonGPU
2. Triton GPU IR (结构化控制流 + 内存操作)
   ↓ TritonGPUCoalesce (合并内存访问)
   ↓ TritonGPUPipeline (优化 shared memory 使用)
   ↓ TritonGPURemoveLayoutConversions (简化 layout 转换)
   ↓ TritonGPUAccelerateMatmul (矩阵乘法加速)
3. Triton GPU IR → LLVM IR
   ↓ TritonGPUToLLVM
4. LLVM IR passes (复用 LLVM 的 pass pipeline)
   ↓ O2 optimization pipeline
5. NVPTX code generation (LLVM 的 NVPTX 后端)
   ↓ PTX assembly
6. NVIDIA PTXAS (NVIDIA 的汇编器)
   ↓ GPU binary (cubin)
```

---

## 示例说明

### 创建 Pass（Legacy PM）

```cpp
// 一个简单的 pass，统计函数中的指令数量
class InstCountPass : public FunctionPass {
    static char ID;
public:
    InstCountPass() : FunctionPass(ID) {}
    
    bool runOnFunction(Function &F) override {
        unsigned Count = 0;
        for (BasicBlock &BB : F) {
            for (Instruction &I : BB) {
                // 不统计 PHI 和 debug 指令
                if (!isa<PHINode>(I) && !I.isDebugOrPseudoInst())
                    Count++;
            }
        }
        errs() << "Function " << F.getName() 
               << " has " << Count << " instructions\n";
        // 这个 pass 不修改 IR
        return false;
    }
    
    void getAnalysisUsage(AnalysisUsage &AU) const override {
        AU.setPreservesAll();  // 不修改任何东西
    }
};

char InstCountPass::ID = 0;
static RegisterPass<InstCountPass> X("inst-count", "Count Instructions");
```

### 创建 Pass（New PM）

```cpp
// 同样的 pass，使用 New PM 实现
class InstCountPass : public PassInfoMixin<InstCountPass> {
public:
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
        unsigned Count = 0;
        for (BasicBlock &BB : F) {
            for (Instruction &I : BB) {
                if (!isa<PHINode>(I) && !I.isDebugOrPseudoInst())
                    Count++;
            }
        }
        errs() << "Function " << F.getName() 
               << " has " << Count << " instructions\n";
        return PreservedAnalyses::all();  // 没修改任何东西
    }
};
```

### 构建自定义 Pass Pipeline

```cpp
// 完整示例：从源码构建一个自定义的 pass pipeline 并运行
int main(int argc, char **argv) {
    // 初始化 LLVM
    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();
    
    // 创建 LLVMContext 和 Module
    LLVMContext Context;
    SMDiagnostic Err;
    std::unique_ptr<Module> M = parseIRFile("input.ll", Err, Context);
    
    // 设置 analysis managers
    LoopAnalysisManager LAM;
    FunctionAnalysisManager FAM;
    CGSCCAnalysisManager CGAM;
    ModuleAnalysisManager MAM;
    
    PassBuilder PB;
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
    
    // 构建自定义管道
    ModulePassManager MPM;
    FunctionPassManager FPM;
    
    // 添加自定义 passes
    FPM.addPass(InstCountPass());    // 统计初始化前的指令数
    FPM.addPass(PromotePass());       // mem2reg
    FPM.addPass(InstCombinePass());   // instcombine
    FPM.addPass(SimplifyCFGPass());   // simplifycfg
    FPM.addPass(InstCountPass());    // 统计优化后的指令数
    
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
    
    // 执行管道
    MPM.run(*M, MAM);
    
    // 输出优化后的 IR
    M->print(outs(), nullptr);
    
    return 0;
}
```

---

## 总结

### 技术要点清单
- Pass 是编译器优化的基本单元，封装变换逻辑、依赖声明和作用域
- Pass Manager 自动化依赖解析、分析缓存和失效传播
- Legacy PM 使用多态 + 宏注册，New PM 使用 CRTP + 模板（性能更好）
- 当前 IR 优化（中端）使用 New PM，CodeGen（后端）仍使用 Legacy PM
- Analysis 是特殊的只读 Pass，其计算结果被缓存直到被失效
- `PreservedAnalyses` 精确描述 Pass 对 IR 的影响，决定哪些分析需要重新计算
- 每个 Pass 必须声明自己的作用域（Module/Function/Loop/MachineFunction）
- `getAnalysisUsage()`（Legacy）或返回值（New PM）是声明依赖和效果的关键
- 管道构建通过 `PassManager::addPass()` 并按添加顺序执行
- MLIR 的 Pass Manager 设计灵感来自 LLVM，但更灵活（嵌套 Operation 作用域、动态管道）

### 实践建议
1. **新代码使用 New PM**：它更快、更安全，是 LLVM 的未来方向
2. **精确声明 PreservedAnalyses**：不要总是返回 `all()` 或 `none()`——这会严重影响编译时间
3. **在管道中重复规范化 passes**：`canonicalize`/`instcombine` 和 `cse` 应该在每次大变换之后运行
4. **使用 `-debug-pass-manager` 调试管道**：理解 pass 的执行顺序和分析失效情况
5. **学习现有管道的设计**：`PB.buildPerModuleDefaultPipeline(O2)` 展示了社区认可的最佳实践
6. **避免在 Pass 中做跨作用域修改**：FunctionPass 不应修改 Module 级别的结构
7. **使用 `-time-passes` 定位编译时间的瓶颈**：特别是 AI 编译器 JIT 场景下编译时间直接影响用户体验

### 进一步学习方向
- LLVM 官方 Pass 文档：https://llvm.org/docs/WritingAnLLVMPass.html
- New Pass Manager 文档：https://llvm.org/docs/NewPassManager.html
- MLIR Pass 文档：https://mlir.llvm.org/docs/PassManagement/
- 阅读 `llvm/lib/Passes/PassBuilder.cpp`——理解默认 O0/O1/O2/O3 管道的设计
- 阅读 IREE 的 pass pipeline 实现（`iree/compiler/`）——理解 AI 编译器的管道设计
- 实践：为 MLIR 的 `linalg` dialect 添加一个新的 pass 并集成到管道中

### 工业界的 Pipeline 设计哲学

**Apple 的 LLVM 使用方式**：
Apple 的编译器团队对 pass pipeline 进行了精细调优：
- 针对 Apple Silicon（AArch64）的特定微架构特征定制 cost model
- 在 O2 和 O3 之间精心选择管道的激进程度
- 使用 PGO（Profile-Guided Optimization）来驱动内联和代码布局决策

**Meta 的构建管道优化**：
Meta 的代码库极大（数千万行 C++），因此构建时间至关重要：
- 使用 ThinLTO（轻量级链接时优化）而非 FullLTO——牺牲少量优化效果换取并行性
- 在 pass pipeline 中集成自定义的 size optimization passes
- 开发专门的 pass ordering 工具来找到最优的 pass 排列

**AI 编译器的管道设计趋势**：
- **自适应管道**：根据输入模型的特性（如大小、操作类型分布）动态选择管道配置
- **编译时间 vs 运行时间权衡**：提供多级优化（如 Triton 的 num_stages、num_warps 调优）
- **硬件感知调度**：pass 顺序根据目标 GPU/加速器的具体特性调整
