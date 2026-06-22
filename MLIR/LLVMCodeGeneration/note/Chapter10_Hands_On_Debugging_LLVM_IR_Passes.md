# Chapter 10: Hands-On Debugging LLVM IR Passes

## 核心概念（详细展开）

调试 LLVM/MLIR passes 是编译器工程师最频繁也最重要的技能。本章覆盖了从发现问题到定位修复的完整调试方法论。对于 AI 编译器工程师而言，调试 GPU kernel 的编译过程比 CPU 代码编译更加困难——错误可能在 PTX 汇编层面才暴露，而非在 LLVM IR 层面。

### 调试方法论的核心流程

```
发现问题（crash/错误输出/性能退化）
  │
  ▼
Step 1: 最小化复现（Minimal Reproduction）
  ├── bugpoint / llvm-reduce
  └── 手动 IR 裁剪
  │
  ▼
Step 2: 验证 IR 合法性
  ├── opt -passes=verify input.ll
  └── opt -verify-each -passes='...' input.ll
  │
  ▼
Step 3: 定位问题 Pass
  ├── -debug-pass-manager（查看流水线结构）
  ├── -print-before/after=<passname>
  └── 二分搜索：逐步增减 passes
  │
  ▼
Step 4: 理解 Pass 行为
  ├── -debug-only=<pass-name>
  ├── LLVM_DEBUG(dbgs() << ...)
  └── 源码级断点（LLDB/GDB）
  │
  ▼
Step 5: 修复并验证
  ├── 修复 pass 代码
  ├── 添加回归测试
  └── 重新运行完整测试套件
```

## LLVM / MLIR 流程（深入）

### LLVM 的调试工具矩阵

| 工具 | 用途 | 典型命令 |
|------|------|---------|
| `bugpoint` | 自动缩小问题 IR | `bugpoint input.ll -passes=<failing>` |
| `llvm-reduce` | 现代 IR 缩减（推荐） | `llvm-reduce input.ll --test=./test.sh` |
| `opt -verify-each` | 每个 pass 后自动验证 | `opt -verify-each -passes='...' input.ll` |
| `opt -debug-pass-manager` | 打印 pass 流水线结构 | `opt -O1 -debug-pass-manager input.ll` |
| `opt -debug-only=<name>` | 指定组件的调试日志 | `opt -debug-only=instcombine -passes=...` |
| `opt -time-passes` | 性能分析 | `opt -time-passes -O2 input.ll -o /dev/null` |
| `opt -print-before/after` | IR 状态快照 | `opt -print-before=mem2reg input.ll` |

### MLIR 调试工具的对应关系

| LLVM 工具 | MLIR 对应 | 说明 |
|----------|----------|------|
| `opt -verify-each` | `mlir-opt -verify-each` | 每个 pass 后验证 IR |
| `-debug-pass-manager` | `-mlir-print-ir-after-all` | 打印 IR 变化 |
| `-print-before=<name>` | `-mlir-print-ir-before=<name>` | 特定 pass 前打印 |
| `-debug-only=<name>` | `-debug-only=<name>` | 相同机制 |
| `bugpoint` | `mlir-reduce` | MLIR 的 IR 缩减工具 |
| `-time-passes` | `-mlir-timing` | 性能分析 |

**MLIR 特有的调试优势**：
- IR 的层次化打印（`--mlir-print-ir-tree-dir` 显示完整的 Op 嵌套关系）
- per-Operation verifier 精确指出哪个 Op 非法（而 LLVM 只能报告模块级错误）
- MLIR 的 `PatternRewriter` 提供 `notifyOperationInserted/Replaced/Erased` hook for comprehensive trace

### MLIR Reduce 的缩减策略

`mlir-reduce` 比 `llvm-reduce` 提供更丰富的缩减策略：

```bash
# 1. 基本使用：给出一个"interesting"测试脚本
mlir-reduce input.mlir --test='mlir-opt -pass-pipeline="..." --verify-diagnostics $1'

# 2. 内置缩减策略（passes）：
#    - 删除无 effects 的 ops
#    - 替换 operands 为常量
#    - 删除函数参数
#    - 删除 block arguments
#    - 用空 block 替换 block body
```

## 关键机制解析（工业视角）

### 1. 日志系统（-debug-only）

LLVM 的调试日志系统基于 `LLVM_DEBUG` 宏和 `DEBUG_TYPE` 定义：

```cpp
#define DEBUG_TYPE "my-pass"
#include "llvm/Support/Debug.h"

bool MyPass::runOnFunction(Function &F) {
  LLVM_DEBUG(dbgs() << "Processing function: " << F.getName() << "\n");
  
  for (auto &BB : F) {
    LLVM_DEBUG(dbgs() << "  BB: " << BB.getName() << "\n");
    for (auto &I : BB) {
      LLVM_DEBUG(dbgs() << "    Inst: " << I << "\n");
    }
  }
  return false;
}
```

**查看所有可用的调试类型**：
```bash
opt -debug-only=help  # 列出所有注册类型
```

**调试类型分类示例**：
- `instcombine`：InstCombine pass 的内部日志
- `licm`：LICM 的决策日志
- `inline`：内联决策日志
- `loop-vectorize`：循环向量化的决策日志
- `isel`：指令选择的日志
- `regalloc`：寄存器分配的日志

**生产经验**：
- 在 debug 构建中启用 `-debug-only` 才有效（`LLVM_ENABLE_ASSERTIONS=ON`）
- `LLVM_DEBUG` 宏在 release 构建中被完全消除，零运行时开销
- 调试日志可能非常冗长，始终搭配 `| head -100` 或 `grep` 使用

### 2. Pass 流水线结构分析

**New PM 流水线结构日志**：
```
Running pass: ModuleToFunctionPassAdaptor on [module]
  Running analysis: InnerAnalysisManagerProxy<...> on [module]
  Running pass: PassManager<Function> on foo
    Running pass: InstCombinePass on foo
      Running analysis: DominatorTreeAnalysis on foo
    Running pass: LICMPass on foo
      Invalidating analysis: DominatorTreeAnalysis on foo
  Running pass: PassManager<Function> on bar
    ...
```

**从流水线日志中发现的问题**：

1. **分析频繁失效**：看到 `Invalidating analysis: DominatorTreeAnalysis` 频繁出现→你的某个 pass 做了不必要的 IR 修改，应检查 `PreservedAnalyses` 返回值是否正确。

2. **Pass 执行顺序错误**：如果 `InstCombine` 在 `LICM` 之后运行且没有正确保留分析→可能需要调整 pass 顺序。

3. **Pass 重复运行**：同一 pass 在一个函数上出现多次→可能是分析失效导致重新调度。

**性能分析命令**：
```bash
# 获取每个 pass 的执行时间
opt -time-passes -passes='default<O2>' input.bc -o /dev/null 2> timing.txt
```

输出格式：
```
===-------------------------------------------------------------------------===
                      Pass execution timing report
===-------------------------------------------------------------------------===
   ---User Time---   --System Time--    ---Wall Time---  --- Name ---
   0.1204 ( 35.2%)   0.0012 (  5.8%)    0.1217 ( 34.1%)  InstCombinePass
   0.0832 ( 24.3%)   0.0003 (  1.4%)    0.0835 ( 23.4%)  GVNPass
```

### 3. IR 缩减（Bugpoint / llvm-reduce）

**llvm-reduce 的使用模式**：

```bash
# 1. 编写测试脚本 test.sh
cat > test.sh << 'EOF'
#!/bin/bash
# 返回 0 表示 bug 仍然存在（interesting）
# 返回非 0 表示 bug 已消失（not interesting）
opt -passes=my-failing-pass reduced.ll -S -o /dev/null 2>&1 | grep -q "assertion failed"
EOF
chmod +x test.sh

# 2. 运行缩减
llvm-reduce large_input.ll --test=./test.sh
# llvm-reduce 输出缩减后的 IR 到 reduced.ll
```

**缩减的内部策略**（自动尝试）：
1. 删除不相关的函数
2. 删除不使用的全局变量
3. 删除基本块
4. 简化函数参数
5. 用 `ret void` 替换函数体
6. 简化控制流

**手动缩减技巧**：
1. 从多函数模块开始，逐个删除函数
2. 对函数内部，删除不参与 bug 计算的基本块
3. 用 `undef` 替代复杂的值定义
4. 使用 `opt -passes=instnamer` 消除隐式变量

### 4. Sanitizers 的使用

**在 LLVM 构建中启用 Sanitizers**：

```bash
# AddressSanitizer（检测堆/栈溢出、use-after-free）
cmake -DLLVM_USE_SANITIZER=Address ...

# MemorySanitizer（检测未初始化读取）
cmake -DLLVM_USE_SANITIZER=Memory ...

# UndefinedBehaviorSanitizer（检测 UB）
cmake -DLLVM_USE_SANITIZER=Undefined ...

# ThreadSanitizer（检测数据竞争）
cmake -DLLVM_USE_SANITIZER=Thread ...
```

**各 Sanitizer 的性能开销和对编译器的适用场景**：

| Sanitizer | 开销 | 检测内容 | AI 编译器常用场景 |
|-----------|------|---------|-----------------|
| ASan | ~2x | 堆/栈越界、uaf | 检测 pass 中 Value 指针悬挂 |
| MSan | ~3x | 未初始化读取 | 检测未初始化的分析结果 |
| UBSan | 低 | 整数溢出、空指针等 | 检测 pass 中的数学错误 |
| TSan | ~5-15x | 数据竞争 | 检测并行 pass 执行中的竞争 |

**生产经验**：在 LLVM/MLIR 开发中，ASan 是最常启用的 sanitizer。许多 pass bug 体现为 use-after-free（某 Value 被 erase 后仍被引用）或 heap-buffer-overflow（迭代器失效）。

### 5. LLDB 源码级调试

**LLDB 调试 LLVM Pass 的标准流程**：

```lldb
# 启动
lldb -- opt -passes=my-pass input.ll -S

# 设置断点
(lldb) b MyPass::runOnFunction
(lldb) b MyPass.cpp:42
(lldb) b Instruction::eraseFromParent

# 条件断点
(lldb) br s -f MyPass.cpp -l 100 -c 'I->getOpcode() == 28'

# 检查 LLVM 对象
(lldb) p I->dump()              # 打印指令的文本表示
(lldb) p F.dump()               # 打印函数
(lldb) p V->getName()           # 获取 Value 的名称
(lldb) p cast<Instruction>(V)->getOpcodeName()
(lldb) p BB->viewCFG()          # 可视化 CFG（需要 dot/graphviz）
(lldb) expr M->dump()           # 打印整个 Module

# 检查类型信息
(lldb) p V->getType()->dump()
(lldb) p dyn_cast<IntegerType>(V->getType())->getBitWidth()

# 在 IR 变换前后检查
# 在关键位置设置断点，执行 'p F.dump()' 查看当前 IR 状态
```

**生产调试技巧**：
- 使用 `RelWithDebInfo` 构建而非 `Debug`——保留了调试信息但性能接近 release
- 学习 LLVM 的内部表示形式（`Value::dump()`, `Type::dump()`）可以极大加速调试
- 当 LLDB 显示指针值时，使用 `p V->dump()` 而非 `p V` 获取有意义的输出
- 条件断点配合 `I->getOpcodeName()` 可以精确定位特定指令的创建/修改点

### 6. Verifier 定位 Pass 错误

**流水线级验证**（定位哪个 pass 产生了非法 IR）：
```bash
opt --passes='default<O2>' -verify-each input.ll
# 输出会精确指出崩溃发生在哪个 pass 之后
# 例如: "Broke after running SimplifyCFGPass"
```

**程序化验证**（在自定义 pass 开发中）：
```cpp
// 在 pass 结束后显式验证
bool MyPass::run(Function &F, FunctionAnalysisManager &AM) {
  // ... 执行变换 ...
  
  // 验证输出
  if (llvm::verifyFunction(F, &errs())) {
    errs() << "MyPass produced invalid IR!\n";
    F.dump();
    report_fatal_error("Invalid IR detected");
  }
  
  return true;
}
```

**New PM 自动验证配置**：
```cpp
StandardInstrumentations SI(Context, /*DebugLogging=*/true,
                            /*VerifyEachPass=*/true,  // ← 启用自动验证
                            PrintPassOpts);
SI.registerCallbacks(PIC, &MAM);
```

### 7. 性能分析 - 深入

**分析 pass 频繁失效导致的性能问题**：

```bash
# 1. 获取流水线结构日志
opt -debug-pass-manager input.bc 2> pipeline.log

# 2. 搜索分析重复运行的模式
grep "Running analysis:" pipeline.log | sort | uniq -c | sort -rn

# 输出示例（分析被重复计算的次数）：
#   47 Running analysis: DominatorTreeAnalysis on bar
#   12 Running analysis: LoopAnalysis on bar
#    3 Running analysis: ScalarEvolutionAnalysis on bar
```

**改善方法**：
1. 确保 pass 正确设置 `PreservedAnalyses`
2. 使用 `DomTreeUpdater` 等 API 增量更新分析结果
3. 避免对不需要的 IR 部分做修改（减少分析失效范围）

## AI 编译器关联

### Debugging MLIR Passes

MLIR 提供了比 LLVM 更强大的调试工具链：

**MLIR 的 IR 打印选项**：
```bash
# 打印每个 pass 前后的 IR
mlir-opt -mlir-print-ir-after-all input.mlir -pass-pipeline='...'

# 带颜色和高亮
mlir-opt -mlir-print-ir-after-all -mlir-print-ir-module-scope input.mlir

# 打印 IR 变化（diff 格式）
mlir-opt -mlir-print-ir-after-change input.mlir -pass-pipeline='...'

# 仅打印特定 Op 的变换
mlir-opt -mlir-print-ir-after=canonicalize input.mlir
```

**MLIR Pass 调试的关键技巧**：

1. **利用 per-Op verifier**：
MLIR 的 `Op::verify()` 比 LLVM 的 `verifyFunction()` 提供更精确的错误定位。如果某个 Op 的类型约束违反，verifier 会精确报告哪个 Op、哪个 Operand 出错。

2. **PatternApplicator 调试**：
```cpp
// 在 canonicalize 过程中跟踪 pattern 应用
LLVM_DEBUG({
  static int patternCount = 0;
  dbgs() << "Applying pattern #" << ++patternCount << ": ";
});
```

3. **Greedy Pattern Rewriter 调试**：
MLIR 的 `GreedyRewriteConfig` 支持设置 maxIterations 防止无限循环，这在调试 rewrite patterns 时非常有用。

### Printing MLIR IR

MLIR 的 IR 打印系统比 LLVM 更丰富：

```bash
# 完整 module 打印
mlir-opt input.mlir -mlir-print-ir-module-scope

# 树形结构打印
mlir-opt input.mlir -mlir-print-ir-tree-dir

# 打印为 LLVM IR 风格（扁平化）
mlir-opt input.mlir -convert-func-to-llvm | mlir-translate --mlir-to-llvmir
```

**调试时的关键打印模式**：
```bash
# 在 lowering pipeline 中插入打印点
mlir-opt input.mlir \
  -pass-pipeline='builtin.module(
    func.func(print-op-graph{label=before-canonicalize}),
    canonicalize,
    func.func(print-op-graph{label=after-canonicalize})
  )'
```

### Reduce Patterns in MLIR

MLIR 的 `mlir-reduce` 比 LLVM 的 `llvm-reduce` 更适合 MLIR 的多层级 IR：

```bash
# MLIR reduce 的标准用法
mlir-reduce input.mlir \
  --test='mlir-opt -pass-pipeline="builtin.module(canonicalize{max-iterations=10})" --verify-diagnostics $1'

# 使用特定的缩减策略
mlir-reduce input.mlir --pass='symbol-pruner,operation-deleter' \
  --test='...'
```

**用于调试 MLIR→LLVM IR lowering 的缩减策略**：
```bash
# 1. 先在高层级缩减
mlir-reduce input.mlir --test='mlir-opt --convert-func-to-llvm --verify-diagnostics $1'
# 2. 降低后进一步缩减
mlir-reduce lowered.mlir --test='mlir-translate --mlir-to-llvmir $1 | opt -verify'
```

### Debugging Triton Compiler Issues

Triton 编译器的调试面临独特的挑战——Python 前端、MLIR pass pipeline、LLVM codegen 三层：

**Triton 调试命令**：
```bash
# 1. 查看完整的 MLIR pipeline（展示每个 stage 的 IR）
TRITON_PRINT_AUTOTUNING=1 python script.py

# 2. 导出 MLIR 用于独立调试
triton-opt --triton-to-llvm input.mlir

# 3. 查看生成的 LLVM IR
TRITON_DUMP_LLVM_IR=1 python script.py

# 4. 查看最终的 PTX 汇编
TRITON_DUMP_PTX=1 python script.py
```

**Triton 编译中的常见调试场景**：

1. **Layout 问题**：Triton GPU Dialect 的 layout 错误（blocked vs MMA vs dot operand）是最常见的 bug。调试时检查每个 Op 的 `encoding` attribute。

2. **Address Space 问题**：LLVM lowering 阶段 shared memory promotion 错误导致地址空间不匹配。使用 `mlir-print-ir-after-all` 追踪地址空间变化。

3. **Barrier 缺失**：Triton 编译器自动插入同步原语。如果同步缺失，kernel 结果不确定（概率性错误）。检查 `gpu.barrier` 的插入点。

4. **Tensor Core 约束违反**：PTX 的 `mma.sync.aligned` 指令严格要求数据对齐和形状。调试时检查 lowering 后的 `nvvm.mma` Op 的 operand 形状。

**Triton 生产调试经验**：
- 始终先在高层级检查 IR 正确性（TTIR → TTGIR 阶段），低层级错误往往是高层级问题的表现
- 使用 `triton-opt --mlir-print-ir-after-all` 生成完整的 pipeline 日志进行离线分析
- 对确定性 kernel 使用 `TRITON_CACHE_DIR` 缓存编译结果，避免每次重新编译

## 示例说明

### 示例 1：用 debug-only 追踪 InstCombine

```bash
# 查看 InstCombine 对特定函数的变换
opt -debug-only=instcombine -passes=instcombine input.ll -S -o /dev/null 2>&1 | head -100
```

典型输出：
```
INSTCOMBINE ITERATION #1 on foo
IC: Visiting:   %add = add i32 %x, %x
IC: Old =   %add = add i32 %x, %x
    New =   <bad> shl i32 %x, 1
IC: ADD:   %add = add i32 %x, %x
IC: ADD:   %mul = shl i32 %x, 1
IC: Mod =   %mul = shl i32 %x, 1
    New =   %add = add i32 %x, %x
IC: ERASE   %add = add i32 %x, %x
```

### 示例 2：定位产生非法 IR 的 Pass

```bash
# 每个 pass 后运行验证器
opt --passes='function(instcombine,licm,simplifycfg),inline' \
    -verify-each input.ll 2>&1 | grep "Broke after"

# 输出示例：
# Broke after running SimplifyCFGPass on function bar
#  → 立即知道是 SimplifyCFG 产生了非法 IR
```

### 示例 3：用 llvm-reduce 创建最小复现

```bash
cat > test.sh << 'EOF'
#!/bin/bash
set -e
# 如果 opt 崩溃（assertion failed），返回 0（interesting）
opt -passes='loop-vectorize' reduced.ll -S -o /dev/null 2>&1 | \
  grep -q "Assertion .* failed" && exit 0
# 否则返回 1（not interesting）
exit 1
EOF
chmod +x test.sh

# 缩减
llvm-reduce --max-pass-iterations=5 huge-input.ll --test=./test.sh

# 结果文件 reduced.ll 是最小的复现用例
```

### 示例 4：性能瓶颈定位

```bash
# 1. 获取时间分析
opt -time-passes -passes='default<O3>' large_module.bc -o /dev/null 2> timing.txt

# 2. 获取流水线结构
opt -debug-pass-manager -passes='default<O3>' large_module.bc -S -o /dev/null 2> structure.txt

# 3. 交叉分析
grep -E "InstCombinePass|Invalidating" structure.txt | head -20
# 如果 InstCombine 被频繁重新运行→某些 pass 没有正确保留分析
```

### 示例 5：MLIR Pass 调试

```bash
# 追踪 MLIR lowering 过程中每个 pass 的 IR 变化
mlir-opt input.mlir \
  --mlir-print-ir-after-all \
  --mlir-print-ir-module-scope \
  -pass-pipeline='builtin.module(
    func.func(canonicalize,cse),
    convert-func-to-llvm
  )' \
  2>&1 | tee pipeline.log

# 在日志中查找期望的 transform 是否生效
grep "arith.addi" pipeline.log  # 追踪某个 Op 的变换
```

## 总结

1. **调试是编译器工程师的核心技能**：本书覆盖的调试工具（日志系统、IR 打印、缩减工具、sanitizers、LLDB）构成了完整的调试工具箱。熟练掌握这些工具可以在数小时内解决原本需要数天的问题。

2. **系统性方法是关键**：
   - 先验证 IR 合法性（verifier）
   - 再定位问题 pass（-verify-each）
   - 创建最小复现（llvm-reduce）
   - 深入理解 pass 行为（-debug-only）
   - 最终修复源码（LLDB）

3. **性能调试与正确性调试同样重要**：
   - `-time-passes` 定位编译时间瓶颈
   - `-debug-pass-manager` 发现分析重复计算
   - 正确设置 `PreservedAnalyses` 是最简单有效的编译时间优化

4. **AI 编译器调试的挑战**：
   - **多层 lowering**：Triton IR → Triton GPU → LLVM IR → PTX → SASS。错误可能在任何一层引入。
   - **概率性 bug**：GPU kernel 中的同步问题可能表现为不确定结果。
   - **不可观察性**：SASS 级别无法通过常规工具反汇编和调试。
   - **缓解策略**：在高层级做尽可能多的验证（MLIR verifier），使用 `mlir-print-ir-after-all` 和 `TRITON_PRINT_AUTOTUNING` 追踪变换。

5. **MLIR 的调试优势**：
   - per-Operation verifier 提供更精确的错误定位
   - `--mlir-print-ir-after-change` 显示最小 diff
   - `mlir-reduce` 支持更多缩减策略
   - Greedy Pattern Rewriter 的 maxIterations 防止无限循环
