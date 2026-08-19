# Chapter 4: Writing Your First Optimization

## 核心概念（详细展开）

### Value（值的深层理解）

Value 是 LLVM 编译器基础设施中最基础的概念。在编译器理论中，Value 和 Variable 的区分是理解 SSA 的关键：

- **Variable（变量）**：一个命名的存储位置，其内容在不同时间可以改变。如 C 语言中的 `int x;`。
- **Value（值）**：程序计算的一个不可变结果。一个变量在不同的程序点可能持有不同的值。

**工业界的实际意义**：当你构建一个 IR 优化 pass 时，你操作的是 Value 而非 Variable。这是微妙但重要的区别——例如，你不能"重复赋值"给一个 Value；你只能创建一个新的 Value 并替换旧 Value 的使用。

在 LLVM 的 C++ 类型系统中：
```cpp
// Value 是所有 IR 实体的基类
class Value {
    // 所有值的基类——包括常数、指令、函数、参数等
    // use_iterator use_begin() / use_end() —— 遍历所有使用
    // unsigned getNumUses() —— 获取使用次数
    // void replaceAllUsesWith(Value *New) —— 全局替换
};

// Instruction 继承自 Value
class Instruction : public User {
    // User 是"使用其他值的 Value"
};
```

### SSA 的工业级理解

SSA（Static Single Assignment）是程序表示的一个属性：每个变量在程序的文本中恰好被定义一次。这个简单的性质带来了深远的影响：

**SSA 的关键优势**：
1. **数据流分析简化**：use-def 关系直接编码在 IR 中——不需要做数据流分析就能找到定义-使用关系
2. **稀疏分析可能**：可以只在有定义/使用的程序点进行分析，而非对每个程序点进行
3. **等价性检测简单**：如果两个 Value 是同一个对象，它们就是等价的

**SSA 的关键挑战**：
- **PHI 节点处理**：在控制流汇合点插入 PHI 指令
- **SESE 区域破坏**：当优化需要跨越基本块边界时，SSA 维护变得复杂
- **内存操作**：内存不是 SSA 的——需要 `MemorySSA` 分析来将内存访问映射到 SSA 形式

**与 MLIR SSA 的对比**：
LLVM 和 MLIR 都使用 SSA，但处理值汇合的方式不同。LLVM 使用 PHI 指令，MLIR 使用 block arguments。在实现层面：
- LLVM PHI 节点是特殊的 Instruction，位于基本块开头
- MLIR block arguments 是 Block 的输入，语义上更接近函数参数
- 两者在内部控制上是等价的，但 MLIR 的方式在结构化控制流场景中更自然

### PHI 指令的运行时语义

```llvm
; %result 的值取决于从哪个前驱进入此基本块
loop_header:
  %i = phi i32 [ 0, %entry ], [ %next, %loop_body ]
  ; 从 entry 进入时，i = 0
  ; 从 loop_body 进入时，i = next（前一次迭代的值）
  %cmp = icmp slt i32 %i, 10
  br i1 %cmp, label %loop_body, label %exit
```

**常见误解澄清**：
- PHI 指令不是条件赋值——它的语义取决于**控制流来自哪里**，而非某个条件
- PHI 指令的所有输入**同时**有效——在进入基本块时选择对应前驱的值
- 在 SSA 构造算法中，PHI 被放置在支配边界（dominance frontier）

### 支配关系的工业应用

支配关系是 SSA 的数学基础。理解支配关系是编写正确优化 pass 的前提：

```cpp
// d 支配 n ↔ 从 entry 到 n 的所有路径都经过 d
// 这意味着：在 d 中定义的值可以在 n 中安全使用

// 立即支配者 (idom)：最接近的直接支配者
// idom(n) 是 n 的唯一"最近"支配者

// 支配树：以 entry 为根，边为 idom 关系
// DominatorTree 类封装了支配树的构建和查询
```

**支配关系的实际使用**：
1. **代码提升（Hoisting）**：将指令移到循环外部时，新的位置必须支配所有原使用点
2. **代码下沉（Sinking）**：将指令移到控制流分支内部时，必须确保不影响其他路径
3. **冗余消除**：如果两个相同的计算出现在被同一节点支配的位置，可以消除重复

### Def-Use 和 Use-Def 链的遍历

这是日常优化开发中最常用的操作：

```cpp
// Def-Use 链：给定一个值，找到所有使用它的地方
// 用途：替换值时需要更新所有使用
Value *OldVal = ...;
Value *NewVal = ...;
OldVal->replaceAllUsesWith(NewVal);

// 手动遍历所有使用
for (Use &U : OldVal->uses()) {
    User *Usr = U.getUser();     // 使用该值的指令
    unsigned OpNo = U.getOperandNo(); // 在第几个操作数位置
    // 做自定义处理...
}

// Use-Def 链：给定一个指令的操作数，找到定义它的指令
// 用途：分析某个操作数的来源
Instruction *I = ...;
Value *Op = I->getOperand(0);     // 获取操作数
if (Instruction *Def = dyn_cast<Instruction>(Op)) {
    // Op 是由 Def 指令定义的
    // 现在可以从 Def 继续追踪
}
```

**跨函数遍历陷阱**：全局变量、函数参数、外部常量等值的 use 列表可能跨越多个函数。如果你的 pass 只作用于一个函数，必须过滤掉其他函数的 uses。

### mem2reg（内存到寄存器的提升）

mem2reg 是将前端生成的基于内存的 IR（alloca/load/store）转换为 SSA 形式的 pass。这是 LLVM IR 优化管道中最重要的 pass 之一，通常作为管道的**第一个 pass** 运行。

```llvm
; 输入：基于内存的 IR（前端生成）
define i32 @foo(i32 %a, i32 %b) {
  %p = alloca i32           ; 分配栈空间
  store i32 %a, ptr %p      ; 存储 a
  %v1 = load i32, ptr %p    ; 加载
  store i32 %b, ptr %p      ; 覆盖为 b
  %v2 = load i32, ptr %p    ; 加载
  %r = add i32 %v1, %v2
  ret i32 %r
}

; 输出：SSA 形式（mem2reg 后）
define i32 @foo(i32 %a, i32 %b) {
  %r = add i32 %a, %b       ; 直接使用 SSA 值
  ret i32 %r
}
```

**mem2reg 的工作原理**：
1. 识别所有的 `alloca` 指令（这些是待提升的栈分配）
2. 对于每个 `alloca`，找到所有的 `store` 和 `load` 到该地址
3. 使用 SSA 构造算法（基于支配边界）插入 PHI 节点
4. 将 `load` 替换为 PHI 的结果，删除 `alloca`/`store`/`load`

### 优化的合法性（Legality）检查

Legality 是优化最基本的约束：你的优化必须保持程序的语义。LLVM 提供了丰富的 API 来辅助合法性检查。

**整数溢出标志**：
```cpp
// NSW (No Signed Wrap): 有符号溢出为未定义行为
if (AddInst->hasNoSignedWrap()) {
    // 可以做基于"不溢出"假设的优化
    // 例如：x * 2 / 2 → x（仅在 NSW 时合法）
}

// NUW (No Unsigned Wrap): 无符号溢出为未定义行为
if (AddInst->hasNoUnsignedWrap()) {
    // 类似 NSW 但适用于无符号操作
}
```

**快速数学标志（Fast-Math Flags）**：
```cpp
// 这些标志允许忽略浮点运算的严格语义约束
// nnans: 没有 NaN——可以假设操作数是有限数
// ninfs: 没有无穷——不需要处理无穷值
// nsz: 没有有符号零——+0 和 -0 等价
// arcp: 允许倒数近似——1/x 可以用近似公式
// contract: 允许 FMA（融合乘加）收缩
// reassoc: 允许重新结合——可以改变运算顺序
// afn: 近似函数——允许不精确的函数实现
// fast: 以上所有标志的组合
```

**使用 Alive2 验证转换**：在生产环境中，任何非平凡的优化转换都应该用 Alive2 验证：
```
; 在 alive2 网站上或本地运行
; 输入：优化前的 IR
; 输出：优化后的 IR
; Alive2 自动检查两者的语义等价性
```

### 优化的有利性（Profitability）评估

即使一个优化是合法的，它也可能不是有利的。例如：
- 一个看似简单的指令替换可能导致寄存器压力增加
- 循环展开可能增加指令缓存（I-cache）的 miss 率
- 常量传播可能增加代码尺寸（如果常量编码比指令本身更大）

**LLVM 的 Cost Model 体系**：

```cpp
// TargetTransformInfo (TTI): 提供指令成本的估计
TargetTransformInfo &TTI = ...;
InstructionCost Cost = TTI.getInstructionCost(I, 
    TargetTransformInfo::TCK_RecipThroughput);
// Cost 是抽象的——不要直接比较数值，使用提供的比较运算符

// TargetLibraryInfo (TLI): 库函数信息
TargetLibraryInfo &TLI = ...;
if (TLI.isFunctionVectorizable("cosf", VF)) {
    // cosf 可以向量化为 VF 宽度的向量操作
}

// DataLayout: 数据类型属性
const DataLayout &DL = ...;
uint64_t Size = DL.getTypeSizeInBits(Ty);
unsigned Align = DL.getPrefTypeAlign(Ty).value();
```

**寄存器压力评估**：在 LLVM IR 级别，寄存器压力不容易直接评估（因为还没有做寄存器分配）。但在 Machine IR 级别：
```cpp
// RegPressureTracker 跟踪寄存器使用
MachineRegisterInfo &MRI = MF.getRegInfo();
// 遍历 live intervals 检查寄存器压力
```

---

## LLVM / MLIR 流程（深入）

### SSA 构造的详细流程

```
1. 前端生成
   Clang 将 C/C++ 源码 → LLVM IR (使用 alloca/load/store 表示变量)
   
2. mem2reg pass
   PromoteMemoryToRegister pass 将 alloca/load/store → SSA values + PHI nodes
   
3. SSA 中间形式
   所有标量值现在在 SSA 形式中，可以直接做数据流优化

4. SSA 破坏（Machine IR 阶段）
   在寄存器分配阶段，SSA 被破坏——虚拟寄存器被映射到物理寄存器后
   不再满足单次定义约束
```

### 优化 Pass 的执行流程

```
输入: Function (在 SSA 形式中)
  │
  ├── Step 1: 获取分析结果
  │   DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  │   LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  │
  ├── Step 2: 遍历 IR
  │   for (BasicBlock &BB : F) {
  │     for (Instruction &I : BB) {
  │
  ├── Step 3: 检查合法性
  │       if (!isLegalToOptimize(I))
  │         continue;
  │
  ├── Step 4: 检查有利性
  │       if (!isProfitableToOptimize(I))
  │         continue;
  │
  ├── Step 5: 执行变换
  │       Value *NewVal = performOptimization(I);
  │       I.replaceAllUsesWith(NewVal);
  │       Changed = true;
  │     }
  │   }
  │
  └── 返回 Changed（是否修改了 IR）
```

### MLIR 的规范化 vs LLVM 的 instcombine

LLVM 的 `instcombine`（指令合并）和 MLIR 的 `canonicalize` 在各自生态中扮演相似的角色：

| 特性 | LLVM instcombine | MLIR canonicalize |
|------|-----------------|-------------------|
| 目标 | 将 IR 转换为规范形式 | 将 IR 转换为规范形式 |
| 实现方式 | 大型的模式匹配 + 重写规则 | 每个操作定义自己的 `getCanonicalizationPatterns` |
| 扩展性 | 修改 instcombine 源码 | 在 dialect 中定义新的 pattern |
| 调用时机 | 在 pass pipeline 中多次调用 | 通常在每个转换 pass 之后调用 |

**MLIR 规范化模式示例**：
```cpp
// 在 MLIR 中定义规范化模式
struct AddIAddConstant : public OpRewritePattern<arith::AddIOp> {
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(arith::AddIOp Op,
                                   PatternRewriter &Rewriter) const override {
        // 匹配: %0 = arith.addi %x, %c1; %1 = arith.addi %0, %c2
        // 重写为: %1 = arith.addi %x, %(c1+c2)
        // ...
    }
};
```

---

## 关键机制解析（工业视角）

### 常量传播优化的完整实现

以下是生产级别的常量传播 pass 的核心逻辑：

```cpp
bool ConstantPropagation::runOnFunction(Function &F) {
    bool Changed = false;
    SmallVector<Instruction *, 16> WorkList;
    
    // Phase 1: 收集所有可能的常量表达式
    for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
            // 跳过 PHI、terminator、以及已经有常量结果的指令
            if (isa<PHINode>(I) || I.isTerminator())
                continue;
            if (all_of(I.operands(), 
                       [](Value *V) { return isa<ConstantInt>(V); })) {
                WorkList.push_back(&I);
            }
        }
    }
    
    // Phase 2: 常量折叠
    for (Instruction *I : WorkList) {
        // 利用 LLVM 的 ConstantFoldInstruction API
        if (Constant *C = ConstantFoldInstruction(
                I, I->getModule()->getDataLayout())) {
            I->replaceAllUsesWith(C);
            I->eraseFromParent();
            Changed = true;
        }
    }
    
    // Phase 3: 传播到 PHI 节点
    // （需要多次迭代直到不动点fixed point）
    bool LocalChanged;
    do {
        LocalChanged = false;
        for (BasicBlock &BB : F) {
            for (Instruction &I : BB) {
                if (auto *PN = dyn_cast<PHINode>(&I)) {
                    // 如果所有输入值都是相同的常量
                    if (Value *Common = PN->hasConstantValue()) {
                        PN->replaceAllUsesWith(Common);
                        PN->eraseFromParent();
                        LocalChanged = true;
                        Changed = true;
                    }
                }
            }
        }
    } while (LocalChanged);
    
    return Changed;
}
```

**关键 API**：
- `ConstantFoldInstruction`：LLVM 内置的常量折叠——利用它而不是手动实现
- `PHINode::hasConstantValue()`：检查 PHI 的所有输入是否收敛到同一常量
- `replaceAllUsesWith`：全局替换值的所有使用——SSA 语义保证替换后程序仍正确

### APInt 的使用（任意精度整数）

LLVM 使用 `APInt`（Arbitrary Precision Integer）处理任意位宽的整数运算。这对跨平台编译至关重要：

```cpp
// 创建各种位宽的常量
APInt Val8(8, 42);          // 8 位值 42
APInt Val32(32, 1000000);   // 32 位值
APInt Val64(64, 0xFFFFFFFFFFFFFFFFULL);

// 常量运算
APInt Val8As32 = Val8.zext(32);
APInt Sum = Val32 + Val8As32;  // 二元运算前必须显式统一位宽
APInt Val32As64 = Val32.sext(64);
APInt Prod = Val32As64 * Val64;

// 创建 ConstantInt
ConstantInt *C8 = ConstantInt::get(Context, Val8);
ConstantInt *C32 = ConstantInt::get(Context, Val32);
```

**为什么需要 APInt 而非 `int64_t`**：LLVM IR 支持任意位宽的整数（`i1`, `i7`, `i19`, `i256`, ...）。只有 `APInt` 能精确表示所有可能的位宽。

**生产陷阱**：`APInt` 不会像 C/C++ 整数那样做 usual arithmetic conversions。多数二元运算要求两侧位宽相同；需要根据 IR 的有符号/无符号语义显式选择 `sext`、`zext` 或 `trunc`。扩展方式选错会在高位为 1 的输入上产生静默错编。

### 使用 isa/dyn_cast/cast 进行类型检查

LLVM 使用自己的 RTTI（运行时类型识别）替代 C++ 的 `dynamic_cast`，以获得更好的性能：

```cpp
// isa<> : 类型检查（返回 bool）
if (isa<ConstantInt>(V)) { ... }

// dyn_cast<> : 安全向下转型（返回指针，失败时返回 nullptr）
if (auto *CI = dyn_cast<ConstantInt>(V)) { ... }

// cast<> : 强制向下转型（失败时触发断言 abort）
ConstantInt *CI = cast<ConstantInt>(V);  // 相信 V 一定是 ConstantInt

// isa_and_nonnull : 检查非空且为特定类型
if (isa_and_nonnull<PHINode>(I)) { ... }
```

**工业最佳实践**：
- 使用 `dyn_cast` 当类型可能不匹配时
- 使用 `cast` 当你确定类型时（逻辑保证 + 断言验证）
- 不要使用 C++ 的 `dynamic_cast`——LLVM 的版本更高效且与构建配置兼容

### 常见优化 Bug 模式

在生产编译器开发中，以下 bug 模式反复出现：

1. **忘记更新 use 列表**：修改了指令但没有更新其他指令对该指令结果的引用
2. **支配关系被破坏**：将指令移到它不支配其 uses 的位置
3. **SSA 属性被破坏**：同一个虚拟寄存器被多次定义（在 Machine IR 中常见）
4. **副作用被错误忽略**：假设某指令没有副作用而删除它，但实际上它有
5. **Undef/Poison 值语义处理不正确**：混淆了 "undefined" 和 "poison" 的语义差异
6. **PHI 循环依赖**：在修改 PHI 节点时产生循环依赖

---

## AI 编译器关联

### MLIR 规范化 Pattern 与 LLVM instcombine

对于 AI 编译器工程师，理解 LLVM instcombine 的设计理念直接帮助你理解 MLIR 的规范化系统：

```
LLVM:  instcombine = 巨大的 switch(opcode) + 模式匹配 + 重写规则
MLIR:  canonicalize = 每个 operation 的 getCanonicalizationPatterns()
```

MLIR 的方法更具可扩展性，因为每个 dialect 可以定义自己的规范化规则。在 AI 编译器中：
- `linalg.generic` 的规范化可能涉及 tiling 模式识别
- `arith.addi` 的规范化类似于 LLVM 的常量折叠
- `scf.for` 的规范化可能涉及循环不变代码外提

### Triton 中的 SSA 使用

Triton 编译器的核心 IR 完全基于 SSA：
- Triton IR 使用结构化控制流，但底层仍然是 SSA 形式
- `tl.load` 和 `tl.store` 返回 SSA 值
- Triton 的优化 passes 使用与传统编译器相同的 use-def 链遍历技术

**Triton 特有的优化挑战**：
- Memory coalescing：多个 `tl.load` 可能合并为更高效的 coalesced load
- Shared memory promotion：自动将数据提升到 shared memory 并管理 bank conflicts
- 线程同步：在插入 barrier 时维护正确的 SSA 属性

### XLA/IREE 中的 Cost Model

AI 编译器的 cost model 比传统编译器更复杂：
- 除了指令延迟，还需要考虑数据传输成本（PCIe bandwidth、HBM bandwidth）
- Memory 布局（row-major vs column-major）对性能影响巨大
- 融合（fusion）决策需要在减少 kernel launch 开销和增加寄存器压力之间权衡

---

## 示例说明

### 示例1：常量传播的完整实现

常量传播是最基础但也是最重要的优化之一。在工业实践中，你应该：

```cpp
// 不要手动实现常量折叠——使用 LLVM 内置的 ConstantFoldInstruction
bool MyConstantFolder::runOnFunction(Function &F) {
    const DataLayout &DL = F.getParent()->getDataLayout();
    bool Changed = false;
    
    // 收集工作列表——不要边遍历边修改（会导致迭代器失效）
    SmallVector<Instruction *, 32> WorkList;
    for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
            WorkList.push_back(&I);
        }
    }
    
    // 处理工作列表
    for (Instruction *I : WorkList) {
        // 跳过可能已被删除的指令
        if (I->use_empty() && isInstructionTriviallyDead(I)) {
            I->eraseFromParent();
            Changed = true;
            continue;
        }
        
        // 尝试常量折叠
        if (auto *C = ConstantFoldInstruction(I, DL)) {
            // 使用 LLVM 的 SSAUpdater 或直接 replaceAllUsesWith
            I->replaceAllUsesWith(C);
            I->eraseFromParent();
            Changed = true;
        }
    }
    
    return Changed;
}
```

### 示例2：def-use 链遍历的完整示例

```cpp
// 遍历一个值的所有使用者（可能跨函数）
void analyzeUsers(Value *V) {
    outs() << "Analyzing uses of: " << *V << "\n";

    for (User *U : V->users()) {
        if (Instruction *I = dyn_cast<Instruction>(U)) {
            Function *UserFunc = I->getFunction();
            outs() << "  Used in function: " << UserFunc->getName()
                   << ", instruction: " << *I << "\n";
        } else if (Constant *C = dyn_cast<Constant>(U)) {
            outs() << "  Used in constant: " << *C << "\n";
            // 注意：常量的使用者可能在不同的函数中
        } else if (GlobalValue *GV = dyn_cast<GlobalValue>(U)) {
            outs() << "  Used in global: " << GV->getName() << "\n";
        }
    }
}
```

---

## 工业落地：一个优化怎样才允许合入

### 正确性门禁

1. **先写不应触发的反例**：有 `nsw/nuw/exact`、poison、undef、不同位宽、向量、地址空间和边界常量时会怎样？
2. **复用公共折叠器**：本章 `example1.cpp` 现在使用 `ConstantFoldInstruction`，避免手写 APInt 时把 `add nsw i8 127, 1` 错折成 `-128`；LLVM 正确结果是 poison。
3. **变换前后都验证**：输入非法时应立即拒绝，变换后用 verifier 捕获支配、类型和 CFG 错误。
4. **非平凡代数改写用 Alive2 或等价的形式化/差分验证**，不要仅凭几个手写样例判断正确。

### 工程验收顺序

```bash
# 1. IR 合法性和单 pass 回归
opt -passes='my-pass,verify' -S input.ll | FileCheck input.ll

# 2. 每个 pass 后验证，定位第一个破坏 IR 的阶段
opt -passes='default<O2>' -verify-each input.bc -o candidate.bc

# 3. baseline/candidate 使用同一输入做语义差分
# 4. 再比较编译时间、代码大小和目标机运行性能
```

FileCheck 只验证结构，不证明语义等价；benchmark 只证明某组输入上的性能，也不证明合法性。
这四类证据必须分开保存。对 `PreservedAnalyses` 拿不准时先返回 `none()`：分析失效声明错误可能让
后续 pass 使用陈旧结果，属于正确性问题，不只是编译速度问题。

## 总结

### 技术要点清单
- Value 是不可变的计算结果；Variable 是可变的存储位置——这是理解 SSA 的基础
- SSA 通过 PHI 指令（LLVM）或 block arguments（MLIR）处理控制流汇合
- 支配关系是 SSA 的数学基础：定义必须支配所有使用
- Def-use 链（use list）自动维护——但遍历时需注意跨函数引用和无序性
- mem2reg 将基于内存的 IR 转换为 SSA 形式，是管道中最重要的 pass 之一
- 优化必须经过合法性（preserves semantics）和有利性（improves performance）的双重检查
- NSW/NUW/FMF 等标志定义了允许哪些"不安全但合法"的优化
- `TargetTransformInfo` 是 LLVM IR 级别 cost model 的核心接口
- `ConstantFoldInstruction` 和 `replaceAllUsesWith` 是编写优化时的核心 API
- 使用 `isa<>`/`dyn_cast<>`/`cast<>` 而不是 C++ `dynamic_cast<>`

### 实践建议
1. **总是先运行 mem2reg**：在 LLVM IR 上做优化之前，确保 IR 在 SSA 形式
2. **利用 LLVM 内置的常量折叠**：不要重新发明 APInt 运算
3. **修改 IR 前复制迭代器**：`for (Instruction &I : make_early_inc_range(BB))` 避免迭代器失效
4. **使用 Alive2 验证非平凡优化**：即使你确定转换正确，验证也能发现边缘情况
5. **不要忽略 Poison 值的语义**：https://llvm.org/docs/LangRef.html#poison-values
6. **跨函数优化时注意 IPO（Inter-Procedural Optimization）的影响**：内联后优化机会大量出现
7. **编写测试时同时包含正面测试和负面测试**：确保优化仅在应该触发时触发

### 进一步学习方向
- 阅读 `llvm/lib/Transforms/InstCombine/` 源码——学习生产级别的指令合并模式
- 学习 Alive2（https://alive2.llvm.org/）——优化正确性的形式化验证
- 阅读 "Engineering a Compiler" 2nd Edition 的第 8-9 章（优化）
- 实践：为 MLIR 的 `arith` dialect 添加新的 canonicalization pattern
- 研究 LLVM 的 ConstantRange 分析——用于更精确的常量传播

### 工业界的优化实践
- **矢量化和并行化**：AI 编译器关注的是数据并行性，传统的 LLVM 循环向量化（LoopVectorize）和 SLP 向量化在 AI 场景中也有效
- **内联策略**：AI 编译器中的 JIT 场景通常使用更激进的内联策略，因为 JIT 编译时间是关键指标
- **Peephole 优化**：在 Machine IR 级别，PeepholeOptimizer 做局部的指令模式替换——这在固定指令集的 GPU 代码生成中尤其重要
