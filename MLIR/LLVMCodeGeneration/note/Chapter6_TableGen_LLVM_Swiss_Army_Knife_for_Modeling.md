# Chapter 6: TableGen - LLVM Swiss Army Knife for Modeling

## 核心概念（详细展开）

TableGen 是 LLVM 生态系统中最重要的领域特定语言（DSL）之一，其核心目标是通过声明式编程来生成高度重复的 C++ 样板代码。对于生产级编译器工程师而言，理解 TableGen 不仅是能读懂 `.td` 文件，更是理解编译器信息如何从高层描述流向后端代码生成的关键环节。

### TableGen 的核心抽象层次

1. **Record（记录）**：TableGen 最基本的原子单位。一个 Record 是有名字的实体，包含若干有类型的字段（field）。所有 TableGen 处理的起点都是将输入文件转换为扁平化的 Record 集合。

2. **Class（类）**：Record 的模板/类型系统。Class 定义了字段及其默认值，可以带参数（类似 C++ 模板）。与 C++ 继承不同，TableGen 的 Class 支持多重继承，且字段名冲突时"最后定义者胜出"——没有虚函数、没有 disambiguation，这是一个常见的生产陷阱。

3. **Def（定义）**：对 Class 的具体实例化。`def` 是创建 Record 的核心关键字。匿名的 `def`（不提供名字）会被自动赋予 `anonymous_N` 的形式。

4. **Multiclass（多类）**：`multiclass` + `defm` 组合允许一次实例化多个 Record。在真实 LLVM 后端中，这是最常用的模式——例如描述一条指令的多种格式变体（寄存器-寄存器、寄存器-立即数等）。

5. **Let 语句**：字段覆写机制。`let ... in { }` 可以对作用域内的所有 Record 统一设置字段值。理解 let 的求值顺序（非 let 赋值先于 let 赋值，let 内部从上到下）是调试 TableGen 行为的关键。

6. **Bang Operators（! 运算符）**：TableGen 的编程能力来源。`!add`, `!sub`, `!range`, `!filter`, `!foreach`, `!size`, `!strconcat` 等提供了在声明式描述中嵌入简单逻辑的能力。

### 类型系统

TableGen 的内建类型反映了其作为编译器建模 DSL 的设计意图：

- `int`：64位整数。所有算术运算都在 64 位上进行。
- `bit`：单比特（0 或 1）。
- `bits<N>`：N 位位向量，用于指令编码描述最常见。
- `string`：字符串，用于标识符、助记符等。
- `list<T>`：类型 T 的列表，支持 `!foreach` 迭代。
- `dag`：有向无环图表达式，结构为 `(operator child1, child2, ...)`。这是指令选择模式匹配的核心概念——描述如 `(add (mul a, b), (div c, d))` 这样的表达式树。
- `code`：代码片段，直接嵌入生成的代码中。

**生产注意**：Records 之间通过引用而非拷贝关联。这既是性能优势（避免大量复制），也是常见 bug 来源——当你修改一个 Record 的字段时，所有引用该 Record 的地方都会受到影响。

## LLVM / MLIR 流程（深入）

### TableGen 在 LLVM 中的完整数据流

```
.td 源文件（开发者编写）
    ↓
llvm-tblgen 前端（解析 + 扁平化 → RecordKeeper）
    ↓
TableGen 后端（遍历 RecordKeeper，应用后端逻辑）
    ↓
.inc 文件（C++ 代码片段，以宏守卫分段）
    ↓
CMake tablegen() 函数（构建时调用 llvm-tblgen）
    ↓
C++ 源码中的 #include "XXX.inc"（在不同上下文中展开）
```

### .inc 文件的结构模式

LLVM 生成的所有 `.inc` 文件遵循统一的模式：

```cpp
#ifdef GET_INTRINSIC_IITINFO
// 第一段：类型枚举表
#endif

#ifdef GET_INTRINSIC_NAME_TABLE
// 第二段：名称映射表
#endif

#ifdef GET_INTRINSIC_TARGET_DATA
// 第三段：目标数据
#endif
```

**为什么用宏守卫而不是头文件？** 这是 LLVM 的设计惯例——同一份 `.inc` 文件可以在不同的 C++ 文件中以不同的宏上下文被包含，从而生成不同的代码。这种模式减少了文件数量，但增加了理解成本。在生产代码中，追踪 `.inc` 的包含点需要 `git grep`。

### 不同驱动的 TableGen 后端

| 驱动工具 | 项目归属 | 典型用法 |
|---------|---------|---------|
| `llvm-tblgen` | LLVM Core | 指令描述、寄存器、intrinsics、GlobalISel |
| `clang-tblgen` | Clang | 命令行选项、诊断信息、built-in 函数 |
| `mlir-tblgen` | MLIR | Dialect 定义、Op 接口、Rewrite Patterns |

**关键区别**：所有驱动共享相同的 TableGen 前端解析器（语法一致），但提供不同的后端。这意味着学会一种 `.td` 文件的语法后，可以在所有 LLVM 子项目中使用。

### MLIR ODS（Operation Definition Specification）vs Traditional TableGen

MLIR 的 Operation Definition Specification（ODS）是 TableGen 在 MLIR 中的核心应用，体现了从"后端描述"到"IR 构建"的范式转变：

**LLVM TableGen 的主要用途**：
- 描述目标硬件特性（寄存器文件、指令集、调用约定）
- 生成选择表（SelectionDAG、GlobalISel）
- 定义 intrinsics

**MLIR ODS 的主要用途**：
- 定义 Dialect 中的 Operation（操作）
- 生成 C++ 操作类（builder、verifier、parser、printer、folder、canonicalizer）
- 定义 Dialect 的 Type 和 Attribute
- 描述 Rewrite Patterns（DRR - Declarative Rewrite Rules）

**关键差异**：

| 维度 | LLVM TableGen | MLIR ODS |
|------|--------------|----------|
| 目标 | 描述目标后端 | 描述 IR 本身 |
| 核心类 | `Instruction`, `Register`, `SubtargetFeature` | `Op`, `Type`, `Attr`, `Pattern` |
| 生成产物 | 查找表、枚举、选择器 | Op 的完整 C++ 类层次 |
| 编译时机 | 构建时（build time） | 构建时（build time） |
| 可扩展性 | 通过后端扩展 | 通过 Dialect + Traits 组合 |

**工业实践**：在 MLIR 中定义一个 Op 只需写：
```tablegen
def MyDialect_MyOp : MyDialect_Op<"my_op"> {
  let summary = "My custom operation";
  let arguments = (ins AnyType:$input, I64Attr:$factor);
  let results = (outs AnyType:$output);
  let hasVerifier = 1;
}
```
然后 MLIR 的 `mlir-tblgen -gen-op-defs` 自动生成 ~200 行的 C++ 样板代码，包括 builder、parser、printer、verifier、folder 等。这使得 AI 编译器团队可以快速迭代 Dialect 设计。

### Triton 如何使用 TableGen 类模式

Triton（OpenAI 的 GPU 编程语言）虽然不是 LLVM 子项目，但其编译器大量借鉴了 TableGen 的设计思想：

1. **Ops 定义**：Triton 的 `TritonOps.td` 使用类似 MLIR ODS 的 TableGen 风格定义 Triton Dialect 的操作。
2. **Rewrite Patterns**：Triton 使用 MLIR 的 `PatternRewriter` 和声明式重写规则（DRR）来执行 Triton IR → Triton GPU IR → LLVM IR 的 lowering。
3. **C++ Code Generation**：类似 TableGen 后端生成 C++，Triton 使用 Python-based Triton DSL 生成 CUDA/ROCm kernel 代码。

## 关键机制解析（工业视角）

### 内建函数的 TableGen 生命周期

以 LLVM IR Intrinsic 为例，完整的生命周期：

1. **定义阶段**（`IntrinsicsXXX.td`）：
```tablegen
let TargetPrefix = "h2blb" in {
  def int_h2blb_widening_smul :
    Intrinsic<[llvm_i32_ty], [llvm_i16_ty, llvm_i16_ty]>;
}
```

2. **生成阶段**（`llvm-tblgen -gen-intrinsic-enums`）：
- 生成 `IntrinsicsXXX.h` 包含 `Intrinsic::ID` 枚举
- 生成 `IntrinsicImpl.inc` 包含各种宏分段（名称表、类型信息、Clang 映射等）

3. **构建集成**：
```cmake
# llvm/include/llvm/IR/CMakeLists.txt
tablegen(LLVM IntrinsicsH2BLB.h -gen-intrinsic-enums -intrinsic-prefix=h2blb)
```

4. **编译使用**：
- `Intrinsics.cpp` 包含生成的 `.inc` 文件
- Clang 通过 `Intrinsic::getIntrinsicForClangBuiltin` 查询 ID
- 前端调用 `CodeGenFunction::EmitTargetBuiltinExpr` 生成 intrinsics 调用

**生产调试经验**：intrinsics 的 TableGen 定义中 `IntrArgMemOnly` vs `IntrWriteMem` 这样的属性直接影响优化器的行为。错误地标注 memory effects 会导致：
- 错误的死代码消除（把有副作用的 intrinsic 消除了）
- 错误的指令调度（把不可移动的 intrinsic 移到了条件分支外面）
- LLVM 不会报错，但会在运行时产生错误结果——这是最危险的 bug

### TableGen 后端开发与调试方法论

**定位故障组件**：

```
1. 获取失败的命令行（从构建日志中提取）
2. 移除 --gen-xxx 选项，添加 --print-records
3. 重新运行：
   - 如果报错消失 → 后端问题
   - 如果报错仍在 → 前端/语法问题
```

**后端调试技巧**：

- 使用 `-debug` 选项启用后端的调试日志（在 llvm-tblgen 命令行后添加）
- 查找 `RecordKeeper::getAllDerivedDefinitions` 调用——这揭示了哪些 Record 和 Class 对该后端是"承重"（load-bearing）的
- 所有后端的入口是 `XXXEmitter::run` 方法（位于 `llvm/utils/TableGen/` 下）
- 使用 `git grep 'class ClassName' -- llvm/include/llvm | grep '\.td'` 查找类的定义文件

**生产环境中的常见 TableGen 问题**：

1. **include 循环/重复**：`.td` 文件没有 include guard，重复包含会报重复定义错误。使用 include 顺序管理依赖是唯一方案。
2. **字段名冲突**：多重继承时字段名冲突不会被报告为错误，而是静默覆盖。这是 LLVM 代码审查时需要特别关注的点。
3. **后端静默失败**：某些 TableGen 后端在遇到不完整定义时不会报错，而是生成错误的 C++ 代码，导致后续编译失败。
4. **构建顺序依赖**：`tablegen()` 的 CMake 调用必须在 `.inc` 文件被使用前完成。`add_public_tablegen_target(intrinsics_gen)` 创建独立的构建目标。

### CMake 中的 TableGen 集成模式

在 LLVM 中，TableGen 的集成模式已经成为其他项目的参考范本：

```cmake
# 基础模式
tablegen(LLVM MyGenFile.inc -gen-my-backend)

# 多文件输入
tablegen(LLVM MyGenFile.inc -gen-my-backend
  SOURCE MyDesc.td
  TARGET MyGenTarget)

# 独立构建目标（加速迭代）
add_public_tablegen_target(my_backend_gen)
# 然后可以单独执行: ninja my_backend_gen
```

**生产优化**：
- 将 TableGen 工具本身用 Release 模式构建（`LLVM_OPTIMIZED_TABLEGEN=ON`），可显著加速构建时间（在大型后端上可节省 50%+ 的 TableGen 运行时间）
- 将 `.td` 文件分层次组织：`TargetBase.td`（通用类）→ `TargetCommon.td`（目标族共享）→ `Target.td`（具体目标）

## AI 编译器关联

### MLIR ODS vs TableGen：操作定义的演进

MLIR ODS 代表了 TableGen 思想在 AI 编译器领域的自然延伸：

1. **DRR（Declarative Rewrite Rules）**：MLIR 的声明式重写规则是 TableGen DAG 模式匹配的进化版，允许直接在 `.td` 文件中描述 IR 变换：

```tablegen
def EliminateIdentityOp : Pat<
  (MyDialect_IdentityOp $input),
  (replaceWithValue $input)
>;
```

2. **Dialect 组合能力**：MLIR 允许通过 TableGen 定义 Traits（`HasParent`, `SameOperandsAndResultType` 等），这些 Traits 在编译期组合到 Op 上，生成对应的 verifier 逻辑。

3. **量化 Dialect 案例**：TFLite 和 IREE 的量化 Dialect 大量使用 ODS 定义 per-channel、per-tensor 等不同的量化参数类型，这是纯 C++ 建模难以维护的。

### Triton 中的 Op 定义模式

Triton 编译器在 `include/triton/Dialect/Triton/IR/` 目录下使用 `.td` 文件定义 Triton Dialect 的所有 Op：

```tablegen
def TT_LoadOp : TT_Op<"load", [MemoryEffects<[...]>]> {
  let arguments = (ins TT_PtrLike:$ptr, TT_MaskLike:$mask, ...);
  let results = (outs TT_Type:$result);
}
```

这种模式使得 Triton 团队可以快速添加新的 GPU 操作，同时自动获得 verifier、canonicalizer 和打印机/解析器的生成。

### IREE 编译栈中的 TableGen 角色

IREE 的 HAL（Hardware Abstraction Layer）Dialect 使用 TableGen 定义后端无关的硬件抽象操作。当添加新的后端（如 Vulkan、CUDA、ROCM）时，只需在各自的 Target dialect 中定义 lowering patterns，而无需修改中间层 IR。

## 示例说明

### 示例 1：定义一个简单的指令集（完整）

```tablegen
// 定义指令基类
class Inst<string mnemonic, bits<8> opcode> {
  string Mnemonic = mnemonic;
  bits<8> Opcode = opcode;
  dag OutOperandList;   // 输出操作数
  dag InOperandList;    // 输入操作数
  string AsmString;     // 汇编字符串
  list<dag> Pattern;    // 选择模式列表
}

// 定义寄存器-寄存器和寄存器-立即数指令（一次定义）
multiclass ArithOp<bits<8> baseOpcode, string mnemonic, SDPatternOperator op> {
  def _rr : Inst<mnemonic # ".rr", baseOpcode> {
    let OutOperandList = (outs GPR:$rd);
    let InOperandList = (ins GPR:$rs1, GPR:$rs2);
    let AsmString = mnemonic # "\t$rd, $rs1, $rs2";
    let Pattern = [(set GPR:$rd, (op GPR:$rs1, GPR:$rs2))];
  }
  def _ri : Inst<mnemonic # ".ri", baseOpcode> {
    bits<8> imm;
    let Opcode = baseOpcode;
    let InOperandList = (ins GPR:$rs1, imm16:$imm);
    let AsmString = mnemonic # "\t$rd, $rs1, $imm";
    let Pattern = [(set GPR:$rd, (op GPR:$rs1, imm16:$imm))];
  }
}

defm ADD : ArithOp<0x01, "add", add>;
// 生成: ADD_rr 和 ADD_ri
```

### 示例 2：TableGen 后端开发骨架

```cpp
// 一个简单的自定义 TableGen 后端
static bool MyBackendEmitter(raw_ostream &OS, RecordKeeper &Records) {
  // 获取所有派生自特定 class 的 record
  auto Recs = Records.getAllDerivedDefinitions("MyClass");
  
  // 生成宏守卫
  OS << "#ifdef GET_MY_GEN_DATA\n";
  
  for (auto *R : Recs) {
    std::string Name = R->getName().str();
    int Value = R->getValueAsInt("MyField");
    OS << "  {" << Name << ", " << Value << "},\n";
  }
  
  OS << "#endif // GET_MY_GEN_DATA\n";
  return false; // false = success
}
```

## 总结

1. **TableGen 的核心价值**：通过声明式 DSL 避免手写大量重复的模板代码，将关注点从"如何生成代码"转移到"描述什么"。

2. **MLIR ODS 是 TableGen 的进化**：MLIR 的 ODS 框架将 TableGen 的思想应用到 IR 操作的定义和重写上，这是 AI 编译器构建的核心生产力工具。

3. **生产级的 TableGen 使用**：
   - 理解后端的工作方式（RecordKeeper → 特定 Record → 遍历生成）
   - 掌握 `.inc` 文件的多段包含机制
   - 熟练使用 `-print-records` 和 `-debug` 选项调试
   - 利用 `ninja intrinsics_gen` 等独立目标加速迭代

4. **常见陷阱**：
   - 多重继承的字段覆盖规则（最后定义者胜出）
   - 没有 include guard 的 `.td` 文件
   - Record 引用而非拷贝的语义
   - Intrinsic 属性错误标注导致的静默 bug

5. **AI 编译器关联**：
   - MLIR Dialect 定义 = ODS TableGen
   - Triton Op 定义 = MLIR-style TableGen
   - IREE Target 后端 = TableGen 生成的 lowering patterns
   - 学习 TableGen 是掌握 MLIR 和现代 AI 编译器栈的基础

---

## 附录：TableGen 深入实践与常见陷阱

### TableGen 中的 Record 引用语义陷阱

TableGen 中 Record 之间是引用关系而非拷贝。这意味着：

```tablegen
class Base {
  int Value = 0;
}
def A : Base { let Value = 10; }
def B : Base { let Value = A.Value; }  // ← 引用 A.Value，而不是拷贝 10
// 如果 A.Value 后来被 let 覆盖，B.Value 也会改变！
```

在生产代码中这种引用语义导致的 bug 极难追踪，因为 Record 可能在多个不同的 `.td` 文件中被修改。

### 多类（Multiclass）与名称空间管理

`multiclass` 是 LLVM 后端中使用最频繁的 TableGen 特性之一。以 RISC-V 后端为例，所有指令变体通过 multiclass 定义：

```tablegen
multiclass RVInstR<bits<7> opcode, string opcodestr, SDPatternOperator op> {
  def _RR : RVInstRBase<opcode, opcodestr, op>;
  def _RI : RVInstIBase<opcode, opcodestr, op>;
  // 变体3：带浮点四舍五入模式
  def _RM : RVInstRBase<opcode, opcodestr, op> { let hasRoundingMode = 1; }
}
defm ADD : RVInstR<0x33, "add", add>;
// 生成: ADD_RR, ADD_RI, ADD_RM
```

**生产经验**：当 multiclass 嵌套超过 2 层时，记录名可能变得不可读。此时应借助 `defvar` 控制命名前缀或分拆为多个更小的 multiclass。

### let-in 作用域的求值顺序详解

`let` 语句的求值顺序遵循以下规则：

1. 所有非 `let` 的赋值首先求值（class constructor 参数、类体中的直接赋值）
2. `let` 语句按文件中从上到下的顺序求值
3. 同名的多个 `let` 赋值，最后的覆盖前面的
4. `let ... in { def X ... }` 的作用域延伸到 `{ }` 内的所有 `def`

```tablegen
class Foo<string _name = "default"> {
  string name = _name;
}
let name = "outer" in {
  let name = "inner" in
  def A : Foo<"from_ctor">;   // → "inner"（最后覆盖）
  def B : Foo<>;               // → "outer"（继承外层 let）
}
def C : Foo<"from_ctor"> {
  let name = "from_body";     // → "from_body"（类体内 let 优先于外部）
}
```

### 构建 TableGen 的 CMake 高级模式

**分离 TableGen 构建目标**（加速迭代开发）：

```cmake
# 定义 TableGen 输入文件
set(LLVM_TARGET_DEFINITIONS AArch64.td)
# 生成多个 .inc 文件
tablegen(LLVM AArch64GenRegisterInfo.inc -gen-register-info)
tablegen(LLVM AArch64GenInstrInfo.inc -gen-instr-info)
tablegen(LLVM AArch64GenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM AArch64GenDAGISel.inc -gen-dag-isel)
tablegen(LLVM AArch64GenGlobalISel.inc -gen-global-isel)
# 分别暴露独立构建目标
add_public_tablegen_target(AArch64CommonTableGen)
# 然后可以: ninja AArch64CommonTableGen
```

**大型项目中的 TableGen 加速策略**：
- `LLVM_OPTIMIZED_TABLEGEN=ON`：用 Release 模式编译 `llvm-tblgen` 工具本身
- `LLVM_TABLEGEN=<path>`：指定预编译的 tblgen 可执行文件
- 使用 `ninja intrinsics_gen` 或 `ninja AArch64CommonTableGen` 只重建所需的 `.inc` 文件

### TableGen 后端开发完整示例

完整的 TableGen 后端需要：

1. 在 `llvm/utils/TableGen/` 下创建 `XXXEmitter.h` 和 `XXXEmitter.cpp`
2. 注册后端到 `llvm-tblgen` 的命令行处理
3. 实现 `EmitXXX` 函数：

```cpp
// 后端入口
void EmitRegisterInfo(RecordKeeper &RK, raw_ostream &OS) {
  // 获取所有 register class 的 records
  auto RegClasses = RK.getAllDerivedDefinitions("RegisterClass");
  
  // 按名称排序（保证确定性输出）
  sort(RegClasses, [](const Record *A, const Record *B) {
    return A->getName() < B->getName();
  });
  
  // 生成 C++ 代码
  OS << "namespace llvm {\n";
  OS << "namespace " << TargetName << " {\n";
  
  // 生成寄存器类 ID 枚举
  OS << "enum {\n";
  for (auto *RC : RegClasses) {
    OS << "  " << RC->getName() << "RegClassID,\n";
  }
  OS << "};\n\n";
  
  // 生成寄存器类描述表
  OS << "const TargetRegisterClass *getRegClass(unsigned i) {\n";
  OS << "  static const TargetRegisterClass *Table[] = {\n";
  for (auto *RC : RegClasses) {
    // 从 record 获取字段值
    int NumRegs = RC->getValueAsInt("NumRegs");
    std::string Name = RC->getValueAsString("Name");
    OS << "    &" << Name << "RegClass,  // " << NumRegs << " regs\n";
  }
  OS << "  };\n";
  OS << "  return Table[i];\n";
  OS << "}\n";
  
  OS << "} // namespace " << TargetName << "\n";
  OS << "} // namespace llvm\n";
}
```

4. 在 `CMakeLists.txt` 中注册：
```cmake
# llvm/utils/TableGen/CMakeLists.txt
set(LLVM_LINK_COMPONENTS Support)
add_tablegen(llvm-tblgen DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
  XXXEmitter.cpp
  ...
)
```

### MLIR ODS 的高级模式

MLIR 的 ODS 支持许多高级建模模式，远超基础 LLVM TableGen：

**Type Constraints（类型约束）**：
```tablegen
def F32OrF64 : AnyTypeOf<[F32, F64]>;
def MyOp : Op<...> {
  let arguments = (ins F32OrF64:$input);  // 限制 input 为 f32 或 f64
}
```

**Builder Methods（构造方法）**：
```tablegen
def MyOp : Op<...> {
  let builders = [
    OpBuilder<(ins "Value":$operand), [{
      // 自定义 C++ builder 代码
      build($_builder, $_state, operand.getType(), operand);
    }]>
  ];
}
```

**Declarative Format（声明式格式）**：
```tablegen
def MyOp : Op<...> {
  let assemblyFormat = "$input `=>` $output attr-dict `:` type($input)";
  // 自动生成 parser/printer（无需手写 ~100 行 C++ 代码）
}
```

这些 MLIR ODS 特性使得定义一个新的 Dialect（包含 20+ Ops）只需 200-300 行 TableGen 代码，而等效的手写 C++ 需要 2000+ 行。
