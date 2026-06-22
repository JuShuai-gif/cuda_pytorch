# Chapter 7: Understanding LLVM IR

## 核心概念（详细展开）

LLVM IR（LLVM Intermediate Representation）是 LLVM 编译基础设施的核心抽象层。对于从事 AI 编译器的工程师而言，深入理解 LLVM IR 不仅是为了读写 `.ll` 文件，更是为了理解 MLIR、Triton IR 等现代编译器 IR 如何设计与 LLVM IR 交互。

### LLVM IR 的哲学基础

LLVM IR 的设计遵循几个核心原则：
1. **SSA（Static Single Assignment）形式**：每个值只被定义一次。这大幅简化了数据流分析和优化算法的实现。
2. **低级但目标无关**：比 C 低级（有显式的内存操作、控制流），但比汇编高级（没有寄存器分配、没有指令编码）。
3. **类型安全**：强类型系统在编译时捕获大量错误，是 Pass verifier 的基础。
4. **可序列化**：同时支持文本格式（`.ll`）和二进制位码（`.bc`），后者具有向后兼容保证。

### 标识符的三个命名空间

| 前缀 | 命名空间 | 作用域 | C++ 基类 | 示例 |
|------|---------|--------|---------|------|
| `%` | 局部值 | 函数内 | `Value` | `%result`, `%1` |
| `@` | 全局值 | 模块级 | `GlobalValue` | `@main`, `@global_var` |
| `!` | 元数据 | 模块级 | `Metadata` | `!dbg !0`, `!llvm.loop` |

**生产注意**：未命名值（隐式变量）使用递增整数如 `%0`, `%1`，数字必须严格递增且不能跳号。手动修改 IR 时务必先运行 `opt -passes=instnamer` 消除隐式变量。

### 类型系统全貌

LLVM IR 的类型系统是其最强大的特性之一，对 AI 编译器尤其重要：

**单值类型（Single-value Types）**：
- `iN`（N 位整数）：`i1`（布尔）、`i8`、`i16`、`i32`、`i64`。N 最大可达 ~8,000,000（2^23）。
- 浮点类型：`half`（16位 IEEE-754）、`bfloat`（16位 BF16，AI 训练常用）、`float`（32位）、`double`（64位）、`fp128`。
- `ptr addrspace(N)`（不透明指针，LLVM 15+）：去除了指针的指向类型，简化了别名分析。
- 向量：`<N x type>`，如 `<4 x float>`、`<16 x i8>`。
- 可伸缩向量：`<vscale x N x type>`，用于 ARM SVE 和 RISC-V V 扩展。

**标签类型（Label）**：`label` 仅用于 `br` 和 `phi` 指令，表示基本块的地址。

**聚合类型（Aggregate Types）**：
- 结构体：`{type1, type2, ...}`（字面量）或 `%MyType = type { ... }`（命名）。
- 数组：`[N x type]`，如 `[3 x [4 x i32]]`。

**对 AI 编译器至关重要的类型**：
- `bfloat` 和 `half`：大量用于量化模型和混合精度训练
- `<vscale x N x type>`：GPU SIMT 模型中 warp 级别的向量化
- `ptr addrspace(N)`：GPU 的不同内存层次（global, shared, local, constant）

### 指令系统

LLVM IR 的指令集是固定的（不可扩展），分为几个类别：

**算术/逻辑**：`add`, `sub`, `mul`, `udiv`, `sdiv`, `urem`, `srem`, `shl`, `lshr`, `ashr`, `and`, `or`, `xor`
- NSW/NUW flags：控制有符号/无符号溢出行为
- FMF（Fast Math Flags）：控制浮点优化激进程度

**内存**：`alloca`, `load`, `store`, `getelementptr`（GEP）
- `alloca` 在 entry block 生成栈空间
- GEP 是最容易出错的指令之一：`getelementptr inbounds <Ty>, ptr %p, i64 idx1, i32 idx2` 是计算 `&p[idx1].field[idx2]` 的地址

**控制流**：`br`（条件/无条件分支）、`switch`、`ret`、`unreachable`
- `phi` 指令：`%result = phi i32 [val1, %block1], [val2, %block2]`

**类型转换**：`trunc`, `zext`, `sext`, `fptrunc`, `fpext`, `bitcast`, `inttoptr`, `ptrtoint`

**聚合操作**：`extractvalue`, `insertvalue`, `extractelement`, `insertelement`, `shufflevector`

**原子操作**：`cmpxchg`, `atomicrmw`

**其他**：`call`, `select`, `va_arg`, `landingpad`, `catchpad`, `cleanuppad`

### 目标特定渗入（Target-Specific Leakage）

LLVM IR 并非真正的"目标无关"——大量目标相关信息不可避免地渗入 IR：

1. **Intrinsics**：
   - 通用 intrinsics：`llvm.vector.reduce.add.*`, `llvm.memcpy.*`
   - 目标特定 intrinsics：`llvm.aarch64.neon.*`, `llvm.nvvm.*`（CUDA）

2. **Triple**：`target triple = "arch-vendor-os-environment"`。控制 ABI、端序、对象文件格式等。

3. **Data Layout**：`target datalayout = "e-p:16:16:16-i32:32:32..."`。控制类型大小、对齐、地址空间默认值、端序。

4. **函数属性**：`"target-cpu"`, `"target-features"`, `"tune-cpu"`。控制微架构特定的指令选择和调度。

5. **ABI 渗入**：前端在生成 LLVM IR 时已经根据目标 ABI 做了参数传递方式的改写（如大结构体通过指针传递 vs 直接返回值）。

**关键结论**：LLVM IR 不是跨目标可移植格式。将 AArch64 的 IR 直接用于 X86 可能在最佳情况下产生次优代码，最坏情况下产生 ABI 不一致的二进制。

## LLVM / MLIR 流程（深入）

### LLVM IR 在编译流水线中的位置

```
Source Code (C/C++/Rust/Swift/...)
  │
  ▼
Frontend (Clang/rustc/...) ─── 生成目标特定的 LLVM IR
  │                              （ABI lower、intrinsics 已在此阶段渗入）
  ▼
LLVM IR (.ll/.bc) ◀── 本章重点
  │
  ├── opt (优化 Pass 流水线)
  │   ├── 规范化 Pass (instcombine, mem2reg, LCSSA)
  │   ├── 分析 Pass (TTI, LoopInfo, AliasAnalysis, DominatorTree)
  │   └── 优化 Pass (inline, LICM, vectorize, ...)
  │
  ▼
Machine IR (MIR) ─── 指令选择、寄存器分配、指令调度
  │
  ▼
Assembly / Object File
```

### MLIR 视角：Dialect 层次与 LLVM IR 的定位

在 MLIR 的多级 Dialect 架构中，LLVM IR 对应最底层的 LLVM Dialect：

```
TensorFlow/PyTorch 计算图
  │
  ▼
TOSA / MHLO / Linalg Dialect (高层 MLIR)
  │
  ▼
Arith + Math + SCF + MemRef Dialects (中层 MLIR)
  │
  ▼
LLVM Dialect (底层 MLIR) ◀── 与 LLVM IR 一一对应
  │
  ▼
LLVM IR (可传递给 llc 或嵌入的 LLVM JIT)
```

MLIR 的 LLVM Dialect 提供了与 LLVM IR 完全对应的操作（`mlir::LLVM::AddOp` ↔ `add`, `mlir::LLVM::LoadOp` ↔ `load` 等），使得 MLIR 编译器可以利用 LLVM 的优化和代码生成基础设施。

### 文本格式 vs 位码格式

| 特性 | 文本格式 (.ll) | 位码格式 (.bc) |
|------|---------------|----------------|
| 人类可读性 | 是 | 否 |
| 文件大小 | 较大 | 紧凑（~1/10） |
| 向后兼容 | 不保证 | 保证（自动升级） |
| 使用场景 | 开发/调试/测试 | 分发/持久化/LTO |
| 修改方式 | 直接文本编辑 | opt 工具转换 |

**位码向后兼容的工作原理**：
```
旧版 .ll → 旧版 opt → 旧版 .bc → 新版 llvm-dis/opt（AutoUpgrade）→ 新版 .ll
```

AutoUpgrade 机制将旧的 IR 结构自动转换为新版本结构（例如，类型化指针升级为不透明指针）。

## 关键机制解析（工业视角）

### GEP（getelementptr）深入

GEP 是 LLVM IR 中最容易误用的指令之一。它计算地址，而非访问内存：

```llvm
; 结构体定义
%struct = type { i32, { float, double }, i64 }

; 访问 struct.field1.1 (即 double 字段)
; 索引: struct[0].field1[1]
%addr = getelementptr %struct, ptr %base, i64 0, i32 1, i32 1
; %addr = &base[0].field1[1]
```

GEP 的索引规则：
- 第一个索引（如 `i64 0`）是数组索引，按整个聚合类型的大小偏移
- 后续索引按字段在结构体中的偏移或数组元素大小偏移
- `inbounds` 关键字保证结果不超出分配对象边界

**生产 bug 案例**：在 GPU 编译器中，错误的 GEP 地址空间计算导致 global memory 的 load 访问了 shared memory 地址范围，产生随机结果而非崩溃——因为 GPU 不提供内存保护。

### undef vs poison

这是 LLVM IR 语义中最微妙的部分之一：

- **undef**：该值的比特位是"我们不关心的"任意模式。每次读取 undef 值的比特可能不同。undef 对应"未初始化"的语义，但行为不是未定义的。
- **poison**：读取 poison 值是未定义行为（UB）。poison 传播能力更强：任何依赖 poison 的指令也产生 poison。这是为了支持推测性执行优化。

**区别示例**：
```llvm
%x = add i32 %y, undef    ; %x 是 undef，但不触发 UB
%z = add i32 %y, poison   ; %z 是 poison，使用 %z 触发 UB
```

**AI 编译器相关性**：在 GPU 上，warp 内的线程可能部分活跃、部分不活跃。不活跃线程的寄存器值应视为 undef（可被任意值替代），而非 poison（不能传播）。

### 地址空间（Address Spaces）与 GPU 编译器

地址空间是 GPU 编译器最核心的 LLVM IR 概念之一：

```
地址空间 0 ─── Generic/Flat（统一寻址，如 AMD GCN）
地址空间 1 ─── Global Memory（全局显存）
地址空间 3 ─── Shared/Local Memory（线程块共享内存）
地址空间 4 ─── Constant Memory
地址空间 5 ─── Private/Local Memory（线程私有寄存器溢出区）
```

在 LLVM IR 中：
```llvm
%g = load float, ptr addrspace(1) %global_ptr   ; 全局内存 load
%s = load float, ptr addrspace(3) %shared_ptr    ; 共享内存 load
```

**MLIR 对应关系**：MLIR 的 GPU Dialect 使用 `gpu.address_space` 属性标注内存空间，最终 lowering 到 LLVM Dialect 时映射为地址空间号。

## AI 编译器关联

### MLIR Dialect 类型 vs LLVM IR 类型

MLIR 的类型系统比 LLVM IR 更丰富：

| 概念 | LLVM IR | MLIR |
|------|---------|------|
| 整数 | `i32` | `IntegerType::get(ctx, 32)` |
| 浮点 | `float` | `Float32Type::get(ctx)` |
| 张量 | 不适用（用数组模拟） | `Tensor<2x3xf32>` |
| 量化张量 | 不适用 | `!quant.uniform<u8:f32, 0.5:0>` |
| 内存 | `alloca` + `ptr` | `memref<2x3xf32>` |
| 函数 | `@func` | `func.func @func(...) -> (...) { }` |

**生产意义**：MLIR 的高层类型（Tensor、MemRef）在 lowering 到 LLVM Dialect 时会被降级为更基础的 LLVM 类型（数组、指针）。这个 lowering 过程由 MLIR 的 Conversion Patterns 完成，通常包含大量缓冲区分配、形状展开等操作。

### Triton IR vs LLVM IR

Triton 语言设计了自定义 IR（Triton IR），然后 lowering 到 MLIR 的 Triton Dialect，最终到 LLVM IR：

```
Triton Language (.py)
  │
  ▼
Triton IR（自定义 SSA IR）
  │
  ▼
Triton GPU Dialect（MLIR）
  │  - 包含 tile-level 操作（load, store, dot, reduce）
  │  - 显式编程模型（program_id, num_programs）
  ▼
LLVM Dialect（MLIR）
  │  - NVVM/AMDGPU intrinsics
  ▼
LLVM IR → PTX/AMDGCN Assembly
```

**Triton IR 相比 LLVM IR 的关键差异**：
- Triton IR 的 `tt.load` 和 `tt.store` 包含隐式的 masking 和 boundary checking
- Triton IR 有内置的 `tl.dot` 操作用于 tensor core 矩阵乘累加
- Triton IR 不暴露内存地址空间，由编译器自动管理 shared memory promotion

### IREE 的 IR Lowering 管道

IREE（MLIR-based 推理编译器）的完整 lowering 管道展示了现代 AI 编译器的层次结构：

```
TensorFlow Lite / PyTorch / JAX
  │
  ▼
MHLO / TOSA / StableHLO Dialects
  │
  ▼
Linalg-on-Tensors（高层计算描述）
  │
  ▼
Linalg-on-Buffers + Arith + SCF（中层，含内存分配）
  │
  ▼
LLVM Dialect + GPU Dialect / SPIRV Dialect
  │
  ▼
LLVM IR / SPIR-V Binary
```

在每个 lowering 阶段，IREE 使用 MLIR 的 pass pipeline 机制（类似 LLVM 的 pass manager）来执行转换和优化。

## 示例说明

### 示例 1：从 C 到 LLVM IR 的完整转换

```c
// input.c
int add_and_multiply(int a, int b) {
    int sum = a + b;
    return sum * 2;
}
```

```llvm
; output.ll (with optimizations)
define i32 @add_and_multiply(i32 %a, i32 %b) #0 {
entry:
  %add = add nsw i32 %a, %b
  %mul = shl i32 %add, 1      ; x * 2 → x << 1 (强度削减)
  ret i32 %mul
}
```

关键观察：
- 变量名（sum）消失了，SSA 形式不保留原始变量名
- `* 2` 被强度削减为 `shl ..., 1`
- `nsw` flag 表示加法不会 signed overflow

### 示例 2：ABI 渗入 IR

同一段 C 代码在不同目标上产生不同的 IR 签名：

```llvm
; AArch64: 返回小结构体通过值
define [2 x i64] @bigStructReturned()

; ARMv7: 返回小结构体通过隐式指针参数
define void @bigStructReturned(ptr sret(%BigStruct) align 4 %agg.result)
```

这说明 LLVM IR 在生成时就已经是目标特定的。跨目标的 IR 移植需要重新经过前端 lowering。

### 示例 3：GPU 地址空间标注

```llvm
; CUDA kernel 在 LLVM IR 中的表示
define void @my_kernel(ptr addrspace(1) %input, ptr addrspace(1) %output) {
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %g_addr = getelementptr float, ptr addrspace(1) %input, i32 %tid
  %val = load float, ptr addrspace(1) %g_addr
  ; ... 计算 ...
  store float %result, ptr addrspace(1) %output
  ret void
}
```

## 总结

1. **LLVM IR 的核心地位**：作为中端优化的唯一表示，LLVM IR 承载了所有的 target-independent 优化。理解 LLVM IR 是理解所有 LLVM-based 编译器（包括 MLIR-based）的基础。

2. **类型系统是优化和验证的基础**：LLVM IR 的强类型系统使得 Verifier 可以在编译时捕获大量结构错误。对 AI 编译器而言，`bfloat`、可伸缩向量和地址空间是最重要的类型概念。

3. **LLVM IR 并非真正可移植**：ABI、intrinsics、data layout、地址空间等目标信息在 IR 生成时就已经渗入。跨目标使用 IR 需要特别谨慎。

4. **MLIR 与 LLVM IR 的关系**：MLIR 的 LLVM Dialect 是 MLIR 与 LLVM 的桥梁。MLIR 的高层 Dialect（Tensor, Linalg）通过逐步 lowering 最终映射到 LLVM Dialect，从而利用 LLVM 的代码生成能力。

5. **AI 编译器关键知识点**：
   - 地址空间是 GPU 内存层次的核心抽象
   - GEP 是理解数组和张量访存模式的基础
   - Intrinsics 是目标特定硬件加速（如 Tensor Core）的接口
    - undef/poison 语义影响推测性优化在 GPU 上的正确性

---

## 附录：LLVM IR 深入参考与工业实践

### Data Layout 字符串完全解读

Data Layout 字符串定义了编译器对内存布局的所有假设。以 AArch64 的典型 data layout 为例：

```
e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n32:64-S128
```

逐段解读：
- `e`：little-endian（小端序）
- `p:64:64:64`：指针，ABI 对齐 64 位，首选对齐 64 位，大小 64 位
- `i1:8:8`：i1 类型，ABI 对齐 8 位（至少 1 字节）
- `i32:32:32`：i32 类型，对齐 32 位（4 字节自然对齐）
- `f32:32:32`：单精度浮点，32 位对齐
- `f64:64:64`：双精度浮点，64 位对齐
- `v64:64:64`：64 位向量，64 位对齐
- `v128:128:128`：128 位向量，128 位对齐
- `a0:0:64`：aggregate types 的对齐
- `n32:64`：原生支持的整数宽度（32 位和 64 位）
- `S128`：栈自然对齐（128 位 = 16 字节）

**Data Layout 对优化的影响**：
- `i32:128`（非自然对齐）→ 所有 i32 load/store 被标注 `align 16` → 无法利用非对齐访问指令 → 可能引入额外指令拆解非对齐访问
- `p:32:32`（32 位指针）→ 所有地址计算在 32 位进行 → GEP 和 inttoptr 的处理方式改变 → 影响 InstCombine 的规范化行为

### 函数属性和参数属性的完整分类

**函数属性（影响代码生成的关键属性）**：

| 属性 | 含义 | AI 编译器使用场景 |
|------|------|-----------------|
| `"target-cpu"="sm_80"` | 目标 NVIDIA GPU 代际 | CUDA kernel 编译 |
| `"target-features"="+ptx70,+sm_80"` | 启用特定硬件特性 | 控制指令集版本 |
| `"noinline"` | 禁止内联 | device function 边界标记 |
| `"alwaysinline"` | 强制内联 | 小 kernel 性能要求 |
| `"optnone"` | 禁止优化 | 调试 kernel |
| `"noreturn"` | 绝不正常返回 | trap 指令的语义建模 |
| `"nounwind"` | 不抛出异常 | GPU kernel（GPU 无异常机制） |
| `"convergent"` | 线程必须汇聚执行 | GPU SIMT 同步操作 |
| `"speculatable"` | 可以推测执行 | 允许将指令移动到条件分支外 |
| `"writeonly"` | 只写内存 | LICM 提升 store |
| `"readnone"` | 不读不写内存 | 纯函数优化 |
| `"argmemonly"` | 仅通过参数指针访问内存 | 别名分析 |

**`convergent` 属性的 GPU 重要性**：
在 GPU 的 SIMT 模型中，warp 内的所有线程必须到达同一点才能继续。如果编译器错误地将 `convergent` 操作（如 `__syncthreads()`）移动到条件分支外，将导致死锁。LLVM 的 `convergent` 属性明确告知优化器这类操作的约束。

### 内建函数（Intrinsics）的分类与 ID 命名规则

LLVM 的 intrinsics 命名遵循严格的规则：

```
llvm.[target_prefix].[intrinsic_name].[type_suffix]
```

- `llvm.memcpy.p0.p0.i64`：通用 intrinsic，操作指针地址空间 0，长度类型 i64
- `llvm.nvvm.barrier0`：NVIDIA 特定，barrier 操作
- `llvm.amdgcn.workitem.id.x`：AMD 特定，workitem ID 查询
- `llvm.riscv.vsetvli`：RISC-V 特定，V 扩展配置

**AI 编译器常用的 intrinsics**：

| Intrinsic | 用途 | 后端 |
|-----------|------|------|
| `llvm.nvvm.mma.m?n?k?` | Tensor Core MMA 操作 | NVIDIA |
| `llvm.amdgcn.mfma.*` | AMD Matrix Core 操作 | AMD |
| `llvm.nvvm.barrier0` | CTA 级同步屏障 | NVIDIA |
| `llvm.amdgcn.s.barrier` | Scalar barrier | AMD |
| `llvm.fmuladd.*` | 融合乘加（FMA） | 所有 |
| `llvm.vector.reduce.add.*` | 向量 reduction | 所有 |
| `llvm.sadd.sat.*` | 饱和加法 | 通用 |

### 位码（Bitcode）的自动升级机制

LLVM 位码文件的向后兼容性通过 `AutoUpgrade` 机制实现：

```
旧版本 .bc 文件读取
    │
    ▼
BitcodeReader 逐条解析
    │
    ▼
遇到旧格式指令/属性/类型
    │
    ▼
AutoUpgrade::UpgradeIntrinsicCall()
AutoUpgrade::UpgradeGlobalVariable()
AutoUpgrade::UpgradeFunctionAttributes()
    │
    ▼
转换为新版本的等价表示
    │
    ▼
加载为新版本的 Module
```

**常见的自动升级示例**：
- 类型化指针（`i32*`）→ 不透明指针（`ptr`）
- `llvm.memcpy` 的属性签名变化
- `!tbaa` 元数据的格式演进
- 指令 flags 的重命名

**生产经验**：在 CI 中缓存 `.bc` 文件（LTO 的中间产物）时务必记录 LLVM 版本号。新旧 `.bc` 文件的混用可能导致静默的错误代码生成，因为自动升级并非对所有改变都保持语义等价。

### GEP（getelementptr）的工业级使用模式

GEP 是 LLVM IR 中与内存布局最密切相关的指令。它的每个索引对应一个"解引用级别"：

```llvm
; 给定结构体
%MyStruct = type { i32, [3 x float], ptr }

; 访问 my_struct.field2（即 ptr 字段）
; 索引: struct[0].field2
%addr1 = getelementptr %MyStruct, ptr %base, i64 0, i32 2
; %addr1 = &base[0].field2

; 访问 my_struct.field1[2]（即 [3 x float] 中的第 3 个元素）
; 索引: struct[0].field1[2]
%addr2 = getelementptr %MyStruct, ptr %base, i64 0, i32 1, i64 2
; %addr2 = &base[0].field1[2]
```

**理解 GEP 索引的角色**：
- **第一个索引**（`i64 0`）：在整个结构体类型上的数组偏移（因为 `%base` 是指向结构体的指针，可以被视为指向结构体数组的第一个元素）
- **第二个索引**（`i32 2`）：在结构体内部的字段偏移（第 2 个字段，即 `ptr` 类型字段）
- **后续索引**：在子字段内部的进一步偏移

**GPU 编译器中的 GEP 优化**：
在 IREE 和 Triton 中，大量 GEP 用于计算张量元素的地址。由于 GPU 的寻址模式有限（通常只支持基址 + 偏移），编译器最终需要将所有 GEP 简化为 `ptr + offset` 的形式。LLVM 的 `SeparateConstOffsetFromGEP` pass 是这一优化的关键环节。

### 元数据（Metadata）系统

LLVM IR 的元数据系统（以 `!` 开头）用于携带不影响程序语义的附加信息：

```llvm
; 调试信息
!0 = !DIFile(filename: "test.c", directory: "/tmp")
!1 = !DILocation(line: 10, column: 5, scope: !2)

; 循环信息
!2 = distinct !{!2, !3}
!3 = !{!"llvm.loop.unroll.enable"}

; 别名分析（TBAA）
!4 = !{!"Simple C++ TBAA"}
!5 = !{!6, !6, i64 0}  ; 访问路径标签
```

**对 AI 编译器重要的元数据**：
- `!llvm.loop.unroll.enable/disable`：控制循环展开
- `!llvm.loop.vectorize.enable`：控制向量化
- `!alias.scope` / `!noalias`：细粒度别名控制
- `!invariant.group`：标记重复加载的不可变性
- `!tbaa`（Type-Based Alias Analysis）：基于类型的别名分析

### 验证器（Verifier）检查项目的完整列表

LLVM IR 的 Verifier 检查包括但不限于：

1. **SSA 属性检查**：
   - 每个 use 必须有对应的 def
   - 定义支配所有使用（Dominance property）

2. **类型一致性检查**：
   - 指令操作数类型必须匹配 Opcode 要求
   - `br i1` 的条件必须是 `i1` 类型
   - `phi` 的所有输入类型一致

3. **控制流检查**：
   - 每个基本块有且仅有一个 terminator
   - Branch 目标必须是已定义的基本块

4. **模块级检查**：
   - `main` 函数的签名
   - 全局变量的初始化器类型匹配

5. **属性验证**：
   - 函数属性与函数内容一致（`noreturn` 的函数不能有 `ret`）
   - inttoptr 和 ptrtoint 的类型关系

6. **链接类型检查**：
   - `internal` 函数不能有声明
   - `external` 函数必须有定义或被声明

**生产经验**：在开发新 pass 时，始终在每个修改 IR 的函数末尾调用 `verifyFunction()`。90% 的 pass bug 会被 verifier 捕获，从而避免更难追踪的下游错误。
