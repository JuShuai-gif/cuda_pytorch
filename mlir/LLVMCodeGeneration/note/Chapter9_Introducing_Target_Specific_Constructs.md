# Chapter 9: Introducing Target-Specific Constructs

## 核心概念（详细展开）

本章以虚构的 H2BLB（How to Build an LLVM Backend）目标为例，完整展示如何从零创建 LLVM 后端并连接到整个工具链。对于 AI 编译器工程师而言，本章的内容直接关系到如何为新的 AI 加速器（NPU、TPU、自定义 GPU 架构）创建编译器后端。

### 创建新后端的六个阶段

1. **构建系统集成**：在 LLVM 源码树中创建目录结构，注册到 CMake
2. **Triple 和 TargetRegistry 注册**：为后端创建唯一的架构标识
3. **TargetMachine 实现**：提供数据布局、重定位模型等基本信息
4. **Clang 前端连接**：实现 `clang::TargetInfo` 子类
5. **Intrinsics 创建**：定义 LLVM IR 级和 Clang 级的内建函数
6. **TargetTransformInfo 实现**：提供目标特定的成本模型

### 构建集成的技术细节

**目录结构规范**（以 H2BLB 为例）：

```
llvm/lib/Target/H2BLB/
├── CMakeLists.txt                   # 顶层构建配置
├── H2BLBTargetMachine.h/.cpp        # TargetMachine 实现
├── MCTargetDesc/                    # MC 层组件
│   ├── CMakeLists.txt
│   └── H2BLBMCTargetDesc.cpp       # LLVMInitializeH2BLBTargetMC
└── TargetInfo/                      # Target 信息
    ├── CMakeLists.txt
    ├── H2BLBTargetInfo.h/.cpp       # LLVMInitializeH2BLBTargetInfo
```

**三个必须提供的初始化函数**：

```cpp
extern "C" {
  void LLVMInitializeH2BLBTargetInfo();  // 注册 Target 实例
  void LLVMInitializeH2BLBTargetMC();    // 注册 MC 组件
  void LLVMInitializeH2BLBTarget();      // 注册 TargetMachine
}
```

**LLVM 特定的 CMake 函数**：
- `add_llvm_component_group(H2BLB)`：创建组件聚合名
- `add_llvm_target(H2BLBCodeGen ...)`：创建目标代码库
- `add_llvm_component_library(LLVMH2BLBInfo ...)`：创建子组件库
- `LINK_COMPONENTS`：声明依赖的 LLVM 库
- `ADD_TO_COMPONENT`：将目标添加到聚合组件

**生产注意**：`LINK_COMPONENTS` 列出的依赖不仅用于链接，还影响 CMake 的构建顺序。遗漏一个间接的头文件依赖（如从 Support 库引用了头文件但未列出 Support）可能导致构建在这样的环境中失败：库 A 先于库 B 构建但 B 的某些类型尚未被处理。

### Triple 和 TargetRegistry

**Triple 架构枚举的修改**：
1. 在 `Triple.h` 的 `ArchType` 枚举中添加 `h2blb`
2. 在 `Triple.cpp` 中填充所有 `switch(ArchType)` 分支：
   - `getArchTypeName`：返回 "h2blb"
   - `getArchTypePrefix`：返回 "h2blb"
   - `isLittleEndian`：返回 true/false
   - `getArchPointerBitWidth`：返回指针位宽（如 16）
   - `getDefaultFormat`：返回对象文件格式（COFF/ELF/Mach-O）

**Target 实例的注册**：

```cpp
Target &llvm::getTheH2BLBTarget() {
  static Target TheH2BLBTarget;
  return TheH2BLBTarget;
}

// 在 LLVMInitializeH2BLBTargetInfo 中：
RegisterTarget<Triple::h2blb, /*HasJIT=*/false> X(
    getTheH2BLBTarget(), "h2blb",
    "How to build an LLVM backend by example", "H2BLB");
```

`RegisterTarget` 是一个 RAII 对象，在构造时将 Target 实例注册到 `TargetRegistry`。它的第一个模板参数是 `ArchType`，第二个模板参数指示是否支持 JIT。

### TargetMachine 构造函数参数详解

```cpp
H2BLBTargetMachine(const Target &T, const Triple &TT,
                   StringRef CPU, StringRef FS,
                   const TargetOptions &Options,
                   std::optional<Reloc::Model> RM,
                   std::optional<CodeModel::Model> CM,
                   CodeGenOptLevel OL, bool JIT);
```

| 参数 | 类型 | 含义 | AI 编译器相关示例 |
|------|------|------|-----------------|
| `CPU` | `StringRef` | CPU 代际名 | "sm_80" (A100), "gfx90a" (MI250X) |
| `FS` | `StringRef` | 特性字符串 | "+ptx70,+sm_80,-fp16" |
| `Options` | `TargetOptions` | 默认行为控制 | FMF 默认值、异常处理 |
| `RM` | `Reloc::Model` | 重定位模型 | Static/PIC/DynamicNoPIC |
| `CM` | `CodeModel::Model` | 代码模型 | Small/Large/Kernel |
| `OL` | `CodeGenOptLevel` | 优化级别 | O0-O3 |
| `JIT` | `bool` | 是否 JIT 编译 | 影响流水线复杂性 |

### Intrinsics 的完整生命周期

**LLVM IR 层面（TableGen 定义）**：

```tablegen
// IntrinsicsH2BLB.td
let TargetPrefix = "h2blb" in {
  def int_h2blb_widening_smul :
    Intrinsic<[llvm_i32_ty], [llvm_i16_ty, llvm_i16_ty]>;
}
```

**Clang 层面（新方法 - TableGen）**：

```tablegen
// BuiltinsH2BLB.td
def WideningSignedMultiply : TargetBuiltin {
  let Spellings = ["__builtin_h2blb_widening_smul"];
  let Attributes = [NoThrow, Const];
  let Prototype = "int(short, short)";
}
```

**自动一对一映射**（使用 `ClangBuiltin` TableGen 类）：

```tablegen
class H2BLB_Intrinsic<string suffix, list<LLVMType> ret_types,
                             list<LLVMType> param_types>
  : ClangBuiltin<!strconcat("__builtin_h2blb_", suffix)>,
    DefaultAttrsIntrinsic<ret_types, param_types>;
```

**N-to-M 映射的处理**（如 `llvm_anyfloat_ty` 对应多个 C 类型）：需要在 Clang 的 `CodeGenFunction::EmitTargetBuiltinExpr` 中手动实现类型分发。

## LLVM / MLIR 流程（深入）

### 新后端在工具链中的连接点

```
┌─ Clang Frontend ─────────────────────────────────────────┐
│  TargetInfo::getTargetBuiltins()                         │
│  TargetInfo::validateAsmConstraint()                     │
│  CodeGenFunction::EmitTargetBuiltinExpr()  ← intrinsics  │
│  AllocateTarget() switch → TargetInfo 子类                │
└───────────────────────┬──────────────────────────────────┘
                        ▼
┌─ LLVM Middle-End ───────────────────────────────────────┐
│  TargetTransformInfo::getInstructionCost()   ← 成本模型  │
│  TargetTransformInfo::getLoadVectorFactor()              │
│  PassBuilder callbacks → registerPassBuilderCallbacks()  │
└───────────────────────┬──────────────────────────────────┘
                        ▼
┌─ LLVM Backend ──────────────────────────────────────────┐
│  TargetMachine::addPassesToEmitFile()      ← 代码生成    │
│  TargetPassConfig::addIRPasses()                         │
│  TargetPassConfig::addInstSelector()                     │
│  TargetLowering::LowerFormalArguments()                  │
└──────────────────────────────────────────────────────────┘
```

### TTI（TargetTransformInfo）的 CRTP 实现模式

TTI 使用 CRTP（Curiously Recurring Template Pattern）实现静态多态：

```cpp
class H2BLBTTIImpl : public BasicTTIImplBase<H2BLBTTIImpl> {
  using BaseT = BasicTTIImplBase<H2BLBTTIImpl>;
  friend BaseT;  // CRTP 需要基类访问派生类的私有方法

  const H2BLBSubtarget &ST;
  const H2BLBTargetLowering &TLI;
  
  // 基类通过 CRTP 调用这些方法
  const H2BLBSubtarget *getST() const { return &ST; }
  const H2BLBTargetLowering *getTLI() const { return &TLI; }
};
```

**TTI 成本模型方法示例**：

```cpp
// 控制向量化行为
unsigned getLoadVectorFactor(unsigned VF, unsigned LoadSize,
                              unsigned ChainSizeInBytes,
                              VectorType *VecTy) const;
// 控制 intrinsic 成本
InstructionCost getIntrinsicInstrCost(const IntrinsicCostAttributes &ICA,
                                      TTI::TargetCostKind CostKind) const;
// 控制循环展开偏好
void getUnrollingPreferences(Loop *L, ScalarEvolution &SE,
                              TTI::UnrollingPreferences &UP) const;
```

## 关键机制解析（工业视角）

### 默认中端流水线的定制

**新 PM 方式**（推荐，通过 PassBuilder callbacks）：

```cpp
void H2BLBTargetMachine::registerPassBuilderCallbacks(PassBuilder &PB) {
  // 在流水线起始位置注入自定义 pass
  PB.registerPipelineStartEPCallback(
    [](ModulePassManager &MPM, OptimizationLevel OptLevel) {
      if (OptLevel == OptimizationLevel::O0) return;
      FunctionPassManager FPM;
      FPM.addPass(H2BLBCustomPass());
      MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
    });
  
  // 在每次 InstCombine 之后注入 peephole pass
  PB.registerPeepholeEPCallback(
    [](FunctionPassManager &FPM, OptimizationLevel OptLevel) {
      FPM.addPass(H2BLBPeepholePass());
    });
}
```

**Legacy PM 方式**（代码生成流水线）：

```cpp
void H2BLBPassConfig::addIRPasses() {
  TargetPassConfig::addIRPasses();  // 先添加默认的 IR passes
  if (getOptLevel() != CodeGenOptLevel::None)
    addPass(createH2BLBCustomLegacyPass());
}
```

**工业经验**：选择正确的 EP（Extension Point）callback 至关重要：
- `PipelineStartEPCallback`：在流水线最开始（适合做自定义规范化）
- `PipelineEarlySimplificationEPCallback`：在早期 simplify 之后
- `PeepholeEPCallback`：在每次 InstCombine 之后（适合 peephole 优化）
- `LateLoopOptimizationsEPCallback`：在循环优化之后
- `ScalarOptimizerLateEPCallback`：在标量优化流水线之后

### 代码生成流水线的骨架搭建

要让 `llc` 不崩溃地运行（即使不生成真正代码），需要提供以下虚假实现：

1. **假的指令选择器**：
```cpp
bool H2BLBPassConfig::addInstSelector() { return false; }
```

2. **TargetLoweringObjectFile 子类**：
```cpp
class H2BLB_ELFTargetObjectFile : public TargetLoweringObjectFileELF {};
class H2BLB_MachoTargetObjectFile : public TargetLoweringObjectFileMachO {};
```

3. **MC 层组件**（MCSubtargetInfo、MCInstrInfo、MCRegisterInfo）：
这些最终需要通过 TableGen 生成，但初期可提供空实现。

## AI 编译器关联

### MLIR Target Dialect（NVVM, SPIRV, AMDGPU）

MLIR 定义了多种 target dialect 直接映射到 GPU 后端：

| MLIR Dialect | 对应的底层 IR | GPU 厂商 |
|-------------|------------|---------|
| `gpu` dialect | 通用 GPU 抽象 | 跨厂商 |
| `nvvm` dialect | NVVM IR (LLVM + NV intrinsics) | NVIDIA |
| `rocdl` dialect | AMDGPU ROCDL IR | AMD |
| `spirv` dialect | SPIR-V binary | 跨厂商（Vulkan/OpenCL） |

**Lowering 路径示例**（NVIDIA）：
```
Triton GPU Dialect → gpu dialect → nvvm dialect → LLVM IR + NV intrinsics → PTX
```

**Lowering 路径示例**（AMD）：
```
Triton GPU Dialect → gpu dialect → rocdl dialect → LLVM IR + AMDGPU intrinsics → AMDGCN ISA
```

**MLIR 中创建类似 TargetMachine 的结构**：

MLIR 的 `mlir::TargetOptions` 和 `mlir::gpu::TargetOptions` 扮演了类似 LLVM `TargetMachine` 参数的角色：

```cpp
// MLIR 中的目标配置
mlir::gpu::TargetOptions targetOptions;
targetOptions.setTarget("cuda");
targetOptions.setChip("sm_80");
targetOptions.setFeatures("+ptx70");
```

### Triton CUDA 后端的注册机制

Triton 的 CUDA 后端注册过程与 LLVM 的 TargetRegistry 机制有相似的设计模式：

1. **编译管线注册**：
```python
# Triton 的 Python 端 target 注册
@triton.runtime.driver.register("cuda")
def compile_cuda(module, target_options):
    # lowering: Triton IR → Triton GPU → LLVM IR
    # codegen: LLVM IR → PTX → cubin
    pass
```

2. **C++ 端 Target backend**：
Triton 在 MLIR 中使用 `triton::TranslateTritonGPUToLLVMIR` pass 执行 lowering，通过 target attribute 区分 CUDA 和 ROCM 后端。

### IREE HAL Target Backends

IREE 的 HAL（Hardware Abstraction Layer）设计体现了多目标后端的最佳实践：

```
IREE Module
  │
  ├── HAL Target: Vulkan-SPIRV
  │   ├── spirv-target-options（capability, extensions, limits）
  │   ├── spirv optimization passes
  │   └── spirv binary serialization
  │
  ├── HAL Target: CUDA
  │   ├── LLVM IR with NV intrinsics
  │   ├── PTX assembly generation
  │   └── cubin object code
  │
  └── HAL Target: CPU-LLVM
      ├── LLVM IR optimization
      └── LLVM JIT or AOT compilation
```

**IREE 的 Target Backend 注册机制**：

```cpp
// 类似 LLVM 的 TargetRegistry
static llvm::cl::opt<std::string> clTargetBackend(
    "iree-hal-target-backend",
    llvm::cl::desc("Target backend to use"),
    llvm::cl::init("dylib-llvm-aot"));

// 注册列表
static TargetBackendRegistration vulkan_spirv("vulkan-spirv", ...);
static TargetBackendRegistration cuda("cuda", ...);
static TargetBackendRegistration llvm_cpu("llvm-cpu", ...);
```

### 自定义 TTI 对 AI 编译器的重要性

在 AI 编译器中，TTI 的实现直接影响生成的代码质量：

**GPU TTI 的关键建模**：
- **SIMT 发散惩罚**：warp 内分支发散导致的性能损失
- **Shared memory 带宽和延迟**：影响 shared memory promotion 的成本判断
- **Tensor Core 利用率**：MMA 操作的 cost 必须反映硬件特性
- **寄存器压力模型**：每个线程的可用寄存器数量限制

**错误 TTI 的后果示例**：
- 如果 `getLoadVectorFactor` 返回了硬件不支持的向量宽度，生成的 PTX 会被 ptxas 拒绝
- 如果 `getMemoryOpCost` 低估了 global memory 延迟，编译器可能过度使用 global memory 而非 shared memory
- 如果 `getArithmeticInstrCost` 没有正确建模 fp16/fp32/fp64 的相对成本，精度选择可能次优

## 示例说明

### 示例 1：自定义 intrinsic 的端到端使用

```c
// C 源码
int widening_signed_multiply(short a, short b) {
  return __builtin_h2blb_widening_smul(a, b);
}
```

```bash
$ clang -target h2blb -O1 -emit-llvm -S input.c -o -
```

```llvm
; 生成的 LLVM IR
define i32 @widening_signed_multiply(i16 signext %a, i16 signext %b) {
entry:
  %0 = tail call i32 @llvm.h2blb.widening.smul(i16 %a, i16 %b)
  ret i32 %0
}
```

### 示例 2：TTI 控制向量化行为

```llvm
; Input: 4 个连续的 i16 加载
%h0 = load i16, ptr %source, align 8
%h1 = load i16, ptr %idx1, align 8
%h2 = load i16, ptr %idx2, align 8
%h3 = load i16, ptr %idx3, align 8
```

由于 TTI 的 `getLoadVectorFactor` 限制 i16 向量宽度最多为 2，`load-store-vectorizer` 产生：
```llvm
%load1 = load <2 x i16>, ptr %source       ; 前两个
%load2 = load <2 x i16>, ptr %source_next   ; 后两个
```

## 总结

1. **创建新 LLVM 后端是系统化工程**：需要依次完成构建系统集成、Triple 注册、TargetMachine、Clang 连接、intrinsics 定义、TTI 实现六个阶段。每个阶段有明确的接口约定。

2. **CRTP 模式是 LLVM 中实现静态多态的核心技术**：`BasicTTIImplBase` 使用 CRTP 避免虚函数调用的运行时开销，这对于编译器中频繁调用的成本查询 API 至关重要。

3. **Intrinsics 是前端与后端的桥梁**：通过 TableGen 的 `ClangBuiltin` 类和 `gen-intrinsic-enums` backend 实现从 Clang builtin 到 LLVM IR intrinsic 的自动一对一映射。

4. **AI 编译器中的目标后端设计**：
   - MLIR 使用 target dialect（nvvm, rocdl, spirv）而非 intrinsic 集合来建模目标特定操作
   - Triton 通过 MLIR pass pipeline 而非单一的 TargetMachine 来管理 lowering 阶段
   - IREE 的 HAL 层提供了多后端注册机制，类似 LLVM 的 TargetRegistry 但更灵活

5. **TTI 是决定代码质量的隐形之手**：正确实现 TTI 比编写优化 passes 本身更具挑战性，因为它影响所有其他 passes 的决策。AI 编译器中 GPU 特定的 TTI 实现（SIMT 发散、shared memory 延迟、tensor core 利用率）是获得高性能的关键。
