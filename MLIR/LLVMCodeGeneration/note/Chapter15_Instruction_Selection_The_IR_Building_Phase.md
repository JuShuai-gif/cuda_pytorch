# Chapter 15: Instruction Selection - The IR Building Phase

## 核心概念（详细展开）

### IR Building Phase 的双重目标

IR Building 是 ISel 的第一个阶段，有两项任务：

1. **翻译 LLVM IR** → 选择器框架的内部 IR（SDNode DAG 或 G_MIR）
   - 纯翻译部分是 **通用的**（目标无关），LLVM 框架自动完成
   - 如 `add i32 %a, %b` → `ISD::ADD` SDNode 或 `G_ADD` MachineInstr

2. **具现化 ABI（Application Binary Interface）**——即 calling conventions
   - 这部分是 **目标特定的**，需要后端作者实现
   - 包括参数传递、返回值处理、栈帧布局约定

**ABI lowering 是后端实现中最容易出错的部分**：
- 调用者的参数准备必须与 callee 的参数接收完全一致
- 结构体返回值的 demotion（sret）逻辑涉及调用者和 callee 双方
- 不同 calling convention（C、fastcall、stdcall）的规则各不相同

### 四大 ABI Lowering Hooks

每个调用边界需要三个方向 + 一个能力查询：

```
┌────────────────────────────────────────────────────┐
│                  Caller (调用者)                     │
│                                                     │
│  LowerCall:                                         │
│    1. 为 outgoing arguments 分配位置                │
│    2. 生成参数准备代码 (COPY to regs, STORE to stack)│
│    3. 生成 call 指令                                │
│    4. 解包返回值（从寄存器/栈取回）                  │
│                                                     │
│  CanLowerReturn: 检查返回值能否放入寄存器            │
│    如果不能: 触发 sret demotion                     │
└──────────────────────┬──────────────────────────────┘
                       │ call instruction
┌──────────────────────┴──────────────────────────────┐
│                  Callee (被调用者)                   │
│                                                     │
│  LowerFormalArguments:                              │
│    1. 从 ABI 位置读取 incoming arguments             │
│    2. 创建虚拟寄存器 + CopyFromReg / Load            │
│                                                     │
│  LowerReturn:                                       │
│    1. 将返回值写入 ABI 指定位置                      │
│    2. 生成 return 指令                              │
└────────────────────────────────────────────────────┘
```

| Hook | 哪一侧 | 职责 |
|------|-------|------|
| `LowerFormalArguments` | Callee | 从 ABI 位置读取传入参数 |
| `LowerReturn` | Callee | 将返回值写入 ABI 位置 |
| `LowerCall` | Caller | 准备 outgoing 参数 + 处理返回值 |
| `CanLowerReturn` | 基础设施查询 | 检查返回值是否适合放在寄存器内 |

### Structure Return Demotion (sret)

如果返回值太大无法用寄存器传递（由 `CanLowerReturn` 判断）：

```
┌─ Caller 侧 ─────────────────────────────────┐
│  1. 在栈上分配 sret 缓冲区                   │
│  2. 将 sret 指针作为额外参数传递（通常是第一个）│
│  3. 从 sret 缓冲区读取最终返回的 struct       │
└──────────────────────────────────────────────┘
┌─ Callee 侧 ─────────────────────────────────┐
│  1. 接收 sret 指针作为额外的形式参数          │
│  2. 将 struct 返回值写入 sret 指针指向的缓冲区 │
└──────────────────────────────────────────────┘
```

## LLVM / MLIR 流程（深入）

### GPU Kernel ABI (CUDA Calling Conventions)

GPU kernel 的 ABI 与 CPU 函数有本质区别：

```
CPU ABI (AArch64 Calling Convention):
  - 参数在 x0-x7 寄存器中
  - 返回值在 x0 寄存器中
  - 栈 16-byte 对齐
  - 有 callee-saved 和 caller-saved 寄存器

GPU Kernel ABI (CUDA):
  - Kernel 参数通过 constant memory 传递（不在寄存器中！）
  - Kernel 无返回值（void return）
  - Thread ID 通过特殊寄存器获取 (%tid.x, %tid.y, %tid.z)
  - Block ID 通过特殊寄存器获取 (%ctaid.x, ...)
  - 无传统意义上的 "calling convention"——kernel 不被 "调用"
```

在 LLVM NVPTX 后端中，kernel ABI 的特殊处理：

```cpp
// NVPTX 的 LowerFormalArguments 实现（简化）
SDValue NVPTXTargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv,
    bool isVarArg, const SmallVectorImpl<ISD::InputArg> &Ins,
    const SDLoc &dl, SelectionDAG &DAG,
    SmallVectorImpl<SDValue> &InVals) const {

    MachineFunction &MF = DAG.getMachineFunction();

    // NVPTX kernel 参数存储在 .param 状态空间
    for (unsigned i = 0; i < Ins.size(); ++i) {
        // 为每个参数创建一个 .param 空间槽位
        // 然后生成 ld.param 来加载
        SDValue Param = DAG.getTargetExternalSymbol(
            paramSymbolName, getPointerTy(DAG.getDataLayout()));
        InVals.push_back(loadFromParamSpace(DAG, dl, Chain, Param, Ins[i].VT));
    }
    return Chain;
}
```

### MLIR 的 gpu.launch_func ABI

MLIR 中 GPU kernel launch 的 ABI 设计：

```mlir
// MLIR 高层表示
gpu.launch_func @my_kernel
    blocks in (%grid_x, %grid_y, %grid_z)
    threads in (%block_x, %block_y, %block_z)
    args(%buf : memref<128xf32, #gpu.address_space<global>>,
         %scalar : f32)

// lowering 到 llvm dialect 后:
// Host 侧:
llvm.call @cudaLaunchKernel(
    @my_kernel_func_ptr,
    %grid_dim, %block_dim,
    %kernel_args_array, %shared_mem_bytes, %stream)

// Device 侧 (kernel 内部):
llvm.func @my_kernel(%arg0: !llvm.ptr<f32>, %arg1: f32) {
    // kernel 参数通过 constant memory / param space 接收
    // ...
}
```

**关键差异**：
- MLIR 的 `gpu.launch_func` 同时包含 host launch 配置和 kernel 参数
- Lowering 后 split 为两部分：host 侧生成 CUDA API 调用，device 侧生成 kernel 函数
- Kernel 的 ABI（参数如何在 device 侧接收）由 NVPTX backend 处理

### Triton Kernel Launch ABI

Triton 的 kernel launch 比标准 CUDA 更复杂，因为 Triton 使用专门的 grid 映射：

```python
# Triton kernel launch
@triton.jit
def matmul(A, B, C, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # ... kernel body ...

# Launch
grid = lambda META: (triton.cdiv(M, META['BLOCK_M']),
                      triton.cdiv(N, META['BLOCK_N']))
matmul[grid](a, b, c, M, N, K, BLOCK_M=128, BLOCK_N=64)
```

Triton 的 kernel ABI 包括：
1. **常量参数**（在 JIT 时折叠）：`BLOCK_M=128`, `BLOCK_N=64` → 编译时常量
2. **动态参数**（运行时传递）：`a`, `b`, `c`（张量指针）, `M`, `N`, `K`（维度）
3. **隐式参数**：`program_id(0)`, `program_id(1)` → 映射到 CUDA 的 `blockIdx`
4. **Triton 自己的 grid 抽象**：Triton 的 grid 是 1D 映射到 CUDA 的 3D grid

Triton 的 LLVM IR 生成器负责：
```cpp
// 生成 CUDA 的 blockIdx 访问
Value loadBlockIdx(MLIRContext &ctx, Location loc, unsigned dim) {
    return builder.CreateCall(
        getBlockIdxIntrinsic(dim), {});  // @llvm.nvvm.read.ptx.sreg.ctaid.x
}

// 将 Triton 的 program_id 映射到 blockIdx
Value programId = loadBlockIdx(ctx, loc, axis);  // program_id(0) = blockIdx.x
```

### Custom Accelerators ABI Design

为自定义 AI 加速器设计 ABI 时的考虑：

```
┌──────────────────────────────────────────────────┐
│ Custom AI Accelerator ABI Design Principles       │
├──────────────────────────────────────────────────┤
│ 1. 标量参数传递                                   │
│    - 少量标量通过配置寄存器传递                    │
│    - 大量标量通过命令缓冲区传递                    │
│                                                   │
│ 2. 张量参数传递                                   │
│    - 通过 HBM/DDR 地址 + descriptor 传递          │
│    - 包含：base_addr, dimensions, strides, dtype │
│                                                   │
│ 3. 工作配置                                       │
│    - tile/block dimensions 通过特殊寄存器传递     │
│    - 类似 CUDA 的 blockDim / gridDim             │
│                                                   │
│ 4. Synchronization                               │
│    - 异步执行模型：launch → signal/wait          │
│    - 不同于同步函数调用的 call/return 语义        │
└──────────────────────────────────────────────────┘
```

## 关键机制解析（工业视角）

### CCValAssign - Calling Convention 的核心数据结构

`CCValAssign` 链接 calling convention 描述与实际参数位置：

```cpp
class CCValAssign {
public:
    // 位置类型
    bool isRegLoc() const;        // 参数在寄存器中
    bool isMemLoc() const;        // 参数在栈中
    unsigned getLocReg() const;   // 如果是寄存器位置，返回物理寄存器号
    unsigned getLocMemOffset() const; // 如果是内存位置，返回栈偏移

    // 类型信息
    MVT getValVT() const;         // 参数的声明类型（如 i16）
    MVT getLocVT() const;         // 位置中的实际类型（如 i32，因为扩展）

    // 变换类型
    LocInfo getLocInfo() const;   // ValVT 和 LocVT 之间需要如何转换
    // LocInfo 可能是:
    //   Full  - 类型完全匹配，无需转换
    //   SExt  - 需要符号扩展到 LocVT
    //   ZExt  - 需要零扩展到 LocVT
    //   AExt  - 需要任意扩展（值高位无关）

    // 参数标识
    unsigned getValNo() const;    // 参数的索引号
    bool isFixed() const;         // true=固定位置，false=可变参数
};
```

**示例：i16 参数在 32-bit 寄存器中**：
```
ValVT = i16, LocVT = i32, LocInfo = SExt
含义：原始是 i16 类型，但 ABI 规定在 32-bit 寄存器中传递，
     需要符号扩展到 32-bit
```

### TableGen Calling Convention Description

```tablegen
// 定义一个 calling convention（H2BLB 示例）
def CC_H2BLB_Common : CallingConv<[
    // 规则 1: v2i16 向量转为 i32 (bitcast)
    CCIfType<[v2i16], CCBitConvertToType<i32>>,

    // 规则 2: i16 和 f16 参数尝试放在 R1,R2,R3 寄存器
    CCIfType<[i16, f16], CCAssignToReg<[R1, R2, R3]>>,

    // 规则 3: i32 和 f32 参数尝试放在 D1 寄存器
    CCIfType<[i32, f32], CCAssignToReg<[D1]>>,

    // 规则 4 (fallback): 所有其他参数放在栈上，2-byte 对齐，2-byte 大小
    CCAssignToStack<2, 2>
]>;

// callee-saved 寄存器列表
def CSR_H2BLB : CalleeSavedRegs<(add R4, R5, R6, D3)>;
```

**规则评估是顺序的**：第一条匹配的规则获胜。如果一个参数无法分配（如指定寄存器已被占用），
则会继续尝试下一条规则。这就是为什么 fallback `CCAssignToStack` 放在最后。

### 连接 gen-callingconv

```cmake
# CMakeLists.txt
tablegen(LLVM XXXGenCallingConv.inc -gen-callingconv)
```

生成的函数签名：
```cpp
// CCAssignFn - 在 LowerFormalArguments / LowerReturn / LowerCall 中使用
bool RetCC_XXX(unsigned ValNo, MVT ValVT, MVT LocVT,
               CCValAssign::LocInfo LocInfo,
               ISD::ArgFlagsTy ArgFlags, CCState &State);
```

使用方式：
```cpp
// 在 LowerFormalArguments 中
CCState CCInfo(CallConv, isVarArg, MF, ArgLocs, *DAG.getContext());
CCInfo.AnalyzeFormalArguments(Ins, CC_XXX);  // CC_XXX 是 TableGen 生成的函数

// 遍历分配结果
for (unsigned i = 0; i < ArgLocs.size(); ++i) {
    CCValAssign &VA = ArgLocs[i];
    if (VA.isRegLoc()) {
        // 参数在寄存器中: VA.getLocReg()
        Register Reg = MF.addLiveIn(VA.getLocReg(), RC);
        SDValue ArgVal = DAG.getCopyFromReg(Chain, dl, Reg, VA.getLocVT());
        // 如果 LocVT != ValVT，需要截断/扩展
        if (VA.getLocInfo() == CCValAssign::SExt)
            ArgVal = DAG.getNode(ISD::AssertSext, dl, VA.getLocVT(), ArgVal,
                                 DAG.getValueType(VA.getValVT()));
        InVals.push_back(ArgVal);
    } else {
        // 参数在栈上: VA.getLocMemOffset()
        int FI = MFI.CreateFixedObject(VA.getLocVT().getSizeInBits() / 8,
                                        VA.getLocMemOffset(), true);
        SDValue Ptr = DAG.getFrameIndex(FI, getPointerTy(DL));
        SDValue Load = DAG.getLoad(VA.getLocVT(), dl, Chain, Ptr,
                                    MachinePointerInfo::getFixedStack(MF, FI));
        InVals.push_back(Load);
    }
}
```

### SDISel ABI Lowering 实现模板

```cpp
// LowerFormalArguments 实现模板
SDValue MyTargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool isVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &dl,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {

    MachineFunction &MF = DAG.getMachineFunction();
    MachineRegisterInfo &MRI = MF.getRegInfo();

    // Step 1: 使用 CCState + TableGen CCAssignFn 分析参数位置
    SmallVector<CCValAssign, 16> ArgLocs;
    CCState CCInfo(CallConv, isVarArg, MF, ArgLocs, *DAG.getContext());
    CCInfo.AnalyzeFormalArguments(Ins, CC_MyTarget);  // TableGen 生成

    // Step 2: 遍历分配结果
    for (unsigned i = 0; i < ArgLocs.size(); ++i) {
        CCValAssign &VA = ArgLocs[i];
        EVT ValVT = VA.getValVT();
        SDValue ArgValue;

        if (VA.isRegLoc()) {
            // 寄存器位置：创建虚拟寄存器 + CopyFromReg
            EVT RegVT = VA.getLocVT();
            const TargetRegisterClass *RC = getRegClassFor(RegVT);
            Register VReg = MRI.createVirtualRegister(RC);
            MRI.addLiveIn(VA.getLocReg(), VReg);  // 标记为 live-in
            ArgValue = DAG.getCopyFromReg(Chain, dl, VReg, RegVT);

            // 处理类型转换
            switch (VA.getLocInfo()) {
            case CCValAssign::SExt:
                ArgValue = DAG.getNode(ISD::AssertSext, dl, RegVT, ArgValue,
                                       DAG.getValueType(ValVT));
                break;
            case CCValAssign::ZExt:
                ArgValue = DAG.getNode(ISD::AssertZext, dl, RegVT, ArgValue,
                                       DAG.getValueType(ValVT));
                break;
            // Full, AExt: no action needed
            default: break;
            }
        } else {
            // 栈位置：创建固定栈对象 + Load
            assert(VA.isMemLoc());
            unsigned ArgSize = VA.getLocVT().getSizeInBits() / 8;
            int FI = MFI.CreateFixedObject(ArgSize, VA.getLocMemOffset(),
                                            /*IsImmutable=*/true);
            SDValue FrameIdx = DAG.getFrameIndex(FI,
                getPointerTy(DAG.getDataLayout()));
            MachinePointerInfo PtrInfo =
                MachinePointerInfo::getFixedStack(MF, FI);
            ArgValue = DAG.getLoad(VA.getLocVT(), dl, Chain, FrameIdx, PtrInfo);
        }
        InVals.push_back(ArgValue);
    }
    return Chain;
}
```

### GlobalISel ABI Lowering - Assigner + Handler 模式

GlobalISel 使用 **assigner + handler** 设计模式：

```
┌──────────────────────────────────────────────┐
│ Assigner                                      │
│  - 使用 CCAssignFn (TableGen 生成) 分配位置   │
│  - 返回 CCValAssign 列表                      │
│                                               │
│ IncomingValueAssigner (for formal arguments)  │
│ CallReturnAssigner (for call return values)   │
│ OutgoingValueAssigner (for call arguments)    │
└───────────────┬──────────────────────────────┘
                │ CCValAssign list
┌───────────────┴──────────────────────────────┐
│ Handler                                       │
│  - 根据 CCValAssign 具现化 IR                │
│  - assignValueToReg(): 寄存器 → COPY         │
│  - assignValueToAddress(): 内存 → LOAD       │
│  - getStackAddress(): 获取栈槽地址            │
└──────────────────────────────────────────────┘
```

```cpp
class MyTargetCallLowering : public CallLowering {
    bool lowerFormalArguments(MachineIRBuilder &MIRBuilder,
                               const Function &F,
                               ArrayRef<ArrayRef<Register>> VRegs,
                               FunctionLoweringInfo &FLI) const override {
        MachineRegisterInfo &MRI = MIRBuilder.getMF().getRegInfo();

        // Step 1: 处理 sret demotion（如果需要）
        SmallVector<ArgInfo, 8> SplitArgs;
        if (!FLI.CanLowerReturn) {
            insertSRetIncomingArgument(F, SplitArgs,
                                        FLI.DemoteRegister, MRI,
                                        MIRBuilder.getMF().getDataLayout());
        }

        // Step 2: Split arguments into value types
        unsigned i = 0;
        for (auto &Arg : F.args()) {
            ArgInfo OrigArg{VRegs[i], Arg, i};
            splitToValueTypes(OrigArg, SplitArgs,
                              MIRBuilder.getMF().getDataLayout(),
                              F.getCallingConv());
            i++;
        }

        // Step 3: Assigner 分配位置
        IncomingValueAssigner Assigner(CCAssignFn_FormalArgs);
        CCState CCInfo(F.getCallingConv(), F.isVarArg(),
                       MIRBuilder.getMF(), ArgLocs,
                       *MIRBuilder.getMF().getContext());
        if (!determineAssignments(Assigner, SplitArgs, CCInfo))
            return false;

        // Step 4: Handler 生成 IR
        IncomingArgHandler Handler(MIRBuilder, MRI);
        return handleAssignments(Handler, SplitArgs, CCInfo,
                                  ArgLocs, MIRBuilder);
    }
};
```

### Handler 的虚方法实现

```cpp
class IncomingArgHandler : public CallLowering::IncomingValueHandler {
protected:
    // 将值从物理寄存器复制到虚拟寄存器
    unsigned assignValueToReg(Register ValVReg, Register PhysReg,
                               const CCValAssign &VA) override {
        MIRBuilder.buildCopy(ValVReg, PhysReg);
        return 1;  // 返回消耗的 CCValAssign 条目数
    }

    // 从内存地址加载值
    unsigned assignValueToAddress(Register ValVReg, Register Addr,
                                   LLT MemTy,
                                   const MachinePointerInfo &MPO,
                                   const CCValAssign &VA) override {
        MachineMemOperand *MMO = MIRBuilder.getMF().getMachineMemOperand(
            MPO, MachineMemOperand::MOLoad, MemTy, Align(4));
        MIRBuilder.buildLoad(ValVReg, Addr, *MMO);
        return 1;
    }

    // 获取栈参数的内存地址
    Register getStackAddress(uint64_t MemSize, int64_t Offset,
                              MachinePointerInfo &MPO,
                              ISD::ArgFlagsTy ArgFlags) override {
        // 创建固定栈对象 + FrameIndex
        MachineFrameInfo &MFI = MIRBuilder.getMF().getFrameInfo();
        int FI = MFI.CreateFixedObject(MemSize, Offset, /*Immutable=*/true);
        MPO = MachinePointerInfo::getFixedStack(MIRBuilder.getMF(), FI);

        // 创建 G_FRAME_INDEX 并添加偏移
        LLT PtrTy = LLT::pointer(0, DL.getPointerSizeInBits(0));
        Register AddrReg = MRI.createGenericVirtualRegister(PtrTy);
        MIRBuilder.buildFrameIndex(AddrReg, FI);
        if (Offset != 0)
            AddrReg = MIRBuilder.buildPtrAdd(PtrTy, AddrReg,
                MIRBuilder.buildConstant(LLT::scalar(64), Offset)).getReg(0);
        return AddrReg;
    }
};
```

## AI 编译器关联

### GPU Kernel ABI 深入

CUDA kernel ABI 的关键特性在 LLVM ISel 中的映射：

```cpp
// NVPTX kernel ABI 在 LowerFormalArguments 中的处理

// CUDA kernel 函数签名:
// __global__ void kernel(float *a, float *b, int n)
//
// LLVM IR 中的表示:
// define void @kernel(ptr %a, ptr %b, i32 %n)
//
// 在 PTX 中:
// .visible .entry kernel(.param .u64 kernel_param_0,
//                         .param .u64 kernel_param_1,
//                         .param .u32 kernel_param_2)

SDValue NVPTXTargetLowering::LowerFormalArguments(...) {
    // NVPTX kernel 参数从 .param 空间通过 ld.param 加载
    for (unsigned i = 0; i < Ins.size(); ++i) {
        if (isKernelFunction(MF)) {
            // Kernel 模式：通过 param space
            SDValue Param = getParamSymbol(DAG, i);
            InVals.push_back(DAG.getLoad(Ins[i].VT, dl, Chain, Param, ...));
        } else {
            // 普通函数模式：标准寄存器 ABI
            // ...
        }
    }
}
```

### MLIR gpu.launch_func → LLVM Call 的 ABI 转换

```mlir
// MLIR: gpu dialect
gpu.launch_func @kernel_func
    blocks in (%cst100, %cst1, %cst1)
    threads in (%cst256, %cst1, %cst1)
    args(%arg0 : memref<32x32xf32>, %arg1 : f32)

// 转换后: LLVM dialect (host 侧)
%kernel_addr = llvm.mlir.addressof @kernel_func : !llvm.ptr
%grid_x = llvm.mlir.constant(100 : i32) : i32
%grid_y = llvm.mlir.constant(1 : i32) : i32
...
// 构建 kernel 参数数组
%args_array = llvm.alloca %num_args x !llvm.ptr
%arg0_ptr = llvm.mlir.undef : !llvm.ptr
// 填充参数数组...
llvm.call @cudaLaunchKernel(%kernel_addr, %grid_dim, %block_dim,
                            %args_array, %shared_mem, %stream)
```

### Triton Kernel Launch ABI 分析

Triton 的 kernel launch 在 LLVM IR 层面生成如下结构：

```llvm
; Triton 生成的 PTX kernel 函数
define void @matmul_kernel_0d1d2d3d4(
    ptr %arg_a,           ; Tensor A
    ptr %arg_b,           ; Tensor B
    ptr %arg_c,           ; Tensor C (output)
    i32 %M,               ; 动态维度参数
    i32 %N,
    i32 %K
) {
    ; Prologue: 读取 block/tread ID
    %pid0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
    %pid1 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()

    ; 将 program_id 映射到数据块
    %row_offset = mul i32 %pid0, 128    ; BLOCK_M = 128 (编译时常量)
    %col_offset = mul i32 %pid1, 64     ; BLOCK_N = 64 (编译时常量)

    ; ... kernel body ...
    ret void
}

; Host 侧 launch（由 Triton runtime 生成）
; 在 LLVM IR 生成时，Triton 将:
; - BLOCK_M, BLOCK_N 作为编译时常量内联到 kernel 体内
; - M, N, K 作为运行时参数传递
; - grid dims 由 Triton 的 grid lambda 计算
```

**Triton 的 ABI 创新**：
1. **编译时 specialization**：block size 等常量在 JIT 编译时内联，避免了动态分支
2. **1D grid 映射**：Triton 自己的 1D/2D grid 到 CUDA 3D grid 的重新映射
3. **自动 coalescing**：Triton 根据 program_id 自动计算 coalesced 的内存访问模式
4. **隐式 shared memory allocation**：Triton 在 LLVM IR 生成时自动管理 shared memory

### Custom AI Accelerator ABI Design 实践

设计自定义 AI 加速器的 ABI lowering 时需要考虑的 ISel 方面：

```
AI Accelerator Calling Convention Design:
┌────────────────────────────────────────────────┐
│ Parameter Classification:                       │
│                                                │
│ Category 1: Configuration Registers             │
│   - Tile dimensions (M, N, K)                  │
│   - Data type encoding (fp16, int8, bf16)      │
│   - Activation function type                   │
│   → ABI: 专用配置寄存器 (如 CFG_REG_0...7)      │
│                                                │
│ Category 2: Tensor Descriptors                  │
│   - Base address + strides + shape              │
│   - Memory space indicator (HBM/L1/SMEM)       │
│   → ABI: Descriptor memory / descriptor regs   │
│                                                │
│ Category 3: Scalar Values                       │
│   - Scale factors, bias values                 │
│   → ABI: 标量通用寄存器 (如 SR_0...15)          │
│                                                │
│ Category 4: Synchronization Tokens              │
│   - Fence IDs, signal handles                  │
│   → ABI: 专用同步寄存器                         │
└────────────────────────────────────────────────┘

// 在 LowerFormalArguments 中处理:
SmallVector<CCValAssign, 16> ArgLocs;
CCInfo.AnalyzeFormalArguments(Ins, CC_AIAccl);
for (auto &VA : ArgLocs) {
    if (VA.isRegLoc()) {
        unsigned PhysReg = VA.getLocReg();
        if (isConfigReg(PhysReg)) {
            // 配置寄存器：通过特殊 move 读取
            ArgValue = emitReadConfigReg(DAG, dl, Chain, PhysReg);
        } else if (isScalarReg(PhysReg)) {
            // 标量寄存器：标准 CopyFromReg
            ArgValue = DAG.getCopyFromReg(Chain, dl, PhysReg, VA.getLocVT());
        }
    } else if (VA.isMemLoc()) {
        // 内存中的 tensor descriptor：通过 descriptor load 加载
        ArgValue = emitLoadDescriptor(DAG, dl, Chain, VA.getLocMemOffset());
    }
}
```

## 示例说明

### 示例 1：完整的 ABI Lowering 流程（SDISel）

```tablegen
// 定义简单的 calling convention
// 第一个 i32 参数 → R0 寄存器
// 第二个 i32 参数 → R1 寄存器
// i64 参数 → (R2, R3) 寄存器对
// 其余参数 → 栈

def CC_Simple : CallingConv<[
    CCIfType<[i32], CCAssignToReg<[R0, R1]>>,
    CCIfType<[i64], CCAssignToRegWithShadow<[R2, R3], [D2]>>,
    CCAssignToStack<4, 4>  // 4-byte 对齐，4-byte 大小
]>;
```

```cpp
// 在 LowerFormalArguments 中的使用
bool MyTargetLowering::LowerFormalArguments(...) {
    // ... setup CCInfo ...

    // 执行分配
    CCInfo.AnalyzeFormalArguments(Ins, CC_Simple);

    // 处理每个参数
    for (auto &VA : ArgLocs) {
        if (VA.isRegLoc()) {
            // i32 参数在 R0/R1 中
            Register VReg = MRI.createVirtualRegister(&GPR32RegClass);
            MF.addLiveIn(VA.getLocReg(), VReg);
            InVals.push_back(DAG.getCopyFromReg(Chain, dl, VReg, MVT::i32));
        } else {
            // i32 参数在栈上
            // ...
        }
    }
}

// 对于调用者侧 (LowerCall):
// 需要确保参数以正确的顺序放在正确的位置
bool MyTargetLowering::LowerCall(...) {
    // 分配 outgoing 参数
    CCInfo.AnalyzeCallOperands(Outs, CC_Simple);

    // 为每个参数生成 COPY to reg 或 STORE to stack
    for (auto &VA : ArgLocs) {
        if (VA.isRegLoc()) {
            // CopyToReg
            Chain = DAG.getCopyToReg(Chain, dl, VA.getLocReg(),
                                      Outs[VA.getValNo()].Val, Chain);
        } else {
            // Store to stack slot
            // ...
        }
    }

    // 生成 call 指令
    Chain = DAG.getNode(MyTargetISD::CALL, dl,
                         DAG.getVTList(MVT::Other, MVT::Glue),
                         {Chain, Callee, ...});

    // 处理返回值（如果有）
    // ...
}
```

### 示例 2：自定义 SDNode 定义

```tablegen
// 在 XXXInstrInfo.td 中定义目标特定的 SDNode
def MyTargetcall : SDNode<"MyTargetISD::CALL",
    SDTypeProfile<0, -1, [SDTCisPtrTy<0>]>,  // 0 results, variadic inputs
    [SDNPHasChain,      // 有 chain 输入/输出
     SDNPOptInGlue,     // 可选的 glue 输入
     SDNPOutGlue,       // 产生 glue 输出
     SDNPVariadic]>;    // 可变数量参数

def MyTargetret : SDNode<"MyTargetISD::RET",
    SDTypeProfile<0, -1, []>,
    [SDNPHasChain, SDNPVariadic]>;
```

```cpp
// 在 C++ 中暴露枚举值
namespace MyTargetISD {
    enum NodeType : unsigned {
        FIRST_NUMBER = ISD::BUILTIN_OP_END,
        CALL,
        RET,
        // ... more custom opcodes ...
    };
}
```

### 示例 3：ISD::CALLSEQ_START / CALLSEQ_END

LLVM 基础设施要求所有后端描述这两种特殊的 DAG 操作：

```cpp
// 标志调用序列的开始和结束
// 它们负责调整栈指针（如果需要）
//
// CALLSEQ_START 在调用参数准备前
// CALLSEQ_END 在调用返回后

// 在 LowerCall 中:
SDValue Chain = DAG.getCALLSEQ_START(Chain, /*StackSizeForCall=*/0,
                                       /*IsTailCall=*/false, dl);

// ... 准备参数 + 发射 call 指令 ...

Chain = DAG.getCALLSEQ_END(Chain,
                            DAG.getIntPtrConstant(/*StackSizeForCall=*/0, dl),
                            DAG.getIntPtrConstant(0, dl),
                            /*Glue=*/CallGlue, dl);

// 即使栈帧调整为零，也必须生成这两个节点
// 因为它们是后端 pass 分析和 ABI 合规性的标记
```

## 总结

IR Building 是 ISel 的第一个阶段，主要有两个目标：
1. **翻译 LLVM IR**（框架自动完成）
2. **具现化 ABI / Calling Convention**（目标特定实现）

ABI lowering 通过四个 hook 实现：
- **LowerFormalArguments**（callee 侧）：从 ABI 位置接收参数
- **LowerReturn**（callee 侧）：将返回值写入 ABI 位置
- **LowerCall**（caller 侧）：准备 outgoing 参数 + 解包返回值
- **CanLowerReturn**：判断返回值是否适合寄存器传递（触发 sret demotion）

Calling convention 通过 TableGen 描述：
- `CallingConv` 记录包含有序的 `CCAction` 规则列表
- `CCIfType<[types], action>` 条件匹配
- `CCAssignToReg<[regs]>` 分配寄存器
- `CCAssignToStack<size, align>` 分配栈空间
- `CCBitConvertToType<ty>` 类型转换
- 规则按顺序评估，第一条匹配的规则获胜

实现辅助：
- **CCValAssign** 链接参数和实际位置（寄存器/栈，值和位置类型，扩展模式）
- **CCState** 管理分配状态（哪些寄存器已占用、栈偏移量）
- SDISel 使用 `DAG.getCopyFromReg()`, `DAG.getCopyToReg()`, `DAG.getLoad()` 等来生成 IR
- GlobalISel 使用 **Assigner + Handler** 模式（模块化、可测试）
- **ISD::CALLSEQ_START / CALLSEQ_END** 标记调用序列，是必须的标准操作

**与 AI 编译器的关系**：
- GPU kernel ABI 与传统 CPU 函数 ABI 有本质差异：
  - Kernel 参数通过 constant memory / param space 传递（而非寄存器）
  - 无传统返回值
  - thread/block ID 通过特殊寄存器获取
- MLIR 的 `gpu.launch_func` 需要两阶段的 ABI lowering：
  - Host 侧生成 CUDA API 调用
  - Device 侧生成 kernel 参数接收逻辑
- Triton 的 kernel launch ABI 包含编译时 specialization（block size 内联）
  和 grid 抽象映射（Triton 1D/2D grid → CUDA 3D grid）
- 自定义 AI 加速器的 ABI 设计应考虑：
  配置寄存器、tensor descriptor、标量值、同步 token 的不同传递方式
- ABI lowering 是后端实现中最容易出 bug 的部分，
  调用者和 callee 的一致性必须精确保证
