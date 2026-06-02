# Chapter 20: Lowering of the Stack Layout

> **From the perspective of a production AI compiler engineer who needs to understand LLVM deeply to work on MLIR/Triton/AI compiler stacks.**

## 核心概念（详细展开）

### 栈布局降低的本质

Stack Lowering 是将抽象栈引用（frame indices）转换为具体栈内存操作的过程。这包括：

1. **Frame Index Abstraction → 物理地址**：将整数 frame index 转换为 SP/FP 相对偏移
2. **Prologue/Epilogue 插入**：分配和回收栈空间的代码
3. **Callee-saved Registers 管理**：保存和恢复被调用者保存的寄存器
4. **Register Scavenging**：在寄存器压力极限时找到可用的临时寄存器

**为什么栈布局对 AI 编译器至关重要：**

1. **GPU 栈 vs CPU 栈的巨大差异**：GPU 的 local memory（即 GPU 栈）位于 global memory，延迟 ~300-800 cycles；CPU 栈在 L1 cache，延迟 ~2-5 cycles
2. **Triton 的 scratch space management**：Triton kernel 使用 shared memory 替代栈做 scratchpad，避免 global memory spill
3. **MLIR 的 buffer allocation 和 stack lowering**：MLIR 的 bufferization phase 决定哪些 buffer 分配在栈上、哪些在堆上、哪些可以复用
4. **Shared memory as scratchpad**：GPU 的 shared memory 可以充当"用户管理的栈"，完全不同于 CPU 的自动栈管理

### 栈相关组件概览

```
┌─────────────────────────────────────────────────────────┐
│           Stack Layout Components in LLVM               │
│                                                        │
│  1. Frame Index (MachineFrameInfo)                     │
│     - 创建：CreateStackObject / CreateFixedObject       │
│     - 管理：大小、对齐、偏移、栈 ID                     │
│                                                        │
│  2. Frame Lowering (TargetFrameLowering)                │
│     - emitPrologue：分配栈空间、保存 FP/RA               │
│     - emitEpilogue：恢复 FP/RA、回收栈空间              │
│     - eliminateFrameIndex：FI → SP/FP + offset         │
│                                                        │
│  3. Prologue/Epilogue Inserter (PEI pass)              │
│     - 在 RA 后运行                                      │
│     - 协调所有栈相关操作                                 │
│                                                        │
│  4. Register Scavenging (RegScavenger)                 │
│     - 在寄存器全部被分配时找空闲的物理寄存器              │
│     - 使用 emergency spill slot 保证永远有可溢出的寄存器   │
└─────────────────────────────────────────────────────────┘
```

## LLVM / MLIR 流程（深入）

### Frame Index 抽象

Frame indices 是在指令选择和合法化阶段创建的抽象栈槽：

```cpp
// 创建栈对象
int FrameIdx = MFI.CreateStackObject(Size, Alignment, /*isSS=*/false);

// 在 MachineInstr 中引用
BuildMI(MBB, MBBI, DL, TII.get(MyTarget::LDR))
    .addReg(DestReg, RegState::Define)
    .addFrameIndex(FrameIdx)     // ← Frame Index（抽象引用）
    .addImm(0);                   // ← 偏移（通常 0）
```

**三种栈对象类型：**

| 类型 | 创建方法 | 用途 |
|------|---------|------|
| **Fixed** | `CreateFixedObject(Size, Offset)` | spill slots（已知偏移）、callee-saved 寄存器、传入参数 |
| **Variable** | `CreateStackObject(Size, Align)` | 局部变量、一般 spills |
| **Variable-Sized** | `CreateVariableSizedObj(Align)` | alloca / VLA（运行时确定大小） |

**Stack ID（`TargetStackID`）区分不同的栈内存空间：**
- `Default`：常规栈内存（通过 SP/FP 访问）
- 目标特定 ID：例如某些架构的独立栈（如异常处理栈、安全栈）

### Frame Index → 物理地址的降低

```
Frame Index (int) → eliminateFrameIndex → SP/FP + 立即数偏移 → 物理地址指令
```

#### TargetFrameLowering 核心方法

```cpp
class MyFrameLowering : public TargetFrameLowering {
public:
  // 核心：将 frame index 转换为物理寄存器 + 偏移
  bool eliminateFrameIndex(MachineBasicBlock::iterator MI, int SPAdj,
                           unsigned FIOperandNum,
                           RegScavenger *RS) const override;

  // Prologue: 函数入口处分配栈
  void emitPrologue(MachineFunction &MF, MachineBasicBlock &MBB) const override;

  // Epilogue: 函数出口处回收栈
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const override;

  // 是否需要 frame pointer
  bool hasFP(const MachineFunction &MF) const override { return true; }

  // 是否保留 call frame（outgoing arguments 的栈空间）
  bool hasReservedCallFrame(const MachineFunction &MF) const override;
};
```

#### Prologue Emission 详解

Prologue 插入在函数体的第一条指令之前：

```cpp
void MyFrameLowering::emitPrologue(MachineFunction &MF,
                                    MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = MBB.begin();
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  DebugLoc DL;

  // Step 1: 保存 frame pointer（如果使用）
  if (hasFP(MF)) {
    // push FP（或等效操作）
    BuildMI(MBB, MBBI, DL, TII.get(MyTarget::PUSH))
        .addReg(TRI->getFrameRegister(MF));
  }

  // Step 2: 设置新 FP = SP（如果需要 frame pointer）
  if (hasFP(MF)) {
    BuildMI(MBB, MBBI, DL, TII.get(MyTarget::MOV))
        .addReg(TRI->getFrameRegister(MF))
        .addReg(getStackPointerRegisterToSaveRestore());
  }

  // Step 3: 调整 SP 分配本地栈空间
  uint64_t StackSize = MFI.getStackSize();
  if (StackSize > 0 && !hasReservedCallFrame(MF)) {
    BuildMI(MBB, MBBI, DL, TII.get(MyTarget::SUBri))
        .addReg(getStackPointerRegisterToSaveRestore())
        .addReg(getStackPointerRegisterToSaveRestore())
        .addImm(StackSize);
  }

  // Step 4: 保存 callee-saved 寄存器
  const std::vector<CalleeSavedInfo> &CSI = MFI.getCalleeSavedInfo();
  for (const CalleeSavedInfo &I : CSI) {
    // store I.getReg() → I.getFrameIdx()（frame index 指向栈位置）
    unsigned Opc = getStoreOpcForRegSize(TRI->getSpillSize(
        *TRI->getMinimalPhysRegClass(I.getReg())));
    BuildMI(MBB, MBBI, DL, TII.get(Opc))
        .addReg(I.getReg())
        .addFrameIndex(I.getFrameIdx())
        .addImm(0);
  }
}
```

#### Epilogue Emission 详解

Epilogue 插入在函数的每个 return 指令之前（或在函数末尾）：

```cpp
void MyFrameLowering::emitEpilogue(MachineFunction &MF,
                                    MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  DebugLoc DL;

  // Step 1: 恢复 callee-saved 寄存器（逆序）
  const std::vector<CalleeSavedInfo> &CSI = MFI.getCalleeSavedInfo();
  for (auto I = CSI.rbegin(); I != CSI.rend(); ++I) {
    unsigned Opc = getLoadOpcForRegSize(TRI->getSpillSize(
        *TRI->getMinimalPhysRegClass(I->getReg())));
    BuildMI(MBB, MBBI, DL, TII.get(Opc))
        .addReg(I->getReg(), RegState::Define)
        .addFrameIndex(I->getFrameIdx())
        .addImm(0);
  }

  // Step 2: 恢复 SP（如果使用 frame pointer）
  if (hasFP(MF)) {
    BuildMI(MBB, MBBI, DL, TII.get(MyTarget::MOV))
        .addReg(getStackPointerRegisterToSaveRestore())
        .addReg(TRI->getFrameRegister(MF));
  }

  // Step 3: 恢复 frame pointer（弹出 FP）
  if (hasFP(MF)) {
    BuildMI(MBB, MBBI, DL, TII.get(MyTarget::POP))
        .addReg(TRI->getFrameRegister(MF));
  }

  // Step 4: 返回指令在 epilogue 之后自动插入
}
```

#### eliminateFrameIndex 详解

这是**核心钩子**——将每个 frame index 引用转换为实际的地址计算：

```cpp
bool MyFrameLowering::eliminateFrameIndex(MachineBasicBlock::iterator MI,
                                           int SPAdj, unsigned FIOperandNum,
                                           RegScavenger *RS) const {
  MachineInstr &MIRef = *MI;
  MachineBasicBlock &MBB = *MI->getParent();
  MachineFunction &MF = *MBB.getParent();
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  MachineFrameInfo &MFI = MF.getFrameInfo();

  // 1. 获取 frame index
  int FrameIndex = MIRef.getOperand(FIOperandNum).getIndex();

  // 2. 计算偏移（相对于 FP 或 SP）
  int Offset = MFI.getObjectOffset(FrameIndex);

  // 3. 确定基址寄存器（FP 或 SP）
  Register BaseReg;
  if (hasFP(MF)) {
    BaseReg = TRI->getFrameRegister(MF);
    Offset += MFI.getStackSize();  // FP-relative: 需要加栈总大小
  } else {
    BaseReg = getStackPointerRegisterToSaveRestore();
    // SP-relative: 不需要调整（SP 已在 prologue 中调整）
  }

  // 4. 替换 frame index 为 基址寄存器 + 立即数偏移
  MIRef.getOperand(FIOperandNum).ChangeToRegister(BaseReg, false);
  MIRef.getOperand(FIOperandNum + 1).ChangeToImmediate(Offset);
  //    ↑ 假设 FI 是第一个操作数，offset 是第二个

  return false;  // false = not an error
}
```

**SP-relative 与 FP-relative 的选择：**
- **SP-relative**：SP 在函数执行中可能变化（alloca、动态栈分配）→ 偏移不稳定
- **FP-relative**：FP 在 prologue 中固定 → 偏移恒定 → 但需要多用一个寄存器做 FP

### Reserved Call Frame

`hasReservedCallFrame` 决定了 outgoing arguments 的栈空间分配策略：

```
Reserved Call Frame (hasReservedCallFrame = true):
  在 prologue 中一次性分配最大 outgoing args 空间
  → 不需要 ADJCALLSTACKDOWN/UP 指令
  → 更简单，但栈空间可能浪费（如果调用多个不同的函数）

Non-reserved Call Frame (hasReservedCallFrame = false):
  每次 call 前动态调整 SP（ADJCALLSTACKDOWN）
  每次 call 后动态恢复 SP（ADJCALLSTACKUP）
  → 栈空间精确，但每次 call 都有额外指令

条件（hasReservedCallFrame 返回 true）：
  - 无 variable-sized objects（alloca/VLA）
  - 或目标总是保留 call frame
```

### Register Scavenging（寄存器回收）

当**所有物理寄存器都已被分配**，但还需要一个临时寄存器时（例如在 eliminateFrameIndex 中需要计算大偏移地址），使用 register scavenging。

```cpp
class RegScavenger {
public:
  // 在当前位置前进（扫描指令以了解寄存器活跃性）
  void forward();

  // 查找指定寄存器类中未使用的寄存器
  Register FindUnusedReg(const TargetRegisterClass *RC) const;

  // 回收一个寄存器（必要时 spill 当前占用者）
  Register scavengeRegister(const TargetRegisterClass *RC,
                            MachineBasicBlock::iterator I, int SPAdj);
};
```

**Emergency Spill Slot：** 回收操作可能找不到空闲寄存器（所有寄存器都在使用且无法 spill）→ 需要预先创建紧急溢出槽：

```cpp
// 在 frame lowering 设置阶段创建
int EmergencySpillSlot = MF.getFrameInfo().CreateStackObject(
    SpillSize, SpillAlign, /*isSS=*/false);
RS->addScavengingFrameIndex(EmergencySpillSlot);
// 现在 RS 可以 spill 任何寄存器到 EmergencySpillSlot
```

## 关键机制解析（工业视角）

### 完整栈布局流程

```
┌─────────────────────────────────────────────────────┐
│  Phase 1: 栈槽创建（在 ISel/Legalization 中）         │
│  - CreateStackObject / CreateFixedObject            │
│  - 插入 frame index 引用到 MachineInstr             │
├─────────────────────────────────────────────────────┤
│  Phase 2: Register Allocation                       │
│  - 引入 spill slots（通过 storeRegToStackSlot）      │
│  - 新 frame indices 被创建用于 spill                │
├─────────────────────────────────────────────────────┤
│  Phase 3: PEI (PrologEpilogInserter) pass           │
│  3a. determineCalleeSaves：决定哪些寄存器需要保存     │
│  3b. 计算所有栈槽的偏移（assignCalleeSavedSpillSlots）│
│  3c. 插入 prologue/epilogue 代码                    │
│  3d. eliminateFrameIndex：替换所有 FI 引用            │
│  3e. scavengeRegister：处理无法直接编码的偏移         │
├─────────────────────────────────────────────────────┤
│  结果：所有 frame indices 被消除                     │
│        所有栈访问使用 SP/FP + 立即数偏移              │
└─────────────────────────────────────────────────────┘
```

### Callee-Saved Registers 的自动管理

LLVM 自动管理 callee-saved 寄存器的保存/恢复：

```cpp
// 在 PEI pass 中，determineCalleeSaves 方法决定：
//  1. 哪些 callee-saved 寄存器在函数中被修改（需要保存）
//  2. 为每个需要保存的寄存器分配栈槽
//  3. 在 prologue 中保存，在 epilogue 中恢复

// 你只需在 TableGen 中定义：
def CSR : CalleeSavedRegs<(add R4, R5, R6, R7)>;
// 表示 R4-R7 是 callee-saved
```

**为什么 FP 通常是 callee-saved 的：** Frame pointer 在函数入口被设置为当前 SP 值，调用者期望被调函数返回后 FP 不变。

## AI 编译器关联

### GPU 栈 vs CPU 栈

GPU 的栈概念与 CPU 完全不同：

```
CPU Stack:
  - 在 L1/L2 cache 中，甚至可能在 L3 cache 中
  - 延迟：~2-5 cycles (cache hit)
  - 通过 SP 寄存器的 push/pop/sub 指令自动管理
  - 硬件预取器可以预取栈内容

GPU "Stack" (= local memory):
  - 物理上位于 global memory (device DRAM)
  - 延迟：~300-800 cycles
  - 每个线程有独立的 local memory 分配
  - L1 cache 可以缓存 local memory（如果配置为 cache local）
  - 编译器在 PTX 中用 .local 声明
  - ptxas 将 .local 映射到 global memory 地址

为什么 GPU 栈如此昂贵：
  - Global memory bandwidth 有限（A100: ~2 TB/s = 100 GB/s per SM）
  - 多个 warp 竞争 DRAM 带宽
  - Local memory access 会占用 memory controller 队列
  - 一次 spill 意味着 thread 被阻塞直到 load 完成
```

### Triton 的 Scratch Space 管理

Triton kernel 不使用栈（local memory）来进行临时存储。它使用 **shared memory** 作为 scratchpad：

```python
@triton.jit
def matmul_kernel(A, B, C, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)

    # Shared memory scratchpad（替代栈）
    # shape: (BLOCK_M, BLOCK_K) 和 (BLOCK_K, BLOCK_N)
    a_sh = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float16)
    b_sh = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float16)

    # 在循环中复用 shared memory（类似 buffer allocation）
    for k in range(0, K, BLOCK_K):
        # Load to shared memory（从 global → shared）
        a_sh = tl.load(A_ptr + offsets)  # coalesced
        b_sh = tl.load(B_ptr + offsets)  # coalesced

        # 在 shared memory 上计算（从 shared → registers → shared）
        tl.dot(a_sh, b_sh, acc)

    tl.store(C_ptr + offsets, acc)
```

**Triton 的 shared memory vs 栈对比：**

| 维度 | CPU 栈 (Local Memory) | GPU Shared Memory (Triton) |
|------|----------------------|---------------------------|
| 延迟 | ~2-5 cycles | ~30 cycles (无 bank conflict) |
| 带宽 | ~100 GB/s per core | ~1.5 TB/s per SM |
| 管理 | 编译器自动（push/pop） | 用户/编译器显式分配 |
| 生命周期 | 函数作用域 | 用户定义（通常是 block 作用域） |
| 容量 | 通常 ~8 MB | ~48 KB - 228 KB per SM |
| 溢出处理 | 自动（递归分配更多栈） | 静态分配（超出则编译失败） |

### MLIR 的 Buffer Allocation 和 Stack Lowering

MLIR 的 bufferization 阶段处理类似栈分配的逻辑，但在更高抽象层：

```mlir
// 输入 MLIR (tensor dialect - 无显式 buffers)
func.func @main() {
  %A = arith.constant dense<...> : tensor<256xf32>
  %B = arith.constant dense<...> : tensor<256xf32>
  %C = linalg.matmul ins(%A, %B : tensor<256x256xf32>)
                     outs(%empty : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %C : tensor<256x256xf32>
}

// Bufferization 后 (memref dialect - 显式 buffers)
func.func @main() {
  %bufA = memref.alloc() : memref<256x256xf32>
  %bufB = memref.alloc() : memref<256x256xf32>
  %bufC = memref.alloc() : memref<256x256xf32>
  // ... fill bufA, bufB ...
  linalg.matmul ins(%bufA, %bufB : memref<256x256xf32>)
               outs(%bufC : memref<256x256xf32>)
  // ... use bufC ...
  memref.dealloc %bufA : memref<256x256xf32>
  memref.dealloc %bufB : memref<256x256xf32>
  memref.dealloc %bufC : memref<256x256xf32>
  return
}

// Buffer 优化 (类似寄存器分配中对栈槽的合并)
// 如果 bufA 和 bufB 的 live ranges 不重叠：
//   bufA 可以复用 bufC 的内存空间（减少总内存）
// 这相当于 LLVM 中对 stack slots 的合并（stack coloring）
```

**MLIR Buffer Allocation 与 LLVM Stack Lowering 的对应关系：**

| 概念 | LLVM Stack Lowering | MLIR Buffer Allocation |
|------|---------------------|----------------------|
| 存储单元 | Frame index (栈槽) | memref (buffer) |
| 生命周期 | Live intervals (SlotIndex ranges) | Liveness analysis (block/op-level) |
| 合并 | Stack slot coloring | Buffer reuse / aliasing |
| 分配 | CreateStackObject | memref.alloc() |
| 定位 | SP/FP + offset | LLVM 的 alloca 或 heap allocation |
| 优化 pass | PEI (PrologEpilogInserter) | One-shot bufferization + buffer deallocation |

### Shared Memory as Scratchpad（共享内存作为便签本）

GPU 的 shared memory 可以看作**用户显式管理的栈**：

```
Shared Memory Layout (CUDA kernel):
┌─────────────────────────────────────────────┐
│  extern __shared__ char smem[];             │
│                                             │
│  手动分区:                                    │
│    double *tile_A = (double*)smem;          │
│    double *tile_B = (double*)&smem[8192];   │
│    int    *flags   = (int*)&smem[16384];    │
│                                             │
│  类似 LLVM 的栈布局:                          │
│    - tile_A 在 offset 0（"frame index 0"）   │
│    - tile_B 在 offset 8192（"frame index 1"）│
│    - flags 在 offset 16384（"frame index 2"）│
└─────────────────────────────────────────────┘

Triton 自动管理 shared memory：
  tl.zeros/shared 分配 → Triton 编译器自动布局
  不同 block 级别变量的 live range 分析 → 复用
```

**Shared memory 相比栈的优势：**
1. 带宽：shared memory ~1.5 TB/s vs local memory ~100 GB/s per SM
2. 可预测延迟：shared memory 延迟一致（~30 cycles），不受 DRAM 调度影响
3. 无 cache miss：shared memory 是 SRAM，不会发生 cache miss（而 local memory 可能 L1 miss）

**Shared memory 相比栈的劣势：**
1. 容量限制：每 SM 仅 48-228 KB，而 local memory 可到 GB 级别
2. 管理复杂：需要显式分配和同步（`__syncthreads()`）
3. Bank conflicts：需要仔细 layout 以避免 bank conflicts

## 示例说明

### 示例 1：完整的 Prologue/Epilogue 栈帧布局

```
假设函数需要：
  - 2 个 callee-saved 寄存器（R4, R5 = 8 bytes total）
  - 3 个局部变量（4+4+8 = 16 bytes）
  - 4 bytes alignment（对齐要求）

栈帧布局（从高地址到低地址，向下增长）：
┌─────────────────────┐ ← 高地址（调用前 SP）
│  R4 (saved)         │ ← FrameOffset = -4
├─────────────────────┤
│  R5 (saved)         │ ← FrameOffset = -8
├─────────────────────┤
│  local_var1 (4B)    │ ← FrameOffset = -12
├─────────────────────┤
│  local_var2 (4B)    │ ← FrameOffset = -16
├─────────────────────┤
│  local_var3 (8B)    │ ← FrameOffset = -24
├─────────────────────┤
│  padding (4B)       │ ← 对齐到 4B: -28
├─────────────────────┤
│  outgoing args (call)│ ← 为 callee 调用预留的空间
└─────────────────────┘ ← 低地址（新 SP = 旧 SP - TotalFrameSize）

Prologue 代码:
  push FP                // 保存调用者的 FP
  mov FP, SP             // 设置新 FP = 当前 SP
  sub SP, #28            // 分配本地栈空间
  str R4, [FP, #-4]     // 保存 callee-saved R4
  str R5, [FP, #-8]     // 保存 callee-saved R5

Epilogue 代码:
  ldr R5, [FP, #-8]     // 恢复 R5
  ldr R4, [FP, #-4]     // 恢复 R4
  mov SP, FP             // 恢复 SP
  pop FP                 // 恢复调用者的 FP
  ret
```

### 示例 2：Triton Shared Memory 作为 Scratchpad 的编译器转换

```
Triton kernel (Python):
  @triton.jit
  def add_kernel(x_ptr, y_ptr, output_ptr, n_elements):
      pid = tl.program_id(0)
      offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
      x = tl.load(x_ptr + offsets)
      y = tl.load(y_ptr + offsets)
      output = x + y                 # 这里 x, y, output 都在寄存器中
      tl.store(output_ptr + offsets, output)

Triton IR (MLIR):
  tt.func @add_kernel(...) {
    %x = tt.load %x_ptr : tensor<128xf32>     # → registers
    %y = tt.load %y_ptr : tensor<128xf32>     # → registers
    %sum = arith.addf %x, %y : tensor<128xf32> # → registers
    tt.store %output_ptr, %sum : tensor<128xf32>
    return
  }

LLVM PTX (最终代码):
  // 无栈使用！所有数据在 registers 中
  ld.global.v4.f32 {%r1, %r2, %r3, %r4}, [%rd1];
  ld.global.v4.f32 {%r5, %r6, %r7, %r8}, [%rd2];
  add.f32 %r9, %r1, %r5;
  add.f32 %r10, %r2, %r6;
  // ...
  st.global.v4.f32 [%rd3], {%r9, %r10, %r11, %r12};
  ret;

  // 对比：如果寄存器不够（spill 发生）→ 会插入 .local 声明
  // .local .align 4 .b8 __local_depot0[512];
  // st.local.f32 [__local_depot0], %r1;  ← spill!
  // ld.local.f32 %r1, [__local_depot0];  ← reload!
```

### 示例 3：MLIR Buffer Allocation 的内存复用

```mlir
// 原始 MLIR（无 buffer 复用）
func.func @example() {
  %buf1 = memref.alloc() : memref<1024xf32>
  call @fill(%buf1) : (memref<1024xf32>) -> ()
  call @use(%buf1) : (memref<1024xf32>) -> ()
  memref.dealloc %buf1 : memref<1024xf32>     // ← buf1 生命周期结束

  %buf2 = memref.alloc() : memref<1024xf32>  // ← 新分配
  call @fill(%buf2) : (memref<1024xf32>) -> ()
  call @use(%buf2) : (memref<1024xf32>) -> ()
  memref.dealloc %buf2 : memref<1024xf32>
  return
}

// 优化后（buffer reuse - 类似 stack slot coloring）
func.func @example() {
  %buf = memref.alloc() : memref<1024xf32>    // ← 只分配一次！
  call @fill(%buf) : (memref<1024xf32>) -> ()
  call @use(%buf) : (memref<1024xf32>) -> ()
  // buf1 的 dealloc 移除（被 buf2 复用）
  // buf2 的 alloc 移除

  call @fill(%buf) : (memref<1024xf32>) -> ()  // 复用同一个 buffer
  call @use(%buf) : (memref<1024xf32>) -> ()
  memref.dealloc %buf : memref<1024xf32>
  return
}
// 节省了 1024*4 = 4KB 的峰值内存
```

## 总结

### 核心要点

1. **Frame indices 是抽象句柄**：`eliminateFrameIndex` 将其转换为 SP/FP + 偏移的物理地址
2. **Prologue/Epilogue 负责**：分配/回收栈空间、保存/恢复 callee-saved 寄存器、设置/恢复 FP
3. **Reserved call frame** 在 prologue 中一次性分配 outgoing args 空间，避免每次 call 都调整 SP
4. **Register scavenging** 是最后的救星：当所有物理寄存器都分配完毕时，通过 emergency spill slot 强制回收一个寄存器

### AI 编译器工程师的关键理解

| 概念 | LLVM 实践 | AI 编译器实践 |
|------|----------|-------------|
| 栈内存位置 | L1/L2 cache（CPU） | Global memory (local memory) on GPU |
| 临时存储策略 | Stack slots via CreateStackObject | Shared memory scratchpad (Triton) |
| 内存复用 | Stack slot coloring | Buffer reuse in MLIR bufferization |
| 分配管理 | Prologue/Epilogue (自动) | 显式 shared memory allocation |
| Spill 处理 | Spill to stack (frame index) | Spill to local memory (灾难性) |
| 对齐 | Stack alignment via TargetFrameLowering | Shared memory bank alignment |

### 进阶话题

- **Stack coloring**：LLVM 的 PEI pass 会合并生命周期不重叠的栈槽，减少栈内存使用（类似于寄存器分配的 live range 合并）
- **Variable-sized objects (alloca)**：当存在 VLA 时，栈帧不是固定大小，需要更复杂的栈管理（base pointer 或 dynamic stack realignment）
- **Shadow stack / SafeStack**：LLVM 支持分离数据栈和返回地址栈以增强安全性
- **GPU 的 `__launch_bounds__` 和 local memory**：CUDA 的 `__launch_bounds__` 提示可以限制编译器使用的寄存器数，从而减少 local memory spill（以 occupancy 换取更少的 spill）
