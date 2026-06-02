# Chapter 19: Register Allocation

> **From the perspective of a production AI compiler engineer who needs to understand LLVM deeply to work on MLIR/Triton/AI compiler stacks.**

## 核心概念（详细展开）

### 寄存器分配的本质

寄存器分配（Register Allocation）是编译器后端中**最关键的性能优化之一**。它将无限的虚拟寄存器映射到有限数量的物理寄存器上，并在物理寄存器不够时生成 spill/reload 代码。

```
输入（pre-RA）:
  %v0 = add i32 %a, %b      ← 虚拟寄存器（无限数量）
  %v1 = mul i32 %v0, %c
  %v2 = add i32 %v1, %d
  return %v2

输出（post-RA）:
  $r0 = add i32 $r1, $r2     ← 物理寄存器（例如 16 个 GPR）
  $r0 = mul i32 $r0, $r3
  $r1 = add i32 $r0, Mem[sp+4]  ← spill %d 到栈
  return $r1
```

**为什么寄存器分配对 AI 编译器至关重要：**

1. **GPU 寄存器文件结构特殊**：NVIDIA GPU 每个 SM 有 65536 个 32-bit 寄存器，但每个线程可用的数量由 occupancy 决定（越多的寄存器/线程 → 越少的线程/SM → 越低的 latency hiding）
2. **Triton 的软件流水线**：Triton 大量使用 loop pipelining（prefetch + compute），这本质上在源码级别管理寄存器分配
3. **MLIR 的寄存器分配**：MLIR 的标准 pipeline 最终通过 LLVM 后端做寄存器分配，但中间层次可能需要自己的 buffer allocation
4. **Spilling 是 GPU kernel 的噩梦**：GPU 的 global memory latency 是 ~300-800 cycles，一次 spill 可能使 kernel 性能下降 10-100x

### LLVM 寄存器分配的两个阶段

```
┌──────────────────────────────────────────────────────┐
│               Register Allocation in LLVM             │
│                                                      │
│  Phase 1: Register Coalescing (RegisterCoalescer)     │
│    - 合并通过 COPY 连接的虚拟寄存器                     │
│    - 使用 AGGRESSIVE coalescing:                       │
│      尽可能消除 COPY（不管寄存器压力）                   │
│                                                      │
│  Phase 2: Register Assignment (Greedy RA / Basic RA)  │
│    - 将虚拟寄存器映射到物理寄存器或内存位置              │
│    - 负责 SPLITTING: 撤销过度 coalescing 造成的压力     │
│    - 负责 SPILLING: 物理寄存器不够时溢出到栈            │
│                                                      │
│  LLVM 的特色：                                         │
│    - SSA 形式虽已消除，但 liveness 信息依然保持 SSA    │
│    - Aggressive coalesce → split → spill              │
│      不同于传统 conservative coalesce 方法             │
└──────────────────────────────────────────────────────┘
```

**Pipeline 中的位置：**

```
Phi Elimination → Register Coalescer → Pre-RA Scheduling
    → Register Assignment (with spill code) → Post-RA Scheduling
    → Prologue/Epilogue Insertion
```

## LLVM / MLIR 流程（深入）

### 启用寄存器分配基础设施

使用默认 machine pass pipeline 时，寄存器分配**自动启用**。自定义 pipeline 时需要手动添加：

```cpp
// 在自定义 TargetPassConfig 中
bool MyPassConfig::addRegAssignAndRewriteOptimized() {
  // Register coalescer
  addPass(createRegisterCoalescerPass());
  // Greedy register allocator（生产环境推荐）
  addPass(createGreedyRegisterAllocator());
  // 或者使用 basic allocator（用于 simple/embedded targets）
  // addPass(createBasicRegisterAllocator());
  return true;
}
```

### Spilling 支持（必须实现的两个方法）

Spilling 发生在物理寄存器不够时。必须实现 `TargetInstrInfo` 的两个方法：

```cpp
// 1. Store register to stack slot
void H2BLBInstrInfo::storeRegToStackSlot(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    Register SrcReg, bool isKill, int FI,
    const TargetRegisterClass *RC, const TargetRegisterInfo *TRI,
    Register VReg, MachineInstr::MIFlag Flags) const {

  MachineFunction &MF = *MBB.getParent();
  MachineFrameInfo &MFI = MF.getFrameInfo();

  // 创建 MachineMemOperand（栈位置 + store 语义 + 大小 + 对齐）
  MachinePointerInfo PtrInfo = MachinePointerInfo::getFixedStack(MF, FI);
  MachineMemOperand *MMO = MF.getMachineMemOperand(
      PtrInfo, MachineMemOperand::MOStore,
      MFI.getObjectSize(FI), MFI.getObjectAlign(FI));

  // 根据寄存器大小选择 opcode
  unsigned Opc = TRI->getSpillSize(*RC) == 2 ? H2BLB::STRSP16
                                              : H2BLB::STRSP32;

  // 标记栈 ID
  MFI.setStackID(FI, TargetStackID::Default);

  // 构建 store 指令
  BuildMI(MBB, MBBI, DebugLoc(), get(Opc))
      .addReg(SrcReg, getKillRegState(isKill))
      .addFrameIndex(FI)
      .addImm(0)
      .addMemOperand(MMO);
}

// 2. Load register from stack slot
void H2BLBInstrInfo::loadRegFromStackSlot(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    Register DestReg, int FI, const TargetRegisterClass *RC,
    const TargetRegisterInfo *TRI, Register VReg,
    MachineInstr::MIFlag Flags) const {

  MachineFunction &MF = *MBB.getParent();
  MachineFrameInfo &MFI = MF.getFrameInfo();

  MachinePointerInfo PtrInfo = MachinePointerInfo::getFixedStack(MF, FI);
  MachineMemOperand *MMO = MF.getMachineMemOperand(
      PtrInfo, MachineMemOperand::MOLoad,
      MFI.getObjectSize(FI), MFI.getObjectAlign(FI));

  unsigned Opc = TRI->getSpillSize(*RC) == 2 ? H2BLB::LDRSP16
                                              : H2BLB::LDRSP32;

  BuildMI(MBB, MBBI, DebugLoc(), get(Opc))
      .addReg(DestReg, RegState::Define)
      .addFrameIndex(FI)
      .addImm(0)
      .addMemOperand(MMO);
}
```

**关键注意：** 必须为每个 RC（Register Class）和每种 spill size 提供 store/load 指令对。否则 alloca 会在溢出的寄存器上崩溃。

### Rematerialization（重计算 vs 溢出）

当值**重计算比 spill 更便宜**时使用 remat：

```tablegen
// 在 TableGen 中标记可重计算的指令
let isReMaterializable = 1 in
def LD16imm7 : H2BLBInstruction<"ldi16", "$dst, $imm7", ...>;
```

**LLVM 仅支持 trivial remat：** 指令的所有输入操作数必须是常数或 trivial 的。如果指令需要从内存读取，重计算可能不比 spill 便宜。

### Slot Indexes（槽索引）

Slot indexes 是理解 LLVM 寄存器分配的关键。它们是 MachineFunction 中所有指令的连续编号。

```
Machine IR dump 中的 slot indexes:
  0B   bb.0 (%ir-block.1):        ← 0B: index=0, slot=Block（基本块起始）
  16B   %3:gpr32 = COPY $w0       ← 16B: index=16, slot=Block
  32B   %4:gpr32 = ADD %3, 1
  ...
  224B  B %bb.1                    ← index=224
  240B bb.1 (%ir-block.5):         ← index=240（有 hole: 224→240=16 间隔）
  256B  ADJCALLSTACKDOWN 0, 0     ← index=256
```

**Slot Index 的关键属性：**
- 单调递增编号（保证指令顺序）
- 编号有 holes（默认间隔 16）以允许指令插入而不重新编号
- 每个 slot index 有 4 个 sub-slots（按执行顺序）：

```
执行顺序 (top → bottom):
  Block (B)      ← 基本块起始标记（无实际操作）
  Early-clobber (e) ← early-clobber 定义在此处开始生存
  Register (r)   ← 常规操作：常规定义在此处开始生存，最后使用在此处死亡
  Dead (d)       ← 死定义的生存范围终点（[16r, 16d) = 仅在 index 内有效）

简写记忆法: B-e-r-d (Berd... "bird")
```

### Live Intervals（活跃区间）

Live intervals 使用 slot index 范围来描述虚拟寄存器在函数中的存活位置。

```
Machine IR:
  0B bb.0
  ...
  80B   %10:gpr32 = COPY $w0               ← 定义
  ...
  208B  Bcc 1, %bb.2, ...
  224B  B %bb.1
  240B bb.1:
  ...
  320B  %10:gpr32 = ... %10                ← 重定义 + 使用
  368B bb.2:
  400B  $w0 = COPY %10:gpr32               ← 最后使用

Live Interval for %10:
  # Segments（活跃片段，{ 表示包含，} 表示不包含）
  [80r,320r:0)    ← value 0: 从定义(80r)到重定义前(320r)
  [320r,368B:1)   ← value 1: 从重定义(320r)到 bb.2 开始(368B)
  [368B,400r:2)   ← value 2: phi 定义(368B)到最后使用(400r)

  # VNInfo（Value Number 信息 — 保持 SSA 形式）
  0@80r           ← value 0 在 80r 定义（实际定义）
  1@320r          ← value 1 在 320r 定义（实际定义）
  2@368B-phi      ← value 2 是 phi 定义（抽象，无实际指令）
```

**关键洞察：** 虽然 Machine IR 已经脱离了 SSA 形式，但 Live Intervals **依然保持 SSA 形式**。这允许在非 SSA 的 Machine IR 中执行 SSA 级别的分析。

#### 使用 Live Intervals 查找 reaching definition

```cpp
// 给定：虚拟寄存器 %10 在 slot index 400 处的一个 use
// 目标：找到这个 use 的 reaching definition(s)

LiveIntervals *LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
LiveInterval &LI = LIS->getInterval(VirtReg);  // 获取 %10 的 live interval
SlotIndex UseIdx = LIS->getInstructionIndex(*UseMI);  // 400

const VNInfo *VNI = LI.getVNInfoAt(UseIdx);  // 返回 VNInfo{2@368B-phi}

if (VNI->isPHIDef()) {
  // 这是 phi 定义 → 需要遍历前驱基本块
  // 在各前驱块的末尾查找 reaching definition
  for (MachineBasicBlock *Pred : MBB->predecessors()) {
    SlotIndex PredEnd = LIS->getMBBEndIdx(Pred);
    const VNInfo *PredVNI = LI.getVNInfoAt(PredEnd);
    // PredVNI 指向 Pred 中的 reaching definition
  }
} else {
  // 这是实际定义
  SlotIndex DefIdx = VNI->def;  // 定义的 slot index
  MachineInstr *DefMI = LIS->getInstructionFromIndex(DefIdx);
}
```

**与传统 reaching definition analysis 的对比：**
- 传统方法：需要迭代数据流分析（O(n²) 或 O(n³)）
- Live Interval 方法：O(1) 查询（通过 VNInfo 直接定位）

#### Key LiveInterval/LiveIntervals API

```cpp
// LiveInterval 查询
bool liveAt(SlotIndex Idx) const;                // 在某个 slot index 是否存活
bool overlaps(const LiveInterval &Other) const;  // 是否与另一个 LI 冲突
const VNInfo *getVNInfoAt(SlotIndex Idx) const;  // 在某个位置的 value number
bool hasSubRanges() const;                       // 是否启用 subregister tracking
iterator_range<subrange_iterator> subranges();   // subregister 子范围（lane masks）

// LiveIntervals 管理
LiveInterval &getInterval(Register Reg);
SlotIndex getInstructionIndex(const MachineInstr &MI) const;
void createEmptyInterval(Register Reg);
void createAndComputeVirtRegInterval(Register Reg);
```

### 维护 Live Intervals

如果在 pass 中修改 IR 且依赖 live intervals，必须手动维护：

| 场景 | 方法 |
|------|------|
| 创建新虚拟寄存器（无 def/use） | `createEmptyInterval(Reg)` |
| 创建新虚拟寄存器（需要计算） | `createAndComputeVirtRegInterval(Reg)` |
| 删除 use | `shrinkToUses(Reg)` |
| 插入 use | `extendToIndices(Reg, Indices)` |
| 插入/删除指令 | `insertMachineInstrInMaps(MI)` / `removeMachineInstrFromMaps(MI)` |

**最佳实践：**
- 总是启用 machine verifier（`-verify-machineinstrs`）来检测一致性错误
- Live intervals 的重计算**编译时间代价高昂**，在 RA pipeline 中间运行时应避免 invalidate

## 关键机制解析（工业视角）

### LLVM 的 Aggressive Coalescing 策略

LLVM 选择 aggressive 而非 conservative coalescing 的原因：

```
Conservative Coalescing（教科书方法）:
  仅当 coalesce 不会增加寄存器压力时才合并
  → 更少的 spill（但更多 COPY 指令）

Aggressive Coalescing（LLVM 方法）:
  尽可能合并（消除 COPY）
  → 可能过度合并 → 增加寄存器压力 → 然后由 RA 做 splitting
  → 好处：RA 阶段的 splitting 比 coalescing 阶段更智能
     （因为 RA 有完整的 liveness 信息）
  → 坏处：RA 阶段更复杂（需要处理 splitting/spilling）
```

### 溢出（Spilling）的代价模型

```
CPU spill 代价：
  store: ~2-5 cycles (L1 cache hit)
  load:  ~2-5 cycles (L1 cache hit)
  → 可控，通常不太影响性能

GPU spill 代价（NVIDIA A100 参考）：
  store: ~30 cycles (shared memory) 或 ~300 cycles (global memory)
  load:  ~30 cycles (shared memory) 或 ~300 cycles (global memory)
  shared memory spill：相对可接受（比 L1 cache 慢但远快于 global memory）
  global memory spill：灾难级（local memory = global memory on GPU）

GPU kernel 中的 spill 影响：
  - 增加 register pressure → 减少 occupancy（线程数/SM）
  - 降低 warp-level latency hiding
  - 可能使 kernel 从 compute-bound 变成 memory-bound
```

### 分配顺序和提示

```cpp
// 分配顺序（AltOrders）
// 在 TableGen 中：
def GPR32 : RegisterClass<"H2BLB", [i32], 32,
  (add R0, R1, R2, R3, R4, R5, R6, R7)> {
  let AltOrders = [(add R7, R6, R5, R4, R3, R2, R1, R0)];
  // 不同顺序可用于不同策略（如优先使用 caller-saved）
}

// 分配提示（Allocation Hints）
const TargetRegisterClass *
MyRegisterInfo::getRegAllocationHints(Register VirtReg,
    ArrayRef<MCPhysReg> Order, SmallVectorImpl<MCPhysReg> &Hints,
    const MachineFunction &MF, const VirtRegMap *VRM,
    const LiveRegMatrix *Matrix) const {
  // 建议首选寄存器（如 hint 到物理寄存器 R0 以减少 COPY）
  Hints.push_back(MyTarget::R0);
  return &MyTarget::GPR32RegClass;
}
```

## AI 编译器关联

### GPU 寄存器文件架构

NVIDIA GPU 的寄存器文件与 CPU 截然不同：

```
A100 SM Register File:
  - 65536 x 32-bit registers per SM
  - 划分为 4 个 processing blocks（各 16384 registers）
  - 每个 register 有唯一物理地址，但被分为多个 bank
  - Bank width: 32 bits（每个 bank 每 cycle 可服务 1 个 32-bit 操作数）

A100 Occupancy 示例:
  寄存器/线程 | 线程/SM | 总寄存器 | Occupancy
  32         | 2048   | 65536   | 100% (max)
  64         | 1024   | 65536   | 100%
  128        | 512    | 65536   | 100%
  255        | 256    | 65280   | ~99.6%
  256        | 256    | 65536   | ~99.6%
  320        | 192    | 61440   | ~93.8% (寄存器文件开始不足)

  NVIDIA limit: 每个线程最多 255 registers（在 PTX 级别）
  编译器可以用更多，但会通过 local memory 实现
```

**GPU 寄存器分配的特殊考量：**
- Register bank conflicts：同一 warp 内不同线程同时访问不同 bank 是高效的；同时访问同一 bank 会产生 conflict
- 寄存器文件 port 限制：每个 SM 分区每 cycle 只能读有限数量的寄存器操作数

### Triton 寄存器分配（软件流水线）

Triton 编译器在生成代码时**不使用** LLVM 的寄存器分配器。它通过软件流水线（SW pipelining）在源级别管理寄存器：

```
Triton kernel 的寄存器管理：

原始 Triton kernel（高 level）：
  for k in range(0, K, BLOCK_K):
      a = tl.load(A_ptr)      # 加载
      b = tl.load(B_ptr)      # 加载
      acc += tl.dot(a, b)     # 计算
      A_ptr += stride_a       # 指针更新
      B_ptr += stride_b

编译器优化后（软件流水线 - prefetch + compute）：
  a_pre = tl.load(A_ptr)           # 预取 tile 0
  b_pre = tl.load(B_ptr)           # 预取 tile 0
  for k in range(1, K//BLOCK_K):
      a_cur, b_cur = a_pre, b_pre          # 轮转
      a_pre = tl.load(A_ptr + stride_a)    # 预取 tile k+1
      b_pre = tl.load(B_ptr + stride_b)    # 预取 tile k+1
      acc += tl.dot(a_cur, b_cur)          # 计算 tile k
      A_ptr += stride_a
      B_ptr += stride_b
  # 最后一轮
  acc += tl.dot(a_pre, b_pre)

寄存器使用分析：
  无流水线：a(1), b(1), acc(1), ptr(2) = 5 个主要值
  有流水线：a_pre(1), b_pre(1), a_cur(1), b_cur(1), acc(1), ptr(2) = 7 个主要值
  额外开销：~2 个寄存器（~40% 增加），但可以隐藏 load 延迟
```

**Triton 寄存器分配的权衡：**
- Triton 的 loop pipelining 本质上是在**编译器源级别**显式管理寄存器
- 优点是精确控制（比 LLVM RA 更确定性的结果）
- 缺点是：对复杂控制流（non-trivial CFG）难以应用

### MLIR 寄存器分配

MLIR 本身不做寄存器分配——它通过 `ConvertToLLVMDialect` 将 IR 转换为 LLVM dialect，然后委托给 LLVM 后端：

```
MLIR pipeline:
  func.func @kernel(%arg0: memref<256xf32>) {
    %0 = memref.load %arg0[%c0] : memref<256xf32>
    %1 = arith.addf %0, %0 : f32
    memref.store %1, %arg0[%c1] : memref<256xf32>
    return
  }
      │ ConvertToLLVMDialect pass
      ▼
  llvm.func @kernel(%arg0: !llvm.ptr) {
    %0 = llvm.load %arg0 : !llvm.ptr -> f32
    %1 = llvm.fadd %0, %0 : f32
    %2 = llvm.getelementptr %arg0[1] : (!llvm.ptr) -> !llvm.ptr
    llvm.store %1, %2 : f32, !llvm.ptr
    llvm.return
  }
      │ LLVM backend (Machine IR → Register Allocation)
      ▼
  Assembly (with physical registers)
```

但 MLIR 在 bufferization 阶段做类似寄存器分配的优化——**buffer allocation**：

```mlir
// MLIR 的 buffer allocation（one-shot bufferization）
// 类似于寄存器分配：决定哪些 buffer 可以复用
%buf0 = memref.alloc() : memref<64xf32>
%buf1 = memref.alloc() : memref<64xf32>
// bufferization 分析后：如果 buf0 和 buf1 不重叠
// → 可以合并为同一个 buffer（减少内存使用）
// → 这与寄存器分配的 live range 分析类似
```

### GPU Kernel 中的 Spilling 噩梦

```
实际案例：A100 上的矩阵乘法 kernel

无 spill（32 registers/thread）：
  - Occupancy: 2048 threads/SM (100%)
  - 共享内存延迟隐藏：优秀
  - 吞吐: ~312 TFLOPS (close to peak)

有 spill（需要 40 registers → 被限制到 32 + spill）：
  - PTX 使用 .local 声明额外数据
  - ptxas 将 .local 映射到 global memory (l1 cache-backed)
  - 每次 spill load: ~30 cycles (L1 hit) 或 ~300 cycles (L1 miss)
  - 吞吐: 可能下降到 ~50-100 TFLOPS（3-6x 降低）

缓解策略：
  1. 使用 __launch_bounds__ 提示编译器目标 occupancy
  2. 手动重写 kernel 减少 live range（拆分大循环）
  3. 使用 warp shuffle 而非 shared memory 做 reduction
  4. 启用 ptxas 的 maxregcount 选项强制限制寄存器使用
```

## 示例说明

### 示例 1：LLVM Coalescing + Splitting 完整流程

```
输入 (SSA Machine IR):
  %v1 = COPY $r1          // argument
  %v2 = ADD %v1, 1
  %v3 = COPY %v2          // copy 1: coalesce candidate
  %v4 = MUL %v3, 2
  %v5 = COPY %v4          // copy 2: coalesce candidate
  $r0 = COPY %v5          // return value

Phase 1: Aggressive Coalescing:
  合并 %v1↔%v2↔%v3↔%v4↔%v5 → 一个大的 live range
  COPY 全部消除！但 live range 覆盖整个函数 → 高寄存器压力

Phase 2: Register Assignment (Greedy RA):
  发现大 live range 导致寄存器不足
  → splitting: 在中点拆分 live range → 产生两个较小的 live range
  → 在 split 点插入 spill/reload（如需要）
  → 分配物理寄存器

最终：
  可能 spill 一些值到栈，但总体代码依然正确且更高效
```

### 示例 2：Live Interval 子范围（Subranges）

```cpp
// 启用 subregister liveness tracking
bool H2BLBSubtarget::enableSubRegLiveness() const override {
  return true;  // 为使用 subregister 的代码提供更精确追踪
}

// 效果：
// 无 subrange:
//   %0:gpr32 的 live interval 覆盖整个函数
//   → 即使只用 sub_low16，整个 gpr32 都被标记为活跃
//   → 任何使用 gpr32 的其他变量都不能共享物理寄存器
//
// 有 subrange:
//   %0:gpr32 的 live interval 分两个子 range:
//     sub_low16:  [0r, 100r)   ← 仅低 16 位活跃
//     sub_high16: [50r, 80r)   ← 仅高 16 位活跃
//   → 其他变量可以在 [80r, 100r] 复用高 16 位
//   → 更好的寄存器利用率
```

### 示例 3：GPU Register Spilling 检测

```bash
# 使用 nvcc 编译并检查 spill
nvcc -Xptxas -v kernel.cu -o kernel

# 输出：
# ptxas info: Used 64 registers, 384 bytes cmem[0], 16 bytes cmem[2]
#               ↓ 关键信息

# 检查是否有 local memory（即 spill）：
cuobjdump -sass kernel | grep STL   # store local = spill store
cuobjdump -sass kernel | grep LDL   # load local = spill load

# 如果看到 STL/LDL 指令 → 发生了 spill → 可能导致严重性能下降
```

## 总结

### 核心要点

1. **LLVM 的两个阶段**：Aggressive Coalescing（消除 COPY）+ Register Assignment（splitting/spilling）
2. **Live Intervals 是 SSA in non-SSA**：这是 LLVM RA 最独特的特性，允许在非 SSA IR 中进行 O(1) reaching definition 查询
3. **Slot Indexes** 的 4 个 sub-slots（Block/Early-clobber/Register/Dead）精确建模了指令执行的时间线
4. **Spilling 是 GPU kernel 的主要性能杀手**：一次 spill 可能使性能下降 10-100x

### AI 编译器工程师的关键理解

| 概念 | LLVM 实践 | AI 编译器实践 |
|------|----------|-------------|
| 寄存器分配算法 | Greedy RA（默认）/ Basic RA / PBQP RA | Triton: 软件流水线预分配；MLIR: LLVM 委托 |
| Live ranges | LiveInterval + VNInfo（SSA in non-SSA） | Triton: 源码级循环展开管理 |
| Spilling 处理 | storeRegToStackSlot / loadRegFromStackSlot | GPU spill = global memory = 灾难 |
| 寄存器压力管理 | Aggressive coalesce → split | Triton: `num_warps` 和 `maxnreg` 提示 |
| Occupancy | N/A（CPU 无此概念） | GPU: 寄存器数/线程 直接影响 occupancy |
| Rematerialization | `isReMaterializable` + trivial 常量 | 在 GPU 上重计算通常优于 spill（避免 mem access） |

### 进阶话题

- **Greedy vs Basic vs PBQP RA**：Greedy RA 是生产环境首选（质量好 + 可接受的编译时间），Basic RA 用于简单/嵌入式目标，PBQP RA 用于需要全局最优分配的场景
- **RegAllocFast**：在 -O0 编译下使用，牺牲分配质量换取编译速度
- **MLIR 的 buffer allocation** 和 register allocation 的关系：buffer allocation 在更高抽象层做类似的 "资源复用" 分析，但处理的是 buffer 而非寄存器
- **迭代 RA + 调度**：理论上可以通过反馈循环来改进——RA 产生 spill 后重新调度（LLVM 目前不支持），这在寄存器稀缺的架构上是一个活跃研究领域
