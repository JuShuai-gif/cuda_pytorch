#pragma once

#include "tile_shape.hpp"
#include "arch_tag.hpp"
#include "operator_class.hpp"

namespace cutlass_style {

// ============================================================================
// MmaPolicy - MMA 策略，Policy-Based Design 的核心
// ============================================================================
//
// WHY CUTLASS 到处都是 Policy-Based Design:
//
// ┌──────────────────────────────────────────────────────────────────┐
// │  Mermaid 图: Policy 组合爆炸与 Policy-Based Design 的解决思路  │
// │                                                                  │
// │  问题: GPU GEMM kernel 的配置维度                               │
// │  ┌─────────────────────────────────────────────────────┐        │
// │  │ 维度1: 架构        Sm70 | Sm75 | Sm80 | Sm90        │        │
// │  │ 维度2: 数据类型    FP16 | BF16 | TF32 | INT8 | FP8  │        │
// │  │ 维度3: Layout      NN | NT | TN | TT                │        │
// │  │ 维度4: Tile Size   64 | 128 | 256                   │        │
// │  │ 维度5: Warp Size   32 | 64                         │        │
// │  │ 维度6: 指令形状    16x16x8 | 16x8x16 | 8x8x16      │        │
// │  └─────────────────────────────────────────────────────┘        │
// │  笛卡尔积 = 4×5×4×3×2×3 = 1440 种组合                          │
// │                                                                  │
// │  传统做法 (❌ 继承):                                             │
// │  class GemmKernel_FP16_Sm80_NN_128x128 { ... };                 │
// │  class GemmKernel_FP16_Sm80_NT_128x128 { ... };                 │
// │  → 写 1440 个类 → 代码重复 → 维护灾难                            │
// │                                                                  │
// │  Policy-Based Design (✅):                                       │
// │  template <typename Policy> class GemmKernel { ... };            │
// │  → 1 个 kernel 类                                               │
// │  → Policy 组合在编译期解析为具体实现                              │
// │  → 每个组合都是独立模板实例化，无运行时开销                       │
// │  → 编译器可以在每个组合中做最优优化                               │
// └──────────────────────────────────────────────────────────────────┘
//
// 类比: Policy 相当于餐厅的"套餐配置"：
//   你不必点 "大份 + 辣 + 加蛋 + 加肠" 这个具体名字的菜，
//   你只需说 "大份"，然后选 "辣度=5"，"加蛋=true"。
//   Policy-Based Design 就是这种 "正交配置" 的 C++ 版本。

// ============================================================================
// MmaPolicy 主模板
// ============================================================================
//
// 编译期参数:
//   WarpShape_        - Warp 处理的子 tile 大小
//   InstructionShape_ - 单条 MMA 指令的形状
//   OperatorClass_    - Simt / TensorOp / Wmma
//   ArchTag_          - GPU 架构标签
//
// 模板展开后 (以 Sm80, FP16, 128x128 tile 为例):
//   MmaPolicy<WarpShape<64,64,32>, InsShape<16,8,16>, TensorOp, Sm80>
//   → 编译器生成:
//     - Warp 数量: (128/64) × (128/64) = 4 warps
//     - 每个 Warp 的 MMA 次数: (64/16)×(64/8)×(32/16) = 4×8×2 = 64 次
//     - Shared memory 大小: 2×(64×32 + 64×32)×2bytes = 16KB
//     - 使用 cp.async (因为 Sm80)
//     - 使用 mma.sync.aligned.m16n8k16 (因为 FP16 + Sm80)

template <
    typename WarpShape_,
    typename InstructionShape_,
    typename OperatorClass_,
    typename ArchTag_
>
struct MmaPolicy {
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;

  // =========================================================================
  // 编译期计算: Warp 的线程布局
  // =========================================================================
  //
  // WHY 线程布局在编译期确定:
  //   每个线程必须知道自己在 warp tile 中的位置，才能计算:
  //     1. 自己负责加载哪些元素到 shared memory
  //     2. 自己持有 fragment 的哪些元素
  //     3. 写回 global memory 时的偏移
  //   这些全部是固定模式，编译期确定 → 零运行时开销。

  // Warp 内部的线程排布: warp_size = M_threads × N_threads
  // 默认: 8×4 = 32 (8 行 4 列，符合访问模式)
  static constexpr int kWarpThreadArrangementM = 8;
  static constexpr int kWarpThreadArrangementN = 4;

  static_assert(
      kWarpThreadArrangementM * kWarpThreadArrangementN == 32,
      "Warp thread arrangement must be 32 (warp size)"
  );

  // 每个线程处理的 A/B/C 元素数
  static constexpr int kElementsPerThreadA =
      WarpShape::M * WarpShape::K /
      (kWarpThreadArrangementM * kWarpThreadArrangementN);

  static constexpr int kElementsPerThreadB =
      WarpShape::N * WarpShape::K /
      (kWarpThreadArrangementM * kWarpThreadArrangementN);

  // =========================================================================
  // 编译期 Shared Memory 大小计算
  // =========================================================================
  //
  // WHY 编译期计算 smem 大小:
  //   CUDA 的 __shared__ 声明需要编译期常量大小。
  //   编译期计算 → 编译器可以:
  //     - 分配静态 shared memory (比动态快)
  //     - 做 bank conflict 分析和优化
  //     - 检查是否超过硬件限制 (编译期报错)

  // A 矩阵 shared memory tile (行优先存储)
  static constexpr int kSharedMemorySizeA =
      WarpShape::M * WarpShape::K * 4;  // 4 bytes per element placeholder

  // B 矩阵 shared memory tile (列优先存储)
  static constexpr int kSharedMemorySizeB =
      WarpShape::N * WarpShape::K * 4;  // 4 bytes per element placeholder

  // 双缓冲: 需要 2 份 shared memory (load 一份时 compute 另一份)
  // 这是 GPU 软件流水线的关键优化
  static constexpr bool kUseDoubleBuffer =
      ArchTag::compute_capability >= 80; // SM80+ 有 cp.async，双缓冲收益大

  static constexpr int kSharedMemoryTotal =
      (kSharedMemorySizeA + kSharedMemorySizeB) * (kUseDoubleBuffer ? 2 : 1);

  // 编译期检查: shared memory 是否超过硬件限制
  // (注释掉的 static_assert，因为静态 constexpr 值取决于 ArchTag，在某些上下文中可能导致问题)
  // 实际 CUTLASS 用 if constexpr 做运行时检查而不是 static_assert
  static constexpr bool kSharedMemoryOk =
      (kSharedMemoryTotal <= ArchTag::max_shared_memory_per_block);

  // =========================================================================
  // 编译期 PTX 指令选择
  // =========================================================================
  //
  // WHY 需要编译期选择 PTX 指令:
  //   不同架构的 MMA 指令名称不同:
  //     SM70: mma.sync.aligned.m16n16k4.row.col.f16.f16.f16.f16
  //     SM80: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
  //     SM90: wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16
  //   如果运行时选择，编译器无法验证指令是否存在。
  //   编译期选择 → 编译时就能发现 "SM70 不支持 wgmma"。

  // 指令形状名称 (用于生成 PTX 助记符)
  static constexpr const char* mma_instruction_mnemonic = []() constexpr {
    if constexpr (is_sm70_v<ArchTag>) return "mma.sync.aligned.m16n16k4";
    else if constexpr (is_sm75_v<ArchTag>) return "mma.sync.aligned.m16n16k8";
    else if constexpr (is_sm80_v<ArchTag>) return "mma.sync.aligned.m16n8k16";
    else if constexpr (is_sm90_v<ArchTag>) return "wgmma.mma_async";
    else return "unknown";
  }();

  // =========================================================================
  // 编译期 Epilogue 配置
  // =========================================================================

  // 累加器类型 (与 Operator 一致)
  using AccumulatorType = typename Operator<OperatorClass, float, float, float>::AccumulatorType;

  // 输出转换: 是否需要类型转换 (如 FP32 acc → FP16 output)
  static constexpr bool kNeedsOutputConversion =
      !std::is_same_v<AccumulatorType, float> || sizeof(AccumulatorType) != 4;
};

// ============================================================================
// 常用 MmaPolicy 配置别名
// ============================================================================

// SM80 A100 最优配置 (FP16)
using Sm80Fp16Policy128x128 = MmaPolicy<
    WarpShape<64, 64, 32>,
    InstructionShape<16, 8, 16>,
    TensorOp,
    Sm80>;

using Sm80Fp16Policy256x128 = MmaPolicy<
    WarpShape<64, 64, 32>,
    InstructionShape<16, 8, 16>,
    TensorOp,
    Sm80>;

// SM90 H100 最优配置 (FP8)
using Sm90Fp8Policy128x256 = MmaPolicy<
    WarpShape<64, 128, 32>,
    InstructionShape<16, 8, 32>,
    TensorOp,
    Sm90>;

// SM75 T4 通用配置
using Sm75Fp16Policy128x128 = MmaPolicy<
    WarpShape<64, 64, 32>,
    InstructionShape<16, 16, 8>,
    TensorOp,
    Sm75>;

} // namespace cutlass_style
