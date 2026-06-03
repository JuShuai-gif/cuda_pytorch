#pragma once

#include <cstdint>
#include <type_traits>

#include "arch_tag.hpp"
#include "layout.hpp"
#include "tile_shape.hpp"
#include "operator_class.hpp"
#include "mma_policy.hpp"

namespace cutlass_style {

// ============================================================================
// GemmKernelTraits - 编译期 Kernel 配置聚合
// ============================================================================
//
// WHY 需要一个"上帝结构体"来聚合所有配置:
//
// ┌──────────────────────────────────────────────────────────────────┐
// │  Mermaid 图: KernelTraits 作为编译期配置中心                    │
// │                                                                  │
// │  GemmKernelTraits                                                │
// │  ┌──────────────────────────────────────────────────────────┐   │
// │  │                                                          │   │
// │  │  ElementA ────┐    LayoutA ────┐    ArchTag             │   │
// │  │  ElementB ────┤    LayoutB ────┤    OperatorClass        │   │
// │  │  ElementC ────┤    LayoutC ────┤    TileShape            │   │
// │  │               │                │    WarpShape            │   │
// │  │               │                │    InstructionShape      │   │
// │  │               │                │                         │   │
// │  │               ├────────────────┤                         │   │
// │  │               │  Operator<>    │                         │   │
// │  │               │                │                         │   │
// │  │               ├────────────────┤                         │   │
// │  │               │  MmaPolicy<>   │                         │   │
// │  │               │                │                         │   │
// │  │               ▼                ▼                         │   │
// │  │  ┌─────────────────────────────────────────────────┐     │   │
// │  │  │ 编译期推导出的配置:                              │     │   │
// │  │  │  - smem_size  (shared memory 分配大小)           │     │   │
// │  │  │  - thread_block_shape (CUDA block 维度)          │     │   │
// │  │  │  - warp_count (一个 block 中的 warp 数)          │     │   │
// │  │  │  - iterations_per_warp (warp 级循环次数)         │     │   │
// │  │  │  - ptx_instruction (生成的 PTX 指令助记符)       │     │   │
// │  │  └─────────────────────────────────────────────────┘     │   │
// │  └──────────────────────────────────────────────────────────┘   │
// │                                                                  │
// │  所有这些计算在编译期完成。运行时的 kernel 代码中:               │
// │    smem_size → 直接写为常量 16384 (而非读取变量)                │
// │    warp_count → 展开为 4 份 warp 代码 (而非循环)                │
// │    ptx_instruction → 直接 inline asm，无分支                      │
// └──────────────────────────────────────────────────────────────────┘
//
// 类比: GemmKernelTraits 相当于 C 构建系统中的"配置头文件"
//       把所有的 #define 集中在一个地方。
//       但 Trait 是类型安全的——类型系统保证配置的一致性。
//       比如: LayoutA=RowMajor, LayoutB=ColumnMajor → 自动推导最优 tile。

template <
    // 数据类型
    typename ElementA_,
    typename ElementB_,
    typename ElementC_,

    // 存储 Layout
    typename LayoutA_,
    typename LayoutB_,
    typename LayoutC_,

    // 架构
    typename ArchTag_,

    // 计算类型
    typename OperatorClass_,

    // Tile 层次
    typename TileShape_,
    typename WarpShape_,
    typename InstructionShape_
>
struct GemmKernelTraits {
  // ── 基本类型别名 ──
  using ElementA = ElementA_;
  using ElementB = ElementB_;
  using ElementC = ElementC_;

  using LayoutA = LayoutA_;
  using LayoutB = LayoutB_;
  using LayoutC = LayoutC_;

  using ArchTag = ArchTag_;
  using OperatorClass = OperatorClass_;

  using TileShape = TileShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;

  // ── 组合类型 ──
  using Operator = cutlass_style::Operator<OperatorClass, ElementA, ElementB, ElementC>;
  using AccumulatorType = typename Operator::AccumulatorType;

  using Policy = MmaPolicy<WarpShape, InstructionShape, OperatorClass, ArchTag>;

  // ── 编译期验证: 配置合法性 ──
  //
  // WHY static_assert 在此: 在编译期就发现配置错误，而非运行时崩溃。
  //   比如: SM70 不支持 BF16 → 编译错误，清晰明确
  static constexpr void validate() {
    // 架构-数据类型兼容性 (注释: 这是伪代码，真实 CUTLASS 使用更精细的 SFINAE)
    // 注: std::is_same_v<ElementA, float> 太宽泛，
    // 实际 CUTLASS 用 __half, __nv_bfloat16 等专用类型来区分
    // 这里为了演示目的做了简化
  }

  // ── 编译期常量: Thread Block 配置 ──

  // Block 中的 Warp 数量
  static constexpr int kWarpCountM = TileShape::M / WarpShape::M;
  static constexpr int kWarpCountN = TileShape::N / WarpShape::N;
  static constexpr int kWarpCount = kWarpCountM * kWarpCountN;

  // Block 中的线程数
  static constexpr int kThreadsPerWarp = 32;
  static constexpr int kThreads = kWarpCount * kThreadsPerWarp;

  // CUDA block 维度 (用于 <<<grid, block>>> 启动配置)
  // WHY 二维 block: warp 在 M 和 N 维度上的排布影响 shared memory bank 访问模式
  static constexpr int kBlockDimX = kWarpCountN * kThreadsPerWarp; // N 方向
  static constexpr int kBlockDimY = kWarpCountM;                    // M 方向 (每个"行" 32 线程)

  // ── 编译期常量: 迭代次数 ──

  // Mainloop 迭代次数 (K 维度上的 tile 迭代)
  // 每次迭代处理 WarpShape::K 个 K 维度元素
  // 整个 K 维度分 (总K / Tile::K) 轮，每轮内分 (Tile::K / Warp::K) 步
  static constexpr int kMainloopIterationsPerTile = TileShape::K / WarpShape::K;

  // Warp 内的 MMA 指令次数
  static constexpr int kMmaIterationsM = WarpShape::M / InstructionShape::M;
  static constexpr int kMmaIterationsN = WarpShape::N / InstructionShape::N;
  static constexpr int kMmaIterationsK = WarpShape::K / InstructionShape::K;
  static constexpr int kMmaIterationsTotal =
      kMmaIterationsM * kMmaIterationsN * kMmaIterationsK;

  // ── 编译期常量: Shared Memory ──
  static constexpr int kSharedMemorySize = Policy::kSharedMemoryTotal;

  // ── 编译期常量: PTX 指令 ──
  static constexpr const char* kPtxMmaInstruction =
      Policy::mma_instruction_mnemonic;

  // ── 性能预估 (编译期) ──

  // 每个 Block 的计算量 (FMA ops)
  static constexpr int kComputeOpsPerBlock =
      TileShape::M * TileShape::N * TileShape::K * 2; // 2 ops per FMA (mul + add)

  // 每个 Block 的数据加载量 (bytes)
  static constexpr int kDataLoadPerBlock =
      TileShape::M * TileShape::K * sizeof(ElementA) +
      TileShape::N * TileShape::K * sizeof(ElementB);

  // 计算密度 (ops/byte) - 高密度 = 计算瓶颈，低密度 = 带宽瓶颈
  static constexpr double kComputeIntensity =
      static_cast<double>(kComputeOpsPerBlock) / kDataLoadPerBlock;
};

// ============================================================================
// 便利 Trait 构建器 - 使用部分默认参数
// ============================================================================
//
// WHY: 大多数情况下 OperatorClass 和某些 shape 可以从
//      ArchTag 和 ElementType 推导出来。提供合理的默认值
//      减少模板参数数量。

template <
    typename ElementA,
    typename ElementB,
    typename ElementC,
    typename LayoutA = RowMajor,
    typename LayoutB = ColumnMajor,
    typename LayoutC = RowMajor,
    typename ArchTag = Sm80,
    typename OperatorClass = TensorOp,
    typename TileShape = TileShape128x128x32,
    typename WarpShape = WarpShape64x64x32,
    typename InstructionShape = Mma16x8x16
>
using DefaultGemmTraits = GemmKernelTraits<
    ElementA, ElementB, ElementC,
    LayoutA, LayoutB, LayoutC,
    ArchTag,
    OperatorClass,
    TileShape, WarpShape, InstructionShape
>;

} // namespace cutlass_style
