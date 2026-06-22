#pragma once

#include <type_traits>
#include <cstdint>

#include "include/kernel_traits.hpp"
#include "include/type_list.hpp"
#include "include/layout.hpp"
#include "default_gemm_config.hpp"

namespace cutlass_style {
namespace dispatch {

// ============================================================================
// KernelDispatch - 编译期 Kernel 选择 (SFINAE 多级分派)
// ============================================================================
//
// WHY SFINAE 分派: CUTLASS 的 dispatch 使用 SFINAE (Substitution Failure
// Is Not An Error) 来在编译期选择 kernel。这允许我们在不同的架构、
// 数据类型和 layout 组合下有不同的候选 kernel，编译器会自动选择
// 第一个匹配的。
//
// ┌──────────────────────────────────────────────────────────────────┐
// │  Mermaid 图: KernelDispatch 多层分派流程                         │
// │                                                                  │
// │  用户调用:                                                      │
// │  gemm<float, half, float, RowMajor, ColMajor, RowMajor>(...)    │
// │                                                                  │
// │  DispatchGemm::Kernel                                           │
// │  ┌──────────────────────────────┐                                │
// │  │ Level 1: 架构分派             │                               │
// │  │ if (sm >= 90) → Sm90Kernel   │ ← is_sm90_v SFINAE            │
// │  │ if (sm >= 80) → Sm80Kernel   │ ← is_sm80_v SFINAE            │
// │  │ if (sm >= 75) → Sm75Kernel   │ ← is_sm75_v SFINAE            │
// │  │ else          → Sm70Kernel   │ ← 兜底                        │
// │  └──────────────┬───────────────┘                                │
// │                 ▼                                                │
// │  ┌──────────────────────────────┐                                │
// │  │ Level 2: 数据类型分派         │                               │
// │  │ if (FP16) → Fp16Traits       │                               │
// │  │ if (BF16) → Bf16Traits       │ ← 只对 SM80+ 有效              │
// │  │ if (INT8) → Int8Traits       │ ← 只对 SM75+ 有效              │
// │  │ if (FP32) → Fp32Traits       │                               │
// │  └──────────────┬───────────────┘                                │
// │                 ▼                                                │
// │  ┌──────────────────────────────┐                                │
// │  │ Level 3: Layout 分派          │                               │
// │  │ if (NN) → GemmNnKernel       │ ← RowMajor × ColumnMajor      │
// │  │ if (NT) → GemmNtKernel       │ ← 需要转置 B                   │
// │  │ if (TN) → GemmTnKernel       │ ← 需要转置 A                   │
// │  │ if (TT) → GemmTtKernel       │ ← A 和 B 都转置               │
// │  └──────────────┬───────────────┘                                │
// │                 ▼                                                │
// │  ┌──────────────────────────────┐                                │
// │  │ Level 4: Tile Size 分派       │                               │
// │  │ DefaultGemmConfiguration     │ ← 自动选择 tile                │
// │  └──────────────┬───────────────┘                                │
// │                 ▼                                                │
// │  ┌──────────────────────────────┐                                │
// │  │ 最终 Kernel:                 │                                │
// │  │ GemmKernel<Traits>           │ ← 单次模板实例化               │
// │  └──────────────────────────────┘                                │
// └──────────────────────────────────────────────────────────────────┘
//
// 类比: KernelDispatch 相当于网络路由器的路由表:
//   每个进来的"包" (GEMM 请求) 经过多级路由规则匹配，
//   最终转发到正确的"端口" (Kernel 实例)。

// ============================================================================
// 编译期 Bool 常量 (用于 SFINAE 条件)
// ============================================================================

template <bool B>
using enable_if_t = std::enable_if_t<B, int>;

// ============================================================================
// DispatchGemm 主模板 - 多级 SFINAE 分派
// ============================================================================
//
// 模板参数顺序是有意设计的:
//   ArchTag 放第一个 → 架构变化最不频繁，放在最外层分派
//   ElementType 放第二 → 数据类型组合有限
//   Layout 放最后 → 同一数据和架构下 layout 变化最多
//
// WHY 这个顺序: 减少模板实例化数量。
//   先按架构分 → 每种架构 1 份 × 数据类型 × layout
//   如果先按 layout 分 → 每种 layout 1 份 × 架构 × 数据类型
//   前者更少。这是"基数排序"的类比。

template <
    typename ArchTag,
    typename ElementA,
    typename LayoutA = RowMajor,
    typename ElementB = float,
    typename LayoutB = ColumnMajor,
    typename ElementC = float,
    typename LayoutC = RowMajor,
    typename OperatorClass = TensorOp
>
struct DispatchGemm {
 private:
  // ── Level 1: 架构分派 ──
  //
  // 使用 if constexpr 链 (C++17 特性) 来选择架构
  // 每个分支内使用不同的 tile 配置和 MMA 指令

  template <typename _ArchTag>
  struct ArchDispatcher {
    // 默认实现: SM70 回退
    template <typename _E = _ArchTag>
    static constexpr auto get_tile_shape() {
      if constexpr (is_sm90_v<_E>) {
        // SM90: 最大 tile (H100 有更多 shared memory)
        return GemmShape<256, 256, 64>{};
      } else if constexpr (is_sm80_v<_E>) {
        // SM80: 大 tile (A100 164KB shared memory)
        return GemmShape<256, 128, 32>{};
      } else if constexpr (is_sm75_v<_E>) {
        // SM75: 中等 tile (T4 64KB shared memory)
        return GemmShape<128, 128, 32>{};
      } else {
        // SM70: 较小 tile (V100 96KB shared memory)
        return GemmShape<128, 64, 32>{};
      }
    }

    template <typename _E = _ArchTag>
    static constexpr auto get_warp_shape() {
      if constexpr (is_sm90_v<_E>) {
        return cutlass_style::WarpShape<64, 128, 64>{};
      } else if constexpr (is_sm80_v<_E>) {
        return cutlass_style::WarpShape<64, 64, 32>{};
      } else if constexpr (is_sm75_v<_E>) {
        return cutlass_style::WarpShape<64, 64, 32>{};
      } else {
        return cutlass_style::WarpShape<32, 64, 32>{};
      }
    }

    template <typename _E = _ArchTag>
    static constexpr auto get_instruction_shape() {
      if constexpr (is_sm90_v<_E>) {
        return cutlass_style::InstructionShape<16, 8, 32>{};   // FP8 MMA for Hopper
      } else if constexpr (is_sm80_v<_E>) {
        return cutlass_style::InstructionShape<16, 8, 16>{};   // FP16 MMA for Ampere
      } else if constexpr (is_sm75_v<_E>) {
        return cutlass_style::InstructionShape<16, 16, 8>{};   // FP16 MMA for Turing
      } else {
        return cutlass_style::InstructionShape<16, 16, 4>{};   // FP16 MMA for Volta
      }
    }
  };

  using TileShape = decltype(ArchDispatcher<ArchTag>::get_tile_shape());
  using WarpShape = decltype(ArchDispatcher<ArchTag>::get_warp_shape());
  using InstructionShape = decltype(ArchDispatcher<ArchTag>::get_instruction_shape());

 public:
  // ── 最终选择的 Kernel Traits ──
  using KernelTraits = GemmKernelTraits<
      ElementA, ElementB, ElementC,
      LayoutA, LayoutB, LayoutC,
      ArchTag,
      OperatorClass,
      TileShape,
      WarpShape,
      InstructionShape
  >;

  // 注: 上面的 decltype 需要在真实代码中替换为直接使用 GemmShape/WarpShape 等值

  // 编译期验证
  static constexpr void validate() {
    KernelTraits::validate();
  }

  // 硬件信息 (用于运行时日志/调试)
  static constexpr const char* arch_name = ArchTag::name;

  static constexpr int tile_m = KernelTraits::TileShape::M;
  static constexpr int tile_n = KernelTraits::TileShape::N;
  static constexpr int tile_k = KernelTraits::TileShape::K;

  static constexpr int block_threads = KernelTraits::kThreads;
  static constexpr int smem_bytes = KernelTraits::kSharedMemorySize;
};

// ============================================================================
// 实际可用的 DispatchGemm (使用默认配置的简便版)
// ============================================================================

// 简化版: 自动从 DefaultGemmConfiguration 选择 tile
template <
    typename ArchTag,
    typename ElementA,
    typename ElementB,
    typename ElementC,
    typename LayoutA = RowMajor,
    typename LayoutB = ColumnMajor,
    typename LayoutC = RowMajor
>
using AutoDispatchGemm = DispatchGemm<
    ArchTag, ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    TensorOp
>;

} // namespace dispatch
} // namespace cutlass_style
