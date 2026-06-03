#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cutlass_style {

// ============================================================================
// GemmShape<M, N, K> - 编译期 Tile 形状
// ============================================================================
//
// WHY 用模板参数而非运行时变量？
//
// ┌──────────────────────────────────────────────────────────────────┐
// │  Mermaid 图: 运行时 tile vs 编译期 tile 的 PTX 代码对比         │
// │                                                                  │
// │  运行时 (❌):                                                    │
// │  for (int k = 0; k < K_tile; k += K_block) {  // 循环边界变量   │
// │      ld.shared r0, [smem + offset];           // offset 运行时算 │
// │      mma.sync r2, r0, r1;                     // MMA size 固定  │
// │  }                                                               │
// │  → 循环展开次数未知，编译器无法优化                               │
// │  → K_block 是变量，shared memory 偏移计算有分支                  │
// │                                                                  │
// │  编译期 (✅):                                                    │
// │  // GemmShape<128, 128, 32> → 编译器已知 K=32                   │
// │  // K 循环被完全展开为 4 次 K_block=8 的迭代                     │
// │  ld.shared r0, [smem + 0x000];                                  │
// │  ld.shared r1, [smem + 0x080];  // 偏移是编译器常量              │
// │  mma.sync r2, r0, r1;             // 编译器可以软件流水线化      │
// │  ld.shared r3, [smem + 0x100];                                  │
// │  ld.shared r4, [smem + 0x180];    // 加载和计算交替              │
// │  mma.sync r5, r3, r4;                                           │
// │  ...                        // 完全展开，无分支，无循环开销       │
// │  → 编译器可做: 寄存器分配优化、指令重排、软件流水线              │
// └──────────────────────────────────────────────────────────────────┘
//
// 类比: GemmShape 之于 GPU kernel，相当于编译期已知大小的
//       std::array<int, 128> 之于运行时可变大小的 std::vector<int>。
//       已知大小 → 栈分配 → 零堆开销 → 编译器可做边界检查消除。

template <int M_, int N_, int K_>
struct GemmShape {
  static constexpr int M = M_;
  static constexpr int N = N_;
  static constexpr int K = K_;

  // Tile 中的总元素数 (用于 shared memory 分配)
  static constexpr int kTileElements = M * N;

  // 验证: 所有维度必须是 2 的幂或常见 tile 大小
  // WHY: GPU warp 大小是 32，tile 必须是 warp 大小的倍数
  //      非对齐的 tile 会导致部分 warp 线程空闲
  static_assert(M % 32 == 0, "M must be multiple of warp size (32)");
  static_assert(N % 32 == 0, "N must be multiple of warp size (32)");

  // 常用 tile 大小列表 (NVIDIA 工程师经过大量实验得出的 sweet spot)
  static constexpr bool is_common_tile =
      (M == 64 || M == 128 || M == 256) &&
      (N == 64 || N == 128 || N == 256) &&
      (K == 8 || K == 16 || K == 32 || K == 64);
};

// ============================================================================
// WarpShape<M, N, K> - Warp 级别 Tiling
// ============================================================================
//
// WHY 需要 Warp Tiling: GPU 的线程层次是 Grid → Block → Warp → Thread。
//   Tile 级别的计算太大，需要进一步切分给各个 Warp。
//   每个 Warp 独立计算自己的子块，然后在 shared memory 中做 reduction。
//
// 类比: Tile 是"工厂的车间"，Warp 是"车间的工位"。
//       一个大订单 (tile) 分给多个车间 (warp)，每个车间独立生产自己的部分。
//
// 设计约束:
//   - WarpShape::M * WarpShape::N 不能超过 Warp 的寄存器容量 (255 per thread on SM80)
//   - WarpShape::K 影响 shared memory 的复用率

template <int M_, int N_, int K_>
struct WarpShape {
  static constexpr int M = M_;
  static constexpr int N = N_;
  static constexpr int K = K_;

  // Warp 大小 (NVIDIA GPU 固定 32 线程)
  static constexpr int kWarpSize = 32;

  // 每个 Warp 处理的元素数
  static constexpr int kElementsPerWarp = M * N;

  // 验证: WarpShape 必须整除 TileShape
  // 类比: 车间的工作量必须能均匀分给各工位
  template <typename TileShape>
  static constexpr bool is_valid_for(TileShape tile) {
    return (tile.M % M == 0) && (tile.N % N == 0) && (tile.K % K == 0);
  }
};

// ============================================================================
// InstructionShape<M, N, K> - MMA 指令形状
// ============================================================================
//
// WHY: Tensor Core 的 MMA 指令是固定形状的:
//   - SM70 (Volta):   m16n16k4  (FP16)
//   - SM75 (Turing):  m16n16k8  (FP16), m8n8k16 (INT8)
//   - SM80 (Ampere):  m16n8k16  (FP16), m16n8k32 (INT8), m16n8k8 (TF32)
//   - SM90 (Hopper):  m16n8k16  (FP8), 支持更大的 group MMA
//
// CUTLASS 将指令形状抽象为模板参数，这样:
//   1. kernel 代码对指令形状保持通用
//   2. 编译器可以根据具体指令做针对性优化
//   3. 不同架构可以通过特化获得最优实现
//
// 类比: InstructionShape 相当于汇编指令的操作数宽度。
//       x86 有 mov eax, [addr] (32bit) 和 mov rax, [addr] (64bit)。
//       你不能在 32bit CPU 上用 64bit mov。
//       同理，你不能在 SM70 上用 m16n8k16 的 MMA 指令。

template <int M_, int N_, int K_>
struct InstructionShape {
  static constexpr int M = M_;
  static constexpr int N = N_;
  static constexpr int K = K_;

  // 一次 MMA 指令的累积运算量 (FMA 数量)
  // 用于计算理论峰值性能
  static constexpr int kAccumulatorCount = M * N * K;

  // 判断是否为有效指令形状
  static constexpr bool is_valid_mma =
      (M == 16 && N == 16 && K == 4)  ||  // SM70 FP16
      (M == 16 && N == 16 && K == 8)  ||  // SM75 FP16
      (M == 8  && N == 8  && K == 16) ||  // SM75 INT8
      (M == 16 && N == 8  && K == 16) ||  // SM80 FP16/TF32
      (M == 16 && N == 8  && K == 32) ||  // SM80 INT8
      (M == 16 && N == 8  && K == 8);     // SM80 TF32
};

// ============================================================================
// 编译期 Tile 计算工具
// ============================================================================
//
// WHY: 这些工具函数在编译期计算 tile 迭代次数、warp 数量等。
//      因为所有维度都是编译期常量，这些计算也在编译期完成，
//      生成的 kernel 代码中直接使用计算结果常量。

template <typename TileShape, typename WarpShape>
struct WarpCount {
  static constexpr int kWarpsM = TileShape::M / WarpShape::M;
  static constexpr int kWarpsN = TileShape::N / WarpShape::N;
  static constexpr int kTotalWarps = kWarpsM * kWarpsN;

  static_assert(TileShape::M % WarpShape::M == 0,
                "Tile M must be divisible by Warp M");
  static_assert(TileShape::N % WarpShape::N == 0,
                "Tile N must be divisible by Warp N");
};

template <typename WarpShape, typename InstructionShape>
struct IterationsPerWarp {
  static constexpr int kIterationsM = WarpShape::M / InstructionShape::M;
  static constexpr int kIterationsN = WarpShape::N / InstructionShape::N;
  static constexpr int kIterationsK = WarpShape::K / InstructionShape::K;

  static constexpr int kTotalMmaInstructions =
      kIterationsM * kIterationsN * kIterationsK;
};

// ============================================================================
// 常用 Tile 配置的别名 (NVIDIA 推荐配置)
// ============================================================================

// SM80 最优配置 (A100)
using TileShape128x128x32  = GemmShape<128, 128, 32>;
using TileShape256x128x32  = GemmShape<256, 128, 32>;
using TileShape128x256x32  = GemmShape<128, 256, 32>;
using TileShape256x256x32  = GemmShape<256, 256, 32>;

// SM70/SM75 最优配置 (V100/T4)
using TileShape64x64x32    = GemmShape<64, 64, 32>;
using TileShape128x64x32   = GemmShape<128, 64, 32>;
using TileShape64x128x32   = GemmShape<64, 128, 32>;

// Warp 级别
using WarpShape64x64x32    = WarpShape<64, 64, 32>;
using WarpShape64x32x32    = WarpShape<64, 32, 32>;
using WarpShape32x64x32    = WarpShape<32, 64, 32>;
using WarpShape32x32x32    = WarpShape<32, 32, 32>;

// MMA 指令级别
using Mma16x16x8  = InstructionShape<16, 16, 8>;   // SM75 FP16
using Mma16x8x16  = InstructionShape<16, 8, 16>;   // SM80 FP16/TF32
using Mma16x8x8   = InstructionShape<16, 8, 8>;    // SM80 TF32
using Mma8x8x16   = InstructionShape<8, 8, 16>;    // SM75 INT8

} // namespace cutlass_style
