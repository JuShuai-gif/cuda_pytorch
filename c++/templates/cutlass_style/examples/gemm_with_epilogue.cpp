#include <iostream>
#include <cstdint>

// ============================================================================
// Mini CUTLASS 示例: 带 Epilogue 的 GEMM (ReLU + Bias + Residual)
// ============================================================================
//
// 这个示例展示 CUTLASS 最强大的特性之一: Epilogue 融合。
// 传统 GPU 编程中，每个操作 (GEMM, BiasAdd, ReLU, ResidualAdd)
// 都是独立的 kernel launch。数据必须在每个 kernel 之间读写 global memory。
// CUTLASS 的 Epilogue 将这些操作全部融合在 GEMM 的寄存器中完成。
//
// ┌──────────────────────────────────────────────────────────────────┐
// │  传统做法 (3 kernel launches)                                    │
// │                                                                  │
// │  Kernel 1: Gemm                                                  │
// │  ┌──────────────────────────────────────────────────┐            │
// │  │ C = A × B                                        │            │
// │  │ write C to global memory                         │            │
// │  └──────────────────────┬───────────────────────────┘            │
// │                         │ Global Memory: C_tmp                   │
// │                         ▼                                        │
// │  Kernel 2: BiasAdd + ReLU                                       │
// │  ┌──────────────────────────────────────────────────┐            │
// │  │ read C_tmp from global memory                    │            │
// │  │ C = C_tmp + bias                                 │            │
// │  │ C = max(0, C)                                    │            │
// │  │ write C to global memory                         │            │
// │  └──────────────────────┬───────────────────────────┘            │
// │                         │ Global Memory: C_activated             │
// │                         ▼                                        │
// │  Kernel 3: Residual Add                                         │
// │  ┌──────────────────────────────────────────────────┐            │
// │  │ read C_activated, residual from global memory     │            │
// │  │ output = C_activated + residual                  │            │
// │  │ write output to global memory                    │            │
// │  └──────────────────────────────────────────────────┘            │
// │                                                                  │
// │  总 Global Memory 流量: 5 reads + 3 writes = 8x 矩阵大小         │
// │                                                                  │
// │  ─────────────────────────────────────────────────────────────  │
// │                                                                  │
// │  CUTLASS Epilogue (1 kernel launch)                             │
// │                                                                  │
// │  ┌──────────────────────────────────────────────────┐            │
// │  │ GemmKernel::launch()                             │            │
// │  │                                                  │            │
// │  │   Mainloop:                                      │            │
// │  │   ┌──────────────────────────────────────┐       │            │
// │  │   │ Load A, B tile → shared memory        │       │            │
// │  │   │ MMA → accumulator (registers)         │       │            │
// │  │   └──────────────────────────────────────┘       │            │
// │  │                                                  │            │
// │  │   Epilogue (in registers, fused):               │            │
// │  │   ┌──────────────────────────────────────┐       │            │
// │  │   │ 1. acc = acc + bias[col]              │ ← 1 read    │    │
// │  │   │ 2. acc = max(0, acc)                  │ ← 0 mem ops │    │
// │  │   │ 3. acc = acc + residual[row][col]     │ ← 1 read    │    │
// │  │   │ 4. cvt.f16.f32 acc → output           │ ← 0 mem ops │    │
// │  │   │ 5. st.global C[row][col] = output     │ ← 1 write   │    │
// │  │   └──────────────────────────────────────┘       │            │
// │  └──────────────────────────────────────────────────┘            │
// │                                                                  │
// │  总 Global Memory 流量: 2 reads + 1 write = 3x 矩阵大小          │
// │  节省: 62.5% 的 global memory 带宽                               │
// └──────────────────────────────────────────────────────────────────┘

#include "include/arch_tag.hpp"
#include "include/layout.hpp"
#include "include/tile_shape.hpp"
#include "include/operator_class.hpp"
#include "include/tensor_ref.hpp"
#include "include/kernel_traits.hpp"
#include "include/type_list.hpp"
#include "include/mma_policy.hpp"
#include "dispatch/default_gemm_config.hpp"
#include "dispatch/kernel_dispatch.hpp"
#include "kernel/gemm_kernel.hpp"
#include "kernel/mainloop.hpp"
#include "kernel/epilogue.hpp"

using namespace cutlass_style;

// ============================================================================
// 展示: 编译期 Epilogue 操作组合
// ============================================================================

template <typename Traits>
void demonstrate_epilogue_fusion() {
  std::cout << "\n=== Epilogue Fusion Demo ===" << std::endl;

  // ── 场景 1: Identity (无 Epilogue) ──
  using EP1 = kernel::EpilogueFactory<kernel::EpilogueType::kIdentity, Traits>::type;
  std::cout << "  Epilogue 1: Identity" << std::endl;
  std::cout << "    Data flow: Accumulator → Type Convert → Global Memory" << std::endl;
  std::cout << "    Kernel launches: 1 (GEMM only)" << std::endl;

  // ── 场景 2: BiasAdd + ReLU (Transformers 中最常见的组合) ──
  using EP2 = kernel::EpilogueFactory<kernel::EpilogueType::kBiasAddReLU, Traits>::type;
  std::cout << "\n  Epilogue 2: BiasAdd + ReLU" << std::endl;
  std::cout << "    Data flow: Accumulator → +Bias → ReLU → Convert → Global Memory" << std::endl;
  std::cout << "    Kernel launches: 1 (vs 3 without fusion)" << std::endl;
  std::cout << "    Bandwidth saved: ~60% (2 extra global reads/writes eliminated)" << std::endl;

  // ── 场景 3: GELU (LLM activation) ──
  using EP3 = kernel::EpilogueFactory<kernel::EpilogueType::kGelu, Traits>::type;
  std::cout << "\n  Epilogue 3: GELU (GPT/BERT activation)" << std::endl;
  std::cout << "    Data flow: Accumulator → GELU(x) → Convert → Global Memory" << std::endl;
  std::cout << "    Kernel launches: 1" << std::endl;

  // ── 自定义组合: BiasAdd + GELU + ResidualAdd ──
  // 这是 GPT/LLaMA 等 Transformer decoder layers 的典型计算模式
  std::cout << "\n  Epilogue 4: Custom (BiasAdd + GELU + ResidualAdd)" << std::endl;
  std::cout << "    This is the typical FFN block in GPT-style models:" << std::endl;
  std::cout << "    output = (W2 * GELU(W1 * x + b1)) + b2 + x" << std::endl;
  std::cout << "    3 GEMM operations, all with fused epilogues." << std::endl;
  std::cout << "    Without fusion: 9 kernel launches" << std::endl;
  std::cout << "    With fusion:     3 kernel launches (saves 67%)" << std::endl;
}

// ============================================================================
// 展示: 不同 Epilogue 类型的编译期选择
// ============================================================================

void explain_epilogue_dispatch() {
  std::cout << "\n=== Epilogue Dispatch Mechanism ===" << std::endl;
  std::cout << std::endl;

  // WHY 编译期 Epilogue 选择:
  std::cout << "  Why compile-time epilogue selection?\n" << std::endl;

  std::cout << "  Runtime approach (❌):" << std::endl;
  std::cout << "    if (use_relu) { c = max(c, 0); }         // branch in kernel" << std::endl;
  std::cout << "    if (use_bias) { c += bias[col]; }         // another branch" << std::endl;
  std::cout << "    if (use_residual) { c += residual; }       // yet another" << std::endl;
  std::cout << "    → 3 runtime branches per output element" << std::endl;
  std::cout << "    → Warp divergence if threads take different paths" << std::endl;
  std::cout << "    → Cannot optimize register allocation (compiler must reserve for all paths)" << std::endl;
  std::cout << std::endl;

  std::cout << "  Compile-time approach (✅ CUTLASS):" << std::endl;
  std::cout << "    using Epilogue = EpilogueFactory<kBiasAddReLU, Traits>::type;" << std::endl;
  std::cout << "    → Compiler knows exactly which operations to fuse" << std::endl;
  std::cout << "    → Zero runtime branches in epilogue" << std::endl;
  std::cout << "    → Optimal register allocation (only allocate what's needed)" << std::endl;
  std::cout << "    → Pipeline can be fully unrolled" << std::endl;
  std::cout << std::endl;

  // 对比表格
  std::cout << "  ┌─────────────────────┬──────────────────┬──────────────────┐" << std::endl;
  std::cout << "  │ Metric              │ Without Fusion   │ With Fusion      │" << std::endl;
  std::cout << "  ├─────────────────────┼──────────────────┼──────────────────┤" << std::endl;
  std::cout << "  │ Kernel launches     │ 5                │ 1                │" << std::endl;
  std::cout << "  │ Global mem reads    │ 5x matrix size   │ 2x               │" << std::endl;
  std::cout << "  │ Global mem writes   │ 5x matrix size   │ 1x               │" << std::endl;
  std::cout << "  │ Register pressure   │ Low (per kernel) │ Higher (fused)   │" << std::endl;
  std::cout << "  │ L1 cache hits       │ ~0%              │ 100% (registers) │" << std::endl;
  std::cout << "  │ Launch latency      │ 5 × ~3μs         │ ~3μs             │" << std::endl;
  std::cout << "  └─────────────────────┴──────────────────┴──────────────────┘" << std::endl;
}

int main() {
  std::cout << "================================================================" << std::endl;
  std::cout << "  Mini CUTLASS: GEMM with Epilogue Fusion Demo" << std::endl;
  std::cout << "================================================================\n" << std::endl;

  // =========================================================================
  // 演示配置 (SM80, FP16)
  // =========================================================================
  using ArchTag = Sm80;
  using ElementA = float;  // half
  using ElementB = float;  // half
  using ElementC = float;  // half

  using Config = dispatch::DefaultGemmConfiguration<ArchTag, ElementA, ElementB, ElementC>;

  using KernelTraits = GemmKernelTraits<
      ElementA, ElementB, ElementC,
      RowMajor, ColumnMajor, RowMajor,
      ArchTag,
      TensorOp,
      typename Config::TileShape,
      typename Config::WarpShape,
      typename Config::MmaInstruction
  >;

  std::cout << "Configuration:" << std::endl;
  std::cout << "  Architecture: " << ArchTag::name << std::endl;
  std::cout << "  Data type:    FP16 (half precision)" << std::endl;
  std::cout << "  Tile:         " << Config::TileShape::M
            << "x" << Config::TileShape::N
            << "x" << Config::TileShape::K << std::endl;

  // =========================================================================
  // Epilogue 融合演示
  // =========================================================================
  demonstrate_epilogue_fusion<KernelTraits>();

  // =========================================================================
  // Epilogue dispatch 机制解释
  // =========================================================================
  explain_epilogue_dispatch();

  // =========================================================================
  // 实际 LLM 推理中的 Epilogue 使用
  // =========================================================================
  std::cout << "\n=== Real LLM Inference Use Cases ===" << std::endl;
  std::cout << std::endl;

  std::cout << "  Transformer FFN Block (fused):" << std::endl;
  std::cout << "  ┌─────────────────────────────────────────────────┐" << std::endl;
  std::cout << "  │ GEMM 1: C1 = W1 × x                            │" << std::endl;
  std::cout << "  │   Epilogue: BiasAdd(b1) + GELU                 │" << std::endl;
  std::cout << "  │ GEMM 2: C2 = W2 × C1                           │" << std::endl;
  std::cout << "  │   Epilogue: BiasAdd(b2) + ResidualAdd(x)        │" << std::endl;
  std::cout << "  │ LayerNorm(C2)                                   │" << std::endl;
  std::cout << "  └─────────────────────────────────────────────────┘" << std::endl;
  std::cout << "  2 GEMM kernels (vs 5-7 without fusion)" << std::endl;
  std::cout << std::endl;

  std::cout << "  Attention Projections (fused):" << std::endl;
  std::cout << "  ┌─────────────────────────────────────────────────┐" << std::endl;
  std::cout << "  │ Q = W_q × x    → Epilogue: BiasAdd(b_q)        │" << std::endl;
  std::cout << "  │ K = W_k × x    → Epilogue: BiasAdd(b_k)        │" << std::endl;
  std::cout << "  │ V = W_v × x    → Epilogue: BiasAdd(b_v)        │" << std::endl;
  std::cout << "  │ C = Attn(Q,K,V) × W_o  → Epilogue: BiasAdd(b_o) │" << std::endl;
  std::cout << "  └─────────────────────────────────────────────────┘" << std::endl;
  std::cout << "  4 GEMM kernels (vs 8 without fusion)" << std::endl;

  std::cout << "\n================================================================" << std::endl;
  std::cout << "  Epilogue fusion: the 'secret sauce' of CUTLASS performance." << std::endl;
  std::cout << "  All operations fuse at compile-time → zero runtime overhead." << std::endl;
  std::cout << "================================================================\n" << std::endl;

  return 0;
}
