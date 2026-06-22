#include <iostream>
#include <cstdint>
#include <string>

#include "include/engine_config.hpp"
#include "include/kernel_registry.hpp"

// ============================================================================
// GEMM Dispatch 完整实现
// ============================================================================
//
// 这个文件展示了 CUTLASS 风格的多级 dispatch 系统。
// 借鉴 CUTLASS 的真实代码结构，但做了大幅简化。
//
// ┌──────────────────────────────────────────────────────────────────┐
// │  编译期 GEMM Dispatch 的分层架构                                 │
// │                                                                  │
// │  Layer 1: 架构分派 (Architecture Dispatch)                      │
// │  ┌──────────────────────────────────────────────────────────┐   │
// │  │ GPU_ARCH ≥ 90 → Sm90GemmDispatch                         │   │
// │  │ GPU_ARCH ≥ 80 → Sm80GemmDispatch                         │   │
// │  │ GPU_ARCH ≥ 75 → Sm75GemmDispatch                         │   │
// │  │ GPU_ARCH ≥ 70 → Sm70GemmDispatch                         │   │
// │  └──────────────────────────────────────────────────────────┘   │
// │                                                                  │
// │  Layer 2: 数据类型分派 (Data Type Dispatch)                     │
// │  ┌──────────────────────────────────────────────────────────┐   │
// │  │ is_fp16 → 使用 Tensor Core fp16 MMA                     │   │
// │  │ is_bf16 → 使用 Tensor Core bf16 MMA (需要 SM80+)         │   │
// │  │ is_int8 → 使用 Tensor Core int8 MMA (需要 SM75+)         │   │
// │  │ is_fp32 → 回退到 SIMT FMA                               │   │
// │  └──────────────────────────────────────────────────────────┘   │
// │                                                                  │
// │  Layer 3: Layout 分派 (Layout Dispatch)                         │
// │  ┌──────────────────────────────────────────────────────────┐   │
// │  │ RowMajor × ColumnMajor  → NN GEMM (标准)                 │   │
// │  │ RowMajor × RowMajor     → NT GEMM (需要转置 B)           │   │
// │  │ ColumnMajor × RowMajor  → TN GEMM (需要转置 A)           │   │
// │  │ ColumnMajor × ColumnMajor → TT GEMM (双转置)             │   │
// │  └──────────────────────────────────────────────────────────┘   │
// │                                                                  │
// │  Layer 4: Tile Policy (性能参数)                                 │
// │  ┌──────────────────────────────────────────────────────────┐   │
// │  │ 根据: 数据类型、架构、K 维度大小                          │   │
// │  │ 输出: 最优 TileShape, WarpShape, InstructionShape         │   │
// │  └──────────────────────────────────────────────────────────┘   │
// │                                                                  │
// │  Layer 5: Kernel Selection (最终 kernel 实例化)                  │
// │  ┌──────────────────────────────────────────────────────────┐   │
// │  │ 综合以上所有信息 → 选择最合适的Kernel                     │   │
// │  │ Kernel::launch(...)                                       │   │
// │  └──────────────────────────────────────────────────────────┘   │
// └──────────────────────────────────────────────────────────────────┘

using namespace mini_inference;

// ============================================================================
// Layer 1: 架构分派
// ============================================================================
//
// WHY if constexpr 而非虚函数:
//   CUDA kernel 函数模板不能是虚函数 (device code 限制)。
//   if constexpr 在编译期消除未使用的分支，比虚函数更快，且不增加 binary size。
//
// 模板展开后:
//   用户调用: dispatch_gemm<Sm80, FP16>(...)
//   → ArchDispatcher<Sm80>::dispatch(...)
//   → if constexpr (Arch >= 90) → false，跳过
//   → if constexpr (Arch >= 80) → true → Sm80GemmDispatch
//   → 其他所有分支的代码不会被编译

template <int Arch>
struct ArchDispatcher {
  template <typename Config>
  static void dispatch(
      const typename Config::RuntimeConfig& rcfg,
      void* A, void* B, void* C,
      int M, int N, int K,
      bool transpose_a, bool transpose_b) {

    std::cout << "  [ArchDispatcher] Arch=" << Arch
              << " (" << Config::arch_name() << ")" << std::endl;

    if constexpr (Arch >= 90) {
      // SM90 (Hopper): 支持 FP8, TMA, wgmma
      std::cout << "    → Dispatching to SM90 kernel path" << std::endl;
      std::cout << "    → Features available: FP8, TMA, wgmma, DSmem" << std::endl;
      dispatch_sm90<Config>(rcfg, A, B, C, M, N, K, transpose_a, transpose_b);
    } else if constexpr (Arch >= 80) {
      // SM80 (Ampere): 支持 BF16, TF32, cp.async
      std::cout << "    → Dispatching to SM80 kernel path" << std::endl;
      std::cout << "    → Features available: BF16, TF32, cp.async" << std::endl;
      dispatch_sm80<Config>(rcfg, A, B, C, M, N, K, transpose_a, transpose_b);
    } else if constexpr (Arch >= 75) {
      // SM75 (Turing): 支持 INT8 Tensor Core
      std::cout << "    → Dispatching to SM75 kernel path" << std::endl;
      std::cout << "    → Features available: INT8, FP16 Tensor Core" << std::endl;
      dispatch_sm75<Config>(rcfg, A, B, C, M, N, K, transpose_a, transpose_b);
    } else {
      // SM70 (Volta): 基础 FP16 Tensor Core
      std::cout << "    → Dispatching to SM70 kernel path" << std::endl;
      std::cout << "    → Features available: FP16 Tensor Core (base)" << std::endl;
      dispatch_sm70<Config>(rcfg, A, B, C, M, N, K, transpose_a, transpose_b);
    }
  }

 private:
  // =========================================================================
  // SM90: Hopper (H100/H200) 专用 dispatch
  // =========================================================================
  template <typename Config>
  static void dispatch_sm90(
      const typename Config::RuntimeConfig& rcfg,
      void* A, void* B, void* C,
      int M, int N, int K,
      bool transpose_a, bool transpose_b) {

    // SM90 最优 tile 配置 (更大的 K-tile 匹配 TMA 能力)
    constexpr int kSm90M = 256, kSm90N = 256, kSm90K = 64;

    std::cout << "      Tile: " << kSm90M << "x"
              << kSm90N << "x" << kSm90K << std::endl;

    // 如果使用 FP8，用专用路径
    if constexpr (Config::DType == DataType::kFloat16) {
      std::cout << "      Using: wgmma.mma_async (Warp Group MMA)" << std::endl;
      // wgmma 是 SM90 的新指令，整个 warp group (128 threads) 协同做 MMA
      // 比普通 mma.sync (32 threads) 效率更高
    }

    // 伪代码: 实际 kernel launch
    // if (transpose_a && !transpose_b) {
    //   using Kernel = Sm90GemmKernel<TN, Sm90Tile, Config>;
    //   Kernel::launch(rcfg, A, B, C, M, N, K);
    // }
  }

  // =========================================================================
  // SM80: Ampere (A100/A6000) 专用 dispatch
  // =========================================================================
  template <typename Config>
  static void dispatch_sm80(
      const typename Config::RuntimeConfig& rcfg,
      void* A, void* B, void* C,
      int M, int N, int K,
      bool transpose_a, bool transpose_b) {

    // SM80 tile 选择策略:
    //   - 小 K (≤ 256): 使用 TileK=32，更多 K 迭代但更高 SM 占用率
    //   - 大 K (> 256): 使用 TileK=64，减少迭代次数
    constexpr int kSmallM = 128, kSmallN = 128, kSmallK = 32;
    constexpr int kLargeM = 256, kLargeN = 128, kLargeK = 32;

    // 运行时 K 值决定 tile 大小
    bool use_large_tile = (K > 256);
    std::cout << "      K=" << K << " → "
              << (use_large_tile ? "Large tile (256x128)" : "Standard tile (128x128)")
              << std::endl;

    std::cout << "      Using: mma.sync.aligned.m16n8k16 (cp.async for data loading)"
              << std::endl;

    // 伪代码: Kernel launch
    // if (use_large_tile) {
    //   select_and_launch<Sm80TileLarge, Config>(...);
    // } else {
    //   select_and_launch<Sm80TileSmall, Config>(...);
    // }
  }

  // =========================================================================
  // SM75: Turing (T4/RTX 2080) 专用 dispatch
  // =========================================================================
  template <typename Config>
  static void dispatch_sm75(
      const typename Config::RuntimeConfig& rcfg,
      void* A, void* B, void* C,
      int M, int N, int K,
      bool transpose_a, bool transpose_b) {

    // SM75 tile: 较小的 shared memory (64KB) 限制 tile 大小
    constexpr int kSm75M = 128, kSm75N = 128, kSm75K = 16;

    std::cout << "      Tile: " << kSm75M << "x"
              << kSm75N << "x" << kSm75K << std::endl;

    std::cout << "      Using: mma.sync.aligned.m16n16k8" << std::endl;

    // INT8: SM75 支持 INT8 Tensor Core (mma.sync.aligned.m8n8k16)
    if constexpr (Config::DType == DataType::kInt8) {
      std::cout << "      Using: mma.sync.aligned.m8n8k16 (INT8 Tensor Core)" << std::endl;
    }
  }

  // =========================================================================
  // SM70: Volta (V100) 专用 dispatch
  // =========================================================================
  template <typename Config>
  static void dispatch_sm70(
      const typename Config::RuntimeConfig& rcfg,
      void* A, void* B, void* C,
      int M, int N, int K,
      bool transpose_a, bool transpose_b) {

    // V100 tile: K 维度更小 (第一条 Tensor Core 限制较多)
    constexpr int kSm70M = 128, kSm70N = 64, kSm70K = 8;

    std::cout << "      Tile: " << kSm70M << "x"
              << kSm70N << "x" << kSm70K << std::endl;

    std::cout << "      Using: mma.sync.aligned.m16n16k4 (first-gen Tensor Core)"
              << std::endl;
  }
};

// ============================================================================
// Layer 2+3: 数据类型 + Layout 分派
// ============================================================================

template <typename Config>
void dispatch_data_type_and_layout(
    const typename Config::RuntimeConfig& rcfg,
    void* A, void* B, void* C,
    int M, int N, int K,
    bool transpose_a, bool transpose_b) {

  std::cout << "  [DtypeDispatch] DataType=" << Config::dtype_name() << std::endl;

  // Layout 组合
  const char* layout_name;
  if (!transpose_a && !transpose_b) {
    layout_name = "NN (A row-major, B col-major) - standard GEMM";
  } else if (!transpose_a && transpose_b) {
    layout_name = "NT (A row-major, B row-major) - B needs transpose";
  } else if (transpose_a && !transpose_b) {
    layout_name = "TN (A col-major, B col-major) - A needs transpose";
  } else {
    layout_name = "TT (A col-major, B row-major) - both need transpose";
  }
  std::cout << "  [LayoutDispatch] " << layout_name << std::endl;

  // 基于数据类型的 kernel 路径选择
  if constexpr (Config::DType == DataType::kFloat16) {
    std::cout << "    → FP16 path: Using Tensor Core MMA (mma.sync)" << std::endl;
    // 如果是 NN layout: 可以直接用 A 的 RowMajor + B 的 ColumnMajor
    // 如果是 NT layout: 需要用 B 的 RowMajor (相当于 B^T 的 ColumnMajor)
  } else if constexpr (Config::DType == DataType::kBFloat16) {
    if constexpr (Config::kHasBf16TensorCore) {
      std::cout << "    → BF16 path: Using Tensor Core with .bf16 suffix" << std::endl;
    } else {
      std::cout << "    → BF16 not supported on this arch, fallback to FP32" << std::endl;
    }
  } else if constexpr (Config::DType == DataType::kInt8) {
    if constexpr (Config::kHasInt8TensorCore) {
      std::cout << "    → INT8 path: Using IMMA (integer MMA)" << std::endl;
      std::cout << "    → Accumulator: INT32, output needs dequant" << std::endl;
    } else {
      std::cout << "    → INT8 not supported on this arch" << std::endl;
    }
  } else {
    std::cout << "    → FP32 path: Using SIMT FMA (no Tensor Core)" << std::endl;
  }

  // 最终: 调用架构分派
  ArchDispatcher<Config::Arch>::template dispatch<Config>(
      rcfg, A, B, C, M, N, K, transpose_a, transpose_b);
}

// ============================================================================
// 公开 API: gemm_dispatch
// ============================================================================
//
// 这是推理引擎对外暴露的 GEMM 接口。
// 调用方式类似 cuBLAS 的 cublasGemmEx:
//
//   gemm_dispatch<ConfigA100Fp16>(rcfg, A, B, C, M, N, K, false, false);
//
// WHY 模板参数 Config:
//   Config 聚合了所有编译期信息 (arch, dtype, layout, tile candidates)。
//   整个 dispatch 链都在编译期展开，运行时只有必要的 switch/case。

template <typename Config>
void gemm_dispatch(
    const typename Config::RuntimeConfig& rcfg,
    void* A, void* B, void* C,
    int M, int N, int K,
    bool transpose_a = false,
    bool transpose_b = false) {

  std::cout << "\n========================================" << std::endl;
  std::cout << "  GEMM Dispatch" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "  Config: Arch=" << Config::arch_name()
            << ", Dtype=" << Config::dtype_name()
            << ", TensorCore=" << (Config::kUseTensorCore ? "YES" : "NO")
            << std::endl;
  std::cout << "  Runtime: " << rcfg.to_string() << std::endl;
  std::cout << "  GEMM: M=" << M << ", N=" << N << ", K=" << K
            << " (trans_a=" << transpose_a
            << ", trans_b=" << transpose_b << ")" << std::endl;
  std::cout << "  Peak performance: " << Config::arch_name() << std::endl;
  std::cout << std::endl;

  // ── 编译期验证 ──
  static_assert(Config::Arch >= 70,
                "GPU architecture must be >= SM70 (Volta)");
  static_assert(Config::DType != DataType::kFloat32 || true,
                "FP32 is supported but slower (no Tensor Core)");

  // ── 多级 Dispatch ──
  dispatch_data_type_and_layout<Config>(
      rcfg, A, B, C, M, N, K, transpose_a, transpose_b);

  std::cout << "  → GEMM dispatch complete\n" << std::endl;
}

// ============================================================================
// Tile Policy 系统 (编译期)
// ============================================================================
//
// WHY Tile Policy:
//   Tile 大小直接影响 shared memory 使用、寄存器压力、occupancy。
//   不同 (arch, dtype, M/N/K) 组合有不同最优 tile。
//   Tile Policy 将这些经验知识编码为编译期决策。

template <int Arch, DataType DType, int M, int N, int K>
struct TilePolicy {
  // 根据矩阵形状选择 tile 策略
  static constexpr bool is_tall_skinny  = (M > 8 * N);   // 高瘦矩阵
  static constexpr bool is_short_fat   = (N > 8 * M);   // 矮胖矩阵
  static constexpr bool is_small_k      = (K <= 256);    // 小 K

  // 默认 tile
  struct DefaultTile { static constexpr int kM = 128, kN = 128, kK = 32; };

  // 高瘦矩阵优化: 增加 M-tile，减少 N-tile
  struct TallSkinnyTile { static constexpr int kM = 256, kN = 64, kK = 32; };

  // 矮胖矩阵优化: 减少 M-tile，增加 N-tile
  struct ShortFatTile { static constexpr int kM = 64, kN = 256, kK = 32; };

  // 小 K 优化: 减少 K-tile 以减少 shared memory，提高 occupancy
  struct SmallKTile { static constexpr int kM = 128, kN = 128, kK = 16; };

  using SelectedTile = std::conditional_t<
      is_tall_skinny, TallSkinnyTile,
      std::conditional_t<
          is_short_fat, ShortFatTile,
          std::conditional_t<
              is_small_k, SmallKTile,
              DefaultTile
          >
      >
  >;

  static constexpr const char* strategy_name =
      is_tall_skinny ? "Tall-Skinny" :
      is_short_fat ? "Short-Fat" :
      is_small_k ? "Small-K" : "Default";
};
