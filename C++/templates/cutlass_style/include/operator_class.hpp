#pragma once

#include <cstdint>
#include <type_traits>

namespace cutlass_style {

// ============================================================================
// OperatorClass - 计算单元类型
// ============================================================================
//
// WHY OperatorClass: GPU 有不同等级的计算单元，性能和精度差异巨大。
//
// ┌──────────────────────────────────────────────────────────────┐
// │  Mermaid 图: 计算单元层次                                   │
// │                                                              │
// │  Simt (CUDA Core)                                            │
// │  ├── FMA: 单独乘法+加法，每周期 1 条指令                      │
// │  ├── 精度: FP64, FP32, FP16                                  │
// │  └── 吞吐: ~19.5 TFLOPS (A100 FP32)                          │
// │                                                              │
// │  TensorOp (Tensor Core)                                      │
// │  ├── MMA: 矩阵乘累加，每周期 m*n*k 次运算                     │
// │  ├── 精度: FP16/BF16/TF32/INT8/FP8 (取决于架构)              │
// │  └── 吞吐: ~312 TFLOPS (A100 BF16)                            │
// │                                                              │
// │  Wmma (Warp Matrix Multiply-Accumulate)                      │
// │  ├── 高层 Tensor Core API，可移植但性能不如直接 MMA           │
// │  └── 适合快速原型和跨架构代码                                  │
// └──────────────────────────────────────────────────────────────┘
//
// 类比: OperatorClass 就像选择"用什么引擎"：
//   Simt     → 普通燃油引擎 (灵活、通用，但效率一般)
//   TensorOp → 火箭引擎 (专用、极快，但需要特定燃料)
//   Wmma     → 混合动力 (兼顾灵活和效率)

struct Simt {
  // SIMT: Single Instruction Multiple Thread
  // 即标准的 CUDA Core 计算模式
  static constexpr const char* name = "SIMT (CUDA Core)";

  // 一次操作的元素数 (标量操作)
  static constexpr int kElementsPerAccess = 1;

  // SIMT 不需要特殊数据布局，支持所有数据类型
  static constexpr bool requires_row_major = false;
  static constexpr bool requires_column_major = false;
};

struct TensorOp {
  // Tensor Core 的 MMA (Matrix Multiply-Accumulate)
  static constexpr const char* name = "TensorOp (Tensor Core)";

  // Tensor Core 一次处理矩阵块
  static constexpr int kElementsPerAccess = 256; // 16x16 block

  // Tensor Core 要求矩阵在 shared memory 中有特定布局
  // (interleaved layout，避免 bank conflict)
  static constexpr bool requires_shared_memory_layout = true;
};

struct Wmma {
  // Warp Matrix Multiply-Accumulate (wmma API)
  static constexpr const char* name = "WMMA";

  // WMMA 作为中间层抽象
  static constexpr int kElementsPerAccess = 256;

  // WMMA 有自己特定的 fragment 类型
  static constexpr bool uses_wmma_fragment = true;
};

// ============================================================================
// Operator<OpClass, ElementA, ElementB, ElementC>
// ============================================================================
//
// WHY Operator 而非单独的类型组合:
//   矩阵乘法涉及 3 种元素类型: A, B, C (accumulator)
//   它们的组合决定了:
//     1. 使用哪种 Tensor Core 指令 (mma.sync vs mma.sync.aligned)
//     2. 累加器的精度 (FP16 acc vs FP32 acc)
//     3. 是否需要类型转换 (INT8 A/B → INT32 C)
//
// 类比: Operator 相当于 BLAS 库的 gemm 函数签名:
//   sgemm → float A, float B, float C (SIMT)
//   hgemm → half A, half B, half/float C (TensorOp)
//   igemm → int8 A, int8 B, int32 C (INT8 TensorOp)

template <typename OpClass_, typename ElementA_, typename ElementB_, typename ElementC_>
struct Operator {
  using OpClass = OpClass_;
  using ElementA = ElementA_;
  using ElementB = ElementB_;
  using ElementC = ElementC_;

  // 编译期验证: TensorOp 要求 half/bf16 等特定类型
  static constexpr bool is_valid = []() constexpr {
    if constexpr (std::is_same_v<OpClass, Simt>) {
      return true; // SIMT 支持所有类型
    } else if constexpr (std::is_same_v<OpClass, TensorOp>) {
      // Tensor Core 只支持特定精度
      return (sizeof(ElementA) <= 2 && sizeof(ElementB) <= 2) || // FP16/BF16/INT8
             (std::is_same_v<ElementA, float> &&
              std::is_same_v<ElementB, float>); // TF32 用 float API
    }
    return false;
  }();

  static_assert(is_valid, "Invalid OperatorClass/ElementType combination");

  // 累加器类型: 通常比输入精度高
  // INT8 输入 → INT32 累加器 (防止溢出)
  // FP16 输入 → FP32 累加器 (提高精度)
  using AccumulatorType = std::conditional_t<
      (sizeof(ElementA) == 1 || sizeof(ElementB) == 1),  // INT8/INT4
      int32_t,                                              // → int32
      float                                                 // → float
      >;
};

// ============================================================================
// 常用 Operator 别名
// ============================================================================

// FP16 Tensor Core (SM70+)
using HgemmTensorOp = Operator<TensorOp, float, float, float>;
// 注: 硬件上 A 和 B 是 half，但 CUDA API 用 half 类型

// INT8 Tensor Core (SM75+)
using IgemmTensorOp = Operator<TensorOp, int8_t, int8_t, int32_t>;

// BF16 Tensor Core (SM80+)
using BfgemmTensorOp = Operator<TensorOp, float, float, float>;
// 注: BF16 用 __nv_bfloat16 类型

// FP32 SIMT (所有架构)
using SgemmSimt = Operator<Simt, float, float, float>;

// TF32 Tensor Core (SM80+)
using Tf32GemmTensorOp = Operator<TensorOp, float, float, float>;
// TF32: 输入是 float，Tensor Core 内部截断为 19bit 尾数

// ============================================================================
// Operator Traits - 编译期特征提取
// ============================================================================

template <typename Operator>
struct OperatorTraits {
  using OpClass = typename Operator::OpClass;
  using ElementA = typename Operator::ElementA;
  using ElementB = typename Operator::ElementB;
  using ElementC = typename Operator::ElementC;
  using AccumulatorType = typename Operator::AccumulatorType;

  static constexpr bool is_tensor_op = std::is_same_v<OpClass, TensorOp>;
  static constexpr bool is_simt = std::is_same_v<OpClass, Simt>;
  static constexpr bool is_wmma = std::is_same_v<OpClass, Wmma>;

  // 每个线程的寄存器需求 (粗略估计)
  // 用于编译期判断是否会 register spill
  static constexpr int estimated_registers_per_thread =
      is_simt ? 32 : 128; // Tensor Core 需要更多寄存器存 fragment
};

} // namespace cutlass_style
