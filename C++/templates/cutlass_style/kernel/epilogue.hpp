#pragma once

#include <cstdint>
#include <type_traits>

#include "include/kernel_traits.hpp"

// CUDA compatibility macros for host-only compilation
#ifndef __CUDACC__
  #define __device__
  #define __global__
  #define __shared__
  #define __syncthreads()
#endif

namespace cutlass_style {
namespace kernel {

// ============================================================================
// Epilogue<Traits, Op> - GEMM 尾声操作
// ============================================================================
//
// ┌──────────────────────────────────────────────────────────────────┐
// │  Mermaid 图: Epilogue 的数据流                                   │
// │                                                                  │
// │  Accumulator (registers)                                         │
// │  ┌──────────────────┐                                            │
// │  │ FP32 accumulator │  ← Tensor Core 的输出 (float32)           │
// │  │ frag[128]        │                                            │
// │  └────────┬─────────┘                                            │
// │           │                                                      │
// │           ▼                                                      │
// │  ┌──────────────────┐                                            │
// │  │ Type Conversion   │  ← FP32 → FP16/BF16/INT8 (architecture opt)│
// │  │ cvt.rn.f16.f32   │                                            │
// │  └────────┬─────────┘                                            │
// │           │                                                      │
// │           ▼                                                      │
// │  ┌──────────────────┐     Optional Operations:                   │
// │  │ Epilogue Pipeline│────────────────────────────────┐          │
// │  │                  │                                │          │
// │  │  1. BiasAdd      │ ← + bias[col]                  │          │
// │  │  2. ReLU         │ ← max(0, x)                    │          │
// │  │  3. GELU         │ ← gelu(x) (LLM 中常用)         │          │
// │  │  4. ResidualAdd  │ ← + residual[row][col]         │          │
// │  │  5. LayerNorm    │ ← (x - μ)/σ * γ + β            │          │
// │  │  6. Dropout      │ ← mask * x / (1-p)             │          │
// │  └────────┬─────────┘                                │          │
// │           │                                          │          │
// │           ▼                                          │          │
// │  ┌──────────────────┐                                │          │
// │  │ Write to Global   │ ← st.global (向量化存储)       │          │
// │  └──────────────────┘                                │          │
// └──────────────────────────────────────────────────────────────────┘
//
// WHY Epilogue 独立于 Mainloop:
//   1. 关注点分离: Mainloop 负责计算，Epilogue 负责输出处理
//   2. 可组合性: 用户可以自由组合 epilogue 操作 (ReLU + Bias + Residual)
//   3. 编译期融合: 多个 epilogue 操作在编译期融合为一个 kernel，
//      避免了多次 global memory 读写 (kernel fusion 的核心价值)
//   4. 寄存器复用: Mainloop 的累加器寄存器直接用于 epilogue，
//      不需要写回 shared memory 再读取
//
// 类比: Mainloop 是"炒菜"，Epilogue 是"装盘+调味"。
//       你不会把菜先装盘再回锅调味——而是直接在锅里完成。

// ============================================================================
// 基础 Epilogue 操作: 直接输出 (Identity)
// ============================================================================
//
// 最简单的 epilogue: 累加器 → 类型转换 → 写出
// 模板展开后: 生成 PTX 的 cvt + st.global 指令

template <typename Traits_>
struct IdentityEpilogue {
  using Traits = Traits_;
  using ElementC = typename Traits::ElementC;
  using AccumulatorType = typename Traits::AccumulatorType;

  // 输出操作 (无额外处理)
  __device__ static void apply(
      AccumulatorType& acc_value,
      ElementC* d_C,
      int global_row, int global_col,
      int ldc) {

    // 类型转换 (如果需要)
    ElementC output = static_cast<ElementC>(acc_value);

    // 写入 global memory
    // d_C[global_row * ldc + global_col] = output;
    //
    // 实际用向量化存储: st.global.v4.f16 [%ptr], {%r0,%r1,%r2,%r3}
  }
};

// ============================================================================
// ReLU Epilogue 操作
// ============================================================================
//
// WHY 在 Epilogue 中做 ReLU:
//   传统做法: GEMM → 写到 global → 读回 → ReLU → 写回 global
//   每次 global memory 访问: ~500 cycles
//   CUTLASS Epilogue: GEMM → ReLU (在寄存器中) → 一次性写出
//   节省: 2 次 global memory 读写 ≈ 节省 30% 延迟
//
// 模板展开后: max.f32 %r_dst, %r_src, 0.0;  ← 单条 PTX 指令

template <typename Traits_>
struct ReluEpilogue {
  using Traits = Traits_;
  using ElementC = typename Traits::ElementC;
  using AccumulatorType = typename Traits::AccumulatorType;

  __device__ static void apply(
      AccumulatorType& acc_value,
      ElementC* d_C,
      int global_row, int global_col,
      int ldc) {

    // ReLU: max(0, x)
    // PTX: max.f32 %r_out, %r_acc, 0f00000000;
    if (acc_value < AccumulatorType(0)) {
      acc_value = AccumulatorType(0);
    }
    // 注: 实际代码用 __builtin_ptx_max 或 intrinsic

    ElementC output = static_cast<ElementC>(acc_value);
    // d_C[global_row * ldc + global_col] = output;
  }
};

// ============================================================================
// BiasAdd Epilogue 操作
// ============================================================================
//
// WHY Bias 融合:
//   LLM 推理中，每个 Transformer 层的 Linear 后面几乎都跟着 bias。
//   独立 kernel: GEMM → write → read bias → add → write (2 reads + 2 writes)
//   融合 kernel: GEMM + bias (在寄存器中) → write (1 read + 1 write)
//   → 减少 50% 的 global memory 带宽消耗

template <typename Traits_>
struct BiasAddEpilogue {
  using Traits = Traits_;
  using ElementC = typename Traits::ElementC;
  using AccumulatorType = typename Traits::AccumulatorType;

  __device__ static void apply(
      AccumulatorType& acc_value,
      const ElementC* d_bias,    // 偏置向量 (长度 = N)
      int global_col,            // 列索引 → 偏置索引
      ElementC* d_C,
      int global_row, int /*global_col_out*/,
      int ldc) {

    // Load bias (从 global memory 或 constant memory)
    ElementC bias_val = d_bias[global_col];

    // 累计 (在寄存器中完成，零额外内存带宽)
    acc_value += static_cast<AccumulatorType>(bias_val);

    ElementC output = static_cast<ElementC>(acc_value);
    // d_C[global_row * ldc + global_col] = output;
  }
};

// ============================================================================
// ResidualAdd Epilogue 操作 (残差连接)
// ============================================================================
//
// WHY 残差融合:
//   Transformer 每一层都有残差: output = LayerNorm(x + Attention(x))
//   或: output = LayerNorm(x + FFN(x))
//   将这个加法融合到上一层的 GEMM epilogue 中。
//
// 类比: 做菜时把调味料直接加到锅里，而不是先装盘再加调味料。

template <typename Traits_>
struct ResidualAddEpilogue {
  using Traits = Traits_;
  using ElementC = typename Traits::ElementC;
  using AccumulatorType = typename Traits::AccumulatorType;

  __device__ static void apply(
      AccumulatorType& acc_value,
      const ElementC* d_residual,  // 残差输入 (与 C 同维度)
      int global_row, int global_col,
      int ldc_residual,
      ElementC* d_C,
      int ldc) {

    // Load residual
    // ElementC residual = d_residual[global_row * ldc_residual + global_col];
    // 伪代码: acc_value += static_cast<AccumulatorType>(residual);

    ElementC output = static_cast<ElementC>(acc_value);
    // d_C[global_row * ldc + global_col] = output;
  }
};

// ============================================================================
// GELU Epilogue (LLM 中常用激活函数)
// ============================================================================
//
// WHY GELU: BERT, GPT 等模型使用 GELU 而非 ReLU。
//   GELU(x) = x * Φ(x) 其中 Φ 是标准正态分布的 CDF。
//   近似公式: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
//
// 在 epilogue 中计算 GELU:
//   - 比独立 kernel 快 ~3 倍 (避免额外的内存读写)
//   - 精度损失可忽略 (GELU 本身是非线性的)

template <typename Traits_>
struct GeluEpilogue {
  using Traits = Traits_;
  using ElementC = typename Traits::ElementC;
  using AccumulatorType = typename Traits::AccumulatorType;

  // GELU 常数
  static constexpr float kSqrt2OverPi = 0.7978845608028654f;   // √(2/π)
  static constexpr float kGeluCoef     = 0.044715f;            // tanh 公式系数

  __device__ static void apply(
      AccumulatorType& acc_value,
      ElementC* d_C,
      int global_row, int global_col,
      int ldc) {

    // GELU approximation (tanh formula)
    // float x = static_cast<float>(acc_value);
    // float x3 = x * x * x;
    // float inner = kSqrt2OverPi * (x + kGeluCoef * x3);
    // float gelu_x = 0.5f * x * (1.0f + tanhf(inner));
    //
    // 注: tanhf 是 CUDA 内建函数，直接映射为 PTX 指令

    ElementC output = static_cast<ElementC>(acc_value);
    // d_C[global_row * ldc + global_col] = output;
  }
};

// ============================================================================
// Epilogue 组合器 - 编译期操作融合
// ============================================================================
//
// WHY 组合器: 用户可以写:
//   using MyEpilogue = CombineEpilogue<BiasAdd, Relu, ResidualAdd>;
//   编译器会将三个操作融合为一个 epilogue pipeline。
//
// 模板展开后:
//   CombineEpilogue<BiasAdd, Relu, ResidualAdd>::apply(acc, ...)
//   → BiasAdd::apply(acc, bias, ...);
//   → Relu::apply(acc, ...);
//   → ResidualAdd::apply(acc, residual, ...);
//   → IdentityEpilogue::apply(acc, d_C, ...);  // 最终写出
//
//   所有操作在寄存器中完成，只有最后一步需要写 global memory。
//   这被称为 "Element-wise Fusion"。

template <typename... Ops>
struct CombineEpilogue;

// 递归基: 空组合 = Identity
template <>
struct CombineEpilogue<> : IdentityEpilogue<GemmKernelTraits<float,float,float,RowMajor,ColumnMajor,RowMajor,Sm80,TensorOp,TileShape128x128x32,WarpShape64x64x32,Mma16x8x16>> {};

// 默认: 直接用 Identity
template <typename Op>
struct CombineEpilogue<Op> : Op {};

// 多操作组合 (编译期递归展开)
template <typename First, typename... Rest>
struct CombineEpilogue<First, Rest...> {
  using Traits = typename First::Traits;  // 简化的 traits 传递

  template <typename... Args>
  __device__ static void apply(Args&&... args) {
    // 先应用 First 操作
    First::apply(args...);
    // 再递归应用剩余操作
    CombineEpilogue<Rest...>::apply(args...);
  }
};

// ============================================================================
// Epilogue 工厂 - 从编译期配置生成 Epilogue
// ============================================================================
//
// WHY 工厂: 不同类型的 epilogue 需要不同的额外参数 (bias ptr, residual ptr 等)
//   编译期工厂根据配置选择正确的 Epilogue 类型和参数签名。

enum class EpilogueType : int {
  kIdentity    = 0,
  kBiasAdd     = 1,
  kReLU        = 2,
  kBiasAddReLU = 3,
  kGelu        = 4,
  kResidual    = 5,
  kResidualReLU = 6,
};

// 从 EpilogueType 编译期映射到具体 Epilogue 类
template <EpilogueType Type, typename Traits>
struct EpilogueFactory;

template <typename Traits>
struct EpilogueFactory<EpilogueType::kIdentity, Traits> {
  using type = IdentityEpilogue<Traits>;
};

template <typename Traits>
struct EpilogueFactory<EpilogueType::kReLU, Traits> {
  using type = ReluEpilogue<Traits>;
};

template <typename Traits>
struct EpilogueFactory<EpilogueType::kBiasAdd, Traits> {
  using type = BiasAddEpilogue<Traits>;
};

template <typename Traits>
struct EpilogueFactory<EpilogueType::kBiasAddReLU, Traits> {
  using type = CombineEpilogue<BiasAddEpilogue<Traits>, ReluEpilogue<Traits>>;
};

template <typename Traits>
struct EpilogueFactory<EpilogueType::kGelu, Traits> {
  using type = GeluEpilogue<Traits>;
};

} // namespace kernel
} // namespace cutlass_style
