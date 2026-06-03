#pragma once

#include <cstdint>
#include <type_traits>
#include <string>

// ============================================================================
// mini_inference_engine - 编译期推理引擎配置
// ============================================================================

namespace mini_inference {

// ============================================================================
// DataType - 推理引擎支持的数据类型
// ============================================================================

enum class DataType : int {
  kFloat32 = 0,
  kFloat16 = 1,
  kBFloat16 = 2,
  kInt8    = 3,
  kInt4    = 4,
};

// 编译期: DataType → C++ 类型映射
template <DataType D>
struct DataTypeToCpp;

template <> struct DataTypeToCpp<DataType::kFloat32>  { using type = float; };
template <> struct DataTypeToCpp<DataType::kFloat16>  { using type = float; }; // __half
template <> struct DataTypeToCpp<DataType::kBFloat16> { using type = float; }; // __nv_bfloat16
template <> struct DataTypeToCpp<DataType::kInt8>     { using type = int8_t; };
template <> struct DataTypeToCpp<DataType::kInt4>     { using type = int8_t; }; // packed

template <DataType D>
using data_type_t = typename DataTypeToCpp<D>::type;

// 编译期: 获取类型大小
template <DataType D>
inline constexpr int data_type_size_v = []() constexpr {
  switch (D) {
    case DataType::kFloat32:  return 4;
    case DataType::kFloat16:  return 2;
    case DataType::kBFloat16: return 2;
    case DataType::kInt8:     return 1;
    case DataType::kInt4:     return 1; // 2个 INT4 打包为 1 byte
  }
  return 4;
}();

// ============================================================================
// LayoutType - 推理引擎布局
// ============================================================================

enum class LayoutType : int {
  kRowMajor    = 0,  // C-contiguous (PyTorch default)
  kColumnMajor = 1,  // Fortran-contiguous (BLAS default)
  kChannelsLast = 2, // NHWC (TensorRT favorite)
};

// ============================================================================
// EngineConfig - 推理引擎配置 (编译期)
// ============================================================================
//
// WHY 编译期配置:
//   LLM 推理对延迟极度敏感 (每个 token ~10ms)。
//   所有可确定的参数都应该在编译期确定，避免运行时分支。
//
// ┌──────────────────────────────────────────────────────────────────┐
// │  Mermaid 图: EngineConfig 中的编译期 vs 运行时参数              │
// │                                                                  │
// │  Compile-time (template params):                                 │
// │  ┌──────────────────────────────────────────────────────┐       │
// │  │ DataType (FP16/BF16/INT8)                           │       │
// │  │ Layout   (RowMajor/ColumnMajor)                     │       │
// │  │ ArchTag  (SM70/SM75/SM80/SM90)                      │       │
// │  │ TileShapeList (候选 tile 列表)                       │       │
// │  │                                                      │       │
// │  │ 这些不变: 驱动 PTX 指令生成、寄存器分配、smem 布局    │       │
// │  └──────────────────────────────────────────────────────┘       │
// │                                                                  │
// │  Runtime (constructor args):                                    │
// │  ┌──────────────────────────────────────────────────────┐       │
// │  │ batch_size  (可变: 1, 4, 8, 32, ...)               │       │
// │  │ seq_len     (可变: 1~2048 for推理，1~8192 for训练)  │       │
// │  │ num_heads   (模型结构确定)                            │       │
// │  │ head_dim    (64/128)                                 │       │
// │  │                                                      │       │
// │  │ 这些变化: 影响 grid/block 尺寸，但不影响 PTX 指令     │       │
// │  └──────────────────────────────────────────────────────┘       │
// └──────────────────────────────────────────────────────────────────┘
//
// 类比: EngineConfig 相当于编译 C 程序时的 -D 宏定义:
//   g++ -DUSE_FP16 -D__CUDA_ARCH__=800 -DTILE_SIZE=128 main.cu
//   每个宏组合编译出一份不同的 binary。
//   EngineConfig 将这些"宏"提升为类型系统的一部分，类型安全。

template <
    DataType DType_,        // 推理精度
    LayoutType Layout_,     // 张量布局
    int Arch_,              // GPU 架构 (70/75/80/90)
    typename TileShapeList_ = void  // 候选 Tile 列表
>
struct EngineConfig {
  static constexpr DataType DType = DType_;
  static constexpr LayoutType Layout = Layout_;
  static constexpr int Arch = Arch_;

  using ElementType = data_type_t<DType>;
  static constexpr int kElementSize = data_type_size_v<DType>;

  using TileShapeList = TileShapeList_;

  // ── 编译期架构特性检测 ──
  static constexpr bool kHasTensorCore = (Arch >= 70);
  static constexpr bool kHasInt8TensorCore = (Arch >= 75);
  static constexpr bool kHasBf16TensorCore = (Arch >= 80);
  static constexpr bool kHasFp8TensorCore = (Arch >= 90);
  static constexpr bool kHasTMA = (Arch >= 90);

  // ── 编译期 kernel 特性 ──
  static constexpr bool kUseTensorCore =
      (DType == DataType::kFloat16 && kHasTensorCore) ||
      (DType == DataType::kBFloat16 && kHasBf16TensorCore) ||
      (DType == DataType::kInt8 && kHasInt8TensorCore);

  // ── 共享内存预算 (bytes) ──
  static constexpr int kSmemBudget = []() constexpr {
    if constexpr (Arch >= 90) return 227 * 1024; // H100
    else if constexpr (Arch >= 80) return 164 * 1024; // A100
    else if constexpr (Arch >= 75) return 64 * 1024;  // T4
    else return 96 * 1024;  // V100
  }();

  // ── 寄存器预算 (per thread) ──
  static constexpr int kRegisterBudget = 255;

  // ── 描述信息 (日志/调试) ──
  static constexpr const char* arch_name() {
    if constexpr (Arch >= 90) return "Hopper (H100/H200)";
    else if constexpr (Arch >= 80) return "Ampere (A100/A6000)";
    else if constexpr (Arch >= 75) return "Turing (T4)";
    else return "Volta (V100)";
  }

  static constexpr const char* dtype_name() {
    switch (DType) {
      case DataType::kFloat32:  return "FP32";
      case DataType::kFloat16:  return "FP16";
      case DataType::kBFloat16: return "BF16";
      case DataType::kInt8:     return "INT8";
      case DataType::kInt4:     return "INT4";
      default: return "UNKNOWN";
    }
  }

  // ── 运行时配置 (由 model 确定，编译期不知道) ──
  struct RuntimeConfig {
    int batch_size = 1;
    int seq_len = 1;
    int num_heads = 32;
    int head_dim = 128;
    int hidden_dim = 4096;
    int intermediate_dim = 11008; // FFN 中间维度

    std::string to_string() const {
      return "RuntimeConfig(batch=" + std::to_string(batch_size) +
             ", seq=" + std::to_string(seq_len) +
             ", heads=" + std::to_string(num_heads) +
             ", head_dim=" + std::to_string(head_dim) + ")";
    }
  };
};

// ============================================================================
// 常用配置别名
// ============================================================================

// FP16 推理 on A100
using ConfigA100Fp16 = EngineConfig<DataType::kFloat16, LayoutType::kRowMajor, 80>;

// BF16 推理 on H100
using ConfigH100Bf16 = EngineConfig<DataType::kBFloat16, LayoutType::kRowMajor, 90>;

// INT8 推理 on T4
using ConfigT4Int8 = EngineConfig<DataType::kInt8, LayoutType::kRowMajor, 75>;

// FP32 推理 on V100
using ConfigV100Fp32 = EngineConfig<DataType::kFloat32, LayoutType::kRowMajor, 70>;

} // namespace mini_inference
