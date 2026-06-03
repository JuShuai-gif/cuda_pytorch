// =============================================================================
// 第 05.3 章：CUTLASS 风格的单头文件设计模式
//
// CUTLASS 是一个仅头文件的模板库。所有代码位于 .h 文件中；
// 没有 .cpp 文件。本文件模拟一个自包含的仅头文件库设计，
// 模仿真实 CUTLASS 的规范：
//
//   1. Include 保护（或 #pragma once）
//   2. 命名空间层级（cutlass::gemm、cutlass::epilogue 等）
//   3. 使用 #if defined(...) 的条件编译
//   4. 通过 #define 宏的编译期配置
//   5. 平台抽象（模拟 CUDA/非 CUDA 路径）
//   6. 通过模板实现静态多态（无虚函数分发）
//   7. 嵌套命名空间中的工具 traits 和辅助函数
//   8. 宏辅助代码生成
//   9. Debug vs Release 构建的条件代码
//
// 编译：g++ -std=c++20 -o 03_cutlass_header_only 03_cutlass_header_only.cpp
// =============================================================================

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <string>
#include <type_traits>

// 模拟 CUTLASS 宏定义（通常来自 cutlass.h）
// 这些通常由构建系统或顶层头文件定义。

#ifndef CUTLASS_DEBUG
#define CUTLASS_DEBUG 0
#endif

#ifndef CUTLASS_ENABLE_TENSOR_CORES
#define CUTLASS_ENABLE_TENSOR_CORES 1
#endif

#ifndef CUTLASS_MAX_SMEM
#define CUTLASS_MAX_SMEM (48 * 1024)
#endif

// =============================================================================
// 模拟：cutlass/platform/platform.h
// =============================================================================
// 平台检测：区分 CUDA 和仅 CPU 构建。
// 在真实 CUTLASS 中，这会检查 __CUDACC__、__CUDA_ARCH__ 等。

#if defined(__CUDACC__) || 1  // 模拟：演示中始终为真
#define CUTLASS_CUDA_ENABLED 1
#else
#define CUTLASS_CUDA_ENABLED 0
#endif

#define CUTLASS_HOST_DEVICE  // 在真实 CUDA 中：__host__ __device__
#define CUTLASS_ALIGN(k) alignas(k)

// =============================================================================
// 模拟：cutlass/numeric_types.h
// =============================================================================

namespace cutlass {

// 半精度模拟（在真实 CUTLASS 中，这包装 __half）
struct half_t {
  unsigned short storage;

  half_t() : storage(0) {}
  explicit half_t(float v) {
    // 简化：仅存储截断的 float 位
    union { float f; unsigned int i; } u;
    u.f = v;
    storage = static_cast<unsigned short>(u.i >> 16);
  }
  explicit operator float() const {
    union { unsigned int i; float f; } u;
    u.i = static_cast<unsigned int>(storage) << 16;
    return u.f;
  }
};

// bfloat16 模拟（在真实 CUTLASS 中：cutlass::bfloat16_t）
struct bfloat16_t {
  unsigned short storage;

  bfloat16_t() : storage(0) {}
  explicit bfloat16_t(float v) {
    union { float f; unsigned int i; } u;
    u.f = v;
    storage = static_cast<unsigned short>(u.i >> 16);
  }
  explicit operator float() const {
    union { unsigned int i; float f; } u;
    u.i = static_cast<unsigned int>(storage) << 16;
    return u.f;
  }
};

// 复数类型
template <typename T>
struct complex {
  T real, imag;
  complex() : real(T{}), imag(T{}) {}
  complex(T r, T i) : real(r), imag(i) {}
};

}  // namespace cutlass

// =============================================================================
// 模拟：cutlass/layout/matrix.h
// =============================================================================

namespace cutlass {
namespace layout {

// 行主序布局标签和工具
struct RowMajor {
  // 对于 [rows x cols] 矩阵：
  //   元素 (r, c) 位于偏移量 r * leading_dim + c
  // leading dimension（步长）通常是 cols（或填充后的值）。

  struct Index {
    int row, col;
  };

  // 静态映射函数
  static CUTLASS_HOST_DEVICE constexpr int offset(int r, int c,
                                                    int /*rows*/,
                                                    int /*cols*/,
                                                    int leading_dim) {
    return r * leading_dim + c;
  }

  // "快速"维度（单位步长）是列维度
  static constexpr int stride_rank = 1;
};

// 列主序布局标签
struct ColumnMajor {
  struct Index {
    int row, col;
  };

  static CUTLASS_HOST_DEVICE constexpr int offset(int r, int c,
                                                    int rows,
                                                    int /*cols*/,
                                                    int leading_dim) {
    return c * leading_dim + r;
  }

  static constexpr int stride_rank = 0;
};

// 检查布局是行主序还是列主序的辅助函数
template <typename Layout>
struct IsRowMajor : std::false_type {};

template <>
struct IsRowMajor<RowMajor> : std::true_type {};

template <typename Layout>
inline constexpr bool is_row_major_v = IsRowMajor<Layout>::value;

}  // namespace layout
}  // namespace cutlass

// =============================================================================
// 模拟：cutlass/gemm/threadblock/default_mma.h
// =============================================================================

namespace cutlass {
namespace gemm {
namespace threadblock {

// MMA 运算符标签（使用哪种 tensor core 指令）
struct MmaSimt {};         // CUDA 核心（SIMT）MMA
struct MmaTensorOpF16 {};  // Tensor Core fp16 MMA
struct MmaTensorOpTF32 {}; // Tensor Core tf32 MMA

// 基于元素类型和 tensor core 可用性的默认 MMA 选择器
template <typename ElementA, typename ElementB, bool TensorCores>
struct DefaultMma {};

// fp32 + 无 tensor cores -> SIMT
template <typename ElementA, typename ElementB>
struct DefaultMma<ElementA, ElementB, false> {
  using type = MmaSimt;
};

// fp16 + tensor cores -> TensorOp
template <typename ElementA, typename ElementB>
struct DefaultMma<ElementA, ElementB, true> {
  using type = MmaTensorOpF16;
};

// 默认 MMA 别名
template <typename ElementA, typename ElementB>
using default_mma_t = typename DefaultMma<
    ElementA, ElementB,
    CUTLASS_ENABLE_TENSOR_CORES>::type;

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass

// =============================================================================
// 模拟：cutlass/gemm/kernel/gemm.h
// =============================================================================

namespace cutlass {
namespace gemm {
namespace kernel {

// 内核配置
template <typename MmaOp_, typename EpilogueOp_, typename TileShape_>
struct Gemm {
  using MmaOp      = MmaOp_;
  using EpilogueOp = EpilogueOp_;
  using TileShape  = TileShape_;

  // 运行时启动参数
  int grid_m, grid_n;

  Gemm(int problem_M, int problem_N)
      : grid_m((problem_M + TileShape::M - 1) / TileShape::M),
        grid_n((problem_N + TileShape::N - 1) / TileShape::N) {}

  void launch() const {
    std::cout << "  [Gemm 内核] 正在启动，grid ("
              << grid_m << ", " << grid_n << ")" << std::endl;
    std::cout << "    MMA：" << MmaOp::name() << std::endl;
    std::cout << "    Epilogue：" << EpilogueOp::name() << std::endl;
    std::cout << "    Tile：" << TileShape::M << "x" << TileShape::N
              << std::endl;
  }
};

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass

// =============================================================================
// 模拟：cutlass/epilogue/thread/linear_combination.h
// =============================================================================

namespace cutlass {
namespace epilogue {
namespace thread {

template <typename ElementOutput, int Count, typename ElementAccumulator = float>
struct LinearCombination {
  using Element = ElementOutput;

  static constexpr int kCount = Count;

  static std::string name() { return "LinearCombination"; }
  static std::string desc() { return "D = alpha * accum + beta * C"; }
};

template <typename ElementOutput, int Count>
struct LinearCombinationRelu {
  using Element = ElementOutput;

  static constexpr int kCount = Count;

  static std::string name() { return "LinearCombinationRelu"; }
  static std::string desc() { return "D = relu(alpha * accum + beta * C)"; }
};

}  // namespace thread
}  // namespace epilogue
}  // namespace cutlass

// =============================================================================
// 模拟：cutlass/core_io.h（Debug/Release 宏）
// =============================================================================

namespace cutlass {

#if CUTLASS_DEBUG
#define CUTLASS_PRINT(msg) \
  std::cout << "[CUTLASS DEBUG] " << msg << std::endl
#else
#define CUTLASS_PRINT(msg) ((void)0)
#endif

}  // namespace cutlass

// =============================================================================
// 模拟：cutlass/arch/mma.h（Tensor Op 标签类型）
// =============================================================================

namespace cutlass {
namespace arch {

// Tensor Core MMA 指令描述符
template <int M, int N, int K, typename ElementA, typename ElementB,
          typename ElementC>
struct MmaInstruction {
  static constexpr int kM = M;
  static constexpr int kN = N;
  static constexpr int kK = K;

  using ElementOpA = ElementA;
  using ElementOpB = ElementB;
  using ElementOpC = ElementC;

  static std::string name() {
    return "mma.sync " + std::to_string(M) + "x" + std::to_string(N) + "x"
           + std::to_string(K);
  }
};

// 常见 MMA 指令
using MmaF16_16x8x8   = MmaInstruction<16, 8, 8, half_t, half_t, float>;
using MmaF16_16x8x16  = MmaInstruction<16, 8, 16, half_t, half_t, float>;
using MmaTF32_16x8x4  = MmaInstruction<16, 8, 4, float, float, float>;
using MmaTF32_16x8x8  = MmaInstruction<16, 8, 8, float, float, float>;

}  // namespace arch
}  // namespace cutlass

// =============================================================================
// 模拟：TileShape（编译期 tile 维度）
// =============================================================================

namespace cutlass {

template <int M_, int N_, int K_>
struct GemmShape {
  static constexpr int M = M_;
  static constexpr int N = N_;
  static constexpr int K = K_;
};

}  // namespace cutlass

// =============================================================================
// 模拟：MMA 操作分发
// =============================================================================

namespace cutlass {
namespace gemm {
namespace threadblock {

// SIMT MMA：基本实现（模拟）
struct SimtMmaOp {
  static std::string name() { return "SIMT MMA（CUDA 核心）"; }
};

// Tensor Core MMA：fp16
struct TensorOpF16MmaOp {
  using Instruction = arch::MmaF16_16x8x8;
  static std::string name() { return "Tensor Core F16 MMA"; }
};

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass

// =============================================================================
//                                   主函数
// =============================================================================

int main() {
  using namespace std;

  cout << "=== 第 05.3 章：CUTLASS 风格的单头文件设计 ===\n" << endl;

  // --- 测试 1：平台检测 ---
  cout << "[测试 1] 平台检测：" << endl;
  cout << "  CUDA_ENABLED：" << CUTLASS_CUDA_ENABLED << endl;
  cout << "  DEBUG：" << CUTLASS_DEBUG << endl;
  cout << "  TENSOR_CORES：" << CUTLASS_ENABLE_TENSOR_CORES << endl;

  // --- 测试 2：布局系统 ---
  cout << "\n[测试 2] 布局偏移计算：" << endl;
  using namespace cutlass::layout;
  int rm_offset = RowMajor::offset(2, 3, 10, 10, 10);
  int cm_offset = ColumnMajor::offset(2, 3, 10, 10, 10);
  cout << "  RowMajor::offset(2,3) = " << rm_offset << "（row*ld+col）"
       << endl;
  cout << "  ColumnMajor::offset(2,3) = " << cm_offset << "（col*ld+row）"
       << endl;
  assert(rm_offset == 2 * 10 + 3);
  assert(cm_offset == 3 * 10 + 2);

  static_assert(is_row_major_v<RowMajor>);
  static_assert(!is_row_major_v<ColumnMajor>);

  // --- 测试 3：MMA 指令描述符 ---
  cout << "\n[测试 3] MMA 指令描述符：" << endl;
  cout << "  " << cutlass::arch::MmaF16_16x8x8::name() << endl;
  cout << "  " << cutlass::arch::MmaF16_16x8x16::name() << endl;
  cout << "  " << cutlass::arch::MmaTF32_16x8x8::name() << endl;

  // --- 测试 4：默认 MMA 选择 ---
  cout << "\n[测试 4] 默认 MMA 选择：" << endl;
  using namespace cutlass::gemm::threadblock;
  using MmaFp32 = default_mma_t<float, float>;
  using MmaFp16 = default_mma_t<cutlass::half_t, cutlass::half_t>;

  cout << "  float MMA："
       << (std::is_same_v<MmaFp32, MmaSimt> ? "SIMT" : "TensorOp") << endl;
  cout << "  half  MMA："
       << (std::is_same_v<MmaFp16, MmaTensorOpF16> ? "TensorOp" : "SIMT")
       << endl;

  // --- 测试 5：Epilogue 操作 ---
  cout << "\n[测试 5] Epilogue 操作：" << endl;
  using namespace cutlass::epilogue::thread;
  using LC = LinearCombination<float, 8>;
  using LCR = LinearCombinationRelu<float, 8>;

  cout << "  " << LC::name() << "：" << LC::desc() << endl;
  cout << "  " << LCR::name() << "：" << LCR::desc() << endl;

  // --- 测试 6：完整内核配置 ---
  cout << "\n[测试 6] 完整内核配置：" << endl;
  using TileS = cutlass::GemmShape<128, 128, 8>;
  using Mma   = SimtMmaOp;
  using Epi   = cutlass::epilogue::thread::LinearCombination<float, 8>;

  cutlass::gemm::kernel::Gemm<Mma, Epi, TileS> gemm(1024, 1024);
  gemm.launch();

  // --- 测试 7：Debug 打印宏 ---
  cout << "\n[测试 7] Debug 打印宏：";
  CUTLASS_PRINT("仅调试版本中会显示");
  cout << "（release 中应为空）" << endl;

  // --- 测试 8：数值类型 ---
  cout << "\n[测试 8] 模拟数值类型：" << endl;
  cutlass::half_t h(3.14f);
  cutlass::bfloat16_t bf(2.71f);
  cout << "  half_t(3.14)：约 " << static_cast<float>(h) << endl;
  cout << "  bfloat16_t(2.71)：约 " << static_cast<float>(bf) << endl;

  // --- 测试 9：复数类型 ---
  cout << "\n[测试 9] 复数类型：" << endl;
  cutlass::complex<float> cf(1.0f, 2.0f);
  cout << "  complex<float>(" << cf.real << "," << cf.imag << ")" << endl;

  cout << "\n所有测试通过！" << endl;
  return 0;
}
