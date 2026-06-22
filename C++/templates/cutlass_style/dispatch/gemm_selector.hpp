#pragma once

#include <cstdint>
#include <iostream>
#include <type_traits>

#include "include/arch_tag.hpp"
#include "include/layout.hpp"
#include "kernel_dispatch.hpp"

namespace cutlass_style {
namespace dispatch {

// ============================================================================
// GemmSelector - 运行时 GEMM 入口
// ============================================================================
//
// WHY 需要 GemmSelector:
//   GPU 程序在运行时才知道硬件能力。用户传 float 运行时变量，
//   但 kernel 必须用编译期常量。GemmSelector 是"运行时→编译期"的桥梁。
//
// ┌──────────────────────────────────────────────────────────────────┐
// │  Mermaid 图: 运行时到编译期的转换                                │
// │                                                                  │
// │  User Code (Runtime)            GemmSelector (Compile-time)      │
// │  ┌──────────────────┐          ┌──────────────────────────────┐  │
// │  │ int arch = 80;    │          │ if constexpr (arch == 80) {  │  │
// │  │ int dtype = FP16; │──────────▶│   using Traits = Sm80Fp16  │  │
// │  │ int layout = NN;  │          │   Traits;                   │  │
// │  │ float *A, *B, *C; │          │ }                            │  │
// │  │ int M, N, K;      │          │ GemmKernel<Traits>::launch( │  │
// │  └──────────────────┘          │   A, B, C, M, N, K);        │  │
// │  变量在运行时才知道               └──────────────────────────────┘  │
// │                                  类型在编译期确定                  │
// │                                                                  │
// │  关键洞察:                                                        │
// │  编译器为所有可能的 arch/dtype/layout 组合生成所有 kernel 变体。  │
// │  运行时只需一个 switch/case 跳转到正确的实例。                    │
// │  这被称为 "编译期多态" (Compile-time Polymorphism)。             │
// └──────────────────────────────────────────────────────────────────┘
//
// 类比: GemmSelector 相当于 C 语言的宏展开 + 函数指针表:
//   #define GEMM(arch, dtype) GEMM_##arch##_##dtype
//   但 CUTLASS 用类型系统 + 模板，享受类型安全。

// ============================================================================
// 运行时架构检测 (真实代码中使用 cudaGetDeviceProperties)
// ============================================================================

// 伪代码: 真实实现会调用 CUDA Runtime API
inline int detect_gpu_architecture() {
  // 实际实现:
  //   cudaDeviceProp prop;
  //   cudaGetDeviceProperties(&prop, 0);
  //   return prop.major * 10 + prop.minor;
  //
  // 或者使用编译期宏:
  //   #if defined(__CUDA_ARCH__)
  //   return __CUDA_ARCH__;
  //   #endif

  return 80; // 默认返回 SM80 (Ampere)
}

// ============================================================================
// GemmSelector - 运行时选择并启动 kernel
// ============================================================================
//
// 模板参数说明:
//   AllConfigs: TypeList of all supported configurations
//     e.g., TypeList<
//       GemmKernelTraits<...Config1...>,
//       GemmKernelTraits<...Config2...>,
//       ...
//     >
//
// 流程:
//   1. 运行时: 用户调用 gemm(A, B, C, M, N, K)
//   2. 运行时: detect_gpu_architecture() 获取硬件信息
//   3. 编译期: for_each_type 遍历所有 kernel 配置
//   4. 编译期: 每个配置检查 arch/dtype/layout 是否匹配
//   5. 运行时: 匹配的 kernel 被 launch

template <typename AllKernelTraits>
class GemmSelector {
 public:
  // 所有支持的 kernel traits 列表 (编译期)
  using KernelTraitsList = AllKernelTraits;

  // =========================================================================
  // select_and_launch - 运行时 kernel 选择和启动
  // =========================================================================
  //
  // 真实代码中，这里会:
  //   1. 获取 GPU 属性 (arch)
  //   2. 检查用户传入的数据类型和 layout (运行时参数)
  //   3. 遍历 KernelTraitsList，找到第一个匹配的
  //   4. 实例化并启动 kernel
  //
  // 每次遍历都不会带来运行时开销——因为只是 switch/case 或函数指针。

  template <typename ElementA, typename ElementB, typename ElementC>
  static void select_and_launch(
      int runtime_arch,
      int runtime_layout_a,
      int runtime_layout_b,
      ElementA* d_A,
      ElementB* d_B,
      ElementC* d_C,
      int M, int N, int K) {

    // 注: 这是伪代码骨架。真实实现:
    //
    // // Step 1: 运行时 arch → 编译期 arch 映射
    // switch (runtime_arch) {
    //   case 70: launch_arch<Sm70>(...); break;
    //   case 75: launch_arch<Sm75>(...); break;
    //   case 80: launch_arch<Sm80>(...); break;
    //   case 90: launch_arch<Sm90>(...); break;
    // }
    //
    // // Step 2: 每个 arch 内部，layout/dtype 用编译期 SFINAE 选择
    // template <typename ArchTag>
    // void launch_arch(...) {
    //   if constexpr (is_row_major_a && is_col_major_b) {
    //     using Traits = DefaultGemmConfiguration<ArchTag, ElemA, ElemB, ElemC>;
    //     using Kernel = GemmKernel<Traits>;
    //     Kernel::launch(d_A, d_B, d_C, M, N, K);
    //   }
    // }

    std::cout << "[GemmSelector] Selecting kernel for arch=" << runtime_arch
              << ", M=" << M << ", N=" << N << ", K=" << K << std::endl;

    // 运行时 arch 分派
    dispatch_by_arch<ElementA, ElementB, ElementC>(
        runtime_arch, d_A, d_B, d_C, M, N, K);
  }

 private:
  // =========================================================================
  // 架构分派: 运行时 int → 编译期 ArchTag
  // =========================================================================
  //
  // WHY 使用 switch/case 而非虚函数:
  //   GPU kernel 启动参数 (grid/block shape) 必须在编译期确定。
  //   如果 CUDA 支持虚函数，也不应该在 GPU 代码中用 (vtable 在 device memory 中极慢)。
  //   所以必须在 host 端就用 switch 完成分派。
  //
  // 模板展开后: 编译器为每个 case 生成独立的 kernel 启动代码。
  //   比如 case 80: 中的代码调用 GemmKernel<Sm80Traits>::launch(...)
  //   → 生成对应 SM80 的 PTX 指令。

  template <typename ElementA, typename ElementB, typename ElementC>
  static void dispatch_by_arch(
      int arch,
      ElementA* d_A, ElementB* d_B, ElementC* d_C,
      int M, int N, int K) {

    switch (arch) {
      case 70:
        dispatch_by_arch_impl<Sm70, ElementA, ElementB, ElementC>(
            d_A, d_B, d_C, M, N, K);
        break;
      case 75:
        dispatch_by_arch_impl<Sm75, ElementA, ElementB, ElementC>(
            d_A, d_B, d_C, M, N, K);
        break;
      case 80:
        dispatch_by_arch_impl<Sm80, ElementA, ElementB, ElementC>(
            d_A, d_B, d_C, M, N, K);
        break;
      case 90:
        dispatch_by_arch_impl<Sm90, ElementA, ElementB, ElementC>(
            d_A, d_B, d_C, M, N, K);
        break;
      default:
        std::cerr << "[GemmSelector] Unsupported GPU architecture: "
                  << arch << std::endl;
        break;
    }
  }

  // =========================================================================
  // 编译期 kernel 实例化和启动
  // =========================================================================
  template <typename ArchTag, typename ElementA, typename ElementB, typename ElementC>
  static void dispatch_by_arch_impl(
      ElementA* d_A, ElementB* d_B, ElementC* d_C,
      int M, int N, int K) {

    // 使用 DefaultGemmConfiguration 自动选择 tile 配置
    using Config = DefaultGemmConfiguration<ArchTag, ElementA, ElementB, ElementC>;

    // 构建完整的 KernelTraits
    using Traits = GemmKernelTraits<
        ElementA, ElementB, ElementC,
        RowMajor, ColumnMajor, RowMajor,
        ArchTag,
        TensorOp,
        typename Config::TileShape,
        typename Config::WarpShape,
        typename Config::MmaInstruction
    >;

    // 编译期验证
    Traits::validate();

    // 日志信息 (发布版本应去掉)
    std::cout << "[GemmSelector] Arch=" << ArchTag::name
              << ", Tile=" << Traits::TileShape::M
              << "x" << Traits::TileShape::N
              << "x" << Traits::TileShape::K
              << ", Threads/block=" << Traits::kThreads
              << ", Smem=" << Traits::kSharedMemorySize << " bytes"
              << std::endl;

    // 启动 kernel (伪代码，真实代码需要 CUDA runtime)
    //
    // dim3 grid(
    //     (M + Traits::TileShape::M - 1) / Traits::TileShape::M,
    //     (N + Traits::TileShape::N - 1) / Traits::TileShape::N
    // );
    // dim3 block(Traits::kThreads);
    //
    // GemmKernel<Traits><<<grid, block, Traits::kSharedMemorySize>>>(
    //     d_A, d_B, d_C, M, N, K
    // );

    std::cout << "[GemmSelector] Kernel would launch with "
              << "grid=("
              << (M + Traits::TileShape::M - 1) / Traits::TileShape::M
              << ", "
              << (N + Traits::TileShape::N - 1) / Traits::TileShape::N
              << "), block=" << Traits::kThreads
              << std::endl;
  }
};

// ============================================================================
// 预注册的 kernel 选择器 (所有支持的 kernel 类型)
// ============================================================================

// 支持的数据类型
using SupportedElementTypes = TypeList<float, double, int32_t>;

// 支持的架构
using SupportedArchitectures = TypeList<Sm70, Sm75, Sm80, Sm90>;

// 全局 selector (单例模式 - 编译期)
// 真实代码中这里会注册所有 kernel 变体
using DefaultSelector = GemmSelector<TypeList<
    // 这里列出所有预编译的 kernel traits 类型
    DefaultGemmTraits<float, float, float>,
    DefaultGemmTraits<double, double, double>
>>;

} // namespace dispatch
} // namespace cutlass_style
