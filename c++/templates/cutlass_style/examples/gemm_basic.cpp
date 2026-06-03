#include <iostream>
#include <cstdint>

// ============================================================================
// Mini CUTLASS 示例: 基本 GEMM 调用
// ============================================================================
//
// 这个示例展示从运行时 API 到编译期 kernel dispatch 的完整流程。
// 模拟的是用户在 PyTorch 中调用 torch.matmul(A, B) 时，
// CUTLASS 后端在底层做的事情。
//
// ┌──────────────────────────────────────────────────────────────────┐
// │  用户调用 (Python)                                               │
// │  ┌──────────────────────────────────────┐                        │
// │  │ C = torch.matmul(A.half(), B.half()) │                        │
// │  └──────────────┬───────────────────────┘                        │
// │                 │ PyTorch dispatcher                             │
// │                 ▼                                                │
// │  ┌──────────────────────────────────────┐                        │
// │  │ CUTLASS GemmSelector                 │ ← dispatch/gemm_selector │
// │  └──────────────┬───────────────────────┘                        │
// │                 │                                                │
// │         ┌───────┴───────┐                                       │
// │         ▼               ▼                                       │
// │  ┌────────────┐  ┌────────────┐                                 │
// │  │ Arch Check │  │ Dtype Check│                                 │
// │  │ SM80? SM90?│  │ FP16? BF16?│                                 │
// │  └─────┬──────┘  └─────┬──────┘                                 │
// │        └───────┬───────┘                                        │
// │                ▼                                                │
// │  ┌──────────────────────────────────────┐                        │
// │  │ Kernel Traits (编译期配置聚合)        │ ← kernel_traits.hpp    │
// │  │ - Element: half, half, half          │                        │
// │  │ - Layout: RowMajor x ColumnMajor     │                        │
// │  │ - Tile: 128x128x32                   │                        │
// │  │ - Arch: Sm80                         │                        │
// │  └──────────────┬───────────────────────┘                        │
// │                 ▼                                                │
// │  ┌──────────────────────────────────────┐                        │
// │  │ GemmKernel<Traits>::launch()         │ ← kernel/gemm_kernel   │
// │  │  → mainloop (load+mma)              │                        │
// │  │  → epilogue (output)                │                        │
// │  └──────────────────────────────────────┘                        │
// └──────────────────────────────────────────────────────────────────┘

// Include the mini-CUTLASS framework
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
#include "dispatch/gemm_selector.hpp"
#include "kernel/gemm_kernel.hpp"
#include "kernel/mainloop.hpp"
#include "kernel/epilogue.hpp"

using namespace cutlass_style;

int main() {
  std::cout << "================================================================" << std::endl;
  std::cout << "  Mini CUTLASS: Basic GEMM Demo" << std::endl;
  std::cout << "================================================================\n" << std::endl;

  // =========================================================================
  // Step 1: 用户定义矩阵维度 (运行时变量)
  // =========================================================================
  //
  // 注意: M, N, K 是运行时变量——但它们不进入 kernel 模板参数。
  // kernel 使用 tile tiling: 将大矩阵切分为 tile，每个 tile 由模板定义。
  // 类比: 一本 1000 页的书 (M=1000, K=1000)，每次读 128 页 (tile)，
  //       读 8 次才能读完。128 是编译期常量，1000 是运行时变量。

  constexpr int M = 1024;   // 输出行数
  constexpr int N = 2048;   // 输出列数
  constexpr int K = 512;    // 内维度

  std::cout << "Matrix dimensions: M=" << M << ", N=" << N << ", K=" << K << std::endl;
  std::cout << "Operation: C[M×N] = A[M×K] × B[K×N]\n" << std::endl;

  // =========================================================================
  // Step 2: 数据类型选择 (编译期常量 → 模板参数)
  // =========================================================================
  //
  // WHY 数据类型是编译期的:
  //   float vs half 生成的 PTX 指令完全不同:
  //     float: fma.rn.f32  (SIMT) 或 tf32 MMA
  //     half:  mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 (Tensor Core)
  //   运行时选择只能把所有可能性编译进去 → 4x 二进制膨胀。
  //   编译期选择 → 只为需要的类型生成代码。

  // 选择 FP16 (half precision) Tensor Core GEMM
  using ElementA = float;  // 实际: __half
  using ElementB = float;  // 实际: __half
  using ElementC = float;  // 实际: __half

  std::cout << "Data types: A=" << "FP16" << ", B=" << "FP16" << ", C=" << "FP16" << std::endl;

  // =========================================================================
  // Step 3: Layout 选择 (编译期常量)
  // =========================================================================
  //
  // 典型的 PyTorch GEMM: A 是 RowMajor (C-contiguous), B 是 RowMajor
  // 但 CUTLASS 内部将 B 视为 ColumnMajor 以避免转置。
  // 这是 BLAS GEMM 的传统约定:
  //   CUTLASS: C = A × B  (A row-major, B column-major) → NN GEMM
  //   PyTorch: C = A × B^T (A row-major, B^T row-major) → NT GEMM

  using LayoutA = RowMajor;
  using LayoutB = ColumnMajor;  // 如果是 NN GEMM
  using LayoutC = RowMajor;

  std::cout << "Layout: A=" << LayoutA::name
            << ", B=" << LayoutB::name
            << ", C=" << LayoutC::name << std::endl;

  // =========================================================================
  // Step 4: GPU 架构选择 (编译期)
  // =========================================================================
  //
  // 实际代码中: #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  // 或者: cudaGetDeviceProperties() 获取运行时架构，然后 switch/case。

  using ArchTag = Sm80;
  std::cout << "GPU Architecture: " << ArchTag::name << "\n" << std::endl;

  // =========================================================================
  // Step 5: 编译期 Kernel 选择
  // =========================================================================
  //
  // 这里展示了 DefaultGemmConfiguration 如何自动选择最优 tile 配置。
  //
  // 模板展开过程 (编译器内部):
  //   1. DefaultGemmConfiguration<Sm80, half, half, half>
  //   2. 寄存器预算 = 255 (SM80)
  //   3. 候选 tile 从大到小尝试:
  //      Config256x256 → 寄存器估算 = 312 > 255 ❌
  //      Config256x128 → 寄存器估算 = 265 > 255 ❌
  //      Config128x256 → 寄存器估算 = 260 > 255 ❌
  //      Config128x128 → 寄存器估算 = 180 <= 255 ✅ ← 选中!
  //   4. 返回: TileShape<128,128,32>, WarpShape<64,64,32>

  using Config = dispatch::DefaultGemmConfiguration<ArchTag, ElementA, ElementB, ElementC>;

  std::cout << "=== Compile-time Kernel Selection ===" << std::endl;
  std::cout << "Strategy: " << Config::description << std::endl;
  std::cout << "Selected Tile: " << Config::TileShape::M
            << "x" << Config::TileShape::N
            << "x" << Config::TileShape::K << std::endl;

  // =========================================================================
  // Step 6: 构建完整的 Kernel Traits
  // =========================================================================
  //
  // GemmKernelTraits 是多维配置的最终聚合点。
  // 这个 traits 对象决定了 kernel 的每一个细节:
  //   - shared memory 大小
  //   - block 线程数
  //   - warp 排布
  //   - MMA 指令形状
  //   - 软件流水线策略

  using KernelTraits = GemmKernelTraits<
      ElementA, ElementB, ElementC,
      LayoutA, LayoutB, LayoutC,
      ArchTag,
      TensorOp,
      typename Config::TileShape,
      typename Config::WarpShape,
      typename Config::MmaInstruction
  >;

  // 编译期验证
  KernelTraits::validate();

  // =========================================================================
  // Step 7: 编译期计算出的 kernel 参数
  // =========================================================================

  std::cout << "\n=== Derived Kernel Parameters (all compile-time) ===" << std::endl;
  std::cout << "  Threads per block:      " << KernelTraits::kThreads << std::endl;
  std::cout << "  Warps per block:        " << KernelTraits::kWarpCount << std::endl;
  std::cout << "  Shared memory:          " << KernelTraits::kSharedMemorySize << " bytes" << std::endl;
  std::cout << "  MMA iterations/warp:    " << KernelTraits::kMmaIterationsTotal << std::endl;
  std::cout << "  PTX instruction:        " << KernelTraits::kPtxMmaInstruction << std::endl;
  std::cout << "  Compute intensity:      " << KernelTraits::kComputeIntensity << " ops/byte" << std::endl;

  // =========================================================================
  // Step 8: 运行时 Grid 计算
  // =========================================================================
  //
  // WHY grid/block 在运行时计算:
  //   M, N 是运行时变量，可大可小。tile 是编译期常量。
  //   Grid 将大任务分解为 tile 大小的子任务。
  //
  // 类比: 铺地砖。地砖大小 (tile) 是固定的 128×128，
  //       但房间大小 (M×N) 每次不同。Grid = 房间大小 / 地砖大小。

  int grid_m = (M + KernelTraits::TileShape::M - 1) / KernelTraits::TileShape::M;
  int grid_n = (N + KernelTraits::TileShape::N - 1) / KernelTraits::TileShape::N;

  std::cout << "\n=== Launch Configuration ===" << std::endl;
  std::cout << "  Grid:  (" << grid_m << ", " << grid_n << ") = "
            << grid_m * grid_n << " thread blocks" << std::endl;
  std::cout << "  Block: " << KernelTraits::kThreads << " threads ("
            << KernelTraits::kWarpCount << " warps)" << std::endl;
  std::cout << "  Total threads: " << grid_m * grid_n * KernelTraits::kThreads << std::endl;

  // =========================================================================
  // Step 9: 理论性能估算
  // =========================================================================

  // 总计算量: 2*M*N*K FLOPS (一次乘+一次加 = 2 ops)
  double total_flops = 2.0 * M * N * K;
  double total_data = (M * K + N * K + M * N) * sizeof(float); // FP16 = 2 bytes output

  std::cout << "\n=== Performance Estimate ===" << std::endl;
  std::cout << "  Total FLOPS:  " << total_flops / 1e9 << " GFLOPS" << std::endl;
  std::cout << "  Total data:   " << total_data / 1e6 << " MB" << std::endl;

  // A100 峰值性能: 312 TFLOPS (FP16 Tensor Core)
  double peak_tflops = 312.0;
  double estimated_time_ms = (total_flops / (peak_tflops * 1e12)) * 1000.0;
  std::cout << "  A100 peak:    " << peak_tflops << " TFLOPS" << std::endl;
  std::cout << "  Est. time:    " << estimated_time_ms << " ms (roofline)" << std::endl;

  // =========================================================================
  // Step 10: 使用 GemmSelector 进行运行时 Dispatch
  // =========================================================================
  //
  // 这是真实 CUTLASS 的使用方式: 运行时 gemm 调用 →
  // GemmSelector 在编译期 kernel 池中找最匹配的。

  std::cout << "\n=== Runtime Dispatch via GemmSelector ===" << std::endl;

  // 伪代码: 实际运行时数据在 GPU 上
  // float *d_A, *d_B, *d_C;
  // cudaMalloc(&d_A, M * K * sizeof(float));
  // cudaMalloc(&d_B, K * N * sizeof(float));
  // cudaMalloc(&d_C, M * N * sizeof(float));

  dispatch::DefaultSelector::select_and_launch(
      ArchTag::compute_capability,  // 运行时 arch
      0,                            // layout_a (0=RowMajor)
      1,                            // layout_b (1=ColumnMajor)
      (float*)nullptr,             // d_A (占位)
      (float*)nullptr,             // d_B (占位)
      (float*)nullptr,             // d_C (占位)
      M, N, K
  );

  std::cout << "\n================================================================" << std::endl;
  std::cout << "  Demo complete. All configuration resolved at compile-time." << std::endl;
  std::cout << "================================================================\n" << std::endl;

  // =========================================================================
  // 附加展示: TypeList 编译期算法
  // =========================================================================
  //
  // 证明 TypeList 在编译期工作的威力

  std::cout << "=== TypeList Compile-time Demo ===" << std::endl;

  using DtypeList = TypeList<float, float, int32_t, double>;
  using ArchList = TypeList<Sm70, Sm75, Sm80, Sm90>;

  std::cout << "  DtypeList size:          " << DtypeList::size << std::endl;
  std::cout << "  ArchList size:           " << ArchList::size << std::endl;
  std::cout << "  Contains int32_t?        " << (contains_v<int32_t, DtypeList> ? "yes" : "no") << std::endl;
  std::cout << "  Contains half?           " << (contains_v<float, DtypeList> ? "yes" : "no") << std::endl;
  std::cout << "  Index of Sm80:           " << index_of_v<Sm80, ArchList> << std::endl;

  return 0;
}
