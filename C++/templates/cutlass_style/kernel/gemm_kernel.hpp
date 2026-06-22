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
// GemmKernel<Traits> - 完整的 GEMM CUDA Kernel 骨架 (伪代码)
// ============================================================================
//
// ┌──────────────────────────────────────────────────────────────────┐
// │  Mermaid 图: GemmKernel 执行流程                                 │
// │                                                                  │
// │  ┌──────────────────────────────────────────────────────┐       │
// │  │                    GEMM Kernel                        │       │
// │  │                                                      │       │
// │  │  1. 线程映射: threadIdx → warp_id → tile 内位置      │       │
// │  │     ┌────────────────────────────────────┐           │       │
// │  │     │ warp_id = threadIdx.x / 32         │           │       │
// │  │     │ warp_m  = warp_id / kWarpCountN     │ ← 编译期 │       │
// │  │     │ warp_n  = warp_id % kWarpCountN     │   常量   │       │
// │  │     └────────────────────────────────────┘           │       │
// │  │                                                      │       │
// │  │  2. 主循环 (Mainloop):                               │       │
// │  │     for k_tile in range(0, K, TileK):               │       │
// │  │       ┌──────────────────────────────────┐          │       │
// │  │       │ Load A_tile: global → shared       │          │       │
// │  │       │ Load B_tile: global → shared       │ ← cp.async │    │
// │  │       │ __syncthreads()                   │          │       │
// │  │       │ for k in range(0, TileK, WarpK):  │          │       │
// │  │       │   Load A_warp: shared → registers │          │       │
// │  │       │   Load B_warp: shared → registers │          │       │
// │  │       │   for m,n in warp tile:           │          │       │
// │  │       │     MMA instruction              │ ← Tensor Core    │
// │  │       │   __syncthreads() (if needed)     │          │       │
// │  │       └──────────────────────────────────┘          │       │
// │  │                                                      │       │
// │  │  3. Epilogue:                                       │       │
// │  │     累加器 → 输出转换 (FP32 acc → FP16 output)      │       │
// │  │     可选: ReLU, Bias, Residual Add                   │       │
// │  │     Write C_warp: registers → shared → global       │       │
// └──────────────────────────────────────────────────────────────────┘
//
// WHY template<Traits> 而非运行时参数:
// - Traits::TileShape::M 是编译期常量 → 循环可完全展开
// - Traits::ArchTag → 生成正确的 PTX 指令 (mma.sync / wgmma)
// - Traits::kSharedMemorySize → 静态 shared memory 分配 (比动态快)
// - 所有偏移计算在编译期完成 → 零运行时分支

template <typename Traits_>
struct GemmKernel {
  using Traits = Traits_;

  using ElementA = typename Traits::ElementA;
  using ElementB = typename Traits::ElementB;
  using ElementC = typename Traits::ElementC;
  using AccumulatorType = typename Traits::AccumulatorType;

  using LayoutA = typename Traits::LayoutA;
  using LayoutB = typename Traits::LayoutB;
  using LayoutC = typename Traits::LayoutC;

  using TileShape = typename Traits::TileShape;
  using WarpShape = typename Traits::WarpShape;
  using InstructionShape = typename Traits::InstructionShape;

  // =========================================================================
  // CUDA Device Function (伪代码 - 不可直接编译，需 CUDA 工具链)
  // =========================================================================
  //
  // 模板展开后: 编译器为每种 Traits 组合生成一份独立的 PTX 代码。
  //   例如: GemmKernel<Sm80Fp16NN128x128> 和 GemmKernel<Sm80Bf16NN128x128>
  //   是完全独立的两个 kernel 函数，尽管源代码只有一份模板。
  //
  // NVIDIA 工程师这样设计的原因:
  //   1. 代码复用: 一份模板覆盖所有变体
  //   2. 零运行时开销: 每个实例化都是针对具体配置优化的
  //   3. 编译期验证: static_assert 在编译期捕获错误配置
  //   4. 编译器优化: 编译器可以看到所有常量，做激进优化

  __device__ static void kernel_impl(
      ElementA* d_A,
      ElementB* d_B,
      ElementC* d_C,
      int M, int N, int K,
      int lda, int ldb, int ldc) {

    // ── Step 1: 线程映射 ──
    // 每个线程计算自己在 block 中的唯一 ID
    //
    // 编译期常量 Traits::kThreads = 128 (例如)
    // 编译器展开: 每个线程的 thread_id 就是 threadIdx.x
    //   没有运行时除法——warp_id 在硬件层面自动提供

    int thread_id = 0; // 实际: threadIdx.x + threadIdx.y * blockDim.x
    int warp_id = 0;   // 实际: thread_id / 32 (硬件支持，单周期)
    int lane_id = 0;   // 实际: thread_id % 32

    // 当前 block 在 M×N grid 中的位置
    int block_m = 0; // 实际: blockIdx.x
    int block_n = 0; // 实际: blockIdx.y

    // ── Step 2: 当前 tile 在全局矩阵中的基地址 ──
    //
    // 模板展开后 (TileShape<128,128,32>):
    //   tile_offset_a_row = block_m * 128  ← 移位而非乘法 (128 = 1<<7)
    //   tile_offset_a_col = k_tile * 32     ← 同上

    int tile_row = block_m * TileShape::M;
    int tile_col = block_n * TileShape::N;

    // ── Step 3: 累加器片段 (每个线程独立持有) ──
    //
    // WHY 每个线程持有累加器片段:
    //   Tensor Core 的 MMA 指令操作 "fragment"——每个线程持有
    //   一部分计算结果（类比: 分布式计算中每个节点持有部分结果）。
    //   fragment 以寄存器存储，速度最快。
    //
    // 模板展开后: 累加器寄存器数量 = (WarpM * WarpN) / 32
    //   对于 WarpShape<64,64,32>: 64*64/32 = 128 个累加器寄存器
    //   每个 float 累加器占 1 个寄存器 → 128 registers
    //   加上 A/B 片段寄存器 ≈ 160 registers (在 255 预算内)

    AccumulatorType accumulator[128] = {0}; // 伪代码; 实际大小编译期确定

    // ── Step 4: Shared Memory 声明 ──
    //
    // WHY 编译期大小:
    //   CUDA 的 __shared__ 必须声明编译期已知大小。
    //   动态 shared memory (extern __shared__) 访问更慢，
    //   因为编译器无法计算偏移和做 bank conflict 分析。
    //
    // 模板展开后: __shared__ half smem_A[128*32];  // 8KB
    //             __shared__ half smem_B[128*32];  // 8KB
    //             → 总计 16KB shared memory (在 164KB A100 限制内)

    // 注: 实际声明为 __shared__
    // ElementA smem_A[Traits::kSharedMemorySize / (2 * sizeof(ElementA))];
    // ElementB smem_B[...];

    // ── Step 5: Mainloop - K 维度分块迭代 ──
    //
    // 模板展开后 (TileShape<128,128,32>):
    //   外层: for k_tile in 0..K step 32   ← 用户 K 维度 / 32 轮
    //   内层: 32 / WarpK 次 warp 迭代       ← 编译期常量

    for (int k_tile = 0; k_tile < K; k_tile += TileShape::K) {

      // ── 5a: 从 Global Memory 加载到 Shared Memory ──
      // SM80+ 使用 cp.async，SM70/SM75 使用普通 ld.global
      //
      // 模板展开后 (如果是 SM80):
      //   asm volatile("cp.async.ca.shared.global [%0], [%1], %2;"
      //                :: "r"(smem_ptr), "l"(global_ptr), "n"(16));
      //   → cp.async 是异步指令，不阻塞线程，允许 overlap 计算和加载
      //
      // 模板展开后 (如果是 SM70):
      //   asm volatile("ld.global.ca.f16 %0, [%1];"
      //                :: "h"(reg), "l"(global_ptr));
      //   → 普通 load，阻塞式

      // 伪代码:
      // load_tile_A_to_smem<Traits>(d_A, smem_A, k_tile, lda);
      // load_tile_B_to_smem<Traits>(d_B, smem_B, k_tile, ldb);

      // cp.async 需要 commit 和 wait group
      // asm volatile("cp.async.commit_group;");

      // ── 5b: 同步 (确保 shared memory 数据就绪) ──
      // SM80+: cp.async.wait_group 0 (比 __syncthreads 更高效)
      // SM70/75: __syncthreads()
      //
      // 模板展开后:
      //   SM80: cp.async.wait_group 0;  ← 只等待异步拷贝完成
      //   SM70: __syncthreads();         ← 全局 barrier

      // ── 5c: Warp 级计算 - 从 Shared Memory 加载到寄存器并 MMA ──
      // 这层循环在编译期完全展开 (K warp 迭代 = 32/32 = 1 次)
      for (int k_warp = 0; k_warp < TileShape::K; k_warp += WarpShape::K) {

        // 从 shared memory 加载 A/B 片段到寄存器
        // load_warp_A<Traits>(smem_A, reg_A_fragment, warp_id, lane_id);
        // load_warp_B<Traits>(smem_B, reg_B_fragment, warp_id, lane_id);

        // ── MMA 指令 ──
        // 这层循环在编译期完全展开
        //
        // 模板展开后 (InstructionShape<16,8,16>):
        //   for m in 0..WarpM step 16:
        //     for n in 0..WarpN step 8:
        //       for k in 0..WarpK step 16:
        //         asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
        //              {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        //             : "=f"(c0),"=f"(c1),...
        //             : "r"(a0),"r"(a1),..., "r"(b0),...,
        //               "f"(c0),"f"(c1),...);
        //
        //   这条 PTX 指令在 1 个 GPU 周期内完成 16×8×16 = 2048 次 FMA 操作！
        //   对比 SIMT FMA: 同期只能完成 1 次 FMA。2048 倍吞吐差异。
        for (int m = 0; m < WarpShape::M; m += InstructionShape::M) {
          for (int n = 0; n < WarpShape::N; n += InstructionShape::N) {
            for (int k = 0; k < WarpShape::K; k += InstructionShape::K) {
              // MMA instruction goes here
              // PTX inline assembly for Tensor Core
            }
          }
        }
      }

      // SM70/75: __syncthreads() 防止 shared memory 被下一轮覆盖
    }

    // ── Step 6: Epilogue - 写回 Global Memory ──
    //
    // 累加器通常是 FP32 (精度)，但输出可能需要 FP16/BF16。
    // 编译期决定是否需要类型转换。
    //
    // 模板展开后:
    //   if constexpr (sizeof(ElementC) == 2) {
    //     // FP32 acc → FP16 output: cvt instruction
    //     asm("cvt.rn.f16.f32 %0, %1;" : "=h"(out) : "f"(acc));
    //   } else {
    //     // 直接写 FP32: st.global
    //     out = acc;
    //   }

    // store_tile_C_to_global<Traits>(d_C, accumulator, ldc);
  }

  // =========================================================================
  // Host 端 launch wrapper (简化版)
  // =========================================================================
  static void launch(
      ElementA* d_A, ElementB* d_B, ElementC* d_C,
      int M, int N, int K,
      int lda, int ldb, int ldc) {

    // 计算 grid 和 block 维度
    // 模板展开: grid 计算是纯编译期算术
    //   gridDim.x = (M + 127) / 128  (当 TileM=128 时)
    //   gridDim.y = (N + 127) / 128
    //
    // 如果 M=1024, N=1024: grid=(8,8), block=128 → 64 blocks
    // 每个 block 128 线程 → 8192 线程并发

    int grid_m = (M + TileShape::M - 1) / TileShape::M;
    int grid_n = (N + TileShape::N - 1) / TileShape::N;

    // dim3 grid(grid_m, grid_n);
    // dim3 block(Traits::kThreads);

    // kernel_impl<<<grid, block, Traits::kSharedMemorySize>>>(
    //     d_A, d_B, d_C, M, N, K, lda, ldb, ldc);
  }
};

// ============================================================================
// 编译期 Kernel 大小分析 (调试用)
// ============================================================================
template <typename Kernel>
struct KernelSizeAnalyzer {
  // 估算 PTX 指令数 (粗略)
  static constexpr int estimated_instructions =
      Kernel::Traits::kMmaIterationsTotal * 3 +  // 3 instr per MMA (load, compute, store)
      Kernel::Traits::kMainloopIterationsPerTile * 10; // load/store overhead

  // 估算寄存器使用数
  static constexpr int estimated_registers =
      (Kernel::Traits::WarpShape::M * Kernel::Traits::WarpShape::N) / 32 + // acc
      (Kernel::Traits::WarpShape::M * Kernel::Traits::WarpShape::K) / 32 + // A fragment
      (Kernel::Traits::WarpShape::N * Kernel::Traits::WarpShape::K) / 32 + // B fragment
      8; // overhead

  static constexpr bool might_spill =
      estimated_registers > Kernel::Traits::ArchTag::max_registers_per_thread;
};

} // namespace kernel
} // namespace cutlass_style
