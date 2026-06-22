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
// Mainloop<Traits> - GEMM 主循环 (Global→Shared Memory 加载)
// ============================================================================
//
// ┌──────────────────────────────────────────────────────────────────┐
// │  Mermaid 图: Mainloop 的软件流水线                               │
// │                                                                  │
// │  Time ──────────────────────────────────────────────────────▶    │
// │                                                                  │
// │  Iteration 0        Iteration 1        Iteration 2               │
// │  ┌────────────┐    ┌────────────┐    ┌────────────┐              │
// │  │ Load Tile 0 │    │ Load Tile 1 │    │ Load Tile 2 │ (global→smem)│
// │  └──────┬─────┘    └──────┬─────┘    └──────┬─────┘              │
// │         │ wait            │ wait            │ wait               │
// │         ▼                 ▼                 ▼                    │
// │  ┌────────────┐    ┌────────────┐    ┌────────────┐              │
// │  │            │    │ MMA Tile 0 │    │ MMA Tile 1 │ (smem→reg→MMA)│
// │  └────────────┘    └────────────┘    └────────────┘              │
// │                                                                  │
// │  双缓冲 (Double Buffering):                                      │
// │  ┌────────────┐    ┌────────────┐                                │
// │  │ Load T0 → S0│    │ MMA T0 ← S0 │   ← Load 和 Compute 交替    │
// │  │            │    │ Load T1 → S1 │   ← 使用不同 smem buffer     │
// │  └────────────┘    │ MMA T1 ← S1 │                              │
// │                    └────────────┘                                │
// └──────────────────────────────────────────────────────────────────┘
//
// WHY 软件流水线 (Software Pipelining):
//   GPU 的内存延迟非常高 (~200-800 cycles for global memory)。
//   如果每次 load 后等数据就绪再做 MMA，流水线会大量空等 (stall)。
//   双缓冲让 load 和 compute 重叠: load tile N+1 的同时 compute tile N。
//   这利用了 GPU 的独立 Load/Store 单元和计算单元的并行性。
//
// 类比: 餐厅厨房的流水线
//   厨师 (Compute) 炒菜同时，助手 (Load) 准备下一道菜的原料。
//   如果厨师必须自己备菜，做完一道才能开始下一道——效率极低。

template <typename Traits_>
struct Mainloop {
  using Traits = Traits_;
  using ElementA = typename Traits::ElementA;
  using ElementB = typename Traits::ElementB;

  using TileShape = typename Traits::TileShape;
  using WarpShape = typename Traits::WarpShape;

  // =========================================================================
  // 编译期常量
  // =========================================================================
  static constexpr int kTileM = TileShape::M;
  static constexpr int kTileN = TileShape::N;
  static constexpr int kTileK = TileShape::K;

  static constexpr int kWarpM = WarpShape::M;
  static constexpr int kWarpN = WarpShape::N;
  static constexpr int kWarpK = WarpShape::K;

  // 每个 Block 中的 Warp 数量
  static constexpr int kWarpCountM = kTileM / kWarpM;
  static constexpr int kWarpCountN = kTileN / kWarpN;
  static constexpr int kWarpCount = kWarpCountM * kWarpCountN;

  // =========================================================================
  // Global → Shared Memory 加载策略
  // =========================================================================
  //
  // WHY 策略模式: 不同架构有不同的加载能力。
  //   SM70/75: 每个线程一次加载 4/8/16 bytes (ld.global.v4.f16)
  //   SM80+: cp.async 一次加载 16 bytes (ld.global.v4.f16 或 cp.async.ca)
  //   SM90+: TMA (Tensor Memory Accelerator) 批量加载整个 tile

  // 共享内存缓冲区大小 (A + B)
  static constexpr int kSmemSizeA = kTileM * kTileK;
  static constexpr int kSmemSizeB = kTileN * kTileK;

  // 每个线程加载的元素数 (对 A 矩阵)
  // 将 kTileM * kTileK 个元素平均分配给所有线程
  static constexpr int kLoadsPerThreadA =
      (kSmemSizeA + Traits::kThreads - 1) / Traits::kThreads;

  // 每个线程加载的元素数 (对 B 矩阵)
  static constexpr int kLoadsPerThreadB =
      (kSmemSizeB + Traits::kThreads - 1) / Traits::kThreads;

  // =========================================================================
  // 迭代次数
  // =========================================================================
  //
  // 每次 Mainloop 迭代处理一个 K-tile (= kTileK 列/行)
  // 需要的迭代次数 = 总 K 维度 / kTileK
  // 模板展开后: 每个迭代处理的偏移是编译期常量

  // =========================================================================
  // 加载函数: Global Memory → Shared Memory (SM70/75 风格)
  // =========================================================================
  //
  // WHY 每个线程负责加载分散的元素:
  //   A 是 RowMajor, B 是 ColumnMajor。
  //   线程 [threadIdx.x] 加载 A[row][k + k_offset] 和 B[col][k + k_offset]
  //   其中 row, col, k_offset 由线程 ID 计算。
  //   这种"交错"加载模式最大化内存带宽利用率。
  //
  // 模板展开后:
  //   load_tile_A_sm70("RowMajor"): 每个线程加载连续元素，合并访问
  //     ld.global.v4.f16 {%r0,%r1,%r2,%r3}, [%ptr];  ← 128-bit 合并访问
  //     st.shared.v4.f16 [%smem], {%r0,%r1,%r2,%r3};
  //
  //   load_tile_B_sm70("ColumnMajor"): B 是列优先，不能用合并访问
  //     需要使用 ld.global.nc (non-coherent) 或 stride 加载
  //     这就是为什么 CUTLASS 偏好 RowMajor×ColumnMajor 组合

  // ── SM70/SM75: 传统加载 (每个线程 128-bit 向量化加载) ──
  __device__ static void load_tile_A_sm70(
      const ElementA* global_A,
      ElementA* smem_A,
      int k_tile,
      int lda,
      int warp_id,
      int lane_id) {

    // 伪代码骨架
    //
    // 每个线程加载 kLoadsPerThreadA 个元素
    // for (int i = 0; i < kLoadsPerThreadA; i++) {
    //   // 计算 global memory 地址 (线程 ID 交错映射)
    //   int row = thread_id / kTileK;
    //   int col = k_tile + (thread_id % kTileK);
    //
    //   // 128-bit 向量化加载 (4 个 half / 2 个 float)
    //   uint4 loaded = *reinterpret_cast<const uint4*>(&global_A[row * lda + col]);
    //
    //   // 写入 shared memory (可能需要 swizzle 避免 bank conflict)
    //   smem_A[row * kTileK + col] = loaded;
    // }
  }

  // ── SM80/SM90: cp.async 异步加载 ──
  __device__ static void load_tile_A_sm80(
      const ElementA* global_A,
      ElementA* smem_A,
      int k_tile,
      int lda,
      int warp_id,
      int lane_id) {

    // 伪代码骨架
    //
    // SM80+ 使用 cp.async 指令 (异步拷贝)
    // cp.async 不阻塞线程，拷贝在后台完成
    // 之后用 cp.async.wait_group 等待完成
    //
    // for (int i = 0; i < kLoadsPerThreadA; i++) {
    //   asm volatile(
    //     "cp.async.ca.shared.global [%0], [%1], %2;\n"
    //     :: "r"(smem_addr), "l"(global_addr), "n"(16)
    //   );
    // }
    //
    // // 提交这组异步拷贝
    // asm volatile("cp.async.commit_group;\n");
  }

  // =========================================================================
  // Shared Memory → 寄存器的加载 (Warp 级别)
  // =========================================================================
  //
  // WHY 从 shared memory 再加载一次到寄存器:
  //   Tensor Core 的 MMA 指令要求操作数在寄存器中。
  //   Shared memory → Register 的加载利用 warp shuffle 和
  //   特殊的矩阵布局来避免 bank conflict。
  //
  // 模板展开后:
  //   每个 warp 独立计算它的 A_fragment (WarpM × WarpK) 和
  //   B_fragment (WarpN × WarpK)。
  //   加载模式由 Policy 的线程布局决定。

  __device__ static void load_warp_fragment_from_smem(
      const ElementA* smem_A,
      const ElementB* smem_B,
      ElementA* reg_A_frag,
      ElementB* reg_B_frag,
      int warp_m, int warp_n,
      int k_warp,
      int lane_id) {

    // 伪代码骨架
    //
    // // A fragment: 每个线程持有 WarpM/WarpThreadsM × WarpK 个元素
    // int m_in_warp = warp_m * kWarpM + (lane_id / 4);   // 行方向
    // int k_offset  = k_warp * kWarpK + (lane_id % 4);   // K 方向
    //
    // for (int k = 0; k < kWarpK; k += 4) {
    //   reg_A_frag[k/4] = smem_A[m_in_warp * kTileK + k_offset + k];
    // }
    //
    // // B fragment: 类似，但是 N 方向
    // int n_in_warp = warp_n * kWarpN + (lane_id / 4);
    // // ... load B ...

    // 用 ldsm (load matrix from shared) 指令 (SM75+)
    // 自动处理 matrix layout 的 swizzle
    // asm volatile("ldsm ...");
  }

  // =========================================================================
  // 完整 Mainloop 迭代控制
  // =========================================================================
  //
  // 模板展开后 (以 kTileK=32, kWarpK=32 为例):
  //   每次 tile 迭代只需 1 次 warp 迭代 (32/32=1)
  //   编译器可以完全展开这个嵌套循环

  __device__ static void iterate(
      const ElementA* global_A,
      const ElementB* global_B,
      ElementA* smem_A,
      ElementB* smem_B,
      ElementA* reg_A,
      ElementB* reg_B,
      int M, int N, int K,
      int lda, int ldb,
      int block_m, int block_n,
      int warp_id, int lane_id) {

    // 当前 warp 在 block 中的位置 (编译期计算)
    int warp_m = (warp_id / kWarpCountN) % kWarpCountM;
    int warp_n = warp_id % kWarpCountN;

    // K 维度 tile 迭代
    // 模板展开: 编译器知道 kTileK，可以预计算迭代次数
    for (int k_tile = 0; k_tile < K; k_tile += kTileK) {

      // Stage 1: 根据架构选择加载方式
      if constexpr (Traits::ArchTag::compute_capability >= 80) {
        // SM80+: 用 cp.async
        load_tile_A_sm80(global_A, smem_A, k_tile, lda, warp_id, lane_id);
        load_tile_A_sm80(
            reinterpret_cast<const ElementA*>(global_B),
            reinterpret_cast<ElementA*>(smem_B), // 伪代码
            k_tile, ldb, warp_id, lane_id);
        // asm("cp.async.commit_group;");
      } else {
        // SM70/75: 传统加载
        load_tile_A_sm70(global_A, smem_A, k_tile, lda, warp_id, lane_id);
        __syncthreads();
      }

      // Stage 2: Warp 级计算 (从 shared memory 加载到寄存器)
      for (int k_warp = 0; k_warp < kTileK; k_warp += kWarpK) {
        load_warp_fragment_from_smem(
            smem_A, smem_B, reg_A, reg_B,
            warp_m, warp_n, k_warp / kWarpK, lane_id);
      }
    }
  }
};

} // namespace kernel
} // namespace cutlass_style
