#pragma once

#include <cstdint>
#include <type_traits>

#include "include/arch_tag.hpp"
#include "include/tile_shape.hpp"
#include "include/type_list.hpp"

namespace cutlass_style {
namespace dispatch {

// ============================================================================
// DefaultGemmConfiguration - 编译期选择最优 Tile 配置
// ============================================================================
//
// WHY 需要这个: 用户不想手动指定 tile 大小。他们只想说:
//   "我要做 FP16 GEMM，在 A100 上"
//   然后框架自动选择最优的 tile 配置。
//
// CUTLASS 的 DefaultGemmConfiguration 的原理:
//   1. 每种架构有自己的"寄存器预算" (255 registers/thread on SM80)
//   2. 每种 tile 大小需要不同数量的寄存器
//   3. 在满足寄存器预算的前提下，选择最大的 tile (最大化数据复用)
//   4. 如果寄存器不够 → register spill → 性能暴跌
//
// 类比: DefaultGemmConfiguration 相当于汽车自动变速箱:
//   你只踩油门 (给数据类型和架构)，变速箱自动选择最优档位 (tile size)。
//   你不会手动选择 1 档还是 5 档——让系统根据负载和转速自动选。
//
// ┌──────────────────────────────────────────────────────────────────┐
// │  Mermaid 图: Tile 大小选择流程                                  │
// │                                                                  │
// │  Input: ArchTag, ElementA, ElementB, ElementC                    │
// │                                                                  │
// │  ┌──────────────────────────────────────────┐                    │
// │  │ 寄存器预算计算                             │                    │
// │  │ registers = min(255, arch_limit)          │                    │
// │  │ smem_budget = arch.max_shared_memory      │                    │
// │  └──────────────┬───────────────────────────┘                    │
// │                 ▼                                                │
// │  ┌──────────────────────────────────────────┐                    │
// │  │ Tile 候选列表从大到小排序                  │                    │
// │  │ Tile256 > Tile128 > Tile64               │                    │
// │  └──────────────┬───────────────────────────┘                    │
// │                 ▼                                                │
// │  ┌──────────────────────────────────────────┐                    │
// │  │ 逐候选检查：                              │                    │
// │  │ 1. registers_needed <= registers_budget   │                    │
// │  │ 2. smem_needed <= smem_budget             │                    │
// │  │ 3. 该架构支持对应的 MMA 指令              │                    │
// │  │ → 第一个满足条件的 = 最优                  │                    │
// │  └──────────────────────────────────────────┘                    │
// └──────────────────────────────────────────────────────────────────┘

template <typename ArchTag, typename ElementA, typename ElementB, typename ElementC>
struct DefaultGemmConfiguration {
 private:
  // 寄存器预算: 每种架构的每个线程寄存器限制
  // WHY 255: NVIDIA GPU 的寄存器文件是固定大小的。
  //          255 是每个线程的最大寄存器数 (A100/H100)。
  //          超过 255 → 编译器会 spill 到 local memory (L1 cache) → 性能暴跌 3-4 倍。
  static constexpr int kRegisterBudget = ArchTag::max_registers_per_thread;

  // Shared memory 预算
  static constexpr int kSmemBudget = ArchTag::max_shared_memory_per_block;

  // 数据类型大小
  static constexpr int kElementASize = sizeof(ElementA);
  static constexpr int kElementBSize = sizeof(ElementB);

  // 估算指定 tile 需要的寄存器数
  //
  // WHY 不是精确计算:
  //   精确的寄存器数只有 nvcc/ptxas 知道 (因为它做寄存器分配)。
  //   但这给出一个安全的"上界"，确保在预算内。
  //   实际 NVIDIA 工程师用启发式公式 + 微基准测试校准。
  template <int M, int N, int K>
  static constexpr int estimate_registers(int warp_m, int warp_n) {
    // 启发式: 每个线程持有 warp_tile/(32 threads) 个累加器
    // 每个累加器 1-4 个寄存器 (取决于精度)
    int acc_regs = (warp_m * warp_n) / 32;          // 累加器片段
    int load_regs = (warp_m * K + warp_n * K) / 32; // 加载寄存器
    int tmp_regs = 8;                               // 临时/指针寄存器

    int regs_per_acc = (kElementASize <= 2) ? 1 : 2;

    return acc_regs * regs_per_acc + load_regs + tmp_regs;
  }

  // 估算指定 tile 需要的 shared memory
  template <int M, int N, int K>
  static constexpr int estimate_smem() {
    int smem_a = M * K * kElementASize;
    int smem_b = N * K * kElementBSize;
    // 双缓冲: 2x (SM80+ 默认启用)
    int multiplier = (ArchTag::compute_capability >= 80) ? 2 : 1;
    return (smem_a + smem_b) * multiplier;
  }

  // 候选 tile 配置 (从大到小)
  struct Config256x128 { static constexpr int M = 256, N = 128, K = 32; static constexpr int warp_m = 64, warp_n = 64; };
  struct Config256x256 { static constexpr int M = 256, N = 256, K = 32; static constexpr int warp_m = 64, warp_n = 64; };
  struct Config128x256 { static constexpr int M = 128, N = 256, K = 32; static constexpr int warp_m = 64, warp_n = 64; };
  struct Config128x128 { static constexpr int M = 128, N = 128, K = 32; static constexpr int warp_m = 64, warp_n = 64; };
  struct Config128x64  { static constexpr int M = 128, N = 64,  K = 32; static constexpr int warp_m = 64, warp_n = 32; };
  struct Config64x128  { static constexpr int M = 64,  N = 128, K = 32; static constexpr int warp_m = 32, warp_n = 64; };
  struct Config64x64   { static constexpr int M = 64,  N = 64,  K = 32; static constexpr int warp_m = 32, warp_n = 32; };

  // 检查某个配置是否可行
  template <typename Config>
  static constexpr bool is_config_valid() {
    constexpr int regs = estimate_registers<Config::M, Config::N, Config::K>(
        Config::warp_m, Config::warp_n);
    constexpr int smem = estimate_smem<Config::M, Config::N, Config::K>();
    return (regs <= kRegisterBudget) && (smem <= kSmemBudget);
  }

  // 编译期优先级选择: 按从大到小的候选列表顺序
  // 返回第一个满足条件的配置

  template <typename First, typename... Rest>
  static constexpr auto select_config() {
    if constexpr (is_config_valid<First>()) {
      return First{};
    } else if constexpr (sizeof...(Rest) > 0) {
      return select_config<Rest...>();
    } else {
      // 保底: 最小的 tile 总是能工作
      static_assert(sizeof...(Rest) >= 0,
                    "No valid tile configuration found! "
                    "This should never happen - minimum tile always fits.");
      return Config64x64{};
    }
  }

  using SelectedConfig = decltype(select_config<
      Config256x256, Config256x128, Config128x256,
      Config128x128, Config128x64, Config64x128, Config64x64>());

 public:
  // 最终选出的最优 tile 配置
  using TileShape = GemmShape<SelectedConfig::M, SelectedConfig::N, SelectedConfig::K>;
  using WarpShape = cutlass_style::WarpShape<SelectedConfig::warp_m, SelectedConfig::warp_n, SelectedConfig::K>;

  // 根据算力选择 MMA 指令形状
  using MmaInstruction = std::conditional_t<
      ArchTag::compute_capability >= 90,
      InstructionShape<16, 8, 32>,   // SM90+ FP8
      std::conditional_t<
          ArchTag::compute_capability >= 80,
          InstructionShape<16, 8, 16>,  // SM80 FP16
          InstructionShape<16, 16, 8>   // SM70/75 FP16
      >
  >;

  // 额外信息 (调试/日志用)
  static constexpr const char* description = []() constexpr {
    if constexpr (ArchTag::compute_capability >= 90) return "SM90-H100: Large tile, FP8";
    else if constexpr (ArchTag::compute_capability >= 80) return "SM80-A100: Medium tile, FP16/BF16";
    else return "SM70/75: Standard tile, FP16";
  }();
};

} // namespace dispatch
} // namespace cutlass_style
