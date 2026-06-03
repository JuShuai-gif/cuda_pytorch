// =============================================================================
// 第 03.2 章：编译期内核配置
//
// 在 GPU 编程（CUDA/HIP）中，内核启动参数通常使用非类型模板参数在编译期确定。
// 本文件模拟：
//   1. 通过编译期常量选择 tile 大小
//   2. 块维度作为非类型参数
//   3. 从问题大小计算 grid 维度
//   4. 在编译期计算的共享内存分配
//   5. 通过 constexpr 进行占用率估算
//   6. 在编译期比较多种启动配置
//
// 编译：g++ -std=c++20 -o 02_compile_time_config 02_compile_time_config.cpp
// =============================================================================

#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <type_traits>

// =============================================================================
// 1. 内核配置结构
// =============================================================================
// 将所有启动参数编码为编译期常量。在真实的 CUDA 代码中，
// 这些作为模板实参传递给内核函数。

template <int BlockDimX_,   // 每块的 X 方向线程数
          int BlockDimY_,   // 每块的 Y 方向线程数
          int TileM_,       // M 维度的 tile 大小（每块）
          int TileN_,       // N 维度的 tile 大小（每块）
          int TileK_,       // K 维度的 tile 大小（每块）
          int NumStages_ = 2>   // 流水线阶段数（共享内存双缓冲）
struct KernelConfig {
  // --- 块 / Grid ---
  static constexpr int block_dim_x     = BlockDimX_;
  static constexpr int block_dim_y     = BlockDimY_;
  static constexpr int threads_per_block = BlockDimX_ * BlockDimY_;

  // --- Tile ---
  static constexpr int tile_M = TileM_;
  static constexpr int tile_N = TileN_;
  static constexpr int tile_K = TileK_;
  static constexpr int elements_per_tile = TileM_ * TileN_;

  // --- 流水线 ---
  static constexpr int num_stages = NumStages_;

  // --- 共享内存估算 ---
  // 对于 GEMM：需要 A tile（MxK）+ B tile（KxN）的共享内存，
  // 可能是双缓冲的。假设为 float（4 字节）。
  static constexpr int smem_per_stage =
      (TileM_ * TileK_ + TileK_ * TileN_) * 4;  // 字节
  static constexpr int total_smem = smem_per_stage * NumStages_;

  // --- 占用率估算 ---
  // 现代 GPU 上每个 SM 的最大线程数 ~ 2048，最大共享内存 ~ 48-100 KB。
  // 基于线程和共享内存限制估算每个 SM 的最大块数。
  static constexpr int max_threads_per_sm = 2048;
  static constexpr int max_smem_per_sm    = 48 * 1024;  // 48 KB

  static constexpr int blocks_by_threads =
      max_threads_per_sm / threads_per_block;
  static constexpr int blocks_by_smem =
      (total_smem > 0) ? max_smem_per_sm / total_smem : blocks_by_threads;

  // 限制因素决定每个 SM 的最大并发块数
  static constexpr int max_blocks_per_sm =
      (blocks_by_threads < blocks_by_smem) ? blocks_by_threads
                                            : blocks_by_smem;
  static constexpr int max_warps_per_sm =
      max_blocks_per_sm * threads_per_block / 32;

  // --- 工具：为给定问题大小计算 grid 维度 ---
  static constexpr int grid_dim_x(int problem_M) {
    return (problem_M + TileM_ - 1) / TileM_;
  }
  static constexpr int grid_dim_y(int problem_N) {
    return (problem_N + TileN_ - 1) / TileN_;
  }

  // --- 美观打印 ---
  static void print() {
    std::cout << "  块：(" << block_dim_x << ", " << block_dim_y << ") = "
              << threads_per_block << " 线程" << std::endl;
    std::cout << "  Tile：（M=" << tile_M << ", N=" << tile_N << ", K="
              << tile_K << "）" << std::endl;
    std::cout << "  流水线阶段数：" << num_stages << std::endl;
    std::cout << "  共享内存 / 块：" << total_smem << " B（"
              << total_smem / 1024 << " KB）" << std::endl;
    std::cout << "  最大块数/SM：" << max_blocks_per_sm
              << "（线程受限=" << blocks_by_threads
              << "，共享内存受限=" << blocks_by_smem << "）" << std::endl;
    std::cout << "  最大 warp 数/SM：" << max_warps_per_sm << std::endl;
  }
};

// =============================================================================
// 2. 问题描述符（运行时 + 编译期混合）
// =============================================================================
// 真正的问题具有运行时大小，但内核配置在编译期固定。
// 此结构混合这两者。

template <typename Config>
struct ProblemDescriptor {
  int problem_M;
  int problem_N;
  int problem_K;

  // 在编译期从 Config 推导
  static constexpr int tile_M = Config::tile_M;
  static constexpr int tile_N = Config::tile_N;

  // 为此问题实例计算 grid 维度
  int grid_x() const {
    return (problem_M + tile_M - 1) / tile_M;
  }
  int grid_y() const {
    return (problem_N + tile_N - 1) / tile_N;
  }

  void print() const {
    std::cout << "问题：M=" << problem_M << ", N=" << problem_N
              << ", K=" << problem_K << std::endl;
    std::cout << "  Grid：(" << grid_x() << ", " << grid_y() << ") 块"
              << std::endl;
    std::cout << "  总块数：" << grid_x() * grid_y() << std::endl;
  }
};

// =============================================================================
// 3. 编译期配置选择（为问题大小选择最优 tile）
// =============================================================================
// 给定问题大小，选择最合适的编译期配置。
// 这就是自动调优器的工作：生成配置列表并进行分发。

template <int M, int N, int K>
struct SelectKernelConfig {
  // 简单启发式：选择接近问题维度的 tile 大小，
  // 但受硬件限制约束。

  // 对于小问题：使用较小的 tile 以避免浪费
  static constexpr int chosen_tile_M = (M < 32)  ? 32  :
                                       (M < 128) ? 64  : 128;
  static constexpr int chosen_tile_N = (N < 32)  ? 32  :
                                       (N < 128) ? 64  : 128;
  static constexpr int chosen_tile_K = (K < 32)  ? 16  :
                                       (K < 128) ? 32  : 64;

  static constexpr int chosen_block_x = 16;
  static constexpr int raw_block_y = (chosen_tile_M * chosen_tile_N) /
                                      (128 * chosen_block_x);
  static constexpr int chosen_block_y = (raw_block_y < 1) ? 1 : raw_block_y;

  using type = KernelConfig<
      chosen_block_x,
      chosen_block_y,
      chosen_tile_M,
      chosen_tile_N,
      chosen_tile_K,
      (chosen_tile_K >= 64 ? 2 : 1)  // 大 K tile 使用双缓冲
  >;
};

// =============================================================================
// 4. 模拟内核启动（无实际 CUDA）
// =============================================================================

template <typename Config>
class SimulatedGemmKernel {
 public:
  // 从 Config 推导的启动参数
  static constexpr int block_x = Config::block_dim_x;
  static constexpr int block_y = Config::block_dim_y;
  static constexpr int threads = Config::threads_per_block;

  void launch(ProblemDescriptor<Config> const& problem) const {
    std::cout << "\n--- 内核启动 ---" << std::endl;
    Config::print();
    problem.print();

    // 模拟：每个块计算其 tile
    int total_blocks = problem.grid_x() * problem.grid_y();
    int flops_per_block = Config::tile_M * Config::tile_N * Config::tile_K * 2;
    long long total_flops = static_cast<long long>(total_blocks) * flops_per_block;

    std::cout << "  模拟 FLOPs：" << total_flops << std::endl;
    std::cout << "  共享内存 / 块：" << Config::total_smem << " B" << std::endl;

    // 占用率
    std::cout << "  理论占用率：" << Config::max_warps_per_sm
              << "/每个 SM 64 warp" << std::endl;
  }
};

// =============================================================================
// 5. 编译期配置比较
// =============================================================================
// 比较两种配置并选择预测性能更好的
//（更高的占用率，更少的浪费线程）。

template <typename ConfigA, typename ConfigB>
struct CompareConfigs {
  static constexpr bool a_higher_occupancy =
      ConfigA::max_warps_per_sm > ConfigB::max_warps_per_sm;
  static constexpr bool a_more_threads =
      ConfigA::threads_per_block > ConfigB::threads_per_block;

  // 简单启发式：优先选择更高占用率
  static constexpr bool better = a_higher_occupancy || a_more_threads;

  using type = std::conditional_t<better, ConfigA, ConfigB>;

  static void report() {
    std::cout << "\n配置比较：" << std::endl;
    std::cout << "  配置 A 占用率：" << ConfigA::max_warps_per_sm
              << " warp/SM" << std::endl;
    std::cout << "  配置 B 占用率：" << ConfigB::max_warps_per_sm
              << " warp/SM" << std::endl;
    std::cout << "  胜出者：" << (better ? "A" : "B") << std::endl;
  }
};

// =============================================================================
// 6. 编译期 2 的幂和对齐检查
// =============================================================================
// GPU 内核要求特定的对齐和 2 的幂约束。

template <int Value>
struct PowerOfTwoChecker {
  static constexpr bool is_power_of_two = (Value > 0) && ((Value & (Value - 1)) == 0);
  static constexpr int next_power_of_two = []() constexpr {
    if constexpr (Value <= 0) return 1;
    int v = Value - 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
  }();
};

// =============================================================================
//                                   主函数
// =============================================================================

int main() {
  using namespace std;

  cout << "=== 第 03.2 章：编译期内核配置 ===\n" << endl;

  // --- 测试 1：手动配置 ---
  cout << "[测试 1] 手动 KernelConfig：" << endl;
  using Cfg1 = KernelConfig<16, 16,   // 块维度：16x16 = 256 线程
                            128, 128,  // tile M,N
                            32,        // tile K
                            2>;        // 2 阶段（双缓冲）
  Cfg1::print();

  static_assert(Cfg1::threads_per_block == 256);
  static_assert(Cfg1::tile_M == 128);
  static_assert(Cfg1::num_stages == 2);
  static_assert(Cfg1::total_smem > 0);

  // --- 测试 2：ProblemDescriptor ---
  cout << "\n[测试 2] ProblemDescriptor：" << endl;
  ProblemDescriptor<Cfg1> problem{1024, 1024, 512};
  problem.print();

  assert(problem.grid_x() == (1024 + 128 - 1) / 128);  // 8
  assert(problem.grid_y() == (1024 + 128 - 1) / 128);  // 8
  cout << "  Grid 维度：" << problem.grid_x() << "x" << problem.grid_y()
       << endl;

  // --- 测试 3：自动配置选择 ---
  cout << "\n[测试 3] 自动配置选择：" << endl;
  using AutoCfg = SelectKernelConfig<256, 256, 64>::type;
  AutoCfg::print();

  static_assert(AutoCfg::tile_M == 128);  // 256 >= 128
  static_assert(AutoCfg::tile_N == 128);
  static_assert(AutoCfg::tile_K == 32);   // 64 < 128
  static_assert(AutoCfg::num_stages == 1);

  using SmallCfg = SelectKernelConfig<16, 16, 32>::type;
  SmallCfg::print();
  static_assert(SmallCfg::tile_M == 32);   // 16 < 32
  static_assert(SmallCfg::tile_N == 32);

  // --- 测试 4：模拟内核启动 ---
  cout << "\n[测试 4] 模拟内核启动：" << endl;
  SimulatedGemmKernel<Cfg1> kernel;
  kernel.launch(problem);

  // --- 测试 5：配置比较 ---
  cout << "\n[测试 5] 配置比较（选择最佳）：" << endl;
  using OptCfg  = KernelConfig<8, 8, 64, 64, 64, 1>;    // 64 线程，低共享内存
  using BaseCfg = KernelConfig<32, 8, 128, 128, 128, 2>;  // 256 线程，高共享内存

  CompareConfigs<OptCfg, BaseCfg>::report();
  // OptCfg 应该能有更多块/SM，因为线程更少、共享内存更少
  static_assert(CompareConfigs<OptCfg, BaseCfg>::better);
  using BetterConfig = CompareConfigs<OptCfg, BaseCfg>::type;
  static_assert(std::is_same_v<BetterConfig, OptCfg>);

  // --- 测试 6：2 的幂检查 ---
  cout << "\n[测试 6] 2 的幂验证：" << endl;
  static_assert(PowerOfTwoChecker<128>::is_power_of_two);
  static_assert(!PowerOfTwoChecker<100>::is_power_of_two);
  static_assert(PowerOfTwoChecker<100>::next_power_of_two == 128);
  cout << "  next_power_of_two(100) = "
       << PowerOfTwoChecker<100>::next_power_of_two << endl;
  cout << "  next_power_of_two(50)  = "
       << PowerOfTwoChecker<50>::next_power_of_two << endl;

  // --- 测试 7：不同问题大小的多种配置 ---
  cout << "\n[测试 7] 各种问题大小的配置：" << endl;
  std::cout << "  问题 64x64x32：   " << std::flush;
  SelectKernelConfig<64, 64, 32>::type::print();
  std::cout << "  问题 256x128x128：" << std::flush;
  SelectKernelConfig<256, 128, 128>::type::print();
  std::cout << "  问题 1024x1024x512：" << std::flush;
  SelectKernelConfig<1024, 1024, 512>::type::print();

  // --- 测试 8：空/单位大小边界情况 ---
  cout << "\n[测试 8] 边界情况（单位 tile）：" << endl;
  using UnitCfg = KernelConfig<1, 1, 1, 1, 1, 1>;
  UnitCfg::print();
  static_assert(UnitCfg::threads_per_block == 1);
  static_assert(UnitCfg::total_smem == (1 * 1 + 1 * 1) * 4 * 1);

  cout << "\n所有测试通过！" << endl;
  return 0;
}
