// =============================================================================
// 第 08.4 章：CUTLASS 风格编译期操作
//
// CUTLASS 执行大量编译期计算来推导 kernel 参数：寄存器分配、
// 共享内存大小、线程映射和分块迭代次数。所有这些都避免了运行期开销。
//
// 本文模拟：
//   1. 每个线程和每个 warp 的寄存器分配估算
//   2. 通过编译期填充避免共享内存 bank 冲突
//   3. 分块大小计算（跨 warp 和线程分割）
//   4. 编译期 warp 调度（warp 级别矩阵乘法分块）
//   5. 流水线阶段数量优化
//   6. 占用率计算
//   7. Block 级别地址计算
//   8. CUTLASS 中使用的编译期整数数学工具
//
// 编译：g++ -std=c++20 -o 04_cutlass_compile_time_ops 04_cutlass_compile_time_ops.cpp
// =============================================================================

#include <array>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <string>
#include <type_traits>

// =============================================================================
// 1. 编译期数学工具（CUTLASS 平台辅助函数）
// =============================================================================

// 2 的幂检查
template <int V>
struct IsPowerOfTwo : std::bool_constant<(V > 0) && ((V & (V - 1)) == 0)> {};

template <int V>
inline constexpr bool is_power_of_two_v = IsPowerOfTwo<V>::value;

// 下一个 2 的幂（使用 constexpr 函数避免模板递归溢出）
constexpr int next_power_of_two_ce(int V) {
  if (V <= 0) return 1;
  int v = V - 1;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return v + 1;
}

// 整数向上除法
template <int A, int B>
inline constexpr int ceil_div_v = (A + B - 1) / B;

// 编译期值的最大值
template <int A, int B>
inline constexpr int max_v = (A > B) ? A : B;

// 编译期值的最小值
template <int A, int B>
inline constexpr int min_v = (A < B) ? A : B;

// GCD（用于对齐计算）
template <int A, int B>
constexpr int gcd_ce() {
  if constexpr (B == 0) return A;
  else return gcd_ce<B, A % B>();
}

// LCM
template <int A, int B>
constexpr int lcm_ce() {
  return A * B / gcd_ce<A, B>();
}

// =============================================================================
// 2. 寄存器分配估算
// =============================================================================
// CUTLASS 在编译期计算每个线程的寄存器使用量。每个线程持有
// 累加器的一个片段。总寄存器数必须适配硬件限制
// （大多数 GPU 上每个线程 255 个，每个 SM 64K）。

template <int ThreadsPerWarp, int AccumulatorElementsPerThread,
          int TempRegisters = 4>
struct RegisterEstimator {
  static constexpr int warp_size = 32;  // NVIDIA warp 大小

  // 每个线程的寄存器数
  static constexpr int regs_per_thread =
      AccumulatorElementsPerThread + TempRegisters;

  // 每个 warp 的寄存器数
  static constexpr int regs_per_warp = regs_per_thread * warp_size;

  // 每个线程的最大寄存器数（硬件限制）
  static constexpr int max_regs_per_thread = 255;

  // 检查寄存器使用量是否在硬件限制内
  static constexpr bool within_limit =
      regs_per_thread <= max_regs_per_thread;

  static_assert(within_limit,
                "Register usage exceeds per-thread limit (255)");

  static void report() {
    std::cout << "  寄存器估算：" << std::endl;
    std::cout << "    累加器元素/线程："
              << AccumulatorElementsPerThread << std::endl;
    std::cout << "    临时寄存器：" << TempRegisters << std::endl;
    std::cout << "    总寄存器/线程：" << regs_per_thread << std::endl;
    std::cout << "    寄存器/warp：" << regs_per_warp << std::endl;
    std::cout << "    是否在限制内：" << (within_limit ? "是" : "否")
              << std::endl;
  }
};

// =============================================================================
// 3. 带填充的共享内存布局（Bank 冲突避免）
// =============================================================================
// CUTLASS 填充共享内存数组以避免 bank 冲突。NVIDIA GPU 上的共享内存
// 有 32 个 bank（每个 4 字节）。warp 中不同线程访问同一个 bank
// 会导致 bank 冲突。
//
// 策略：添加填充列，使连续行不会落入同一个 bank。

template <int Rows, int Cols, int ElementSize, int BankCount = 32,
          int BankWidth = 4>
struct SharedMemoryLayout {
  // 每个 bank 的元素数
  static constexpr int elements_per_bank = BankWidth / ElementSize;
  // 无填充时每行所需的 bank 数
  static constexpr int banks_per_row_raw = Cols / elements_per_bank;

  // 填充：添加足够的元素，使连续行不会别名到同一个 bank
  // 常见策略：填充到奇数个 bank（或添加 1 个 bank 宽的填充）
  static constexpr int padding_elements = []() constexpr {
    // 如果 Cols 恰好填满 bank，添加 1 个元素来错开对齐
    if ((Cols * ElementSize) % (BankCount * BankWidth) == 0) {
      return elements_per_bank;  // 错开一个 bank 宽度
    }
    return 0;
  }();

  static constexpr int padded_cols = Cols + padding_elements;
  static constexpr int smem_size_bytes = Rows * padded_cols * ElementSize;

  static void report() {
    std::cout << "  共享内存布局：" << std::endl;
    std::cout << "    行数：" << Rows << "，列数：" << Cols << std::endl;
    std::cout << "    元素大小：" << ElementSize << " 字节" << std::endl;
    std::cout << "    填充元素数：" << padding_elements << std::endl;
    std::cout << "    填充后列数：" << padded_cols << std::endl;
    std::cout << "    共享内存总计：" << smem_size_bytes << " B ("
              << smem_size_bytes / 1024.0 << " KB)" << std::endl;
  }
};

// =============================================================================
// 4. 分块细分（Threadblock -> Warp -> Thread）
// =============================================================================
// CUTLASS 将 threadblock 分块细分为 warp 级别分块，然后再细分为
// 每个线程的片段。所有划分都在编译期进行。

template <int ThreadblockM, int ThreadblockN, int ThreadblockK,
          int WarpsM, int WarpsN, int ThreadsM, int ThreadsN>
struct TileSubdivision {
  // --- Threadblock 级别 ---
  static constexpr int tb_M = ThreadblockM;
  static constexpr int tb_N = ThreadblockN;
  static constexpr int tb_K = ThreadblockK;

  // --- Warp 级别 ---
  static constexpr int warp_M = tb_M / WarpsM;
  static constexpr int warp_N = tb_N / WarpsN;
  static constexpr int warp_K = tb_K;  // 所有 warp 共享 K 分块
  static constexpr int total_warps = WarpsM * WarpsN;

  // --- 线程级别 ---
  static constexpr int thread_M = warp_M / ThreadsM;
  static constexpr int thread_N = warp_N / ThreadsN;
  static constexpr int threads_per_warp = ThreadsM * ThreadsN;

  // --- 每个线程的累加器元素数 ---
  static constexpr int accum_elements = thread_M * thread_N;

  // --- 每个 threadblock 的共享内存 ---
  // A 分块 + B 分块（双重缓冲时可选 x2）
  static constexpr int smem_a_bytes = tb_M * tb_K * 4;  // float = 4 字节
  static constexpr int smem_b_bytes = tb_N * tb_K * 4;

  // --- 验证 ---
  static_assert(tb_M % WarpsM == 0,
                "Threadblock M 必须能被 M 方向的 warp 数整除");
  static_assert(tb_N % WarpsN == 0,
                "Threadblock N 必须能被 N 方向的 warp 数整除");
  static_assert(warp_M % ThreadsM == 0,
                "Warp M 必须能被 M 方向的线程数整除");
  static_assert(warp_N % ThreadsN == 0,
                "Warp N 必须能被 N 方向的线程数整除");

  static void report() {
    std::cout << "  分块细分：" << std::endl;
    std::cout << "    Threadblock：" << tb_M << "x" << tb_N << "x" << tb_K
              << std::endl;
    std::cout << "    Warps：" << WarpsM << "x" << WarpsN << " = "
              << total_warps << " warps" << std::endl;
    std::cout << "    Warp 分块：" << warp_M << "x" << warp_N << "x" << warp_K
              << std::endl;
    std::cout << "    线程/warp：" << ThreadsM << "x" << ThreadsN << " = "
              << threads_per_warp << std::endl;
    std::cout << "    线程片段：" << thread_M << "x" << thread_N
              << " = " << accum_elements << " 个元素" << std::endl;
    std::cout << "    共享内存 A：" << smem_a_bytes << " B" << std::endl;
    std::cout << "    共享内存 B：" << smem_b_bytes << " B" << std::endl;
  }
};

// =============================================================================
// 5. 流水线阶段优化
// =============================================================================
// CUTLASS 根据可用共享内存和分块 K 维度来决定使用多少个流水线阶段。
// 更多阶段可以更好地隐藏延迟，但使用更多共享内存。

template <int SmemA, int SmemB, int MaxSmemBytes = 48 * 1024>
struct PipelineOptimizer {
  // 单缓冲的基准共享内存
  static constexpr int base_smem = SmemA + SmemB;

  // 检查能容纳多少个阶段
  static constexpr int max_stages_by_smem = MaxSmemBytes / base_smem;

  // 最少 1 个阶段，最多通常 4-6 个阶段（收益递减）
  static constexpr int optimal_stages = []() constexpr {
    if (max_stages_by_smem >= 3) return 3;
    if (max_stages_by_smem >= 2) return 2;
    return 1;
  }();

  static constexpr int total_smem_used = base_smem * optimal_stages;

  static void report() {
    std::cout << "  流水线优化器：" << std::endl;
    std::cout << "    基准共享内存：" << base_smem << " B" << std::endl;
    std::cout << "    SMEM 限制的最大阶段数：" << max_stages_by_smem << std::endl;
    std::cout << "    最优阶段数：" << optimal_stages << std::endl;
    std::cout << "    SMEM 总用量：" << total_smem_used << " B ("
              << total_smem_used / 1024.0 << " KB)" << std::endl;
  }
};

// =============================================================================
// 6. 占用率计算
// =============================================================================
// 占用率 = 每个 SM 的活动 warp 数 / 每个 SM 的最大 warp 数。
// CUTLASS 在编译期计算此值以指导 kernel 选择。

template <int ThreadsPerBlock, int SmemPerBlock, int RegsPerThread,
          int MaxThreadsPerSM = 2048, int MaxSmemPerSM = 48 * 1024,
          int MaxRegsPerSM = 65536, int MaxBlocksPerSM = 32>
struct OccupancyCalculator {
  static constexpr int warp_size = 32;

  // 线程数限制的最大 block 数
  static constexpr int blocks_by_threads = MaxThreadsPerSM / ThreadsPerBlock;

  // 共享内存限制的最大 block 数
  static constexpr int blocks_by_smem =
      (SmemPerBlock > 0) ? MaxSmemPerSM / SmemPerBlock : MaxBlocksPerSM;

  // 寄存器数限制的最大 block 数
  static constexpr int regs_per_block = RegsPerThread * ThreadsPerBlock;
  static constexpr int blocks_by_regs =
      (regs_per_block > 0) ? MaxRegsPerSM / regs_per_block : MaxBlocksPerSM;

  // 瓶颈约束
  static constexpr int max_active_blocks =
      min_v<blocks_by_threads, min_v<blocks_by_smem, blocks_by_regs>>;

  static constexpr int active_warps =
      max_active_blocks * ThreadsPerBlock / warp_size;

  static constexpr int max_warps_per_sm = MaxThreadsPerSM / warp_size;

  // 占用率百分比（0-100）
  static constexpr int occupancy_percent =
      active_warps * 100 / max_warps_per_sm;

  static constexpr bool is_thread_limited =
      blocks_by_threads <= blocks_by_smem && blocks_by_threads <= blocks_by_regs;
  static constexpr bool is_smem_limited =
      blocks_by_smem < blocks_by_threads && blocks_by_smem <= blocks_by_regs;
  static constexpr bool is_reg_limited =
      blocks_by_regs < blocks_by_threads && blocks_by_regs < blocks_by_smem;

  static void report() {
    std::cout << "  占用率计算器：" << std::endl;
    std::cout << "    线程/block：" << ThreadsPerBlock << std::endl;
    std::cout << "    SMEM/block：" << SmemPerBlock << " B" << std::endl;
    std::cout << "    寄存器/线程：" << RegsPerThread << std::endl;
    std::cout << "    线程限制的 block 数：" << blocks_by_threads << std::endl;
    std::cout << "    SMEM 限制的 block 数：" << blocks_by_smem << std::endl;
    std::cout << "    寄存器限制的 block 数：" << blocks_by_regs << std::endl;
    std::cout << "    SM 最大活动 block 数：" << max_active_blocks
              << std::endl;
    std::cout << "    SM 活动 warp 数：" << active_warps << "/"
              << max_warps_per_sm << std::endl;
    std::cout << "    占用率：" << occupancy_percent << "%" << std::endl;

    if (is_thread_limited)
      std::cout << "    瓶颈：线程数" << std::endl;
    if (is_smem_limited)
      std::cout << "    瓶颈：共享内存" << std::endl;
    if (is_reg_limited)
      std::cout << "    瓶颈：寄存器数" << std::endl;
  }
};

// =============================================================================
// 7. Block 级别地址计算
// =============================================================================
// 计算网格中每个 block 的全局内存地址偏移。
// 这通常在 kernel 启动配置中完成。

template <int BlockIdxM, int BlockIdxN, int TileM, int TileN,
          int LeadingDimK>
struct BlockAddress {
  // 此 block 在全局内存中的起始行和列
  static constexpr int start_row = BlockIdxM * TileM;
  static constexpr int start_col = BlockIdxN * TileN;

  // 线性偏移（假设 RowMajor）
  static constexpr int linear_offset = start_row * LeadingDimK + start_col;

  static void report() {
    std::cout << "  Block (" << BlockIdxM << ", " << BlockIdxN << ")："
              << std::endl;
    std::cout << "    start_row = " << start_row
              << ", start_col = " << start_col << std::endl;
    std::cout << "    linear_offset = " << linear_offset << std::endl;
  }
};

// =============================================================================
//                                   MAIN
// =============================================================================

int main() {
  using namespace std;

  cout << "=== 第 08.4 章：CUTLASS 风格编译期操作 ===\n"
       << endl;

  // --- 测试 1：编译期数学工具 ---
  cout << "[Test 1] 编译期数学工具：" << endl;
  static_assert(is_power_of_two_v<32>);
  static_assert(!is_power_of_two_v<100>);
  static_assert(next_power_of_two_ce(100) == 128);
  static_assert(ceil_div_v<10, 3> == 4);
  static_assert(max_v<5, 9> == 9);
  static_assert(min_v<5, 9> == 5);
  static_assert(gcd_ce<12, 18>() == 6);
  static_assert(lcm_ce<12, 18>() == 36);

  cout << "  is_power_of_two(32) = " << is_power_of_two_v<32> << endl;
  cout << "  next_power_of_two(100) = " << next_power_of_two_ce(100) << endl;
  cout << "  ceil_div(10, 3) = " << ceil_div_v<10, 3> << endl;
  cout << "  gcd(12, 18) = " << gcd_ce<12, 18>() << endl;
  cout << "  lcm(12, 18) = " << lcm_ce<12, 18>() << endl;

  // --- 测试 2：寄存器估算 ---
  cout << "\n[Test 2] 寄存器估算：" << endl;
  // 每个线程 8 个累加器元素 + 4 个临时寄存器 = 12 寄存器/线程
  RegisterEstimator<32, 8, 4>::report();

  // 更高用量：64 个累加器元素
  cout << "\n  更高寄存器用量：" << endl;
  RegisterEstimator<32, 64, 8>::report();

  static_assert(RegisterEstimator<32, 8, 4>::regs_per_thread == 12);
  static_assert(RegisterEstimator<32, 8, 4>::within_limit);

  // --- 测试 3：带填充的共享内存布局 ---
  cout << "\n[Test 3] 共享内存布局：" << endl;
  // 64x64 浮点数分块（如果不对齐到整 bank，无需填充）
  SharedMemoryLayout<64, 64, 4>::report();

  // 64x8 分块（窄，可能跨越 bank）
  cout << "\n  窄分块：" << endl;
  SharedMemoryLayout<64, 8, 4>::report();

  // --- 测试 4：分块细分 ---
  cout << "\n[Test 4] 分块细分：" << endl;
  using Tile = TileSubdivision<
      /* tb_M */ 128, /* tb_N */ 128, /* tb_K */ 8,
      /* WarpsM */ 2, /* WarpsN */ 2,
      /* ThreadsM */ 4, /* ThreadsN */ 8>;

  Tile::report();

  static_assert(Tile::warp_M == 64);
  static_assert(Tile::warp_N == 64);
  static_assert(Tile::total_warps == 4);
  static_assert(Tile::thread_M == 16);
  static_assert(Tile::thread_N == 8);
  static_assert(Tile::accum_elements == 128);
  static_assert(Tile::smem_a_bytes == 128 * 8 * 4);
  static_assert(Tile::smem_b_bytes == 128 * 8 * 4);

  // --- 测试 5：流水线优化 ---
  cout << "\n[Test 5] 流水线优化：" << endl;
  using Pipe = PipelineOptimizer<Tile::smem_a_bytes, Tile::smem_b_bytes>;
  Pipe::report();

  static_assert(Pipe::optimal_stages >= 1);

  // --- 测试 6：占用率计算 ---
  cout << "\n[Test 6] 占用率计算：" << endl;
  // 256 线程/block，16KB smem，32 寄存器/线程
  using Occ = OccupancyCalculator<256, 16 * 1024, 32>;
  Occ::report();

  static_assert(Occ::max_active_blocks > 0);
  static_assert(Occ::active_warps > 0);

  // 更高 SMEM 用量 -> 更低占用率
  cout << "\n  更高 SMEM 用量：" << endl;
  using Occ2 = OccupancyCalculator<256, 32 * 1024, 64>;
  Occ2::report();

  // --- 测试 7：Block 地址计算 ---
  cout << "\n[Test 7] Block 地址计算：" << endl;
  cout << "  1024x1024 问题，128x128 分块的网格：" << endl;
  BlockAddress<0, 0, 128, 128, 1024>::report();
  BlockAddress<0, 1, 128, 128, 1024>::report();
  BlockAddress<1, 0, 128, 128, 1024>::report();

  static_assert(BlockAddress<0, 0, 128, 128, 1024>::linear_offset == 0);
  static_assert(BlockAddress<0, 1, 128, 128, 1024>::linear_offset == 128);
  static_assert(BlockAddress<1, 0, 128, 128, 1024>::linear_offset == 131072);

  // --- 测试 8：汇总 ---
  cout << "\n[Test 8] 完整编译期 kernel 配置：" << endl;
  cout << "  配置汇总：" << endl;
  cout << "  =======================" << endl;
  Tile::report();
  Pipe::report();
  Occ::report();

  cout << "\n所有测试通过！" << endl;
  return 0;
}
