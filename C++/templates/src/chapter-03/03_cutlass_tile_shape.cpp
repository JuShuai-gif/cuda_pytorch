// =============================================================================
// 第 03.3 章：CUTLASS 风格的 GemmShape 编译期形状系统
//
// CUTLASS 使用编译期形状系统，其中矩阵维度是非类型模板参数。
// 形状在整个内核层级中传递：
//
//   GemmShape<M, N, K>      -- 线程块级 tile 几何
//   WarpShape<M, N, K>      -- warp 级 tile 几何
//   InstructionShape<M,N,K> -- MMA 指令级形状
//
// 每一级都是对上一级的细分。所有维度在编译期已知，
// 从而实现激进的循环展开和寄存器分配。
//
// 本文件实现：
//   1. GemmShape 作为根形状
//   2. 子形状推导（从 GemmShape 到 WarpShape）
//   3. 形状划分和迭代次数计算
//   4. 可读的形状打印输出
//   5. 形状组合（嵌套形状）
//   6. 编译期验证（2 的幂、可整除性）
//
// 编译：g++ -std=c++20 -o 03_cutlass_tile_shape 03_cutlass_tile_shape.cpp
// =============================================================================

#include <cassert>
#include <cstddef>
#include <iostream>
#include <string>
#include <type_traits>

// =============================================================================
// 1. 工具：编译期 2 的幂和可整除性检查
// =============================================================================

template <int V>
constexpr bool is_power_of_two_v = (V > 0) && ((V & (V - 1)) == 0);

template <int A, int B>
constexpr bool is_divisible_v = (A % B == 0);

template <int A, int B>
constexpr int ceil_div_v = (A + B - 1) / B;

// =============================================================================
// 2. GemmShape -- 根线程块级 tile 形状
// =============================================================================

template <int M_, int N_, int K_>
struct GemmShape {
  static constexpr int M = M_;  // C tile 的行数
  static constexpr int N = N_;  // C tile 的列数
  static constexpr int K = K_;  // 内层维度 tile

  // --- 验证 ---
  static_assert(M > 0 && N > 0 && K > 0,
                "GemmShape 维度必须为正数");
  static_assert(is_power_of_two_v<M> && is_power_of_two_v<N>,
                "GemmShape M、N 必须是 2 的幂以符合 CUTLASS 要求");
  static_assert(K % 8 == 0, "GemmShape K 必须是 8 的倍数");

  // --- 推导值 ---
  static constexpr int total_elements = M * N;
  static constexpr int total_flops    = M * N * K * 2;  // 融合乘加
  static constexpr int a_tile_bytes   = M * K * 4;       // 假设 float（4B）
  static constexpr int b_tile_bytes   = K * N * 4;
  static constexpr int c_tile_bytes   = M * N * 4;

  // 线程块级的迭代次数（每次迭代沿 K 方向消耗一个 tile）
  static constexpr int k_iterations = K / 8;  // 8 宽向量

  static std::string to_string() {
    return "GemmShape<" + std::to_string(M) + "," + std::to_string(N) + ","
           + std::to_string(K) + ">";
  }

  static void print() {
    std::cout << to_string() << std::endl;
    std::cout << "  元素数：" << total_elements << std::endl;
    std::cout << "  FLOPs：" << total_flops << std::endl;
    std::cout << "  K 迭代次数：" << k_iterations << std::endl;
    std::cout << "  A tile：" << a_tile_bytes << " B" << std::endl;
    std::cout << "  B tile：" << b_tile_bytes << " B" << std::endl;
    std::cout << "  C tile：" << c_tile_bytes << " B" << std::endl;
  }
};

// =============================================================================
// 3. WarpShape -- 将 GemmShape 细分为每个 warp 的 tile
// =============================================================================

template <typename GemmShape_, int WarpM_, int WarpN_, int WarpK_>
struct WarpShape {
  using ThreadblockShape = GemmShape_;

  static constexpr int M = WarpM_;
  static constexpr int N = WarpN_;
  static constexpr int K = WarpK_;

  // --- 验证 ---
  static_assert(M > 0 && N > 0 && K > 0,
                "WarpShape 维度必须为正数");
  static_assert(is_divisible_v<GemmShape_::M, WarpM_>,
                "GemmShape M 必须能被 WarpShape M 整除");
  static_assert(is_divisible_v<GemmShape_::N, WarpN_>,
                "GemmShape N 必须能被 WarpShape N 整除");

  // --- 推导值 ---
  // M 和 N 方向的 warp 数
  static constexpr int warp_count_M = GemmShape_::M / WarpM_;
  static constexpr int warp_count_N = GemmShape_::N / WarpN_;
  static constexpr int total_warps  = warp_count_M * warp_count_N;

  // 每个 warp tile 的元素数
  static constexpr int elements_per_warp = M * N;
  static constexpr int flops_per_warp    = M * N * K * 2;

  static std::string to_string() {
    return "WarpShape<" + std::to_string(M) + "," + std::to_string(N) + ","
           + std::to_string(K) + ">";
  }

  static void print() {
    std::cout << to_string() << std::endl;
    std::cout << "  每个线程块的 warp 数：" << total_warps
              << "（" << warp_count_M << "x" << warp_count_N << "）" << std::endl;
    std::cout << "  每个 warp 的元素数：" << elements_per_warp << std::endl;
    std::cout << "  每个 warp 的 FLOPs：" << flops_per_warp << std::endl;
  }
};

// =============================================================================
// 4. InstructionShape -- MMA 指令级形状
// =============================================================================
// Tensor Core 指令在小的 mma 形状上操作，如 16x8x8 或 16x8x16。
// WarpShape 被细分为 InstructionShape 大小的片段。

template <typename WarpShape_, int InstM_, int InstN_, int InstK_>
struct InstructionShape {
  using Warp = WarpShape_;

  static constexpr int M = InstM_;
  static constexpr int N = InstN_;
  static constexpr int K = InstK_;

  // --- 验证 ---
  static_assert(is_divisible_v<Warp::M, InstM_>,
                "WarpShape M 必须能被 Instruction M 整除");
  static_assert(is_divisible_v<Warp::N, InstN_>,
                "WarpShape N 必须能被 Instruction N 整除");
  static_assert(is_divisible_v<Warp::K, InstK_>,
                "WarpShape K 必须能被 Instruction K 整除");

  // 每个 warp 沿各维度的指令数
  static constexpr int inst_count_M = Warp::M / InstM_;
  static constexpr int inst_count_N = Warp::N / InstN_;
  static constexpr int inst_count_K = Warp::K / InstK_;
  static constexpr int total_instructions =
      inst_count_M * inst_count_N * inst_count_K;

  // 每条指令的累加器元素数
  static constexpr int accum_elements = M * N;

  static std::string to_string() {
    return "InstructionShape<" + std::to_string(M) + "," + std::to_string(N)
           + "," + std::to_string(K) + ">";
  }

  static void print() {
    std::cout << to_string() << std::endl;
    std::cout << "  每个 warp 的指令数：" << total_instructions
              << "（" << inst_count_M << "x" << inst_count_N << "x"
              << inst_count_K << "）" << std::endl;
    std::cout << "  累加器元素数：" << accum_elements << std::endl;
  }
};

// =============================================================================
// 5. 完整形状层级组合器
// =============================================================================
// 将整个形状层级编码为嵌套模板参数。

template <typename GemmShape_,
          int WarpM, int WarpN, int WarpK,
          int InstM, int InstN, int InstK>
struct ShapeHierarchy {
  using Threadblock = GemmShape_;
  using Warp        = WarpShape<GemmShape_, WarpM, WarpN, WarpK>;
  using Instruction = InstructionShape<Warp, InstM, InstN, InstK>;

  // 所有层级的 K 迭代总数
  static constexpr int total_k_iterations =
      GemmShape_::K / InstK;  // 指令级的 K 遍历

  // 每个 warp 的寄存器估算（累加器）
  static constexpr int registers_per_warp =
      Warp::elements_per_warp;  // 简化

  static void print() {
    std::cout << "========================================" << std::endl;
    std::cout << "形状层级" << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << "[线程块] ";
    Threadblock::print();

    std::cout << "[Warp] ";
    Warp::print();

    std::cout << "[指令] ";
    Instruction::print();

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "K 迭代总数：" << total_k_iterations << std::endl;
    std::cout << "估算寄存器/warp：" << registers_per_warp << std::endl;
    std::cout << "========================================" << std::endl;
  }
};

// =============================================================================
// 6. 形状比例 / 比较工具
// =============================================================================
// 比较两个形状：哪个更大，计算比例等。

template <typename ShapeA, typename ShapeB>
struct ShapeRatio {
  static constexpr double ratio_M = static_cast<double>(ShapeA::M) / ShapeB::M;
  static constexpr double ratio_N = static_cast<double>(ShapeA::N) / ShapeB::N;
  static constexpr double ratio_K = static_cast<double>(ShapeA::K) / ShapeB::K;
  static constexpr double ratio_elements =
      static_cast<double>(ShapeA::total_elements) / ShapeB::total_elements;

  static void print() {
    std::cout << ShapeA::to_string() << " vs " << ShapeB::to_string()
              << std::endl;
    std::cout << "  M 比例：" << ratio_M << std::endl;
    std::cout << "  N 比例：" << ratio_N << std::endl;
    std::cout << "  K 比例：" << ratio_K << std::endl;
    std::cout << "  元素比例：" << ratio_elements << std::endl;
  }
};

// =============================================================================
// 7. 常见 CUTLASS 形状预设（供参考）
// =============================================================================
// CUTLASS 附带许多预定义的形状。以下是典型的几种。

using GemmShape_128x128x8   = GemmShape<128, 128, 8>;
using GemmShape_128x64x8    = GemmShape<128, 64, 8>;
using GemmShape_64x64x32    = GemmShape<64, 64, 32>;
using GemmShape_256x128x16  = GemmShape<256, 128, 16>;
using GemmShape_128x256x8   = GemmShape<128, 256, 8>;

// 典型 warp 形状
// WarpShape<Threadblock, 64, 64, K> -- 每个线程块 4 个 warp（2x2）
// WarpShape<Threadblock, 32, 64, K> -- 每个线程块 8 个 warp（4x2）

// 典型指令形状（模拟 Tensor Core mma.sync 形状）
// InstructionShape<Warp, 16, 8, 8>   -- Volta/Turing fp16
// InstructionShape<Warp, 16, 8, 16>  -- Ampere tf32

// =============================================================================
// 8. 形状缩放（将形状乘以一个因子）
// =============================================================================
// 缩放形状的元函数：ScaleShape<Shape, Scale>
// 仅作演示用途；真正的 CUTLASS 形状是固定的。

template <typename Shape, int Scale>
struct ScaleShape {
  using type = GemmShape<Shape::M * Scale, Shape::N * Scale, Shape::K>;
};

template <typename Shape, int Scale>
using scale_shape_t = typename ScaleShape<Shape, Scale>::type;

// =============================================================================
//                                   主函数
// =============================================================================

int main() {
  using namespace std;

  cout << "=== 第 03.3 章：CUTLASS 风格的 GemmShape 系统 ===\n" << endl;

  // --- 测试 1：基本 GemmShape ---
  cout << "[测试 1] 基本 GemmShape：" << endl;
  using GS = GemmShape<128, 128, 8>;
  GS::print();

  static_assert(GS::M == 128 && GS::N == 128 && GS::K == 8);
  static_assert(GS::total_elements == 128 * 128);
  static_assert(GS::total_flops == 128 * 128 * 8 * 2);
  static_assert(GS::k_iterations == 1);

  // --- 测试 2：WarpShape ---
  cout << "\n[测试 2] WarpShape：" << endl;
  using WS = WarpShape<GS, 64, 64, 8>;
  WS::print();

  static_assert(WS::M == 64 && WS::N == 64);
  static_assert(WS::warp_count_M == 2 && WS::warp_count_N == 2);
  static_assert(WS::total_warps == 4);
  static_assert(WS::elements_per_warp == 64 * 64);

  // 验证坏 warp 形状会被捕获
  // WarpShape<GS, 63, 64, 8> 会 static_assert 失败（63 不能整除 128）

  // --- 测试 3：InstructionShape ---
  cout << "\n[测试 3] InstructionShape：" << endl;
  using IS = InstructionShape<WS, 16, 8, 8>;
  IS::print();

  static_assert(IS::M == 16 && IS::N == 8 && IS::K == 8);
  static_assert(IS::inst_count_M == 4);   // 64/16
  static_assert(IS::inst_count_N == 8);   // 64/8
  static_assert(IS::inst_count_K == 1);   // 8/8
  static_assert(IS::total_instructions == 32);

  // --- 测试 4：完整形状层级 ---
  cout << "\n[测试 4] 完整 ShapeHierarchy：" << endl;
  using Hierarchy = ShapeHierarchy<GS, 64, 64, 8, 16, 8, 8>;
  Hierarchy::print();

  static_assert(Hierarchy::Threadblock::M == 128);
  static_assert(Hierarchy::Warp::total_warps == 4);
  static_assert(Hierarchy::Instruction::total_instructions == 32);
  static_assert(Hierarchy::total_k_iterations == 1);  // 8/8
  static_assert(Hierarchy::registers_per_warp == 64 * 64);

  // --- 测试 5：形状比较 ---
  cout << "\n[测试 5] 形状比较：" << endl;
  using GS_Small = GemmShape<64, 64, 32>;
  cout << "  比例（GS / GS_Small）：" << endl;
  ShapeRatio<GS, GS_Small>::print();

  static_assert(ShapeRatio<GS, GS_Small>::ratio_M == 2.0);
  static_assert(ShapeRatio<GS, GS_Small>::ratio_N == 2.0);
  static_assert(ShapeRatio<GS, GS_Small>::ratio_K == 0.25);

  // --- 测试 6：预定义形状 ---
  cout << "\n[测试 6] 预定义形状：" << endl;
  cout << "  "; GemmShape_128x128x8::print();
  cout << "  "; GemmShape_64x64x32::print();
  cout << "  "; GemmShape_256x128x16::print();

  // --- 测试 7：缩放形状 ---
  cout << "\n[测试 7] 缩放形状：" << endl;
  using Scaled = scale_shape_t<GS_Small, 2>;
  cout << "  将 GS_Small 缩放 x2：";
  Scaled::print();
  static_assert(Scaled::M == 128 && Scaled::N == 128 && Scaled::K == 32);

  // --- 测试 8：不同形状的不同层级 ---
  cout << "\n[测试 8] 替代层级（Turing 风格）：" << endl;
  using TuringShape = GemmShape<256, 128, 16>;
  using TuringHierarchy = ShapeHierarchy<TuringShape, 64, 64, 16, 16, 8, 8>;
  TuringHierarchy::print();

  static_assert(TuringHierarchy::Warp::total_warps == 8);  // 4x2
  static_assert(TuringHierarchy::Instruction::total_instructions ==
                4 * 8 * 2);  // 64

  // --- 测试 9：边界情况 -- 最小有效形状 ---
  cout << "\n[测试 9] 最小有效 GemmShape：" << endl;
  using TinyShape = GemmShape<8, 8, 8>;
  cout << "  "; TinyShape::print();
  static_assert(TinyShape::M == 8 && TinyShape::N == 8);
  static_assert(TinyShape::total_flops == 8 * 8 * 8 * 2);

  // K=16：2 次迭代
  using ShapeK16 = GemmShape<128, 128, 16>;
  static_assert(ShapeK16::k_iterations == 2);

  cout << "\n所有测试通过！" << endl;
  return 0;
}
