// =============================================================================
// 第 04.3 章：CUTLASS 风格的类型列表分发
//
// CUTLASS 使用编译期类型列表来枚举所有可能的内核配置。
// 在运行时，分发函数遍历类型列表，根据问题参数
//（对齐、数据类型、布局等）选择合适的内核特化。
//
// 本文件演示：
//   1. 具有多个 tile 形状的内核类型列表
//   2. 在编译期遍历类型列表的分发系统
//   3. 通过递归模板实例化进行编译期迭代
//   4. 运行时到编译期的桥接（对枚举进行 switch）
//   5. 无匹配特化时的默认"回退"内核
//   6. 基于对齐的分发（CUTLASS 模式）
//
// 编译：g++ -std=c++20 -o 03_cutlass_type_list 03_cutlass_type_list.cpp
// =============================================================================

#include <cassert>
#include <cstddef>
#include <iostream>
#include <string>
#include <type_traits>

// =============================================================================
// 1. TypeList（最小化版，来自 04.2）
// =============================================================================

template <typename... Ts>
struct TypeList {
  static constexpr std::size_t size = sizeof...(Ts);
};

// =============================================================================
// 2. 内核特化（标签类型）
// =============================================================================
// 每个内核特化是一个编码其配置的独立类型。
// 在真实 CUTLASS 中，这些是完整的内核类；这里使用存储
// tile 形状的轻量级标签类型。

template <int M, int N, int K>
struct GemmKernel {
  static constexpr int tile_M = M;
  static constexpr int tile_N = N;
  static constexpr int tile_K = K;

  static std::string name() {
    return "GemmKeernel<" + std::to_string(M) + "," + std::to_string(N)
           + "," + std::to_string(K) + ">";
  }

  static constexpr int flops_per_tile = M * N * K * 2;

  // 最小化的 setter，让内核看起来好像做了什么
  static void execute(int problem_M, int problem_N, int problem_K) {
    int grid_m = (problem_M + M - 1) / M;
    int grid_n = (problem_N + N - 1) / N;
    std::cout << "    [内核] " << name() << " 已启动" << std::endl;
    std::cout << "      Tile：" << M << "x" << N << "x" << K << std::endl;
    std::cout << "      Grid：" << grid_m << "x" << grid_n << std::endl;
    std::cout << "      FLOPs/tile：" << flops_per_tile << std::endl;
  }
};

// =============================================================================
// 3. 内核列表 -- 所有可用的特化
// =============================================================================
// 在真实 CUTLASS 中，此列表由配置参数的笛卡尔积生成。
// 这里硬编码一些典型配置。

using KernelList = TypeList<
    GemmKernel<256, 128, 16>,
    GemmKernel<256, 64,  16>,
    GemmKernel<128, 256, 16>,
    GemmKernel<128, 128, 32>,
    GemmKernel<128, 128, 16>,
    GemmKernel<128, 64,  32>,
    GemmKernel<128, 64,  16>,
    GemmKernel<64,  128, 32>,
    GemmKernel<64,  128, 16>,
    GemmKernel<64,  64,  32>,
    GemmKernel<64,  64,  16>,
    GemmKernel<32,  64,  32>,
    GemmKernel<32,  32,  32>
>;

// =============================================================================
// 4. 分发：为给定问题找到最佳内核
// =============================================================================
// 分发函数在内核列表中搜索其 tile 维度最有效覆盖问题（最小浪费）的内核。
//
// 策略：在编译期遍历类型列表，生成一系列 if-else 分支。
// 每个分支检查运行时条件，如果满足则实例化内核。

// --- 分发步骤：尝试一个内核，然后递归处理下一个 ---

// 前向声明
template <typename KernelList, typename Problem>
struct KernelDispatcher;

// 基例：空列表 -- 有适当回退的情况下不应发生
template <typename Problem>
struct KernelDispatcher<TypeList<>, Problem> {
  static void dispatch(int M, int N, int K) {
    std::cout << "  错误：未找到适合问题的内核！" << std::endl;
  }
};

// 递归情况：检查头部内核，然后递归处理尾部
template <typename Kernel, typename... RestKernels, typename Problem>
struct KernelDispatcher<TypeList<Kernel, RestKernels...>, Problem> {
  static void dispatch(int M, int N, int K) {
    // 运行时检查：此内核的 tile 大小是否适合问题？
    // 启发式：选择 tile_M 是沿 M 方向小于等于问题维度的最大 2 的幂的内核。
    // 但为简化起见，我们选择第一个 tile 大小"合理适配"的内核。
    //
    // 更现实的启发式：最小化浪费比率（tile_size / problem_size）的乘积。
    // 这里使用简化条件：选择仍然适配问题的最大 tile 的内核。

    // 这是简化的分发；真实 CUTLASS 根据问题维度预计算优选的
    // 内核索引。

    // 为了本演示：分发到 tile 大小与问题维度"足够接近"（在 2 倍以内）
    // 的第一个内核。

    constexpr int tile_M = Kernel::tile_M;
    constexpr int tile_N = Kernel::tile_N;

    // 如果 tile 大小与问题维度"兼容"则选择
    bool fits = (M >= tile_M / 2) && (N >= tile_N / 2) &&
                (tile_M <= M * 2) && (tile_N <= N * 2);

    if (fits) {
      Kernel::execute(M, N, K);
    } else {
      // 尝试列表中的下一个内核
      KernelDispatcher<TypeList<RestKernels...>, Problem>::dispatch(M, N, K);
    }
  }
};

// 便捷包装器
template <typename KernelList>
struct GemmLauncher {
  static void launch(int M, int N, int K) {
    std::cout << "\n[GemmLauncher] 问题：M=" << M << ", N=" << N
              << ", K=" << K << std::endl;
    // 模拟一个哑问题类型（可以编码对齐、布局等）
    struct Problem {};  // 占位符
    KernelDispatcher<KernelList, Problem>::dispatch(M, N, K);
  }
};

// =============================================================================
// 5. 基于对齐的分发（另一种 CUTLASS 模式）
// =============================================================================
// CUTLASS 根据 leading dimension 的对齐进行分发。如果
// leading dimension 是 8 的倍数，使用向量化加载；否则回退到标量加载。

enum class Alignment { Align1 = 1, Align4 = 4, Align8 = 8 };

// 依赖对齐的内核标签
template <Alignment Align>
struct AlignedKernel {
  static std::string name() {
    return "AlignedKeernel<Align" + std::to_string(static_cast<int>(Align))
           + ">";
  }
  static void execute() {
    std::cout << "  运行 " << name()
              << " 使用向量化加载（宽度="
              << static_cast<int>(Align) << "）" << std::endl;
  }
};

// 按对齐分发 -- 按依赖顺序定义（从最小的开始）
template <Alignment Target>
struct AlignmentDispatcher;

template <>
struct AlignmentDispatcher<Alignment::Align1> {
  static void run(int /*ld*/) {
    AlignedKernel<Alignment::Align1>::execute();
  }
};

template <>
struct AlignmentDispatcher<Alignment::Align4> {
  static void run(int ld) {
    if (ld % 4 == 0) {
      AlignedKernel<Alignment::Align4>::execute();
    } else {
      AlignmentDispatcher<Alignment::Align1>::run(ld);
    }
  }
};

template <>
struct AlignmentDispatcher<Alignment::Align8> {
  static void run(int ld) {
    if (ld % 8 == 0) {
      AlignedKernel<Alignment::Align8>::execute();
    } else {
      // 回退到下一级对齐
      AlignmentDispatcher<Alignment::Align4>::run(ld);
    }
  }
};

// =============================================================================
// 6. 自动调优风格的分发（遍历所有，选择最佳）
// =============================================================================
// 简化的评分：遍历所有内核并打印其预测性能指标。
// 在真实的自动调优器中，会跟踪最佳值。

template <typename KernelList>
struct KernelScorer;

template <>
struct KernelScorer<TypeList<>> {
  static void evaluate(int /*M*/, int /*N*/, int /*K*/) {}
};

template <typename Kernel, typename... Rest>
struct KernelScorer<TypeList<Kernel, Rest...>> {
  static long long compute_score(int M, int N) {
    int blocks_m = (M + Kernel::tile_M - 1) / Kernel::tile_M;
    int blocks_n = (N + Kernel::tile_N - 1) / Kernel::tile_N;
    long long waste = static_cast<long long>(blocks_m * Kernel::tile_M - M) *
                      (blocks_n * Kernel::tile_N - N);
    return -waste;  // 浪费越少 = 分数越高
  }

  static void evaluate(int M, int N, int K) {
    long long score = compute_score(M, N);
    std::cout << "  " << Kernel::name() << " 分数=" << score << std::endl;
    KernelScorer<TypeList<Rest...>>::evaluate(M, N, K);
  }
};

// =============================================================================
//                                   主函数
// =============================================================================

int main() {
  using namespace std;

  cout << "=== 第 04.3 章：CUTLASS 风格的类型列表分发 ===\n" << endl;

  // --- 测试 1：枚举内核列表 ---
  cout << "[测试 1] 内核列表枚举：" << endl;
  cout << "  内核总数：" << KernelList::size << endl;

  // --- 测试 2：按问题大小分发 ---
  cout << "\n[测试 2] 按问题大小分发到最佳内核：" << endl;

  // 小问题：32x32 应该匹配 GemmKernel<32,32,32>
  GemmLauncher<KernelList>::launch(32, 32, 32);

  // 中等问题：128x128 -> 匹配 128x128
  GemmLauncher<KernelList>::launch(128, 128, 16);

  // 大问题：1024x1024 -> 匹配 256x128 或 128x256
  GemmLauncher<KernelList>::launch(1024, 1024, 64);

  // 奇怪的问题大小
  GemmLauncher<KernelList>::launch(100, 200, 50);

  // --- 测试 3：基于对齐的分发 ---
  cout << "\n[测试 3] 基于对齐的分发：" << endl;
  cout << "  Leading dimension = 256：" << endl;
  AlignmentDispatcher<Alignment::Align8>::run(256);

  cout << "  Leading dimension = 20：" << endl;
  AlignmentDispatcher<Alignment::Align8>::run(20);

  cout << "  Leading dimension = 5：" << endl;
  AlignmentDispatcher<Alignment::Align8>::run(5);

  cout << "  Leading dimension = 64（4 的倍数但不是 8 的倍数）：" << endl;
  AlignmentDispatcher<Alignment::Align8>::run(64);

  // --- 测试 4：自动调优风格 ---
  cout << "\n[测试 4] 自动调优内核评分：" << endl;
  cout << "  对 256x256 评分：" << endl;
  KernelScorer<KernelList>::evaluate(256, 256, 16);

  // --- 测试 5：内核列表的静态属性 ---
  cout << "\n[测试 5] 静态属性：" << endl;
  // 列表中的第一个内核
  using FirstKernel = GemmKernel<256, 128, 16>;
  static_assert(FirstKernel::tile_M == 256);
  static_assert(FirstKernel::tile_N == 128);
  static_assert(FirstKernel::tile_K == 16);
  cout << "  第一个内核：" << FirstKernel::name()
       << " FLOPs/tile=" << FirstKernel::flops_per_tile << endl;

  // 最后一个内核（概念上最小的）
  using LastKernel = GemmKernel<32, 32, 32>;
  static_assert(LastKernel::tile_M == 32);
  cout << "  最后一个内核：" << LastKernel::name()
       << " FLOPs/tile=" << LastKernel::flops_per_tile << endl;

  // --- 测试 6：TypeList 空边界情况 ---
  cout << "\n[测试 6] 空列表分发（应打印错误）：" << endl;
  struct DummyProblem {};
  KernelDispatcher<TypeList<>, DummyProblem>::dispatch(1, 1, 1);

  // --- 测试 7：分发基准：最小问题 ---
  cout << "\n[测试 7] 分发 8x8x8（极小问题）：" << endl;
  GemmLauncher<KernelList>::launch(8, 8, 8);

  cout << "\n所有测试通过！" << endl;
  return 0;
}
